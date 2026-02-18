import numpy as np

class SingleMPPIPlanner:
    """
    MPPI para dron como punto-masa:
      x = [p(3), v(3)]
      u = a_cmd(3)  (aceleración deseada)
    Obstáculos: esferas y cajas AABB (axis-aligned).
    """

    def __init__(
        self,
        dt:float = 0.1,
        horizon: int = 30,           # pasos
        num_samples: int = 300,      # rollouts
        lambda_: float = 1.0,            # temperatura MPPI (exploración)
        noise_sigma: np.ndarray = np.array([2.0, 2.0, 4.0]),  # ruido en Newton step
        F_min: np.ndarray = np.array([-8.0, -8.0,  0.0]),
        F_max: np.ndarray = np.array([ 8.0,  8.0,  25.0]),
        w_goal: float = 6.0,         # goal weighting (tracking)
        w_terminal: float = 50.0,    # terminal cost weighting
        w_F: float = 0.05,           # control cost weighting
        w_smooth: float = 0.02,      # smoothness cost weighting (delta u)
        w_obs: float = 80.0,         # obstacle cost weighting
        obs_margin: float = 0.20,    # margen de seguridad para obstáculos
        obs_softness: float = 0.15,  # suavidad de la penalización de obstáculos
        goal_tolerance: float = 0.10, # distancia al goal para considerar "llegado"
        z_min: float = 0.5,         # altura mínima (suelo)
        z_max: float = 2.0,         # altura máxima (techo)
        z_margin: float = 0.15,      # “zona amarilla” cerca de límites
        w_z: float = 200.0,          # peso altura durante rollout
        w_z_terminal: float = 400.0, # peso terminal altura
        rng_seed: int = 0,
    ):
        self.dt = float(dt)
        self.H = int(horizon)
        self.K = int(num_samples)
        self.lambda_ = float(lambda_)

        self.noise_sigma = noise_sigma.astype(float)
        self.F_min = F_min.astype(float)
        self.F_max = F_max.astype(float)

        self.w_goal = float(w_goal)
        self.w_terminal = float(w_terminal)
        self.w_F = float(w_F)
        self.w_smooth = float(w_smooth)
        self.w_obs = float(w_obs)
        self.obs_margin = float(obs_margin)
        self.obs_softness = float(obs_softness)
        self.goal_tolerance = float(goal_tolerance)

        self.goal = np.zeros(3, dtype=float)
        
        # altura
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.z_margin = float(z_margin)
        self.w_z = float(w_z)
        self.w_z_terminal = float(w_z_terminal)

        # model
        self.model = None  # <-- por defecto no hay modelo

        # Secuencia nominal de fuerzas (H x 3)
        self.F_nom = np.zeros((self.H, 3), dtype=float)

        # Obstáculos: lista de dicts
        self.obstacles = []

        # RNG
        self.rng = np.random.default_rng(rng_seed)

    def set_goal(self, goal_xyz):
        self.goal = np.array(goal_xyz, dtype=float)

    def set_obstacles(self, obstacles):
        """
        obstacles: lista de dicts, ejemplos:
          {"type":"sphere", "c":[x,y,z], "r":0.25}
          {"type":"box", "c":[x,y,z], "h":[hx,hy,hz]}   # AABB half-sizes
        """
        self.obstacles = obstacles

    # ---------------------------
    # Distancias a obstáculos
    # ---------------------------
    def _sphere_penalty(self, p, c, r):
        # distancia al borde (positiva fuera)
        d = np.linalg.norm(p - c) - (r + self.obs_margin)
        # penalización suave: grande cerca / dentro
        # inside => d<0 => exp(-d/s) crece muchísimo
        return np.exp(-d / self.obs_softness)

    def _box_penalty(self, p, c, h):
        # AABB: distancia signed-like (aprox) usando distancia a caja
        q = np.abs(p - c) - h - self.obs_margin
        outside = np.maximum(q, 0.0)
        d_out = np.linalg.norm(outside)  # 0 si está dentro/proyectado
        # Si está dentro: q tiene componentes negativas
        inside = np.all(q <= 0.0)
        if inside:
            # dentro => penaliza muy fuerte
            return np.exp(+2.0 / self.obs_softness)
        else:
            return np.exp(-d_out / self.obs_softness)

    def _obstacle_cost(self, p):
        if not self.obstacles:
            return 0.0
        cost = 0.0
        for obs in self.obstacles:
            if obs["type"] == "sphere":
                cost += self._sphere_penalty(p, np.array(obs["c"], float), float(obs["r"]))
            elif obs["type"] == "box":
                cost += self._box_penalty(p, np.array(obs["c"], float), np.array(obs["h"], float))
        return self.w_obs * cost
    

    # ---------------------------
    # Costo por rango de altura
    # ---------------------------
    def _height_cost(self, z: float, terminal: bool = False) -> float:
        """
        Penaliza salir de [z_min, z_max]. Dentro del rango => ~0.
        En la zona de margen, empieza a subir suave.
        Fuera del rango, sube fuerte.
        """
        # Queremos que (z_min + margin) <= z <= (z_max - margin) sea “seguro”
        low = (self.z_min + self.z_margin) - z   # >0 si está por debajo de zona segura
        high = z - (self.z_max - self.z_margin)  # >0 si está por encima de zona segura

        # softplus hace transición suave, luego elevamos al cuadrado para empuje fuerte
        pen = self.softplus(low) ** 2 + self.softplus(high) ** 2
        w = self.w_z_terminal if terminal else self.w_z
        return w * float(pen)
    

    # ---------------------------
    # Dinámica discreta simple
    # ---------------------------
    def define_model(self, model):
        """
        model debe tener: step(pos, vel, action) -> (new_pos, new_vel, ...)
        """
        if not hasattr(model, "step"):
            raise TypeError("El modelo debe tener un método .step(pos, vel, action)")
        self.model = model
    
    def _step_dynamics(self, p, v, u):
        if self.model is None:
            # fallback a semi-implicit Euler
            v2 = v + u * self.dt
            p2 = p + v2 * self.dt
            return p2, v2

        out = self.model.step(p, v, u)
        # soporta (p2, v2) o (p2, v2, pos_var, vel_var, ...)
        return out[0], out[1]


    # ---------------------------
    # Funciones auxiliares
    # ---------------------------
    def clamp_vec(self, x, lo, hi):
        return np.minimum(np.maximum(x, lo), hi)
    
    def softplus(self,x):
        # estable numéricamente
        return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)

    # ---------------------------
    # MPPI core
    # ---------------------------
    def compute_action(self, p0, v0):
        p0 = np.array(p0, dtype=float)
        v0 = np.array(v0, dtype=float)

        # Si ya estás cerca del goal, apaga aceleración (suaviza)
        if np.linalg.norm(p0 - self.goal) < self.goal_tolerance:
            self.F_nom[:] = 0.0
            return np.zeros(3), p0, v0

        # Muestreo de ruido: (K, H, 3)
        eps = self.rng.normal(0.0, 1.0, size=(self.K, self.H, 3)) * self.noise_sigma

        # Construir controles muestreados F = clamp(F_nom + eps)
        Fsamp = self.F_nom[None, :, :] + eps
        Fsamp = self.clamp_vec(Fsamp, self.F_min[None, None, :], self.F_max[None, None, :])

        costs = np.zeros(self.K, dtype=float) # costo total de cada rollout

        # Rollouts
        for k in range(self.K):
            p = p0.copy()
            v = v0.copy()
            J = 0.0 # costo total del rollout
            F_prev = None
            for t in range(self.H):
                F = Fsamp[k, t]

                # costos de tracking + control
                # (puedes cambiar w_goal si quieres que sea más agresivo)
                dp = p - self.goal
                J += self.w_goal * np.dot(dp, dp) # tracking al goal
                J += self.w_F * np.dot(F, F)    # costo de control (penaliza fuerza grande)

                if F_prev is not None:
                    dF = (F - F_prev)
                    J += self.w_smooth * np.dot(dF, dF) # penaliza cambios bruscos (smoothness)
                F_prev = F

                # obstáculo
                J += self._obstacle_cost(p) # costo de obstáculos en cada paso

                # altura
                J += self._height_cost(p[2], terminal=False) # costo por altura en cada paso

                # dinámica
                p, v = self._step_dynamics(p, v, F) # actualiza estado con la dinámica (modelo)

            # costo terminal
            J += self.w_terminal * np.dot(p - self.goal, p - self.goal) # costo terminal (tracking final)
            J += self._obstacle_cost(p) # costo de obstáculos en estado final

            costs[k] = J # costo total del rollout k

        # Pesos MPPI
        beta = np.min(costs) # baseline para estabilidad numérica
        w = np.exp(-(costs - beta) / self.lambda_) # pesos sin normalizar
        w_sum = np.sum(w) + 1e-12 # evitar división por cero
        w = w / w_sum # normalizar pesos

        # Actualiza secuencia nominal F_nom con promedio ponderado
        # F_nom <- sum_k w_k * Fsamp_k
        self.F_nom = np.tensordot(w, Fsamp, axes=(0, 0)) # nuevo plan nominal (H x 3)

        # Acción receding-horizon: primer control
        F0 = self.F_nom[0].copy()

        # Shift del plan (para el próximo step)
        self.F_nom[:-1] = self.F_nom[1:] # shift left
        self.F_nom[-1] = self.F_nom[-2] * 0.5  # decaimiento suave

        # Predicción 1-step (para p_d, v_d)
        p1, v1 = self._step_dynamics(p0, v0, F0)
        return F0, p1, v1
    



##############################################################################################
##############################################################################################
################ Single MPPI with torch (para comparación) ###################################
##############################################################################################
##############################################################################################

import numpy as np
import torch


class SingleMPPIPlannerTorch:
    """
    MPPI para dron como punto-masa:
      x = [p(3), v(3)]
      u = a_cmd(3)  (aceleración deseada)
    Obstáculos: esferas y cajas AABB (axis-aligned).

    NOTA (Camino 1): aquí u es FUERZA en WORLD (N), aunque mantengo el naming "compute_action"
    para que encaje con tu código.
    """

    def __init__(
        self,
        dt: float = 0.1,
        horizon: int = 30,            # pasos
        num_samples: int = 300,       # rollouts
        lambda_: float = 1.0,         # temperatura MPPI (exploración)
        noise_sigma: np.ndarray = np.array([2.0, 2.0, 4.0]),  # ruido en Newton step (N)
        F_min: np.ndarray = np.array([-8.0, -8.0,  0.0]),
        F_max: np.ndarray = np.array([ 8.0,  8.0,  25.0]),
        w_goal: float = 6.0,          # goal weighting (tracking)
        w_terminal: float = 50.0,     # terminal cost weighting
        w_F: float = 0.05,            # control cost weighting
        w_smooth: float = 0.02,       # smoothness cost weighting (delta u)
        w_obs: float = 80.0,          # obstacle cost weighting
        obs_margin: float = 0.20,     # margen de seguridad para obstáculos
        obs_softness: float = 0.15,   # suavidad de la penalización de obstáculos
        goal_tolerance: float = 0.10, # distancia al goal para considerar "llegado"

        z_min: float = 0.5,           # altura mínima (suelo)
        z_max: float = 2.0,           # altura máxima (techo)
        z_margin: float = 0.15,       # “zona amarilla” cerca de límites
        w_z: float = 200.0,           # peso altura durante rollou
        w_z_terminal: float = 400.0,  # peso terminal altura

        v_max: float = 5.0,            # velocidad máxima (m/s)
        v_margin: float = 0.5,         # margen para penalizar velocidad alta
        w_v: float = 50.0,             # peso para penalizar velocidad alta
        w_v_terminal: float = 100.0,     # peso terminal para velocidad alta

        rng_seed: int = 0,
        device: str | None = None,    # "cuda" / "cpu" / None => auto
        dtype: torch.dtype = torch.float32,
    ):
        self.dt = float(dt)
        self.H = int(horizon)
        self.K = int(num_samples)
        self.lambda_ = float(lambda_)

        # device / dtype
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype

        # numpy copies (por compatibilidad / debug)
        self.noise_sigma_np = noise_sigma.astype(float).reshape(3,)
        self.F_min_np = F_min.astype(float).reshape(3,)
        self.F_max_np = F_max.astype(float).reshape(3,)

        # torch buffers
        self.noise_sigma = torch.as_tensor(self.noise_sigma_np, device=self.device, dtype=self.dtype)
        self.F_min = torch.as_tensor(self.F_min_np, device=self.device, dtype=self.dtype)
        self.F_max = torch.as_tensor(self.F_max_np, device=self.device, dtype=self.dtype)

        self.w_goal = float(w_goal)
        self.w_terminal = float(w_terminal)
        self.w_F = float(w_F)
        self.w_smooth = float(w_smooth)
        self.w_obs = float(w_obs)
        self.obs_margin = float(obs_margin)
        self.obs_softness = float(obs_softness)
        self.goal_tolerance = float(goal_tolerance)

        # altura
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.z_margin = float(z_margin)
        self.w_z = float(w_z)
        self.w_z_terminal = float(w_z_terminal)

        # velocidad
        self.v_max = float(v_max)
        self.v_margin = float(v_margin)
        self.w_v = float(w_v)
        self.w_v_terminal = float(w_v_terminal)

        # goal
        self.goal = torch.zeros(3, device=self.device, dtype=self.dtype)

        # model
        # Para GPU de verdad, el modelo debe operar en torch y en el mismo device.
        # Si le pasas un modelo numpy, caerá a CPU y perderás el beneficio.
        self.model = None  # <-- por defecto no hay modelo

        # Secuencia nominal de fuerzas (H x 3)
        self.F_nom = torch.zeros((self.H, 3), device=self.device, dtype=self.dtype)

        # Obstáculos: guardados como tensores para vectorizar
        self.obstacles = []
        self._obs_tensors_built = False
        self._spheres_c = None  # (Ns,3)
        self._spheres_r = None  # (Ns,)
        self._boxes_c = None    # (Nb,3)
        self._boxes_h = None    # (Nb,3)

        # RNG torch (para reproducibilidad en GPU/CPU)
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(rng_seed))

    def set_goal(self, goal_xyz):
        self.goal = torch.as_tensor(np.array(goal_xyz, dtype=float).reshape(3,), device=self.device, dtype=self.dtype)

    def set_obstacles(self, obstacles):
        """
        obstacles: lista de dicts, ejemplos:
          {"type":"sphere", "c":[x,y,z], "r":0.25}
          {"type":"box", "c":[x,y,z], "h":[hx,hy,hz]}   # AABB half-sizes
        """
        self.obstacles = obstacles
        self._obs_tensors_built = False  # rebuild lazy

    # ---------------------------
    # Distancias a obstáculos (torch, vectorizado)
    # ---------------------------
    def _build_obstacle_tensors(self):
        spheres_c, spheres_r = [], []
        boxes_c, boxes_h = [], []

        for obs in self.obstacles:
            if obs["type"] == "sphere":
                spheres_c.append(obs["c"])
                spheres_r.append(obs["r"])
            elif obs["type"] == "box":
                boxes_c.append(obs["c"])
                boxes_h.append(obs["h"])

        if spheres_c:
            self._spheres_c = torch.as_tensor(np.array(spheres_c, float), device=self.device, dtype=self.dtype)  # (Ns,3)
            self._spheres_r = torch.as_tensor(np.array(spheres_r, float), device=self.device, dtype=self.dtype)  # (Ns,)
        else:
            self._spheres_c = None
            self._spheres_r = None

        if boxes_c:
            self._boxes_c = torch.as_tensor(np.array(boxes_c, float), device=self.device, dtype=self.dtype)  # (Nb,3)
            self._boxes_h = torch.as_tensor(np.array(boxes_h, float), device=self.device, dtype=self.dtype)  # (Nb,3)
        else:
            self._boxes_c = None
            self._boxes_h = None

        self._obs_tensors_built = True

    def _sphere_penalty(self, p, c, r):
        # p: (...,3), c:(Ns,3), r:(Ns,)
        # distancia al borde (positiva fuera)
        d = torch.linalg.norm(p.unsqueeze(-2) - c, dim=-1) - (r + self.obs_margin)  # (..., Ns)
        return torch.exp(-d / self.obs_softness)  # (..., Ns)

    def _box_penalty(self, p, c, h):
        # AABB penalty vectorizado
        # p: (...,3), c:(Nb,3), h:(Nb,3)
        q = torch.abs(p.unsqueeze(-2) - c) - h - self.obs_margin  # (..., Nb, 3)
        outside = torch.clamp(q, min=0.0)                         # (..., Nb, 3)
        d_out = torch.linalg.norm(outside, dim=-1)                # (..., Nb)
        inside = (q <= 0.0).all(dim=-1)                           # (..., Nb)
        # dentro => penaliza muy fuerte
        pen_inside = torch.exp(torch.tensor(2.0 / self.obs_softness, device=self.device, dtype=self.dtype))
        pen_outside = torch.exp(-d_out / self.obs_softness)
        return torch.where(inside, pen_inside, pen_outside)       # (..., Nb)

    def _obstacle_cost(self, p):
        # p: (...,3)
        if not self.obstacles:
            return torch.zeros(p.shape[:-1], device=self.device, dtype=self.dtype)

        if not self._obs_tensors_built:
            self._build_obstacle_tensors()

        cost = torch.zeros(p.shape[:-1], device=self.device, dtype=self.dtype)

        if self._spheres_c is not None:
            cost = cost + self._sphere_penalty(p, self._spheres_c, self._spheres_r).sum(dim=-1)

        if self._boxes_c is not None:
            cost = cost + self._box_penalty(p, self._boxes_c, self._boxes_h).sum(dim=-1)

        return self.w_obs * cost

    # ---------------------------
    # Costo por rango de altura (torch)
    # ---------------------------
    def softplus(self, x):
        # torch.softplus es estable y rápida
        return torch.nn.functional.softplus(x)

    def _height_cost(self, z, terminal: bool = False):
        """
        Penaliza salir de [z_min, z_max]. Dentro del rango => ~0.
        En la zona de margen, empieza a subir suave.
        Fuera del rango, sube fuerte.
        """
        low = (self.z_min + self.z_margin) - z
        high = z - (self.z_max - self.z_margin)
        pen = self.softplus(low) ** 2 + self.softplus(high) ** 2
        w = self.w_z_terminal if terminal else self.w_z
        return w * pen
    
    # ---------------------------
    # Costo por rango de velocidad (torch)
    # ---------------------------
    def _velocity_cost(self, v, terminal=False):
        """
        Penaliza cuando ||v|| > v_max.
        """
        speed = torch.linalg.norm(v, dim=-1)

        excess = speed - (self.v_max - self.v_margin)
        pen = torch.nn.functional.softplus(excess) ** 2

        w = self.w_v_terminal if terminal else self.w_v
        return w * pen


    # ---------------------------
    # Dinámica discreta simple (torch)
    # ---------------------------
    def define_model(self, model):
        """
        model debe tener: step(pos, vel, action) -> (new_pos, new_vel, ...)
        Para GPU real, step debe aceptar/retornar torch tensors en self.device.
        """
        if not hasattr(model, "step"):
            raise TypeError("El modelo debe tener un método .step(pos, vel, action)")
        self.model = model

    def _step_dynamics(self, p, v, F):
        """
        p,v,F: (...,3) torch
        """
        if self.model is None:
            # fallback a semi-implicit Euler (sin g ni masa)
            v2 = v + F * self.dt
            p2 = p + v2 * self.dt
            return p2, v2

        out = self.model.step(p, v, F)
        # soporta (p2, v2) o (p2, v2, ...)
        return out[0], out[1]

    # ---------------------------
    # Funciones auxiliares
    # ---------------------------
    def clamp_vec(self, x, lo, hi):
        return torch.clamp(x, min=lo, max=hi)

    # ---------------------------
    # MPPI core (GPU torch, vectorizado)
    # ---------------------------
    @torch.no_grad()
    def compute_action(self, p0, v0):
        """
        p0, v0 pueden ser numpy o torch. Devuelve numpy por compatibilidad:
          F0 (3,), p1 (3,), v1 (3,)
        """
        # a torch
        p0 = torch.as_tensor(np.array(p0, dtype=float).reshape(3,), device=self.device, dtype=self.dtype)
        v0 = torch.as_tensor(np.array(v0, dtype=float).reshape(3,), device=self.device, dtype=self.dtype)

        # Si ya estás cerca del goal, apaga
        if torch.linalg.norm(p0 - self.goal).item() < self.goal_tolerance:
            self.F_nom.zero_()
            z = torch.zeros(3, device=self.device, dtype=self.dtype)
            return z.cpu().numpy(), p0.cpu().numpy(), v0.cpu().numpy()

        # Muestreo de ruido: (K, H, 3)
        eps = torch.randn((self.K, self.H, 3), generator=self.gen, device=self.device, dtype=self.dtype) * self.noise_sigma

        # Construir controles muestreados F = clamp(F_nom + eps)
        Fsamp = self.F_nom.unsqueeze(0) + eps  # (K,H,3)
        Fsamp = self.clamp_vec(Fsamp, self.F_min.view(1, 1, 3), self.F_max.view(1, 1, 3))

        # Rollouts vectorizados (K)
        p = p0.view(1, 3).repeat(self.K, 1)  # (K,3)
        v = v0.view(1, 3).repeat(self.K, 1)  # (K,3)

        costs = torch.zeros((self.K,), device=self.device, dtype=self.dtype)
        F_prev = None

        for t in range(self.H):
            F = Fsamp[:, t, :]  # (K,3)

            # tracking
            dp = p - self.goal.view(1, 3)
            costs = costs + self.w_goal * (dp * dp).sum(dim=-1)

            # control effort
            costs = costs + self.w_F * (F * F).sum(dim=-1)

            # smoothness
            if F_prev is not None:
                dF = F - F_prev
                costs = costs + self.w_smooth * (dF * dF).sum(dim=-1)
            F_prev = F

            # obstáculo
            costs = costs + self._obstacle_cost(p)

            # altura
            costs = costs + self._height_cost(p[:, 2], terminal=False)

            # velocidad
            costs = costs + self._velocity_cost(v, terminal=False)

            # dinámica
            p, v = self._step_dynamics(p, v, F)

        # costo terminal
        dpT = p - self.goal.view(1, 3)
        costs = costs + self.w_terminal * (dpT * dpT).sum(dim=-1)
        costs = costs + self._obstacle_cost(p)
        costs = costs + self._height_cost(p[:, 2], terminal=True)

        # Pesos MPPI
        beta = torch.min(costs)
        w = torch.exp(-(costs - beta) / self.lambda_)
        w = w / (torch.sum(w) + 1e-12)

        # Actualiza secuencia nominal F_nom con promedio ponderado
        # F_nom <- sum_k w_k * Fsamp_k
        self.F_nom = torch.tensordot(w, Fsamp, dims=([0], [0]))  # (H,3)

        # Acción receding-horizon: primer control
        F0 = self.F_nom[0].clone()
        # Accion completa (para debug / comparación): toda la secuencia nominal antes del shift
        F_seq = self.F_nom.clone()   # (H,3) en torch

        # Shift del plan (para el próximo step)
        self.F_nom = torch.roll(self.F_nom, shifts=-1, dims=0)
        self.F_nom[-1] = 0.5 * self.F_nom[-2]

        # Predicción 1-step
        p1, v1 = self._step_dynamics(p0, v0, F0)

        return F0.cpu().numpy(), p1.cpu().numpy(), v1.cpu().numpy(), F_seq.cpu().numpy()








##############################################################################################
##############################################################################################
################################## Multi MPPI with torch #####################################
##############################################################################################
##############################################################################################

import numpy as np
import torch


class MultiMPPIPlannerTorch:
    """
    MPPI para N drones como punto-masa (multi-agente):
      x = [p(N,3), v(N,3)]
      u = F_cmd(N,3)  (fuerza deseada en WORLD, Newtons)

    Obstáculos: esferas y cajas AABB (axis-aligned).
    + Restricción suave de altura por dron (z_min/z_max)
    + Límite suave de velocidad por dron (v_max)
    + Evitación entre drones (distancia mínima)
    """

    def __init__(
        self,
        N: int,
        dt: float = 0.1,
        horizon: int = 30,            # pasos
        num_samples: int = 512,       # rollouts
        lambda_: float = 1.0,         # temperatura MPPI

        # Fuerza (WORLD, N) por componente
        noise_sigma: np.ndarray = np.array([2.0, 2.0, 4.0]),   # N (por dron)
        F_min: np.ndarray = np.array([-8.0, -8.0, 0.0]),       # N
        F_max: np.ndarray = np.array([ 8.0,  8.0, 25.0]),      # N

        # Costos
        w_goal: float = 6.0,
        w_terminal: float = 50.0,
        w_F: float = 0.05,
        w_smooth: float = 0.02,
        w_obs: float = 80.0,

        # Obstáculos
        obs_margin: float = 0.20,
        obs_softness: float = 0.15,

        # “llegado”
        goal_tolerance: float = 0.10,

        # Altura (por dron)
        z_min: float = 0.5,
        z_max: float = 2.0,
        z_margin: float = 0.15,
        w_z: float = 200.0,
        w_z_terminal: float = 400.0,

        # Velocidad max (por dron)
        v_max: float = 3.0,
        v_margin: float = 0.30,
        w_v: float = 50.0,
        w_v_terminal: float = 100.0,

        # Evitación entre drones
        agent_radius: float = 0.10,   # radio efectivo por dron (m)
        sep_margin: float = 0.10,     # margen adicional (m)
        sep_softness: float = 0.10,   # suavidad penalización
        w_sep: float = 200.0,         # peso de separación

        # Torch
        rng_seed: int = 0,
        device: str | None = None,    # "cuda"/"cpu"/None(auto)
        dtype: torch.dtype = torch.float32,
    ):
        self.N = int(N)
        self.dt = float(dt)
        self.H = int(horizon)
        self.K = int(num_samples)
        self.lambda_ = float(lambda_)

        # device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.dtype = dtype

        # parámetros (floats)
        self.w_goal = float(w_goal)
        self.w_terminal = float(w_terminal)
        self.w_F = float(w_F)
        self.w_smooth = float(w_smooth)
        self.w_obs = float(w_obs)
        self.obs_margin = float(obs_margin)
        self.obs_softness = float(obs_softness)
        self.goal_tolerance = float(goal_tolerance)

        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.z_margin = float(z_margin)
        self.w_z = float(w_z)
        self.w_z_terminal = float(w_z_terminal)

        self.v_max = float(v_max)
        self.v_margin = float(v_margin)
        self.w_v = float(w_v)
        self.w_v_terminal = float(w_v_terminal)

        self.agent_radius = float(agent_radius)
        self.sep_margin = float(sep_margin)
        self.sep_softness = float(sep_softness)
        self.w_sep = float(w_sep)

        # tensores de límites y ruido (por dron, 3)
        noise_sigma = np.array(noise_sigma, float).reshape(3,)
        F_min = np.array(F_min, float).reshape(3,)
        F_max = np.array(F_max, float).reshape(3,)

        self.noise_sigma = torch.as_tensor(noise_sigma, device=self.device, dtype=self.dtype)  # (3,)
        self.F_min = torch.as_tensor(F_min, device=self.device, dtype=self.dtype)            # (3,)
        self.F_max = torch.as_tensor(F_max, device=self.device, dtype=self.dtype)            # (3,)

        # goals: (N,3)
        self.goal = torch.zeros((self.N, 3), device=self.device, dtype=self.dtype)

        # modelo acoplado (torch recomendado)
        self.model = None

        # Secuencia nominal: (H, N, 3)
        self.F_nom = torch.zeros((self.H, self.N, 3), device=self.device, dtype=self.dtype)

        # obstáculos
        self.obstacles = []
        self._obs_built = False
        self._spheres_c = None  # (Ns,3)
        self._spheres_r = None  # (Ns,)
        self._boxes_c = None    # (Nb,3)
        self._boxes_h = None    # (Nb,3)

        # RNG
        self.gen = torch.Generator(device=self.device)
        self.gen.manual_seed(int(rng_seed))

    # ---------------------------
    # Setters
    # ---------------------------
    def set_goals(self, goals_xyz):
        """
        goals_xyz: (N,3) array-like
        """
        g = np.array(goals_xyz, dtype=float)
        assert g.shape == (self.N, 3)
        self.goal = torch.as_tensor(g, device=self.device, dtype=self.dtype)

    def set_obstacles(self, obstacles):
        """
        obstacles: lista de dicts:
          {"type":"sphere", "c":[x,y,z], "r":0.25}
          {"type":"box",    "c":[x,y,z], "h":[hx,hy,hz]}
        """
        self.obstacles = obstacles
        self._obs_built = False

    def define_model(self, model):
        """
        model debe tener:
          step(p, v, F) -> (p_next, v_next, ...)
        donde p,v,F son torch tensors en self.device.
        Shapes esperadas:
          p: (K,N,3) o (N,3)
          v: (K,N,3) o (N,3)
          F: (K,N,3) o (N,3)
        """
        if not hasattr(model, "step"):
            raise TypeError("El modelo debe tener un método .step(p, v, F)")
        self.model = model

    # ---------------------------
    # Aux
    # ---------------------------
    def _clamp_vec(self, x, lo, hi):
        return torch.clamp(x, min=lo, max=hi)

    def _softplus(self, x):
        return torch.nn.functional.softplus(x)

    # ---------------------------
    # Obstáculos (vectorizado)
    # ---------------------------
    def _build_obstacle_tensors(self):
        spheres_c, spheres_r = [], []
        boxes_c, boxes_h = [], []

        for obs in self.obstacles:
            if obs["type"] == "sphere":
                spheres_c.append(obs["c"])
                spheres_r.append(obs["r"])
            elif obs["type"] == "box":
                boxes_c.append(obs["c"])
                boxes_h.append(obs["h"])

        if spheres_c:
            self._spheres_c = torch.as_tensor(np.array(spheres_c, float), device=self.device, dtype=self.dtype)
            self._spheres_r = torch.as_tensor(np.array(spheres_r, float), device=self.device, dtype=self.dtype)
        else:
            self._spheres_c = None
            self._spheres_r = None

        if boxes_c:
            self._boxes_c = torch.as_tensor(np.array(boxes_c, float), device=self.device, dtype=self.dtype)
            self._boxes_h = torch.as_tensor(np.array(boxes_h, float), device=self.device, dtype=self.dtype)
        else:
            self._boxes_c = None
            self._boxes_h = None

        self._obs_built = True

    def _sphere_penalty(self, p, c, r):
        # p: (K,N,3), c:(Ns,3), r:(Ns,)
        d = torch.linalg.norm(p.unsqueeze(-2) - c, dim=-1) - (r + self.obs_margin)  # (K,N,Ns)
        return torch.exp(-d / self.obs_softness)

    def _box_penalty(self, p, c, h):
        # p: (K,N,3), c:(Nb,3), h:(Nb,3)
        q = torch.abs(p.unsqueeze(-2) - c) - h - self.obs_margin  # (K,N,Nb,3)
        outside = torch.clamp(q, min=0.0)
        d_out = torch.linalg.norm(outside, dim=-1)               # (K,N,Nb)
        inside = (q <= 0.0).all(dim=-1)                          # (K,N,Nb)
        pen_inside = torch.exp(torch.tensor(2.0 / self.obs_softness, device=self.device, dtype=self.dtype))
        pen_outside = torch.exp(-d_out / self.obs_softness)
        return torch.where(inside, pen_inside, pen_outside)

    def _obstacle_cost(self, p):
        # p: (K,N,3)
        if not self.obstacles:
            return torch.zeros((p.shape[0],), device=self.device, dtype=self.dtype)

        if not self._obs_built:
            self._build_obstacle_tensors()

        cost = torch.zeros((p.shape[0],), device=self.device, dtype=self.dtype)  # (K,)

        if self._spheres_c is not None:
            cost = cost + self._sphere_penalty(p, self._spheres_c, self._spheres_r).sum(dim=(-1, -2))  # sum over N and Ns

        if self._boxes_c is not None:
            cost = cost + self._box_penalty(p, self._boxes_c, self._boxes_h).sum(dim=(-1, -2))          # sum over N and Nb

        return self.w_obs * cost

    # ---------------------------
    # Altura / velocidad (por dron)
    # ---------------------------
    def _height_cost(self, z, terminal=False):
        # z: (K,N)
        low = (self.z_min + self.z_margin) - z
        high = z - (self.z_max - self.z_margin)
        pen = self._softplus(low) ** 2 + self._softplus(high) ** 2
        w = self.w_z_terminal if terminal else self.w_z
        return w * pen.sum(dim=-1)  # sum over drones -> (K,)

    def _velocity_cost(self, v, terminal=False):
        # v: (K,N,3)
        speed = torch.linalg.norm(v, dim=-1)  # (K,N)
        excess = speed - (self.v_max - self.v_margin)
        pen = self._softplus(excess) ** 2
        w = self.w_v_terminal if terminal else self.w_v
        return w * pen.sum(dim=-1)  # (K,)

    # ---------------------------
    # Separación entre drones (evitar colisión)
    # ---------------------------
    def _separation_cost(self, p):
        # p: (K,N,3)
        if self.N < 2:
            return torch.zeros((p.shape[0],), device=self.device, dtype=self.dtype)

        # dist(i,j)
        diff = p.unsqueeze(2) - p.unsqueeze(1)        # (K,N,N,3)
        dist = torch.linalg.norm(diff, dim=-1)        # (K,N,N)

        # máscara para i!=j
        eye = torch.eye(self.N, device=self.device, dtype=torch.bool).unsqueeze(0)  # (1,N,N)
        dist = torch.where(eye, torch.full_like(dist, 1e6), dist)

        d_min = (2.0 * self.agent_radius) + self.sep_margin
        # penaliza si dist < d_min (softplus sobre d_min - dist)
        pen = self._softplus((d_min - dist) / self.sep_softness) ** 2  # (K,N,N)

        # cada par se cuenta dos veces; dividimos por 2
        return self.w_sep * 0.5 * pen.sum(dim=(1, 2))  # (K,)

    # ---------------------------
    # Dinámica
    # ---------------------------
    def _step_dynamics(self, p, v, F):
        if self.model is None:
            # fallback simple (sin masa/gravedad): NO recomendado para drones reales
            v2 = v + F * self.dt
            p2 = p + v2 * self.dt
            return p2, v2
        out = self.model.step(p, v, F)
        return out[0], out[1]

    # ---------------------------
    # MPPI core
    # ---------------------------
    @torch.no_grad()
    def compute_action(self, p0, v0, return_sequence: bool = False):
        """
        p0, v0: (N,3) array-like (numpy) o torch
        Retorna:
          F0: (N,3) numpy
          p1: (N,3) numpy
          v1: (N,3) numpy
          opcional: F_seq (H,N,3) numpy
        """
        p0 = torch.as_tensor(np.array(p0, float), device=self.device, dtype=self.dtype).reshape(self.N, 3)
        v0 = torch.as_tensor(np.array(v0, float), device=self.device, dtype=self.dtype).reshape(self.N, 3)

        # si todos los drones están cerca de sus goals -> apaga
        dist = torch.linalg.norm(p0 - self.goal, dim=-1)  # (N,)
        if torch.all(dist < self.goal_tolerance):
            self.F_nom.zero_()
            z = torch.zeros((self.N, 3), device=self.device, dtype=self.dtype)
            if return_sequence:
                return z.cpu().numpy(), p0.cpu().numpy(), v0.cpu().numpy(), self.F_nom.cpu().numpy()
            return z.cpu().numpy(), p0.cpu().numpy(), v0.cpu().numpy()

        # ruido: (K,H,N,3)
        eps = torch.randn((self.K, self.H, self.N, 3), generator=self.gen, device=self.device, dtype=self.dtype)
        eps = eps * self.noise_sigma.view(1, 1, 1, 3)

        # muestras: (K,H,N,3)
        Fsamp = self.F_nom.unsqueeze(0) + eps
        Fsamp = self._clamp_vec(Fsamp,
                                self.F_min.view(1, 1, 1, 3),
                                self.F_max.view(1, 1, 1, 3))

        # rollouts batched en K
        p = p0.unsqueeze(0).repeat(self.K, 1, 1)  # (K,N,3)
        v = v0.unsqueeze(0).repeat(self.K, 1, 1)  # (K,N,3)

        costs = torch.zeros((self.K,), device=self.device, dtype=self.dtype)
        F_prev = None

        for t in range(self.H):
            F = Fsamp[:, t, :, :]  # (K,N,3)

            # tracking (sum over drones)
            dp = p - self.goal.unsqueeze(0)                   # (K,N,3)
            costs = costs + self.w_goal * (dp * dp).sum(dim=(1, 2))

            # esfuerzo
            costs = costs + self.w_F * (F * F).sum(dim=(1, 2))

            # smoothness
            if F_prev is not None:
                dF = F - F_prev
                costs = costs + self.w_smooth * (dF * dF).sum(dim=(1, 2))
            F_prev = F

            # obstáculos / altura / velocidad / separación
            costs = costs + self._obstacle_cost(p)
            costs = costs + self._height_cost(p[:, :, 2], terminal=False)
            costs = costs + self._velocity_cost(v, terminal=False)
            costs = costs + self._separation_cost(p)

            # dinámica
            p, v = self._step_dynamics(p, v, F)

        # terminal
        dpT = p - self.goal.unsqueeze(0)
        costs = costs + self.w_terminal * (dpT * dpT).sum(dim=(1, 2))
        costs = costs + self._obstacle_cost(p)
        costs = costs + self._height_cost(p[:, :, 2], terminal=True)
        costs = costs + self._velocity_cost(v, terminal=True)
        costs = costs + self._separation_cost(p)

        # pesos MPPI
        beta = torch.min(costs)
        w = torch.exp(-(costs - beta) / self.lambda_)
        w = w / (torch.sum(w) + 1e-12)

        # update nominal: (H,N,3)
        self.F_nom = torch.tensordot(w, Fsamp, dims=([0], [0]))

        # primer control
        F0 = self.F_nom[0].clone() # F0: (N,3)

        # secuencia nominal completa (para debug / comparación)
        F_seq = self.F_nom.clone()  # (H,N,3) en torch

        # shift (evita solapamiento)
        self.F_nom[:-1] = self.F_nom[1:].clone()
        self.F_nom[-1] = 0.5 * self.F_nom[-2]

        # predicción 1-step
        p1, v1 = self._step_dynamics(p0, v0, F0)

        if return_sequence:
            return F0.cpu().numpy(), p1.cpu().numpy(), v1.cpu().numpy(), F_seq.cpu().numpy()
        return F0.cpu().numpy(), p1.cpu().numpy(), v1.cpu().numpy()

