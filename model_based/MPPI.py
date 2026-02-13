import numpy as np

def clamp_vec(x, lo, hi):
    return np.minimum(np.maximum(x, lo), hi)

class SingleMPPIPlanner:
    """
    MPPI para dron como punto-masa:
      x = [p(3), v(3)]
      u = a_cmd(3)  (aceleración deseada)
    Obstáculos: esferas y cajas AABB (axis-aligned).
    """

    def __init__(
        self,
        dt,
        horizon=30,           # pasos
        num_samples=300,      # rollouts
        lambda_=1.0,
        noise_sigma=np.array([2.0, 2.0, 2.0]),  # ruido en aceleración
        u_min=np.array([-6.0, -6.0, -4.0]),
        u_max=np.array([ 6.0,  6.0,  4.0]),
        w_goal=6.0,
        w_terminal=50.0,
        w_u=0.05,
        w_smooth=0.02,
        w_obs=80.0,
        obs_margin=0.20,
        obs_softness=0.15,
        goal_tolerance=0.10
    ):
        self.dt = float(dt)
        self.H = int(horizon)
        self.K = int(num_samples)
        self.lambda_ = float(lambda_)

        self.noise_sigma = noise_sigma.astype(float)
        self.u_min = u_min.astype(float)
        self.u_max = u_max.astype(float)

        self.w_goal = float(w_goal)
        self.w_terminal = float(w_terminal)
        self.w_u = float(w_u)
        self.w_smooth = float(w_smooth)
        self.w_obs = float(w_obs)
        self.obs_margin = float(obs_margin)
        self.obs_softness = float(obs_softness)
        self.goal_tolerance = float(goal_tolerance)

        self.goal = np.zeros(3, dtype=float)

        # Secuencia nominal (H x 3)
        self.u_nom = np.zeros((self.H, 3), dtype=float)

        # Obstáculos: lista de dicts
        self.obstacles = []

        # RNG
        self.rng = np.random.default_rng(0)

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
    # Dinámica discreta simple
    # ---------------------------
    def _step_dynamics(self, p, v, u):
        # semi-implicit Euler
        v2 = v + u * self.dt
        p2 = p + v2 * self.dt
        return p2, v2

    # ---------------------------
    # MPPI core
    # ---------------------------
    def compute_action(self, p0, v0):
        p0 = np.array(p0, dtype=float)
        v0 = np.array(v0, dtype=float)

        # Si ya estás cerca del goal, apaga aceleración (suaviza)
        if np.linalg.norm(p0 - self.goal) < self.goal_tolerance:
            self.u_nom[:] = 0.0
            return np.zeros(3), p0, v0

        # Muestreo de ruido: (K, H, 3)
        eps = self.rng.normal(0.0, 1.0, size=(self.K, self.H, 3)) * self.noise_sigma

        # Construir controles muestreados u = clamp(u_nom + eps)
        U = self.u_nom[None, :, :] + eps
        U = clamp_vec(U, self.u_min[None, None, :], self.u_max[None, None, :])

        costs = np.zeros(self.K, dtype=float)

        # Rollouts
        for k in range(self.K):
            p = p0.copy()
            v = v0.copy()
            J = 0.0
            u_prev = None
            for t in range(self.H):
                u = U[k, t]

                # costos de tracking + control
                # (puedes cambiar w_goal si quieres que sea más agresivo)
                J += self.w_goal * np.dot(p - self.goal, p - self.goal)
                J += self.w_u * np.dot(u, u)

                if u_prev is not None:
                    du = (u - u_prev)
                    J += self.w_smooth * np.dot(du, du)
                u_prev = u

                # obstáculo
                J += self._obstacle_cost(p)

                # dinámica
                p, v = self._step_dynamics(p, v, u)

            # costo terminal
            J += self.w_terminal * np.dot(p - self.goal, p - self.goal)
            J += self._obstacle_cost(p)

            costs[k] = J

        # Pesos MPPI
        beta = np.min(costs)
        w = np.exp(-(costs - beta) / self.lambda_)
        w_sum = np.sum(w) + 1e-12
        w = w / w_sum

        # Actualiza u_nom con promedio ponderado
        # u_nom <- sum_k w_k * U_k
        self.u_nom = np.tensordot(w, U, axes=(0, 0))

        # Acción receding-horizon: primer control
        u0 = self.u_nom[0].copy()

        # Shift del plan (para el próximo step)
        self.u_nom[:-1] = self.u_nom[1:]
        self.u_nom[-1] = self.u_nom[-2] * 0.5  # decaimiento suave

        # Predicción 1-step (para p_d, v_d)
        p1, v1 = self._step_dynamics(p0, v0, u0)
        return u0, p1, v1
