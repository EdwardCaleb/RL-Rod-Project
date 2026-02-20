
# Definimos el modelo dinámico para el MPPI

import numpy as np

class SingleMassDynamicModel:
    def __init__(self, dt=0.1, mass=1.0):
        self.dt = dt
        self.mass = mass
        self.mg = np.array([0.0, 0.0, -9.81 * self.mass])  # Gravedad
    
    def step(self, pos, vel, action):
        # pos: [x, y, z]
        # vel: [vx, vy, vz]
        # action: [Fx, Fy, Fz]
        
        a = (action + self.mg) / self.mass
        new_vel = vel + a * self.dt  # Update velocities
        new_pos = pos + vel * self.dt# + 0.5 * a * self.dt**2  # Update positions
        # uncertainty variance (simple model, could be more complex)
        pos_var = 0.01 * np.eye(3)  # Variance en posición
        vel_var = 0.01 * np.eye(3)  # Variance en velocidad
        return new_pos, new_vel, pos_var, vel_var


##############################################################################
##############################################################################
########################### Modelo dinámico con TORCH ########################
##############################################################################
##############################################################################
import numpy as np
import torch

class SingleMassDynamicModelTorch:
    def __init__(self, dt=0.1, mass=1.0, device="cuda", dtype=torch.float32):
        self.dt = float(dt)
        self.mass = float(mass)
        self.device = torch.device(device)
        self.dtype = dtype
        self.mg = torch.tensor([0.0, 0.0, -9.81 * self.mass], device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def step(self, pos, vel, action):
        # pos, vel, action pueden ser (3,) o (K,3)
        a = (action + self.mg) / self.mass
        new_vel = vel + a * self.dt
        new_pos = pos + vel * self.dt + 0.5 * a * (self.dt ** 2)
        return new_pos, new_vel





##############################################################################
##############################################################################
###################### Multi Modelo dinámico con TORCH #######################
##############################################################################
##############################################################################
import torch


class MultiBodySpringDamperDynamicsTorch:
    """
    Dinámica multi-body (N puntos-masa) con resortes/amortiguadores entre pares (edges).
    Todo en Torch (GPU friendly) y vectorizado en batch.

    Estado:
      p: (..., N, 3)   posiciones WORLD
      v: (..., N, 3)   velocidades WORLD
    Acción:
      F_ext: (..., N, 3) fuerzas externas WORLD (ej: MPPI por dron)

    Fuerzas internas (por edge i<->j):
      F_spring = k * (dist - L0) * dir
      F_damp   = c * ( (v_j - v_i) · dir ) * dir
      F_ij = (F_spring + F_damp)  aplicada a i (hacia j)
      F_ji = -F_ij aplicada a j

    Integración:
      a = (F_ext + F_internal + m*g)/m
      v_next = v + a*dt
      p_next = p + v*dt + 0.5*a*dt^2

    Escalable: N=2,3,... y batch K rollouts: p.shape puede ser (K,N,3).
    """

    def __init__(
        self,
        dt: float,
        mass: float | torch.Tensor,
        edges: list[tuple[int, int]],
        k: float | torch.Tensor = 20.0,         # stiffness (N/m)
        c: float | torch.Tensor = 2.0,          # damping (N*s/m)
        rest_length: float | torch.Tensor = 0.5,# L0 (m)
        gravity: float = 9.81,
        device: str | torch.device = "cuda",
        dtype: torch.dtype = torch.float32,
        eps: float = 1e-8,
    ):
        self.dt = float(dt)
        self.device = torch.device(device)
        self.dtype = dtype
        self.eps = float(eps)

        # masa: escalar o (N,)
        self.mass = torch.as_tensor(mass, device=self.device, dtype=self.dtype)
        self.g = torch.tensor([0.0, 0.0, -float(gravity)], device=self.device, dtype=self.dtype)

        # edges -> tensores (E,)
        if len(edges) == 0:
            raise ValueError("edges no puede estar vacío (define al menos un resorte).")
        ii, jj = zip(*edges)
        self.i = torch.tensor(ii, device=self.device, dtype=torch.long)
        self.j = torch.tensor(jj, device=self.device, dtype=torch.long)
        self.E = self.i.numel()

        # parámetros por edge: escalar o (E,)
        self.k = torch.as_tensor(k, device=self.device, dtype=self.dtype)
        self.c = torch.as_tensor(c, device=self.device, dtype=self.dtype)
        self.L0 = torch.as_tensor(rest_length, device=self.device, dtype=self.dtype)

        # los dejamos listos para broadcast por-edge
        if self.k.ndim == 0:  self.k = self.k.repeat(self.E)
        if self.c.ndim == 0:  self.c = self.c.repeat(self.E)
        if self.L0.ndim == 0: self.L0 = self.L0.repeat(self.E)

    @torch.no_grad()
    def step(self, pos, vel, action_F):
        """
        pos, vel, action_F pueden ser:
          - (N,3)
          - (K,N,3)

        Devuelve:
          new_pos, new_vel
        """
        pos = self._to_torch(pos)
        vel = self._to_torch(vel)
        Fext = self._to_torch(action_F)

        # soporta (N,3) y (K,N,3)
        batched = (pos.ndim == 3)
        if not batched:
            pos = pos.unsqueeze(0)  # (1,N,3)
            vel = vel.unsqueeze(0)
            Fext = Fext.unsqueeze(0)

        B, N, _ = pos.shape

        # masa -> (1,N,1) para broadcast
        if self.mass.ndim == 0:
            m = self.mass.view(1, 1, 1)
        else:
            # (N,) -> (1,N,1)
            m = self.mass.view(1, N, 1)

        # -------------------------
        # Fuerzas internas por edges (vectorizado)
        # -------------------------
        pi = pos[:, self.i, :]   # (B,E,3)
        pj = pos[:, self.j, :]   # (B,E,3)
        vi = vel[:, self.i, :]
        vj = vel[:, self.j, :]

        d = pj - pi                              # (B,E,3)
        dist = torch.linalg.norm(d, dim=-1)      # (B,E)
        dir_ = d / (dist.unsqueeze(-1) + self.eps)

        # spring: k*(dist-L0)
        stretch = dist - self.L0.view(1, self.E)                   # (B,E)
        F_s = (self.k.view(1, self.E) * stretch).unsqueeze(-1) * dir_  # (B,E,3)

        # damping: c*((v_rel · dir) ) dir
        vrel = vj - vi
        vrel_along = (vrel * dir_).sum(dim=-1)                     # (B,E)
        F_d = (self.c.view(1, self.E) * vrel_along).unsqueeze(-1) * dir_  # (B,E,3)

        F_edge = F_s + F_d                                         # fuerza sobre i hacia j (B,E,3)

        # -------------------------
        # Acumular fuerzas a nodos con scatter/index_add
        # -------------------------
        Fint = torch.zeros((B, N, 3), device=self.device, dtype=self.dtype)

        # sumar a i: +F_edge
        Fint.index_add_(1, self.i, F_edge)
        # sumar a j: -F_edge
        Fint.index_add_(1, self.j, -F_edge)

        # -------------------------
        # Aceleración total + integración
        # -------------------------
        # gravedad en fuerza: m*g
        Fg = m * self.g.view(1, 1, 3)            # (B,N,3) por broadcast

        Ftot = Fext + Fint + Fg
        a = Ftot / m

        dt = self.dt
        new_vel = vel + a * dt
        new_pos = pos + vel * dt + 0.5 * a * (dt * dt)

        if not batched:
            return new_pos.squeeze(0), new_vel.squeeze(0)
        return new_pos, new_vel

    def _to_torch(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(device=self.device, dtype=self.dtype)
        return torch.as_tensor(x, device=self.device, dtype=self.dtype)


# -------------------------
# Ejemplos de uso
# -------------------------
if __name__ == "__main__":
    # 2 cuerpos unidos
    dyn2 = MultiBodySpringDamperDynamicsTorch(
        dt=0.01,
        mass=0.3,
        edges=[(0, 1)],
        k=30.0,
        c=3.0,
        rest_length=0.5,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # 3 cuerpos en cadena 0-1-2
    dyn3 = MultiBodySpringDamperDynamicsTorch(
        dt=0.01,
        mass=torch.tensor([0.3, 0.3, 0.3]),
        edges=[(0, 1), (1, 2)],
        k=torch.tensor([30.0, 30.0]),
        c=torch.tensor([3.0, 3.0]),
        rest_length=torch.tensor([0.5, 0.5]),
        device="cuda" if torch.cuda.is_available() else "cpu",
    )





# if __name__ == "__main__":
#     # Test del modelo dinámico
#     model = SingleMassDynamicModel(dt=0.1, mass=1.0)
#     pos = np.array([0.0, 0.0, 0.0])
#     vel = np.array([0.0, 0.0, 0.0])
#     action = np.array([1.0, 0.0, 0.0])  # Fuerza en x
#     new_pos, new_vel, pos_var, vel_var = model.step(pos, vel, action)
#     print("New Position:", new_pos)
#     print("New Velocity:", new_vel)
#     print("Position Variance:\n", pos_var)
#     print("Velocity Variance:\n", vel_var)


# TEST DEL MODELO DINÁMICO EN TORCH
if __name__ == "__main__":
    model = SingleMassDynamicModelTorch(dt=0.1, mass=1.0, device="cuda")
    pos = torch.tensor([0.0, 0.0, 0.0], device=model.device, dtype=model.dtype)
    vel = torch.tensor([0.0, 0.0, 0.0], device=model.device, dtype=model.dtype)
    action = torch.tensor([1.0, 0.0, 0.0], device=model.device, dtype=model.dtype)
    new_pos, new_vel = model.step(pos, vel, action)
    print("New Position:", new_pos.cpu().numpy())
    print("New Velocity:", new_vel.cpu().numpy())

    #verificar que cuda se está usando
    print("Device used:", model.device)


