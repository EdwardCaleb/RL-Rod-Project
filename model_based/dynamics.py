
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
        new_pos = pos + vel * self.dt + 0.5 * a * self.dt**2  # Update positions
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
########################### Modelo dinámico con SVGP #########################
##############################################################################
##############################################################################

from SVGP import OnlineSVGP1D
import os

# ============================================================
# ---- Drone Dynamics with 6 independent SVGPs ---------------
# ============================================================
class SVGPDroneDynamicModel:
    """
    Learns dynamics:
      X = [p(3), v(3)]  (6)
      input Z = [Xprev(6), F(3)] (9)
      target Y = ΔX (6)  where ΔX = Xnext - Xprev

    Provides:
      step(pos, vel, force) -> new_pos, new_vel, pos_var(3), vel_var(3)

    Online training:
      add_transition(...) stores and (optionally) trains every N steps.
    """

    def __init__(
        self,
        dt: float = 0.1,
        mass: float = 1.0,
        gravity: float = 9.81,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,

        # GP hyperparams
        kernel: str = "RBF",
        lr: float = 0.01,
        batch_size: int = 256,
        num_inducing: int = 256,
        init_train_steps: int = 800,

        # Online schedule
        train_every: int = 20,
        online_steps: int = 50,
        min_points_to_train: int = 300,

        # If True, reset model+stats each episode (but can still load rollouts from disk)
        reset_each_episode: bool = False,
    ):
        self.dt = float(dt)
        self.mass = float(mass)
        self.g = float(gravity)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self.input_dim = 9
        self.out_dim = 6

        self.kernel = kernel
        self.lr = lr
        self.batch_size = batch_size
        self.num_inducing = num_inducing
        self.init_train_steps = init_train_steps

        self.train_every = int(train_every)
        self.online_steps = int(online_steps)
        self.min_points_to_train = int(min_points_to_train)
        self.reset_each_episode = bool(reset_each_episode)

        # 6 independent SVGPs for Δx_i
        self.gps = [
            OnlineSVGP1D(
                input_dim=self.input_dim,
                kernel=self.kernel,
                lr=self.lr,
                batch_size=self.batch_size,
                num_inducing=self.num_inducing,
                init_train_steps=self.init_train_steps,
                device=self.device,
                dtype=self.dtype,
            )
            for _ in range(self.out_dim)
        ]

        # rollout buffers (numpy, for saving/loading)
        self.Z_buf = []   # each: (9,)
        self.dX_buf = []  # each: (6,)

        self._step_counter = 0
        self._trained_once = False

    # ---------- utils ----------
    def _fallback_step(self, pos: np.ndarray, vel: np.ndarray, force: np.ndarray):
        """
        Point-mass fallback:
          a = (F + [0,0,-mg]) / m
          x_{t+1} = x_t + ...
        """
        pos = np.asarray(pos, dtype=float).reshape(3,)
        vel = np.asarray(vel, dtype=float).reshape(3,)
        F = np.asarray(force, dtype=float).reshape(3,)

        a = (F + np.array([0.0, 0.0, -self.mass * self.g])) / self.mass
        v2 = vel + a * self.dt
        p2 = pos + vel * self.dt + 0.5 * a * (self.dt ** 2)

        pos_var = np.ones(3) * 1e-3
        vel_var = np.ones(3) * 1e-3
        return p2, v2, pos_var, vel_var

    def _make_input_Z(self, pos, vel, force) -> np.ndarray:
        pos = np.asarray(pos, dtype=float).reshape(3,)
        vel = np.asarray(vel, dtype=float).reshape(3,)
        F = np.asarray(force, dtype=float).reshape(3,)
        X = np.concatenate([pos, vel], axis=0)
        Z = np.concatenate([X, F], axis=0)  # (9,)
        return Z

    # ---------- public API ----------
    def step(self, pos, vel, action_force):
        """
        Predict next state and per-dimension variance.
        If not trained yet -> fallback physics.
        """
        if not self._trained_once or not all(gp.trained for gp in self.gps):
            return self._fallback_step(pos, vel, action_force)

        Z = self._make_input_Z(pos, vel, action_force)
        Zt = torch.as_tensor(Z[None, :], device=self.device, dtype=self.dtype)  # (1,9)

        dX_mean = np.zeros(6, dtype=float)
        dX_var = np.zeros(6, dtype=float)
        for i in range(6):
            m, v = self.gps[i].predict_torch(Zt)
            dX_mean[i] = float(m.item())
            dX_var[i] = float(v.item())

        X = np.concatenate([np.asarray(pos, float).reshape(3,), np.asarray(vel, float).reshape(3,)])
        X2 = X + dX_mean

        new_pos = X2[0:3]
        new_vel = X2[3:6]

        pos_var = dX_var[0:3]
        vel_var = dX_var[3:6]
        return new_pos, new_vel, pos_var, vel_var

    def add_transition(self, pos, vel, force, new_pos, new_vel, train_online: bool = True):
        """
        Store transition and optionally perform online training.
        Transition format:
          Z = [p,v,F]
          dX = [p2-p, v2-v]
        """
        Z = self._make_input_Z(pos, vel, force)
        X = np.concatenate([np.asarray(pos, float).reshape(3,), np.asarray(vel, float).reshape(3,)])
        X2 = np.concatenate([np.asarray(new_pos, float).reshape(3,), np.asarray(new_vel, float).reshape(3,)])
        dX = X2 - X

        self.Z_buf.append(Z.astype(np.float32))
        self.dX_buf.append(dX.astype(np.float32))

        self._step_counter += 1

        if not train_online:
            return

        # train every N steps, once enough data exists
        if (self._step_counter % self.train_every) == 0 and len(self.Z_buf) >= self.min_points_to_train:
            self.train_online(steps=self.online_steps)

    def train_full(self):
        """
        Full (re)train from scratch using current buffers.
        """
        if len(self.Z_buf) < self.min_points_to_train:
            return

        Z = np.stack(self.Z_buf, axis=0)   # (N,9)
        dX = np.stack(self.dX_buf, axis=0) # (N,6)

        for i in range(6):
            self.gps[i].fit(Z, dX[:, i])
            print(f"Full trained GP {i+1}/6 on {Z.shape[0]} points.")

        self._trained_once = True

    # 
    def train_online(self, steps: int | None = None):
        """
        Light online training using add_data() for each dimension.
        """
        if len(self.Z_buf) < self.min_points_to_train:
            return

        steps = int(self.online_steps if steps is None else steps)

        # Use only the latest chunk for online update (cheap)
        # You can change this window size if you want.
        window = min(5 * self.train_every, len(self.Z_buf))
        Z_new = np.stack(self.Z_buf[-window:], axis=0)    # (W,9)
        dX_new = np.stack(self.dX_buf[-window:], axis=0)  # (W,6)

        # If never trained -> do full first (stable normalization)
        if not self._trained_once:
            self.train_full()
            return

        for i in range(6):
            self.gps[i].add_data(Z_new, dX_new[:, i], online_steps=steps)
            print(f"Online trained GP {i+1}/6 on {Z_new.shape[0]} points.")

        self._trained_once = True

    def reset_episode(self, clear_buffers: bool = False):
        """
        Optionally reset model+buffers at episode start (starting from zero).
        """
        self._step_counter = 0
        if clear_buffers:
            self.Z_buf.clear()
            self.dX_buf.clear()

        if self.reset_each_episode:
            # wipe GP models
            self.gps = [
                OnlineSVGP1D(
                    input_dim=self.input_dim,
                    kernel=self.kernel,
                    lr=self.lr,
                    batch_size=self.batch_size,
                    num_inducing=self.num_inducing,
                    init_train_steps=self.init_train_steps,
                    device=self.device,
                    dtype=self.dtype,
                )
                for _ in range(self.out_dim)
            ]
            self._trained_once = False

    # ---------- persistence ----------
    def save_rollouts(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if len(self.Z_buf) == 0:
            np.savez(path, Z=np.zeros((0, 9), np.float32), dX=np.zeros((0, 6), np.float32))
            return
        Z = np.stack(self.Z_buf, axis=0)
        dX = np.stack(self.dX_buf, axis=0)
        np.savez(path, Z=Z, dX=dX)

    def load_rollouts(self, path: str, append: bool = True):
        data = np.load(path)
        Z = data["Z"].astype(np.float32)
        dX = data["dX"].astype(np.float32)
        if Z.shape[0] != dX.shape[0] or Z.shape[1] != 9 or dX.shape[1] != 6:
            raise ValueError(f"Bad rollout file shapes: Z{Z.shape}, dX{dX.shape}")

        if not append:
            self.Z_buf = []
            self.dX_buf = []

        for i in range(Z.shape[0]):
            self.Z_buf.append(Z[i])
            self.dX_buf.append(dX[i])

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        bundle = {
            "dt": self.dt,
            "mass": self.mass,
            "g": self.g,
            "device": self.device,
            "dtype": str(self.dtype),
            "kernel": self.kernel,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "num_inducing": self.num_inducing,
            "init_train_steps": self.init_train_steps,
            "train_every": self.train_every,
            "online_steps": self.online_steps,
            "min_points_to_train": self.min_points_to_train,
            "reset_each_episode": self.reset_each_episode,
            "_trained_once": self._trained_once,
            "gps": [gp.state_dict_bundle() for gp in self.gps],
        }
        torch.save(bundle, path)
    

    def load_model(self, path: str, map_location: str | None = None):
        bundle = torch.load(path, map_location=map_location or "cpu")

        self.dt = float(bundle["dt"])
        self.mass = float(bundle["mass"])
        self.g = float(bundle["g"])
        self.kernel = bundle["kernel"]
        self.lr = float(bundle["lr"])
        self.batch_size = int(bundle["batch_size"])
        self.num_inducing = int(bundle["num_inducing"])
        self.init_train_steps = int(bundle["init_train_steps"])
        self.train_every = int(bundle["train_every"])
        self.online_steps = int(bundle["online_steps"])
        self.min_points_to_train = int(bundle["min_points_to_train"])
        self.reset_each_episode = bool(bundle["reset_each_episode"])
        self._trained_once = bool(bundle.get("_trained_once", False))

        # rebuild gps
        self.gps = [
            OnlineSVGP1D(
                input_dim=self.input_dim,
                kernel=self.kernel,
                lr=self.lr,
                batch_size=self.batch_size,
                num_inducing=self.num_inducing,
                init_train_steps=self.init_train_steps,
                device=self.device,     # keep current runtime device
                dtype=self.dtype,
            )
            for _ in range(self.out_dim)
        ]
        for i in range(6):
            self.gps[i].load_from_bundle(bundle["gps"][i])
        self._trained_once = all(gp.trained for gp in self.gps)

    
    # ---------- torch step for MPPI ----------
    @torch.no_grad()
    def step_torch(self, pos: torch.Tensor, vel: torch.Tensor, action_force: torch.Tensor):
        """
        Torch/GPU step for MPPI rollouts.
        Inputs:
        pos, vel, action_force: (...,3) torch on self.device
        Returns:
        new_pos, new_vel: (...,3) torch on self.device
        """
        # fallback si no entrenado
        if (not self._trained_once) or (not all(gp.trained for gp in self.gps)):
            mg = torch.tensor([0.0, 0.0, -self.mass * self.g], device=self.device, dtype=self.dtype)
            a = (action_force + mg) / self.mass
            new_vel = vel + a * self.dt
            new_pos = pos + vel * self.dt + 0.5 * a * (self.dt ** 2)
            return new_pos, new_vel

        # asegurar device/dtype
        pos = pos.to(self.device, dtype=self.dtype)
        vel = vel.to(self.device, dtype=self.dtype)
        action_force = action_force.to(self.device, dtype=self.dtype)

        X = torch.cat([pos, vel], dim=-1)            # (...,6)
        Z = torch.cat([X, action_force], dim=-1)     # (...,9)
        Zflat = Z.reshape(-1, 9)                     # (B,9)

        dX_mean = []
        for i in range(6):
            mi, _vi = self.gps[i].predict_torch(Zflat)   # (B,)
            dX_mean.append(mi)
        dX_mean = torch.stack(dX_mean, dim=-1)       # (B,6)

        X2 = Zflat[:, :6] + dX_mean                  # (B,6)
        X2 = X2.reshape(*Z.shape[:-1], 6)            # (...,6)

        new_pos = X2[..., 0:3]
        new_vel = X2[..., 3:6]
        return new_pos, new_vel


    













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


