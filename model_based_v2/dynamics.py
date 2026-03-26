
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
    



class DoubleMassDynamicModel:
    def __init__(self, dt=0.1, mass1=1.0, mass2=1.0):
        self.dt = dt
        self.mass1 = mass1
        self.mass2 = mass2
        self.mg1 = np.array([0.0, 0.0, -9.81 * self.mass1])  # Gravedad
        self.mg2 = np.array([0.0, 0.0, -9.81 * self.mass2])  # Gravedad
    
    def step(self, pos, vel, action1, action2):
        # pos: [x, y, z]
        # vel: [vx, vy, vz]
        # action: [Fx, Fy, Fz]
        
        a1 = (action1 + self.mg1) / self.mass1
        a2 = (action2 + self.mg2) / self.mass2
        a = np.concatenate([a1, a2], axis=0)  # (6,)
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


class DoubleMassDynamicModelTorch:
    def __init__(self, dt=0.1, mass1=1.0, mass2=1.0, device="cuda", dtype=torch.float32):
        self.dt = float(dt)
        self.mass1 = float(mass1)
        self.mass2 = float(mass2)
        self.device = torch.device(device)
        self.dtype = dtype
        self.mg1 = torch.tensor([0.0, 0.0, -9.81 * self.mass1], device=self.device, dtype=self.dtype)
        self.mg2 = torch.tensor([0.0, 0.0, -9.81 * self.mass2], device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def step(self, pos, vel, action1, action2):
        # pos, vel, action pueden ser (3,) o (K,3)
        a1 = (action1 + self.mg1) / self.mass1
        a2 = (action2 + self.mg2) / self.mass2
        a = torch.cat([a1, a2], dim=-1)  # (6,)
        new_vel = vel + a * self.dt
        new_pos = pos + vel * self.dt + 0.5 * a * (self.dt ** 2)
        return new_pos, new_vel



##############################################################################
##############################################################################
########################### Modelo dinámico con SVGP #########################
##############################################################################
##############################################################################

from SVGP import OnlineSVGP1D, OnlineSVGPBatch, OnlineSVGPBatchAxiswise1D
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

        # If False, skip predictive variances (much faster). Variances returned are zeros.
        predict_variance: bool = False,

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
        self.predict_variance = bool(predict_variance)

        # One batched multi-output SVGP (independent outputs via batch dim).
        # This replaces 6 separate SVGPs called in a Python loop.
        self.gp = OnlineSVGPBatch(
            input_dim=self.input_dim,
            output_dim=self.out_dim,
            kernel=self.kernel,
            lr=self.lr,
            batch_size=self.batch_size,
            num_inducing=self.num_inducing,
            init_train_steps=self.init_train_steps,
            device=self.device,
            dtype=self.dtype,
        )

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
        # Prefer batched GP if available
        if getattr(self, "gp", None) is not None and getattr(self.gp, "trained", False):
            Z = self._make_input_Z(pos, vel, action_force)
            Zt = torch.as_tensor(Z[None, :], device=self.device, dtype=self.dtype)  # (1,9)

            m, v = self.gp.predict_torch(Zt, return_var=self.predict_variance)
            dX_mean = m[0].detach().cpu().numpy().astype(float)
            if self.predict_variance:
                dX_var = v[0].detach().cpu().numpy().astype(float)
            else:
                dX_var = np.zeros(6, dtype=float)

            X = np.concatenate([np.asarray(pos, float).reshape(3,), np.asarray(vel, float).reshape(3,)])
            X2 = X + dX_mean

            new_pos = X2[0:3]
            new_vel = X2[3:6]
            pos_var = dX_var[0:3]
            vel_var = dX_var[3:6]
            return new_pos, new_vel, pos_var, vel_var

        # Backward compat: per-dimension models
        if getattr(self, "gps", None) is not None and all(gp.trained for gp in self.gps):
            Z = self._make_input_Z(pos, vel, action_force)
            Zt = torch.as_tensor(Z[None, :], device=self.device, dtype=self.dtype)  # (1,9)
            dX_mean = np.zeros(6, dtype=float)
            dX_var = np.zeros(6, dtype=float)
            for i in range(6):
                m, v = self.gps[i].predict_torch(Zt, return_var=self.predict_variance)
                dX_mean[i] = float(m.item())
                dX_var[i] = float(v.item()) if self.predict_variance else 0.0

            X = np.concatenate([np.asarray(pos, float).reshape(3,), np.asarray(vel, float).reshape(3,)])
            X2 = X + dX_mean
            new_pos = X2[0:3]
            new_vel = X2[3:6]
            pos_var = dX_var[0:3]
            vel_var = dX_var[3:6]
            return new_pos, new_vel, pos_var, vel_var

        return self._fallback_step(pos, vel, action_force)

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

        if getattr(self, "gp", None) is not None:
            self.gp.fit(Z, dX)
            print(f"Full trained batched GP on {Z.shape[0]} points (out_dim={self.out_dim}).")
        else:
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

        if getattr(self, "gp", None) is not None:
            self.gp.add_data(Z_new, dX_new, online_steps=steps)
            print(f"Online trained batched GP on {Z_new.shape[0]} points (out_dim={self.out_dim}).")
        else:
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
            self.gp = OnlineSVGPBatch(
                input_dim=self.input_dim,
                output_dim=self.out_dim,
                kernel=self.kernel,
                lr=self.lr,
                batch_size=self.batch_size,
                num_inducing=self.num_inducing,
                init_train_steps=self.init_train_steps,
                device=self.device,
                dtype=self.dtype,
            )
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
            "predict_variance": self.predict_variance,
            "gp_type": "batch" if getattr(self, "gp", None) is not None else "list",
            "gp": self.gp.state_dict_bundle() if getattr(self, "gp", None) is not None else None,
            "gps": None if getattr(self, "gp", None) is not None else [gp.state_dict_bundle() for gp in self.gps],
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

        self.predict_variance = bool(bundle.get("predict_variance", False))

        # New format: batched GP
        if bundle.get("gp_type", None) == "batch" and ("gp" in bundle):
            self.gp = OnlineSVGPBatch(
                input_dim=self.input_dim,
                output_dim=self.out_dim,
                kernel=self.kernel,
                lr=self.lr,
                batch_size=self.batch_size,
                num_inducing=self.num_inducing,
                init_train_steps=self.init_train_steps,
                device=self.device,
                dtype=self.dtype,
            )
            self.gp.load_from_bundle(bundle["gp"])
            self._trained_once = bool(self.gp.trained)
            return

        # Backward-compat: old format with per-dimension SVGPs
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

        # Disable batched GP if we loaded a legacy checkpoint
        self.gp = None

        # Wrap old list into a new batched GP (one-time conversion for runtime speed)
        if self._trained_once:
            # NOTE: we cannot exactly fuse parameters from 6 independent models without
            # re-training, so we keep the old list for prediction in this legacy case.
            # If you want the speedup, re-train and save with gp_type='batch'.
            pass

    
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
        use_batch = getattr(self, "gp", None) is not None
        use_list = getattr(self, "gps", None) is not None
        trained_ok = (use_batch and getattr(self.gp, "trained", False)) or (use_list and all(gp.trained for gp in self.gps))

        if (not self._trained_once) or (not trained_ok):
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

        # Mean-only prediction for rollouts (fast)
        if use_batch:
            dX_mean, _ = self.gp.predict_torch(Zflat, return_var=False)  # (B,6)
        else:
            dX_list = []
            for i in range(6):
                mi, _ = self.gps[i].predict_torch(Zflat, return_var=False)
                dX_list.append(mi)
            dX_mean = torch.stack(dX_list, dim=-1)

        X2 = Zflat[:, :6] + dX_mean                  # (B,6)
        X2 = X2.reshape(*Z.shape[:-1], 6)            # (...,6)

        new_pos = X2[..., 0:3]
        new_vel = X2[..., 3:6]
        return new_pos, new_vel


    


# ============================================================
# ---- Drone Dynamics learning axis accelerations only --------
# ============================================================
class _BaseSVGPDroneAxisAccelDynamicModel:
    """Base class for translational dynamics driven by axis-wise acceleration GPs.

    State:
      X = [p(3), v(3)]

    Integration used here is the one requested by the user:
      p_{t+1} = p_t + dt * v_t
      v_{t+1} = v_t + dt * a_t

    The GP learns one scalar function per axis using only the corresponding
    force component as input. The three axis models are packed into a batched
    GPU-friendly SVGP manager to avoid Python loops during prediction/training.
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

        # If False, skip predictive variances (faster).
        predict_variance: bool = False,

        # If True, reset model+stats each episode.
        reset_each_episode: bool = False,
    ):
        self.dt = float(dt)
        self.mass = float(mass)
        self.g = float(gravity)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self.input_dim = 3
        self.out_dim = 3

        self.kernel = kernel
        self.lr = lr
        self.batch_size = batch_size
        self.num_inducing = num_inducing
        self.init_train_steps = init_train_steps

        self.train_every = int(train_every)
        self.online_steps = int(online_steps)
        self.min_points_to_train = int(min_points_to_train)
        self.reset_each_episode = bool(reset_each_episode)
        self.predict_variance = bool(predict_variance)

        self.gp = OnlineSVGPBatchAxiswise1D(
            output_dim=self.out_dim,
            kernel=self.kernel,
            lr=self.lr,
            batch_size=self.batch_size,
            num_inducing=self.num_inducing,
            init_train_steps=self.init_train_steps,
            device=self.device,
            dtype=self.dtype,
        )

        # rollout buffers (numpy)
        self.F_buf = []   # each: (3,) -> [Fx, Fy, Fz]
        self.A_buf = []   # each: (3,) -> learned target accelerations

        self._step_counter = 0
        self._trained_once = False

    # ---------- utils ----------
    def _gravity_vec_np(self) -> np.ndarray:
        return np.array([0.0, 0.0, -self.g], dtype=float)

    def _gravity_vec_torch(self, like: torch.Tensor | None = None) -> torch.Tensor:
        device = self.device if like is None else like.device
        dtype = self.dtype if like is None else like.dtype
        return torch.tensor([0.0, 0.0, -self.g], device=device, dtype=dtype)

    def _make_force(self, force) -> np.ndarray:
        return np.asarray(force, dtype=float).reshape(3,)

    def _make_state(self, pos, vel) -> np.ndarray:
        pos = np.asarray(pos, dtype=float).reshape(3,)
        vel = np.asarray(vel, dtype=float).reshape(3,)
        return np.concatenate([pos, vel], axis=0)

    def _nominal_acc_numpy(self, force: np.ndarray) -> np.ndarray:
        return np.zeros(3, dtype=float)

    def _nominal_acc_torch(self, force: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(force)

    def _fallback_step(self, pos: np.ndarray, vel: np.ndarray, force: np.ndarray):
        pos = np.asarray(pos, dtype=float).reshape(3,)
        vel = np.asarray(vel, dtype=float).reshape(3,)
        F = np.asarray(force, dtype=float).reshape(3,)

        # Physics fallback for stability before the GP is trained
        a = (F / self.mass) + self._gravity_vec_np()
        new_vel = vel + a * self.dt
        new_pos = pos + vel * self.dt
        pos_var = np.zeros(3, dtype=float)
        vel_var = np.ones(3, dtype=float) * 1e-3
        return new_pos, new_vel, pos_var, vel_var

    def _learned_acc_target_numpy(self, vel, new_vel, force) -> np.ndarray:
        vel = np.asarray(vel, dtype=float).reshape(3,)
        new_vel = np.asarray(new_vel, dtype=float).reshape(3,)
        force = np.asarray(force, dtype=float).reshape(3,)
        acc_obs = (new_vel - vel) / self.dt
        return acc_obs - self._nominal_acc_numpy(force)

    def _predict_total_acc_numpy(self, force: np.ndarray):
        F = np.asarray(force, dtype=np.float32).reshape(1, 3)
        Ft = torch.as_tensor(F, device=self.device, dtype=self.dtype)
        a_gp, a_var = self.gp.predict_torch(Ft, return_var=self.predict_variance)

        a_gp_np = a_gp[0].detach().cpu().numpy().astype(float)
        if self.predict_variance:
            a_var_np = a_var[0].detach().cpu().numpy().astype(float)
        else:
            a_var_np = np.zeros(3, dtype=float)

        a_total = self._nominal_acc_numpy(force) + a_gp_np
        return a_total, a_var_np

    def _predict_total_acc_torch(self, force: torch.Tensor):
        force = force.to(self.device, dtype=self.dtype)
        flat_force = force.reshape(-1, 3)
        a_gp, a_var = self.gp.predict_torch(flat_force, return_var=self.predict_variance)
        a_total = self._nominal_acc_torch(flat_force) + a_gp
        a_total = a_total.reshape(*force.shape[:-1], 3)
        a_var = a_var.reshape(*force.shape[:-1], 3)
        return a_total, a_var

    # ---------- public API ----------
    def step(self, pos, vel, action_force):
        if getattr(self, "gp", None) is not None and getattr(self.gp, "trained", False):
            pos_np = np.asarray(pos, dtype=float).reshape(3,)
            vel_np = np.asarray(vel, dtype=float).reshape(3,)
            force_np = np.asarray(action_force, dtype=float).reshape(3,)

            a_mean, a_var = self._predict_total_acc_numpy(force_np)
            new_pos = pos_np + self.dt * vel_np
            new_vel = vel_np + self.dt * a_mean

            pos_var = np.zeros(3, dtype=float)
            vel_var = (self.dt ** 2) * a_var
            return new_pos, new_vel, pos_var, vel_var

        return self._fallback_step(pos, vel, action_force)

    def add_transition(self, pos, vel, force, new_pos, new_vel, train_online: bool = True):
        F = self._make_force(force).astype(np.float32)
        A = self._learned_acc_target_numpy(vel, new_vel, force).astype(np.float32)

        self.F_buf.append(F)
        self.A_buf.append(A)
        self._step_counter += 1

        if not train_online:
            return

        if (self._step_counter % self.train_every) == 0 and len(self.F_buf) >= self.min_points_to_train:
            self.train_online(steps=self.online_steps)

    def train_full(self):
        if len(self.F_buf) < self.min_points_to_train:
            return

        F = np.stack(self.F_buf, axis=0)
        A = np.stack(self.A_buf, axis=0)
        self.gp.fit(F, A)
        self._trained_once = True

    def train_online(self, steps: int | None = None):
        if len(self.F_buf) < self.min_points_to_train:
            return

        steps = int(self.online_steps if steps is None else steps)
        window = min(5 * self.train_every, len(self.F_buf))
        F_new = np.stack(self.F_buf[-window:], axis=0)
        A_new = np.stack(self.A_buf[-window:], axis=0)

        if not self._trained_once:
            self.train_full()
            return

        self.gp.add_data(F_new, A_new, online_steps=steps)
        self._trained_once = True

    def reset_episode(self, clear_buffers: bool = False):
        self._step_counter = 0
        if clear_buffers:
            self.F_buf.clear()
            self.A_buf.clear()

        if self.reset_each_episode:
            self.gp = OnlineSVGPBatchAxiswise1D(
                output_dim=self.out_dim,
                kernel=self.kernel,
                lr=self.lr,
                batch_size=self.batch_size,
                num_inducing=self.num_inducing,
                init_train_steps=self.init_train_steps,
                device=self.device,
                dtype=self.dtype,
            )
            self._trained_once = False

    # ---------- persistence ----------
    def save_rollouts(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if len(self.F_buf) == 0:
            np.savez(path, F=np.zeros((0, 3), np.float32), A=np.zeros((0, 3), np.float32))
            return
        F = np.stack(self.F_buf, axis=0)
        A = np.stack(self.A_buf, axis=0)
        np.savez(path, F=F, A=A)

    def load_rollouts(self, path: str, append: bool = True):
        data = np.load(path)
        F = data["F"].astype(np.float32)
        A = data["A"].astype(np.float32)
        if F.shape[0] != A.shape[0] or F.shape[1] != 3 or A.shape[1] != 3:
            raise ValueError(f"Bad rollout file shapes: F{F.shape}, A{A.shape}")

        if not append:
            self.F_buf = []
            self.A_buf = []

        for i in range(F.shape[0]):
            self.F_buf.append(F[i])
            self.A_buf.append(A[i])

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
            "predict_variance": self.predict_variance,
            "model_form": self.__class__.__name__,
            "gp_type": "axiswise_batch_1d",
            "gp": self.gp.state_dict_bundle() if getattr(self, "gp", None) is not None else None,
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
        self.predict_variance = bool(bundle.get("predict_variance", False))

        if bundle.get("model_form", self.__class__.__name__) != self.__class__.__name__:
            raise RuntimeError(
                f"Checkpoint was saved for {bundle.get('model_form')} but you are loading it into {self.__class__.__name__}."
            )

        if bundle.get("gp_type", None) != "axiswise_batch_1d" or bundle.get("gp", None) is None:
            raise RuntimeError(
                f"Unsupported checkpoint type for {self.__class__.__name__}: {bundle.get('gp_type', None)}"
            )

        self.gp = OnlineSVGPBatchAxiswise1D(
            output_dim=self.out_dim,
            kernel=self.kernel,
            lr=self.lr,
            batch_size=self.batch_size,
            num_inducing=self.num_inducing,
            init_train_steps=self.init_train_steps,
            device=self.device,
            dtype=self.dtype,
        )
        self.gp.load_from_bundle(bundle["gp"])
        self._trained_once = bool(self.gp.trained)

    # ---------- torch step for MPPI ----------
    @torch.no_grad()
    def step_torch(self, pos: torch.Tensor, vel: torch.Tensor, action_force: torch.Tensor):
        pos = pos.to(self.device, dtype=self.dtype)
        vel = vel.to(self.device, dtype=self.dtype)
        action_force = action_force.to(self.device, dtype=self.dtype)

        use_gp = getattr(self, "gp", None) is not None and getattr(self.gp, "trained", False)
        if (not self._trained_once) or (not use_gp):
            a = (action_force / self.mass) + self._gravity_vec_torch(action_force)
            new_vel = vel + a * self.dt
            new_pos = pos + vel * self.dt
            return new_pos, new_vel

        a_total, _ = self._predict_total_acc_torch(action_force)
        new_vel = vel + a_total * self.dt
        new_pos = pos + vel * self.dt
        return new_pos, new_vel

    @torch.no_grad()
    def predict_acceleration_torch(self, action_force: torch.Tensor, return_var: bool = True):
        action_force = action_force.to(self.device, dtype=self.dtype)
        flat_force = action_force.reshape(-1, 3)

        if getattr(self, "gp", None) is None or (not getattr(self.gp, "trained", False)):
            a = (flat_force / self.mass) + self._gravity_vec_torch(flat_force)
            a = a.reshape(*action_force.shape[:-1], 3)
            v = torch.zeros_like(a)
            return a, v

        a_gp, a_var = self.gp.predict_torch(flat_force, return_var=return_var)
        a_total = self._nominal_acc_torch(flat_force) + a_gp
        a_total = a_total.reshape(*action_force.shape[:-1], 3)
        a_var = a_var.reshape(*action_force.shape[:-1], 3)
        return a_total, a_var


class SVGPDroneAccelerationDynamicModel(_BaseSVGPDroneAxisAccelDynamicModel):
    """Learns only translational accelerations as axis-wise functions of force.

    Learned model:
      ax = g_x(Fx), ay = g_y(Fy), az = g_z(Fz)

    State integration:
      p_{t+1} = p_t + dt * v_t
      v_{t+1} = v_t + dt * a_t
    """

    def _nominal_acc_numpy(self, force: np.ndarray) -> np.ndarray:
        return np.zeros(3, dtype=float)

    def _nominal_acc_torch(self, force: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(force)


class SVGPDroneResidualAccelerationDynamicModel(_BaseSVGPDroneAxisAccelDynamicModel):
    """Learns a residual on top of a point-mass translational prior.

    Residual model:
      a = (F / m) + [0, 0, -g] + g_res(F)

    In x/y this is exactly the requested structure F_axis/m + g(F_axis).
    In z we keep gravity explicit so the residual GP does not waste capacity
    learning the constant -g term.
    """

    def _nominal_acc_numpy(self, force: np.ndarray) -> np.ndarray:
        force = np.asarray(force, dtype=float).reshape(3,)
        return (force / self.mass) + self._gravity_vec_np()

    def _nominal_acc_torch(self, force: torch.Tensor) -> torch.Tensor:
        return (force / self.mass) + self._gravity_vec_torch(force)








###################################################################################################
###################################################################################################
########################### 2 Drones - Modelo dinámico con SVGP ###################################
###################################################################################################
###################################################################################################



import os
import numpy as np
import torch

# Reusa tu OnlineSVGP1D (y por debajo SparseGPModel, gpytorch, etc.)
# from your_svgp_module import OnlineSVGP1D


class SVGPDoubleDroneDynamicModel:
    """
    Two-drone dynamics model (translational only):

    State:
      X = [p1(3), v1(3), p2(3), v2(3)]  (12)
    Action:
      U = [F1(3), F2(3)]  (6)

    GP input:
      Z = [Xprev(12), U(6)]  (18)
    GP target:
      dX = Xnext - Xprev  (12)

    Uses 12 independent 1D SVGPs.
    """

    def __init__(
        self,
        dt: float = 0.1,
        mass1: float = 1.0,
        mass2: float = 1.0,
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

        # If False, skip predictive variances (much faster). Variances returned are zeros.
        predict_variance: bool = False,

        reset_each_episode: bool = False,
    ):
        self.dt = float(dt)
        self.m1 = float(mass1)
        self.m2 = float(mass2)
        self.g = float(gravity)

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype

        self.input_dim = 18
        self.out_dim = 12

        self.kernel = kernel
        self.lr = lr
        self.batch_size = batch_size
        self.num_inducing = num_inducing
        self.init_train_steps = init_train_steps

        self.train_every = int(train_every)
        self.online_steps = int(online_steps)
        self.min_points_to_train = int(min_points_to_train)
        self.reset_each_episode = bool(reset_each_episode)
        self.predict_variance = bool(predict_variance)

        self.gp = OnlineSVGPBatch(
            input_dim=self.input_dim,
            output_dim=self.out_dim,
            kernel=self.kernel,
            lr=self.lr,
            batch_size=self.batch_size,
            num_inducing=self.num_inducing,
            init_train_steps=self.init_train_steps,
            device=self.device,
            dtype=self.dtype,
        )

        # buffers (numpy)
        self.Z_buf = []   # (18,)
        self.dX_buf = []  # (12,)

        self._step_counter = 0
        self._trained_once = False

    # -------------------- utils --------------------
    def _make_X(self, p1, v1, p2, v2) -> np.ndarray:
        p1 = np.asarray(p1, float).reshape(3,)
        v1 = np.asarray(v1, float).reshape(3,)
        p2 = np.asarray(p2, float).reshape(3,)
        v2 = np.asarray(v2, float).reshape(3,)
        return np.concatenate([p1, v1, p2, v2], axis=0)  # (12,)

    def _make_U(self, F1, F2) -> np.ndarray:
        F1 = np.asarray(F1, float).reshape(3,)
        F2 = np.asarray(F2, float).reshape(3,)
        return np.concatenate([F1, F2], axis=0)  # (6,)

    def _make_Z(self, p1, v1, p2, v2, F1, F2) -> np.ndarray:
        X = self._make_X(p1, v1, p2, v2)
        U = self._make_U(F1, F2)
        return np.concatenate([X, U], axis=0)  # (18,)

    def _fallback_step_numpy(self, p1, v1, p2, v2, F1, F2):
        """
        Two independent point-masses with gravity.
        """
        p1 = np.asarray(p1, float).reshape(3,)
        v1 = np.asarray(v1, float).reshape(3,)
        p2 = np.asarray(p2, float).reshape(3,)
        v2 = np.asarray(v2, float).reshape(3,)
        F1 = np.asarray(F1, float).reshape(3,)
        F2 = np.asarray(F2, float).reshape(3,)

        a1 = (F1 + np.array([0.0, 0.0, -self.m1 * self.g])) / self.m1
        a2 = (F2 + np.array([0.0, 0.0, -self.m2 * self.g])) / self.m2

        v1n = v1 + a1 * self.dt
        p1n = p1 + v1 * self.dt + 0.5 * a1 * (self.dt ** 2)

        v2n = v2 + a2 * self.dt
        p2n = p2 + v2 * self.dt + 0.5 * a2 * (self.dt ** 2)

        # simple small variance fallback
        pvar1 = np.ones(3) * 1e-3
        vvar1 = np.ones(3) * 1e-3
        pvar2 = np.ones(3) * 1e-3
        vvar2 = np.ones(3) * 1e-3
        return p1n, v1n, p2n, v2n, pvar1, vvar1, pvar2, vvar2

    # -------------------- public API (numpy) --------------------
    def step(self, p1, v1, p2, v2, F1, F2):
        """
        Returns:
          p1n, v1n, p2n, v2n, pvar1(3), vvar1(3), pvar2(3), vvar2(3)
        """
        if getattr(self, "gp", None) is not None and getattr(self.gp, "trained", False):
            Z = self._make_Z(p1, v1, p2, v2, F1, F2).astype(np.float32)
            Zt = torch.as_tensor(Z[None, :], device=self.device, dtype=self.dtype)  # (1,18)

            m, v = self.gp.predict_torch(Zt, return_var=self.predict_variance)
            dX_mean = m[0].detach().cpu().numpy().astype(float)
            if self.predict_variance:
                dX_var = v[0].detach().cpu().numpy().astype(float)
            else:
                dX_var = np.zeros(12, dtype=float)

            X = self._make_X(p1, v1, p2, v2)
            Xn = X + dX_mean

            p1n, v1n = Xn[0:3],  Xn[3:6]
            p2n, v2n = Xn[6:9],  Xn[9:12]

            pvar1, vvar1 = dX_var[0:3], dX_var[3:6]
            pvar2, vvar2 = dX_var[6:9], dX_var[9:12]
            return p1n, v1n, p2n, v2n, pvar1, vvar1, pvar2, vvar2

        if getattr(self, "gps", None) is not None and all(gp.trained for gp in self.gps):
            Z = self._make_Z(p1, v1, p2, v2, F1, F2).astype(np.float32)
            Zt = torch.as_tensor(Z[None, :], device=self.device, dtype=self.dtype)  # (1,18)
            dX_mean = np.zeros(12, dtype=float)
            dX_var = np.zeros(12, dtype=float)
            for i in range(12):
                m, v = self.gps[i].predict_torch(Zt, return_var=self.predict_variance)
                dX_mean[i] = float(m.item())
                dX_var[i] = float(v.item()) if self.predict_variance else 0.0

            X = self._make_X(p1, v1, p2, v2)
            Xn = X + dX_mean
            p1n, v1n = Xn[0:3],  Xn[3:6]
            p2n, v2n = Xn[6:9],  Xn[9:12]

            pvar1, vvar1 = dX_var[0:3], dX_var[3:6]
            pvar2, vvar2 = dX_var[6:9], dX_var[9:12]
            return p1n, v1n, p2n, v2n, pvar1, vvar1, pvar2, vvar2

        return self._fallback_step_numpy(p1, v1, p2, v2, F1, F2)

    def add_transition(self, p1, v1, p2, v2, F1, F2, p1n, v1n, p2n, v2n, train_online: bool = True):
        """
        Store transition (Z -> dX).
        """
        Z = self._make_Z(p1, v1, p2, v2, F1, F2).astype(np.float32)
        X0 = self._make_X(p1, v1, p2, v2).astype(np.float32)
        X1 = self._make_X(p1n, v1n, p2n, v2n).astype(np.float32)
        dX = (X1 - X0).astype(np.float32)

        self.Z_buf.append(Z)
        self.dX_buf.append(dX)
        self._step_counter += 1

        if not train_online:
            return

        if (self._step_counter % self.train_every) == 0 and len(self.Z_buf) >= self.min_points_to_train:
            self.train_online(steps=self.online_steps)

    def train_full(self):
        if len(self.Z_buf) < self.min_points_to_train:
            return
        Z = np.stack(self.Z_buf, axis=0)   # (N,18)
        dX = np.stack(self.dX_buf, axis=0) # (N,12)
        if getattr(self, "gp", None) is not None:
            self.gp.fit(Z, dX)
        else:
            for i in range(12):
                self.gps[i].fit(Z, dX[:, i])
        self._trained_once = True

    def train_online(self, steps: int | None = None):
        if len(self.Z_buf) < self.min_points_to_train:
            return
        steps = int(self.online_steps if steps is None else steps)

        # update with recent window (cheap)
        window = min(5 * self.train_every, len(self.Z_buf))
        Z_new = np.stack(self.Z_buf[-window:], axis=0)    # (W,18)
        dX_new = np.stack(self.dX_buf[-window:], axis=0)  # (W,12)

        if not self._trained_once:
            self.train_full()
            return

        if getattr(self, "gp", None) is not None:
            self.gp.add_data(Z_new, dX_new, online_steps=steps)
        else:
            for i in range(12):
                self.gps[i].add_data(Z_new, dX_new[:, i], online_steps=steps)
        self._trained_once = True

    def reset_episode(self, clear_buffers: bool = False):
        self._step_counter = 0
        if clear_buffers:
            self.Z_buf.clear()
            self.dX_buf.clear()

        if self.reset_each_episode:
            self.gp = OnlineSVGPBatch(
                input_dim=self.input_dim,
                output_dim=self.out_dim,
                kernel=self.kernel,
                lr=self.lr,
                batch_size=self.batch_size,
                num_inducing=self.num_inducing,
                init_train_steps=self.init_train_steps,
                device=self.device,
                dtype=self.dtype,
            )
            self._trained_once = False

    # -------------------- persistence --------------------
    def save_rollouts(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        if len(self.Z_buf) == 0:
            np.savez(path, Z=np.zeros((0, 18), np.float32), dX=np.zeros((0, 12), np.float32))
            return
        np.savez(path, Z=np.stack(self.Z_buf, 0), dX=np.stack(self.dX_buf, 0))

    def load_rollouts(self, path: str, append: bool = True):
        data = np.load(path)
        Z = data["Z"].astype(np.float32)
        dX = data["dX"].astype(np.float32)
        if Z.shape[1] != 18 or dX.shape[1] != 12 or Z.shape[0] != dX.shape[0]:
            raise ValueError(f"Bad rollout shapes: Z{Z.shape}, dX{dX.shape}")

        if not append:
            self.Z_buf, self.dX_buf = [], []

        for i in range(Z.shape[0]):
            self.Z_buf.append(Z[i])
            self.dX_buf.append(dX[i])

    def save_model(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        bundle = {
            "dt": self.dt,
            "m1": self.m1,
            "m2": self.m2,
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
            "predict_variance": self.predict_variance,
            "gp_type": "batch",
            "gp": self.gp.state_dict_bundle() if getattr(self, "gp", None) is not None else None,
            "gps": None if getattr(self, "gp", None) is not None else [gp.state_dict_bundle() for gp in self.gps],
        }
        torch.save(bundle, path)

    def load_model(self, path: str, map_location: str | None = None):
        bundle = torch.load(path, map_location=map_location or "cpu")
        self.dt = float(bundle["dt"])
        self.m1 = float(bundle["m1"])
        self.m2 = float(bundle["m2"])
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

        self.predict_variance = bool(bundle.get("predict_variance", False))

        if bundle.get("gp_type", None) == "batch" and bundle.get("gp", None) is not None:
            self.gp = OnlineSVGPBatch(
                input_dim=self.input_dim,
                output_dim=self.out_dim,
                kernel=self.kernel,
                lr=self.lr,
                batch_size=self.batch_size,
                num_inducing=self.num_inducing,
                init_train_steps=self.init_train_steps,
                device=self.device,
                dtype=self.dtype,
            )
            self.gp.load_from_bundle(bundle["gp"])
            self._trained_once = bool(self.gp.trained)
            return

        # backward compat
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
        for i in range(12):
            self.gps[i].load_from_bundle(bundle["gps"][i])
        self._trained_once = all(gp.trained for gp in self.gps)

        self.gp = None

    # -------------------- torch step for MPPI --------------------
    @torch.no_grad()
    def step_torch(self, X: torch.Tensor, U: torch.Tensor):
        """
        Torch/GPU step for MPPI rollouts.

        Inputs:
          X: (...,12) [p1,v1,p2,v2]
          U: (..., 6) [F1,F2]
        Returns:
          Xnext: (...,12)
        """
        X = X.to(self.device, dtype=self.dtype)
        U = U.to(self.device, dtype=self.dtype)

        use_batch = getattr(self, "gp", None) is not None
        use_list = getattr(self, "gps", None) is not None
        trained_ok = (use_batch and getattr(self.gp, "trained", False)) or (use_list and all(gp.trained for gp in self.gps))

        # fallback
        if (not self._trained_once) or (not trained_ok):
            mg1 = torch.tensor([0.0, 0.0, -self.m1 * self.g], device=self.device, dtype=self.dtype)
            mg2 = torch.tensor([0.0, 0.0, -self.m2 * self.g], device=self.device, dtype=self.dtype)

            p1, v1 = X[..., 0:3], X[..., 3:6]
            p2, v2 = X[..., 6:9], X[..., 9:12]
            F1, F2 = U[..., 0:3], U[..., 3:6]

            a1 = (F1 + mg1) / self.m1
            a2 = (F2 + mg2) / self.m2

            v1n = v1 + a1 * self.dt
            p1n = p1 + v1 * self.dt + 0.5 * a1 * (self.dt ** 2)

            v2n = v2 + a2 * self.dt
            p2n = p2 + v2 * self.dt + 0.5 * a2 * (self.dt ** 2)

            return torch.cat([p1n, v1n, p2n, v2n], dim=-1)

        Z = torch.cat([X, U], dim=-1)          # (...,18)
        Zflat = Z.reshape(-1, 18)              # (B,18)

        if use_batch:
            dX, _ = self.gp.predict_torch(Zflat, return_var=False)  # (B,12)
        else:
            dX_list = []
            for i in range(12):
                mi, _ = self.gps[i].predict_torch(Zflat, return_var=False)  # (B,)
                dX_list.append(mi)
            dX = torch.stack(dX_list, dim=-1)      # (B,12)

        Xnext = Zflat[:, :12] + dX             # (B,12)
        Xnext = Xnext.reshape(*Z.shape[:-1], 12)
        return Xnext












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

