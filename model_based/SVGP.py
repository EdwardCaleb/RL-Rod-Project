
import torch
import gpytorch
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


# ============================================================
# ---- Sparse GP model (SVGP con inducing points) ------------
# ============================================================
class SparseGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, kernel='RBF', ard_dims=None):
        """
        inducing_points: tensor [M, D] ya en el device correcto
        """
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.size(0)
        )

        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )

        super().__init__(variational_strategy)

        # Mean
        self.mean_module = gpytorch.means.ConstantMean()

        # Kernel
        if kernel == 'RBF':
            base_kernel = gpytorch.kernels.RBFKernel(ard_num_dims=ard_dims)
        elif kernel == 'Matern':
            base_kernel = gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=ard_dims)
        elif kernel == 'RQ':
            base_kernel = gpytorch.kernels.RationalQuadraticKernel(ard_num_dims=ard_dims)
        else:
            raise ValueError(f"Unsupported kernel type: {kernel}")

        self.covar_module = gpytorch.kernels.ScaleKernel(base_kernel)

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



# ============================================================
# ---- Online Sparse GP Manager ------------------------------
# ============================================================

class OnlineSparseGPManager:
    def __init__(
        self,
        kernel='RBF',
        lr=0.01,
        iters=1000,
        num_inducing=1000,
        batch_size=128,
        device=None,
    ):
        """
        Online GP basado en SVGP (Sparse Variational GP).
        - kernel: 'RBF', 'Matern', 'RQ'
        - lr: learning rate Adam
        - iters: nº máximo de steps de entrenamiento "full" en fit()
        - num_inducing: nº de inducing points M
        - batch_size: tamaño de minibatch para VI
        - device: 'cuda' o 'cpu'
        """
        self.kernel = kernel
        self.lr = lr
        self.iters = iters
        self.num_inducing = num_inducing
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Datos en escala original
        self.X = None      # [N, D]
        self.Y = None      # [N]

        # Datos normalizados
        self.Xn = None     # [N, D]
        self.Yn = None     # [N]

        # Normalización (fijamos tras el primer fit)
        self.X_mean = None
        self.X_std = None
        self.Y_mean = None
        self.Y_std = None

        # Modelo SVGP + likelihood
        self.likelihood = None
        self.model = None
        self.mll = None

        self.trained = False


    # ----------------------------- #
    #        FIT / INITIAL TRAIN    #
    # ----------------------------- #
    def fit(self, X, Y):
        """
        Entrenamiento inicial con todos los datos pasados.
        X: array/tensor [N, D]
        Y: array/tensor [N] o [N, 1]
        """
        # Convertir a tensores en device
        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        Y = torch.as_tensor(Y, dtype=torch.float32, device=self.device).flatten()

        self.X = X.clone()
        self.Y = Y.clone()

        # Normalización (la fijamos aquí para todo el online)
        self._compute_normalization(initial=True)

        # Inicializar modelo SVGP + likelihood
        self._init_model()

        # Entrenar varias iteraciones (full training)
        self._train(max_steps=self.iters)

        self.trained = True


    # ----------------------------- #
    #       ADD NEW DATA POINTS     #
    # ----------------------------- #
    def add_data(self, X_new, Y_new, retrain=True, online_steps=50):
        """
        Añadir nuevos datos (online) y entrenar unas pocas steps.
        X_new: [N_new, D]
        Y_new: [N_new]
        online_steps: nº de steps de optimización tras añadir datos
                      (típicamente << self.iters)
        """
        # Convertir a tensores
        X_new = torch.as_tensor(X_new, dtype=torch.float32, device=self.device)
        Y_new = torch.as_tensor(Y_new, dtype=torch.float32, device=self.device).flatten()

        if self.X is None or self.Y is None:
            # Si no había datos, esto equivale a fit()
            self.fit(X_new, Y_new)
            return

        # Apilar a dataset existente (escala original)
        self.X = torch.cat([self.X, X_new], dim=0)
        self.Y = torch.cat([self.Y, Y_new], dim=0)

        # ✅ Importante: mantenemos la normalización ORIGINAL
        # para no romper el modelo. Solo normalizamos los nuevos
        # datos con las mismas X_mean, X_std, etc.
        self._update_normalized_data()

        # Actualizar num_data del ELBO
        self.mll.num_data = self.Yn.size(0)

        if retrain:
            # Entrenamiento "ligero" online (unas pocas steps)
            self._train(max_steps=online_steps)


    # ----------------------------- #
    #         INTERNAL UTILS        #
    # ----------------------------- #
    def _compute_normalization(self, initial=False):
        """
        Calcula X_mean, X_std, Y_mean, Y_std a partir de self.X, self.Y.
        Si initial=True se fijan para todo el proceso online.
        """
        if initial or self.X_mean is None:
            self.X_mean = self.X.mean(dim=0)
            self.X_std = self.X.std(dim=0) + 1e-8
            self.Y_mean = self.Y.mean(dim=0)
            self.Y_std = self.Y.std(dim=0) + 1e-8

        # Normalización usando SIEMPRE las mismas medias/varianzas
        self.Xn = (self.X - self.X_mean) / self.X_std
        self.Yn = (self.Y - self.Y_mean) / self.Y_std

    def _update_normalized_data(self):
        """
        Recalcula Xn, Yn con los mismos X_mean, X_std, Y_mean, Y_std
        (no se vuelven a estimar, solo se aplican a los nuevos datos).
        """
        self.Xn = (self.X - self.X_mean) / self.X_std
        self.Yn = (self.Y - self.Y_mean) / self.Y_std

    def dataset(self):
        """
        Devuelve los datos en escala original como numpy (para debug).
        """
        X_np = self.X.detach().cpu().numpy()
        Y_np = self.Y.detach().cpu().numpy()
        return X_np, Y_np

    def _init_model(self):
        """
        Crea el modelo SVGP + likelihood a partir de self.Xn, self.Yn.
        """
        # Elegir inducing points: subset aleatorio de Xn
        N = self.Xn.size(0)
        M = min(self.num_inducing, N)
        perm = torch.randperm(N, device=self.device)
        inducing_points = self.Xn[perm[:M]].contiguous()

        # Likelihood
        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.likelihood.noise_covar.initialize(noise=1e-3)

        # Modelo SVGP
        self.model = SparseGPModel(
            inducing_points=inducing_points,
            kernel=self.kernel,
            ard_dims=self.Xn.shape[-1],
        ).to(self.device)

        # ELBO variacional
        self.mll = gpytorch.mlls.VariationalELBO(
            likelihood=self.likelihood,
            model=self.model,
            num_data=self.Yn.size(0),
        )

    def _train(self, max_steps=None):
        """
        Entrenamiento variacional (SVGP) con minibatches.
        max_steps: nº máximo de pasos de optimización
        """
        if max_steps is None:
            max_steps = self.iters

        self.model.train()
        self.likelihood.train()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

        dataset = TensorDataset(self.Xn, self.Yn)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        step = 0
        while step < max_steps:
            for Xb, Yb in loader:
                optimizer.zero_grad()
                output = self.model(Xb)
                loss = -self.mll(output, Yb)
                loss.backward()
                optimizer.step()

                step += 1
                if step >= max_steps:
                    break

        self.model.eval()
        self.likelihood.eval()
        self.trained = True


    # ----------------------------- #
    #            PREDICT            #
    # ----------------------------- #
    @torch.no_grad()
    def predict(self, X):
        """
        Predicción de media y varianza en escala ORIGINAL.
        X: [N_test, D]
        Devuelve:
          mean_np: [N_test]
          var_np:  [N_test]
        """
        if not self.trained:
            raise RuntimeError("GP has not been trained yet.")

        X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        Xn = (X - self.X_mean) / self.X_std

        with gpytorch.settings.fast_pred_var():
            pred_dist = self.likelihood(self.model(Xn))
            mean = pred_dist.mean * self.Y_std + self.Y_mean
            var = pred_dist.variance * (self.Y_std ** 2)

        return mean.detach().cpu().numpy(), var.detach().cpu().numpy()


    # ----------------------------- #
    #        PREDICT TORCH          #
    # ----------------------------- #
    @torch.no_grad()
    def predict_torch(self, X: torch.Tensor):
        """
        Predicción en torch y en el mismo device, en escala ORIGINAL.
        X: (B, D) torch
        Devuelve:
        mean: (B,)
        var:  (B,)
        """
        if not self.trained:
            raise RuntimeError("GP has not been trained yet.")

        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        else:
            X = X.to(device=self.device, dtype=torch.float32)

        Xn = (X - self.X_mean) / self.X_std

        with gpytorch.settings.fast_pred_var():
            pred_dist = self.likelihood(self.model(Xn))
            mean = pred_dist.mean * self.Y_std + self.Y_mean
            var = pred_dist.variance * (self.Y_std ** 2)

        return mean, var
    


    # ----------------------------- #
    #          LOAD - SAVE          #
    # ----------------------------- #
    def get_config(self) -> dict:
        """Config mínima para reconstruir el manager al cargar."""
        return dict(
            kernel=self.kernel,
            lr=self.lr,
            iters=self.iters,
            num_inducing=self.num_inducing,
            batch_size=self.batch_size,
            device=str(self.device),
        )

    def save(self, path: str, save_data: bool = True):
        """
        Guarda un checkpoint robusto.
        save_data=True guarda X,Y (útil si quieres seguir entrenando online tras cargar).
        """
        if not self.trained or self.model is None or self.likelihood is None:
            raise RuntimeError("No puedes guardar: GP no entrenado o modelo no inicializado.")

        ckpt = {
            "config": self.get_config(),
            "trained": bool(self.trained),

            # normalización
            "X_mean": self.X_mean.detach().cpu(),
            "X_std":  self.X_std.detach().cpu(),
            "Y_mean": self.Y_mean.detach().cpu(),
            "Y_std":  self.Y_std.detach().cpu(),

            # inducing points (MUY importante en SVGP)
            "inducing_points": self.model.variational_strategy.inducing_points.detach().cpu(),

            # pesos del modelo/likelihood
            "model_state": self.model.state_dict(),
            "likelihood_state": self.likelihood.state_dict(),

            # dataset acumulado (opcional)
            "X": self.X.detach().cpu() if (save_data and self.X is not None) else None,
            "Y": self.Y.detach().cpu() if (save_data and self.Y is not None) else None,
        }
        torch.save(ckpt, path)

    @classmethod
    def load(cls, path: str, device: str | None = None, map_location: str | None = None):
        """
        Carga el checkpoint y reconstruye el SVGP en el device deseado.
        - device: 'cuda' o 'cpu' (si None usa lo guardado en config)
        - map_location: lo que acepta torch.load (ej: 'cpu')
        """
        ckpt = torch.load(path, map_location=map_location or "cpu")

        cfg = ckpt["config"]
        if device is not None:
            cfg["device"] = device

        mgr = cls(**cfg)

        # restaura dataset (opcional)
        if ckpt.get("X", None) is not None:
            mgr.X = ckpt["X"].to(mgr.device)
        if ckpt.get("Y", None) is not None:
            mgr.Y = ckpt["Y"].to(mgr.device)

        # restaura normalización
        mgr.X_mean = ckpt["X_mean"].to(mgr.device)
        mgr.X_std  = ckpt["X_std"].to(mgr.device)
        mgr.Y_mean = ckpt["Y_mean"].to(mgr.device)
        mgr.Y_std  = ckpt["Y_std"].to(mgr.device)

        # re-crea Xn/Yn si hay data
        if mgr.X is not None and mgr.Y is not None:
            mgr.Xn = (mgr.X - mgr.X_mean) / mgr.X_std
            mgr.Yn = (mgr.Y - mgr.Y_mean) / mgr.Y_std

        # reconstruye el modelo con inducing points EXACTOS
        inducing = ckpt["inducing_points"].to(mgr.device).contiguous()

        # (re)crear likelihood + model
        mgr.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(mgr.device)
        mgr.model = SparseGPModel(
            inducing_points=inducing,
            kernel=mgr.kernel,
            ard_dims=inducing.shape[-1],
        ).to(mgr.device)

        # recrear ELBO (num_data si tenemos data)
        num_data = int(mgr.Yn.size(0)) if (mgr.Yn is not None) else 1
        mgr.mll = gpytorch.mlls.VariationalELBO(
            likelihood=mgr.likelihood,
            model=mgr.model,
            num_data=num_data,
        )

        # cargar pesos
        mgr.model.load_state_dict(ckpt["model_state"])
        mgr.likelihood.load_state_dict(ckpt["likelihood_state"])

        mgr.model.eval()
        mgr.likelihood.eval()
        mgr.trained = bool(ckpt.get("trained", True))
        return mgr




# ============================================================
# ---- Online SVGP for 1D target: y = f(x) -------------------
# ============================================================
class OnlineSVGP1D:
    """
    SVGP 1D output manager:
      X: (N, D)
      Y: (N,)
    """

    def __init__(
        self,
        input_dim: int,
        kernel: str = "RBF",
        lr: float = 0.01,
        batch_size: int = 256,
        num_inducing: int = 256,
        init_train_steps: int = 800,
        device: str | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.input_dim = int(input_dim)
        self.kernel = kernel
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.num_inducing = int(num_inducing)
        self.init_train_steps = int(init_train_steps)
        self.dtype = dtype
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        # raw data
        self.X: torch.Tensor | None = None   # (N,D)
        self.Y: torch.Tensor | None = None   # (N,)

        # normalization stats (fixed after first fit)
        self.X_mean: torch.Tensor | None = None
        self.X_std: torch.Tensor | None = None
        self.Y_mean: torch.Tensor | None = None
        self.Y_std: torch.Tensor | None = None

        # normalized data
        self.Xn: torch.Tensor | None = None
        self.Yn: torch.Tensor | None = None

        self.model: SparseGPModel | None = None
        self.likelihood: gpytorch.likelihoods.GaussianLikelihood | None = None
        self.mll: gpytorch.mlls.VariationalELBO | None = None

        self.trained = False

    def _ensure_stats(self):
        if self.X_mean is None:
            raise RuntimeError("Normalization stats not initialized. Call fit() first.")

    def _compute_normalization_initial(self):
        self.X_mean = self.X.mean(dim=0)
        self.X_std = self.X.std(dim=0) + 1e-8
        self.Y_mean = self.Y.mean()
        self.Y_std = self.Y.std() + 1e-8
        self.Xn = (self.X - self.X_mean) / self.X_std
        self.Yn = (self.Y - self.Y_mean) / self.Y_std

    def _apply_normalization(self):
        self._ensure_stats()
        self.Xn = (self.X - self.X_mean) / self.X_std
        self.Yn = (self.Y - self.Y_mean) / self.Y_std

    def _init_model(self):
        assert self.Xn is not None and self.Yn is not None
        N = self.Xn.size(0)
        M = min(self.num_inducing, N)
        perm = torch.randperm(N, device=self.device)
        inducing_points = self.Xn[perm[:M]].contiguous()

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.likelihood.noise_covar.initialize(noise=1e-3)

        self.model = SparseGPModel(inducing_points, kernel=self.kernel, ard_dims=self.input_dim).to(self.device)

        self.mll = gpytorch.mlls.VariationalELBO(
            likelihood=self.likelihood,
            model=self.model,
            num_data=self.Yn.size(0),
        )

    def _train_steps(self, steps: int):
        assert self.model is not None and self.likelihood is not None and self.mll is not None
        assert self.Xn is not None and self.Yn is not None

        self.model.train()
        self.likelihood.train()

        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        ds = TensorDataset(self.Xn, self.Yn)
        dl = DataLoader(ds, batch_size=self.batch_size, shuffle=True)

        step = 0
        while step < steps:
            for xb, yb in dl:
                opt.zero_grad()
                out = self.model(xb)
                loss = -self.mll(out, yb)
                loss.backward()
                opt.step()

                step += 1
                if step >= steps:
                    break

        self.model.eval()
        self.likelihood.eval()
        self.trained = True

    def fit(self, X: np.ndarray, Y: np.ndarray):
        X = torch.as_tensor(np.asarray(X, dtype=np.float32), device=self.device, dtype=self.dtype)
        Y = torch.as_tensor(np.asarray(Y, dtype=np.float32).reshape(-1), device=self.device, dtype=self.dtype)

        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"X must be (N,{self.input_dim}), got {tuple(X.shape)}")
        if Y.ndim != 1 or Y.shape[0] != X.shape[0]:
            raise ValueError(f"Y must be (N,), got {tuple(Y.shape)} with N={X.shape[0]}")

        self.X = X.clone()
        self.Y = Y.clone()
        self._compute_normalization_initial()
        self._init_model()
        self._train_steps(self.init_train_steps)

    def add_data(self, X_new: np.ndarray, Y_new: np.ndarray, online_steps: int = 50):
        X_new = torch.as_tensor(np.asarray(X_new, dtype=np.float32), device=self.device, dtype=self.dtype)
        Y_new = torch.as_tensor(np.asarray(Y_new, dtype=np.float32).reshape(-1), device=self.device, dtype=self.dtype)

        if self.X is None:
            self.fit(X_new.detach().cpu().numpy(), Y_new.detach().cpu().numpy())
            return

        self.X = torch.cat([self.X, X_new], dim=0)
        self.Y = torch.cat([self.Y, Y_new], dim=0)

        # keep old stats (online-stable)
        self._apply_normalization()

        assert self.mll is not None
        self.mll.num_data = self.Yn.size(0)

        if online_steps > 0:
            self._train_steps(int(online_steps))

    @torch.no_grad()
    def predict_torch(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.trained:
            raise RuntimeError("SVGP not trained yet.")

        if not isinstance(X, torch.Tensor):
            X = torch.as_tensor(X, device=self.device, dtype=self.dtype)
        else:
            X = X.to(self.device, dtype=self.dtype)

        self._ensure_stats()
        Xn = (X - self.X_mean) / self.X_std

        with gpytorch.settings.fast_pred_var():
            pred = self.likelihood(self.model(Xn))
            mean = pred.mean * self.Y_std + self.Y_mean
            var = pred.variance * (self.Y_std ** 2)
        return mean, var

    def state_dict_bundle(self) -> dict:
        """
        Everything needed to restore:
        - model/likelihood weights
        - inducing points are inside model variational strategy
        - normalization stats
        """
        if self.model is None or self.likelihood is None:
            raise RuntimeError("Model not initialized.")
        bundle = {
            "input_dim": self.input_dim,
            "kernel": self.kernel,
            "lr": self.lr,
            "batch_size": self.batch_size,
            "num_inducing": self.num_inducing,
            "init_train_steps": self.init_train_steps,
            "dtype": str(self.dtype),
            "trained": self.trained,
            "X_mean": None if self.X_mean is None else self.X_mean.detach().cpu(),
            "X_std": None if self.X_std is None else self.X_std.detach().cpu(),
            "Y_mean": None if self.Y_mean is None else self.Y_mean.detach().cpu(),
            "Y_std": None if self.Y_std is None else self.Y_std.detach().cpu(),
            "model_state": self.model.state_dict(),
            "likelihood_state": self.likelihood.state_dict(),
        }
        return bundle

    def load_from_bundle(self, bundle: dict):
        self.input_dim = int(bundle["input_dim"])
        self.kernel = bundle["kernel"]
        self.lr = float(bundle["lr"])
        self.batch_size = int(bundle["batch_size"])
        self.num_inducing = int(bundle["num_inducing"])
        self.init_train_steps = int(bundle["init_train_steps"])
        self.trained = bool(bundle.get("trained", False))

        # restore stats
        def to_dev(x):
            return None if x is None else x.to(self.device, dtype=self.dtype)

        self.X_mean = to_dev(bundle["X_mean"])
        self.X_std = to_dev(bundle["X_std"])
        self.Y_mean = to_dev(bundle["Y_mean"])
        self.Y_std = to_dev(bundle["Y_std"])

        model_state = bundle["model_state"]
        like_state = bundle["likelihood_state"]

        # ✅ CRITICAL: infer M from checkpoint, not from current config
        if "variational_strategy.inducing_points" not in model_state:
            raise RuntimeError("Checkpoint missing inducing_points in model_state.")

        M_ckpt = model_state["variational_strategy.inducing_points"].shape[0]

        # build model with correct M
        inducing_dummy = torch.zeros((M_ckpt, self.input_dim), device=self.device, dtype=self.dtype)

        self.likelihood = gpytorch.likelihoods.GaussianLikelihood().to(self.device)
        self.model = SparseGPModel(inducing_dummy, kernel=self.kernel, ard_dims=self.input_dim).to(self.device)

        # load weights
        self.model.load_state_dict(model_state, strict=True)
        self.likelihood.load_state_dict(like_state, strict=True)

        # rebuild ELBO (num_data placeholder, updated when training/adding data)
        self.mll = gpytorch.mlls.VariationalELBO(self.likelihood, self.model, num_data=1)

        self.model.eval()
        self.likelihood.eval()