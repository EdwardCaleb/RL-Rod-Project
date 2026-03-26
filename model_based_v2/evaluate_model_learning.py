import os
import csv
import math
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
import matplotlib.pyplot as plt

# Keep same OpenMP workaround used elsewhere in project.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "True")

from dynamics import (
    SingleMassDynamicModelTorch,
    SVGPDroneDynamicModel,
    SVGPDroneAccelerationDynamicModel,
    SVGPDroneResidualAccelerationDynamicModel,
)


MODEL_FILENAMES = {
    "GP_DX": "svgp_dyn_dx.pt",
    "GP_ACCEL": "svgp_dyn_accel.pt",
    "GP_RES_ACCEL": "svgp_dyn_res_accel.pt",
}

ROLLOUT_FILENAMES = {
    "GP_DX": "rollouts_dx_all.npz",
    "GP_ACCEL": "rollouts_accel_all.npz",
    "GP_RES_ACCEL": "rollouts_res_accel_all.npz",
}

DEFAULT_HORIZONS = [1, 5, 10, 20, 50]


@dataclass
class EvalDatasetDX:
    Z: np.ndarray   # (N,9) = [pos(3), vel(3), F(3)]
    dX: np.ndarray  # (N,6) = [Δpos(3), Δvel(3)]
    dt: float
    mass: float
    gravity: float

    @property
    def X(self) -> np.ndarray:
        return self.Z[:, :6]

    @property
    def F(self) -> np.ndarray:
        return self.Z[:, 6:9]

    @property
    def X_next(self) -> np.ndarray:
        return self.X + self.dX

    @property
    def A(self) -> np.ndarray:
        return self.dX[:, 3:6] / self.dt


@dataclass
class SplitDX:
    train: EvalDatasetDX
    test: EvalDatasetDX


# -----------------------------------------------------------------------------
# Helpers: loading and metrics
# -----------------------------------------------------------------------------
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _dtype_from_str(s: str) -> torch.dtype:
    s = str(s).lower()
    if "float64" in s or "double" in s:
        return torch.float64
    return torch.float32


def load_checkpoint_bundle(path: str) -> dict:
    return torch.load(path, map_location="cpu")


def load_dx_rollouts(path: str, dt: float, mass: float, gravity: float) -> EvalDatasetDX:
    data = np.load(path)
    if "Z" not in data or "dX" not in data:
        raise ValueError(
            f"El archivo {path} no es un rollout tipo GP_DX. Se esperaban claves 'Z' y 'dX'."
        )
    Z = data["Z"].astype(np.float32)
    dX = data["dX"].astype(np.float32)
    if Z.ndim != 2 or dX.ndim != 2 or Z.shape[1] != 9 or dX.shape[1] != 6 or Z.shape[0] != dX.shape[0]:
        raise ValueError(f"Shapes inválidos en {path}: Z{Z.shape}, dX{dX.shape}")
    return EvalDatasetDX(Z=Z, dX=dX, dt=float(dt), mass=float(mass), gravity=float(gravity))


def temporal_split_dx(ds: EvalDatasetDX, test_frac: float) -> SplitDX:
    n = ds.Z.shape[0]
    n_test = max(1, int(round(n * test_frac)))
    n_train = max(1, n - n_test)
    if n_train >= n:
        n_train = n - 1
    return SplitDX(
        train=EvalDatasetDX(ds.Z[:n_train], ds.dX[:n_train], ds.dt, ds.mass, ds.gravity),
        test=EvalDatasetDX(ds.Z[n_train:], ds.dX[n_train:], ds.dt, ds.mass, ds.gravity),
    )


def rmse(y_true: np.ndarray, y_pred: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=axis))


def mae(y_true: np.ndarray, y_pred: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.mean(np.abs(y_true - y_pred), axis=axis)


def r2(y_true: np.ndarray, y_pred: np.ndarray, axis: int = 0) -> np.ndarray:
    ss_res = np.sum((y_true - y_pred) ** 2, axis=axis)
    y_bar = np.mean(y_true, axis=axis, keepdims=True)
    ss_tot = np.sum((y_true - y_bar) ** 2, axis=axis)
    return 1.0 - (ss_res / np.maximum(ss_tot, 1e-12))


def flatten_metric_dict(prefix: str, y_true: np.ndarray, y_pred: np.ndarray, dim_names: List[str]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    rmse_v = rmse(y_true, y_pred)
    mae_v = mae(y_true, y_pred)
    r2_v = r2(y_true, y_pred)
    for i, name in enumerate(dim_names):
        out[f"{prefix}_rmse_{name}"] = float(rmse_v[i])
        out[f"{prefix}_mae_{name}"] = float(mae_v[i])
        out[f"{prefix}_r2_{name}"] = float(r2_v[i])
    out[f"{prefix}_rmse_mean"] = float(np.mean(rmse_v))
    out[f"{prefix}_mae_mean"] = float(np.mean(mae_v))
    out[f"{prefix}_r2_mean"] = float(np.mean(r2_v))
    return out


# -----------------------------------------------------------------------------
# Model builders / predictions
# -----------------------------------------------------------------------------
def build_model_from_type(model_type: str, dt: float, mass: float, gravity: float, device: str, dtype: torch.dtype):
    if model_type == "MASS":
        return SingleMassDynamicModelTorch(dt=dt, mass=mass, device=device, dtype=dtype)
    if model_type == "GP_DX":
        return SVGPDroneDynamicModel(dt=dt, mass=mass, gravity=gravity, device=device, dtype=dtype)
    if model_type == "GP_ACCEL":
        return SVGPDroneAccelerationDynamicModel(dt=dt, mass=mass, gravity=gravity, device=device, dtype=dtype)
    if model_type == "GP_RES_ACCEL":
        return SVGPDroneResidualAccelerationDynamicModel(dt=dt, mass=mass, gravity=gravity, device=device, dtype=dtype)
    raise ValueError(f"Tipo de modelo desconocido: {model_type}")


def available_checkpoint_paths(log_dir: str) -> Dict[str, str]:
    out = {}
    for k, fname in MODEL_FILENAMES.items():
        path = os.path.join(log_dir, fname)
        if os.path.exists(path):
            out[k] = path
    return out


def infer_common_eval_rollouts(log_dir: str, explicit_path: Optional[str]) -> Optional[str]:
    if explicit_path:
        return explicit_path
    auto = os.path.join(log_dir, ROLLOUT_FILENAMES["GP_DX"])
    if os.path.exists(auto):
        return auto
    return None


def load_checkpoint_models(paths: Dict[str, str], device: str) -> Dict[str, object]:
    models = {}
    for model_type, path in paths.items():
        bundle = load_checkpoint_bundle(path)
        dt = float(bundle["dt"])
        mass = float(bundle["mass"])
        gravity = float(bundle.get("g", 9.81))
        dtype = _dtype_from_str(bundle.get("dtype", "float32"))
        model = build_model_from_type(model_type, dt=dt, mass=mass, gravity=gravity, device=device, dtype=dtype)
        model.load_model(path, map_location=device)
        models[model_type] = model
    return models


@torch.no_grad()
def predict_next_state_batch(model, model_type: str, pos: np.ndarray, vel: np.ndarray, force: np.ndarray, device: str) -> Tuple[np.ndarray, np.ndarray]:
    dtype = getattr(model, "dtype", torch.float32)
    pos_t = torch.as_tensor(pos, device=device, dtype=dtype)
    vel_t = torch.as_tensor(vel, device=device, dtype=dtype)
    F_t = torch.as_tensor(force, device=device, dtype=dtype)

    if model_type == "MASS":
        p2, v2 = model.step(pos_t, vel_t, F_t)
    else:
        p2, v2 = model.step_torch(pos_t, vel_t, F_t)
    return p2.detach().cpu().numpy(), v2.detach().cpu().numpy()


@torch.no_grad()
def predict_acc_batch(model, model_type: str, vel: np.ndarray, vel_next_true: np.ndarray, force: np.ndarray, dt: float, device: str) -> np.ndarray:
    if model_type in {"GP_ACCEL", "GP_RES_ACCEL"}:
        dtype = getattr(model, "dtype", torch.float32)
        F_t = torch.as_tensor(force, device=device, dtype=dtype)
        a_pred, _ = model.predict_acceleration_torch(F_t, return_var=False)
        return a_pred.detach().cpu().numpy()

    pos_dummy = np.zeros_like(force, dtype=np.float32)
    _, v2_pred = predict_next_state_batch(model, model_type, pos_dummy, vel, force, device=device)
    return (v2_pred - vel) / dt


# -----------------------------------------------------------------------------
# Rollout evaluation on common DX dataset
# -----------------------------------------------------------------------------
def evaluate_one_step_on_dx_dataset(model, model_type: str, ds: EvalDatasetDX, device: str) -> Dict[str, object]:
    pos = ds.Z[:, 0:3]
    vel = ds.Z[:, 3:6]
    F = ds.Z[:, 6:9]
    X_next_true = ds.X_next
    A_true = ds.A

    pos_pred, vel_pred = predict_next_state_batch(model, model_type, pos, vel, F, device=device)
    X_next_pred = np.concatenate([pos_pred, vel_pred], axis=1)
    A_pred = predict_acc_batch(model, model_type, vel, X_next_true[:, 3:6], F, dt=ds.dt, device=device)

    metrics = {}
    metrics.update(flatten_metric_dict("next_pos", X_next_true[:, 0:3], X_next_pred[:, 0:3], ["x", "y", "z"]))
    metrics.update(flatten_metric_dict("next_vel", X_next_true[:, 3:6], X_next_pred[:, 3:6], ["vx", "vy", "vz"]))
    metrics.update(flatten_metric_dict("next_state", X_next_true, X_next_pred, ["x", "y", "z", "vx", "vy", "vz"]))
    metrics.update(flatten_metric_dict("acc", A_true, A_pred, ["ax", "ay", "az"]))

    return {
        "metrics": metrics,
        "pred": {
            "X_next_true": X_next_true,
            "X_next_pred": X_next_pred,
            "A_true": A_true,
            "A_pred": A_pred,
        },
    }


def continuity_mask_from_dx(ds: EvalDatasetDX, tol: float = 1e-4) -> np.ndarray:
    x_now = ds.X[:-1]
    x_next_from_dx = ds.X[:-1] + ds.dX[:-1]
    x_next_logged = ds.X[1:]
    err = np.max(np.abs(x_next_from_dx - x_next_logged), axis=1)
    return err < tol


@torch.no_grad()
def rollout_open_loop_final_error(model, model_type: str, ds: EvalDatasetDX, horizons: List[int], device: str, continuity_tol: float = 1e-4) -> List[Dict[str, float]]:
    cont = continuity_mask_from_dx(ds, tol=continuity_tol)
    results = []

    for H in horizons:
        valid_starts = []
        for i in range(0, max(0, len(ds.Z) - H)):
            if H == 1:
                valid_starts.append(i)
                continue
            if np.all(cont[i:i + H - 1]):
                valid_starts.append(i)

        if len(valid_starts) == 0:
            results.append({
                "horizon": H,
                "n_windows": 0,
                "final_pos_rmse": float("nan"),
                "final_vel_rmse": float("nan"),
                "final_state_rmse": float("nan"),
            })
            continue

        final_true = []
        final_pred = []

        for i in valid_starts:
            pos = ds.Z[i, 0:3].copy()
            vel = ds.Z[i, 3:6].copy()

            pos_pred = pos[None, :]
            vel_pred = vel[None, :]
            for k in range(H):
                Fk = ds.Z[i + k:i + k + 1, 6:9]
                pos_pred, vel_pred = predict_next_state_batch(model, model_type, pos_pred, vel_pred, Fk, device=device)

            x_true = ds.X[i].copy()
            for k in range(H):
                x_true = x_true + ds.dX[i + k]

            final_true.append(x_true)
            final_pred.append(np.concatenate([pos_pred[0], vel_pred[0]], axis=0))

        final_true = np.stack(final_true, axis=0)
        final_pred = np.stack(final_pred, axis=0)

        pos_rmse = float(np.mean(rmse(final_true[:, 0:3], final_pred[:, 0:3])))
        vel_rmse = float(np.mean(rmse(final_true[:, 3:6], final_pred[:, 3:6])))
        state_rmse = float(np.mean(rmse(final_true, final_pred)))
        results.append({
            "horizon": H,
            "n_windows": len(valid_starts),
            "final_pos_rmse": pos_rmse,
            "final_vel_rmse": vel_rmse,
            "final_state_rmse": state_rmse,
        })

    return results


# -----------------------------------------------------------------------------
# Fair retraining from a common DX dataset
# -----------------------------------------------------------------------------
def get_default_train_cfg() -> dict:
    return dict(
        kernel="RBF",
        lr=0.01,
        batch_size=256,
        num_inducing=2 ** 5,
        init_train_steps=800,
        train_every=20,
        online_steps=50,
        min_points_to_train=300,
        reset_each_episode=False,
        predict_variance=False,
    )


def cfg_from_checkpoint_if_available(model_type: str, checkpoint_paths: Dict[str, str]) -> dict:
    cfg = get_default_train_cfg()
    path = checkpoint_paths.get(model_type)
    if not path:
        return cfg

    bundle = load_checkpoint_bundle(path)
    for k in [
        "kernel", "lr", "batch_size", "num_inducing", "init_train_steps",
        "train_every", "online_steps", "min_points_to_train", "reset_each_episode", "predict_variance"
    ]:
        if k in bundle:
            cfg[k] = bundle[k]
    return cfg


def build_trainable_model(model_type: str, ds: EvalDatasetDX, device: str, dtype: torch.dtype, cfg: dict):
    if model_type == "GP_DX":
        return SVGPDroneDynamicModel(dt=ds.dt, mass=ds.mass, gravity=ds.gravity, device=device, dtype=dtype, **cfg)
    if model_type == "GP_ACCEL":
        return SVGPDroneAccelerationDynamicModel(dt=ds.dt, mass=ds.mass, gravity=ds.gravity, device=device, dtype=dtype, **cfg)
    if model_type == "GP_RES_ACCEL":
        return SVGPDroneResidualAccelerationDynamicModel(dt=ds.dt, mass=ds.mass, gravity=ds.gravity, device=device, dtype=dtype, **cfg)
    raise ValueError(model_type)


def preload_train_split_into_model(model, model_type: str, ds_train: EvalDatasetDX):
    if model_type == "GP_DX":
        model.Z_buf = [z.astype(np.float32) for z in ds_train.Z]
        model.dX_buf = [dx.astype(np.float32) for dx in ds_train.dX]
        return

    F = ds_train.F.astype(np.float32)
    A_total = ds_train.A.astype(np.float32)

    if model_type == "GP_ACCEL":
        A_target = A_total
    elif model_type == "GP_RES_ACCEL":
        nominal = (F / float(ds_train.mass)) + np.array([0.0, 0.0, -float(ds_train.gravity)], dtype=np.float32)[None, :]
        A_target = A_total - nominal
    else:
        raise ValueError(model_type)

    model.F_buf = [f.astype(np.float32) for f in F]
    model.A_buf = [a.astype(np.float32) for a in A_target]


def retrain_models_from_common_dx(split: SplitDX, checkpoint_paths: Dict[str, str], device: str) -> Dict[str, object]:
    models = {}
    dtype = torch.float32
    for model_type in ["GP_DX", "GP_ACCEL", "GP_RES_ACCEL"]:
        cfg = cfg_from_checkpoint_if_available(model_type, checkpoint_paths)
        model = build_trainable_model(model_type, split.train, device=device, dtype=dtype, cfg=cfg)
        preload_train_split_into_model(model, model_type, split.train)
        model.train_full()
        models[model_type] = model
    return models


# -----------------------------------------------------------------------------
# Reporting
# -----------------------------------------------------------------------------
def save_rows_csv(path: str, rows: List[Dict[str, object]]):
    if len(rows) == 0:
        return
    fieldnames = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def plot_bar_summary(rows: List[Dict[str, object]], out_path: str, title: str):
    if len(rows) == 0:
        return

    names = [r["model"] for r in rows]
    pos_rmse = [float(r["next_pos_rmse_mean"]) for r in rows]
    vel_rmse = [float(r["next_vel_rmse_mean"]) for r in rows]
    acc_rmse = [float(r["acc_rmse_mean"]) for r in rows]

    x = np.arange(len(names))
    w = 0.25

    fig = plt.figure(figsize=(10, 5))
    plt.bar(x - w, pos_rmse, width=w, label="Pos RMSE")
    plt.bar(x, vel_rmse, width=w, label="Vel RMSE")
    plt.bar(x + w, acc_rmse, width=w, label="Acc RMSE")
    plt.xticks(x, names, rotation=20)
    plt.ylabel("RMSE")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_rollout_horizons(rows: List[Dict[str, object]], out_path: str, title: str):
    if len(rows) == 0:
        return

    fig = plt.figure(figsize=(9, 5))
    models = sorted(set(r["model"] for r in rows))
    for model in models:
        sub = sorted([r for r in rows if r["model"] == model], key=lambda x: int(x["horizon"]))
        x = [int(r["horizon"]) for r in sub]
        y = [float(r["final_state_rmse"]) for r in sub]
        plt.plot(x, y, marker="o", label=model)

    plt.xlabel("Horizonte open-loop")
    plt.ylabel("RMSE estado final")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close(fig)


def write_markdown_report(
    path: str,
    args,
    dataset_path: str,
    split: SplitDX,
    checkpoint_rows: List[Dict[str, object]],
    checkpoint_rollouts: List[Dict[str, object]],
    retrain_rows: List[Dict[str, object]],
    retrain_rollouts: List[Dict[str, object]],
):
    def best_model(rows: List[Dict[str, object]], key: str) -> str:
        if not rows:
            return "N/D"
        row = min(rows, key=lambda r: float(r.get(key, float("inf"))))
        return f"{row['model']} ({row[key]:.6f})"

    lines = []
    lines.append("# Evaluación amplia de modelos dinámicos\n")
    lines.append("## Configuración\n")
    lines.append(f"- Dataset común usado: `{dataset_path}`\n")
    lines.append(f"- Transiciones train: **{split.train.Z.shape[0]}**\n")
    lines.append(f"- Transiciones test: **{split.test.Z.shape[0]}**\n")
    lines.append(f"- dt: **{split.test.dt:.6f} s**\n")
    lines.append(f"- masa: **{split.test.mass:.6f} kg**\n")
    lines.append(f"- gravedad: **{split.test.gravity:.6f} m/s²**\n")
    lines.append(f"- device solicitado: `{args.device}`\n")
    lines.append(f"- horizontes open-loop: `{args.horizons}`\n")

    lines.append("\n## Cómo interpretar los resultados\n")
    lines.append("- **Checkpoint eval**: mide los modelos ya entrenados tal como están guardados en disco.\n")
    lines.append("- **Retrain eval**: re-entrena cada arquitectura desde cero sobre el mismo split train/test. Esta es la comparación más justa para decidir qué arquitectura aprende mejor.\n")
    lines.append("- **Next-state RMSE**: error a un paso sobre `[x,y,z,vx,vy,vz]`.\n")
    lines.append("- **Acc RMSE**: error sobre aceleración `[ax,ay,az]`, derivada desde `Δv/dt`.\n")
    lines.append("- **Open-loop final-state RMSE**: error al propagar el modelo durante varios pasos consecutivos.\n")

    if checkpoint_rows:
        lines.append("\n## Resumen rápido — modelos guardados (checkpoint eval)\n")
        lines.append(f"- Mejor en `next_state_rmse_mean`: **{best_model(checkpoint_rows, 'next_state_rmse_mean')}**\n")
        lines.append(f"- Mejor en `acc_rmse_mean`: **{best_model(checkpoint_rows, 'acc_rmse_mean')}**\n")
        if checkpoint_rollouts:
            ck_final_rows = [r for r in checkpoint_rollouts if int(r['horizon']) == max(args.horizons)]
            lines.append(f"- Mejor en rollout horizonte {max(args.horizons)}: **{best_model(ck_final_rows, 'final_state_rmse')}**\n")

    if retrain_rows:
        lines.append("\n## Resumen rápido — reentrenando desde cero (retrain eval)\n")
        lines.append(f"- Mejor en `next_state_rmse_mean`: **{best_model(retrain_rows, 'next_state_rmse_mean')}**\n")
        lines.append(f"- Mejor en `acc_rmse_mean`: **{best_model(retrain_rows, 'acc_rmse_mean')}**\n")
        if retrain_rollouts:
            rt_final_rows = [r for r in retrain_rollouts if int(r['horizon']) == max(args.horizons)]
            lines.append(f"- Mejor en rollout horizonte {max(args.horizons)}: **{best_model(rt_final_rows, 'final_state_rmse')}**\n")

    lines.append("\n## Archivos generados\n")
    lines.append("- `metrics_checkpoint_common_dx.csv`\n")
    lines.append("- `rollout_checkpoint_common_dx.csv`\n")
    lines.append("- `metrics_retrain_common_dx.csv`\n")
    lines.append("- `rollout_retrain_common_dx.csv`\n")
    lines.append("- `summary_checkpoint.png`\n")
    lines.append("- `summary_retrain.png`\n")
    lines.append("- `rollout_checkpoint.png`\n")
    lines.append("- `rollout_retrain.png`\n")
    lines.append("- `report.md`\n")

    lines.append("\n## Recomendación práctica\n")
    lines.append(
        "Para decidir qué modelo usar en control, yo miraría en este orden: `open-loop final_state_rmse`, luego `next_state_rmse_mean`, y después `acc_rmse_mean`. "
        "Si un modelo gana sólo a un paso pero se degrada mucho al propagar 10-50 pasos, normalmente será peor dentro de MPPI.\n"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.writelines(lines)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluación amplia de precisión de aprendizaje para GP_DX / GP_ACCEL / GP_RES_ACCEL.")
    parser.add_argument("--log_dir", type=str, default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs"))
    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--eval_rollouts_dx", type=str, default=None, help="Archivo NPZ común tipo GP_DX con Z,dX para evaluar a todos sobre el mismo dataset.")
    parser.add_argument("--test_frac", type=float, default=0.2)
    parser.add_argument("--device", type=str, default=("cuda" if torch.cuda.is_available() else "cpu"))
    parser.add_argument("--horizons", type=int, nargs="+", default=DEFAULT_HORIZONS)
    parser.add_argument("--include_mass", action="store_true", help="Incluye el baseline MASS en la evaluación sobre el dataset común.")
    parser.add_argument("--skip_checkpoint_eval", action="store_true")
    parser.add_argument("--retrain_from_rollouts", action="store_true", help="Reentrena desde cero cada arquitectura sobre el mismo split train/test. Es la comparación más justa.")
    parser.add_argument("--continuity_tol", type=float, default=1e-4)
    args = parser.parse_args()

    out_dir = args.out_dir or os.path.join(args.log_dir, "evaluation_learning")
    ensure_dir(out_dir)

    checkpoint_paths = available_checkpoint_paths(args.log_dir)
    eval_rollouts_path = infer_common_eval_rollouts(args.log_dir, args.eval_rollouts_dx)
    if eval_rollouts_path is None:
        raise FileNotFoundError(
            "No encontré un dataset común tipo GP_DX para comparar todos los modelos. "
            "Pásalo con --eval_rollouts_dx o guarda log_dir/rollouts_dx_all.npz"
        )

    if "GP_DX" in checkpoint_paths:
        bundle_ref = load_checkpoint_bundle(checkpoint_paths["GP_DX"])
    elif checkpoint_paths:
        bundle_ref = load_checkpoint_bundle(next(iter(checkpoint_paths.values())))
    else:
        raise FileNotFoundError(
            "No encontré checkpoints de modelos GP. Necesito al menos uno para inferir dt/mass/gravity."
        )

    dt = float(bundle_ref["dt"])
    mass = float(bundle_ref["mass"])
    gravity = float(bundle_ref.get("g", 9.81))

    ds_all = load_dx_rollouts(eval_rollouts_path, dt=dt, mass=mass, gravity=gravity)
    split = temporal_split_dx(ds_all, test_frac=args.test_frac)

    checkpoint_rows: List[Dict[str, object]] = []
    checkpoint_rollout_rows: List[Dict[str, object]] = []
    retrain_rows: List[Dict[str, object]] = []
    retrain_rollout_rows: List[Dict[str, object]] = []

    # 1) Evaluate saved checkpoints on the common dataset.
    if not args.skip_checkpoint_eval:
        models_ckpt = load_checkpoint_models(checkpoint_paths, device=args.device)
        if args.include_mass:
            models_ckpt["MASS"] = build_model_from_type("MASS", dt=dt, mass=mass, gravity=gravity, device=args.device, dtype=torch.float32)

        for model_name, model in models_ckpt.items():
            out = evaluate_one_step_on_dx_dataset(model, model_name, split.test, device=args.device)
            row = {"model": model_name, **out["metrics"]}
            checkpoint_rows.append(row)

            rollout_rows = rollout_open_loop_final_error(
                model, model_name, split.test, horizons=args.horizons, device=args.device, continuity_tol=args.continuity_tol
            )
            for rr in rollout_rows:
                checkpoint_rollout_rows.append({"model": model_name, **rr})

        save_rows_csv(os.path.join(out_dir, "metrics_checkpoint_common_dx.csv"), checkpoint_rows)
        save_rows_csv(os.path.join(out_dir, "rollout_checkpoint_common_dx.csv"), checkpoint_rollout_rows)
        plot_bar_summary(checkpoint_rows, os.path.join(out_dir, "summary_checkpoint.png"), "Checkpoint eval — one-step metrics")
        plot_rollout_horizons(checkpoint_rollout_rows, os.path.join(out_dir, "rollout_checkpoint.png"), "Checkpoint eval — open-loop rollout")

    # 2) Fair retraining on same split.
    if args.retrain_from_rollouts:
        models_retrain = retrain_models_from_common_dx(split, checkpoint_paths, device=args.device)
        if args.include_mass:
            models_retrain["MASS"] = build_model_from_type("MASS", dt=dt, mass=mass, gravity=gravity, device=args.device, dtype=torch.float32)

        for model_name, model in models_retrain.items():
            out = evaluate_one_step_on_dx_dataset(model, model_name, split.test, device=args.device)
            row = {"model": model_name, **out["metrics"]}
            retrain_rows.append(row)

            rollout_rows = rollout_open_loop_final_error(
                model, model_name, split.test, horizons=args.horizons, device=args.device, continuity_tol=args.continuity_tol
            )
            for rr in rollout_rows:
                retrain_rollout_rows.append({"model": model_name, **rr})

        save_rows_csv(os.path.join(out_dir, "metrics_retrain_common_dx.csv"), retrain_rows)
        save_rows_csv(os.path.join(out_dir, "rollout_retrain_common_dx.csv"), retrain_rollout_rows)
        plot_bar_summary(retrain_rows, os.path.join(out_dir, "summary_retrain.png"), "Retrain eval — one-step metrics")
        plot_rollout_horizons(retrain_rollout_rows, os.path.join(out_dir, "rollout_retrain.png"), "Retrain eval — open-loop rollout")

    write_markdown_report(
        os.path.join(out_dir, "report.md"),
        args=args,
        dataset_path=eval_rollouts_path,
        split=split,
        checkpoint_rows=checkpoint_rows,
        checkpoint_rollouts=checkpoint_rollout_rows,
        retrain_rows=retrain_rows,
        retrain_rollouts=retrain_rollout_rows,
    )

    meta = {
        "log_dir": args.log_dir,
        "out_dir": out_dir,
        "eval_rollouts_dx": eval_rollouts_path,
        "models_found": sorted(list(checkpoint_paths.keys())),
        "include_mass": bool(args.include_mass),
        "retrain_from_rollouts": bool(args.retrain_from_rollouts),
        "test_frac": float(args.test_frac),
        "horizons": [int(h) for h in args.horizons],
        "device": args.device,
    }
    with open(os.path.join(out_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] evaluación guardada en: {out_dir}")
    for fname in sorted(os.listdir(out_dir)):
        print("   -", fname)


if __name__ == "__main__":
    main()
