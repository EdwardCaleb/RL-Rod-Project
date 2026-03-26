import os
import time
import argparse
import numpy as np
import mujoco
from mujoco import viewer
import torch
import tqdm

from mellinguer import MellinguerControllerForce
from MPPI import SingleMPPIPlannerTorch
from helper import (
    quat_to_rotmat,
    ForceArrow,
    enforce_tilt_and_thrust_limits,
    get_drone_state,
    set_goal_for_episode,
    reset_episode,
)
from dynamics import (
    SingleMassDynamicModelTorch,
    SVGPDroneDynamicModel,
    SVGPDroneAccelerationDynamicModel,
    SVGPDroneResidualAccelerationDynamicModel,
)


# libiomp5md.dll, but found libiomp5md.dll already initialized. PROBLEM with OpenMP runtime!
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # UNSAFE workaround, only used for experimentation.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_XML_PATH = os.path.join(BASE_DIR, "system", "single_quad_obs_u1234.xml")
DEFAULT_LOG_DIR = os.path.join(BASE_DIR, "logs")


# ============================================================
# --------- 1) CONFIGURACIÓN GENERAL -------------------------
# ============================================================
NUM_EPISODES = 10
STEPS_PER_EP = 700
RENDER = True
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# MPPI scheduler
MPPI_EVERY = 1

# MPPI plan fallback
YAW_TARGET = 0.0
HEIGHT_TARGET = 1.25

# Si quieres que el GP se resetee cada episodio (empezar de cero):
RESET_GP_EACH_EP = False

# Guardar cada episodio
SAVE_EACH_EP = False

# Guardar y entrenar modelo final al terminar todos los episodios
SAVE_FINAL = False

# Modelo que aprende online con los datos de la simulación.
# Opciones:
#   - "GP_DX"         : aprende ΔX completo (modelo original)
#   - "GP_ACCEL"      : aprende a = g(F)
#   - "GP_RES_ACCEL"  : aprende a = F/m + [0,0,-g] + g(F)
LEARNED_DYN_MODEL_TYPE = "GP_DX"

# Modelo que usa MPPI para planear.
# Opciones:
#   - "MASS"          : punto-masa nominal
#   - "GP_DX"
#   - "GP_ACCEL"
#   - "GP_RES_ACCEL"
PLANNER_DYN_MODEL_TYPE = "GP_RES_ACCEL"


# ============================================================
#                 HELPERS DE CONFIG Y FACTORY
# ============================================================
GP_MODEL_FILENAME = {
    "GP_DX": "svgp_dyn_dx.pt",
    "GP_ACCEL": "svgp_dyn_accel.pt",
    "GP_RES_ACCEL": "svgp_dyn_res_accel.pt",
}

ROLLOUTS_FILENAME = {
    "GP_DX": "rollouts_dx_all.npz",
    "GP_ACCEL": "rollouts_accel_all.npz",
    "GP_RES_ACCEL": "rollouts_res_accel_all.npz",
}

EP_ROLLOUTS_FILENAME = {
    "GP_DX": "rollouts_dx_ep{:03d}.npz",
    "GP_ACCEL": "rollouts_accel_ep{:03d}.npz",
    "GP_RES_ACCEL": "rollouts_res_accel_ep{:03d}.npz",
}


GP_COMMON_CFG = dict(
    kernel="RBF",
    lr=0.01,
    batch_size=256,
    num_inducing=2**5,
    init_train_steps=800,
    train_every=20,
    online_steps=50,
    min_points_to_train=300,
    reset_each_episode=RESET_GP_EACH_EP,
    predict_variance=False,
)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_artifact_paths(log_dir: str, model_type: str):
    if model_type not in GP_MODEL_FILENAME:
        raise ValueError(f"No hay artefactos persistentes definidos para {model_type}.")
    return (
        os.path.join(log_dir, GP_MODEL_FILENAME[model_type]),
        os.path.join(log_dir, ROLLOUTS_FILENAME[model_type]),
        os.path.join(log_dir, EP_ROLLOUTS_FILENAME[model_type]),
    )


def build_dynamics_model(model_type: str, dt: float, mass: float, gravity: float, device: str, **extra_kwargs):
    if model_type == "MASS":
        return SingleMassDynamicModelTorch(dt=dt, mass=mass, device=device)

    kwargs = dict(dt=dt, mass=mass, gravity=gravity, device=device)
    kwargs.update(GP_COMMON_CFG)
    kwargs.update(extra_kwargs)

    if model_type == "GP_DX":
        return SVGPDroneDynamicModel(**kwargs)
    if model_type == "GP_ACCEL":
        return SVGPDroneAccelerationDynamicModel(**kwargs)
    if model_type == "GP_RES_ACCEL":
        return SVGPDroneResidualAccelerationDynamicModel(**kwargs)

    raise ValueError(f"Tipo de modelo dinámico desconocido: {model_type}")


def is_gp_model(model_type: str) -> bool:
    return model_type in {"GP_DX", "GP_ACCEL", "GP_RES_ACCEL"}


def get_num_transitions(dyn_model) -> int:
    if hasattr(dyn_model, "Z_buf"):
        return len(dyn_model.Z_buf)
    if hasattr(dyn_model, "F_buf"):
        return len(dyn_model.F_buf)
    return 0


# ============================================================
#                      MAIN
# ============================================================
def main(xml_path: str = DEFAULT_XML_PATH, log_dir: str = DEFAULT_LOG_DIR):
    ensure_dir(log_dir)

    # ============================================================
    # --------- 1) CONFIGURACIÓN MUJOCO --------------------------
    # ============================================================
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    drone_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone_3d")
    if drone_id < 0:
        raise ValueError("No existe el body 'drone_3d' en el XML.")

    mujoco.mj_forward(model, data)
    p0 = data.xpos[drone_id].copy()
    m = float(model.body_subtreemass[drone_id])

    print(f"dt: {dt:.4f}  mass: {m:.3f} kg  p0: {p0}")
    print(f"device: {DEVICE}")
    print(f"learner: {LEARNED_DYN_MODEL_TYPE} | planner: {PLANNER_DYN_MODEL_TYPE}")

    # actuadores
    act_u1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "u1_thrust")
    act_u2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "u2_tau_x")
    act_u3 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "u3_tau_y")
    act_u4 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "u4_tau_z")
    if min(act_u1, act_u2, act_u3, act_u4) < 0:
        raise ValueError("No encontré actuadores u1_thrust/u2_tau_x/u3_tau_y/u4_tau_z. Revisa el XML.")

    arrow = ForceArrow(model, data, arrow_idx=1) if RENDER else None

    # ============================================================
    # --------- 2) Mellinguer Controller -------------------------
    # ============================================================
    gains = [
        np.array([5.0, 5.0, 10.0]),
        np.array([3.0, 3.0, 6.0]),
        np.array([8.0, 8.0, 2.0]),
        np.array([0.2, 0.2, 0.1]),
    ]
    controller = MellinguerControllerForce(mass=m, gravity=9.81, gains=gains)

    # ============================================================
    # --------- 3) Modelo a aprender online ----------------------
    # ============================================================
    if not is_gp_model(LEARNED_DYN_MODEL_TYPE):
        raise ValueError("LEARNED_DYN_MODEL_TYPE debe ser un modelo GP.")

    model_path, rollouts_path, ep_rollouts_fmt = get_artifact_paths(log_dir, LEARNED_DYN_MODEL_TYPE)

    dyn_gp = build_dynamics_model(
        LEARNED_DYN_MODEL_TYPE,
        dt=dt,
        mass=m,
        gravity=9.81,
        device=DEVICE,
    )

    if os.path.exists(model_path):
        print(f"[LOAD] model: {model_path}")
        dyn_gp.load_model(model_path)

    if os.path.exists(rollouts_path):
        print(f"[LOAD] rollouts: {rollouts_path}")
        dyn_gp.load_rollouts(rollouts_path, append=True)
        if not getattr(dyn_gp, "_trained_once", False):
            print("[TRAIN] full from loaded rollouts...")
            dyn_gp.train_full()

    # ============================================================
    # --------- 4) MPPI Planner ----------------------------------
    # ============================================================
    Fxy = 8.0 * m
    Fz_min = 0.0
    Fz_max = 25.0 * m

    planner = SingleMPPIPlannerTorch(
        dt=dt,
        horizon=50,
        num_samples=1024 // 2,
        lambda_=10.0,
        noise_sigma=np.array([2.0 * m, 2.0 * m, 4.0 * m]),
        F_min=np.array([-Fxy, -Fxy, Fz_min]),
        F_max=np.array([Fxy, Fxy, Fz_max]),
        w_goal=6.0,
        w_terminal=50.0,
        w_F=0.03,
        w_smooth=0.02,
        w_obs=80.0,
        obs_margin=0.20,
        obs_softness=0.15,
        goal_tolerance=0.10,
        z_min=HEIGHT_TARGET - 0.25,
        z_max=HEIGHT_TARGET + 0.25,
        z_margin=0.15,
        w_z=200.0,
        w_z_terminal=400.0,
        v_max=5.0,
        v_margin=0.5,
        w_v=100.0,
        w_v_terminal=200.0,
        rng_seed=0,
        device=DEVICE,
    )

    if PLANNER_DYN_MODEL_TYPE == "MASS":
        planner_dyn_model = build_dynamics_model(
            "MASS",
            dt=dt,
            mass=m,
            gravity=9.81,
            device=DEVICE,
        )
    else:
        planner_dyn_model = build_dynamics_model(
            PLANNER_DYN_MODEL_TYPE,
            dt=dt,
            mass=m,
            gravity=9.81,
            device=DEVICE,
        )

        planner_model_path, _, _ = get_artifact_paths(log_dir, PLANNER_DYN_MODEL_TYPE)
        if os.path.exists(planner_model_path):
            print(f"[LOAD planner model] {planner_model_path}")
            planner_dyn_model.load_model(planner_model_path)
        else:
            print(f"[WARN] No existe checkpoint para planner en {planner_model_path}. Se usará fallback físico del modelo.")

    planner.define_model(planner_dyn_model)

    obstacles = [
        {"type": "sphere", "c": [1.0, 0.0, 1.0], "r": 0.25},
        {"type": "box", "c": [1.6, 0.4, 1.0], "h": [0.20, 0.20, 0.40]},
        {"type": "box", "c": [1.6, -0.4, 1.0], "h": [0.20, 0.20, 0.40]},
    ]
    planner.set_obstacles(obstacles)

    vector_r = []

    def run_episodes(vis=None):
        nonlocal p0

        for ep in range(NUM_EPISODES):
            print(f"\n=== EPISODE {ep+1}/{NUM_EPISODES} ===")

            reset_episode(model, data)
            dyn_gp.reset_episode(clear_buffers=False)

            goal = set_goal_for_episode(p0, HEIGHT_TARGET, ep)
            planner.set_goal(goal)

            print(f"goal: {goal}")

            plan_idx = MPPI_EVERY
            F_plan = None
            F_fallback = np.array([0.0, 0.0, m * 9.81], dtype=float)

            tilt_max = np.deg2rad(35.0)
            u1_max = Fz_max

            for step in tqdm.tqdm(range(STEPS_PER_EP)):
                if vis is not None and not vis.is_running():
                    print("Viewer closed.")
                    return

                r, v_world, R, omega_body = get_drone_state(model, data, drone_id)
                vector_r.append(r)

                if plan_idx >= MPPI_EVERY or F_plan is None:
                    _, _, _, F_seq = planner.compute_action(r, v_world)
                    F_plan = F_seq
                    plan_idx = 0

                if plan_idx < len(F_plan):
                    F_des = F_plan[plan_idx].copy()
                else:
                    F_des = F_fallback.copy()
                plan_idx += 1

                F_des[2] = max(F_des[2], 0.0)

                if np.linalg.norm(r - goal) < planner.goal_tolerance:
                    F_des = np.array([0.0, 0.0, m * 9.81]) - np.array([2.0, 2.0, 3.0]) * v_world

                F_des = enforce_tilt_and_thrust_limits(F_des, u1_max=u1_max, tilt_max_rad=tilt_max)

                if arrow is not None:
                    arrow.update_force_arrow_mocap(
                        p0_world=r,
                        F_world=F_des,
                        scale=0.03,
                        max_len=0.8,
                        radius=0.01,
                    )

                u = controller.step(F_des, R, omega_body, psi_T=YAW_TARGET)

                data.ctrl[act_u1] = float(u[0])
                data.ctrl[act_u2] = float(u[1])
                data.ctrl[act_u3] = float(u[2])
                data.ctrl[act_u4] = float(u[3])

                pos0 = r.copy()
                vel0 = v_world.copy()
                F0 = F_des.copy()

                mujoco.mj_step(model, data)

                r2, v2_world, _, _ = get_drone_state(model, data, drone_id)
                dyn_gp.add_transition(pos0, vel0, F0, r2, v2_world, train_online=False)

                if vis is not None:
                    vis.sync()

            print(f"[EP {ep}] steps collected: {get_num_transitions(dyn_gp)}")

            if SAVE_EACH_EP:
                ep_path = ep_rollouts_fmt.format(ep)
                dyn_gp.save_rollouts(ep_path)
                print(f"[SAVE] episode rollouts: {ep_path}")

                if os.path.exists(rollouts_path):
                    tmp = build_dynamics_model(
                        LEARNED_DYN_MODEL_TYPE,
                        dt=dt,
                        mass=m,
                        gravity=9.81,
                        device=DEVICE,
                    )
                    tmp.load_rollouts(rollouts_path, append=False)
                    tmp.load_rollouts(ep_path, append=True)
                    tmp.save_rollouts(rollouts_path)
                else:
                    dyn_gp.save_rollouts(rollouts_path)

                dyn_gp.save_model(model_path)
                print(f"[SAVE] model: {model_path}")
                print(f"[SAVE] rollouts_all: {rollouts_path}")

        if SAVE_FINAL:
            print("\n[FINAL TRAIN] training full on all collected data...")
            print(f"total collected transitions: {get_num_transitions(dyn_gp)}")
            dyn_gp.train_full()
            dyn_gp.save_model(model_path)
            dyn_gp.save_rollouts(rollouts_path)

    if RENDER:
        with viewer.launch_passive(model, data) as vis:
            run_episodes(vis=vis)
    else:
        run_episodes(vis=None)

    import matplotlib.pyplot as plt

    vector_r_np = np.array(vector_r)
    print(f"p0: {p0}")
    goal = set_goal_for_episode(p0, HEIGHT_TARGET, 0)

    print(f"goal: {goal}")
    if len(vector_r_np) > 0:
        print(f"final position: {vector_r_np[-1]}")

        plt.figure()
        plt.plot(vector_r_np[:, 0], vector_r_np[:, 2], label="Trayectoria del dron")
        plt.scatter(goal[0], goal[2], color="red", label="Goal", marker="X", s=100)
        plt.xlabel("x (m)")
        plt.ylabel("z (m)")
        plt.title("Side view (x-z)")
        plt.legend()
        plt.xlim(p0[0] - 0.5, p0[0] + 3.0)
        plt.ylim(p0[2] - 0.5, p0[2] + 1.5)
        plt.show()

        plt.figure()
        plt.plot(vector_r_np[:, 0], vector_r_np[:, 1], label="Trayectoria del dron")
        plt.scatter(goal[0], goal[1], color="red", label="Goal", marker="X", s=100)
        plt.xlabel("x (m)")
        plt.ylabel("y (m)")
        plt.title("Top view (x-y)")
        plt.legend()
        plt.xlim(p0[0] - 0.5, p0[0] + 3.0)
        plt.ylim(p0[1] - 1.5, p0[1] + 1.5)
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--xml", type=str, default=DEFAULT_XML_PATH, help="Ruta al XML de MuJoCo")
    parser.add_argument("--log_dir", type=str, default=DEFAULT_LOG_DIR, help="Directorio para checkpoints y rollouts")
    args = parser.parse_args()
    main(xml_path=args.xml, log_dir=args.log_dir)
