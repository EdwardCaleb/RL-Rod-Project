import os
import numpy as np
import mujoco
from mujoco import viewer
import tqdm

from mellinguer import MellinguerControllerForce
from MPPI import DoubleMPPIPlannerTorch
from helper import ForceArrow, enforce_tilt_and_thrust_limits, get_drone_state, reset_episode, set_goal_for_episode
from dynamics import SVGPDoubleDroneDynamicModel

# OpenMP workaround (Windows)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

# ============================================================
# CONFIG
# ============================================================
XML_PATH = "D:\\Lehigh\\1st Semester\\Indepedent Study\\Code\\RL-Rod-Project\\model_based_fixed\\system\\2_quad_flexrod_fixed_connect.xml"

NUM_EPISODES = 10
STEPS_PER_EP = 3000
RENDER = True

LOG_DIR = "D:\\Lehigh\\1st Semester\\Indepedent Study\\Code\\RL-Rod-Project\\model_based_fixed\\logs_2drones"
MODEL_PATH = os.path.join(LOG_DIR, "svgp_dyn_2drones.pt")
ROLLOUTS_PATH = os.path.join(LOG_DIR, "rollouts_all_2drones.npz")
EP_ROLLOUTS_FMT = os.path.join(LOG_DIR, "rollouts_ep{:03d}_2drones.npz")
os.makedirs(LOG_DIR, exist_ok=True)

MPPI_EVERY = 1
YAW_TARGET_1 = 0.0
YAW_TARGET_2 = 0.0
HEIGHT_TARGET = 1.25

RESET_GP_EACH_EP = False
SAVE_EACH_EP = False
SAVE_FINAL = False

# "GP" usa el GP (con fallback interno a point-mass si no está entrenado)
DYN_MODEL_TYPE = "GP"  # "GP" o "MASS" (MASS aquí lo dejo usando fallback del propio GP)


def main():
    # ---------------- MuJoCo ----------------
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    # Bodies (según tu XML 2_quad_flexrod_obs_u1234.xml)
    drone_id_1 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "core_1")
    drone_id_2 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "core_2")
    if drone_id_1 < 0 or drone_id_2 < 0:
        raise ValueError("No encontré bodies 'core_1' y/o 'core_2' en el XML.")

    mujoco.mj_forward(model, data)
    p0_1 = data.xpos[drone_id_1].copy()
    p0_2 = data.xpos[drone_id_2].copy()

    m1 = float(model.body_subtreemass[drone_id_1])
    m2 = float(model.body_subtreemass[drone_id_2])

    print(f"dt={dt:.4f}  m1={m1:.3f}  m2={m2:.3f}  p0_1={p0_1}  p0_2={p0_2}")

    # ---------------- actuators ----------------
    def aid(name: str) -> int:
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if idx < 0:
            raise ValueError(f"No encontré actuator '{name}' en el XML.")
        return idx

    act = {
        "u1_1": aid("u1_thrust_1"), "u2_1": aid("u2_tau_x_1"), "u3_1": aid("u3_tau_y_1"), "u4_1": aid("u4_tau_z_1"),
        "u1_2": aid("u1_thrust_2"), "u2_2": aid("u2_tau_x_2"), "u3_2": aid("u3_tau_y_2"), "u4_2": aid("u4_tau_z_2"),
    }

    arrow1 = ForceArrow(model, data, arrow_idx=1) if RENDER else None
    arrow2 = ForceArrow(model, data, arrow_idx=2) if RENDER else None

    # ---------------- controllers ----------------
    gains = [
        np.array([5.0, 5.0, 10.0]),
        np.array([3.0, 3.0, 6.0]),
        np.array([8.0, 8.0, 2.0]),
        np.array([0.2, 0.2, 0.1]),
    ]
    ctrl1 = MellinguerControllerForce(mass=m1, gravity=9.81, gains=gains)
    ctrl2 = MellinguerControllerForce(mass=m2, gravity=9.81, gains=gains)

    # ---------------- GP dynamics model (2 drones) ----------------
    dyn_gp = SVGPDoubleDroneDynamicModel(
        dt=dt,
        mass1=m1,
        mass2=m2,
        gravity=9.81,
        device="cuda",
        kernel="RBF",
        lr=0.01,
        batch_size=256,
        num_inducing=256,
        init_train_steps=800,
        train_every=20,
        online_steps=50,
        min_points_to_train=300,
        reset_each_episode=RESET_GP_EACH_EP,
        predict_variance=False,   # <- importante para que no calcule var (más rápido)
    )

    # cargar modelo si existe
    if os.path.exists(MODEL_PATH):
        print(f"[LOAD] model: {MODEL_PATH}")
        dyn_gp.load_model(MODEL_PATH)

    # cargar rollouts si existe
    if os.path.exists(ROLLOUTS_PATH):
        print(f"[LOAD] rollouts: {ROLLOUTS_PATH}")
        dyn_gp.load_rollouts(ROLLOUTS_PATH, append=True)
        if not getattr(dyn_gp, "_trained_once", False):
            print("[TRAIN] full from loaded rollouts...")
            dyn_gp.train_full()

    # ---------------- MPPI (2 drones) ----------------
    Fxy1 = 8.0 * m1
    Fxy2 = 8.0 * m2
    Fz_min1, Fz_max1 = 0.0, 25.0 * m1
    Fz_min2, Fz_max2 = 0.0, 25.0 * m2

    U_min = np.array([-Fxy1, -Fxy1, Fz_min1,   -Fxy2, -Fxy2, Fz_min2], dtype=float)
    U_max = np.array([ Fxy1,  Fxy1, Fz_max1,    Fxy2,  Fxy2, Fz_max2], dtype=float)

    noise_sigma = np.array([2.0*m1, 2.0*m1, 4.0*m1,   2.0*m2, 2.0*m2, 4.0*m2], dtype=float)

    planner = DoubleMPPIPlannerTorch(
        dt=dt,
        horizon=40,
        num_samples=1024//2,
        lambda_=10.0,
        noise_sigma=noise_sigma,
        U_min=U_min,
        U_max=U_max,
        w_goal=6.0,
        w_terminal=50.0,
        w_u=0.03,
        w_smooth=0.02,
        w_obs=80.0,
        obs_margin=0.20,
        obs_softness=0.15,
        # altura/velocidad
        z_min=HEIGHT_TARGET-0.25, z_max=HEIGHT_TARGET+0.25, z_margin=0.15, w_z=200.0, w_z_terminal=400.0,
        v_max=5.0, v_margin=0.5, w_v=100.0, w_v_terminal=200.0,
        # separación
        w_sep=80.0, sep_min=0.45, sep_softness=0.10,
        rng_seed=0,
        device="cuda",
    )

    if DYN_MODEL_TYPE in ("GP", "MASS"):
        planner.define_model(dyn_gp)  # usa GP double (tiene fallback interno)
    else:
        raise ValueError(f"DYN_MODEL_TYPE desconocido: {DYN_MODEL_TYPE}")

    obstacles = [
        {"type": "sphere", "c": [1.0,  0.0, 1.0], "r": 0.25},
        {"type": "box",   "c": [1.6,  0.4, 1.0], "h": [0.20, 0.20, 0.40]},
        {"type": "box",   "c": [1.6, -0.4, 1.0], "h": [0.20, 0.20, 0.40]},
    ]
    planner.set_obstacles(obstacles)

    # ---------------- run episodes ----------------
    vector_r1, vector_r2 = [], []
    def run_episodes(vis=None):
        nonlocal p0_1, p0_2

        for ep in range(NUM_EPISODES):
            print(f"\n=== EPISODE {ep+1}/{NUM_EPISODES} ===")

            reset_episode(model, data)
            mujoco.mj_forward(model, data)

            p0_1 = data.xpos[drone_id_1].copy()
            p0_2 = data.xpos[drone_id_2].copy()

            vector_r1.append(p0_1)
            vector_r2.append(p0_2)

            dyn_gp.reset_episode(clear_buffers=False)

            goal1 = set_goal_for_episode(p0_1, HEIGHT_TARGET, ep)
            goal2 = set_goal_for_episode(p0_2, HEIGHT_TARGET, ep)
            planner.set_goals(goal1, goal2)
            print(f"goals: g1={goal1}, g2={goal2}")

            plan_idx = MPPI_EVERY
            U_plan = None

            F_hover_1 = np.array([0.0, 0.0, m1 * 9.81], dtype=float)
            F_hover_2 = np.array([0.0, 0.0, m2 * 9.81], dtype=float)
            U_fallback = np.concatenate([F_hover_1, F_hover_2], axis=0)

            tilt_max = np.deg2rad(35.0)

            for _step in tqdm.tqdm(range(STEPS_PER_EP)):
                if vis is not None and not vis.is_running():
                    print("Viewer closed.")
                    return

                r1, v1, R1, omega1 = get_drone_state(model, data, drone_id_1)
                r2, v2, R2, omega2 = get_drone_state(model, data, drone_id_2)
                X0 = np.concatenate([r1, v1, r2, v2], axis=0)  # (12,)

                if plan_idx >= MPPI_EVERY or U_plan is None:
                    U0, _X1_pred, U_seq = planner.compute_action(X0)
                    U_plan = U_seq
                    plan_idx = 0

                U_des = U_plan[plan_idx].copy() if plan_idx < len(U_plan) else U_fallback.copy()
                plan_idx += 1

                F1 = U_des[0:3].copy()
                F2 = U_des[3:6].copy()

                F1[2] = max(F1[2], 0.0)
                F2[2] = max(F2[2], 0.0)

                F1 = enforce_tilt_and_thrust_limits(F1, u1_max=Fz_max1, tilt_max_rad=tilt_max)
                F2 = enforce_tilt_and_thrust_limits(F2, u1_max=Fz_max2, tilt_max_rad=tilt_max)

                if arrow1 is not None:
                    arrow1.update_force_arrow_mocap(p0_world=r1, F_world=F1, scale=0.03, max_len=0.8, radius=0.01)
                if arrow2 is not None:
                    arrow2.update_force_arrow_mocap(p0_world=r2, F_world=F2, scale=0.03, max_len=0.8, radius=0.01)

                u1 = ctrl1.step(F1, R1, omega1, psi_T=YAW_TARGET_1)
                u2 = ctrl2.step(F2, R2, omega2, psi_T=YAW_TARGET_2)

                data.ctrl[act["u1_1"]] = float(u1[0])
                data.ctrl[act["u2_1"]] = float(u1[1])
                data.ctrl[act["u3_1"]] = float(u1[2])
                data.ctrl[act["u4_1"]] = float(u1[3])

                data.ctrl[act["u1_2"]] = float(u2[0])
                data.ctrl[act["u2_2"]] = float(u2[1])
                data.ctrl[act["u3_2"]] = float(u2[2])
                data.ctrl[act["u4_2"]] = float(u2[3])

                # transición (antes de step)
                p1_0, v1_0 = r1.copy(), v1.copy()
                p2_0, v2_0 = r2.copy(), v2.copy()

                mujoco.mj_step(model, data)

                r1n, v1n, _, _ = get_drone_state(model, data, drone_id_1)
                r2n, v2n, _, _ = get_drone_state(model, data, drone_id_2)

                vector_r1.append(r1n)
                vector_r2.append(r2n)

                dyn_gp.add_transition(
                    p1_0, v1_0, p2_0, v2_0,
                    F1, F2,
                    r1n, v1n, r2n, v2n,
                    train_online=False,
                )

                if vis is not None:
                    vis.sync()

            print(f"[EP {ep}] collected transitions total: {len(dyn_gp.Z_buf)}")

            if SAVE_EACH_EP:
                ep_path = EP_ROLLOUTS_FMT.format(ep)
                dyn_gp.save_rollouts(ep_path)
                print(f"[SAVE] episode rollouts: {ep_path}")

                dyn_gp.save_model(MODEL_PATH)
                print(f"[SAVE] model: {MODEL_PATH}")

    # run
    if RENDER:
        with viewer.launch_passive(model, data) as vis:
            run_episodes(vis)
    else:
        run_episodes(None)

    # final train/save
    if SAVE_FINAL:
        print("\n[FINAL TRAIN] training full on all collected data...")
        print(f"total transitions: {len(dyn_gp.Z_buf)}")
        dyn_gp.train_full()
        dyn_gp.save_model(MODEL_PATH)
        dyn_gp.save_rollouts(ROLLOUTS_PATH)
        print(f"[SAVE] model: {MODEL_PATH}")
        print(f"[SAVE] rollouts: {ROLLOUTS_PATH}")

    # plot trajectories (opcional)
    import matplotlib.pyplot as plt
    # 'SVGPDoubleDroneDynamicModel' object has no attribute 'X1_buf'
    # Si quieres plotear las trayectorias, necesitas guardar los estados durante la simulación (ejemplo: vector_r1.append(r1) y vector_r2.append(r2) después de leer los estados), y luego plotear esos vectores aquí. El GP no guarda automáticamente las trayectorias, solo las transiciones para entrenamiento.
    # 
    traj1 = np.array(vector_r1)[:, 0:3]
    traj2 = np.array(vector_r2)[:, 0:3]
    goal1 = set_goal_for_episode(p0_1, HEIGHT_TARGET, 0)
    goal2 = set_goal_for_episode(p0_2, HEIGHT_TARGET, 0)
    plt.figure()
    plt.plot(traj1[:,0], traj1[:,2], label='Dron 1')
    plt.plot(traj2[:,0], traj2[:,2], label='Dron 2')
    plt.scatter(goal1[0], goal1[2], color='red', label='Goal 1', marker='X', s=100)
    plt.scatter(goal2[0], goal2[2], color='blue', label='Goal 2', marker='X', s=100)
    plt.xlabel('x (m)')
    plt.ylabel('z (m)')
    plt.title('Side view (x-z)')
    plt.legend()
    plt.show()
    plt.figure()
    plt.plot(traj1[:,0], traj1[:,1], label='Dron 1')
    plt.plot(traj2[:,0], traj2[:,1], label='Dron 2')
    plt.scatter(goal1[0], goal1[1], color='red', label='Goal 1', marker='X', s=100)
    plt.scatter(goal2[0], goal2[1], color='blue', label='Goal 2', marker='X', s=100)
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.title('Top view (x-y)')
    plt.legend()
    plt.show()





if __name__ == "__main__":
    main()