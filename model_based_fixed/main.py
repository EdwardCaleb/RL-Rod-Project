import os
import time
import numpy as np
import mujoco
from mujoco import viewer

from mellinguer import MellinguerControllerForce
from MPPI import SingleMPPIPlannerTorch
from helper import quat_to_rotmat, ForceArrow, enforce_tilt_and_thrust_limits, get_drone_state, set_goal_for_episode, reset_episode
from dynamics import SingleMassDynamicModelTorch, SVGPDroneDynamicModel
import tqdm


# libiomp5md.dll, but found libiomp5md.dll already initialized. PROBLEM with OpenMP runtime!
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"  # UNSAFE workaround, only used for experimentation.

# ============================================================
# --------- 1) CONFIGURACIÓN MUJOCO --------------------------
# ============================================================
# XML_PATH = "system/single_quad_obs_u1234.xml"
XML_PATH = "D:\\Lehigh\\1st Semester\\Indepedent Study\\Code\\RL-Rod-Project\\model_based\\system\\single_quad_obs_u1234.xml"
# Episodios
NUM_EPISODES = 10 # Episodios
STEPS_PER_EP = 1500

# Render
RENDER = True

# Paths persistencia
LOG_DIR = "D:\\Lehigh\\1st Semester\\Indepedent Study\\Code\\RL-Rod-Project\\model_based\\logs"
MODEL_PATH = os.path.join(LOG_DIR, "svgp_dyn.pt")
ROLLOUTS_PATH = os.path.join(LOG_DIR, "rollouts_all.npz")
EP_ROLLOUTS_FMT = os.path.join(LOG_DIR, "rollouts_ep{:03d}.npz")

os.makedirs(LOG_DIR, exist_ok=True)

# MPPI scheduler
MPPI_EVERY = 1

# MPPI plan fallback
YAW_TARGET = 0.0

# Si quieres que el GP se resetee cada episodio (empezar de cero):
RESET_GP_EACH_EP = False

# Guardar cada episodio
SAVE_EACH_EP = False

# Guardar y entrenar modelo final al terminar todos los episodios
SAVE_FINAL = False

# Dybamics model para MPPI: puedes usar punto-masa o el GP cuando ya esté entrenado
DYN_MODEL_TYPE = "GP"  # opciones: "MASS" o "GP"

# ============================================================
#                 HELPERS
# ============================================================
#-----------\ import helpers
# from helper import quat_to_rotmat, ForceArrow, enforce_tilt_and_thrust_limits, get_drone_state, set_goal_for_episode


# ============================================================
#                      MAIN
# ============================================================
def main():
    # ============================================================
    # --------- 1) CONFIGURACIÓN MUJOCO --------------------------
    # ============================================================
    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data = mujoco.MjData(model)
    dt = model.opt.timestep

    drone_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "drone_3d")
    if drone_id < 0:
        raise ValueError("No existe el body 'drone_3d' en el XML.")

    mujoco.mj_forward(model, data)
    p0 = data.xpos[drone_id].copy()
    m = float(model.body_subtreemass[drone_id])

    print(f"dt: {dt:.4f}  mass: {m:.3f} kg  p0: {p0}")

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
        np.array([5.0, 5.0, 10.0]),         # Kp posición (diag)
        np.array([3.0, 3.0, 6.0]),          # Kv velocidad (diag)
        np.array([8.0, 8.0, 2.0]),    # KR actitud (diag)
        np.array([0.2, 0.2, 0.1])         # Komega velocidad angular (diag)
    ]
    controller = MellinguerControllerForce(mass=m, gravity=9.81, gains=gains)

    # ============================================================
    # --------- 3) GP dynamics model (learn online) --------------
    # ============================================================
    dyn_gp = SVGPDroneDynamicModel(
        dt=dt,
        mass=m,
        gravity=9.81,
        device="cuda",
        kernel="RBF",
        lr=0.01,
        batch_size=256,
        num_inducing=2**5,  # puedes ajustar según tu GPU, más inducing points = modelo más preciso pero más lento
        init_train_steps=800, # pasos iniciales para llenar el buffer antes de entrenar la primera vez
        train_every=20,          # entrenar cada N steps (ajusta, más frecuencia = modelo más actualizado pero más tiempo de cómputo)
        online_steps=50,         # grad steps online (puedes ajustar, más pasos = modelo más actualizado pero más tiempo de cómputo)
        min_points_to_train=300, # mínimo de puntos en el buffer para empezar a entrenar (ajusta, si es muy bajo puede ser inestable, si es muy alto tarda más en empezar a mejorar)
        reset_each_episode=RESET_GP_EACH_EP, # si True, el modelo se reseteará cada episodio (útil para probar aprendizaje desde cero cada vez, pero si quieres acumular experiencia a lo largo de episodios, déjalo en False
    )

    # cargar modelo si existe
    if os.path.exists(MODEL_PATH):
        print(f"[LOAD] model: {MODEL_PATH}")
        dyn_gp.load_model(MODEL_PATH)

    # cargar rollouts si existe
    if os.path.exists(ROLLOUTS_PATH):
        print(f"[LOAD] rollouts: {ROLLOUTS_PATH}")
        dyn_gp.load_rollouts(ROLLOUTS_PATH, append=True)
        # si cargaste datos y no está entrenado, puedes entrenar full
        if not getattr(dyn_gp, "_trained_once", False):
            print("[TRAIN] full from loaded rollouts...")
            dyn_gp.train_full()

    # ============================================================
    # --------- 4) MPPI Planner ----------------------------------
    # ============================================================
    # límites MPPI (fuerza total en WORLD)
    Fxy = 8.0 * m
    Fz_min = 0.0
    Fz_max = 25.0 * m

    planner = SingleMPPIPlannerTorch(
        dt=dt,
        horizon=40,            # pasos
        num_samples=1024//2,       # rollouts
        lambda_=10.0,            # temperatura
        noise_sigma=np.array([2.0*m, 2.0*m, 4.0*m]),  # ruido en newton step
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
        # altura
        z_min=1.0,
        z_max=1.5,
        z_margin=0.15,
        w_z=200.0,
        w_z_terminal=400.0,
        #velocidad
        v_max=5.0,
        v_margin=0.5,
        w_v=100.0,
        w_v_terminal=200.0,
        rng_seed=0,
        device="cuda")

    # modelo dinámico para MPPI: puedes usar punto-masa o el GP cuando ya esté entrenado
    if DYN_MODEL_TYPE == "GP": # si quieres usar el GP desde el inicio (puede ser inestable al principio, pero es para probar)
         mass_model = SVGPDroneDynamicModel(dt=dt, mass=m, device="cuda")
         mass_model.load_model(MODEL_PATH)  # carga el modelo entrenado (si existe)
    elif DYN_MODEL_TYPE == "MASS": # si quieres usar un modelo de punto-masa simple al inicio
         mass_model = SingleMassDynamicModelTorch(dt=dt, mass=m, device="cuda")
    else:
        raise ValueError(f"DYN_MODEL_TYPE desconocido: {DYN_MODEL_TYPE}")

    planner.define_model(mass_model)

    # obstáculos (ejemplo)
    obstacles = [
        {"type": "sphere", "c": [1.0,  0.0, 1.0], "r": 0.25},
        {"type": "box",   "c": [1.6,  0.4, 1.0], "h": [0.20, 0.20, 0.40]},
        {"type": "box",   "c": [1.6, -0.4, 1.0], "h": [0.20, 0.20, 0.40]},
    ]
    planner.set_obstacles(obstacles)


    def run_episodes(vis=None):
        nonlocal p0

        for ep in range(NUM_EPISODES):
            print(f"\n=== EPISODE {ep+1}/{NUM_EPISODES} ===")

            # reset sim
            reset_episode(model, data)
            # mujoco.mj_resetData(model, data)
            # mujoco.mj_forward(model, data)
            # p0 = data.xpos[drone_id].copy()

            # reset episode (GP buffers, optionally model)
            dyn_gp.reset_episode(clear_buffers=False) # si clear_buffers=True, se borra el buffer de rollouts del GP cada episodio, si False, se mantiene (útil para acumular datos a lo largo de episodios)

            # set goal per episode
            goal = set_goal_for_episode(p0, ep)
            planner.set_goal(goal)

            print(f"goal: {goal}")
            

            # MPPI plan buffer
            plan_idx = MPPI_EVERY
            F_plan = None
            F_fallback = np.array([0.0, 0.0, m * 9.81], dtype=float)

            # safety limits for realizability
            tilt_max = np.deg2rad(35.0)
            u1_max = Fz_max  # thrust total max coherente con F_des (N)

            for step in tqdm.tqdm(range(STEPS_PER_EP)):
                if vis is not None and not vis.is_running():
                    print("Viewer closed.")
                    return

                # (1) read state
                r, v_world, R, omega_body = get_drone_state(model, data, drone_id)

                # (2) MPPI (replan each MPPI_EVERY)
                if plan_idx >= MPPI_EVERY or F_plan is None:
                    F0, _, _, F_seq = planner.compute_action(r, v_world)
                    F_plan = F_seq
                    plan_idx = 0

                if plan_idx < len(F_plan):
                    F_des = F_plan[plan_idx].copy()
                else:
                    F_des = F_fallback.copy()
                plan_idx += 1

                # Z no negativa
                F_des[2] = max(F_des[2], 0.0)

                # cerca del goal => hover + damping para no caer
                if np.linalg.norm(r - goal) < planner.goal_tolerance:
                    F_des = np.array([0.0, 0.0, m * 9.81]) - np.array([2.0, 2.0, 3.0]) * v_world

                # realizable
                F_des = enforce_tilt_and_thrust_limits(F_des, u1_max=u1_max, tilt_max_rad=tilt_max)

                # (optional) draw desired force
                if arrow is not None:
                    arrow.update_force_arrow_mocap(
                        p0_world=r,
                        F_world=F_des,
                        scale=0.03,
                        max_len=0.8,
                        radius=0.01,
                    )

                # (3) low-level controller
                u = controller.step(F_des, R, omega_body, psi_T=YAW_TARGET)

                # (4) apply to actuators
                data.ctrl[act_u1] = float(u[0])
                data.ctrl[act_u2] = float(u[1])
                data.ctrl[act_u3] = float(u[2])
                data.ctrl[act_u4] = float(u[3])

                # ---- collect transition for GP ----
                pos0 = r.copy()
                vel0 = v_world.copy()
                F0 = F_des.copy()

                # (5) sim step
                mujoco.mj_step(model, data)

                # new state
                r2, v2_world, _, _ = get_drone_state(model, data, drone_id)

                # online add + (optional) online train each N steps
                # agrega la transición al buffer del GP, y si train_online=True, entrena cada online_steps pasos (ajusta en la configuración del modelo)
                dyn_gp.add_transition(pos0, vel0, F0, r2, v2_world, train_online=False) # si train_online=True, se entrena cada online_steps pasos (ajusta en la configuración del modelo)

                if vis is not None:
                    vis.sync()

            # end episode: train a bit more + save
            print(f"[EP {ep}] steps collected: {len(dyn_gp.Z_buf)}")

            # # opcional: un entrenamiento final del episodio
            # dyn_gp.train_online(steps=200)

            if SAVE_EACH_EP:
                ep_path = EP_ROLLOUTS_FMT.format(ep)
                dyn_gp.save_rollouts(ep_path)
                print(f"[SAVE] episode rollouts: {ep_path}")

                # append + save global rollouts
                # (guardamos buffer del episodio en un archivo global acumulado)
                if os.path.exists(ROLLOUTS_PATH):
                    # cargar y re-guardar acumulado (simple y robusto)
                    tmp = SVGPDroneDynamicModel(dt=dt, mass=m, device="cuda")
                    tmp.load_rollouts(ROLLOUTS_PATH, append=False)
                    tmp.load_rollouts(ep_path, append=True)
                    tmp.save_rollouts(ROLLOUTS_PATH)
                else:
                    dyn_gp.save_rollouts(ROLLOUTS_PATH)

                dyn_gp.save_model(MODEL_PATH)
                print(f"[SAVE] model: {MODEL_PATH}")
                print(f"[SAVE] rollouts_all: {ROLLOUTS_PATH}")

        # fin de todos los episodios: entrenar full y guardar
        if SAVE_FINAL:
            print("\n[FINAL TRAIN] training full on all collected data...")
            # print size of collected data
            print(f"total collected transitions: {len(dyn_gp.Z_buf)}")
            dyn_gp.train_full()
            dyn_gp.save_model("logs/svgp_dyn.pt")
            dyn_gp.save_rollouts("logs/rollouts_all.npz")

    if RENDER:
        with viewer.launch_passive(model, data) as vis:
            run_episodes(vis=vis)
    else:
        run_episodes(vis=None)


if __name__ == "__main__":
    main()