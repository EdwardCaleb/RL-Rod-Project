
import numpy as np
import mujoco

# Función auxiliar para convertir quaternion a matriz de rotación
def quat_to_rotmat(q):
    """Quaternion (w,x,y,z) -> matriz de rotación 3x3."""
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
    ], dtype=float)


class ForceArrow:
    def __init__(self, model, data, arrow_idx):
        self.model = model
        self.data = data
        self.arrow_idx = arrow_idx

        self.arrow_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"force_arrow_{arrow_idx}")
        if self.arrow_body < 0:
            raise ValueError(f"No existe el body 'force_arrow_{arrow_idx}' en el XML.")
        self.mocap_id = self.model.body_mocapid[self.arrow_body]
        if self.mocap_id < 0:
            raise ValueError(f"'force_arrow_{arrow_idx}' no es mocap. ¿Pusiste mocap='true'?")
        self.arrow_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"force_geom_{arrow_idx}")
        if self.arrow_geom < 0:
            raise ValueError(f"No existe el geom 'force_geom_{arrow_idx}' en el XML.")
        
    # Función para actualizar la flecha de fuerza en Mujoco
    def update_force_arrow_mocap(self, p0_world, F_world,
                                scale=0.03, max_len=0.8, radius=0.01):
        """
        Actualiza la flecha mocap para visualizar la fuerza F_world desde p0_world.
        - scale: metros por Newton (m/N)
        """
        F = np.asarray(F_world, dtype=float)
        n = np.linalg.norm(F)
        if n < 1e-6:
            return

        dir_ = F / n
        length = min(max_len, scale * n)          # longitud total (m)

        # Coloca el body en el punto medio de la flecha
        p_mid = np.asarray(p0_world, dtype=float) + 0.5 * length * dir_
        self.data.mocap_pos[self.mocap_id] = p_mid

        # Quaternion que alinea +Z con dir_
        quat = np.zeros(4, dtype=float)
        mujoco.mju_quatZ2Vec(quat, dir_)
        self.data.mocap_quat[self.mocap_id] = quat

        # Ajusta tamaño del capsule (radius, half_length)
        self.model.geom_size[self.arrow_geom][0] = float(radius)
        self.model.geom_size[self.arrow_geom][1] = float(length * 0.5)






def enforce_tilt_and_thrust_limits(F, u1_max, tilt_max_rad):
    Fx, Fy, Fz = float(F[0]), float(F[1]), float(F[2])

    # thrust no negativo
    Fz = max(Fz, 1e-3)

    # tilt limit: ||Fxy|| <= Fz * tan(tilt_max)
    Fxy = np.array([Fx, Fy], dtype=float)
    nxy = np.linalg.norm(Fxy)
    max_xy = Fz * np.tan(tilt_max_rad)
    if nxy > max_xy and nxy > 1e-9:
        Fxy *= (max_xy / nxy)

    F2 = np.array([Fxy[0], Fxy[1], Fz], dtype=float)

    # thrust magnitude limit (total)
    nF = np.linalg.norm(F2)
    if nF > u1_max and nF > 1e-9:
        F2 *= (u1_max / nF)

    return F2





def get_drone_state(model, data, drone_id):
    r = data.xpos[drone_id].copy()
    q = data.xquat[drone_id].copy()
    R = quat_to_rotmat(q)

    # cvel = [wx, wy, wz, vx, vy, vz] (world-aligned)
    cvel = data.cvel[drone_id].copy()
    omega_world = cvel[0:3]
    v_world = cvel[3:6]

    omega_body = R.T @ omega_world
    return r, v_world, R, omega_body


def reset_episode(model, data):
    mujoco.mj_resetData(model, data)     # vuelve a qpos0, qvel=0, etc.
    # opcional: randomizar qpos/qvel aquí
    mujoco.mj_forward(model, data)
    # reset actions a 0 (importante para evitar spikes de acción en el primer paso)
    data.ctrl[:] = 0.0


def set_goal_for_episode(p0, height, ep_idx):
    # ejemplo: goals distintos por episodio
    # (cámbialo a random si quieres)
    #altura fija, pero varía en x/y cada episodio
    offset = np.array([2.5, -0.4, 0.0])
    if ep_idx % 2 == 1:
        offset = np.array([2.5, 0.4, 0.0])
    goal = p0 + offset

    goal[2] = height
    # goal =np.array([2.5, 0.0, height])
    return goal