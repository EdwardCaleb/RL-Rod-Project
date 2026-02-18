
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
    def __init__(self, model, data):
        self.model = model
        self.data = data

        self.arrow_body = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "force_arrow")
        if self.arrow_body < 0:
            raise ValueError("No existe el body 'force_arrow' en el XML.")
        self.mocap_id = self.model.body_mocapid[self.arrow_body]
        if self.mocap_id < 0:
            raise ValueError("'force_arrow' no es mocap. ¿Pusiste mocap='true'?")
        self.arrow_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "force_geom")
        if self.arrow_geom < 0:
            raise ValueError("No existe el geom 'force_geom' en el XML.")
        
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
