"""
Controlador de Mellinger (geométrico) para tu modelo MuJoCo "quadrotorplus".

- Implementa: control de posición -> fuerza deseada -> actitud deseada -> momentos -> asignación a 4 motores.
- Asume que cada actuador <motor> aplica:
    * fuerza +Z del site (gear ... 0 0 1 ...)
    * torque de yaw en el site (último componente gear ... ±0.1)

Requisitos:
    pip install mujoco mujoco-python-viewer   (o usa mujoco.viewer si lo tienes)
Uso:
    python mellinger_mujoco.py quadrotorplus.xml
"""

import sys
import time
import numpy as np
import mujoco_viewer

try:
    import mujoco
except ImportError as e:
    raise SystemExit("Necesitas `pip install mujoco`") from e


# -------------------------
# Utilidades SO(3)
# -------------------------
def hat(w: np.ndarray) -> np.ndarray:
    wx, wy, wz = w
    return np.array([[0, -wz, wy],
                     [wz, 0, -wx],
                     [-wy, wx, 0]], dtype=float)

def vee(M: np.ndarray) -> np.ndarray:
    # Inversa de hat para matrices skew
    return np.array([M[2, 1], M[0, 2], M[1, 0]], dtype=float)

def normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n


# -------------------------
# Controlador Mellinger
# -------------------------
class MellingerController:
    def __init__(self, model: "mujoco.MjModel"):
        self.model = model

        # --- IDs útiles ---
        self.core_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "core")
        if self.core_bid < 0:
            raise ValueError("No encuentro body 'core' en el XML.")

        # Sitios de motores (según tu XML)
        self.motor_sites = ["motor0", "motor1", "motor2", "motor3"]
        self.site_ids = []
        for s in self.motor_sites:
            sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, s)
            if sid < 0:
                raise ValueError(f"No encuentro site '{s}' en el XML.")
            self.site_ids.append(sid)

        # Actuadores: asumimos 4 y en el mismo orden que en <actuator>
        if model.nu != 4:
            raise ValueError(f"Esperaba 4 actuadores, pero model.nu={model.nu}.")

        # --- Parámetros físicos aproximados del cuerpo 'core' ---
        # Masa total del modelo (aprox): sum de body masses
        self.m = float(np.sum(model.body_mass))
        # Inercia del body "core" (diagonal en frame del cuerpo)
        Jdiag = model.body_inertia[self.core_bid].copy()
        self.J = np.diag(Jdiag)

        # --- Ganancias (ajústalas) ---
        # Posición
        self.Kp = np.diag([3.0, 3.0, 6.0])
        self.Kd = np.diag([2.5, 2.5, 4.0])
        # Actitud
        self.KR = np.diag([8.0, 8.0, 2.0])
        self.Kw = np.diag([0.25, 0.25, 0.08])

        # Gravedad de MuJoCo (viene en model.opt.gravity, típicamente [0,0,-9.81])
        self.gvec = model.opt.gravity.copy()  # en mundo
        # e3 mundo (up)
        self.e3 = np.array([0.0, 0.0, 1.0])

        # Para evitar singularidad si la fuerza requerida es muy chica
        self.force_eps = 1e-6

    def desired_trajectory(self, t: float):
        """
        Trayectoria deseada.
        Aquí: hover fijo a z=3 (como tu pos inicial), yaw=0.
        Puedes reemplazar por la tuya: x_d(t), v_d(t), a_d(t), yaw_d(t).
        """
        # x_d = np.array([0.0, 0.0, 2.0])
        x_d = np.array([1.0, 1.0, 0.5*np.sin(t)+1])
        v_d = np.array([0.0, 0.0, 0.0])
        a_d = np.array([0.0, 0.0, 0.0])
        yaw_d = 0.0
        yaw_rate_d = 0.0
        return x_d, v_d, a_d, yaw_d, yaw_rate_d

    def build_Rd_from_b3_yaw(self, b3d: np.ndarray, yaw_d: float) -> np.ndarray:
        """
        Construye R_d tal como en Mellinger:
          - b3d fija la dirección del empuje
          - yaw fija heading en el plano horizontal
        """
        b3d = normalize(b3d)
        # Vector "c" en el plano XY para definir yaw deseado
        b1c = np.array([np.cos(yaw_d), np.sin(yaw_d), 0.0])
        # b2d = (b3d x b1c) / ||...||
        b2d = np.cross(b3d, b1c)
        if np.linalg.norm(b2d) < 1e-9:
            # Si b3d casi alineado con b1c, elige otro heading auxiliar
            b1c = np.array([1.0, 0.0, 0.0])
            b2d = np.cross(b3d, b1c)
        b2d = normalize(b2d)
        b1d = np.cross(b2d, b3d)
        Rd = np.column_stack((b1d, b2d, b3d))
        return Rd

    def allocation_matrix(self, data: "mujoco.MjData") -> np.ndarray:
        """
        Matriz A tal que:
            [f; Mx; My; Mz] = A @ u
        donde u = [u0,u1,u2,u3] son controles (0..1) en tu XML.

        - Thrust de cada motor: u_i (porque gear fuerza = [0,0,1] => 1 N por unidad ctrl, aprox)
        - Momento por brazo: r_i x (thrust * e3_body)
        - Momento yaw adicional: km_sign * u_i, con km_sign = ±0.1 (del gear último componente)
        """
        # Pose del core
        R = data.xmat[self.core_bid].reshape(3, 3)
        x_core = data.xpos[self.core_bid].copy()

        # Extraer km_sign desde model.actuator_gear (para cada actuator)
        # gear es longitud 6: [fx,fy,fz, tx,ty,tz] para site transmission
        km = []
        for i in range(self.model.nu):
            gear6 = self.model.actuator_gear[i].copy()  # shape (6,)
            km.append(gear6[5])  # tz en frame del site
        km = np.array(km, dtype=float)  # típicamente ±0.1

        A = np.zeros((4, 4), dtype=float)

        # En body frame: thrust apunta +z_body
        e3b = np.array([0.0, 0.0, 1.0])

        for i, sid in enumerate(self.site_ids):
            # r_w = pos_motor - pos_core (en mundo)
            r_w = data.site_xpos[sid].copy() - x_core
            # r_b = R^T r_w
            r_b = R.T @ r_w

            # Torque por brazo: r_b x (u_i * e3b)
            # => (r_b x e3b) * u_i
            arm_torque = np.cross(r_b, e3b)  # 3x1

            A[0, i] = 1.0                   # contribuye a thrust total
            A[1, i] = arm_torque[0]         # tau_x
            A[2, i] = arm_torque[1]         # tau_y
            A[3, i] = arm_torque[2] + km[i] # tau_z (arm + yaw motor)
            # Nota: arm_torque[2] suele ser ~0 porque r_b está en XY y e3b en Z => tau_z=0.

        return A

    def step(self, data: "mujoco.MjData", t: float) -> np.ndarray:
        """
        Devuelve ctrl (nu,) para MuJoCo.
        """
        # Estado actual del core
        x = data.xpos[self.core_bid].copy()
        R = data.xmat[self.core_bid].reshape(3, 3).copy()

        # Para free joint, qvel:
        #   v (world) = qvel[0:3]
        #   w (body)  = qvel[3:6]
        # El free joint en tu XML se llama "root", y suele ser el primer qpos/qvel del modelo.
        v = data.qvel[0:3].copy()
        w = data.qvel[3:6].copy()

        # Deseado
        x_d, v_d, a_d, yaw_d, yaw_rate_d = self.desired_trajectory(t)

        # Errores
        ex = x - x_d
        ev = v - v_d

        # Aceleración "virtual" (PD)
        a_cmd = a_d - (self.Kp @ ex) - (self.Kd @ ev)

        # Fuerza requerida en mundo:
        # m * xdd = m*a_cmd
        # dinámica: m xdd = m g + f R e3 + Fe  => aquí Fe=0 (si tienes Fe(x,R), réstala aquí)
        # Ojo: model.opt.gravity suele ser [0,0,-9.81], por eso sumamos m*gvec
        # F_req = self.m * a_cmd - self.m * self.gvec  # porque gvec ya apunta hacia abajo


        # Si quieres incluir fuerza externa conocida Fe(x,R) (en mundo):
        # F_req = self.m * a_cmd - self.m * self.gvec - Fe(x,R)
        Fe = self.Fe_world(x, R, v, w, t)   # (3,)
        F_req = self.m * a_cmd - self.m * self.gvec - Fe


        # Construye actitud deseada desde b3d = F_req/||F_req||
        if np.linalg.norm(F_req) < self.force_eps:
            b3d = R @ self.e3  # mantiene dirección actual
        else:
            b3d = normalize(F_req)

        Rd = self.build_Rd_from_b3_yaw(b3d, yaw_d)

        # Thrust escalar (proyección sobre z_body actual)
        # f = F_req · (R e3)
        zb = R @ self.e3
        f = float(F_req.dot(zb))

        # Errores geométricos en SO(3)
        # e_R = 0.5 vee( Rd^T R - R^T Rd )
        eR_mat = 0.5 * (Rd.T @ R - R.T @ Rd)
        e_R = vee(eR_mat)

        # w_d (body) ~ [0,0,yaw_rate_d] en el frame deseado, aproximación común
        w_d = np.array([0.0, 0.0, yaw_rate_d])
        # e_w = w - R^T Rd w_d
        e_w = w - (R.T @ Rd @ w_d)

        # Momento deseado (con compensación giroscópica)
        M = -(self.KR @ e_R) - (self.Kw @ e_w) + np.cross(w, self.J @ w)

        # Asignación a motores: A u = [f, Mx, My, Mz]
        A = self.allocation_matrix(data)
        y = np.array([f, M[0], M[1], M[2]], dtype=float)

        # Resolver con pseudo-inversa y saturar [0,1]
        u = np.linalg.pinv(A) @ y
        u = np.clip(u, 0.0, 1.0)

        return u


    def Fe_world(self, x: np.ndarray, R: np.ndarray, v: np.ndarray, w: np.ndarray, t: float) -> np.ndarray:
        """
        Fuerza externa conocida en el frame MUNDO.

        Implementa aquí tu Fe(x, R). Debe devolver un vector (3,).

        x: posición del core en mundo
        R: rotación (mundo <- cuerpo)
        v: velocidad lineal en mundo
        w: velocidad angular en cuerpo
        t: tiempo

        EJEMPLOS (descomenta uno):
        """

        # --- Ejemplo 0: sin fuerza externa ---
        # return np.zeros(3)

        # --- Ejemplo 1: "muelle" hacia el origen (fuerza depende de posición) ---
        k = 0.5
        return -k * x

        # --- Ejemplo 2: fuerza constante horizontal (como viento constante) ---
        # return np.array([0.2, 0.0, 0.0])

        # --- Ejemplo 3: fuerza dependiente de orientación (ej: empuja en eje x del cuerpo) ---
        # # fuerza en frame cuerpo:
        # Fe_body = np.array([0.3, 0.0, 0.0])
        # # pasar a mundo:
        # return R @ Fe_body

        # --- Ejemplo 4: fuerza tipo drag aerodinámico simple (depende de v) ---
        # c = 0.1
        # return -c * v




# -------------------------
# Simulación
# -------------------------
def run(xml_path: str, realtime: bool = True):
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)

    ctrl = MellingerController(model)

    viewer = mujoco_viewer.MujocoViewer(model, data)
    while viewer.is_alive:
        u = ctrl.step(data, data.time)
        data.ctrl[:] = u
        mujoco.mj_step(model, data)
        viewer.render()


if __name__ == "__main__":
    run("SingleDrone/single_quadrotor.xml")

