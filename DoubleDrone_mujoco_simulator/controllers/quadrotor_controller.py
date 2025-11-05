import numpy as np

def quat_to_euler(q):
    w, x, y, z = q / np.linalg.norm(q)  # ← Normaliza el cuaternión
    # Limita el argumento del asin para evitar NaN por errores numéricos
    sinp = 2 * (w*y - z*x)
    sinp = np.clip(sinp, -1.0, 1.0)
    roll  = np.arctan2(2*(w*x + y*z), 1 - 2*(x**2 + y**2))
    pitch = np.arcsin(sinp)
    yaw   = np.arctan2(2*(w*z + x*y), 1 - 2*(y**2 + z**2))
    return np.array([roll, pitch, yaw])

def attitude_to_motors(roll, pitch, yaw, thrust):
    """Convierte comandos de actitud en señales de motor (4 rotores)."""
    m0 = thrust - roll + pitch + yaw
    m1 = thrust + roll + pitch - yaw
    m2 = thrust + roll - pitch + yaw
    m3 = thrust - roll - pitch - yaw
    return np.clip([m0, m1, m2, m3], 0.0, 1.0)


class QuadrotorControllerPID:
    """Controlador PID simple de posición y actitud para un dron."""

    def __init__(self):
        # Ganancias PID básicas
        self.kp_z, self.kd_z = 5.0e-0, 1.5e-0
        self.kp_roll, self.kp_pitch, self.kp_yaw = 1.0e-2, 1.0e-2, 0.5e-2
        self.z_target = 1.0  # altura deseada (m)

    def update(self, model, data):
        """Calcula los motores basándose en el estado actual."""

        # Extraer sensores
        pos = data.sensordata[0:3]
        quat = data.sensordata[3:7]
        rpy = quat_to_euler(quat)

        # Error en z
        z_error = self.z_target - pos[2]

        # Control de altura tipo PD
        thrust = 0.7e-6 + self.kp_z*z_error - self.kd_z*data.qvel[2]

        # Control de nivelación
        roll_cmd  = -self.kp_roll * rpy[0]
        pitch_cmd = -self.kp_pitch * rpy[1]
        yaw_cmd   = -self.kp_yaw   * rpy[2]

        return attitude_to_motors(roll_cmd, pitch_cmd, yaw_cmd, thrust)
