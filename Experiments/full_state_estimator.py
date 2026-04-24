
import numpy as np

# acc_vel_estimator.py based on position and rotation measurements, using a simple finite difference method
class FullStateEstimator:
    def __init__(self, dt, alpha=0.5):
        self.dt = dt
        self.alpha = alpha  # filtro low-pass

        self.initialized = False

        self.position = np.zeros(3)
        self.prev_position = np.zeros(3)

        self.velocity = np.zeros(3)
        self.prev_velocity = np.zeros(3)

        self.acceleration = np.zeros(3)

        self.q = np.array([1, 0, 0, 0])
        self.R = np.eye(3)




    def update_state_p_q(self, p, q, dt=None):
        if dt is not None:
            self.dt = dt

        # Normalizar quaternion
        q = q / np.linalg.norm(q)

        if not self.initialized:
            self.position = p
            self.prev_position = p
            self.velocity = np.zeros(3)
            self.prev_velocity = np.zeros(3)
            self.acceleration = np.zeros(3)
            self.q = q
            self.R = self.quat_to_rotmat(q)
            self.initialized = True

            return self.position, self.velocity, self.acceleration, self.R

        # Posición
        self.prev_position = self.position
        self.position = p

        # Velocidad (con filtro)
        raw_velocity = (self.position - self.prev_position) / self.dt
        self.velocity = self.alpha * raw_velocity + (1 - self.alpha) * self.velocity

        # Aceleración (con filtro)
        raw_acc = (self.velocity - self.prev_velocity) / self.dt
        self.acceleration = self.alpha * raw_acc + (1 - self.alpha) * self.acceleration

        self.prev_velocity = self.velocity

        # Orientación
        self.q = q
        self.R = self.quat_to_rotmat(q)

        return self.position, self.velocity, self.acceleration, self.R




    # Función auxiliar para convertir quaternion a matriz de rotación
    def quat_to_rotmat(self, q):
        """Quaternion (w,x,y,z) -> matriz de rotación 3x3."""
        w, x, y, z = q
        return np.array([
            [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
            [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
            [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)]
        ], dtype=float)


