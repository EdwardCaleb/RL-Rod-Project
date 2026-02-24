
import numpy as np

# Mellinguer controller implementation

class MellinguerController:
    def __init__(self, mass, gravity, gains):      
        self.mass = mass
        self.gravity = gravity
        self.gains = gains
        self.Kp = np.array(gains[0], dtype=float)  # Proportional gain for position
        self.Kv = np.array(gains[1], dtype=float)  # Damping gain for velocity
        self.KR = np.array(gains[2], dtype=float)  # Proportional gain for attitude
        self.K_omega = np.array(gains[3], dtype=float)  # Damping gain for angular velocity

        # Permite escalar, vector o matriz 3x3
        self._is_mat = lambda K: (np.ndim(K) == 2 and K.shape == (3, 3))


    # AUXILIARY FUNCTIONS
    def _apply_gain(self, K, x):
        # Permite aplicar una ganancia que puede ser escalar, vector (diagonal) o matriz completa
        K = np.array(K, dtype=float)
        x = np.array(x, dtype=float)
        if np.ndim(K) == 0:
            return K * x
        if np.ndim(K) == 1:
            return K * x               # diagonal como vector
        if np.ndim(K) == 2:
            return K @ x               # matriz completa
        raise ValueError("Ganancia K con forma inválida")
    


    # PIPELINE FUNCTIONS
    def position_control(self, r_T, v_T, a_T, r, v):
        # Control law for position
        e_p = r - r_T
        e_v = v - v_T
        zW = np.array([0.0, 0.0, 1.0])
        F_des = (
            - self._apply_gain(self.Kp, e_p)
            - self._apply_gain(self.Kv, e_v)
            + self.mass * self.gravity * zW
            + self.mass * a_T
        )

        if np.linalg.norm(F_des) < 1e-9: # seguridad numérica, evitar fuerza nula que puede causar problemas en orientación deseada
            F_des = np.array([0.0, 0.0, self.mass * self.gravity])

        return F_des


    def total_thrust(self, F_des, R):
        # Compute total thrust required
        z_B = R[:, 2]  # z-axis of the body frame
        u1 = float(F_des.dot(z_B))
        return u1
    

    def desired_orientation(self, F_des, psi_T, R_current=None):
        # Compute desired orientation based on desired force and target yaw
        z_B_des = F_des / np.linalg.norm(F_des)
        x_C_des = np.array([np.cos(psi_T), np.sin(psi_T), 0])
        y_B_des_cross = np.cross(z_B_des, x_C_des)
        if np.linalg.norm(y_B_des_cross) < 1e-6:
            # Si F_des está alineado con x_C_des, elegimos otro vector para evitar singularidad
            # fallback con Y mundo; si falla, X mundo
            for alt in (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])):
                y_B_des_cross = np.cross(z_B_des, alt)
                if np.linalg.norm(y_B_des_cross) >= 1e-6:
                    break
        
        y_B_des = y_B_des_cross / np.linalg.norm(y_B_des_cross)
        x_B_des = np.cross(y_B_des, z_B_des)
        x_B_des = x_B_des / np.linalg.norm(x_B_des) # seguridad numérica, ortogonalización

        R_des = np.column_stack((x_B_des, y_B_des, z_B_des))

        # Paper: alternativa (-xB, -yB, zB) también válida
        if R_current is not None:
            R_alt = np.column_stack((-x_B_des, -y_B_des, z_B_des))
            score1 = np.trace(R_current.T @ R_des)
            score2 = np.trace(R_current.T @ R_alt)
            if score2 > score1:
                R_des = R_alt

        return R_des
    

    def attitude_control(self, R_des, R_current):
        # Control law for attitude
        R_error = R_des.T @ R_current - R_current.T @ R_des
        e_R = 0.5 * np.array([R_error[2, 1], R_error[0, 2], R_error[1, 0]])
        return e_R
    

    def angular_velocity_error(self, omega_des, omega_current):
        # Control law for angular velocity
        e_omega = omega_current - omega_des
        return e_omega
    

    def moment_control(self, e_R, e_omega):
        # Control law for moments
        u2u3u4 = (
            - self._apply_gain(self.KR, e_R)
            - self._apply_gain(self.K_omega, e_omega)
        )
        return u2u3u4
    

    def propeller_angular_velocities(self, u1, u2u3u4):
        # Convert control inputs to propeller angular velocities (placeholder)
        # Aquí deberías implementar la lógica específica para convertir u1, u2, u3, u4 a velocidades angulares de los motores
        # Esto depende de la configuración de tu drone y cómo se relacionan los momentos con las velocidades de los motores
        omega = np.zeros(4)  # placeholder
        return omega
    

    def motor_mixing(self, u1, u2u3u4):
        # Simple motor mixing (this is a placeholder - actual implementation depends on the drone's motor layout)
        u = np.array([u1, u2u3u4[0], u2u3u4[1], u2u3u4[2]], dtype=float)
        return u
    
    
    # u to F
    def ctrl_to_force(self, u, L, kF, kM):
        # L = 0.1  # distancia del motor al centro de masa
        # kF = 1.0  # constante de fuerza del motor
        # kM = 0.1  # constante de momento del motor
        k = kM / kF  # relación momento-fuerza
        matrix = np.array([
            [1/4,  0        , -1/(2*L) ,  1/(4*k)],
            [1/4,  1/(2*L)  ,  0       , -1/(4*k)],
            [1/4,  0        ,  1/(2*L) ,  1/(4*k)],
            [1/4, -1/(2*L)  ,  0       , -1/(4*k)]])
        
        prop_F = matrix @ u
        return prop_F
    

    def ctrl_to_limited_force(self, u, L, kF, kM, Fmin, Fmax):
        # si u1 es imposible con límites, lo saturas (si no, no hay forma)
        u1 = float(np.clip(u[0], 4*Fmin, 4*Fmax))
        u = u.copy()
        u[0] = u1

        F_ideal = self.ctrl_to_force(u, L=L, kF=kF, kM=kM)  # recalcula F_ideal con u1 saturado

        Fbase = u[0] / 4.0
        d = F_ideal - Fbase  # diferencial; suma ~0

        # alpha máximo para que Fbase + alpha*d esté dentro de [Fmin, Fmax]
        alphas = [1.0]
        for di in d:
            if di > 0:
                alphas.append((Fmax - Fbase) / di)
            elif di < 0:
                alphas.append((Fmin - Fbase) / di)  # di<0 => ratio positivo
        alpha = max(0.0, min(alphas))

        F = Fbase + alpha * d
        F = np.clip(F, Fmin, Fmax)  # seguridad numérica
        return F




    def step(self, r_T, v_T, a_T, r, v, R_current, omega_current, psi_T, omega_des=None):
        F_des = self.position_control(r_T, v_T, a_T, r, v)
        u1 = self.total_thrust(F_des, R_current)
        R_des = self.desired_orientation(F_des, psi_T, R_current)
        e_R = self.attitude_control(R_des, R_current)
        if omega_des is None:
            omega_des = np.zeros(3)  # Asumimos que queremos mantener la orientación deseada sin rotación
        e_omega = self.angular_velocity_error(omega_des, omega_current)
        u2u3u4 = self.moment_control(e_R, e_omega)
        u = self.motor_mixing(u1, u2u3u4)
        return u



# ejemplo de uso
if __name__ == "__main__":
    # Parámetros del quadrotor
    mass = 0.2  # kg
    gravity = 9.81  # m/s^2
    Kp = np.array([1.0, 1.0, 2.0])
    Kv = np.array([1.0, 1.0, 1.5])
    KR = np.array([8.0, 8.0, 2.0])
    K_omega = np.array([0.2, 0.2, 0.1])

    gains = (Kp, Kv, KR, K_omega)


    # Crear una instancia del controlador
    controller = MellinguerController(mass, gravity, gains)
    # Ejemplo de estado y referencia
    r_T = np.array([0.0, 0.0, 1.0])  # posición deseada
    v_T = np.zeros(3)  # velocidad deseada
    a_T = np.zeros(3)  # aceleración deseada
    r = np.array([0.0, 0.0, 0.5])  # posición actual
    v = np.zeros(3)  # velocidad actual
    R_current = np.eye(3)  # orientación actual (identidad)
    omega_current = np.zeros(3)  # velocidad angular actual
    psi_T = 0.0  # yaw deseado
    u = controller.step(r_T, v_T, a_T, r, v, R_current, omega_current, psi_T)
    # motor_commands = controller.ctrl_to_force(u)
    kF = 1.0
    kM = 0.1
    L = 0.1
    motor_commands = controller.ctrl_to_limited_force(u, L=L, kF=kF, kM=kM, Fmin=0.0, Fmax=20.0)
    print("Control inputs (u):", u)
    print("Motor commands:", motor_commands)







#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################
#######################################################





# Mellinguer modified, only force as input

class MellinguerControllerModified:
    def __init__(self, mass, gravity, gains):      
        self.mass = mass
        self.gravity = gravity
        self.gains = gains
        self.Kp = np.array(gains[0], dtype=float)  # Proportional gain for position
        self.Kv = np.array(gains[1], dtype=float)  # Damping gain for velocity
        self.KR = np.array(gains[2], dtype=float)  # Proportional gain for attitude
        self.K_omega = np.array(gains[3], dtype=float)  # Damping gain for angular velocity

        # Permite escalar, vector o matriz 3x3
        self._is_mat = lambda K: (np.ndim(K) == 2 and K.shape == (3, 3))


    # AUXILIARY FUNCTIONS
    def _apply_gain(self, K, x):
        # Permite aplicar una ganancia que puede ser escalar, vector (diagonal) o matriz completa
        K = np.array(K, dtype=float)
        x = np.array(x, dtype=float)
        if np.ndim(K) == 0:
            return K * x
        if np.ndim(K) == 1:
            return K * x               # diagonal como vector
        if np.ndim(K) == 2:
            return K @ x               # matriz completa
        raise ValueError("Ganancia K con forma inválida")
    


    # PIPELINE FUNCTIONS
    def position_control(self, r_T, v_T, a_T, r, v):
        # Control law for position
        e_p = r - r_T
        e_v = v - v_T
        zW = np.array([0.0, 0.0, 1.0])
        F_des = (
            - self._apply_gain(self.Kp, e_p)
            - self._apply_gain(self.Kv, e_v)
            + self.mass * self.gravity * zW
            + self.mass * a_T
        )

        if np.linalg.norm(F_des) < 1e-9: # seguridad numérica, evitar fuerza nula que puede causar problemas en orientación deseada
            F_des = np.array([0.0, 0.0, self.mass * self.gravity])

        return F_des


    def total_thrust(self, F_des, R):
        # Compute total thrust required
        z_B = R[:, 2]  # z-axis of the body frame
        u1 = float(F_des.dot(z_B))
        return u1
    

    def desired_orientation(self, F_des, psi_T, R_current=None):
        # Compute desired orientation based on desired force and target yaw
        z_B_des = F_des / np.linalg.norm(F_des)
        x_C_des = np.array([np.cos(psi_T), np.sin(psi_T), 0])
        y_B_des_cross = np.cross(z_B_des, x_C_des)
        if np.linalg.norm(y_B_des_cross) < 1e-6:
            # Si F_des está alineado con x_C_des, elegimos otro vector para evitar singularidad
            # fallback con Y mundo; si falla, X mundo
            for alt in (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])):
                y_B_des_cross = np.cross(z_B_des, alt)
                if np.linalg.norm(y_B_des_cross) >= 1e-6:
                    break
        
        y_B_des = y_B_des_cross / np.linalg.norm(y_B_des_cross)
        x_B_des = np.cross(y_B_des, z_B_des)
        x_B_des = x_B_des / np.linalg.norm(x_B_des) # seguridad numérica, ortogonalización

        R_des = np.column_stack((x_B_des, y_B_des, z_B_des))

        # Paper: alternativa (-xB, -yB, zB) también válida
        if R_current is not None:
            R_alt = np.column_stack((-x_B_des, -y_B_des, z_B_des))
            score1 = np.trace(R_current.T @ R_des)
            score2 = np.trace(R_current.T @ R_alt)
            if score2 > score1:
                R_des = R_alt

        return R_des
    

    def attitude_control(self, R_des, R_current):
        # Control law for attitude
        R_error = R_des.T @ R_current - R_current.T @ R_des
        e_R = 0.5 * np.array([R_error[2, 1], R_error[0, 2], R_error[1, 0]])
        return e_R
    

    def angular_velocity_error(self, omega_des, omega_current):
        # Control law for angular velocity
        e_omega = omega_current - omega_des
        return e_omega
    

    def moment_control(self, e_R, e_omega):
        # Control law for moments
        u2u3u4 = (
            - self._apply_gain(self.KR, e_R)
            - self._apply_gain(self.K_omega, e_omega)
        )
        return u2u3u4
    

    def propeller_angular_velocities(self, u1, u2u3u4):
        # Convert control inputs to propeller angular velocities (placeholder)
        # Aquí deberías implementar la lógica específica para convertir u1, u2, u3, u4 a velocidades angulares de los motores
        # Esto depende de la configuración de tu drone y cómo se relacionan los momentos con las velocidades de los motores
        omega = np.zeros(4)  # placeholder
        return omega
    

    def motor_mixing(self, u1, u2u3u4):
        # Simple motor mixing (this is a placeholder - actual implementation depends on the drone's motor layout)
        u = np.array([u1, u2u3u4[0], u2u3u4[1], u2u3u4[2]], dtype=float)
        return u
    
    
    # u to F
    def ctrl_to_force(self, u, L, kF, kM):
        # L = 0.1  # distancia del motor al centro de masa
        # kF = 1.0  # constante de fuerza del motor
        # kM = 0.1  # constante de momento del motor
        k = kM / kF  # relación momento-fuerza
        matrix = np.array([
            [1/4,  0        , -1/(2*L) ,  1/(4*k)],
            [1/4,  1/(2*L)  ,  0       , -1/(4*k)],
            [1/4,  0        ,  1/(2*L) ,  1/(4*k)],
            [1/4, -1/(2*L)  ,  0       , -1/(4*k)]])
        
        prop_F = matrix @ u
        return prop_F
    

    def ctrl_to_limited_force(self, u, L, kF, kM, Fmin, Fmax):
        # si u1 es imposible con límites, lo saturas (si no, no hay forma)
        u1 = float(np.clip(u[0], 4*Fmin, 4*Fmax))
        u = u.copy()
        u[0] = u1

        F_ideal = self.ctrl_to_force(u, L=L, kF=kF, kM=kM)  # recalcula F_ideal con u1 saturado

        Fbase = u[0] / 4.0
        d = F_ideal - Fbase  # diferencial; suma ~0

        # alpha máximo para que Fbase + alpha*d esté dentro de [Fmin, Fmax]
        alphas = [1.0]
        for di in d:
            if di > 0:
                alphas.append((Fmax - Fbase) / di)
            elif di < 0:
                alphas.append((Fmin - Fbase) / di)  # di<0 => ratio positivo
        alpha = max(0.0, min(alphas))

        F = Fbase + alpha * d
        F = np.clip(F, Fmin, Fmax)  # seguridad numérica
        return F




    def step(self, F_des, R_current, omega_current, psi_T, omega_des=None):
        u1 = self.total_thrust(F_des, R_current)
        R_des = self.desired_orientation(F_des, psi_T, R_current)
        e_R = self.attitude_control(R_des, R_current)
        if omega_des is None:
            omega_des = np.zeros(3)  # Asumimos que queremos mantener la orientación deseada sin rotación
        e_omega = self.angular_velocity_error(omega_des, omega_current)
        u2u3u4 = self.moment_control(e_R, e_omega)
        u = self.motor_mixing(u1, u2u3u4)
        return u