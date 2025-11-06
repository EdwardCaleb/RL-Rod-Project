import numpy as np

class QuadrotorControllerPID:
    """
    Controlador PID para un dron 2D acoplado a una viga flexible.
    Controla la altura (z) y puede adaptarse a otros ejes si se amplía.
    """

    def __init__(self, kp=5.0, ki=0.5, kd=2.0, u_min=0.0, u_max=0.8):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.u_min = u_min
        self.u_max = u_max

        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

        # objetivo (setpoint)
        self.z_target = 0.8  # altura deseada [m]

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0
        self.prev_time = None

    def update(self, model, data):
        """
        Calcula la acción de control (thrust promedio para las hélices)
        en función del error en z.
        """
        # 🔹 Obtener posición actual del dron (sensor framepos)
        z_actual = data.sensor("pos_drone").data[2]

        # 🔹 Error de altura
        error = self.z_target - z_actual

        # 🔹 Delta tiempo
        t = data.time
        if self.prev_time is None:
            dt = 0.0
        else:
            dt = t - self.prev_time

        # 🔹 PID clásico
        if dt > 0:
            self.integral += error * dt
            derivative = (error - self.prev_error) / dt
        else:
            derivative = 0.0

        u = self.kp * error + self.ki * self.integral + self.kd * derivative

        # 🔹 Saturar salida al rango del actuador
        u = np.clip(u, self.u_min, self.u_max)

        # 🔹 Guardar valores para siguiente paso
        self.prev_error = error
        self.prev_time = t

        # 🔹 Devuelve mismo empuje en ambas hélices (hover)
        return np.array([u, u])
