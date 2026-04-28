
import numpy as np

class PathGenerator:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.dp = np.zeros(3)  # posición actual del drone
        self.dv = np.zeros(3)  # velocidad actual del drone
        self.da = np.zeros(3)  # aceleración actual del drone
        self.t = 0.0          # tiempo actual
        self.prev_dp = np.zeros(3)  # posición anterior (para cálculo de velocidad numérica)
        self.prev_dv = np.zeros(3)  # velocidad anterior (para cálculo de aceleración numérica)
        self.prev_da = np.zeros(3)  # aceleración anterior (para cálculo de jerk numérico)

    def update_des_state(self, dp, dv, da, t):
        self.dp = dp
        self.dv = dv
        self.da = da
        self.t = t

    def update_prev_des_state(self):
        self.prev_dp = self.dp
        self.prev_dv = self.dv
        self.prev_da = self.da

    def fixed_point(self, point):
        # Genera una trayectoria que simplemente mantiene el drone en un punto fijo
        dp = np.array(point)
        dv = np.zeros(3)
        da = np.zeros(3)
        return dp, dv, da
    

    def do_circle_xz(self, t, center, radius, omega):
        # Genera una trayectoria circular en el plano XZ alrededor de un centro dado en
        x_c, y_c, z_c = center
        dpx = x_c + radius * np.cos(omega * t)
        dpy = y_c  # Mantiene la coordenada Y constante
        dpz = z_c + radius * np.sin(omega * t)
        dp = np.array([dpx, dpy, dpz])
        # hallamos velocidad derivando la posición
        dvx = -radius * omega * np.sin(omega * t)
        dvy = 0
        dvz = radius * omega * np.cos(omega * t)
        dv = np.array([dvx, dvy, dvz])
        # hallamos aceleración derivando la velocidad
        dax = -radius * omega**2 * np.cos(omega * t)
        day = 0
        daz = -radius * omega**2 * np.sin(omega * t)
        da = np.array([dax, day, daz])
        return dp, dv, da

    def do_circle_xy(self, t, center, radius, omega):
        # Genera una trayectoria circular en el plano XY alrededor de un centro dado
        x_c, y_c, z_c = center
        dpx = x_c + radius * np.cos(omega * t)
        dpy = y_c + radius * np.sin(omega * t)
        dpz = z_c  # Mantiene la coordenada Z constante
        dp = np.array([dpx, dpy, dpz])
        # hallamos velocidad derivando la posición
        dvx = -radius * omega * np.sin(omega * t)
        dvy = radius * omega * np.cos(omega * t)
        dvz = 0
        dv = np.array([dvx, dvy, dvz])
        # hallamos aceleración derivando la velocidad
        dax = -radius * omega**2 * np.cos(omega * t)
        day = -radius * omega**2 * np.sin(omega * t)
        daz = 0
        da = np.array([dax, day, daz])
        return dp, dv, da
    
    def do_elipse_xy(self, t, center, radius_x, radius_y, omega):
        # Genera una trayectoria elíptica en el plano XY alrededor de un centro dado
        x_c, y_c, z_c = center
        dpx = x_c + radius_x * np.cos(omega * t)
        dpy = y_c + radius_y * np.sin(omega * t)
        dpz = z_c  # Mantiene la coordenada Z constante
        dp = np.array([dpx, dpy, dpz])
        # hallamos velocidad derivando la posición
        dvx = -radius_x * omega * np.sin(omega * t)
        dvy = radius_y * omega * np.cos(omega * t)
        dvz = 0
        dv = np.array([dvx, dvy, dvz])
        # hallamos aceleración derivando la velocidad
        dax = -radius_x * omega**2 * np.cos(omega * t)
        day = -radius_y * omega**2 * np.sin(omega * t)
        daz = 0
        da = np.array([dax, day, daz])
        return dp, dv, da
    
    # hacer vaiven en en x
    def do_sine_x(self, t, center, amplitude, frequency):
        x_c, y_c, z_c = center
        dpx = x_c + amplitude * np.sin(2 * np.pi * frequency * t)
        dpy = y_c
        dpz = z_c
        dp = np.array([dpx, dpy, dpz])
        # hallamos velocidad derivando la posición
        dvx = amplitude * 2 * np.pi * frequency * np.cos(2 * np.pi * frequency * t)
        dvy = 0
        dvz = 0
        dv = np.array([dvx, dvy, dvz])
        # hallamos aceleración derivando la velocidad
        dax = -amplitude * (2 * np.pi * frequency)**2 * np.sin(2 * np.pi * frequency * t)
        day = 0
        daz = 0
        da = np.array([dax, day, daz])
        return dp, dv, da
    
    def do_chirp_x(self, t, center, amplitude, f0, f1, T): # use do_sine_x fcn with frequency = f0 + (f1 - f0) * (t / T)
        frequency = f0 + (f1 - f0) * (t / T)
        return self.do_sine_x(t, center, amplitude, frequency)

    def do_square_xz(self, t, center, side_length, omega):
        # Genera una trayectoria cuadrada en el plano XZ alrededor de un centro dado
        x_c, y_c, z_c = center
        period = 4 * side_length / (omega * side_length)  # Tiempo para completar un ciclo completo
        t_mod = t % period
        
        if t_mod < period / 4:
            dpx = x_c + side_length * (t_mod / (period / 4))
            dpz = z_c
        elif t_mod < period / 2:
            dpx = x_c + side_length
            dpz = z_c + side_length * ((t_mod - period / 4) / (period / 4))
        elif t_mod < 3 * period / 4:
            dpx = x_c + side_length * (1 - (t_mod - period / 2) / (period / 4))
            dpz = z_c + side_length
        else:
            dpx = x_c
            dpz = z_c + side_length * (1 - (t_mod - 3 * period / 4) / (period / 4))
        
        dpy = y_c
        dp = np.array([dpx, dpy, dpz])
        
        # Para la velocidad y aceleración, podríamos usar aproximaciones numéricas o derivar analíticamente cada segmento.
        # Aquí usaremos aproximaciones numéricas simples para mantenerlo sencillo.
        
        return dp, np.zeros(3), np.zeros(3)
    
    
    def do_square_signal_x(self, t, center, amplitude, frequency):
        # Genera una trayectoria de señal cuadrada en el eje X alrededor de un centro dado
        x_c, y_c, z_c = center
        dpx = x_c + amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
        dpy = y_c
        dpz = z_c
        dp = np.array([dpx, dpy, dpz])
        
        # Para la velocidad y aceleración, podríamos usar aproximaciones numéricas o derivar analíticamente cada segmento.
        # Aquí usaremos aproximaciones numéricas simples para mantenerlo sencillo.
        
        return dp, np.zeros(3), np.zeros(3)
    

    # def do_spherical_spiral(self, t, center, radius, omega, vertical_speed):
    #     # Genera una trayectoria de espiral esférica alrededor de un centro dado
    #     x_c, y_c, z_c = center
    #     dpx = x_c + radius * np.cos(omega * t) * np.cos(vertical_speed * t)
    #     dpy = y_c + radius * np.cos(omega * t) * np.sin(vertical_speed * t)
    #     dpz = z_c + radius * np.sin(omega * t)
    #     dp = np.array([dpx, dpy, dpz])
        
    #     # Para la velocidad y aceleración, podríamos usar aproximaciones numéricas o derivar analíticamente cada segmento.
    #     # Aquí usaremos aproximaciones numéricas simples para mantenerlo sencillo.
        
    #     return dp, np.zeros(3), np.zeros(3)


    def do_spherical_spiral(self, t, center, radius, omega, vertical_speed):
        # Genera una trayectoria de espiral esférica alrededor de un centro dado
        x_c, y_c, z_c = center
        dpx = x_c + radius * np.cos(omega * t) * np.cos(vertical_speed * t)
        dpy = y_c + radius * np.sin(omega * t) * np.cos(vertical_speed * t)
        dpz = z_c + radius * np.sin(vertical_speed * t)
        dp = np.array([dpx, dpy, dpz])
        
        # Para la velocidad y aceleración, podríamos usar aproximaciones numéricas o derivar analíticamente cada segmento.
        # Aquí usaremos aproximaciones numéricas simples para mantenerlo sencillo.
        
        return dp, np.zeros(3), np.zeros(3)


    def do_fill_spherical_spiral(self, t, center, radius, omega, vertical_speed):

        self.update_prev_des_state()

        # Parámetro normalizado (0 → 1 → 0 para ir y volver)
        u = 0.5 * (1 + np.sin(0.25 * omega * t))
        
        # Distribución correcta en volumen
        var_r = radius * (u ** (1/3))

        x_c, y_c, z_c = center
        dpx = x_c + var_r * np.cos(omega * t) * np.cos(vertical_speed * t)
        dpy = y_c + var_r * np.sin(omega * t) * np.cos(vertical_speed * t)
        dpz = z_c + var_r * np.sin(vertical_speed * t)
        dp = np.array([dpx, dpy, dpz])

        return dp, np.zeros(3), np.zeros(3)
    


    def get_vel_acc_from_pos(self, dp, t):
        # Aproximación numérica de la velocidad a partir de la posición
        self.dp = dp

        self.update_des_state(dp, np.zeros(3), np.zeros(3), t)

        dvx = (self.dp[0] - self.prev_dp[0]) / self.dt
        dvy = (self.dp[1] - self.prev_dp[1]) / self.dt
        dvz = (self.dp[2] - self.prev_dp[2]) / self.dt
        dv = np.array([dvx, dvy, dvz])

        self.dv = dv

        dax = (dv[0] - self.prev_dv[0]) / self.dt
        day = (dv[1] - self.prev_dv[1]) / self.dt
        daz = (dv[2] - self.prev_dv[2]) / self.dt
        da = np.array([dax, day, daz])

        self.da = da

        return dv, da


        
    # def do_multistep_z(self, t, center, step_height, n_steps, step_duration):
    #     # Genera una trayectoria que sube en escalones en el eje Z alrededor de un centro dado
    #     x_c, y_c, z_c = center
    #     total_duration = n_steps * step_duration
    #     t_mod = t % total_duration
    #     current_step = int(t_mod // step_duration)
    #     dpx = x_c
    #     dpy = y_c
    #     dpz = z_c + step_height * current_step
    #     dp = np.array([dpx, dpy, dpz])
    #     return dp, np.zeros(3), np.zeros(3)
    

    def do_linear_z(self, t, center, z_speed):
        # Genera una trayectoria que sube linealmente en el eje Z alrededor de un centro dado
        x_c, y_c, z_c = center
        dpx = x_c
        dpy = y_c
        dpz = z_c + z_speed * t
        dp = np.array([dpx, dpy, dpz])
        return dp, np.zeros(3), np.zeros(3)
    


    