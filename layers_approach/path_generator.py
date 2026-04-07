
import numpy as np

class PathGenerator:
    def __init__(self):
        self.p = np.zeros(3)  # posición actual del drone
        self.v = np.zeros(3)  # velocidad actual del drone
        self.a = np.zeros(3)  # aceleración actual del drone
        self.t = 0.0          # tiempo actual

    def update_state(self, p, v, a, t):
        self.p = p
        self.v = v
        self.a = a
        self.t = t

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
    
    