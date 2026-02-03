# sistema que simula 2 puntos con masa unidos por un resorte
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation



class double_mass_dynamics_2d:
    def __init__(self, mass1, mass2, position1, position2,velocity1=np.zeros(2), velocity2=np.zeros(2)):
        self.mass1 = mass1
        self.mass2 = mass2
        self.r1 = np.array(position1, dtype=float)   # posición vectorial del punto 1
        self.r2 = np.array(position2, dtype=float)   # posición vectorial del punto 2
        self.v1 = np.array(velocity1, dtype=float)                        # velocidad vectorial del punto 1
        self.v2 = np.array(velocity2, dtype=float)                        # velocidad vectorial del punto 2
        self.a1 = np.zeros(2)                        # aceleración vectorial del punto 1
        self.a2 = np.zeros(2)                        # aceleración vectorial del punto 2
        self.spr_force = np.zeros(2)
    
    def non_linear_spring_force(self):
        damping = 0.2  # coeficiente de amortiguamiento
        L0 =1.0  # longitud natural del resorte (m)
        Lm = 1.5  # longitud maxima del resorte (m)
        kc = 0.001  # constante de compresión del resorte
        kt = 50.0   # constante de tracción del resorte
        x1, y1 = self.r1
        x2, y2 = self.r2
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        dL = L - L0
        dv = self.v2 - self.v1

        if L == 0:
            self.spr_force = np.array([0.0, 0.0])
            return np.zeros(2)
        
        if dL < L-Lm:
            Fx = -kc * (dL)**3 * (x2 - x1) / L - damping * dv[0]
            Fy = -kc * (dL)**3 * (y2 - y1) / L - damping * dv[1]
            self.spr_force = np.array([Fx, Fy])
            return self.spr_force
        elif dL >= L-Lm:
            Fx = (-kt * (dL) + -kc * (Lm)**3) * (x2 - x1) / L - damping * dv[0]
            Fy = (-kt * (dL) + -kc * (Lm)**3) * (y2 - y1) / L - damping * dv[1]
            self.spr_force = np.array([Fx, Fy])
            return self.spr_force
        else:
            self.spr_force = np.array([0.0, 0.0])
            return self.spr_force
    
    def step(self, dt, force1, force2):
        # Calcular la fuerza del resorte
        Fspr = self.non_linear_spring_force()
        # Calcular la aceleración
        self.a1 = (force1 - Fspr) / self.mass1
        self.a2 = (force2 + Fspr) / self.mass2
        # calcular la nueva velocidad
        self.v1 = self.v1 + self.a1 * dt
        self.v2 = self.v2 + self.a2 * dt
        # Calcular la nueva posición
        self.r1 = self.r1 + self.v1 * dt
        self.r2 = self.r2 + self.v2 * dt



def external_force_x(t):
    # Fuerza externa en x (por ejemplo, una fuerza oscilatoria)
    F_ext_x = 5.0 * np.sin(2 * np.pi * 0.5 * t) - 0.3 # Amplitud de 5 N, frecuencia de 0.5 Hz
    F_ext_y = 5.0 * np.sin(2 * np.pi * 0.5 * t) - 0.3 # Amplitud de 5 N, frecuencia de 0.5 Hz
    return np.array([F_ext_x, F_ext_y])




# Parámetros del sistema
m1 = 1.0  # masa del punto 1 (kg)
m2 = 1.0  # masa del punto 2 (kg)

# Condiciones iniciales
p1_0 = np.array([0.0, 0.0]) # posición inicial del punto 1
p2_0 = np.array([1.0, 0.0]) # posición inicial del punto 2
v1_0 = np.array([0.0, 0.0]) # velocidad inicial del punto 1
v2_0 = np.array([0.0, 0.0]) # velocidad inicial del punto 2


# Tiempo de simulación
t_start = 0.0  # tiempo inicial (s)
t_end = 20.0  # tiempo final (s)
dt = 0.01  # paso de tiempo (s)


# Inicializar las masas
system = double_mass_dynamics_2d(m1, m2, p1_0, p2_0, v1_0, v2_0)

# Listas para almacenar las posiciones
x1_list = []
y1_list = []
x2_list = []
y2_list = []

# --- Listas nuevas para vectores---
Fext1_list = []
Fext2_list = []
Fspr1_list = []
Fspr2_list = []

# Simulación
t = t_start

while t < t_end:

    # crear fuerzas externas si es necesario
    Fext1 = external_force_x(t)
    Fext2 = -0.55 * Fext1

    # Actualizar las masas
    system.step(dt, Fext1, Fext2)
    
    # Almacenar las posiciones
    x1_list.append(system.r1[0])
    y1_list.append(system.r1[1])
    x2_list.append(system.r2[0])
    y2_list.append(system.r2[1])

    # --- Almacenar fuerzas ---
    Fext1_list.append(Fext1.copy())
    Fext2_list.append(Fext2.copy())
    Fspr1_list.append(-system.spr_force.copy())
    Fspr2_list.append(system.spr_force.copy())
    
    # Avanzar en el tiempo
    t += dt

# Crear la animación en jupyter notebook
fig, ax = plt.subplots()
ax.set_xlim(-3, 5)
ax.set_ylim(-3, 3)
line, = ax.plot([], [], 'o:', lw=2)
# poner un punto rojo solo a la masa 1
point1, = ax.plot([], [], 'ro', markersize=8)
point2, = ax.plot([], [], 'bo', markersize=8)
# quivers
force_scale = 0.05
qext1 = ax.quiver([0.0], [0.0], [0.0], [0.0], angles='xy', scale_units='xy', scale=1/force_scale, color='m', alpha=0.5, width=0.005)
qext2 = ax.quiver([0.0], [0.0], [0.0], [0.0], angles='xy', scale_units='xy', scale=1/force_scale, color='c', alpha=0.5, width=0.005)
qspr1 = ax.quiver([0.0], [0.0], [0.0], [0.0], angles='xy', scale_units='xy', scale=1/force_scale, color='r', alpha=0.5, width=0.005)
qspr2 = ax.quiver([0.0], [0.0], [0.0], [0.0], angles='xy', scale_units='xy', scale=1/force_scale, color='b', alpha=0.5, width=0.005)




def init():
    line.set_data([], [])
    point1.set_data([], [])
    point2.set_data([], [])
    qext1.set_offsets(np.array([[0.0, 0.0]])); qext1.set_UVC([0.0], [0.0])
    qext2.set_offsets(np.array([[0.0, 0.0]])); qext2.set_UVC([0.0], [0.0])
    qspr1.set_offsets(np.array([[0.0, 0.0]])); qspr1.set_UVC([0.0], [0.0])
    qspr2.set_offsets(np.array([[0.0, 0.0]])); qspr2.set_UVC([0.0], [0.0])
    return line, point1, point2, qext1, qext2, qspr1, qspr2


def update(frame):
    x1, y1 = x1_list[frame], y1_list[frame]
    x2, y2 = x2_list[frame], y2_list[frame]

    line.set_data([x1, x2], [y1, y2])
    point1.set_data([x1], [y1])
    point2.set_data([x2], [y2])

    Fx1, Fy1 = Fext1_list[frame]
    Fx2, Fy2 = Fext2_list[frame]
    Fx1_spr, Fy1_spr = Fspr1_list[frame]
    Fx2_spr, Fy2_spr = Fspr2_list[frame]

    qext1.set_offsets(np.array([[x1, y1]])); qext1.set_UVC([Fx1], [Fy1])
    qspr1.set_offsets(np.array([[x1, y1]])); qspr1.set_UVC([Fx1_spr], [Fy1_spr])
    qext2.set_offsets(np.array([[x2, y2]])); qext2.set_UVC([Fx2], [Fy2])
    qspr2.set_offsets(np.array([[x2, y2]])); qspr2.set_UVC([Fx2_spr], [Fy2_spr])

    return line, point1, point2, qext1, qext2, qspr1, qspr2



ani = FuncAnimation(fig, update, frames=len(x1_list), init_func=init, blit=False, interval=10)
plt.show()
