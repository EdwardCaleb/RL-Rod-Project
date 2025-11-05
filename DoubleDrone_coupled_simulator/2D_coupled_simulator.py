"""
Simulación 2D de dos drones unidos por una barra flexible (PyElastica)
con visualización mejorada (hélices).
Autor: ChatGPT (2025)
Compatible con PyElastica 0.3.3.post2
"""

import numpy as np
import matplotlib.pyplot as plt
from elastica import *
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate
import contextlib, io


# ---------------------------------------------------------------------
# 1. Modelo 2D de dron con gravedad
# ---------------------------------------------------------------------
class Drone2D:
    """Modelo simplificado de dron en 2D (x, z, theta)."""

    def __init__(self, P0, m=1.0, I=0.01, g=9.81, dt=1e-3, arm=0.1):
        self.m = m
        self.I = I
        self.g = g
        self.dt = dt
        self.arm = arm  # distancia entre hélices

        self.P = np.array(P0, dtype=float).reshape(3, 1)   # [x, z, θ]
        self.V = np.zeros((3, 1))  # [vx, vz, ω]
        self.A = np.zeros((3, 1))
        self.F_ext = np.zeros((2, 1))
        self.T_ext = 0.0

    def apply_external_force(self, F):
        self.F_ext += np.array(F, dtype=float).reshape(2, 1)

    def apply_external_torque(self, T):
        self.T_ext += float(T)

    def command_to_U(self, thrust, torque):
        """Control directo: thrust (N) y torque (N·m)."""
        return np.array([[thrust], [torque]])

    def system(self, U):
        """Actualiza la dinámica del dron para un paso de tiempo."""
        T, tau = U.flatten()
        x, z, theta = self.P.flatten()
        vx, vz, omega = self.V.flatten()

        # Aceleraciones
        ax = (T / self.m) * np.sin(theta) + self.F_ext[0, 0] / self.m
        az = (T / self.m) * np.cos(theta) - self.g + self.F_ext[1, 0] / self.m
        alpha = tau / self.I + self.T_ext / self.I

        self.A[:] = [[ax], [az], [alpha]]
        self.V += self.A * self.dt
        self.P += self.V * self.dt

        # Reset de fuerzas externas
        self.F_ext[:] = 0.0
        self.T_ext = 0.0

        return self.P

    def get_rotor_positions(self):
        """Devuelve las posiciones (x,z) de las dos hélices."""
        x, z, theta = self.P.flatten()
        offset = np.array([
            [-self.arm / 2, self.arm / 2],  # dos hélices
            [0.0, 0.0]
        ])
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        rotors = R @ offset + np.array([[x], [z]])
        return rotors  # shape (2, 2)


# ---------------------------------------------------------------------
# 2. Sistema PyElastica 2D
# ---------------------------------------------------------------------
class DroneRodSystem(BaseSystemCollection, Constraints, Forcing, Damping):
    """Contenedor de la barra flexible y los drones."""

    def __init__(self, gravity=np.array([0, -9.81, 0])):
        super().__init__()
        self.gravity = gravity

    def apply_gravity(self):
        for system in self.systems():
            if hasattr(system, "mass") and hasattr(system, "external_forces"):
                gravity_force = np.expand_dims(self.gravity, axis=1) * system.mass
                system.external_forces += gravity_force

    def quiet_integrate(self, stepper, final_time, n_steps=1, progress_bar=False):
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            integrate(stepper, self, final_time, n_steps, progress_bar=progress_bar)

    def step(self, stepper, dt):
        self.apply_gravity()
        self.quiet_integrate(stepper, final_time=dt, n_steps=1)


# ---------------------------------------------------------------------
# 3. Configuración de simulación
# ---------------------------------------------------------------------
sim = DroneRodSystem(gravity=np.array([0, -9.81, 0]))

# Barra flexible (en el plano XZ)
n_elem = 15
start = np.array([0.0, 0.0, 0.0])
direction = np.array([1.0, 0.0, 0.0])
normal = np.array([0.0, 0.0, 1.0])
base_length = 1.0
base_radius = 0.01
density = 500
E = 5e5
nu = 0.5
G = E / (2 * (1 + nu))
dt = 1e-3

rod = CosseratRod.straight_rod(
    n_elements=n_elem,
    start=start,
    direction=direction,
    normal=normal,
    base_length=base_length,
    base_radius=base_radius,
    density=density,
    youngs_modulus=E,
    shear_modulus=G,
)
sim.append(rod)

# Drones (uno a cada extremo)
droneA = Drone2D(P0=[0, 0, 0])
droneB = Drone2D(P0=[1, 0, 0])

# Fijar extremo izquierdo
sim.constrain(rod).using(
    OneEndFixedBC,
    constrained_position_idx=(0,),
    constrained_director_idx=(0,),
)

# Amortiguamiento lineal
sim.dampen(rod).using(
    AnalyticalLinearDamper,
    damping_constant=5e-4,
    time_step=dt,
)

integrator = PositionVerlet()

# ---------------------------------------------------------------------
# 3. Configuración de simulación
# ---------------------------------------------------------------------
sim = DroneRodSystem(gravity=np.array([0, -9.81, 0]))

# Barra flexible (en el plano XZ)
n_elem = 15
start = np.array([0.0, 0.0, 0.0])
direction = np.array([1.0, 0.0, 0.0])
normal = np.array([0.0, 0.0, 1.0])
base_length = 1.0
base_radius = 0.01
density = 500
E = 5e5
nu = 0.5
G = E / (2 * (1 + nu))
dt = 1e-3

rod = CosseratRod.straight_rod(
    n_elements=n_elem,
    start=start,
    direction=direction,
    normal=normal,
    base_length=base_length,
    base_radius=base_radius,
    density=density,
    youngs_modulus=E,
    shear_modulus=G,
)
sim.append(rod)

# Drones (uno a cada extremo)
droneA = Drone2D(P0=[0, 0, 0])
droneB = Drone2D(P0=[1, 0, 0])

# Amortiguamiento lineal
sim.dampen(rod).using(
    AnalyticalLinearDamper,
    damping_constant=5e-4,
    time_step=dt,
)

integrator = PositionVerlet()


# ---------------------------------------------------------------------
# 4. Bucle de simulación (con unión rígida en ambos extremos)
# ---------------------------------------------------------------------
T_final = 3.0
steps = int(T_final / dt)

rod_hist, A_hist, B_hist = [], [], []

for i in range(steps):
    # Vincular ambos extremos del rod a los drones
    rod.position_collection[..., 0] = np.array([droneA.P[0, 0], 0.0, droneA.P[1, 0]])
    rod.position_collection[..., -1] = np.array([droneB.P[0, 0], 0.0, droneB.P[1, 0]])

    # Avanzar PyElastica un paso
    sim.step(integrator, dt)

    # Obtener fuerzas internas en ambos extremos
    F_A = rod.internal_forces[:, 0]
    F_B = -rod.internal_forces[:, -1]

    # Proyectar a 2D (X-Z)
    F_A_2d = F_A[[0, 2]]
    F_B_2d = F_B[[0, 2]]

    # Aplicar fuerzas de reacción a los drones
    droneA.apply_external_force(-F_A_2d)
    droneB.apply_external_force(-F_B_2d)

    # Control: thrust = 0 o m*g según tu prueba
    U_A = droneA.command_to_U(thrust=droneA.m * droneA.g + 0.2, torque=0)
    U_B = droneB.command_to_U(thrust=droneB.m * droneB.g + 0.2, torque=0)

    # Actualizar dinámica de los drones
    droneA.system(U_A)
    droneB.system(U_B)

    # Guardar historia para graficar
    if i % 10 == 0:
        rod_hist.append(rod.position_collection[[0, 2], :].copy())
        A_hist.append(droneA.P.flatten())
        B_hist.append(droneB.P.flatten())

print("✅ Simulación 2D completada (ambos extremos conectados a drones).")


# print(rod_hist)

# ---------------------------------------------------------------------
# 5. Visualización 2D con hélices
# ---------------------------------------------------------------------
plt.ion()
fig, ax = plt.subplots(figsize=(8, 4))
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(-0.8, 0.8)
ax.set_xlabel("X [m]")
ax.set_ylabel("Z [m]")
ax.set_title("Simulación 2D: Drones con hélices y barra flexible")

for k in range(0, len(rod_hist), 2):
    ax.cla()
    ax.plot(rod_hist[k][0, :], rod_hist[k][1, :], "b-", lw=2, label="Barra flexible")

    # Dibuja los drones
    for drone, color, label in zip([droneA, droneB], ["r", "g"], ["Drone A", "Drone B"]):
        pos = [drone.P[0, 0], drone.P[1, 0]]
        rotors = drone.get_rotor_positions()
        ax.plot(rotors[0, :], rotors[1, :], color=color, lw=3)
        ax.scatter(rotors[0, :], rotors[1, :], s=30, c=color)
        ax.scatter(pos[0], pos[1], c=color, s=40, edgecolor="k", label=label)

    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.8, 0.8)
    ax.legend(loc="upper right")
    plt.draw()
    plt.pause(0.05)
    if k % 10 == 0:
        print(f"w{k}")

plt.ioff()
plt.show()
