"""
Simulación y visualización 3D de dos drones conectados por una barra flexible.
Versión compatible con PyElastica 0.3.3.post2
Autor: ChatGPT (2025)
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from elastica import *
from elastica.timestepper.symplectic_steppers import PositionVerlet
from elastica.timestepper import integrate
import contextlib, io


# ------------------------------------------------------------------
# 1. Clase simple para el dron
# ------------------------------------------------------------------
class SimpleDrone:
    def __init__(self, mass, position, velocity=np.zeros(3)):
        self.m = mass
        self.pos = np.array(position, dtype=float)
        self.vel = np.array(velocity, dtype=float)
        self.force = np.zeros(3)

    def apply_force(self, F):
        self.force += np.array(F, dtype=float)

    def step(self, dt):
        acc = self.force / self.m
        self.vel += acc * dt
        self.pos += self.vel * dt
        self.force[:] = 0.0


# ------------------------------------------------------------------
# 2. Clase PyElastica principal
# ------------------------------------------------------------------
class DroneRodConnection(BaseSystemCollection, Constraints, Forcing, Damping):
    """Sistema que combina drones y barra flexible."""

    def __init__(self):
        super().__init__()

    def quiet_integrate(self, stepper, final_time, n_steps=1, progress_bar=False):
        """Ejecuta la integración sin imprimir mensajes en consola."""
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            integrate(stepper, self, final_time, n_steps, progress_bar=progress_bar)

    def step_simulation(self, stepper, dt):
        """Avanza un paso completo de simulación silenciosamente."""
        self.quiet_integrate(stepper, final_time=dt, n_steps=1)


# ------------------------------------------------------------------
# 3. Configuración de la simulación
# ------------------------------------------------------------------
sim = DroneRodConnection()

# Barra flexible
n_elements = 20
start = np.array([0.0, 0.0, 0.0])
direction = np.array([1.0, 0.0, 0.0])
normal = np.array([0.0, 1.0, 0.0])
base_length = 1.0
base_radius = 0.01
density = 1000
E = 1e6
nu = 0.5
G = E / (2 * (1 + nu))

rod = CosseratRod.straight_rod(
    n_elements=n_elements,
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

# Drones
droneA = SimpleDrone(mass=1.0, position=start)
droneB = SimpleDrone(mass=1.0, position=start + direction * base_length)

# Condición de frontera fija
sim.constrain(rod).using(
    OneEndFixedBC,
    constrained_position_idx=(0,),
    constrained_director_idx=(0,),
)

# Amortiguamiento
sim.dampen(rod).using(
    AnalyticalLinearDamper,
    damping_constant=1e-3,
    time_step=1e-3,
)

integrator = PositionVerlet()

# ------------------------------------------------------------------
# 4. Simulación (solo cálculo, sin gráficos)
# ------------------------------------------------------------------
final_time = 3.0
dt = 1e-3
num_steps = int(final_time / dt)

rod_history = []
droneA_history = []
droneB_history = []

for step in range(num_steps):
    # Actualizar extremo libre (dron B)
    rod.position_collection[..., -1] = droneB.pos

    # Integrar simulación
    sim.step_simulation(integrator, dt)

    # Calcular fuerzas
    F_A = rod.internal_forces[..., 0]
    F_B = -rod.internal_forces[..., -1]

    droneA.apply_force(-F_A)
    droneB.apply_force(-F_B)

    # Pequeño movimiento oscilatorio del dron B
    droneB.apply_force(np.array([0.5 * np.sin(step * dt * 2 * np.pi), 0, 0]))

    # Avanzar drones
    droneA.step(dt)
    droneB.step(dt)

    # Guardar datos
    if step % 10 == 0:
        rod_history.append(rod.position_collection.copy())
        droneA_history.append(droneA.pos.copy())
        droneB_history.append(droneB.pos.copy())

print("✅ Simulación completada. Guardando datos para visualización...")


# ------------------------------------------------------------------
# 5. Clase de visualización 3D
# ------------------------------------------------------------------
class Plot3DSimulation:
    def __init__(self, rod_history, droneA_history, droneB_history):
        self.rod_history = rod_history
        self.droneA_history = droneA_history
        self.droneB_history = droneB_history

    def animate(self, step_skip=1):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 0.5)
        ax.set_zlim(-0.2, 0.2)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title("Drones conectados por barra flexible")

        rod_line, = ax.plot([], [], [], "b-", lw=2)
        droneA_point, = ax.plot([], [], [], "ro", markersize=8)
        droneB_point, = ax.plot([], [], [], "go", markersize=8)

        for i in range(0, len(self.rod_history), step_skip):
            rod_pos = self.rod_history[i]
            droneA_pos = self.droneA_history[i]
            droneB_pos = self.droneB_history[i]

            xs, ys, zs = rod_pos
            rod_line.set_data(xs, ys)
            rod_line.set_3d_properties(zs)

            droneA_point.set_data([droneA_pos[0]], [droneA_pos[1]])
            droneA_point.set_3d_properties([droneA_pos[2]])

            droneB_point.set_data([droneB_pos[0]], [droneB_pos[1]])
            droneB_point.set_3d_properties([droneB_pos[2]])

            plt.pause(0.02)

        plt.show()


# ------------------------------------------------------------------
# 6. Ejecutar visualización
# ------------------------------------------------------------------
viz = Plot3DSimulation(rod_history, droneA_history, droneB_history)
viz.animate(step_skip=5)
