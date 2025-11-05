"""
Simulación de dos drones 3DOF conectados por una barra flexible (PyElastica),
con gravedad y visualización 3D simplificada (4 hélices por dron).
Compatible con PyElastica 0.3.3.post2
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
# 1. Modelo de cuadricóptero 6DOF con gravedad
# ------------------------------------------------------------------
class Quad3DOF:
    """Dron con dinámica completa (traslación y rotación) + fuerzas externas."""

    def __init__(self, P0, m, g, Ix, Iy, Iz, dt):
        self.m = m
        self.g = g
        self.Ix = Ix
        self.Iy = Iy
        self.Iz = Iz
        self.dt = dt

        self.P = np.array(P0, dtype=float).reshape(6, 1)  # [x, y, z, φ, θ, ψ]
        self.V = np.zeros((6, 1))
        self.A = np.zeros((6, 1))
        self.external_forces = np.zeros((3, 1))
        self.external_torques = np.zeros((3, 1))

    def system(self, U):
        """Actualiza el estado del dron dadas las entradas U=[T, τx, τy, τz]."""
        m, g = self.m, self.g
        Ix, Iy, Iz = self.Ix, self.Iy, self.Iz
        U1, U2, U3, U4 = U.flatten()
        φ, θ, ψ = self.P[3, 0], self.P[4, 0], self.P[5, 0]
        φ_dot, θ_dot, ψ_dot = self.V[3, 0], self.V[4, 0], self.V[5, 0]

        # --- Aceleraciones lineales ---
        ax = (U1 / m) * (np.cos(ψ) * np.sin(θ) * np.cos(φ) + np.sin(ψ) * np.sin(φ))
        ay = (U1 / m) * (np.sin(ψ) * np.sin(θ) * np.cos(φ) - np.cos(ψ) * np.sin(φ))
        az = (U1 / m) * (np.cos(θ) * np.cos(φ)) - g

        # Añadir fuerzas externas (p. ej. de la barra)
        ax += self.external_forces[0, 0] / m
        ay += self.external_forces[1, 0] / m
        az += self.external_forces[2, 0] / m

        # --- Aceleraciones angulares ---
        αx = ((Iy - Iz) / Ix) * θ_dot * ψ_dot + (U2 + self.external_torques[0, 0]) / Ix
        αy = ((Iz - Ix) / Iy) * φ_dot * ψ_dot + (U3 + self.external_torques[1, 0]) / Iy
        αz = ((Ix - Iy) / Iz) * φ_dot * θ_dot + (U4 + self.external_torques[2, 0]) / Iz

        self.A = np.array([[ax], [ay], [az], [αx], [αy], [αz]])
        self.V += self.A * self.dt
        self.P += self.V * self.dt

        # Resetear fuerzas externas
        self.external_forces[:] = 0.0
        self.external_torques[:] = 0.0

        return self.P

    def apply_external_force(self, F):
        self.external_forces += np.array(F, dtype=float).reshape(3, 1)

    def apply_external_torque(self, T):
        self.external_torques += np.array(T, dtype=float).reshape(3, 1)

    def command_to_U(self, roll_cmd, pitch_cmd, yaw_rate_cmd, thrust_cmd,
                     k_roll=0.01, k_pitch=0.01, k_yaw=0.002):
        """Convierte comandos tipo Crazyflie (roll, pitch, yaw_rate, thrust) a U1..U4."""
        φ, θ, ψ = self.P[3, 0], self.P[4, 0], self.P[5, 0]
        φ_dot, θ_dot, ψ_dot = self.V[3, 0], self.V[4, 0], self.V[5, 0]

        e_roll = roll_cmd - φ
        e_pitch = pitch_cmd - θ
        e_yaw = yaw_rate_cmd - ψ_dot

        U1 = thrust_cmd
        U2 = k_roll * e_roll
        U3 = k_pitch * e_pitch
        U4 = k_yaw * e_yaw

        return np.array([[U1], [U2], [U3], [U4]])


# ------------------------------------------------------------------
# 2. Sistema PyElastica con gravedad
# ------------------------------------------------------------------
class DroneRodConnection(BaseSystemCollection, Constraints, Forcing, Damping):
    """Contenedor de la barra flexible y las conexiones con los drones."""

    def __init__(self, gravity=np.array([0, 0, -9.81])):
        super().__init__()
        self.gravity = gravity

    def apply_gravity(self):
        """Aplica la gravedad a todos los sistemas con masa."""
        for system in self.systems():  # ✅ llamada correcta: systems()
            if hasattr(system, "mass") and hasattr(system, "external_forces"):
                gravity_force = np.expand_dims(self.gravity, axis=1) * system.mass
                system.external_forces += gravity_force

    def quiet_integrate(self, stepper, final_time, n_steps=1, progress_bar=False):
        """Ejecuta un paso de integración sin imprimir mensajes."""
        import io, contextlib
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            integrate(stepper, self, final_time, n_steps, progress_bar=progress_bar)

    def step_simulation(self, stepper, dt):
        """Aplica gravedad y avanza un paso de integración."""
        self.apply_gravity()
        self.quiet_integrate(stepper, final_time=dt, n_steps=1)


# ------------------------------------------------------------------
# 3. Configuración de simulación
# ------------------------------------------------------------------
sim = DroneRodConnection(gravity=np.array([0, 0, -9.81]))

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
dt = 1e-3

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
droneA = Quad3DOF(P0=[0, 0, 0, 0, 0, 0], m=1.0, g=9.81, Ix=0.01, Iy=0.01, Iz=0.02, dt=dt)
droneB = Quad3DOF(P0=[1, 0, 0, 0, 0, 0], m=1.0, g=9.81, Ix=0.01, Iy=0.01, Iz=0.02, dt=dt)

# Fijar extremo izquierdo
sim.constrain(rod).using(
    OneEndFixedBC,
    constrained_position_idx=(0,),
    constrained_director_idx=(0,),
)

# Amortiguamiento
sim.dampen(rod).using(
    AnalyticalLinearDamper,
    damping_constant=1e-3,
    time_step=dt,
)

integrator = PositionVerlet()


# ------------------------------------------------------------------
# 4. Simulación
# ------------------------------------------------------------------
final_time = 3.0
num_steps = int(final_time / dt)

rod_history, droneA_history, droneB_history = [], [], []

for step in range(num_steps):
    rod.position_collection[..., -1] = droneB.P[0:3, 0]
    sim.step_simulation(integrator, dt)

    F_A = rod.internal_forces[..., 0]
    F_B = -rod.internal_forces[..., -1]

    droneA.apply_external_force(-F_A)
    droneB.apply_external_force(-F_B)

    U_A = droneA.command_to_U(roll_cmd=0, pitch_cmd=0, yaw_rate_cmd=0, thrust_cmd=droneA.m * droneA.g)
    U_B = droneB.command_to_U(roll_cmd=0, pitch_cmd=0, yaw_rate_cmd=0, thrust_cmd=droneB.m * droneB.g)

    droneA.system(U_A)
    droneB.system(U_B)

    if step % 10 == 0:
        rod_history.append(rod.position_collection.copy())
        droneA_history.append(droneA.P[:6, 0].copy())
        droneB_history.append(droneB.P[:6, 0].copy())

print("✅ Simulación completada. Guardando datos para visualización...")


# ------------------------------------------------------------------
# 5. Visualización 3D con hélices
# ------------------------------------------------------------------
class Plot3DSimulation:
    def __init__(self, rod_history, droneA_history, droneB_history):
        self.rod_history = rod_history
        self.droneA_history = droneA_history
        self.droneB_history = droneB_history

    # ------------------------------------------------------------
    def draw_drone(self, ax, pos, angles, color="r", arm_length=0.1):
        """Dibuja un cuadricóptero con 4 hélices."""
        φ, θ, ψ = angles
        # Matriz de rotación cuerpo->mundo
        R = np.array([
            [np.cos(ψ)*np.cos(θ),
             np.cos(ψ)*np.sin(θ)*np.sin(φ) - np.sin(ψ)*np.cos(φ),
             np.cos(ψ)*np.sin(θ)*np.cos(φ) + np.sin(ψ)*np.sin(φ)],
            [np.sin(ψ)*np.cos(θ),
             np.sin(ψ)*np.sin(θ)*np.sin(φ) + np.cos(ψ)*np.cos(φ),
             np.sin(ψ)*np.sin(θ)*np.cos(φ) - np.cos(ψ)*np.sin(φ)],
            [-np.sin(θ),
             np.cos(θ)*np.sin(φ),
             np.cos(θ)*np.cos(φ)]
        ])

        # Posiciones de hélices en el marco del cuerpo
        body_points = np.array([
            [arm_length, 0, 0],
            [-arm_length, 0, 0],
            [0, arm_length, 0],
            [0, -arm_length, 0]
        ]).T

        world_points = R @ body_points + pos.reshape(3, 1)

        # Dibuja el cuerpo central y hélices
        ax.scatter(pos[0], pos[1], pos[2], c=color, s=30)
        ax.scatter(world_points[0, :], world_points[1, :], world_points[2, :], c=color, s=10)

    # ------------------------------------------------------------
    def set_axes_equal(self, ax):
        """Hace que los ejes tengan la misma escala (sin distorsión)."""
        limits = np.array([
            ax.get_xlim3d(),
            ax.get_ylim3d(),
            ax.get_zlim3d()
        ])
        spans = abs(limits[:, 1] - limits[:, 0])
        centers = np.mean(limits, axis=1)
        radius = 0.5 * max(spans)
        ax.set_xlim3d([centers[0] - radius, centers[0] + radius])
        ax.set_ylim3d([centers[1] - radius, centers[1] + radius])
        ax.set_zlim3d([centers[2] - radius, centers[2] + radius])

    # ------------------------------------------------------------
    def animate(self, step_skip=1):
        fig = plt.figure(figsize=(8, 6))
        ax = fig.add_subplot(111, projection="3d")

        for i in range(0, len(self.rod_history), step_skip):
            ax.cla()

            # Ejes y título
            ax.set_title("Drones 3DOF conectados por barra flexible con gravedad")
            ax.set_xlabel("X [m]")
            ax.set_ylabel("Y [m]")
            ax.set_zlabel("Z [m]")

            # Extraer posiciones
            rod_pos = self.rod_history[i]
            droneA_pos = self.droneA_history[i][:3]
            droneB_pos = self.droneB_history[i][:3]
            droneA_ang = self.droneA_history[i][3:]
            droneB_ang = self.droneB_history[i][3:]

            # Dibujar barra
            ax.plot(rod_pos[0], rod_pos[1], rod_pos[2], "b-", lw=2)

            # Dibujar drones
            self.draw_drone(ax, droneA_pos, droneA_ang, color="r")
            self.draw_drone(ax, droneB_pos, droneB_ang, color="g")

            # Limites del espacio
            ax.set_xlim(-0.5, 1.5)
            ax.set_ylim(-0.5, 0.5)
            ax.set_zlim(-0.5, 0.5)

            # Igualar ejes (sin distorsión)
            self.set_axes_equal(ax)

            plt.pause(0.02)

        plt.show()


# ------------------------------------------------------------------
# 6. Ejecutar visualización
# ------------------------------------------------------------------
viz = Plot3DSimulation(rod_history, droneA_history, droneB_history)
viz.animate(step_skip=5)
