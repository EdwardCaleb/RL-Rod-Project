import numpy as np
import tqdm

# ============================================================
# 1) Dinámica: doble integrador 2D (single_mass_dynamics)
#    Estado: x = [px, py, vx, vy]
#    Control: u = [ax, ay]
# ============================================================

class single_mass_dynamics:
    def __init__(self, dt=0.02, u_min=(-3.0, -3.0), u_max=(3.0, 3.0)):
        self.dt = float(dt)
        self.u_min = np.array(u_min, dtype=float)
        self.u_max = np.array(u_max, dtype=float)

    def clamp_u(self, u):
        return np.clip(u, self.u_min, self.u_max)

    def step(self, x, u):
        """
        Un paso Euler.
        x: (4,)  [px, py, vx, vy]
        u: (2,)  [ax, ay]
        """
        u = self.clamp_u(u)
        dt = self.dt

        px, py, vx, vy = x
        ax, ay = u

        # Integración
        px_next = px + vx * dt
        py_next = py + vy * dt
        vx_next = vx + ax * dt
        vy_next = vy + ay * dt

        return np.array([px_next, py_next, vx_next, vy_next], dtype=float)

    def rollout(self, x0, U):
        """
        Propaga una secuencia de controles.
        x0: (4,)
        U:  (T,2)
        Retorna X: (T+1,4)
        """
        T = U.shape[0]
        X = np.zeros((T + 1, 4), dtype=float)
        X[0] = x0
        x = x0.copy()
        for t in range(T):
            x = self.step(x, U[t])
            X[t + 1] = x
        return X


# ============================================================
# 2) MPPI Controller
# ============================================================

class MPPI:
    def __init__(
        self,
        dynamics: single_mass_dynamics,
        horizon=40,
        num_samples=1500,
        lambda_=1.0,
        noise_sigma=(0.6, 0.6),
        u_min=(-3.0, -3.0),
        u_max=(3.0, 3.0),
        seed=0,
    ):
        """
        horizon: T
        num_samples: K
        lambda_: temperatura MPPI
        noise_sigma: std del ruido por dimensión del control
        """
        self.dyn = dynamics
        self.T = int(horizon)
        self.K = int(num_samples)
        self.lmbda = float(lambda_)

        self.u_min = np.array(u_min, dtype=float)
        self.u_max = np.array(u_max, dtype=float)

        self.noise_sigma = np.array(noise_sigma, dtype=float)
        assert self.noise_sigma.shape == (2,)

        self.rng = np.random.default_rng(seed)

        # Secuencia nominal (warm start)
        self.U = np.zeros((self.T, 2), dtype=float)

    def _clamp_U(self, U):
        return np.clip(U, self.u_min, self.u_max)

    @staticmethod
    def _softmin_weights(costs, lambda_):
        """
        Pesos ~ exp(-cost/lambda) con estabilización numérica.
        """
        c_min = np.min(costs)
        w = np.exp(-(costs - c_min) / max(lambda_, 1e-9))
        w_sum = np.sum(w) + 1e-12
        return w / w_sum

    def command(self, x0, cost_fn):
        """
        Calcula el primer control u0 usando MPPI.

        x0: (4,)
        cost_fn: función que recibe (X, U) y retorna costo escalar
                 X: (T+1,4) trayectoria
                 U: (T,2) controles aplicados
        """
        # 1) Sample noise: (K, T, 2)
        eps = self.rng.normal(loc=0.0, scale=1.0, size=(self.K, self.T, 2))
        eps = eps * self.noise_sigma[None, None, :]

        # 2) Controles perturbados
        U_pert = self.U[None, :, :] + eps
        U_pert = np.clip(U_pert, self.u_min, self.u_max)

        # 3) Evaluar costos
        costs = np.zeros((self.K,), dtype=float)
        for k in range(self.K):
            Xk = self.dyn.rollout(x0, U_pert[k])
            costs[k] = cost_fn(Xk, U_pert[k])

        # 4) Pesos y actualización
        w = self._softmin_weights(costs, self.lmbda)

        # MPPI update: U <- sum_k w_k * U_k
        self.U = np.sum(w[:, None, None] * U_pert, axis=0)
        self.U = self._clamp_U(self.U)

        # 5) Output = primer control, y shift del horizonte
        u0 = self.U[0].copy()
        self.U[:-1] = self.U[1:]
        self.U[-1] = 0.0  # o repetir último
        return u0, costs, w


# ============================================================
# 3) Costo: ir a la meta + evitar obstáculos + suavidad
#    Obstáculos circulares: (cx, cy, radius)
# ============================================================

def make_cost_fn(
    goal_xy,
    obstacles,
    w_goal=12.0,
    w_obs=80.0,
    w_ctrl=0.2,
    w_smooth=0.5,
    safe_margin=0.10,
):
    goal_xy = np.array(goal_xy, dtype=float)
    obstacles = [tuple(map(float, o)) for o in obstacles]

    def cost_fn(X, U):
        """
        X: (T+1,4), U: (T,2)
        """
        pos = X[:, 0:2]  # (T+1,2)

        # --- 1) Costo por meta (terminal y/o running)
        goal_err_terminal = np.sum((pos[-1] - goal_xy) ** 2)
        goal_err_running = np.mean(np.sum((pos - goal_xy) ** 2, axis=1))
        J_goal = 0.8 * goal_err_terminal + 0.2 * goal_err_running

        # --- 2) Obstáculos: penalización fuerte si entra en zona insegura
        J_obs = 0.0
        for (cx, cy, r) in obstacles:
            c = np.array([cx, cy], dtype=float)
            d = np.linalg.norm(pos - c[None, :], axis=1)  # distancia al centro
            # distancia "libre" a la superficie (positiva afuera)
            free = d - (r + safe_margin)

            # Penaliza si free < 0 (dentro de margen)
            # Usamos hinge cuadrática suave: max(0, -free)^2
            viol = np.maximum(0.0, -free)
            J_obs += np.sum(viol ** 2)

        # --- 3) Esfuerzo de control
        J_ctrl = np.sum(U ** 2)

        # --- 4) Suavidad (variación entre controles)
        dU = np.diff(U, axis=0)
        J_smooth = np.sum(dU ** 2) if dU.shape[0] > 0 else 0.0

        return (w_goal * J_goal) + (w_obs * J_obs) + (w_ctrl * J_ctrl) + (w_smooth * J_smooth)

    return cost_fn


# ============================================================
# 4) Ejemplo de uso: simulación cerrada con el MISMO modelo
# ============================================================

if __name__ == "__main__":
    dt = 0.03
    dyn = single_mass_dynamics(dt=dt, u_min=(-3, -3), u_max=(3, 3))

    # Meta y obstáculos (círculos)
    goal = (6.0, 4.0)
    obstacles = [
        (2.5, 2.0, 0.7),
        (4.0, 3.0, 0.6),
        (3.2, 1.0, 0.5),
    ]

    cost_fn = make_cost_fn(
        goal_xy=goal,
        obstacles=obstacles,
        w_goal=10.0,
        w_obs=120.0,
        w_ctrl=0.15,
        w_smooth=0.4,
        safe_margin=0.15,
    )

    mppi = MPPI(
        dynamics=dyn,
        horizon=45,
        num_samples=2000,
        lambda_=1.3,
        noise_sigma=(0.7, 0.7),
        u_min=(-3, -3),
        u_max=(3, 3),
        seed=42,
    )

    # Estado inicial: [px, py, vx, vy]
    x = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)

    # Simulación
    N_steps = 250
    traj = [x.copy()]
    controls = []

    for k in tqdm.trange(N_steps):
        u, costs, w = mppi.command(x, cost_fn)
        x = dyn.step(x, u)  # simulador (misma dinámica)
        traj.append(x.copy())
        controls.append(u.copy())

        # criterio simple de llegada
        if np.linalg.norm(x[0:2] - np.array(goal)) < 0.25 and np.linalg.norm(x[2:4]) < 0.4:
            break

    traj = np.array(traj)
    controls = np.array(controls)

    print("Pasos simulados:", len(traj) - 1)
    print("Pos final:", traj[-1, 0:2], "Vel final:", traj[-1, 2:4])
    print("Dist a meta:", np.linalg.norm(traj[-1, 0:2] - np.array(goal)))









import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# Asegúrate de tener:
# traj: (N+1, 4)  columnas [px, py, vx, vy]
# controls: (N, 2) columnas [ax, ay]
# obstacles: lista de (cx, cy, r)
# goal: (gx, gy)

# -------------------------
# Preparar listas (como tu estilo)
# -------------------------
x_list = traj[:, 0]
y_list = traj[:, 1]

# Para el quiver, hacemos que tenga el mismo largo que traj
# (último control se repite o se pone cero)
u_list = np.zeros((len(traj), 2))
u_list[:-1] = controls
u_list[-1] = controls[-1] if len(controls) > 0 else np.zeros(2)

# -------------------------
# Figura
# -------------------------
fig, ax = plt.subplots()

# Ajusta límites automáticamente con margen
xmin = min(np.min(x_list), min([o[0]-o[2] for o in obstacles], default=np.min(x_list))) - 1.0
xmax = max(np.max(x_list), max([o[0]+o[2] for o in obstacles], default=np.max(x_list))) + 1.0
ymin = min(np.min(y_list), min([o[1]-o[2] for o in obstacles], default=np.min(y_list))) - 1.0
ymax = max(np.max(y_list), max([o[1]+o[2] for o in obstacles], default=np.max(y_list))) + 1.0

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect('equal', adjustable='box')
ax.grid(True, alpha=0.3)

# Trayectoria (línea con puntos)
line, = ax.plot([], [], 'o:', lw=2)

# Punto del agente
point, = ax.plot([], [], 'ro', markersize=8)

# Meta
goal_point, = ax.plot([goal[0]], [goal[1]], 'g*', markersize=14)

# Obstáculos como círculos
obs_patches = []
for (cx, cy, r) in obstacles:
    c = Circle((cx, cy), r, fill=True, alpha=0.25)
    ax.add_patch(c)
    obs_patches.append(c)

# Quiver del control (aceleración)
force_scale = 0.2  # ajusta a gusto
q_u = ax.quiver(
    [0.0], [0.0], [0.0], [0.0],
    angles='xy', scale_units='xy',
    scale=1/force_scale,
    color='m', alpha=0.7, width=0.006
)

title = ax.set_title("MPPI: doble integrador 2D (evitando obstáculos)")

# -------------------------
# Init / Update (igual a tu estilo)
# -------------------------
def init():
    line.set_data([], [])
    point.set_data([], [])
    q_u.set_offsets(np.array([[0.0, 0.0]]))
    q_u.set_UVC([0.0], [0.0])
    title.set_text("MPPI: inicializando...")
    return (line, point, q_u, goal_point, title, *obs_patches)

def update(frame):
    # estado actual
    x = x_list[frame]
    y = y_list[frame]

    # trayectoria hasta ahora
    line.set_data(x_list[:frame+1], y_list[:frame+1])
    point.set_data([x], [y])

    # control (aceleración)
    ax_u, ay_u = u_list[frame]
    q_u.set_offsets(np.array([[x, y]]))
    q_u.set_UVC([ax_u], [ay_u])

    title.set_text(f"Frame {frame}/{len(x_list)-1} | pos=({x:.2f},{y:.2f}) | u=({ax_u:.2f},{ay_u:.2f})")
    return (line, point, q_u, goal_point, title, *obs_patches)

ani = FuncAnimation(
    fig, update,
    frames=len(x_list),
    init_func=init,
    blit=False,
    interval=20
)

plt.show()
