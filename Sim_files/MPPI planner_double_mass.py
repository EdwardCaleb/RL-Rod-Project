import numpy as np
import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Circle

# ============================================================
# 1) Dinámica: 2 masas en 2D con resorte + amortiguador
#    Estado:
#      x = [x1, y1, x2, y2, vx1, vy1, vx2, vy2]  (8,)
#    Control (fuerza externa):
#      u = [Fx1, Fy1, Fx2, Fy2]                  (4,)
# ============================================================

class double_mass_dynamics:
    def __init__(
        self,
        dt=0.02,
        m1=1.0, m2=1.0,
        k=0.1,          # constante resorte
        c=0.1,           # amortiguamiento (a lo largo del resorte)
        l0=1.0,          # longitud natural
        u_min=(-4.0, -4.0, -4.0, -4.0),
        u_max=( 4.0,  4.0,  4.0,  4.0),
        eps=1e-9,
    ):
        self.dt = float(dt)
        self.m1 = float(m1)
        self.m2 = float(m2)
        self.k = float(k)
        self.c = float(c)
        self.l0 = float(l0)
        self.u_min = np.array(u_min, dtype=float)
        self.u_max = np.array(u_max, dtype=float)
        self.eps = float(eps)

    def clamp_u(self, u):
        return np.clip(u, self.u_min, self.u_max)

    def spring_forces(self, x):
        """
        Fuerzas del resorte+amortiguador sobre cada masa (solo componente a lo largo del resorte).
        Retorna:
          Fspr1 (2,), Fspr2 (2,) (Fspr2 = -Fspr1)
        """
        x1, y1, x2, y2, vx1, vy1, vx2, vy2 = x
        p1 = np.array([x1, y1], dtype=float)
        p2 = np.array([x2, y2], dtype=float)
        v1 = np.array([vx1, vy1], dtype=float)
        v2 = np.array([vx2, vy2], dtype=float)

        d = p2 - p1
        dist = np.linalg.norm(d)
        if dist < self.eps:
            # si están casi en el mismo lugar, fuerza de resorte indefinida -> 0
            return np.zeros(2), np.zeros(2)

        dir_ = d / dist  # unit vector de 1->2

        # elongación: (dist - l0)
        stretch = dist - self.l0

        # velocidad relativa a lo largo del resorte
        rel_v = v2 - v1
        rel_v_along = np.dot(rel_v, dir_)

        # magnitud de fuerza a lo largo del resorte:
        # resorte: k*stretch
        # amortiguador: c*rel_v_along
        # fuerza sobre masa1 hacia masa2:
        f_mag = self.k * stretch + self.c * rel_v_along

        Fspr1 = f_mag * dir_
        Fspr2 = -Fspr1
        return Fspr1, Fspr2

    def step(self, x, u):
        """
        Euler explícito.
        x: (8,), u: (4,)
        """
        u = self.clamp_u(u)
        dt = self.dt

        x1, y1, x2, y2, vx1, vy1, vx2, vy2 = x
        Fext1 = u[0:2]
        Fext2 = u[2:4]

        Fspr1, Fspr2 = self.spring_forces(x)

        # Aceleraciones
        a1 = (Fext1 + Fspr1) / self.m1
        a2 = (Fext2 + Fspr2) / self.m2

        # Integración
        x1n = x1 + vx1 * dt
        y1n = y1 + vy1 * dt
        x2n = x2 + vx2 * dt
        y2n = y2 + vy2 * dt

        vx1n = vx1 + a1[0] * dt
        vy1n = vy1 + a1[1] * dt
        vx2n = vx2 + a2[0] * dt
        vy2n = vy2 + a2[1] * dt

        return np.array([x1n, y1n, x2n, y2n, vx1n, vy1n, vx2n, vy2n], dtype=float)

    def rollout(self, x0, U):
        """
        x0: (8,), U: (T,4)  -> X: (T+1,8)
        """
        T = U.shape[0]
        X = np.zeros((T + 1, 8), dtype=float)
        X[0] = x0
        x = x0.copy()
        for t in range(T):
            x = self.step(x, U[t])
            X[t + 1] = x
        return X


# ============================================================
# 2) MPPI (generalizado a dim_u=4)
# ============================================================

class MPPI:
    def __init__(
        self,
        dynamics: double_mass_dynamics,
        horizon=50,
        num_samples=2500,
        lambda_=1.2,
        noise_sigma=(1.0, 1.0, 1.0, 1.0),
        u_min=(-4, -4, -4, -4),
        u_max=( 4,  4,  4,  4),
        seed=0,
    ):
        self.dyn = dynamics
        self.T = int(horizon)
        self.K = int(num_samples)
        self.lmbda = float(lambda_)
        self.u_min = np.array(u_min, dtype=float)
        self.u_max = np.array(u_max, dtype=float)

        self.noise_sigma = np.array(noise_sigma, dtype=float)
        assert self.noise_sigma.shape == (4,)

        self.rng = np.random.default_rng(seed)

        # warm-start
        self.U = np.zeros((self.T, 4), dtype=float)

    def _clamp_U(self, U):
        return np.clip(U, self.u_min, self.u_max)

    @staticmethod
    def _softmin_weights(costs, lambda_):
        c_min = np.min(costs)
        w = np.exp(-(costs - c_min) / max(lambda_, 1e-9))
        return w / (np.sum(w) + 1e-12)

    def command(self, x0, cost_fn):
        # ruido (K,T,4)
        eps = self.rng.normal(0.0, 1.0, size=(self.K, self.T, 4)) * self.noise_sigma[None, None, :]

        # controles perturbados
        U_pert = self.U[None, :, :] + eps
        U_pert = np.clip(U_pert, self.u_min, self.u_max)

        # costos
        costs = np.zeros((self.K,), dtype=float)
        for k in range(self.K):
            Xk = self.dyn.rollout(x0, U_pert[k])
            costs[k] = cost_fn(Xk, U_pert[k])

        # pesos
        w = self._softmin_weights(costs, self.lmbda)

        # update
        self.U = np.sum(w[:, None, None] * U_pert, axis=0)
        self.U = self._clamp_U(self.U)

        # salida y shift
        u0 = self.U[0].copy()
        self.U[:-1] = self.U[1:]
        self.U[-1] = 0.0
        return u0, costs, w


# ============================================================
# 3) Costo: meta + evitar obstáculos + regularizaciones
#    - penaliza obstáculos para masa1 y masa2
#    - penaliza estiramiento del resorte
#    - penaliza control y suavidad
# ============================================================

def make_cost_fn_two_masses(
    goal_xy=(6.0, 0.0),              # meta para masa2 por defecto
    obstacles=(),
    safe_margin=0.15,
    w_goal=12.0,
    w_obs=120.0,
    w_ctrl=0.05,
    w_smooth=0.2,
    w_spring_stretch=2.0,
    l0=1.0,
):
    goal = np.array(goal_xy, dtype=float)
    obstacles = [tuple(map(float, o)) for o in obstacles]

    def cost_fn(X, U):
        # posiciones (T+1,2)
        p1 = X[:, 0:2]
        p2 = X[:, 2:4]

        # --- meta (masa2)
        goal_err_terminal = np.sum((p2[-1] - goal) ** 2)
        goal_err_running = np.mean(np.sum((p2 - goal) ** 2, axis=1))
        J_goal = 0.8 * goal_err_terminal + 0.2 * goal_err_running

        # --- obstáculos (masa1 y masa2)
        J_obs = 0.0
        for (cx, cy, r) in obstacles:
            c = np.array([cx, cy], dtype=float)

            d1 = np.linalg.norm(p1 - c[None, :], axis=1)
            d2 = np.linalg.norm(p2 - c[None, :], axis=1)

            free1 = d1 - (r + safe_margin)
            free2 = d2 - (r + safe_margin)

            viol1 = np.maximum(0.0, -free1)
            viol2 = np.maximum(0.0, -free2)

            J_obs += np.sum(viol1**2) + np.sum(viol2**2)

        # --- estiramiento del resorte
        d12 = p2 - p1
        dist = np.linalg.norm(d12, axis=1)
        stretch = dist - l0
        J_spring = np.sum(stretch**2)

        # --- esfuerzo control
        J_ctrl = np.sum(U**2)

        # --- suavidad
        dU = np.diff(U, axis=0)
        J_smooth = np.sum(dU**2) if dU.shape[0] > 0 else 0.0

        return (w_goal * J_goal) + (w_obs * J_obs) + (w_spring_stretch * J_spring) + (w_ctrl * J_ctrl) + (w_smooth * J_smooth)

    return cost_fn


# ============================================================
# 4) Simulación cerrada + logging para animación tipo tu ejemplo
# ============================================================

# --- parámetros del mundo
dt = 0.02
dyn = double_mass_dynamics(
    dt=dt,
    m1=1.0, m2=1.0,
    k=35.0,
    c=3.0,
    l0=1.0,
    u_min=(-6, -6, -6, -6),
    u_max=( 6,  6,  6,  6),
)

goal = (6.0, 0.0)
obstacles = [
    (2.5, 0.2, 0.6),
    (4.0, -0.6, 0.55),
    (3.5, 0.9, 0.5),
]

cost_fn = make_cost_fn_two_masses(
    goal_xy=goal,
    obstacles=obstacles,
    safe_margin=0.18,
    w_goal=14.0,
    w_obs=180.0,
    w_ctrl=0.03,
    w_smooth=0.15,
    w_spring_stretch=1.5,
    l0=dyn.l0,
)

mppi = MPPI(
    dynamics=dyn,
    horizon=55,
    num_samples=300,
    lambda_=1.4,
    noise_sigma=(1.2, 1.2, 1.2, 1.2),
    u_min=dyn.u_min,
    u_max=dyn.u_max,
    seed=42,
)

# Estado inicial:
# masa1 en (0,0), masa2 en (1,0), velocidades 0
x = np.array([0.0, 0.0, 1.0, 0.0,  0.0, 0.0, 0.0, 0.0], dtype=float)

N_steps = 260

# logs para animación (como tu snippet)
x1_list, y1_list, x2_list, y2_list = [], [], [], []
Fext1_list, Fext2_list = [], []
Fspr1_list, Fspr2_list = [], []

# log inicial
x1_list.append(x[0]); y1_list.append(x[1])
x2_list.append(x[2]); y2_list.append(x[3])
Fspr1, Fspr2 = dyn.spring_forces(x)
Fext1_list.append(np.array([0.0, 0.0])); Fext2_list.append(np.array([0.0, 0.0]))
Fspr1_list.append(Fspr1); Fspr2_list.append(Fspr2)

for k in tqdm.trange(N_steps):
    u, costs, w = mppi.command(x, cost_fn)
    u = dyn.clamp_u(u)

    # fuerzas para log (antes del step)
    Fext1 = u[0:2].copy()
    Fext2 = u[2:4].copy()
    Fspr1, Fspr2 = dyn.spring_forces(x)

    # sim
    x = dyn.step(x, u)

    # log
    x1_list.append(x[0]); y1_list.append(x[1])
    x2_list.append(x[2]); y2_list.append(x[3])
    Fext1_list.append(Fext1); Fext2_list.append(Fext2)
    Fspr1_list.append(Fspr1); Fspr2_list.append(Fspr2)

    # criterio de llegada (masa2)
    if np.linalg.norm(x[2:4] - np.array(goal)) < 0.25 and np.linalg.norm(x[6:8]) < 0.4:
        break

print("Pasos simulados:", len(x1_list) - 1)
print("Pos final masa2:", (x2_list[-1], y2_list[-1]), "dist a meta:", np.linalg.norm(np.array([x2_list[-1], y2_list[-1]]) - np.array(goal)))


# ============================================================
# 5) Animación en estilo "tu snippet"
# ============================================================

fig, ax = plt.subplots()

# límites automáticos con margen
all_x = np.array(x1_list + x2_list)
all_y = np.array(y1_list + y2_list)
xmin = min(all_x.min(), min([o[0]-o[2] for o in obstacles], default=all_x.min())) - 1.0
xmax = max(all_x.max(), max([o[0]+o[2] for o in obstacles], default=all_x.max())) + 1.0
ymin = min(all_y.min(), min([o[1]-o[2] for o in obstacles], default=all_y.min())) - 1.0
ymax = max(all_y.max(), max([o[1]+o[2] for o in obstacles], default=all_y.max())) + 1.0

ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.set_aspect('equal', adjustable='box')
ax.grid(True, alpha=0.3)

# obstáculos
for (cx, cy, r) in obstacles:
    ax.add_patch(Circle((cx, cy), r, fill=True, alpha=0.25))

# meta
ax.plot([goal[0]], [goal[1]], 'g*', markersize=14)

# línea entre masas (con puntos)
line, = ax.plot([], [], 'o:', lw=2)

# puntos de masas
point1, = ax.plot([], [], 'ro', markersize=8)
point2, = ax.plot([], [], 'bo', markersize=8)

# quivers (como tu ejemplo)
force_scale = 0.05
qext1 = ax.quiver([0.0], [0.0], [0.0], [0.0],
                  angles='xy', scale_units='xy', scale=1/force_scale,
                  color='m', alpha=0.5, width=0.005)
qext2 = ax.quiver([0.0], [0.0], [0.0], [0.0],
                  angles='xy', scale_units='xy', scale=1/force_scale,
                  color='c', alpha=0.5, width=0.005)
qspr1 = ax.quiver([0.0], [0.0], [0.0], [0.0],
                  angles='xy', scale_units='xy', scale=1/force_scale,
                  color='r', alpha=0.5, width=0.005)
qspr2 = ax.quiver([0.0], [0.0], [0.0], [0.0],
                  angles='xy', scale_units='xy', scale=1/force_scale,
                  color='b', alpha=0.5, width=0.005)

title = ax.set_title("MPPI: 2 masas + resorte (evitando obstáculos)")

def init():
    line.set_data([], [])
    point1.set_data([], [])
    point2.set_data([], [])

    qext1.set_offsets(np.array([[0.0, 0.0]])); qext1.set_UVC([0.0], [0.0])
    qext2.set_offsets(np.array([[0.0, 0.0]])); qext2.set_UVC([0.0], [0.0])
    qspr1.set_offsets(np.array([[0.0, 0.0]])); qspr1.set_UVC([0.0], [0.0])
    qspr2.set_offsets(np.array([[0.0, 0.0]])); qspr2.set_UVC([0.0], [0.0])

    title.set_text("MPPI: inicializando...")
    return line, point1, point2, qext1, qext2, qspr1, qspr2, title

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

    title.set_text(f"Frame {frame}/{len(x1_list)-1} | m2=({x2:.2f},{y2:.2f})")

    return line, point1, point2, qext1, qext2, qspr1, qspr2, title

ani = FuncAnimation(fig, update, frames=len(x1_list), init_func=init, blit=False, interval=15)
plt.show()
