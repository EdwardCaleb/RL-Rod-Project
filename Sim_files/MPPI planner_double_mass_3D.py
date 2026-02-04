"""
MPPI para sistema doble masa 3D con resorte NO lineal + gravedad.
- Usa TU clase double_mass_dynamics_3d como simulador y como modelo interno (rollouts).
- MPPI controla fuerzas externas en ambas masas: u = [Fx1,Fy1,Fz1,Fx2,Fy2,Fz2]
- Compensa gravedad con feedforward (-m*g) (opcional, recomendado).
- Costo: meta (masa2), obstáculos esféricos, suelo (no caer), resorte, control y suavidad.
- Animación 3D estable (sin ax.collections.clear), con línea, puntos y quivers.
"""

import numpy as np
import tqdm
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


# ============================================================
# 1) TU dinámica (tal cual la pegaste)
# ============================================================

class double_mass_dynamics_3d:
    def __init__(self, mass1, mass2, position1, position2, velocity1=np.zeros(3), velocity2=np.zeros(3)):
        self.mass1 = mass1
        self.mass2 = mass2
        self.r1 = np.array(position1, dtype=float)   # posición vectorial del punto 1
        self.r2 = np.array(position2, dtype=float)   # posición vectorial del punto 2
        self.v1 = np.array(velocity1, dtype=float)   # velocidad vectorial del punto 1
        self.v2 = np.array(velocity2, dtype=float)   # velocidad vectorial del punto 2
        self.a1 = np.zeros(3)                        # aceleración vectorial del punto 1
        self.a2 = np.zeros(3)                        # aceleración vectorial del punto 2
        self.spr_force = np.zeros(3)
        self.g = np.array([0.0, 0.0, -9.81])         # gravedad (m/s^2)

    def non_linear_spring_force(self):
        damping = 0.2
        L0 = 1.0
        Lm = 1.5
        kc = 0.001
        kt = 50.0

        x1, y1, z1 = self.r1
        x2, y2, z2 = self.r2
        L = np.sqrt((x2 - x1)**2 + (y2 - y1)**2 + (z2 - z1)**2)
        dL = L - L0
        dv = self.v2 - self.v1

        if L == 0:
            self.spr_force = np.array([0.0, 0.0, 0.0])
            return np.zeros(3)

        # Mantengo exactamente tu lógica para no cambiar el modelo.
        if dL < L - Lm:
            Fx = -kc * (dL)**3 * (x2 - x1) / L - damping * dv[0]
            Fy = -kc * (dL)**3 * (y2 - y1) / L - damping * dv[1]
            Fz = -kc * (dL)**3 * (z2 - z1) / L - damping * dv[2]
            self.spr_force = np.array([Fx, Fy, Fz])
            return self.spr_force
        elif dL >= L - Lm:
            Fx = (-kt * (dL) + -kc * (Lm)**3) * (x2 - x1) / L - damping * dv[0]
            Fy = (-kt * (dL) + -kc * (Lm)**3) * (y2 - y1) / L - damping * dv[1]
            Fz = (-kt * (dL) + -kc * (Lm)**3) * (z2 - z1) / L - damping * dv[2]
            self.spr_force = np.array([Fx, Fy, Fz])
            return self.spr_force
        else:
            self.spr_force = np.array([0.0, 0.0, 0.0])
            return self.spr_force

    def step(self, dt, force1, force2, en_gravity=True):
        Fspr = self.non_linear_spring_force()
        self.a1 = (force1 - Fspr + en_gravity * self.g * self.mass1) / self.mass1
        self.a2 = (force2 + Fspr + en_gravity * self.g * self.mass2) / self.mass2
        self.v1 = self.v1 + self.a1 * dt
        self.v2 = self.v2 + self.a2 * dt
        self.r1 = self.r1 + self.v1 * dt
        self.r2 = self.r2 + self.v2 * dt


# ============================================================
# 2) Wrapper MPPI-friendly (estado vector + rollout)
#    Estado: [r1(3), r2(3), v1(3), v2(3)] => (12,)
#    Control: [F1(3), F2(3)] => (6,)
# ============================================================

class double_mass_dynamics_3d_wrapper:
    def __init__(self, dyn: double_mass_dynamics_3d, dt=0.02, u_min=None, u_max=None, en_gravity=True):
        self.dyn = dyn
        self.dt = float(dt)
        self.en_gravity = bool(en_gravity)

        if u_min is None:
            u_min = [-30, -30, -60,  -30, -30, -60]
        if u_max is None:
            u_max = [ 30,  30,  60,   30,  30,  60]

        self.u_min = np.array(u_min, dtype=float).reshape(6,)
        self.u_max = np.array(u_max, dtype=float).reshape(6,)

    def clamp_u(self, u):
        return np.clip(u, self.u_min, self.u_max)

    def get_state(self):
        return np.concatenate([self.dyn.r1, self.dyn.r2, self.dyn.v1, self.dyn.v2]).astype(float)

    def set_state(self, x):
        x = np.array(x, dtype=float).reshape(12,)
        self.dyn.r1 = x[0:3].copy()
        self.dyn.r2 = x[3:6].copy()
        self.dyn.v1 = x[6:9].copy()
        self.dyn.v2 = x[9:12].copy()

    def spring_force_from_state(self, x):
        # clon ligero
        d = self.dyn
        tmp = double_mass_dynamics_3d(
            mass1=d.mass1, mass2=d.mass2,
            position1=x[0:3], position2=x[3:6],
            velocity1=x[6:9], velocity2=x[9:12],
        )
        tmp.g = d.g.copy()
        return tmp.non_linear_spring_force()

    def step_from_state(self, x, u):
        u = self.clamp_u(u)
        F1 = u[0:3]
        F2 = u[3:6]

        d = self.dyn
        tmp = double_mass_dynamics_3d(
            mass1=d.mass1, mass2=d.mass2,
            position1=x[0:3], position2=x[3:6],
            velocity1=x[6:9], velocity2=x[9:12],
        )
        tmp.g = d.g.copy()
        tmp.step(self.dt, F1, F2, en_gravity=self.en_gravity)

        return np.concatenate([tmp.r1, tmp.r2, tmp.v1, tmp.v2]).astype(float)

    def rollout(self, x0, U):
        T = U.shape[0]
        X = np.zeros((T + 1, 12), dtype=float)
        X[0] = x0
        x = x0.copy()
        for t in range(T):
            x = self.step_from_state(x, U[t])
            X[t + 1] = x
        return X


# ============================================================
# 3) MPPI (dim_u=6) con feedforward de gravedad
# ============================================================

class MPPI:
    def __init__(
        self,
        dynamics: double_mass_dynamics_3d_wrapper,
        horizon=65,
        num_samples=3000,
        lambda_=1.7,
        noise_sigma=None,
        seed=42,
        use_gravity_ff=True,
    ):
        self.dyn = dynamics
        self.T = int(horizon)
        self.K = int(num_samples)
        self.lmbda = float(lambda_)
        self.rng = np.random.default_rng(seed)

        self.dim_u = 6
        if noise_sigma is None:
            # Menos ruido en Z ayuda a no "tironear" altura
            noise_sigma = [4, 4, 2,  4, 4, 2]
        self.noise_sigma = np.array(noise_sigma, dtype=float).reshape(6,)

        # Secuencia nominal sin FF (guardamos la parte "delta")
        self.U = np.zeros((self.T, self.dim_u), dtype=float)
        self.use_gravity_ff = bool(use_gravity_ff)

    def _softmin_weights(self, costs):
        cmin = np.min(costs)
        w = np.exp(-(costs - cmin) / max(self.lmbda, 1e-9))
        return w / (np.sum(w) + 1e-12)

    def _gravity_feedforward(self):
        if not self.use_gravity_ff:
            return np.zeros((6,), dtype=float)
        g = self.dyn.dyn.g
        m1 = self.dyn.dyn.mass1
        m2 = self.dyn.dyn.mass2
        # Para cancelar gravedad: fuerza hacia arriba = -m*g
        F1_ff = -m1 * g
        F2_ff = -m2 * g
        return np.concatenate([F1_ff, F2_ff]).astype(float)

    def command(self, x0, cost_fn):
        ff = self._gravity_feedforward()

        eps = self.rng.normal(0.0, 1.0, size=(self.K, self.T, self.dim_u)) * self.noise_sigma[None, None, :]

        # U_pert = U_nominal + ruido + ff (aplicado)
        U_pert = self.U[None, :, :] + eps + ff[None, None, :]
        U_pert = self.dyn.clamp_u(U_pert)

        costs = np.zeros((self.K,), dtype=float)
        for k in range(self.K):
            Xk = self.dyn.rollout(x0, U_pert[k])
            costs[k] = cost_fn(Xk, U_pert[k])

        w = self._softmin_weights(costs)

        # Actualiza U guardando la parte "sin ff"
        U_weighted = np.sum(w[:, None, None] * U_pert, axis=0)
        self.U = self.dyn.clamp_u(U_weighted - ff[None, :])

        # output (con ff)
        u0 = self.dyn.clamp_u(self.U[0] + ff)

        # shift
        self.U[:-1] = self.U[1:]
        self.U[-1] = 0.0
        return u0, costs, w


# ============================================================
# 4) Costo 3D (meta + obstáculos esféricos + suelo + resorte + regularización)
# ============================================================

def make_cost_fn_3d(
    goal_r2=(6.0, 0.0, 1.0),
    obstacles=(),              # esferas: (cx,cy,cz,r)
    safe_margin=0.25,
    # pesos:
    w_goal=14.0,
    w_obs=260.0,
    w_floor=500.0,
    w_spring=0.5,
    w_ctrl=0.015,
    w_smooth=0.04,
    # resorte:
    L0=1.0,
    # suelo:
    z_floor=0.2,
):
    goal = np.array(goal_r2, dtype=float)
    obstacles = [tuple(map(float, o)) for o in obstacles]

    def cost_fn(X, U):
        r1 = X[:, 0:3]
        r2 = X[:, 3:6]

        # meta (masa2)
        eT = np.sum((r2[-1] - goal) ** 2)
        eR = np.mean(np.sum((r2 - goal) ** 2, axis=1))
        J_goal = 0.8 * eT + 0.2 * eR

        # obstáculos (ambas masas)
        J_obs = 0.0
        for (cx, cy, cz, rad) in obstacles:
            c = np.array([cx, cy, cz], dtype=float)
            d1 = np.linalg.norm(r1 - c[None, :], axis=1) - (rad + safe_margin)
            d2 = np.linalg.norm(r2 - c[None, :], axis=1) - (rad + safe_margin)
            J_obs += np.sum(np.maximum(0.0, -d1)**2) + np.sum(np.maximum(0.0, -d2)**2)

        # suelo (no caer)
        z1 = r1[:, 2]
        z2 = r2[:, 2]
        J_floor = np.sum(np.maximum(0.0, z_floor - z1)**2 + np.maximum(0.0, z_floor - z2)**2)

        # resorte: mantener distancia ~ L0
        dist = np.linalg.norm(r2 - r1, axis=1)
        J_spring = np.sum((dist - L0) ** 2)

        # control y suavidad
        J_ctrl = np.sum(U**2)
        dU = np.diff(U, axis=0)
        J_smooth = np.sum(dU**2) if dU.shape[0] else 0.0

        return (w_goal*J_goal + w_obs*J_obs + w_floor*J_floor +
                w_spring*J_spring + w_ctrl*J_ctrl + w_smooth*J_smooth)

    return cost_fn


# ============================================================
# 5) Animación 3D estable (sin ax.collections.clear)
# ============================================================

def animate_3d(r1_list, r2_list, Fext1_list, Fext2_list, Fspr1_list, Fspr2_list, obstacles, goal, interval=25):
    r1_arr = np.array(r1_list)
    r2_arr = np.array(r2_list)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    all_pts = np.vstack([r1_arr, r2_arr])
    pad = 1.0
    ax.set_xlim(all_pts[:, 0].min()-pad, all_pts[:, 0].max()+pad)
    ax.set_ylim(all_pts[:, 1].min()-pad, all_pts[:, 1].max()+pad)
    ax.set_zlim(max(0.0, all_pts[:, 2].min()-pad), all_pts[:, 2].max()+pad)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    def draw_sphere(cx, cy, cz, r):
        u = np.linspace(0, 2*np.pi, 18)
        v = np.linspace(0, np.pi, 12)
        xs = cx + r*np.outer(np.cos(u), np.sin(v))
        ys = cy + r*np.outer(np.sin(u), np.sin(v))
        zs = cz + r*np.outer(np.ones_like(u), np.cos(v))
        ax.plot_wireframe(xs, ys, zs, linewidth=0.4, alpha=0.35)

    # FIX: en vez del helper de arriba (que tiene un typo r*npouter),
    # lo definimos bien aquí:
    def draw_sphere(cx, cy, cz, r):
        uu = np.linspace(0, 2*np.pi, 18)
        vv = np.linspace(0, np.pi, 12)
        xs = cx + r*np.outer(np.cos(uu), np.sin(vv))
        ys = cy + r*np.outer(np.sin(uu), np.sin(vv))
        zs = cz + r*np.outer(np.ones_like(uu), np.cos(vv))
        ax.plot_wireframe(xs, ys, zs, linewidth=0.4, alpha=0.35)

    # obstáculos (una vez)
    for (cx, cy, cz, rad) in obstacles:
        draw_sphere(cx, cy, cz, rad)

    # meta (una vez)
    ax.scatter([goal[0]], [goal[1]], [goal[2]], marker='*', s=140)

    # artistas actualizables
    line, = ax.plot([], [], [], 'o:', lw=2)
    p1, = ax.plot([], [], [], 'ro', markersize=7)
    p2, = ax.plot([], [], [], 'bo', markersize=7)

    quiver_handles = []
    force_scale = 0.06

    def init():
        line.set_data([], [])
        line.set_3d_properties([])
        p1.set_data([], [])
        p1.set_3d_properties([])
        p2.set_data([], [])
        p2.set_3d_properties([])
        return (line, p1, p2)

    def update(frame):
        nonlocal quiver_handles

        a = r1_arr[frame]
        b = r2_arr[frame]

        line.set_data([a[0], b[0]], [a[1], b[1]])
        line.set_3d_properties([a[2], b[2]])

        p1.set_data([a[0]], [a[1]])
        p1.set_3d_properties([a[2]])

        p2.set_data([b[0]], [b[1]])
        p2.set_3d_properties([b[2]])

        # remover quivers previos
        for h in quiver_handles:
            try:
                h.remove()
            except Exception:
                pass
        quiver_handles = []

        F1 = np.array(Fext1_list[frame])
        F2 = np.array(Fext2_list[frame])
        S1 = np.array(Fspr1_list[frame])
        S2 = np.array(Fspr2_list[frame])

        def q(pos, vec):
            return ax.quiver(
                pos[0], pos[1], pos[2],
                vec[0], vec[1], vec[2],
                length=force_scale,
                normalize=False,
                arrow_length_ratio=0.25,
                linewidth=1.0,
                alpha=0.6,
            )

        quiver_handles.append(q(a, F1))
        quiver_handles.append(q(b, F2))
        quiver_handles.append(q(a, S1))
        quiver_handles.append(q(b, S2))

        ax.set_title(f"Frame {frame}/{len(r1_arr)-1} | r2=({b[0]:.2f},{b[1]:.2f},{b[2]:.2f})")

        return (line, p1, p2, *quiver_handles)

    ani = FuncAnimation(fig, update, frames=len(r1_arr), init_func=init, interval=interval, blit=False)
    plt.show()
    return ani


# ============================================================
# 6) MAIN (simulación + animación)
# ============================================================

if __name__ == "__main__":
    # ----------- Config del sistema -----------
    mass1 = 1.0
    mass2 = 1.2
    dt = 0.02

    dm = double_mass_dynamics_3d(
        mass1=mass1,
        mass2=mass2,
        position1=[0.0, 0.0, 1.0],
        position2=[1.0, 0.0, 1.0],
        velocity1=np.zeros(3),
        velocity2=np.zeros(3),
    )

    dyn = double_mass_dynamics_3d_wrapper(
        dm, dt=dt,
        u_min=[-30, -30, -60,  -30, -30, -60],
        u_max=[ 30,  30,  60,   30,  30,  60],
        en_gravity=True,
    )

    # ----------- Mundo -----------
    goal = (6.0, 0.0, 1.0)
    obstacles = [
        (2.5, 0.0, 1.0, 0.6),
        (3.8, 0.6, 1.0, 0.55),
        (4.2, -0.7, 1.0, 0.55),
    ]

    cost_fn = make_cost_fn_3d(
        goal_r2=goal,
        obstacles=obstacles,
        safe_margin=0.25,
        w_goal=14.0,
        w_obs=260.0,
        w_floor=600.0,
        w_spring=0.5,
        w_ctrl=0.01,
        w_smooth=0.03,
        L0=1.0,
        z_floor=1.00,
    )

    mppi = MPPI(
        dynamics=dyn,
        horizon=25,
        num_samples=1000,
        lambda_=1.7,
        noise_sigma=[4, 4, 2,  4, 4, 2],
        seed=42,
        use_gravity_ff=True,
    )

    # ----------- Simulación -----------
    x = dyn.get_state()
    N_steps = 260

    r1_list, r2_list = [], []
    Fext1_list, Fext2_list = [], []
    Fspr1_list, Fspr2_list = [], []

    # log inicial
    r1_list.append(x[0:3].copy())
    r2_list.append(x[3:6].copy())
    Fspr = dyn.spring_force_from_state(x)
    Fspr1_list.append((-Fspr).copy())
    Fspr2_list.append(( Fspr).copy())
    Fext1_list.append(np.zeros(3))
    Fext2_list.append(np.zeros(3))

    for k in tqdm.trange(N_steps):
        u, costs, w = mppi.command(x, cost_fn)

        F1 = u[0:3].copy()
        F2 = u[3:6].copy()

        # resorte (del estado actual para logging)
        Fspr = dyn.spring_force_from_state(x)
        Fspr1 = (-Fspr).copy()
        Fspr2 = ( Fspr).copy()

        # paso real (misma clase)
        dm.step(dt, F1, F2, en_gravity=True)
        x = dyn.get_state()

        # log
        r1_list.append(x[0:3].copy())
        r2_list.append(x[3:6].copy())
        Fext1_list.append(F1)
        Fext2_list.append(F2)
        Fspr1_list.append(Fspr1)
        Fspr2_list.append(Fspr2)

        # llegada (masa2)
        if np.linalg.norm(x[3:6] - np.array(goal)) < 0.35 and np.linalg.norm(x[9:12]) < 0.6:
            break

    print("Pasos simulados:", len(r1_list) - 1)
    print("Pos final masa2:", r2_list[-1], "dist a meta:", np.linalg.norm(r2_list[-1] - np.array(goal)))

    # ----------- Animación -----------
    animate_3d(r1_list, r2_list, Fext1_list, Fext2_list, Fspr1_list, Fspr2_list, obstacles, goal, interval=25)
