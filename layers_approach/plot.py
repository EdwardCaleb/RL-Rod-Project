import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory(x, y, xlabel='', ylabel='', title='', xlim=None, ylim=None):
    plt.figure()
    plt.plot(x, y, label='Trayectoria del dron')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    #ajustar límites y agregar referencia deseada
    if xlim is not None: plt.xlim(xlim)
    if ylim is not None: plt.ylim(ylim)

    plt.legend()
    plt.show()

def plot_traj_and_ref(x, y, ref_x, ref_y, xlabel='', ylabel='', title='', xlim=None, ylim=None):
    plt.figure()
    plt.plot(x, y, label='Trayectoria del dron')
    plt.plot(ref_x, ref_y, label='Referencia')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    #ajustar límites y agregar referencia deseada
    if xlim is not None: plt.xlim(xlim)
    if ylim is not None: plt.ylim(ylim)

    plt.legend()
    plt.show()


def plot_force_vs_time(axt, F_1, name_1=None, linestyle_1='-' , color_1='blue',
                       F_2=None, name_2=None, linestyle_2='--', color_2='orange',
                       F_3=None, name_3=None, linestyle_3='-.', color_3='green',
                       F_4=None, name_4=None, linestyle_4=':' , color_4='red',
                       F_5=None, name_5=None, linestyle_5='-' , color_5='purple',
                       F_6=None, name_6=None, linestyle_6='--', color_6='brown',
                       F_7=None, name_7=None, linestyle_7='-.', color_7='pink'):
    plt.figure()
    if F_1 is not None:
        plt.plot(axt, F_1, label=name_1, linestyle=linestyle_1, color=color_1)
    if F_2 is not None:
        plt.plot(axt, F_2, label=name_2, linestyle=linestyle_2, color=color_2)
    if F_3 is not None:
        plt.plot(axt, F_3, label=name_3, linestyle=linestyle_3, color=color_3)
    if F_4 is not None:
        plt.plot(axt, F_4, label=name_4, linestyle=linestyle_4, color=color_4)
    if F_5 is not None:
        plt.plot(axt, F_5, label=name_5, linestyle=linestyle_5, color=color_5)
    if F_6 is not None:
        plt.plot(axt, F_6, label=name_6, linestyle=linestyle_6, color=color_6)
    if F_7 is not None:
        plt.plot(axt, F_7, label=name_7, linestyle=linestyle_7, color=color_7)
    plt.xlabel('Paso de tiempo (s)')
    plt.ylabel('Modulo de Fuerza (N)')
    plt.legend()
    plt.show()


def plot(ejex,var_1,name_1='',var_2=None,name_2='',var_3=None,name_3='',var_4=None,name_4='',xlabel='',ylabel='',title=''):
    plt.figure()
    if var_1 is not None:
        plt.plot(ejex, var_1, label=name_1, linestyle='-')
    if var_2 is not None:
        plt.plot(ejex, var_2, label=name_2, linestyle='--')
    if var_3 is not None:
        plt.plot(ejex, var_3, label=name_3, linestyle='-.')
    if var_4 is not None:
        plt.plot(ejex, var_4, label=name_4, linestyle=':')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()



def plot_vector_field_3d(model, grid_lim=1.5, center=np.array([0.0, 0.0, 2.0]), n_points=6, scale=1.0, title="Campo de fuerzas"):
    x = np.linspace(center[0] - grid_lim, center[0] + grid_lim, n_points)
    y = np.linspace(center[1] - grid_lim, center[1] + grid_lim, n_points)
    z = np.linspace(center[2] - grid_lim, center[2] + grid_lim, n_points)

    X, Y, Z = np.meshgrid(x, y, z)

    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    W = np.zeros_like(Z)
    magnitude = np.zeros_like(X)

    # Evaluar modelo
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                phi = np.array([X[i,j,k], Y[i,j,k], Z[i,j,k], 1.0])
                F = model.predict(phi)

                U[i,j,k] = F[0]
                V[i,j,k] = F[1]
                W[i,j,k] = F[2]

                magnitude[i,j,k] = np.linalg.norm(F)

    # Escalar vectores (opcional pero recomendado)
    U_scaled = U * scale
    V_scaled = V * scale
    W_scaled = W * scale

    # Normalizar magnitud para colores
    mag_norm = magnitude / (np.max(magnitude) + 1e-8)

    colors = plt.cm.viridis(mag_norm.flatten())

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.quiver(
        X, Y, Z,
        U_scaled, V_scaled, W_scaled,
        color=colors,
        length=1.0,   # base length
        normalize=False
    )

    # añadir un punto en el centro para referencia, el punto no es muy grande para no tapar las flechas, pero se ve claramente
    ax.scatter(center[0], center[1], center[2], color='red', s=20, label=f'Referencia ({center[0]},{center[1]},{center[2]})')
    
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()



# plot_true_vector_field_3d(vector_F_res, vector_r, grid_lim=2.5, n_points=10, scale=0.03, title="Campo de fuerzas real del resorte")
def plot_true_vector_field_3d(
        force_fn,
        k=10.0, rest_length=1.0,
        grid_lim=1.5, 
        center=np.array([0.0, 0.0, 2.0]), 
        n_points=6, 
        scale=1.0, 
        title="Campo de fuerzas"):

    x = np.linspace(center[0] - grid_lim, center[0] + grid_lim, n_points)
    y = np.linspace(center[1] - grid_lim, center[1] + grid_lim, n_points)
    z = np.linspace(center[2] - grid_lim, center[2] + grid_lim, n_points)

    X, Y, Z = np.meshgrid(x, y, z)

    U = np.zeros_like(X)
    V = np.zeros_like(Y)
    W = np.zeros_like(Z)
    magnitude = np.zeros_like(X)

    # Evaluar modelo
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            for k in range(X.shape[2]):
                phi = np.array([X[i,j,k], Y[i,j,k], Z[i,j,k], 1.0])
                
                r = np.array([X[i,j,k], Y[i,j,k], Z[i,j,k]])
                F = force_fn(r, center, k=k, rest_length=rest_length)

                U[i,j,k] = F[0]
                V[i,j,k] = F[1]
                W[i,j,k] = F[2]

                magnitude[i,j,k] = np.linalg.norm(F)

    # Escalar vectores (opcional pero recomendado)
    U_scaled = U * scale
    V_scaled = V * scale
    W_scaled = W * scale

    # Normalizar magnitud para colores
    mag_norm = magnitude / (np.max(magnitude) + 1e-8)

    colors = plt.cm.viridis(mag_norm.flatten())

    # Plot
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    ax.quiver(
        X, Y, Z,
        U_scaled, V_scaled, W_scaled,
        color=colors,
        length=1.0,   # base length
        normalize=False
    )

    # añadir un punto en el centro para referencia, el punto no es muy grande para no tapar las flechas, pero se ve claramente
    ax.scatter(center[0], center[1], center[2], color='red', s=20, label=f'Referencia ({center[0]},{center[1]},{center[2]})')
    
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    plt.show()






import numpy as np
import matplotlib.pyplot as plt

def plot_vector_field_2d(
    model=None,
    force_fn=None,
    k=10.0, rest_length=1.0,
    plane="xy",                 # "xy", "xz", "yz"
    fixed_value=0.0,            # valor del eje que se fija
    center=np.array([0,0,0]),
    grid_lim=1.5,
    n_points=20,
    scale=1.0,
    title="Campo vectorial 2D"
):
    """
    Puedes usar:
    - model.predict(phi)
    o
    - force_fn(r, center)
    """

    axis_map = {
        "xy": (0, 1, 2),
        "xz": (0, 2, 1),
        "yz": (1, 2, 0)
    }

    i1, i2, i_fixed = axis_map[plane]

    # Crear grid 2D
    a = np.linspace(center[i1] - grid_lim, center[i1] + grid_lim, n_points)
    b = np.linspace(center[i2] - grid_lim, center[i2] + grid_lim, n_points)

    A, B = np.meshgrid(a, b)

    U = np.zeros_like(A)
    V = np.zeros_like(B)
    magnitude = np.zeros_like(A)

    # Evaluar campo
    for i in range(A.shape[0]):
        for j in range(A.shape[1]):

            r = np.zeros(3)
            r[i1] = A[i,j]
            r[i2] = B[i,j]
            r[i_fixed] = fixed_value

            # Elegir fuente
            if model is not None:
                phi = np.array([r[0], r[1], r[2], 1.0])
                F = np.array(model.predict(phi))
            elif force_fn is not None:
                F = force_fn(r, center, k=k, rest_length=rest_length)
            else:
                raise ValueError("Debes pasar model o force_fn")

            U[i,j] = F[i1]
            V[i,j] = F[i2]
            magnitude[i,j] = np.linalg.norm(F)

    # Dirección + magnitud
    norm = np.sqrt(U**2 + V**2) + 1e-8
    U_plot = (U / norm) * magnitude * scale
    V_plot = (V / norm) * magnitude * scale

    # Colores
    mag_norm = magnitude / (np.max(magnitude) + 1e-8)

    # Plot
    plt.figure()
    plt.quiver(
        A, B,
        U_plot, V_plot,
        mag_norm,
        cmap="viridis"
    )

    plt.colorbar(label="Magnitud de la fuerza")

    labels = ["X", "Y", "Z"]
    plt.xlabel(labels[i1])
    plt.ylabel(labels[i2])

    plt.title(f"{title} ({plane} @ {labels[i_fixed]}={fixed_value:.2f})")

    # punto centro proyectado
    plt.scatter(center[i1], center[i2], color="red", s=30)

    plt.axis("equal")
    plt.grid(True)

    plt.show()







import numpy as np
import matplotlib.pyplot as plt

def plot_spring_damper_fields_2d(
    model=None,
    force_fn=None,
    k=10.0, rest_length=1.0, c=1.0,
    center=np.array([0,0,0]),
    plane="xy",
    fixed_value=0.0,
    grid_lim=1.5,
    n_points=20,
    force_clip=100.0,   # 🔑 nuevo: limita fuerzas
    title="Campos vectoriales (posición vs velocidad)"
):
    axis_map = {
        "xy": (0, 1, 2),
        "xz": (0, 2, 1),
        "yz": (1, 2, 0)
    }

    i1, i2, i_fixed = axis_map[plane]

    a = np.linspace(center[i1] - grid_lim, center[i1] + grid_lim, n_points)
    b = np.linspace(center[i2] - grid_lim, center[i2] + grid_lim, n_points)

    A, B = np.meshgrid(a, b)

    # =========================
    # HELPER: obtener fuerza
    # =========================
    def get_force(r, v):
        if model is not None:
            phi = np.array([r[0], r[1], r[2], v[0], v[1], v[2], 1.0])
            F = np.array(model.predict(phi)).flatten()
        elif force_fn is not None:
            F = force_fn(r, v, center, k=k, rest_length=rest_length, c=c)
        else:
            raise ValueError("Debes pasar model o force_fn")

        # 🔑 limpiar NaN / inf
        F = np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)

        # 🔑 evitar explosiones numéricas
        F = np.clip(F, -force_clip, force_clip)

        return F

    # =========================
    # 1) CAMPO DE POSICIÓN
    # =========================
    U_pos = np.zeros_like(A)
    V_pos = np.zeros_like(B)

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):

            r = np.zeros(3)
            r[i1] = A[i,j]
            r[i2] = B[i,j]
            r[i_fixed] = fixed_value

            v = np.zeros(3)

            F = get_force(r, v)

            U_pos[i,j] = F[i1]
            V_pos[i,j] = F[i2]

    # =========================
    # 2) CAMPO DE VELOCIDAD
    # =========================
    U_vel = np.zeros_like(A)
    V_vel = np.zeros_like(B)

    r_fixed = center + 0.1*np.array([1.0, 0.0, 0.0])

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):

            v = np.zeros(3)
            v[i1] = A[i,j]
            v[i2] = B[i,j]
            v[i_fixed] = 0.0

            F = get_force(r_fixed, v)

            U_vel[i,j] = F[i1]
            V_vel[i,j] = F[i2]

    # 🔑 limpieza final (por seguridad extra)
    U_pos = np.nan_to_num(U_pos)
    V_pos = np.nan_to_num(V_pos)
    U_vel = np.nan_to_num(U_vel)
    V_vel = np.nan_to_num(V_vel)

    # magnitudes (para color)
    mag_pos = np.sqrt(U_pos**2 + V_pos**2)
    mag_vel = np.sqrt(U_vel**2 + V_vel**2)

    # =========================
    # PLOTS
    # =========================
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    labels = ["X", "Y", "Z"]

    # --- POSICIÓN ---
    q1 = axs[0].quiver(
        A, B,

        U_pos, V_pos,
        mag_pos,
        cmap="viridis",
        scale=None   # 🔑 evita warnings
    )
    axs[0].set_title("Campo por posición (v=0)")
    axs[0].set_xlabel(labels[i1])
    axs[0].set_ylabel(labels[i2])
    axs[0].scatter(center[i1], center[i2], color="red", s=30)
    axs[0].axis("equal")
    axs[0].grid(True)
    fig.colorbar(q1, ax=axs[0])

    # --- VELOCIDAD ---
    q2 = axs[1].quiver(
        A, B,
        U_vel, V_vel,
        mag_vel,
        cmap="plasma",
        scale=None   # 🔑 evita warnings
    )
    axs[1].set_title("Campo por velocidad (r fijo)")
    axs[1].set_xlabel(f"v{labels[i1]}")
    axs[1].set_ylabel(f"v{labels[i2]}")
    axs[1].axis("equal")
    axs[1].grid(True)
    fig.colorbar(q2, ax=axs[1])

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()










# # visualización de trayectoria (opcional)
# import matplotlib.pyplot as plt
# vector_r = np.array(vector_r)
# plt.figure()
# plt.plot(vector_r[:,0], vector_r[:,2], label='Trayectoria del dron')
# plt.xlabel('x (m)')
# plt.ylabel('z (m)')

# #ajustar límites y agregar referencia deseada
# # plt.xlim(p0[0] - 1, p0[0] + 1)
# # plt.ylim(p0[2] - 1, p0[2] + 1)

# plt.legend()
# plt.show()