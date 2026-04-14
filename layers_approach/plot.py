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
    plt.ylabel('Fuerza (N)')
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