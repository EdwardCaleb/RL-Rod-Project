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