'''
Este test hace que el drone vuele siguiendo una trayectoria generada por el PathGenerator,
mientras se le envían posiciones externas desde el tracker de OptiTrack
'''


import time
import threading
import numpy as np

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

from optitrack_natnet_main import tracker

from path_generator import PathGenerator

URI = 'radio://0/90/2M/E7E7E7E705'
robot_id = 537


# =========================
# 🔵 PATH GENERATOR
# =========================
# class PathGenerator:
#     def __init__(self, dt=0.01):
#         self.dt = dt

#     def do_sine_x(self, t, center, amplitude, frequency):
#         x_c, y_c, z_c = center

#         dpx = x_c + amplitude * np.sin(2 * np.pi * frequency * t)
#         dpy = y_c
#         dpz = z_c

#         dp = np.array([dpx, dpy, dpz])

#         return dp


# =========================
# 🔵 OPTITRACK
# =========================
tracker = tracker.OptiTrackClient(
    client_address="192.168.0.99",
    server_address="192.168.0.4"
)
tracker.start()


def send_extpos_loop(cf):
    while True:
        pos, quat = tracker.get_full_pose(robot_id)

        if pos is not None:
            x, y, z = pos
            cf.extpos.send_extpos(x, y, z)

        time.sleep(0.01)


# =========================
# 🔵 CONTROL
# =========================
def takeoff(cf, height=0.5, duration=2.0):
    steps = int(duration / 0.02)
    for i in range(steps):
        z = height * (i / steps)
        cf.commander.send_position_setpoint(0, 0, z, 0)
        time.sleep(0.02)


def land(cf, height=0.5, duration=2.0):
    steps = int(duration / 0.02)
    for i in range(steps):
        z = height * (1 - i / steps)
        cf.commander.send_position_setpoint(0, 0, z, 0)
        time.sleep(0.02)

    cf.commander.send_stop_setpoint()


def run_trajectory(cf):
    print("Iniciando trayectoria...")

    # 🔴 IMPORTANTE
    cf.param.set_value('stabilizer.estimator', '2')
    cf.param.set_value('commander.enHighLevel', '0')

    time.sleep(2)

    path = PathGenerator(dt=0.01)

    # 🛫 DESPEGUE
    takeoff(cf, height=0.5, duration=2.0)

    t0 = time.time()

    try:
        while True:
            t = time.time() - t0

            # Trayectoria seno en X
            # dp = path.do_sine_x(
            #     t,
            #     center=[0.0, 0.0, 0.5],
            #     amplitude=0.5,
            #     frequency=0.2
            # )

            dp = path.do_fill_spherical_spiral(t, center=np.array([0.0, 0.0, 2.0]), radius=0.5, omega=0.5*np.sqrt(2), vertical_speed=0.5*(2)**(1/4))
            # dp = path.do_square_xz(t, center=np.array([1.0, 0.0, 1.5]), side_length=0.5, omega=1.0)


            # x, y, z = dp
            dp = np.array(dp).flatten()
            x = float(dp[0])
            y = float(dp[1])
            z = float(dp[2])

            cf.commander.send_position_setpoint(x, y, z, 0.0)
            # cf.commander.send_position_setpoint(
            #     float(x),
            #     float(y),
            #     float(z),
            #     0.0
            # )

            time.sleep(path.dt)

    except KeyboardInterrupt:
        print("Interrumpido → aterrizando")

    # 🛬 ATERRIZAJE
    land(cf, height=0.5, duration=2.0)


# =========================
# 🔵 MAIN
# =========================
if __name__ == "__main__":
    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie()) as scf:

        # Hilo de posición externa
        threading.Thread(
            target=send_extpos_loop,
            args=(scf.cf,),
            daemon=True
        ).start()

        run_trajectory(scf.cf)