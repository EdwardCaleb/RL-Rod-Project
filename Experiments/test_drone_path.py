'''
Este test hace que el drone vuele siguiendo una trayectoria generada por el PathGenerator,
mientras se le envían posiciones externas desde el tracker de OptiTrack.
También imprime las posiciones deseadas XYZ y las posiciones actuales XYZ medidas por OptiTrack.
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
robot_id = 542


# =========================
# 🔵 VARIABLES GLOBALES
# =========================
current_pos = None
current_quat = None
pose_lock = threading.Lock()


# =========================
# 🔵 OPTITRACK
# =========================
tracker = tracker.OptiTrackClient(
    client_address="192.168.0.32",
    server_address="192.168.0.4"
)
tracker.start()


def send_extpos_loop(cf):
    """
    Lee la posición del drone desde OptiTrack y la envía al Crazyflie
    como posición externa. También guarda la posición actual para imprimirla.
    """

    global current_pos, current_quat

    while True:
        pos, quat = tracker.get_full_pose(robot_id)

        if pos is not None:
            x, y, z = pos

            # Enviar posición externa al Crazyflie
            cf.extpos.send_extpos(
                float(x),
                float(y),
                float(z)
            )

            # Guardar posición actual medida por OptiTrack
            with pose_lock:
                current_pos = np.array([x, y, z], dtype=float)
                current_quat = quat

        time.sleep(0.01)


# =========================
# 🔵 CONTROL
# =========================
def takeoff(cf, height=0.5, duration=2.0):
    """
    Despegue usando setpoints de posición.
    """

    steps = int(duration / 0.02)

    for i in range(steps):
        z = height * (i / steps)
        cf.commander.send_position_setpoint(0.0, 0.0, z, 0.0)
        time.sleep(0.02)


def land(cf, height=0.5, duration=2.0):
    """
    Aterrizaje usando setpoints de posición.
    """

    steps = int(duration / 0.02)

    for i in range(steps):
        z = height * (1 - i / steps)
        cf.commander.send_position_setpoint(0.0, 0.0, z, 0.0)
        time.sleep(0.02)

    cf.commander.send_stop_setpoint()


def run_trajectory(cf):
    """
    Ejecuta la trayectoria deseada e imprime:
    - posición deseada
    - posición actual medida por OptiTrack
    - error de posición
    """

    print("Iniciando trayectoria...")

    # Usar Kalman estimator
    cf.param.set_value('stabilizer.estimator', '2')

    # Usar Mellinger controller
    cf.param.set_value('stabilizer.controller', '2')

    # Desactivar high level commander
    cf.param.set_value('commander.enHighLevel', '0')

    time.sleep(2)

    print("Estimator:", cf.param.get_value('stabilizer.estimator'))
    print("Controller:", cf.param.get_value('stabilizer.controller'))
    print("Mellinger ki_xy:", cf.param.get_value('ctrlMel.ki_xy'))
    print("Mellinger ki_z:", cf.param.get_value('ctrlMel.ki_z'))
    print("Mellinger ki_m_xy:", cf.param.get_value('ctrlMel.ki_m_xy'))
    print("Mellinger ki_m_z:", cf.param.get_value('ctrlMel.ki_m_z'))

    path = PathGenerator(dt=0.01)

    # 🛫 DESPEGUE
    takeoff(cf, height=0.5, duration=2.0)

    t0 = time.time()
    last_print = 0.0

    try:
        while True:
            t = time.time() - t0

            # =========================
            # 🔵 TRAYECTORIA DESEADA
            # =========================

            # Ejemplo 1: punto fijo
            dp = path.fixed_point([0.0, 0.0, 1.5])

            # Ejemplo 2: seno en X
            # dp = path.do_sine_x(
            #     t,
            #     center=[0.0, 0.0, 0.5],
            #     amplitude=0.5,
            #     frequency=0.2
            # )

            # Ejemplo 3: espiral esférica
            # dp = path.do_fill_spherical_spiral(
            #     t,
            #     center=np.array([0.0, 0.0, 2.0]),
            #     radius=0.5,
            #     omega=0.5 * np.sqrt(2),
            #     vertical_speed=0.5 * (2)**(1/4)
            # )

            # Ejemplo 4: cuadrado en XZ
            # dp = path.do_square_xz(
            #     t,
            #     center=np.array([0.0, 0.0, 1.5]),
            #     side_length=0.5,
            #     omega=1.0
            # )

            # Ejemplo 5: círculo en XZ
            # dp = path.do_circle_xz(
            #     t,
            #     center=np.array([0.0, 0.0, 2.0]),
            #     radius=0.3,
            #     omega=1.0
            # )

            # dp = path.do_square_xy(
            #     t,
            #     center=np.array([0.0, 0.0, 1.0]),
            #     side_length=1.0,
            #     omega=1.0
            # )

            # Asegurar formato correcto
            dp = np.array(dp).flatten()

            x_des = float(dp[0])
            y_des = float(dp[1])
            z_des = float(dp[2])

            desired_pos = np.array([x_des, y_des, z_des], dtype=float)

            # Enviar setpoint deseado al drone
            cf.commander.send_position_setpoint(
                x_des,
                y_des,
                z_des,
                0.0
            )

            # =========================
            # 🔵 POSICIÓN ACTUAL
            # =========================
            with pose_lock:
                actual_pos = None if current_pos is None else current_pos.copy()

            # =========================
            # 🔵 IMPRIMIR DATOS
            # =========================
            # Imprime a 10 Hz, aunque el control corra a 100 Hz
            if t - last_print >= 0.1:
                if actual_pos is not None:
                    error = desired_pos - actual_pos

                    print(
                        f"t={t:6.2f} | "
                        f"DES x={desired_pos[0]:+7.3f}, "
                        f"y={desired_pos[1]:+7.3f}, "
                        f"z={desired_pos[2]:+7.3f} | "
                        f"ACT x={actual_pos[0]:+7.3f}, "
                        f"y={actual_pos[1]:+7.3f}, "
                        f"z={actual_pos[2]:+7.3f} | "
                        f"ERR x={error[0]:+7.3f}, "
                        f"y={error[1]:+7.3f}, "
                        f"z={error[2]:+7.3f}",
                        flush=True
                    )

                else:
                    print(
                        f"t={t:6.2f} | "
                        "Esperando posición actual de OptiTrack...",
                        flush=True
                    )

                last_print = t

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

        # Hilo de posición externa desde OptiTrack
        threading.Thread(
            target=send_extpos_loop,
            args=(scf.cf,),
            daemon=True
        ).start()

        run_trajectory(scf.cf)