import time
import threading
import numpy as np
import logging
import os

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig

from optitrack_natnet_main import tracker
from path_generator import PathGenerator

from full_state_estimator import FullStateEstimator


# =========================
# CONFIG
# =========================
URI = 'radio://0/90/2M/E7E7E7E705'
robot_id = 537

logging.basicConfig(level=logging.ERROR)
os.environ["CFCLIENT_CACHE_DIR"] = "./cache"

stop_event = threading.Event()


# =========================
# OPTITRACK
# =========================
tracker = tracker.OptiTrackClient(
    client_address="192.168.0.99",
    server_address="192.168.0.4"
)
tracker.start()


def send_extpos_loop(cf, estimator):
    t_prev = time.time()

    while not stop_event.is_set():
        pos, quat = tracker.get_full_pose(robot_id)

        if pos is not None and quat is not None:
            now = time.time()
            dt = now - t_prev
            t_prev = now

            p = np.array(pos)
            q = np.array(quat)

            # 🔴 ESTIMACIÓN COMPLETA
            p, v, a, R = estimator.update_state_p_q(p, q, dt)

            # enviar posición al Crazyflie
            cf.extpos.send_extpos(float(p[0]), float(p[1]), float(p[2]))

            # 🔍 DEBUG (cada ~100 ms)
            if int(now * 10) % 10 == 0:
                print(f"v: {v}, a: {a}")

        time.sleep(0.01)


# =========================
# LOGGING
# =========================
def log_callback(timestamp, data, logconf):
    total_pwm = data.get('propForce.totalPwm', None)
    total_uncapped = data.get('propForce.totalUncapped', None)
    total_si = data.get('propForce.totalSi', None)
    vbat = data.get('pm.vbat', None)

    # imprimir cada ~100 ms
    if timestamp % 100 == 0:
        print(f"[{timestamp}] PWM: {total_pwm} | Force(N): {total_si} | Vbat: {vbat}")


def start_logging(cf):
    log_conf = LogConfig(name='PropForces', period_in_ms=10)

    log_conf.add_variable('propForce.totalPwm', 'uint32_t')
    log_conf.add_variable('propForce.totalUncapped', 'uint32_t')
    log_conf.add_variable('propForce.totalSi', 'float')
    log_conf.add_variable('pm.vbat', 'float')

    cf.log.add_config(log_conf)

    if log_conf.valid:
        log_conf.data_received_cb.add_callback(log_callback)
        log_conf.start()
        print("Logging started...")
        return log_conf
    else:
        print("Log config invalid")
        return None


# =========================
# CONTROL
# =========================
def takeoff(cf, height=1.0, duration=2.0):
    steps = int(duration / 0.02)
    for i in range(steps):
        z = height * (i / steps)
        cf.commander.send_position_setpoint(0.0, 0.0, float(z), 0.0)
        time.sleep(0.02)


def land(cf, height=1.0, duration=2.0):
    steps = int(duration / 0.02)
    for i in range(steps):
        z = height * (1 - i / steps)
        cf.commander.send_position_setpoint(0.0, 0.0, float(z), 0.0)
        time.sleep(0.02)

    cf.commander.send_stop_setpoint()


def run_trajectory(cf):
    print("Iniciando trayectoria...")

    path = PathGenerator(dt=0.01)

    # 🛫 TAKEOFF
    takeoff(cf, height=1.5, duration=2.0)

    t0 = time.time()

    try:
        while not stop_event.is_set():
            t = time.time() - t0

            # trayectoria (elige una)
            # dp = path.do_sine_x(t, center=[0,0,1.5], amplitude=0.3, frequency=0.2)
            dp, _ , _ = path.fixed_point([0.0, 0.0, 1.5])

            dp = np.array(dp).flatten()
            x, y, z = map(float, dp)

            cf.commander.send_position_setpoint(x, y, z, 0.0)

            time.sleep(path.dt)

    except KeyboardInterrupt:
        print("Interrumpido")

    finally:
        print("Aterrizando...")
        land(cf, height=1.5, duration=2.0)


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:

        cf = scf.cf

        print("Inicializando...")
        time.sleep(2)

        # 🔧 CONFIGURACIÓN CONTROL
        cf.param.set_value('stabilizer.estimator', '2')
        cf.param.set_value('stabilizer.controller', '2')
        cf.param.set_value('commander.enHighLevel', '0')

        # opcional: masa
        # cf.param.set_value('ctrlMel.mass', '0.148')

        time.sleep(0.5)

        print("Armando dron...")
        cf.platform.send_arming_request(True)
        time.sleep(1.0)

        print("Controlador:", cf.param.get_value('stabilizer.controller'))
        print("Estimador:", cf.param.get_value('stabilizer.estimator'))

        # 🔹 Thread extpos
        estimator = FullStateEstimator(dt=0.01, alpha=0.5)
        extpos_thread = threading.Thread(
            target=send_extpos_loop,
            args=(cf, estimator),
            daemon=True
        )
        extpos_thread.start()

        # 🔹 Logging
        log_conf = start_logging(cf)

        try:
            run_trajectory(cf)

        except KeyboardInterrupt:
            print("Ctrl+C detectado")

        finally:
            print("Cerrando...")

            stop_event.set()
            time.sleep(0.5)

            if log_conf is not None:
                log_conf.stop()

            cf.commander.send_stop_setpoint()
            time.sleep(0.1)

            cf.platform.send_arming_request(False)
            time.sleep(0.2)

            extpos_thread.join(timeout=1)

            print("Finished")