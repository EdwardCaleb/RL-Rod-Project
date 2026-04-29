
'''
En este test:
- Leemos el thrust en PWM y voltaje desde el Crazyflie
- Estimamos la fuerza de thrust en newtons usando un modelo PWM a fuerza
'''

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

from PWM_to_force_thrust import PwMToForceThrust


# =========================
# CONFIG
# =========================
URI = 'radio://0/90/2M/E7E7E7E705'
robot_id = 538

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
            # optitrack devuelve quaterniones como [qx, qy, qz, qw], tomar eso en consideración para la estimación
            q = np.array(quat)

            # 🔴 ESTIMACIÓN COMPLETA
            p, v, a, R = estimator.update_state_p_q(p, q, dt, input_format="xyzw")

            # enviar posición al Crazyflie
            cf.extpos.send_extpos(float(p[0]), float(p[1]), float(p[2]))

            # # 🔍 DEBUG (cada ~100 ms)
            # if int(now * 10) % 10 == 0:
            #     print(f"v: {v}, a: {a}", f"R: \n{R}")

        time.sleep(0.01)


# =========================
# LOGGING + PWM model
# =========================
def log_callback(timestamp, data, logconf, pwm_model):
    total_pwm = data.get('propForce.totalPwm', None)
    vbat = data.get('pm.vbat', None)

    if total_pwm is None or vbat is None:
        return
    
    force_est = pwm_model.pwm_to_force(total_pwm, vbat)


    # imprimir cada ~100 ms
    if (timestamp//10) % 100 == 0:
        print(
            f"[{timestamp}] "
            f"PWM: {total_pwm} | "
            f"Vbat: {vbat:.2f} | "
            f"Force Est(N): {force_est:.3f}"
        )


def start_logging(cf, pwm_model):
    log_conf = LogConfig(name='PropForces', period_in_ms=10)

    log_conf.add_variable('propForce.totalPwm', 'uint32_t')
    log_conf.add_variable('pm.vbat', 'float')

    cf.log.add_config(log_conf)

    if log_conf.valid:
        log_conf.data_received_cb.add_callback(
            lambda t, d, l: log_callback(t, d, l, pwm_model)
        )
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
    takeoff(cf, height=0.5, duration=2.0)

    t0 = time.time()

    try:
        while not stop_event.is_set():
            t = time.time() - t0

            # trayectoria (elige una)
            # dp, _ , _  path.do_sine_x(t, center=[0,0,1.5], amplitude=0.3, frequency=0.2)
            # dp, _ , _  = path.do_fill_spherical_spiral(t, center=np.array([0.0, 0.0, 1.5]), radius=0.5, omega=0.5*np.sqrt(2), vertical_speed=0.5*(2)**(1/4))
            # dp, _ , _  = path.do_square_xz(t, center=np.array([0.0, 0.0, 1.5]), side_length=1.0, omega=1.0)
            # dp, _ , _ = path.do_multistep_z(t, center=np.array([0.0, 0.0, 0.3]), step_height=0.4, n_steps=4, step_duration=4.0)
            dp, _ , _ = path.do_linear_z(t, center=np.array([0.0, 0.0, 0.5]), z_speed=0.10)
            # dp, _ , _ = path.fixed_point([0.0, 0.0, 1.5])

            dp = np.array(dp).flatten()
            x, y, z = map(float, dp)

            cf.commander.send_position_setpoint(x, y, z, 0.0)

            time.sleep(path.dt)

    except KeyboardInterrupt:
        print("Interrumpido")

    finally:
        print("Aterrizando...")
        land(cf, height=0.5, duration=2.0)


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


        # Estimador de estado completo
        estimator = FullStateEstimator(dt=0.01, alpha=0.5)

        # Modelo PWM a fuerza
        pwm_model = PwMToForceThrust()
        
        # 🔹 Thread extpos
        extpos_thread = threading.Thread(
            target=send_extpos_loop,
            args=(cf, estimator),
            daemon=True
        )
        extpos_thread.start()

        # 🔹 Logging
        log_conf = start_logging(cf, pwm_model)

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