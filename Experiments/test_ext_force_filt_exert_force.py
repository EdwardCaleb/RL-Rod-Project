
'''
En este test:
- Leemos el thrust en PWM y voltaje desde el Crazyflie
- Estimamos la fuerza de thrust en newtons usando un modelo PWM a fuerza
- Estimamos la fuerza exogena aplicada al drone en x, y, z usando un observador de fuerzas
- Filtramos la fuerza exogena estimada para obtener una versión más suave y menos ruidosa
- Incluimos la adicion de una fuerza exogena manual (empujón) durante la trayectoria para observar su efecto en la estimación de fuerza y su filtrado
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
from force_observer import ForceObserver
from force_filter import MovingAverageForgettingFactorFilter


# =========================
# CONFIG
# =========================
URI = 'radio://0/90/2M/E7E7E7E705'
robot_id = 542

DRONE_MASS = 0.153
GRAVITY = 9.81

logging.basicConfig(level=logging.ERROR)
os.environ["CFCLIENT_CACHE_DIR"] = "./cache"

stop_event = threading.Event()

latest_state = {
    "p": None,
    "v": None,
    "a": None,
    "R": None,
    "force_thrust": None,
    "force_external": None,
}


# =========================
# OPTITRACK
# =========================
tracker = tracker.OptiTrackClient(
    client_address="192.168.0.32",
    server_address="192.168.0.4"
)
tracker.start()


def send_extpos_loop(cf, estimator):
    t_prev = time.time()

    while not stop_event.is_set():
        pos, quat = tracker.get_full_pose(robot_id)

        if pos is not None and quat is not None:
            now = time.time()
            dt = max(min(now - t_prev, 0.02), 0.005)
            t_prev = now

            p = np.array(pos, dtype=float)

            # OptiTrack entrega quaternion como (x,y,z,w)
            q = np.array(quat, dtype=float)

            p, v, a, R = estimator.update_state_p_q(
                p, q, dt, input_format="xyzw"
            )

            latest_state["p"] = p
            latest_state["v"] = v
            latest_state["a"] = a
            latest_state["R"] = R

            cf.extpos.send_extpos(float(p[0]), float(p[1]), float(p[2]))

        time.sleep(0.01)


# =========================
# VIRTUAL FORCE COMMAND
# =========================
def set_virtual_force(cf, f):
    cf.param.set_value('ctrlMel.fvirt_x', str(float(f[0])))
    cf.param.set_value('ctrlMel.fvirt_y', str(float(f[1])))
    cf.param.set_value('ctrlMel.fvirt_z', str(float(f[2])))


def virtual_force_loop(cf):
    """
    Fuerza virtual:
    - 0 a 5 s: apagada
    - 5 a 10 s: encendida
    - 10 a 15 s: apagada
    - 15 a 20 s: encendida
    - luego apagada
    """

    f_on = np.array([0.5, 0.0, 0.0])  # N, empieza pequeño
    f_off = np.array([0.0, 0.0, 0.0])

    t0 = time.time()
    last_state = None

    while not stop_event.is_set():
        t = time.time() - t0

        if 5.0 <= t < 10.0:
            f = f_on
            state = "ON_1"
        elif 15.0 <= t < 20.0:
            f = f_on
            state = "ON_2"
        else:
            f = f_off
            state = "OFF"

        set_virtual_force(cf, f)

        if state != last_state:
            print(f"Fvirt {state}: [{f[0]:.3f}, {f[1]:.3f}, {f[2]:.3f}] N")
            last_state = state

        time.sleep(0.05)

    set_virtual_force(cf, f_off)


# =========================
# LOGGING + FORCE PIPELINE
# =========================
def log_callback(timestamp, data, logconf, pwm_model, force_observer, force_filter):
    pwm = data.get('propForce.totalPwm', None)
    vbat = data.get('pm.vbat', None)

    if pwm is None or vbat is None:
        return

    force_est = pwm_model.pwm_to_force(pwm, vbat)
    latest_state["force_thrust"] = force_est

    a = latest_state["a"]
    R = latest_state["R"]

    if a is not None and R is not None:
        f_ext = force_observer.observe(force_est, a, R)
        f_ext_filt = force_filter.filter(f_ext)

        latest_state["force_external"] = f_ext_filt

        if timestamp % 100 == 0:
            print(
                f"[{timestamp}] "
                f"F_raw: [{f_ext[0]:.2f}, {f_ext[1]:.2f}, {f_ext[2]:.2f}] | "
                f"F_filt: [{f_ext_filt[0]:.2f}, {f_ext_filt[1]:.2f}, {f_ext_filt[2]:.2f}]"
            )
        if (timestamp//10) % 10 == 0:
            print(
                f"[{timestamp}] "
                f"PWM: {pwm} | "
                f"Vbat: {vbat:.2f} | "
                f"Force Est(N): {force_est:.3f} | "
                f"External Force Est(N): [{f_ext[0]:.3f}, {f_ext[1]:.3f}, {f_ext[2]:.3f}] N | "
                f"External Force Est Filtrada(N): [{f_ext_filt[0]:.3f}, {f_ext_filt[1]:.3f}, {f_ext_filt[2]:.3f}] N"
            )


def start_logging(cf, pwm_model, force_observer, force_filter):
    log_conf = LogConfig(name='PropForces', period_in_ms=10)

    log_conf.add_variable('propForce.totalPwm', 'uint32_t')
    log_conf.add_variable('pm.vbat', 'float')

    cf.log.add_config(log_conf)

    if log_conf.valid:
        log_conf.data_received_cb.add_callback(
            lambda t, d, l: log_callback(
                t, d, l, pwm_model, force_observer, force_filter
            )
        )
        log_conf.start()
        print("Logging started...")
        return log_conf

    print("Log config invalid")
    return None


# =========================
# CONTROL
# =========================
def takeoff(cf, height=0.5, duration=2.0):
    steps = int(duration / 0.02)

    for i in range(steps):
        if stop_event.is_set():
            break

        z = height * (i / steps)
        cf.commander.send_position_setpoint(0.0, 0.0, float(z), 0.0)
        time.sleep(0.02)


def land(cf, height=0.5, duration=2.0):
    steps = int(duration / 0.02)

    for i in range(steps):
        z = height * (1 - i / steps)
        cf.commander.send_position_setpoint(0.0, 0.0, float(z), 0.0)
        time.sleep(0.02)

    cf.commander.send_stop_setpoint()


def run_trajectory(cf):
    print("Iniciando trayectoria...")

    path = PathGenerator(dt=0.01)

    takeoff(cf, height=0.5, duration=2.0)

    t0 = time.time()

    try:
        while not stop_event.is_set():
            t = time.time() - t0

            # dp, _, _ = path.do_linear_z(
            #     t,
            #     center=np.array([0.0, 0.0, 0.5]),
            #     z_speed=0.1
            # )

            # Alternativas:
            dp, _, _ = path.fixed_point([0.0, 0.0, 1.5])
            # dp, _, _ = path.do_sine_x(t, center=[0, 0, 0.5], amplitude=0.3, frequency=0.2)

            dp = np.array(dp, dtype=float).flatten()
            x, y, z = map(float, dp[:3])

            cf.commander.send_position_setpoint(x, y, z, 0.0)

            time.sleep(path.dt)

    except KeyboardInterrupt:
        print("Interrumpido")

    finally:
        print("Aterrizando...")

        # Apagar fuerza virtual antes de aterrizar
        set_virtual_force(cf, np.array([0.0, 0.0, 0.0]))

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

        cf.param.set_value('stabilizer.estimator', '2')
        cf.param.set_value('stabilizer.controller', '2')
        cf.param.set_value('commander.enHighLevel', '0')

        # Asegura fuerza virtual inicial cero
        set_virtual_force(cf, np.array([0.0, 0.0, 0.0]))

        # opcional:
        # cf.param.set_value('ctrlMel.mass', str(DRONE_MASS))

        time.sleep(0.5)

        print("Armando...")
        cf.platform.send_arming_request(True)
        time.sleep(1.0)

        print("Controlador:", cf.param.get_value('stabilizer.controller'))
        print("Estimador:", cf.param.get_value('stabilizer.estimator'))
        print("Fvirt x:", cf.param.get_value('ctrlMel.fvirt_x'))
        print("Fvirt y:", cf.param.get_value('ctrlMel.fvirt_y'))
        print("Fvirt z:", cf.param.get_value('ctrlMel.fvirt_z'))

        estimator = FullStateEstimator(dt=0.01, alpha=0.2)
        pwm_model = PwMToForceThrust(drone_mass=DRONE_MASS, gravity=GRAVITY)
        force_observer = ForceObserver(drone_mass=DRONE_MASS, gravity=GRAVITY)
        force_filter = MovingAverageForgettingFactorFilter(
            window_size=5,
            forgetting_factor=0.8
        )

        extpos_thread = threading.Thread(
            target=send_extpos_loop,
            args=(cf, estimator),
            daemon=True
        )
        extpos_thread.start()

        fvirt_thread = threading.Thread(
            target=virtual_force_loop,
            args=(cf,),
            daemon=True
        )
        fvirt_thread.start()

        log_conf = start_logging(
            cf,
            pwm_model,
            force_observer,
            force_filter
        )

        try:
            run_trajectory(cf)

        except KeyboardInterrupt:
            print("Ctrl+C detectado")

        finally:
            print("Cerrando...")

            stop_event.set()
            time.sleep(0.5)

            set_virtual_force(cf, np.array([0.0, 0.0, 0.0]))

            if log_conf is not None:
                log_conf.stop()

            cf.commander.send_stop_setpoint()
            time.sleep(0.1)

            cf.platform.send_arming_request(False)
            time.sleep(0.2)

            extpos_thread.join(timeout=1)
            fvirt_thread.join(timeout=1)

            print("Finished")