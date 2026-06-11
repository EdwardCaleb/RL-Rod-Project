
'''
En este test:
- Leemos el thrust en PWM y voltaje desde el Crazyflie
- Estimamos la fuerza de thrust en newtons usando un modelo PWM a fuerza
- Estimamos la fuerza exogena aplicada al drone en x, y, z usando un observador de fuerzas
- Filtramos la fuerza exogena estimada para obtener una versión más suave y menos ruidosa
- Incluimos la adicion de una fuerza exogena manual (empujón) durante la trayectoria para observar su efecto en la estimación de fuerza y su filtrado
- Compensamos la trayectoria del drone en tiempo real para contrarrestar la fuerza exogena aplicada, usando la fuerza exogena filtrada como referencia para el empuje compensatorio
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
    "force_external_raw": None,
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
            q = np.array(quat, dtype=float)  # OptiTrack: x, y, z, w

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
# VIRTUAL FORCE
# =========================
def has_param(cf, name):
    try:
        cf.param.get_value(name)
        return True
    except KeyError:
        return False


def set_virtual_force(cf, f, verbose=False):
    params = [
        ('ctrlMel.fvirt_x', float(f[0])),
        ('ctrlMel.fvirt_y', float(f[1])),
        ('ctrlMel.fvirt_z', float(f[2])),
    ]

    for name, value in params:
        try:
            cf.param.set_value(name, str(value))
        except KeyError:
            if verbose:
                print(f"WARNING: {name} no existe en param TOC.")
            return False
        except Exception as e:
            if verbose:
                print(f"WARNING: no pude enviar {name}: {e}")
            return False

    return True


def limit_vector_norm(v, max_norm):
    norm = np.linalg.norm(v)

    if norm > max_norm:
        return v * (max_norm / norm)

    return v


def external_force_feedback_loop(
    cf,
    gain=0.3,
    max_force=0.15,
    deadband=0.05,
    rate_hz=10,
    start_delay=5.0
):
    """
    Compensación activa:
        fvirt = -gain * F_ext_filtered

    La fuerza virtual se suma en Mellinger en coordenadas mundo.
    """

    required_params = [
        'ctrlMel.fvirt_x',
        'ctrlMel.fvirt_y',
        'ctrlMel.fvirt_z',
    ]

    if not all(has_param(cf, p) for p in required_params):
        print("WARNING: ctrlMel.fvirt_* no disponible. Feedback desactivado.")
        return

    print("External force feedback loop iniciado")

    t0 = time.time()
    dt = 1.0 / rate_hz
    last_print = 0.0

    while not stop_event.is_set():
        t = time.time() - t0

        if t < start_delay:
            f_cmd = np.zeros(3)
        else:
            f_ext = latest_state["force_external"]

            if f_ext is None:
                f_cmd = np.zeros(3)
            else:
                f_ext = np.array(f_ext, dtype=float).flatten()

                if np.linalg.norm(f_ext) < deadband:
                    f_cmd = np.zeros(3)
                else:
                    # hacer negativo al valor z del F_cmd
                    # f_ext[2] = -f_ext[2]

                    f_cmd = -gain * f_ext
                    f_cmd = limit_vector_norm(f_cmd, max_force)

        ok = set_virtual_force(cf, f_cmd, verbose=True)

        if not ok:
            print("WARNING: se perdió acceso a ctrlMel.fvirt_*. Feedback detenido.")
            break

        if t - last_print > 0.5:
            print(
                f"Fvirt feedback: "
                f"[{f_cmd[0]:.3f}, {f_cmd[1]:.3f}, {f_cmd[2]:.3f}] N"
            )
            last_print = t

        time.sleep(dt)

    set_virtual_force(cf, np.zeros(3), verbose=False)


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
        f_ext_raw = force_observer.observe(force_est, a, R)
        f_ext_filt = force_filter.filter(f_ext_raw)

        latest_state["force_external_raw"] = f_ext_raw
        latest_state["force_external"] = f_ext_filt

        if timestamp % 100 == 0:
            print(
                f"[{timestamp}] "
                f"F_raw: [{f_ext_raw[0]:.2f}, {f_ext_raw[1]:.2f}, {f_ext_raw[2]:.2f}] | "
                f"F_filt: [{f_ext_filt[0]:.2f}, {f_ext_filt[1]:.2f}, {f_ext_filt[2]:.2f}]"
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
            dp, _, _ = path.fixed_point([0.0, 0.0, 0.5])
            # dp, _, _ = path.do_sine_x(t, center=[0, 0, 0.5], amplitude=0.3, frequency=0.2)

            dp = np.array(dp, dtype=float).flatten()
            x, y, z = map(float, dp[:3])

            cf.commander.send_position_setpoint(x, y, z, 0.0)

            time.sleep(path.dt)

    except KeyboardInterrupt:
        print("Interrumpido")

    finally:
        print("Aterrizando...")

        set_virtual_force(cf, np.zeros(3), verbose=True)

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

        time.sleep(0.5)

        print("Fvirt disponible x:", has_param(cf, 'ctrlMel.fvirt_x'))
        print("Fvirt disponible y:", has_param(cf, 'ctrlMel.fvirt_y'))
        print("Fvirt disponible z:", has_param(cf, 'ctrlMel.fvirt_z'))

        set_virtual_force(cf, np.zeros(3), verbose=True)

        # Opcional:
        # cf.param.set_value('ctrlMel.mass', str(DRONE_MASS))

        print("Armando...")
        cf.platform.send_arming_request(True)
        time.sleep(1.0)

        print("Controlador:", cf.param.get_value('stabilizer.controller'))
        print("Estimador:", cf.param.get_value('stabilizer.estimator'))

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

        feedback_thread = threading.Thread(
            target=external_force_feedback_loop,
            args=(cf,),
            kwargs={
                "gain": 0.1,
                "max_force": 0.15,
                "deadband": 0.005,
                "rate_hz": 5,
                "start_delay": 5.0,
            },
            daemon=True
        )
        feedback_thread.start()

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

            set_virtual_force(cf, np.zeros(3), verbose=True)

            if log_conf is not None:
                log_conf.stop()

            cf.commander.send_stop_setpoint()
            time.sleep(0.1)

            cf.platform.send_arming_request(False)
            time.sleep(0.2)

            extpos_thread.join(timeout=1)
            feedback_thread.join(timeout=1)

            print("Finished")