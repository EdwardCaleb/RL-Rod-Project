import time
import threading
import logging

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig

from optitrack_natnet_main import tracker

URI = 'radio://0/90/2M/E7E7E7E701'
robot_id = 536

logging.basicConfig(level=logging.ERROR)

# =========================
# OPTITRACK
# =========================
tracker = tracker.OptiTrackClient(
    client_address="192.168.0.99",
    server_address="192.168.0.4"
)
tracker.start()

# =========================
# LOGGING CALLBACK
# =========================
def log_callback(timestamp, data, logconf):
    total_pwm = data.get('propForce.totalPwm', None)
    total_uncapped = data.get('propForce.totalUncapped', None)
    total_si = data.get('propForce.totalSi', None)
    vbat = data.get('pm.vbat', None)

    print(f"[{timestamp}] PWM: {total_pwm} | Uncapped: {total_uncapped} | Force(N): {total_si} | Vbat: {vbat}")


# =========================
# EXTPOS LOOP
# =========================
def send_extpos_loop(cf):
    while True:
        pos, quat = tracker.get_full_pose(robot_id)

        if pos is not None:
            x, y, z = pos
            cf.extpos.send_extpos(x, y, z)

        time.sleep(0.01)  # 100 Hz


# =========================
# LOGGING THREAD
# =========================
def start_logging(cf):
    log_conf = LogConfig(name='PropForces', period_in_ms=10)  # 100 Hz

    log_conf.add_variable('propForce.totalPwm', 'uint32_t')
    log_conf.add_variable('propForce.totalUncapped', 'uint32_t')
    log_conf.add_variable('propForce.totalSi', 'float')
    log_conf.add_variable('pm.vbat', 'float')

    cf.log.add_config(log_conf)

    if log_conf.valid:
        log_conf.data_received_cb.add_callback(log_callback)
        log_conf.start()
        print("Logging started...")
    else:
        print("Log config invalid")


# =========================
# FLIGHT SEQUENCE
# =========================
def run_sequence(cf):
    cf.param.set_value('stabilizer.estimator', '2')
    cf.param.set_value('commander.enHighLevel', '1')

    commander = cf.high_level_commander

    time.sleep(2)

    commander.takeoff(1.5, 2.0)
    time.sleep(3)

    commander.go_to(1.0, 0.0, 0.5, 0, 3.0)
    time.sleep(4)

    commander.go_to(0.0, 0.0, 0.5, 0, 3.0)
    time.sleep(4)

    commander.land(0.0, 2.0)
    time.sleep(3)

    commander.stop()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:

        cf = scf.cf

        # 🔹 Hilo OptiTrack → Crazyflie
        threading.Thread(target=send_extpos_loop, args=(cf,), daemon=True).start()

        # 🔹 Iniciar logging
        start_logging(cf)

        # 🔹 Ejecutar vuelo
        run_sequence(cf)

        print("Finished")