import logging
import time

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.log import LogConfig
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

URI = 'radio://0/90/2M/E7E7E7E701'  # cambia si usas otro canal

logging.basicConfig(level=logging.ERROR)

def log_callback(timestamp, data, logconf):
    total_pwm = data.get('propForce.totalPwm', None)
    total_uncapped = data.get('propForce.totalUncapped', None)
    total_si = data.get('propForce.totalSi', None)

    print(f"[{timestamp}] PWM: {total_pwm} | Uncapped: {total_uncapped} | Force(N): {total_si}")


def main():
    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache')) as scf:
        cf = scf.cf

        # Configuración del log (100 Hz → 10 ms)
        log_conf = LogConfig(name='PropForces', period_in_ms=10)

        log_conf.add_variable('propForce.totalPwm', 'uint32_t')
        log_conf.add_variable('propForce.totalUncapped', 'uint32_t')
        log_conf.add_variable('propForce.totalSi', 'float')

        cf.log.add_config(log_conf)

        if log_conf.valid:
            log_conf.data_received_cb.add_callback(log_callback)
            log_conf.start()
            print("Logging started...")

            time.sleep(10)  # lee por 10 segundos

            log_conf.stop()
        else:
            print("Log configuration not valid")

if __name__ == '__main__':
    main()