import time
import threading
import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie

from optitrack_natnet_main import tracker

URI = 'radio://0/90/2M/E7E7E7E701'
robot_id = 536

# iniciar tracker
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


def run_sequence(cf):
    cf.param.set_value('stabilizer.estimator', '2')
    cf.param.set_value('commander.enHighLevel', '1')

    commander = cf.high_level_commander

    time.sleep(2)

    commander.takeoff(1.5, 2.0) # altura, tiempo para despegar
    time.sleep(3)

    commander.go_to(1.0, 0.0, 0.5, 0, 3.0) # x, y, z, yaw, tiempo para llegar
    time.sleep(4)

    commander.go_to(0.0, 0.0, 0.5, 0, 3.0) # x, y, z, yaw, tiempo para llegar
    time.sleep(4)

    commander.land(0.0, 2.0) # altura, tiempo para aterrizary
    time.sleep(3)

    commander.stop()


if __name__ == "__main__":
    cflib.crtp.init_drivers()

    with SyncCrazyflie(URI, cf=Crazyflie()) as scf:

        # hilo para posición externa
        threading.Thread(target=send_extpos_loop, args=(scf.cf,), daemon=True).start()

        run_sequence(scf.cf)





