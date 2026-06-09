import time
import cflib.crtp
from cflib.utils.power_switch import PowerSwitch

uri = "radio://0/90/2M/E7E7E7E705"

cflib.crtp.init_drivers()

ps = None

try:
    ps = PowerSwitch(uri)

    print("Reiniciando Crazyflie Bolt...")
    ps.stm_power_cycle()

    time.sleep(3)

    print("Reinicio completado")

except Exception as e:
    print(f"Error: {e}")

finally:
    if ps:
        ps.close()