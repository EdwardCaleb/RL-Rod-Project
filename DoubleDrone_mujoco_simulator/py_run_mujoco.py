import mujoco
import mujoco.viewer
import time

# Cargar el modelo XML
model = mujoco.MjModel.from_xml_path("flexible_rod_dual_drone.xml")
data = mujoco.MjData(model)

# Crear el viewer interactivo
with mujoco.viewer.launch_passive(model, data) as viewer:
    print("Simulando hoja de papel... (cerrar ventana para terminar)")
   
    start = time.time()
    while viewer.is_running():
        step_start = time.time()
       
        # Avanza la simulación
        mujoco.mj_step(model, data)
       
        # Actualiza la visualización
        viewer.sync()
       
        # Mantiene velocidad de simulación real
        time_until_next_step = model.opt.timestep - (time.time() - step_start)
        if time_until_next_step > 0:
            time.sleep(time_until_next_step)
   
    print(f"Simulación finalizada tras {time.time() - start:.1f} segundos")


