import mujoco as mj
from mujoco.glfw import glfw
import numpy as np
import os
import time

from controllers.quadrotor_controller_pid import QuadrotorControllerPID
from models.cantilever_2d_drone import get_xml



# --- Carga del modelo XML ---

if True:  # Cambiar a False para cargar desde archivo XML
    XML = get_xml(num_segmentos=10, longitud_total=0.7)
    model = mj.MjModel.from_xml_string(XML)  # MuJoCo model
else:
    xml_path = "simplified/models/cantilever_2d_drone.xml"  # nombre del archivo XML
    xml_path = os.path.abspath(xml_path)
    model = mj.MjModel.from_xml_path(xml_path)

data = mj.MjData(model)




# --- Configuración básica ---
simend = 50                      # tiempo total de simulación
print_camera_config = 0          # imprimir configuración de cámara

# --- Variables globales de interacción ---
button_left = button_middle = button_right = False
lastx = lasty = 0

# --- Funciones de control ---
def init_controller(model, data):
    pass


def keyboard(window, key, scancode, act, mods):
    if act == glfw.PRESS and key == glfw.KEY_BACKSPACE:
        mj.mj_resetData(model, data)
        mj.mj_forward(model, data)

def mouse_button(window, button, act, mods):
    global button_left, button_middle, button_right
    button_left = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS)
    button_middle = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_MIDDLE) == glfw.PRESS)
    button_right = (glfw.get_mouse_button(window, glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS)
    glfw.get_cursor_pos(window)

def mouse_move(window, xpos, ypos):
    global lastx, lasty, button_left, button_middle, button_right
    dx, dy = xpos - lastx, ypos - lasty
    lastx, lasty = xpos, ypos
    if not (button_left or button_middle or button_right): return

    width, height = glfw.get_window_size(window)
    mod_shift = glfw.get_key(window, glfw.KEY_LEFT_SHIFT) == glfw.PRESS or glfw.get_key(window, glfw.KEY_RIGHT_SHIFT) == glfw.PRESS

    if button_right:
        action = mj.mjtMouse.mjMOUSE_MOVE_H if mod_shift else mj.mjtMouse.mjMOUSE_MOVE_V
    elif button_left:
        action = mj.mjtMouse.mjMOUSE_ROTATE_H if mod_shift else mj.mjtMouse.mjMOUSE_ROTATE_V
    else:
        action = mj.mjtMouse.mjMOUSE_ZOOM

    mj.mjv_moveCamera(model, action, dx/height, dy/height, scene, cam)

def scroll(window, xoffset, yoffset):
    mj.mjv_moveCamera(model, mj.mjtMouse.mjMOUSE_ZOOM, 0.0, -0.05*yoffset, scene, cam)



# --- Inicialización de ventana y render ---
glfw.init()
window = glfw.create_window(1200, 900, "Quadrotor Simulation", None, None)
glfw.make_context_current(window)
glfw.swap_interval(1)

cam = mj.MjvCamera()
opt = mj.MjvOption()
mj.mjv_defaultCamera(cam)
mj.mjv_defaultOption(opt)
scene = mj.MjvScene(model, maxgeom=10000)
context = mj.MjrContext(model, mj.mjtFontScale.mjFONTSCALE_150.value)

# --- Callbacks ---
glfw.set_key_callback(window, keyboard)
glfw.set_cursor_pos_callback(window, mouse_move)
glfw.set_mouse_button_callback(window, mouse_button)
glfw.set_scroll_callback(window, scroll)


#--------------------


# --- Controladores ---
controller_pid = QuadrotorControllerPID(kp=8.0, ki=0.8, kd=3.0)
controller_pid.z_target = 1.0

# --- Control callback ---
def controller(model, data):
    thrusts = controller_pid.update(model, data)
    data.ctrl[0] = thrusts[0]
    data.ctrl[1] = thrusts[1]
    # print("Thrusts:", data.ctrl[0], data.ctrl[1])



init_controller(model, data)
mj.set_mjcb_control(controller)


# --- Bucle principal ---
while not glfw.window_should_close(window):
    time_prev = data.time
    while (data.time - time_prev < 1.0/60.0):
        mj.mj_step(model, data)

    if data.time >= simend:
        break
    
    viewport_width, viewport_height = glfw.get_framebuffer_size(window)
    viewport = mj.MjrRect(0, 0, viewport_width, viewport_height)

    if print_camera_config:
        print("cam.azimuth =", cam.azimuth, "; cam.elevation =", cam.elevation, "; cam.distance =", cam.distance)
        print("cam.lookat =", cam.lookat)

    mj.mjv_updateScene(model, data, opt, None, cam, mj.mjtCatBit.mjCAT_ALL.value, scene)
    mj.mjr_render(viewport, scene, context)

    # time.sleep(1/10)   # ralentizar simulacion

    glfw.swap_buffers(window)
    glfw.poll_events()

glfw.terminate()
