# import xml.etree.ElementTree as ET
# from xml.dom import minidom

# # ==============================
# # 🔹 Funciones auxiliares
# # ==============================

# def crear_quadrotor(idx, pos, color):
#     """Crea un bloque <body> con un quadrotor completo."""
#     body = ET.Element("body", name=f"core_{idx}", pos=f"{pos[0]} {pos[1]} {pos[2]}")

#     # núcleo del dron
#     ET.SubElement(body, "geom", {
#         "name": f"core_geom_{idx}",
#         "type": "box",
#         "pos": "0 0 0",
#         "size": ".04 .04 .02",
#         "rgba": f"{color[0]} {color[1]} {color[2]} 1",
#         "mass": ".1"
#     })

#     ET.SubElement(body, "joint", {
#         "name": f"root_{idx}",
#         "type": "free",
#         "damping": "0",
#         "armature": "0",
#         "limited": "false"
#     })

#     # Brazos y hélices
#     brazos = [
#         ("front",  ".12 0 0",  ".01 0 .005", ".01 .005 .01"),
#         ("back",  "-.12 0 0", "-.01 0 .005", ".01 .005 .01"),
#         ("left",   "0 .12 0",  "0 .01 .005", ".01 .005 .01"),
#         ("right",  "0 -.12 0", "0 -.01 .005", ".01 .005 .01")
#     ]

#     for i, (nombre, pos_arm, pos_geom, size_geom) in enumerate(brazos):
#         arm = ET.SubElement(body, "body", name=f"arm_{nombre}_{idx}", pos=pos_arm)
#         ET.SubElement(arm, "geom", {
#             "type": "box",
#             "pos": pos_geom,
#             "size": size_geom,
#             "rgba": "1 .1 0 1",
#             "mass": ".02"
#         })
#         thruster = ET.SubElement(arm, "body", name=f"thruster{i}_{idx}", pos="0 0 0.015")
#         ET.SubElement(thruster, "geom", {
#             "type": "cylinder",
#             "pos": "0 0 .0025",
#             "size": ".05 .0025",
#             "rgba": ".3 1 .3 0.3",
#             "mass": ".005"
#         })
#         ET.SubElement(thruster, "site", {
#             "name": f"motor{i}_{idx}",
#             "type": "cylinder",
#             "pos": "0 0 .0025",
#             "size": ".01 .0025",
#             "rgba": ".3 1 .3 0.3"
#         })

#     # Puntos de unión (para tendones)
#     ET.SubElement(body, "site", {"name": f"attach_{'left' if idx==1 else 'right'}", "pos": "0 0 -0.03"})

#     return body


# def crear_barra(num_segmentos=5, longitud_total=1.0):
#     """Crea una barra articulada con num_segmentos y longitud_total."""
#     longitud_seg = longitud_total / num_segmentos
#     bar = ET.Element("body", name="bar_seg1", pos="-0.5 0 0.1")
#     ET.SubElement(bar, "joint", {"name": "bar_free", "type": "free", "damping": "0.0005"})
#     ET.SubElement(bar, "geom", {
#         "type": "capsule",
#         "fromto": f"0 0 0 {longitud_seg} 0 0",
#         "size": "0.005",
#         "rgba": "0.3 0.9 0.3 1",
#         "mass": "0.05"
#     })
#     ET.SubElement(bar, "site", {"name": "bar_left", "pos": "0 0 0"})

#     parent = bar
#     for i in range(2, num_segmentos + 1):
#         child = ET.SubElement(parent, "body", name=f"bar_seg{i}", pos=f"{longitud_seg} 0 0")
#         ET.SubElement(child, "joint", {
#             "type": "hinge",
#             "axis": "0 1 0",
#             "range": "-30 30",
#             "damping": "0.0005",
#             "stiffness": "0.001"
#         })
#         ET.SubElement(child, "geom", {
#             "type": "capsule",
#             "fromto": f"0 0 0 {longitud_seg} 0 0",
#             "size": "0.005",
#             "rgba": f"0.3 {0.9 - 0.1*i:.1f} {0.3 + 0.1*i:.1f} 1",
#             "mass": "0.01"
#         })
#         parent = child

#     ET.SubElement(parent, "site", {"name": "bar_right", "pos": f"{longitud_seg} 0 0"})
#     return bar


# def crear_tendones(num_drones, stiffness=20, damping=3):
#     """Genera tendones entre los drones y la barra."""
#     tendon = ET.Element("tendon")
#     if num_drones >= 1:
#         ET.SubElement(tendon, "spatial", {
#             "name": "tendon_left",
#             "limited": "true",
#             "range": "0.01 0.03",
#             "springlength": "0.1",
#             "stiffness": str(stiffness),
#             "damping": str(damping),
#             "rgba": "0.2 1 0.2 0.5",
#             "width": "0.005"
#         })
#         ET.SubElement(tendon[-1], "site", {"site": "attach_left"})
#         ET.SubElement(tendon[-1], "site", {"site": "bar_left"})

#     if num_drones >= 2:
#         ET.SubElement(tendon, "spatial", {
#             "name": "tendon_right",
#             "limited": "true",
#             "range": "0.01 0.03",
#             "springlength": "0.25",
#             "stiffness": str(stiffness),
#             "damping": str(damping),
#             "rgba": "0.2 1 0.2 0.5",
#             "width": "0.005"
#         })
#         ET.SubElement(tendon[-1], "site", {"site": "attach_right"})
#         ET.SubElement(tendon[-1], "site", {"site": "bar_right"})

#     return tendon


# def crear_actuadores(num_drones):
#     """Crea motores para cada quadrotor."""
#     actuator = ET.Element("actuator")
#     for idx in range(1, num_drones + 1):
#         for i in range(4):
#             ET.SubElement(actuator, "motor", {
#                 "site": f"motor{i}_{idx}",
#                 "ctrlrange": "0 2",
#                 "gear": f"0 0 1 0 0 {(-1)**i * 0.1}"
#             })
#     return actuator


# def crear_sensores(num_drones):
#     """Crea sensores de posición y orientación para cada dron."""
#     sensor = ET.Element("sensor")
#     for idx in range(1, num_drones + 1):
#         ET.SubElement(sensor, "framepos", {"name": f"pos_core_{idx}", "objtype": "body", "objname": f"core_{idx}"})
#         ET.SubElement(sensor, "framequat", {"name": f"quat_core_{idx}", "objtype": "body", "objname": f"core_{idx}"})
#     return sensor


# # ==============================
# # 🔸 Generador principal
# # ==============================

# def generar_modelo(num_drones=2, distancia=1.0, num_segmentos=5, longitud_barra=1.0):
#     mujoco = ET.Element("mujoco", model="quadrotorplus")

#     worldbody = ET.SubElement(mujoco, "worldbody")

#     # Añadir drones
#     for i in range(num_drones):
#         x = -distancia / 2 + i * distancia
#         color = (0.8 - 0.3*i, 0.2 + 0.3*i, 0.2 + 0.3*i)
#         worldbody.append(crear_quadrotor(i+1, (x, 0, 0.2), color))

#     # Añadir barra flexible
#     worldbody.append(crear_barra(num_segmentos, longitud_barra))

#     # Tendones, actuadores y sensores
#     mujoco.append(crear_tendones(num_drones))
#     mujoco.append(crear_actuadores(num_drones))
#     mujoco.append(crear_sensores(num_drones))

#     return ET.ElementTree(mujoco)


# def get_xml(**params) -> str:
#     import xml.etree.ElementTree as ET
#     from xml.dom import minidom

#     tree = generar_modelo(**params)
#     xml_bytes = ET.tostring(tree.getroot(), encoding='utf-8')

#     parsed = minidom.parseString(xml_bytes)
#     pretty_xml = parsed.toprettyxml(indent="    ")
#     pretty_xml = "\n".join([line for line in pretty_xml.splitlines() if line.strip()])

#     # ❌ Quitamos la línea con <?xml ...?>
#     pretty_xml = "\n".join(line for line in pretty_xml.splitlines() if not line.startswith("<?xml"))

#     # ✅ Dejamos el string directamente (sin encabezado)
#     xml_string = pretty_xml
#     return xml_string





# =====================================
# 🔹 1. Bloques estáticos del modelo
# =====================================

XML_HEADER = r"""
<mujoco model="quadrotorplus">
    <compiler angle="degree" inertiafromgeom="true" coordinate="local"/>
    <option	timestep="0.01" gravity="0 0 -9.81" density="1" viscosity="1e-5" integrator="RK4"/>

    <default>
        <joint armature="1" damping="1" limited="true"/>
        <geom conaffinity="0" condim="3" density="5.0" friction="1 0.5 0.5" margin="0.01" rgba="0.8 0.6 0.4 1"/>
    </default>

    <asset>
        <texture builtin="gradient" type="skybox" height="100" width="100" rgb1="1 1 1" rgb2="0 0 0"/>
        <texture name="texgeom" builtin="flat" height="1278" mark="cross" markrgb="1 1 1" random="0.01" rgb1="0.8 0.6 0.4" rgb2="0.8 0.6 0.4" type="cube" width="127"/>
        <texture name="texplane" builtin="checker" height="100" width="100" rgb1="0 0 0" rgb2="0.8 0.8 0.8" type="2d"/>
        <material name="MatPlane" reflectance="0.5" shininess="1" specular="1" texrepeat="60 60" texture="texplane"/>
        <material name="geom" texture="texgeom" texuniform="true"/>
    </asset>

    <worldbody>
        <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
        <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.4 0.3 0.3 1" size="5 5 0.2" type="plane"/>
"""

XML_DRONES = r"""
        <!-- FIRST QUADROTOR -->
        <body name="core_1" pos="-0.5 0 0.2">
            <geom name="core_geom_1" type="box" pos="0 0 0" quat="1 0 0 0" size=".04 .04 .02" rgba=".8 .2 0 1" mass=".1"/>
            <joint name="root_1" type="free" damping="0" armature="0" pos="0 0 0" limited="false"/>
            <!-- brazos y motores -->
            <geom name="arm_front0_1" type="box" pos=".08 0 0" size=".04 .005 .005" quat="1 0 0 0" rgba=".8 .8 .8 1" mass=".02"/>
            <geom name="arm_back0_1" type="box" pos="-.08 0 0" size=".04 .005 .005" quat="0 0 0 1" rgba=".8 .8 .8 1" mass=".02"/>
            <geom name="arm_left0_1" type="box" pos="0 .08 0" size=".04 .005 .005" quat=".707 0 0 .707" rgba=".8 .8 .8 1" mass=".02"/>
            <geom name="arm_right0_1" type="box" pos="0 -.08 0" size=".04 .005 .005" quat=".707 0 0 -.707" rgba=".8 .8 .8 1" mass=".02"/>

            <body name="arm_front1_1" pos=".12 0 0">
                <geom type="box" pos=".01 0 .005" size=".01 .005 .01" quat="1 0 0 0" rgba="1 .1 0 1" mass=".02"/>
                <body name="thruster0_1" pos="0.01 0 0.015">
                    <geom type="cylinder" pos="0 0 .0025" size=".05 .0025" quat="1 0 0 0" rgba=".3 1 .3 0.3" mass=".005"/>
                    <site name="motor0_1" type="cylinder" pos="0 0 .0025" size=".01 .0025" quat="1 0 0 0" rgba=".3 1 .3 0.3"/>
                </body>
            </body>

            <body name="arm_back1_1" pos="-.12 0 0">
                <geom type="box" pos="-.01 0 .005" size=".01 .005 .01" quat="0 0 0 1" rgba="1 .1 0 1" mass=".02"/>
                <body name="thruster1_1" pos="-0.01 0 .015">
                    <geom type="cylinder" pos="0 0 .0025" size=".05 .0025" quat="1 0 0 0" rgba=".3 1 .3 0.3" mass=".005"/>
                    <site name="motor1_1" type="cylinder" pos="0 0 .0025" size=".01 .0025" quat="1 0 0 0" rgba=".3 1 .3 0.3"/>
                </body>
            </body>

            <body name="arm_left1_1" pos="0 .12 0">
                <geom type="box" pos="0 .01 .005" size=".01 .005 .01" quat=".7071068 0 0 .7071068" rgba="1 .1 0 1" mass=".02"/>
                <body name="thruster2_1" pos="0 0.01 0.015">
                    <geom type="cylinder" pos="0 0 .0025" size=".05 .0025" quat="1 0 0 0" rgba=".3 1 .3 0.3" mass=".005"/>
                    <site name="motor2_1" type="cylinder" pos="0 0 .0025" size=".01 0.0025" quat="1 0 0 0" rgba=".3 1 .3 0.3"/>
                </body>
            </body>

            <body name="arm_right1_1" pos="0 -.12 0">
                <geom type="box" pos="0 -.01 .005" size=".01 .005 .01" quat=".7071068 0 0 -.7071068" rgba="1 .1 0 1" mass=".02"/>
                <body name="thruster3_1" pos="0 -0.01 .015">
                    <geom type="cylinder" pos="0 0 .0025" size=".05 .0025" quat="1 0 0 0" rgba=".3 1 .3 0.3" mass=".005"/>
                    <site type="cylinder" pos="0 0 .0025" size=".01 .0025" quat="1 0 0 0" rgba=".3 1 .3 0.3" name="motor3_1"/>
                </body>
            </body>

            <site name="x_axis_1" type="box" pos=".1 .0 .0" size=".15 .005 .005" quat="1 0 0 0" rgba="1 0 0 0.2"/>
            <site name="y_axis_1" type="box" pos=".0 .1 .0" size=".15 .005 .005" quat=".707 0 0 .707" rgba="0 1 0 0.2"/>
            <site name="z_axis_1" type="box" pos=".0 .0 .1" size=".15 .005 .005" quat="-.707 0 .707 0" rgba="0 0 1 0.2"/>

            <site name="attach_left" pos="0 0 -0.03"/>
        </body>
        
        <!-- SECOND QUADROTOR -->
        <body name="core_2" pos="0.5 0 0.2">
            <geom name="core_geom_2" type="box" pos="0 0 0" quat="1 0 0 0" size=".04 .04 .02" rgba="0 0.2 0.8 1" mass=".1"/>
            <joint name="root_2" type="free" damping="0" armature="0" pos="0 0 0" limited="false"/>
            <geom name="arm_front0_2" type="box" pos=".08 0 0" size=".04 .005 .005" quat="1 0 0 0" rgba=".8 .8 .8 1" mass=".02"/>
            <geom name="arm_back0_2" type="box" pos="-.08 0 0" size=".04 .005 .005" quat="0 0 0 1" rgba=".8 .8 .8 1" mass=".02"/>
            <geom name="arm_left0_2" type="box" pos="0 .08 0" size=".04 .005 .005" quat=".707 0 0 .707" rgba=".8 .8 .8 1" mass=".02"/>
            <geom name="arm_right0_2" type="box" pos="0 -.08 0" size=".04 .005 .005" quat=".707 0 0 -.707" rgba=".8 .8 .8 1" mass=".02"/>

            <body name="arm_front1_2" pos=".12 0 0">
                <geom type="box" pos=".01 0 .005" size=".01 .005 .01" quat="1 0 0 0" rgba="1 .1 0 1" mass=".02"/>
                <body name="thruster0_2" pos="0.01 0 0.015">
                    <geom type="cylinder" pos="0 0 .0025" size=".05 .0025" quat="1 0 0 0" rgba=".3 1 .3 0.3" mass=".005"/>
                    <site name="motor0_2" type="cylinder" pos="0 0 .0025" size=".01 .0025" quat="1 0 0 0" rgba=".3 .3 1 0.3"/>
                </body>
            </body>

            <body name="arm_back1_2" pos="-.12 0 0">
                <geom type="box" pos="-.01 0 .005" size=".01 .005 .01" quat="0 0 0 1" rgba="1 .1 0 1" mass=".02"/>
                <body name="thruster1_2" pos="-0.01 0 .015">
                    <geom type="cylinder" pos="0 0 .0025" size=".05 .0025" quat="1 0 0 0" rgba=".3 1 .3 0.3" mass=".005"/>
                    <site name="motor1_2" type="cylinder" pos="0 0 .0025" size=".01 .0025" quat="1 0 0 0" rgba=".3 .3 1 0.3"/>
                </body>
            </body>

            <body name="arm_left1_2" pos="0 .12 0">
                <geom type="box" pos="0 .01 .005" size=".01 .005 .01" quat=".7071068 0 0 .7071068" rgba="1 .1 0 1" mass=".02"/>
                <body name="thruster2_2" pos="0 0.01 0.015">
                    <geom type="cylinder" pos="0 0 .0025" size=".05 .0025" quat="1 0 0 0" rgba=".3 1 .3 0.3" mass=".005"/>
                    <site name="motor2_2" type="cylinder" pos="0 0 .0025" size=".01 0.0025" quat="1 0 0 0" rgba=".3 .3 1 0.3"/>
                </body>
            </body>

            <body name="arm_right1_2" pos="0 -.12 0">
                <geom type="box" pos="0 -.01 .005" size=".01 .005 .01" quat=".7071068 0 0 -.7071068" rgba="1 .1 0 1" mass=".02"/>
                <body name="thruster3_2" pos="0 -0.01 .015">
                    <geom type="cylinder" pos="0 0 .0025" size=".05 .0025" quat="1 0 0 0" rgba=".3 1 .3 0.3" mass=".005"/>
                    <site type="cylinder" pos="0 0 .0025" size=".01 .0025" quat="1 0 0 0" rgba=".3 .3 1 0.3" name="motor3_2"/>
                </body>
            </body>

            <site name="x_axis_2" type="box" pos=".1 .0 .0" size=".15 .005 .005" quat="1 0 0 0" rgba="1 0 0 0.2"/>
            <site name="y_axis_2" type="box" pos=".0 .1 .0" size=".15 .005 .005" quat=".707 0 0 .707" rgba="0 1 0 0.2"/>
            <site name="z_axis_2" type="box" pos=".0 .0 .1" size=".15 .005 .005" quat="-.707 0 .707 0" rgba="0 0 1 0.2"/>

            <site name="attach_right" pos="0 0 -0.03"/>
        </body>
"""

XML_FOOTER = r"""
    </worldbody>

    <!-- 🌈 Tendones elásticos que unen drones y barra -->
    <tendon>
        <spatial name="tendon_left"
                 limited="true" range="0.01 0.03"
                 springlength="0.1" stiffness="20" damping="3"
                 rgba="0.2 1 0.2 0.5" width="0.005">
            <site site="attach_left"/>
            <site site="bar_left"/>
        </spatial>

        <spatial name="tendon_right"
                 limited="true" range="0.01 0.03"
                 springlength="0.25" stiffness="20" damping="3"
                 rgba="0.2 1 0.2 0.5" width="0.005">
            <site site="attach_right"/>
            <site site="bar_right"/>
        </spatial>
    </tendon>

    <actuator>
        <!-- actuadores del primer dron -->
        <motor site="motor0_1" ctrlrange="0 2" gear="0  0. 1. 0. 0. -0.1"/>
        <motor site="motor1_1" ctrlrange="0 2" gear="0  0. 1. 0. 0.  0.1"/>
        <motor site="motor2_1" ctrlrange="0 2" gear="0  0. 1. 0. 0. -0.1"/>
        <motor site="motor3_1" ctrlrange="0 2" gear="0  0. 1. 0. 0.  0.1"/>

        <!-- actuadores del segundo dron -->
        <motor site="motor0_2" ctrlrange="0 2" gear="0  0. 1. 0. 0. -0.1"/>
        <motor site="motor1_2" ctrlrange="0 2" gear="0  0. 1. 0. 0.  0.1"/>
        <motor site="motor2_2" ctrlrange="0 2" gear="0  0. 1. 0. 0. -0.1"/>
        <motor site="motor3_2" ctrlrange="0 2" gear="0  0. 1. 0. 0.  0.1"/>
    </actuator>

    <sensor>
        <!-- Primer dron -->
        <framepos  name="pos_core_1"  objtype="body" objname="core_1"/>
        <framequat name="quat_core_1" objtype="body" objname="core_1"/>

        <!-- Segundo dron -->
        <framepos  name="pos_core_2"  objtype="body" objname="core_2"/>
        <framequat name="quat_core_2" objtype="body" objname="core_2"/>
    </sensor>

</mujoco>
"""




def generar_barra(num_segmentos=5, longitud_total=1.0, damping=0.0005, stiffness=0.001, mass=0.2) -> str:
    longitud_seg = longitud_total / num_segmentos
    xml = []
    xml.append(f'        <!-- 🔗 Barra flexible con {num_segmentos} segmentos articulados -->')
    xml.append(f'        <body name="bar_seg1" pos="-0.500 0 0.1">')
    xml.append(f'            <joint name="bar_free" type="free" damping="{damping:.3f}" stiffness="{stiffness:.3f}"/>')
    xml.append(f'            <geom type="capsule" fromto="0 0 0 {longitud_seg:.3f} 0 0" size="0.005" rgba="0.3 0.9 0.3 1" mass="{mass/num_segmentos:.3f}"/>')
    xml.append(f'            <site name="bar_left" pos="0 0 0"/>')

    # Crear segmentos internos
    for i in range(2, num_segmentos + 1):
        color_g = 0.9 - (i * 0.1)
        color_b = 0.3 + (i * 0.1)
        xml.append(f'            <body name="bar_seg{i}" pos="{longitud_seg:.3f} 0 0">')
        xml.append(f'                <joint type="hinge" axis="0 1 0" range="-30 30" damping="0.0005" stiffness="0.001"/>')
        xml.append(f'                <geom type="capsule" fromto="0 0 0 {longitud_seg:.3f} 0 0" size="0.005" rgba="0.3 {color_g:.2f} {color_b:.2f} 1" mass="{mass/num_segmentos:.3f}"/>')

    # sitio final (último segmento)
    xml.append(f'                <site name="bar_right" pos="{longitud_seg:.3f} 0 0"/>')

    # 🔴 Cierre correcto: cerrar SOLO los cuerpos abiertos
    xml.extend(["            </body>" for _ in range(num_segmentos - 1)])  # antes: num_segmentos
    xml.append("        </body>")  # cerrar bar_seg1

    return "\n".join(xml)



def get_xml(num_segmentos=5, longitud_total=1.0) -> str:
    barra_xml = generar_barra(num_segmentos, longitud_total, damping=0.05, stiffness=0.1, mass=0.01)
    XML = XML_HEADER + XML_DRONES + barra_xml + XML_FOOTER
    return XML


# if __name__ == "__main__":
#     xml = get_xml(num_segmentos=6, longitud_total=1.2)
#     print(xml)  # para ver el inicio del XML