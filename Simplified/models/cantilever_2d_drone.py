# =====================================
# 🔹 1. Bloques estáticos del modelo
# =====================================

XML_HEADER = r"""
<mujoco model="cantilever_2d_drone">
    <compiler angle="degree" inertiafromgeom="true" coordinate="local"/>
    <option timestep="0.01" gravity="0 0 -9.81" density="1" viscosity="1e-5" integrator="RK4"/>

    <default>
        <joint damping="0.0005" stiffness="0.001" limited="true"/>
        <geom conaffinity="0" condim="3" density="500" friction="1 0.5 0.5" margin="0.01" rgba="0.8 0.6 0.4 1"/>
    </default>

    <asset>
        <texture builtin="gradient" type="skybox" height="100" width="100" rgb1="1 1 1" rgb2="0.6 0.8 1"/>
        <texture name="texplane" builtin="checker" height="100" width="100" rgb1="0.7 0.7 0.7" rgb2="0.9 0.9 0.9" type="2d"/>
        <material name="MatPlane" reflectance="0.3" shininess="1" specular="0.5" texrepeat="5 5" texture="texplane"/>
        <material name="MatBeam" specular="0.3" shininess="0.5" rgba="0.3 0.7 0.3 1"/>
        <material name="MatDrone" specular="0.5" shininess="1" rgba="0.8 0.2 0.2 1"/>
    </asset>

    <worldbody>
        <light cutoff="100" diffuse="1 1 1" dir="-0 0 -1.3" directional="true" exponent="1" pos="0 0 1.3" specular=".1 .1 .1"/>
        <geom conaffinity="1" condim="3" material="MatPlane" name="floor" pos="0 0 0" rgba="0.4 0.3 0.3 1" size="5 5 0.2" type="plane"/>
        <geom name="wall" type="box" size="0.05 0.05 0.15" pos="0 0 0.25" rgba="0.2 0.3 0.8 1"/>

        <!-- ⚙️ Base fija de la viga -->
        <body name="beam_base" pos="0.05 0 1.0">
            <site name="base_site" pos="0 0 0" size="0.01" rgba="1 0 0 1"/>
"""

# Dron centrado respecto al extremo de la barra
XML_DRONE_2D = r"""
                        <!-- 🚁 Dron 2D fijo al extremo libre -->
                        <body name="drone_2d" pos="0 0 0">
                            <geom name="drone_body" type="box" size="0.08 0.01 0.01" material="MatDrone" mass="0.15"/>
                            <!-- brazos -->
                            <geom name="arm_left"  type="box" pos="-0.08 0 0.02" size="0.05 0.005 0.005" rgba="0.8 0.8 0.8 1"/>
                            <geom name="arm_right" type="box" pos=" 0.08 0 0.02" size="0.05 0.005 0.005" rgba="0.8 0.8 0.8 1"/>
                            <!-- hélices -->
                            <body name="prop_left" pos="-0.13 0 0.03">
                                <geom type="cylinder" size="0.02 0.003" rgba="0.3 1 0.3 0.6"/>
                                <site name="motor_left" pos="0 0 0.003" size="0.005" rgba="0 1 0 0.7" type="cylinder"/>
                            </body>
                            <body name="prop_right" pos="0.13 0 0.03">
                                <geom type="cylinder" size="0.02 0.003" rgba="0.3 1 0.3 0.6"/>
                                <site name="motor_right" pos="0 0 0.003" size="0.005" rgba="0 1 0 0.7" type="cylinder"/>
                            </body>
                            <!-- ejes de referencia -->
                            <site name="x_axis" type="box" pos="0.15 0 0" size="0.15 0.002 0.002" rgba="1 0 0 0.3"/>
                            <site name="z_axis" type="box" pos="0 0 0.1" size="0.002 0.002 0.15" rgba="0 0 1 0.3"/>
                        </body>
"""

XML_FOOTER = r"""
        </body>
    </worldbody>

    <!-- ⚙️ Actuadores (2 hélices) -->
    <actuator>
        <motor site="motor_left"  ctrlrange="0 2" gear="0 0 1 0 0 -0.05"/>
        <motor site="motor_right" ctrlrange="0 2" gear="0 0 1 0 0  0.05"/>
    </actuator>

    <!-- 📡 Sensores -->
    <sensor>
        <framepos  name="pos_drone"  objtype="body" objname="drone_2d"/>
        <framequat name="quat_drone" objtype="body" objname="drone_2d"/>
        <framelinvel name="linvel_drone" objtype="body" objname="drone_2d"/>
        <frameangvel name="rotvel_drone" objtype="body" objname="drone_2d"/>
    </sensor>

    <!-- 🧩 Restricción 2D (bloquear movimiento en Y) -->
    <equality>
        <weld name="2d_constraint" active="true" body1="world" body2="beam_base" solimp="0.9 0.95 0.001" solref="0.02 1"/>
    </equality>

</mujoco>
"""

# =====================================
# 🔹 2. Generador de la barra articulada
# =====================================

def generar_barra(num_segmentos=5, longitud_total=1.0, damping=0.0005, stiffness=0.0001, mass=0.1) -> str:
    """
    Genera una viga flexible 2D con un dron centrado en el extremo libre.
    El centro del dron coincide con el extremo del último segmento.
    """
    longitud_seg = longitud_total / num_segmentos
    xml = []
    xml.append(f'            <!-- 🔗 Barra flexible con {num_segmentos} segmentos -->')
    xml.append(f'            <body name="seg1" pos="0.1 0 0">')
    xml.append(f'                <joint name="hinge1" type="hinge" axis="0 1 0" range="-15 15"/>')
    xml.append(f'                <geom type="capsule" fromto="0 0 0 {longitud_seg:.3f} 0 0" size="0.01" material="MatBeam" mass="{mass/num_segmentos:.3f}"/>')

    # generar segmentos intermedios
    for i in range(2, num_segmentos + 1):
        g = 0.9 - i * 0.1
        b = 0.3 + i * 0.1
        xml.append(f'                <body name="seg{i}" pos="{longitud_seg:.3f} 0 0">')
        xml.append(f'                    <joint name="hinge{i}" type="hinge" axis="0 1 0" range="-15 15" damping="{damping}" stiffness="{stiffness}"/>')
        xml.append(f'                    <geom type="capsule" fromto="0 0 0 {longitud_seg:.3f} 0 0" size="0.01" rgba="0.3 {g:.2f} {b:.2f} 1" mass="{mass/num_segmentos:.3f}"/>')

    # 🔹 el dron se coloca centrado en el extremo final (0 0 0)
    xml.append(XML_DRONE_2D)

    # cerrar jerarquía de cuerpos
    xml.extend(["                </body>" for _ in range(num_segmentos - 1)])
    xml.append("            </body>")
    return "\n".join(xml)

# =====================================
# 🔹 3. Ensamblador completo
# =====================================

def get_xml(num_segmentos=5, longitud_total=1.0) -> str:
    barra_xml = generar_barra(num_segmentos, longitud_total)
    return XML_HEADER + barra_xml + XML_FOOTER

# =====================================
# 🔹 4. Ejemplo de uso
# =====================================

if __name__ == "__main__":
    xml_text = get_xml(num_segmentos=6, longitud_total=1.2)
    with open("cantilever_drone2d_aligned.xml", "w", encoding="utf-8") as f:
        f.write(xml_text)
    print("✅ Archivo 'cantilever_drone2d_aligned.xml' generado correctamente.")
    print(xml_text)
    
