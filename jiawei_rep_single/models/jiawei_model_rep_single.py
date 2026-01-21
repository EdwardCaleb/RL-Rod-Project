# =====================================
# 🔹 1. Bloques estáticos del modelo (3D)
# =====================================

XML_HEADER = r"""
<mujoco model="cantilever_3d_drone">
    <compiler angle="degree" inertiafromgeom="true" coordinate="local"/>
    <option timestep="0.01" gravity="0 0 -9.81" density="1" viscosity="1e-5" integrator="RK4"/>

    <default>
        <!-- Un poco de amortiguamiento para estabilidad -->
        <joint damping="0.001" limited="false"/>
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
            <!-- Sitio de referencia del soporte -->
            <site name="base_site" pos="0 0 0" size="0.01" rgba="1 0 0 1"/>
"""

# =====================================
# 🔹 2. Dron 3D (cuerpo libre, sin weld)
# =====================================

# NOTA: el dron queda como body "drone_3d" con joint free para que pueda moverse/rotar en 3D.
# Luego lo conectamos al último segmento con equality connect (ball).
XML_DRONE_3D = r"""
                        <!-- 🚁 Dron 3D (cuerpo libre) -->
                        <body name="drone_3d" pos="0 0 0">
                            <!-- Joint libre (6DOF): permite que el dron exista físicamente en 3D -->
                            <joint name="drone_free" type="free"/>

                            <!-- Cuerpo principal -->
                            <geom name="drone_body" type="box" size="0.08 0.04 0.02" material="MatDrone" mass="0.15"/>

                            <!-- Brazos (solo visual/simple) -->
                            <geom name="arm_x_pos" type="box" pos=" 0.10 0 0" size="0.06 0.01 0.01" rgba="0.8 0.8 0.8 1"/>
                            <geom name="arm_x_neg" type="box" pos="-0.10 0 0" size="0.06 0.01 0.01" rgba="0.8 0.8 0.8 1"/>
                            <geom name="arm_y_pos" type="box" pos="0  0.10 0" size="0.01 0.06 0.01" rgba="0.8 0.8 0.8 1"/>
                            <geom name="arm_y_neg" type="box" pos="0 -0.10 0" size="0.01 0.06 0.01" rgba="0.8 0.8 0.8 1"/>

                            <!-- Hélices / motores (sites para actuadores) -->
                            <body name="prop_0" pos=" 0.13  0.00 0.03">
                                <geom type="cylinder" size="0.02 0.003" rgba="0.3 1 0.3 0.6"/>
                                <site name="motor_0" pos="0 0 0.003" size="0.006" rgba="0 1 0 0.7" type="cylinder"/>
                            </body>
                            <body name="prop_1" pos="-0.13  0.00 0.03">
                                <geom type="cylinder" size="0.02 0.003" rgba="0.3 1 0.3 0.6"/>
                                <site name="motor_1" pos="0 0 0.003" size="0.006" rgba="0 1 0 0.7" type="cylinder"/>
                            </body>
                            <body name="prop_2" pos=" 0.00  0.13 0.03">
                                <geom type="cylinder" size="0.02 0.003" rgba="0.3 1 0.3 0.6"/>
                                <site name="motor_2" pos="0 0 0.003" size="0.006" rgba="0 1 0 0.7" type="cylinder"/>
                            </body>
                            <body name="prop_3" pos=" 0.00 -0.13 0.03">
                                <geom type="cylinder" size="0.02 0.003" rgba="0.3 1 0.3 0.6"/>
                                <site name="motor_3" pos="0 0 0.003" size="0.006" rgba="0 1 0 0.7" type="cylinder"/>
                            </body>

                            <!-- Sitio donde se conecta la rótula (ball) con la viga -->
                            <site name="drone_attach" pos="0 0 0" size="0.01" rgba="1 0 1 1"/>

                            <!-- Ejes de referencia -->
                            <site name="x_axis" type="box" pos="0.20 0 0" size="0.20 0.005 0.005" rgba="1 0 0 0.3"/>
                            <site name="y_axis" type="box" pos="0 0.20 0" size="0.005 0.20 0.005" rgba="0 1 0 0.3"/>
                            <site name="z_axis" type="box" pos="0 0 0.20" size="0.005 0.005 0.20" rgba="0 0 1 0.3"/>
                        </body>
"""

# =====================================
# 🔹 3. Footer (actuadores + sensores + unión esférica)
# =====================================

XML_FOOTER = r"""
        </body>
    </worldbody>

    <!-- ⚙️ Actuadores (4 hélices) -->
    <actuator>
        <!-- ctrlrange más amplio para que sea más fácil controlar -->
        <motor site="motor_0" ctrlrange="0 2" gear="0 0 1 0 0 -0.05"/>
        <motor site="motor_1" ctrlrange="0 2" gear="0 0 1 0 0  0.05"/>
        <motor site="motor_2" ctrlrange="0 2" gear="0 0 1 0 0 -0.05"/>
        <motor site="motor_3" ctrlrange="0 2" gear="0 0 1 0 0  0.05"/>
    </actuator>

    <!-- 📡 Sensores -->
    <sensor>
        <framepos    name="pos_drone"    objtype="body" objname="drone_3d"/>
        <framequat   name="quat_drone"   objtype="body" objname="drone_3d"/>
        <framelinvel name="linvel_drone" objtype="body" objname="drone_3d"/>
        <frameangvel name="rotvel_drone" objtype="body" objname="drone_3d"/>
    </sensor>

    <!-- 🧩 Unión esférica (ball) entre último segmento y dron -->
    <equality>
        <!-- Conecta el sitio del último segmento (end_site) con el sitio del dron (drone_attach)
             Esto actúa como una rótula: coinciden en posición, pero el dron puede rotar libremente -->
        <connect name="ball_to_drone" site1="end_site" site2="drone_attach"
                 solimp="0.9 0.95 0.001" solref="0.02 1"/>
    </equality>

</mujoco>
"""


# =====================================
# 🔹 4. Generador de la barra articulada (3D)
# =====================================

def generar_barra_3d(
    num_segmentos=5,
    longitud_total=1.0,
    damping=0.002,
    stiffness=0.0,
    mass=0.1
) -> str:
    """
    Genera una viga flexible 3D con joints tipo BALL.
    - Cada segmento tiene un joint ball (3 DOF rotacionales).
    - Al final crea un site "end_site" para conectar el dron con equality/connect.
    """
    longitud_seg = longitud_total / num_segmentos
    xml = []

    xml.append(f'            <!-- 🔗 Barra flexible 3D con {num_segmentos} segmentos -->')
    xml.append(f'            <body name="seg1" pos="0.1 0 0">')

    # Primer joint: ball (rótula)
    xml.append(f'                <joint name="ball1" type="ball" damping="{damping}"/>')

    # Primer segmento geom
    xml.append(
        f'                <geom type="capsule" fromto="0 0 0 {longitud_seg:.3f} 0 0" '
        f'size="0.01" material="MatBeam" mass="{mass/num_segmentos:.3f}"/>'
    )

    # Segmentos intermedios
    for i in range(2, num_segmentos + 1):
        g = 0.9 - i * 0.08
        b = 0.3 + i * 0.08
        xml.append(f'                <body name="seg{i}" pos="{longitud_seg:.3f} 0 0">')
        xml.append(f'                    <joint name="ball{i}" type="ball" damping="{damping}"/>')
        # Si quieres "semi-rigidez", en MuJoCo la rigidez real suele trabajarse con equality/solref,
        # aquí lo mantenemos simple: solo damping.
        xml.append(
            f'                    <geom type="capsule" fromto="0 0 0 {longitud_seg:.3f} 0 0" '
            f'size="0.01" rgba="0.3 {g:.2f} {b:.2f} 1" mass="{mass/num_segmentos:.3f}"/>'
        )

    # Site al final del último segmento para conectar el dron con rótula
    xml.append(f'                    <!-- Punto de conexión (extremo libre) -->')
    xml.append(f'                    <site name="end_site" pos="{longitud_seg:.3f} 0 0" size="0.01" rgba="1 1 0 1"/>')

    # Colocar el dron "cerca" del extremo. OJO:
    # Como el dron tiene joint free, si lo dejas exactamente en el mismo sitio puede explotar numéricamente
    # por colisiones. Lo dejamos muy cerca en z.
    xml.append('                    <!-- Dron cerca del extremo, se une con connect(ball) -->')
    xml.append('                    <body name="drone_mount" pos="{:.3f} 0 0.02">'.format(longitud_seg))
    xml.append(XML_DRONE_3D)
    xml.append('                    </body>')

    # Cerrar jerarquía de cuerpos
    xml.extend(["                </body>" for _ in range(num_segmentos - 1)])
    xml.append("            </body>")

    return "\n".join(xml)


# =====================================
# 🔹 5. Ensamblador completo
# =====================================

def get_xml(num_segmentos=5, longitud_total=1.0) -> str:
    barra_xml = generar_barra_3d(num_segmentos, longitud_total)
    return XML_HEADER + barra_xml + XML_FOOTER


# =====================================
# 🔹 6. Ejemplo de uso
# =====================================

if __name__ == "__main__":
    xml_text = get_xml(num_segmentos=6, longitud_total=1.2)

    with open("cantilever_drone3d_ball.xml", "w", encoding="utf-8") as f:
        f.write(xml_text)

    print("✅ Archivo 'cantilever_drone3d_ball.xml' generado correctamente.")
    print(xml_text)
