import numpy as np

class ForceObserver:
    def __init__(self, drone_mass, gravity):
        self.drone_mass = drone_mass
        self.gravity = gravity

    def observe(self, u1_sclr, drone_acceleration, R):
        z_B = R[:, 2]  # Extract the z-axis of the body frame in world coordinates
        u1_force = u1_sclr * z_B  # Convert scalar thrust to force vector in world frame
        estimated_force = self.drone_mass*drone_acceleration  +  self.drone_mass*self.gravity*np.array([0.0, 0.0, 1.0]) - u1_force
        return estimated_force

    # def observe(self, F_des, a=np.zeros(3)):
    #     # Estima la fuerza exogena aplicada al drone, por ejemplo por viento, basándose en la diferencia entre F_des (que es lo que el controlador quiere) y F_ideal (lo que se supone que se genera con los comandos dados).
    #     # OJO: La aceleracion debe ser la real, no la deseada, para que esta estimación tenga sentido. En este ejemplo, por simplicidad, se asume que a_T es la aceleración real, pero en un caso real necesitarías medirla o estimarla de alguna forma (por ejemplo con un filtro de Kalman).
    #     F_estimated = self.drone_mass * a - F_des + self.drone_mass * self.gravity * np.array([0.0, 0.0, 1.0])  # simplificación: asumiendo que sin perturbaciones el drone estaría en hover con F_des = peso
    #     return F_estimated


# F_estimated = self.mass * a - F_des + self.mass * self.gravity * np.array([0.0, 0.0, 1.0])
# F_estimated = self.mass*a  -  F_des  +  self.mass*self.gravity*np.array([0.0, 0.0, 1.0])

