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

    # # Método alternativo usando la fórmula directa sin separar u1_force
    # def observe_out(self, F_des, a=np.zeros(3)):
    #     # Estima la fuerza exogena aplicada al drone, por ejemplo por viento, basándose en la diferencia entre F_des (que es lo que el controlador quiere) y F_ideal (lo que se supone que se genera con los comandos dados).
    #     # OJO: La aceleracion debe ser la real, no la deseada, para que esta estimación tenga sentido. En este ejemplo, por simplicidad, se asume que a_T es la aceleración real, pero en un caso real necesitarías medirla o estimarla de alguna forma (por ejemplo con un filtro de Kalman).
    #     F_estimated = self.drone_mass * a - F_des + self.drone_mass * self.gravity * np.array([0.0, 0.0, 1.0])  # simplificación: asumiendo que sin perturbaciones el drone estaría en hover con F_des = peso
    #     return F_estimated




class MovingAverageForgettingFactorFilter:
    def __init__(self, window_size = 5, forgetting_factor = 0.9):
        self.window_size = window_size
        self.forgetting_factor = forgetting_factor
        self.force_history = []

    # limit the force depending on its magnitude, to avoid outliers dominating the estimation. This is optional and can be tuned based on expected force ranges.
    def limiter(self, force_measurement, max_force=50.0):
        if np.linalg.norm(force_measurement) > max_force:
            return max_force * force_measurement / np.linalg.norm(force_measurement)  # scale down to max_force while keeping the direction
        return force_measurement

    def filter(self, force_measurement, limit=True, max_force=50.0):
        if limit:
            force_measurement = self.limiter(force_measurement, max_force=max_force)
        self.force_history.append(force_measurement)
        if len(self.force_history) > self.window_size:
            self.force_history.pop(0)
        
        weights = [self.forgetting_factor ** i for i in range(len(self.force_history))]
        weighted_sum = sum(f * w for f, w in zip(self.force_history, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    






