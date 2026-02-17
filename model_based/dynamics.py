
# Definimos el modelo dinámico para el MPPI

import numpy as np

class SingleMassDynamicModel:
    def __init__(self, dt=0.1, mass=1.0):
        self.dt = dt
        self.mass = mass
        self.mg = np.array([0.0, 0.0, -9.81 * self.mass])  # Gravedad
    
    def step(self, pos, vel, action):
        # pos: [x, y, z]
        # vel: [vx, vy, vz]
        # action: [Fx, Fy, Fz]
        
        a = (action + self.mg) / self.mass
        new_vel = vel + a * self.dt  # Update velocities
        new_pos = pos + vel * self.dt + 0.5 * a * self.dt**2  # Update positions
        # uncertainty variance (simple model, could be more complex)
        pos_var = 0.01 * np.eye(3)  # Variance en posición
        vel_var = 0.01 * np.eye(3)  # Variance en velocidad
        return new_pos, new_vel, pos_var, vel_var


##############################################################################
##############################################################################
############################# Modelo dinámico con TORCH ######################
##############################################################################
##############################################################################
import numpy as np
import torch

class SingleMassDynamicModelTorch:
    def __init__(self, dt=0.1, mass=1.0, device="cuda", dtype=torch.float32):
        self.dt = float(dt)
        self.mass = float(mass)
        self.device = torch.device(device)
        self.dtype = dtype
        self.mg = torch.tensor([0.0, 0.0, -9.81 * self.mass], device=self.device, dtype=self.dtype)

    @torch.no_grad()
    def step(self, pos, vel, action):
        # pos, vel, action pueden ser (3,) o (K,3)
        a = (action + self.mg) / self.mass
        new_vel = vel + a * self.dt
        new_pos = pos + vel * self.dt + 0.5 * a * (self.dt ** 2)
        return new_pos, new_vel



# if __name__ == "__main__":
#     # Test del modelo dinámico
#     model = SingleMassDynamicModel(dt=0.1, mass=1.0)
#     pos = np.array([0.0, 0.0, 0.0])
#     vel = np.array([0.0, 0.0, 0.0])
#     action = np.array([1.0, 0.0, 0.0])  # Fuerza en x
#     new_pos, new_vel, pos_var, vel_var = model.step(pos, vel, action)
#     print("New Position:", new_pos)
#     print("New Velocity:", new_vel)
#     print("Position Variance:\n", pos_var)
#     print("Velocity Variance:\n", vel_var)


# TEST DEL MODELO DINÁMICO EN TORCH
if __name__ == "__main__":
    model = SingleMassDynamicModelTorch(dt=0.1, mass=1.0, device="cuda")
    pos = torch.tensor([0.0, 0.0, 0.0], device=model.device, dtype=model.dtype)
    vel = torch.tensor([0.0, 0.0, 0.0], device=model.device, dtype=model.dtype)
    action = torch.tensor([1.0, 0.0, 0.0], device=model.device, dtype=model.dtype)
    new_pos, new_vel = model.step(pos, vel, action)
    print("New Position:", new_pos.cpu().numpy())
    print("New Velocity:", new_vel.cpu().numpy())

    #verificar que cuda se está usando
    print("Device used:", model.device)


