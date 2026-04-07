
import numpy as np

# acceleration_estimator.py based on velocity measurements, using a simple finite difference method
class AccelerationEstimator:
    def __init__(self, dt):
        self.dt = dt
        self.prev_velocity = 0.0
    
    # add noise to the acceleration estimate to simulate real-world conditions
    def add_noise(self, acceleration, noise_std=0.1):
        noise = np.random.normal(0, noise_std, size=acceleration.shape)
        return acceleration + noise

    def estimate(self, velocity_measurement, noise=False, noise_std=0.1):
        # Simple finite difference method to estimate acceleration
        acceleration = (velocity_measurement - self.prev_velocity) / self.dt
        if noise:
            acceleration = self.add_noise(acceleration, noise_std=noise_std)  # add noise to the acceleration estimate
        self.prev_velocity = velocity_measurement
        return acceleration
