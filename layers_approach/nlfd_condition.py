import numpy as np
from collections import deque

class NLFDCondition:
    def __init__(self, dt=0.01):
        self.dt = dt
        self.last_sample = None
        self.sample_accumulator = 0.0
        self.delay_buffer = deque()

    def apply_noise(self, vector=None, std=1.0, mean=0.0):
        if vector is None:
            return None
        noise = np.random.normal(mean, std, size=vector.shape)
        return vector + noise
    
    def apply_subsampling(self, vector=None, resample_rate=10):
        if vector is None:
            return None
        
        # si sample rate es 10, significa que resampleamos a 10 muestras por segundo, entonces cada 1/resample_rate segundos tomamos una muestra
        if self.last_sample is None:
            self.last_sample = vector
            return vector
        self.sample_accumulator += self.dt
        if self.sample_accumulator >= 1.0 / resample_rate:
            self.sample_accumulator = 0.0
            self.last_sample = vector
            return vector
        else:
            return self.last_sample


    def apply_delay(self, vector=None, delay_time=0.1):
        if vector is None:
            return None
        
        self.delay_buffer.append(vector)
        
        delay_steps = int(delay_time / self.dt)
        if len(self.delay_buffer) <= delay_steps:
            return self.delay_buffer[0]
        
        return self.delay_buffer.popleft()
    
# Ejemplo de uso

nlfd = NLFDCondition()

vector = np.array([1.0, 2.0, 3.0])
for _ in range(20):
    vector = vector + 0.1  # Simula una señal que cambia con el tiempo
    noisy_vector = nlfd.apply_noise(std=0.5, mean=0.0, vector=vector)
    subsampled_vector = nlfd.apply_subsampling(resample_rate=3, vector=noisy_vector)
    delayed_vector = nlfd.apply_delay(delay_time=0.02, vector=subsampled_vector)
    print(f"Noisy: {noisy_vector}, Subsampled: {subsampled_vector}, Delayed: {delayed_vector}")

