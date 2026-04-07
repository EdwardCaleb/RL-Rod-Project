

class LowPassForceFilter:
    def __init__(self, alpha = 0.5):
        self.alpha = alpha
        self.filtered_force = 0.0

    def filter(self, force_measurement):
        self.filtered_force = self.alpha * force_measurement + (1 - self.alpha) * self.filtered_force
        return self.filtered_force



class MovingAverageForceFilter:
    def __init__(self, window_size = 5):
        self.window_size = window_size
        self.force_history = []

    def filter(self, force_measurement):
        self.force_history.append(force_measurement)
        if len(self.force_history) > self.window_size:
            self.force_history.pop(0)
        return sum(self.force_history) / len(self.force_history)



class MovingAverageForgettingFactorFilter:
    def __init__(self, window_size = 5, forgetting_factor = 0.9):
        self.window_size = window_size
        self.forgetting_factor = forgetting_factor
        self.force_history = []

    def filter(self, force_measurement):
        self.force_history.append(force_measurement)
        if len(self.force_history) > self.window_size:
            self.force_history.pop(0)
        
        weights = [self.forgetting_factor ** i for i in range(len(self.force_history))]
        weighted_sum = sum(f * w for f, w in zip(self.force_history, weights))
        total_weight = sum(weights)
        
        return weighted_sum / total_weight if total_weight > 0 else 0.0
    


# -------------------------------------------------------
# test code
if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt

    # Simulated noisy force measurements
    n = 100
    true_force = np.sin(0.1 * np.arange(n)) * 10.0
    noise = np.random.normal(0, 2, n) # n measurements with mean 0 and std dev 2
    measurements = true_force + noise

    # Create filters
    low_pass_filter = LowPassForceFilter(alpha=0.3)
    moving_average_filter = MovingAverageForceFilter(window_size=5)
    forgetting_factor_filter = MovingAverageForgettingFactorFilter(window_size=5, forgetting_factor=0.8)

    # Apply filters to measurements
    low_pass_results = [low_pass_filter.filter(m) for m in measurements]
    moving_average_results = [moving_average_filter.filter(m) for m in measurements]
    forgetting_factor_results = [forgetting_factor_filter.filter(m) for m in measurements]

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(measurements, label='Noisy Measurements', alpha=0.5)
    plt.plot(low_pass_results, label='Low-Pass Filter', linewidth=2)
    plt.plot(moving_average_results, label='Moving Average Filter', linewidth=2)
    plt.plot(forgetting_factor_results, label='Moving Average with Forgetting Factor', linewidth=2)
    plt.plot(true_force, color='k', linestyle='--', label='True Force')
    plt.legend()
    plt.title('Force Measurement Filtering')
    plt.xlabel('Time Step')
    plt.ylabel('Force')
    plt.grid()
    plt.show()
