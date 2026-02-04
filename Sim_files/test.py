import numpy as np

# low pass filter of estimated forces
class LPF_filter:
    def __init__(self):
        self.prev_f1_est = np.array([0.0, 0.0, 0.0])
        self.prev_f2_est = np.array([0.0, 0.0, 0.0])

    @staticmethod
    def filter(f1_spring_est, f2_spring_est, alpha=0.1):
        f1_est = LPF_filter.prev_f1_est = alpha * f1_spring_est + (1 - alpha) * LPF_filter.prev_f1_est
        f2_est = LPF_filter.prev_f2_est = alpha * f2_spring_est + (1 - alpha) * LPF_filter.prev_f2_est
        return f1_est, f2_est

# def LPF_filter(f1_spring_est, f2_spring_est, f1_spring_est_prev, f2_spring_est_prev, alpha=0.1):
#     f1_est = LPF_filter.prev_f1_est = alpha * f1_spring_est + (1 - alpha) * f1_spring_est_prev
#     f2_est = LPF_filter.prev_f2_est = alpha * f2_spring_est + (1 - alpha) * f2_spring_est_prev
#     return f1_est, f2_est





# example usage of LPF_filter

# for i in range(10):
#     f1_spring_est = i* np.array([0.1, 0.1, 0.1])
#     f2_spring_est = np.array([0.0, 0.0, 0.0])
#     f1_spring_est_prev = np.array([0.0, 0.0, 0.0])
#     f2_spring_est_prev = np.array([0.0, 0.0, 0.0])
#     f1_est, f2_est = LPF_filter(f1_spring_est, f2_spring_est, f1_spring_est_prev, f2_spring_est_prev)
#     print(f"Filtered F1: {f1_est}, Filtered F2: {f2_est}")





def suma_acumulada(n):
    if not hasattr(suma_acumulada, 'prev_sum'):
        suma_acumulada.prev_sum = 0
    suma = suma_acumulada.prev_sum = n + suma_acumulada.prev_sum
    suma_acumulada.prev_sum = suma
    return suma

print(suma_acumulada(5))

for i in range(1, 6):
    print(suma_acumulada(i))
