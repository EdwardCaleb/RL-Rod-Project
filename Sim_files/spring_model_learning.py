import numpy as np


# receives r1,r2,v1,v1, gives d,d',alpha,alpha'
def spring_estimator(r1, r2, v1, v2):
    # Compute relative position and velocity
    dr = r2 - r1
    dv = v2 - v1

    # Compute distance and its derivative
    d = np.linalg.norm(dr)
    if d == 0:
        d_dot = 0
    else:
        d_dot = np.dot(dr, dv) / d

    # Compute angle vertical inclination (assuming z-axis is up)
    if d == 0:
        alpha = 0
    else:
        alpha = np.arcsin(dr[2] / d)  # Vertical angle (radians)

    # Compute angular velocity (assuming z-axis is up)
    if d == 0:
        alpha_dot = 0
    else:
        alpha_dot = (dv[2] * d - dr[2] * d_dot) / (d**2)

    return d, d_dot, alpha, alpha_dot


# example_r1 = np.array([0.0, 0.0, 0.0])
# example_r2 = np.array([1.0, 1.0, 1.0])
# example_v1 = np.array([0.0, 0.0, 0.0])
# example_v2 = np.array([0.1, 0.1, 0.1])
# d, d_dot, alpha, alpha_dot = spring_estimator(example_r1, example_r2, example_v1, example_v2)
# print(f"Distance: {d}, Distance rate: {d_dot}, Angle: {alpha}, Angle rate: {alpha_dot}")



# receives F1,F2 and d, d', alpha, alpha', and learns to predict F_spring
def gaussian_process_learning(F1, F2, d, d_dot, alpha, alpha_dot):
    pass







# spring force observer, receives u1,u2, r1,r2,v1,v2, and estimates F_1spring and F_2spring
def spring_force_observer(u1, u2, r1, r2, v1, v2, m1, m2, dt):

    a1 = (v1 - spring_force_observer.prev_v1) / dt
    a2 = (v2 - spring_force_observer.prev_v2) / dt

    gz = np.array([0.0, 0.0, 9.81])  # gravitational force per unit mass

    f1spring_est = m1*a1 + m1*gz - u1
    f2spring_est = m2*a2 + m2*gz - u2

    spring_force_observer.prev_v1 = v1
    spring_force_observer.prev_v2 = v2

    return f1spring_est, f2spring_est


# low pass filter of estimated forces
def LPF_filter(f1_spring_est, f2_spring_est, f1_spring_est_prev, f2_spring_est_prev, alpha=0.1):
    f1_est = LPF_filter.prev_f1_est = alpha * f1_spring_est + (1 - alpha) * f1_spring_est_prev
    f2_est = LPF_filter.prev_f2_est = alpha * f2_spring_est + (1 - alpha) * f2_spring_est_prev
    return f1_est, f2_est







