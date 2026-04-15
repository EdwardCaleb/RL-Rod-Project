import numpy as np

def linear_spring_force(r, center, k=10.0):
    r_rel = r - center
    norm = np.linalg.norm(r_rel) + 1e-8  # evitar división por cero
    
    r_eq = r_rel / norm
    F = -k * (r_rel - r_eq)
    
    return F
