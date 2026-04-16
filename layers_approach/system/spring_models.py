import numpy as np

def linear_spring_force(r, center, k=10.0, rest_length=1.0):
    r_rel = r - center
    norm = np.linalg.norm(r_rel) + 1e-8  # evitar división por cero
    
    r_eq = r_rel / norm
    F = -k * (r_rel - r_eq * rest_length)
    
    return F

def spring_damping_force(r, v, center, k=10.0, rest_length=1.0, c=1.0):
    
    r_rel = r - center
    norm = np.linalg.norm(r_rel) + 1e-8
    r_eq = r_rel / norm
    
    F_spring = -k * (r_rel - r_eq * rest_length)


    v_rel = v - np.zeros_like(v)  # velocidad relativa (asumiendo que el centro está fijo)
    v_radial = np.dot(v_rel, r_eq) * r_eq  # componente radial de la velocidad
    F_damping = -c * v_radial
    
    return F_spring + F_damping