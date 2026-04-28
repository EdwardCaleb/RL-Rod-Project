import numpy as np

class PwMToForceThrust:
    def __init__(self, bias=-575.580758, k_pwm=2.204870e-03, k_vbat=4.082917e+01, drone_mass=0.153, gravity=9.81):
        self.bias = bias
        self.k_pwm = k_pwm
        self.k_vbat = k_vbat
        self.drone_mass = drone_mass
        self.gravity = gravity
        

    def calibrate(self, pwm_values, force_values):
        pass

    def pwm_to_force(self, pwm_value, vbat):
        # f_grams(PWM, V) = -422.580758 + (2.204870e-03)*PWM + (4.082917e+01)*Voltage
        f_grams = (self.bias  +  self.drone_mass*1000)  +  self.k_pwm*pwm_value  +  self.k_vbat*vbat
        
        # limitar a fuerza máxima de 400 gramos y minima de 0 gramos
        f_grams = max(0, min(f_grams, 400))

        f_newtons = f_grams * self.gravity / 1000  # Convertir gramos a Newtons
        return f_newtons
    

    
    