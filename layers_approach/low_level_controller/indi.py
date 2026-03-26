
import numpy as np

# INDI controller


import numpy as np


class INDIController:
    def __init__(
        self,
        mass,
        gravity,
        inertia,
        gains,
        acc_alpha=0.2,
        angacc_alpha=0.2,
        u1_limits=(0.0, None),
        moment_limits=None,
    ):
        """
        gains[0] = Kp_pos
        gains[1] = Kv_pos
        gains[2] = KR_att
        gains[3] = Komega_att
        """

        self.mass = float(mass)
        self.gravity = float(gravity)
        self.J = np.array(inertia, dtype=float)

        self.Kp = np.array(gains[0], dtype=float)
        self.Kv = np.array(gains[1], dtype=float)
        self.KR = np.array(gains[2], dtype=float)
        self.K_omega = np.array(gains[3], dtype=float)

        self.acc_alpha = float(acc_alpha)
        self.angacc_alpha = float(angacc_alpha)

        self.u1_limits = u1_limits
        self.moment_limits = moment_limits

        self.reset()

    # =========================
    # AUXILIARY FUNCTIONS
    # =========================

    def reset(self):
        # Estado interno del INDI
        self.v_prev = None
        self.omega_prev = None
        self.a_filt_prev = None
        self.alpha_filt_prev = None

        # Inicialización razonable: hover
        self.u_prev = np.array([self.mass * self.gravity, 0.0, 0.0, 0.0], dtype=float)

    def _apply_gain(self, K, x):
        K = np.array(K, dtype=float)
        x = np.array(x, dtype=float)

        if np.ndim(K) == 0:
            return K * x
        if np.ndim(K) == 1:
            return K * x
        if np.ndim(K) == 2:
            return K @ x

        raise ValueError("Ganancia K con forma inválida")

    def _lowpass(self, x, x_prev, alpha):
        x = np.array(x, dtype=float)
        if x_prev is None:
            return x.copy()
        return alpha * x + (1.0 - alpha) * x_prev

    def _clip_scalar_with_none(self, x, limits):
        lo, hi = limits
        if lo is not None:
            x = max(lo, x)
        if hi is not None:
            x = min(hi, x)
        return x

    def _clip_vector_symmetric(self, x, lim):
        x = np.array(x, dtype=float)
        if lim is None:
            return x
        lim = np.array(lim, dtype=float)
        if lim.ndim == 0:
            return np.clip(x, -lim, lim)
        return np.clip(x, -lim, lim)

    def estimated_acceleration(self, v, v_prev, dt):
        if v_prev is None or dt <= 0.0:
            return np.zeros(3)
        return (np.array(v) - np.array(v_prev)) / dt

    def estimated_angular_acceleration(self, omega, omega_prev, dt):
        if omega_prev is None or dt <= 0.0:
            return np.zeros(3)
        return (np.array(omega) - np.array(omega_prev)) / dt

    # =========================
    # OUTER LOOP
    # =========================

    def position_control(self, r_T, v_T, a_T, r, v):
        """
        Genera aceleración neta deseada en marco mundo:
            a_cmd = -Kp*ep - Kv*ev + a_T

        Luego:
            F_des = m * (a_cmd + g*e3)
        """
        e_p = np.array(r, dtype=float) - np.array(r_T, dtype=float)
        e_v = np.array(v, dtype=float) - np.array(v_T, dtype=float)

        e3 = np.array([0.0, 0.0, 1.0])

        a_cmd = (
            - self._apply_gain(self.Kp, e_p)
            - self._apply_gain(self.Kv, e_v)
            + np.array(a_T, dtype=float)
        )

        F_des = self.mass * (a_cmd + self.gravity * e3)

        if np.linalg.norm(F_des) < 1e-9:
            F_des = np.array([0.0, 0.0, self.mass * self.gravity])

        return a_cmd, F_des

    def desired_orientation(self, F_des, psi_T, R_current=None):
        """
        Igual que en tu Mellinguer:
        construye R_des a partir de la fuerza deseada y yaw deseado.
        """
        z_B_des = F_des / np.linalg.norm(F_des)
        x_C_des = np.array([np.cos(psi_T), np.sin(psi_T), 0.0])

        y_B_des_cross = np.cross(z_B_des, x_C_des)
        if np.linalg.norm(y_B_des_cross) < 1e-6:
            for alt in (np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])):
                y_B_des_cross = np.cross(z_B_des, alt)
                if np.linalg.norm(y_B_des_cross) >= 1e-6:
                    break

        y_B_des = y_B_des_cross / np.linalg.norm(y_B_des_cross)
        x_B_des = np.cross(y_B_des, z_B_des)
        x_B_des = x_B_des / np.linalg.norm(x_B_des)

        R_des = np.column_stack((x_B_des, y_B_des, z_B_des))

        if R_current is not None:
            R_alt = np.column_stack((-x_B_des, -y_B_des, z_B_des))
            score1 = np.trace(R_current.T @ R_des)
            score2 = np.trace(R_current.T @ R_alt)
            if score2 > score1:
                R_des = R_alt

        return R_des

    # =========================
    # ATTITUDE ERRORS
    # =========================

    def attitude_control(self, R_des, R_current):
        R_error = R_des.T @ R_current - R_current.T @ R_des
        e_R = 0.5 * np.array([
            R_error[2, 1],
            R_error[0, 2],
            R_error[1, 0]
        ])
        return e_R

    def angular_velocity_error(self, omega_des, omega_current):
        return np.array(omega_current, dtype=float) - np.array(omega_des, dtype=float)

    def desired_angular_acceleration(self, e_R, e_omega, alpha_des_ff=None):
        if alpha_des_ff is None:
            alpha_des_ff = np.zeros(3)

        alpha_cmd = (
            - self._apply_gain(self.KR, e_R)
            - self._apply_gain(self.K_omega, e_omega)
            + np.array(alpha_des_ff, dtype=float)
        )
        return alpha_cmd

    # =========================
    # INDI CORE
    # =========================

    def thrust_indi(self, a_cmd, a_meas, R_current):
        """
        INDI simplificado para empuje total.

        Dinámica aproximada:
            a ≈ (u1/m) * z_B - g*e3 + perturbaciones

        Incremento:
            Δu1 = m * z_B^T (a_cmd - a_meas)

        donde:
        - a_cmd  : aceleración neta deseada en mundo
        - a_meas : aceleración neta medida en mundo
        """
        z_B = R_current[:, 2]
        delta_u1 = self.mass * float(np.dot(z_B, (a_cmd - a_meas)))
        u1 = self.u_prev[0] + delta_u1
        u1 = self._clip_scalar_with_none(u1, self.u1_limits)
        return u1, delta_u1

    def moment_indi(self, alpha_cmd, alpha_meas):
        """
        INDI para momentos.

        Dinámica rotacional:
            alpha ≈ J^{-1} M + términos no modelados

        Usando INDI:
            ΔM = J (alpha_cmd - alpha_meas)
            M  = M_prev + ΔM
        """
        delta_M = self.J @ (np.array(alpha_cmd) - np.array(alpha_meas))
        M = self.u_prev[1:] + delta_M
        M = self._clip_vector_symmetric(M, self.moment_limits)
        return M, delta_M

    # =========================
    # MIXING / ACTUATORS
    # =========================

    def motor_mixing(self, u1, moments):
        return np.array([u1, moments[0], moments[1], moments[2]], dtype=float)

    def ctrl_to_force(self, u, L, kF, kM):
        k = kM / kF
        matrix = np.array([
            [1/4,  0        , -1/(2*L),  1/(4*k)],
            [1/4,  1/(2*L)  ,  0      , -1/(4*k)],
            [1/4,  0        ,  1/(2*L),  1/(4*k)],
            [1/4, -1/(2*L)  ,  0      , -1/(4*k)]
        ])
        prop_F = matrix @ np.array(u, dtype=float)
        return prop_F

    def ctrl_to_limited_force(self, u, L, kF, kM, Fmin, Fmax):
        u = np.array(u, dtype=float).copy()

        # Saturación básica del thrust total
        u1 = float(np.clip(u[0], 4 * Fmin, 4 * Fmax))
        u[0] = u1

        F_ideal = self.ctrl_to_force(u, L=L, kF=kF, kM=kM)

        Fbase = u[0] / 4.0
        d = F_ideal - Fbase

        alphas = [1.0]
        for di in d:
            if di > 0:
                alphas.append((Fmax - Fbase) / di)
            elif di < 0:
                alphas.append((Fmin - Fbase) / di)

        alpha = max(0.0, min(alphas))
        F = Fbase + alpha * d
        F = np.clip(F, Fmin, Fmax)
        return F

    # =========================
    # DISTURBANCE ESTIMATION
    # =========================

    def exerted_force_estimation(self, a_meas, u1, R_current):
        """
        Estimación de fuerza externa en marco mundo:
            m a = u1 z_B - m g e3 + F_ext
            => F_ext = m a - u1 z_B + m g e3
        """
        e3 = np.array([0.0, 0.0, 1.0])
        z_B = R_current[:, 2]
        F_ext = self.mass * np.array(a_meas) - u1 * z_B + self.mass * self.gravity * e3
        return F_ext

    # =========================
    # MAIN STEP
    # =========================

    def step(
        self,
        r_T, v_T, a_T,
        r, v,
        R_current, omega_current,
        psi_T,
        dt,
        a_meas=None,
        omega_des=None,
        alpha_des_ff=None,
    ):
        """
        Entradas:
        - r_T, v_T, a_T       : referencia de posición, velocidad y aceleración (mundo)
        - r, v                : estado actual de posición y velocidad (mundo)
        - R_current           : matriz de rotación body->world
        - omega_current       : velocidad angular actual (body)
        - psi_T               : yaw deseado
        - dt                  : paso de simulación/control
        - a_meas              : aceleración lineal medida en mundo
                                Si no se da, se estima con diferencia de velocidades
        - omega_des           : body-rate deseada
        - alpha_des_ff        : aceleración angular feedforward

        Salida:
        - u = [u1, Mx, My, Mz]
        - F_des
        - F_ext_est
        """

        r = np.array(r, dtype=float)
        v = np.array(v, dtype=float)
        R_current = np.array(R_current, dtype=float)
        omega_current = np.array(omega_current, dtype=float)

        # 1) Medidas / estimaciones
        if a_meas is None:
            a_meas = self.estimated_acceleration(v, self.v_prev, dt)
        else:
            a_meas = np.array(a_meas, dtype=float)

        alpha_meas = self.estimated_angular_acceleration(omega_current, self.omega_prev, dt)

        # 2) Filtros (muy importantes en INDI)
        a_filt = self._lowpass(a_meas, self.a_filt_prev, self.acc_alpha)
        alpha_filt = self._lowpass(alpha_meas, self.alpha_filt_prev, self.angacc_alpha)

        # 3) Outer loop
        a_cmd, F_des = self.position_control(r_T, v_T, a_T, r, v)

        # 4) Orientación deseada
        R_des = self.desired_orientation(F_des, psi_T, R_current)

        # 5) Errores de actitud
        e_R = self.attitude_control(R_des, R_current)

        if omega_des is None:
            omega_des = np.zeros(3)

        e_omega = self.angular_velocity_error(omega_des, omega_current)

        # 6) Aceleración angular deseada
        alpha_cmd = self.desired_angular_acceleration(e_R, e_omega, alpha_des_ff)

        # 7) INDI thrust
        u1, delta_u1 = self.thrust_indi(a_cmd, a_filt, R_current)

        # 8) INDI moments
        M, delta_M = self.moment_indi(alpha_cmd, alpha_filt)

        # 9) Salida
        u = self.motor_mixing(u1, M)

        # 10) Estimación de fuerza externa
        F_ext_est = self.exerted_force_estimation(a_filt, u1, R_current)

        # 11) Actualizar memoria
        self.v_prev = v.copy()
        self.omega_prev = omega_current.copy()
        self.a_filt_prev = a_filt.copy()
        self.alpha_filt_prev = alpha_filt.copy()
        self.u_prev = u.copy()

        return u, F_des, F_ext_est