from __future__ import annotations
from dataclasses import dataclass
import numpy as np


def _skew_to_vee(S: np.ndarray) -> np.ndarray:
    """
    vee map: so(3) -> R^3
    For a skew-symmetric matrix:
        [  0  -z   y]
        [  z   0  -x]  -> [x, y, z]
        [ -y   x   0]
    """
    return np.array([S[2, 1], S[0, 2], S[1, 0]], dtype=float)


def _normalize(v: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < eps:
        raise ValueError("No se puede normalizar un vector con norma ~0.")
    return v / n


@dataclass
class QuadrotorParams:
    m: float = 1.0
    g: float = 9.81

    # Solo necesarios si quieres mezclar a velocidades de motor
    kF: float | None = None
    kM: float | None = None
    L: float | None = None


class MellingerController2011:
    """
    Controlador no lineal de Mellinger & Kumar (2011), Sección IV (Control).

    Entradas típicas:
      - estado: r, r_dot, R (world<-body), omega (en body)
      - referencia: rT, rT_dot, rT_ddot, psiT
      - opcional: omegaT (velocidad angular deseada en body, default 0)

    Salidas:
      - u = [u1, u2, u3, u4] (thrust + momentos en el body frame)
      - adicionalmente: F_des, R_des, eR, eomega (útiles para debug)
    """

    def __init__(
        self,
        params: QuadrotorParams,
        Kp: np.ndarray,
        Kv: np.ndarray,
        KR: np.ndarray,
        Komega: np.ndarray,
    ):
        self.p = params

        self.Kp = np.array(Kp, dtype=float)
        self.Kv = np.array(Kv, dtype=float)
        self.KR = np.array(KR, dtype=float)
        self.Komega = np.array(Komega, dtype=float)

        # Checks simples
        for M, name in [(self.Kp, "Kp"), (self.Kv, "Kv"), (self.KR, "KR"), (self.Komega, "Komega")]:
            if M.shape != (3, 3):
                raise ValueError(f"{name} debe ser 3x3.")

    @staticmethod
    def _zB_from_R(R: np.ndarray) -> np.ndarray:
        """
        zB (en mundo) es la tercera columna de R = [xB yB zB].
        R debe ser (3,3).
        """
        return R[:, 2]

    @staticmethod
    def _xC_from_yaw(psi: float) -> np.ndarray:
        return np.array([np.cos(psi), np.sin(psi), 0.0], dtype=float)

    def compute_R_des(self, F_des: np.ndarray, psiT: float, R_current: np.ndarray | None = None) -> np.ndarray:
        """
        Construye R_des como en el paper:
          zB_des = F_des / ||F_des||
          xC_des = [cos psiT, sin psiT, 0]
          yB_des = normalize(zB_des x xC_des)
          xB_des = yB_des x zB_des
          R_des = [xB_des yB_des zB_des]

        El paper menciona la singularidad cuando zB_des || xC_des.
        Aquí:
          - si casi paralelos, hacemos un fallback usando un eje alternativo
          - además, si R_current se pasa, elegimos la solución más cercana
            (tal como comentan: usar signos alternativos para evitar saltos).
        """
        zB_des = _normalize(F_des)

        xC_des = self._xC_from_yaw(psiT)

        cross = np.cross(zB_des, xC_des)
        if np.linalg.norm(cross) < 1e-6:
            # fallback: usa otro eje "xC" alternativo (por ejemplo eje Y mundo)
            xC_alt = np.array([0.0, 1.0, 0.0], dtype=float)
            cross = np.cross(zB_des, xC_alt)
            if np.linalg.norm(cross) < 1e-6:
                # último recurso: eje X mundo
                xC_alt = np.array([1.0, 0.0, 0.0], dtype=float)
                cross = np.cross(zB_des, xC_alt)

            yB_des = _normalize(cross)
        else:
            yB_des = _normalize(cross)

        xB_des = np.cross(yB_des, zB_des)

        R_des = np.column_stack((xB_des, yB_des, zB_des))

        # Paper: -xB_des y -yB_des también válidos (misma z y yaw)
        # Elegimos el más cercano a R actual si está disponible:
        if R_current is not None:
            R_current = np.asarray(R_current, dtype=float)
            R_alt = np.column_stack((-xB_des, -yB_des, zB_des))
            # métrica simple: mayor tr(R_current^T R_candidate)
            score1 = np.trace(R_current.T @ R_des)
            score2 = np.trace(R_current.T @ R_alt)
            if score2 > score1:
                R_des = R_alt

        return R_des

    def compute_control(
        self,
        r: np.ndarray,
        r_dot: np.ndarray,
        R: np.ndarray,
        omega: np.ndarray,
        rT: np.ndarray,
        rT_dot: np.ndarray,
        rT_ddot: np.ndarray,
        psiT: float,
        omegaT: np.ndarray | None = None,
    ) -> dict:
        """
        Implementa Sección IV del paper.

        Devuelve un dict con:
          u, u1, u2u3u4, F_des, R_des, e_p, e_v, e_R, e_omega
        """
        r = np.asarray(r, dtype=float).reshape(3)
        r_dot = np.asarray(r_dot, dtype=float).reshape(3)
        R = np.asarray(R, dtype=float).reshape(3, 3)
        omega = np.asarray(omega, dtype=float).reshape(3)

        rT = np.asarray(rT, dtype=float).reshape(3)
        rT_dot = np.asarray(rT_dot, dtype=float).reshape(3)
        rT_ddot = np.asarray(rT_ddot, dtype=float).reshape(3)

        if omegaT is None:
            omegaT = np.zeros(3)
        else:
            omegaT = np.asarray(omegaT, dtype=float).reshape(3)

        # 1) errores traslacionales
        e_p = r - rT
        e_v = r_dot - rT_dot

        # 2) F_des = -Kp ep - Kv ev + mg zW + m rT_ddot
        zW = np.array([0.0, 0.0, 1.0], dtype=float)
        F_des = -(self.Kp @ e_p) - (self.Kv @ e_v) + self.p.m * self.p.g * zW + self.p.m * rT_ddot

        if np.linalg.norm(F_des) < 1e-9:
            raise ValueError("||F_des|| ~ 0; referencia / ganancias generan fuerza nula.")

        # 3) u1 = F_des · zB
        zB = self._zB_from_R(R)  # en mundo
        u1 = float(F_des.dot(zB))

        # 4) construir R_des
        R_des = self.compute_R_des(F_des=F_des, psiT=psiT, R_current=R)

        # 5) error de rotación eR = 1/2 (Rdes^T R - R^T Rdes)^vee
        S = (R_des.T @ R) - (R.T @ R_des)
        e_R = 0.5 * _skew_to_vee(S)

        # 6) error de velocidad angular
        e_omega = omega - omegaT

        # 7) momentos: [u2,u3,u4]^T = -KR eR - Komega eomega
        u2u3u4 = -(self.KR @ e_R) - (self.Komega @ e_omega)

        u = np.array([u1, u2u3u4[0], u2u3u4[1], u2u3u4[2]], dtype=float)

        return {
            "u": u,
            "u1": u1,
            "u2u3u4": u2u3u4,
            "F_des": F_des,
            "R_des": R_des,
            "e_p": e_p,
            "e_v": e_v,
            "e_R": e_R,
            "e_omega": e_omega,
        }

    def mix_to_motor_omegas_squared(self, u: np.ndarray) -> np.ndarray:
        """
        Usa la matriz (2) del paper:
          u = A * [w1^2, w2^2, w3^2, w4^2]^T

        Retorna w_sq (4,)
        Requiere kF, kM y L en params.
        """
        if self.p.kF is None or self.p.kM is None or self.p.L is None:
            raise ValueError("Para mezclar a motores necesitas params.kF, params.kM, params.L.")

        u = np.asarray(u, dtype=float).reshape(4)
        kF, kM, L = self.p.kF, self.p.kM, self.p.L

        A = np.array(
            [
                [kF, kF, kF, kF],
                [0.0, kF * L, 0.0, -kF * L],
                [-kF * L, 0.0, kF * L, 0.0],
                [kM, -kM, kM, -kM],
            ],
            dtype=float,
        )
        w_sq = np.linalg.solve(A, u)

        # opcional: clamp a >=0
        w_sq = np.maximum(w_sq, 0.0)
        return w_sq


# -------------------------
# Ejemplo mínimo de uso
# -------------------------
if __name__ == "__main__":
    params = QuadrotorParams(m=1.0, g=9.81)

    # Ganancias típicas (ajusta a tu modelo)
    Kp = np.diag([6.0, 6.0, 8.0])
    Kv = np.diag([4.0, 4.0, 5.0])
    KR = np.diag([8.0, 8.0, 2.0])
    Komega = np.diag([0.2, 0.2, 0.1])

    ctrl = MellingerController2011(params, Kp, Kv, KR, Komega)

    # Estado actual (ejemplo)
    r = np.array([0.0, 0.0, 0.5])
    r_dot = np.zeros(3)
    R = np.eye(3)      # nivelado
    omega = np.zeros(3)

    # Referencia
    rT = np.array([1.0, 1.0, 1.0])
    rT_dot = np.zeros(3)
    rT_ddot = np.zeros(3)
    psiT = 0.0

    out = ctrl.compute_control(r, r_dot, R, omega, rT, rT_dot, rT_ddot, psiT)
    print("u = [u1,u2,u3,u4] =", out["u"])
    print("F_des =", out["F_des"])
