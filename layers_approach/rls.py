import numpy as np

class AdaptiveForceEstimator:
    def __init__(self, n_features, lambda_=0.99, P0_scale=1000.0):
        """
        n_features: tamaño del vector phi
        lambda_: factor de olvido (0 < lambda <= 1)
        P0_scale: valor inicial de P (confianza inicial baja -> grande)
        """
        self.n = n_features
        self.lambda_ = lambda_

        # Parámetros estimados (W_hat)
        self.W_hat = np.zeros((n_features, 1))

        # Matriz P (covarianza inversa)
        self.P = P0_scale * np.eye(n_features)

    def predict(self, phi):
        """
        Estima la fuerza: f_hat = W_hat^T phi
        phi: vector columna (n x 1)
        """
        return float(self.W_hat.T @ phi)

    def update(self, phi, f_real):
        """
        Actualiza el estimador con una nueva medición

        phi: (n x 1)
        f_real: escalar (fuerza real proyectada)
        """
        phi = phi.reshape(-1, 1)

        # Predicción
        f_hat = self.predict(phi)

        # Error
        error = f_real - f_hat

        # Ganancia (tipo RLS)
        denom = self.lambda_ + phi.T @ self.P @ phi
        K = (self.P @ phi) / denom

        # Actualización de parámetros
        self.W_hat = self.W_hat + K * error

        # Actualización de P
        self.P = (1 / self.lambda_) * (
            self.P - K @ phi.T @ self.P
        )

        return f_hat, error
    



class VectorRLS:
    def __init__(self, n_features, n_outputs, lambda_=0.99, P0_scale=1000.0):
        self.n = n_features
        self.m = n_outputs
        self.lambda_ = lambda_

        # W: (n_features x n_outputs)
        self.W = np.zeros((n_features, n_outputs))

        # P: (n_features x n_features)
        self.P = P0_scale * np.eye(n_features)

    def predict(self, phi):
        phi = phi.reshape(-1, 1)  # (n x 1)
        return (self.W.T @ phi).flatten()  # (m,)

    def update(self, phi, F_real):
        phi = phi.reshape(-1, 1)        # (n x 1)
        F_real = F_real.reshape(-1, 1)  # (m x 1)

        # Predicción
        F_hat = self.W.T @ phi          # (m x 1)

        # Error vectorial
        error = F_real - F_hat          # (m x 1)

        # Ganancia
        denom = self.lambda_ + phi.T @ self.P @ phi
        K = (self.P @ phi) / denom      # (n x 1)

        # Update de W (nota: outer product)
        self.W = self.W + K @ error.T   # (n x m)

        # Update de P
        self.P = (1 / self.lambda_) * (
            self.P - K @ phi.T @ self.P
        )

        return F_hat.flatten(), error.flatten()
    







import numpy as np

class RLS:
    """
    Recursive Least Squares (RLS) Algorithm
    Notación basada en:
        π(n), k(n), ξ(n), P(n), w(n)

    Modelo:
        d(n) ≈ w^T u(n)

    Soporta:
        - salida escalar
        - salida vectorial (multi-output)
    """

    def __init__(self, n_features, n_outputs=1, lambda_=0.99, delta=1.0):
        """
        Parámetros:
        ----------
        n_features : int
            Dimensión de u(n) (vector de entrada)
        
        n_outputs : int
            Dimensión de la salida (1 = escalar, >1 = vector)
        
        lambda_ : float
            Factor de olvido (0 < λ ≤ 1)
        
        delta : float
            Inicialización de P(0) = (1/δ) I
            - pequeño δ → aprendizaje rápido
            - grande δ → más robusto al ruido
        """

        self.n = n_features
        self.m = n_outputs
        self.lambda_ = lambda_

        # Parámetros estimados w(n)
        # Dimensión: (n_features x n_outputs)
        self.w = np.zeros((self.n, self.m))

        # Matriz de covarianza inversa P(n)
        self.P = (1.0 / delta) * np.eye(self.n)

    def getWmatrix(self):
        return self.w
    

    def predict(self, u):
        """
        Predicción:
            d_hat(n) = w^T u(n)
        """
        u = u.reshape(-1, 1)  # (n x 1)
        d_hat = self.w.T @ u  # (m x 1)
        return d_hat.flatten()

    def update(self, u, d):
        """
        Ejecuta UNA iteración del algoritmo RLS:

        Entrada:
        --------
        u : (n,)
            vector de entrada u(n)
        
        d : (m,) o escalar
            salida real d(n)

        Retorna:
        --------
        d_hat : predicción
        xi    : error
        """

        # Asegurar dimensiones correctas
        u = u.reshape(-1, 1)          # (n x 1)
        d = np.array(d).reshape(-1, 1)  # (m x 1)

        # --------------------------------------------------
        # 1. π(n) = P(n-1) u(n)
        # --------------------------------------------------
        pi = self.P @ u               # (n x 1)

        # --------------------------------------------------
        # 2. k(n) = π(n) / (λ + u^T π(n))
        # --------------------------------------------------
        denom = self.lambda_ + (u.T @ pi)  # escalar
        k = pi / denom                     # (n x 1)

        # --------------------------------------------------
        # 3. ξ(n) = d(n) - w^T(n-1) u(n)
        # --------------------------------------------------
        d_hat = self.w.T @ u              # (m x 1)
        xi = d - d_hat                    # (m x 1)

        # --------------------------------------------------
        # 4. w(n) = w(n-1) + k(n) ξ(n)^T
        # (outer product si m > 1)
        # --------------------------------------------------
        self.w = self.w + k @ xi.T        # (n x m)

        # --------------------------------------------------
        # 5. P(n) = λ⁻¹ [P(n-1) - k(n) u^T(n) P(n-1)]
        # --------------------------------------------------
        self.P = (1.0 / self.lambda_) * (
            self.P - k @ (u.T @ self.P)
        )

        return d_hat.flatten(), xi.flatten()