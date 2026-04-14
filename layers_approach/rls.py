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