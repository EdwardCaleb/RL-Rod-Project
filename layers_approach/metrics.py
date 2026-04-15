import numpy as np

def evaluate_vector_field(model, true_force_fn, center, grid_lim=1.5, n_points=8):
    x = np.linspace(center[0] - grid_lim, center[0] + grid_lim, n_points)
    y = np.linspace(center[1] - grid_lim, center[1] + grid_lim, n_points)
    z = np.linspace(center[2] - grid_lim, center[2] + grid_lim, n_points)

    errors = []
    rel_errors = []
    angular_errors_deg = []

    y_true_all = []
    y_pred_all = []

    for xi in x:
        for yi in y:
            for zi in z:
                r = np.array([xi, yi, zi], dtype=float)

                # Fuerza real
                F_true = true_force_fn(r, center)

                # Predicción del modelo
                phi = np.array([xi, yi, zi, 1.0], dtype=float)
                F_pred = np.array(model.predict(phi), dtype=float)

                # Guardar
                y_true_all.append(F_true)
                y_pred_all.append(F_pred)

                # Error absoluto vectorial
                err = np.linalg.norm(F_true - F_pred)
                errors.append(err)

                # Error relativo
                rel_err = err / (np.linalg.norm(F_true) + 1e-8)
                rel_errors.append(rel_err)

                # Error angular
                norm_true = np.linalg.norm(F_true)
                norm_pred = np.linalg.norm(F_pred)

                if norm_true > 1e-8 and norm_pred > 1e-8:
                    cos_theta = np.dot(F_true, F_pred) / (norm_true * norm_pred)
                    cos_theta = np.clip(cos_theta, -1.0, 1.0)
                    theta_deg = np.degrees(np.arccos(cos_theta))
                    angular_errors_deg.append(theta_deg)

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    mse = np.mean(np.sum((y_true_all - y_pred_all)**2, axis=1))
    rmse = np.sqrt(mse)

    metrics = {
        "MSE": mse,
        "RMSE": rmse,
        "Mean_Absolute_Vector_Error": float(np.mean(errors)),
        "Mean_Relative_Error": float(np.mean(rel_errors)),
        "Median_Relative_Error": float(np.median(rel_errors)),
        "Mean_Angular_Error_deg": float(np.mean(angular_errors_deg)) if angular_errors_deg else np.nan,
        "Max_Vector_Error": float(np.max(errors)),
    }

    return metrics