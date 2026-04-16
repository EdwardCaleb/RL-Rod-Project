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






import numpy as np

def evaluate_vector_field_damping(
    model,
    true_force_fn,
    center,
    grid_lim=1.5,
    vel_lim=1.0,
    n_points_pos=6,
    n_points_vel=4
):
    """
    Evalúa modelo F(r,v) vs ground truth

    phi = [x,y,z,vx,vy,vz,1]
    """

    # grids
    x = np.linspace(center[0] - grid_lim, center[0] + grid_lim, n_points_pos)
    y = np.linspace(center[1] - grid_lim, center[1] + grid_lim, n_points_pos)
    z = np.linspace(center[2] - grid_lim, center[2] + grid_lim, n_points_pos)

    vx = np.linspace(-vel_lim, vel_lim, n_points_vel)
    vy = np.linspace(-vel_lim, vel_lim, n_points_vel)
    vz = np.linspace(-vel_lim, vel_lim, n_points_vel)

    errors = []
    rel_errors = []
    angular_errors_deg = []

    y_true_all = []
    y_pred_all = []

    for xi in x:
        for yi in y:
            for zi in z:
                r = np.array([xi, yi, zi], dtype=float)

                for vxi in vx:
                    for vyi in vy:
                        for vzi in vz:
                            v = np.array([vxi, vyi, vzi], dtype=float)

                            # =========================
                            # Fuerza real
                            # =========================
                            F_true = true_force_fn(r, v, center)

                            # =========================
                            # Predicción modelo
                            # =========================
                            phi = np.array([xi, yi, zi, vxi, vyi, vzi, 1.0], dtype=float)

                            F_pred = np.array(
                                model.predict(phi.reshape(1, -1))
                            ).flatten()

                            # limpieza numérica
                            F_pred = np.nan_to_num(F_pred)

                            # guardar
                            y_true_all.append(F_true)
                            y_pred_all.append(F_pred)

                            # =========================
                            # MÉTRICAS
                            # =========================

                            # error absoluto
                            err = np.linalg.norm(F_true - F_pred)
                            errors.append(err)

                            # error relativo
                            rel_err = err / (np.linalg.norm(F_true) + 1e-8)
                            rel_errors.append(rel_err)

                            # error angular
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
        "Num_Samples": len(errors)
    }

    return metrics





