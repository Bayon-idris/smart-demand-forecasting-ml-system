from sklearn.metrics import mean_squared_error, mean_absolute_error


def compute_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)

    return {"rmse": rmse, "mae": mae}
