import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.utils import paths


def compute_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mae = mean_absolute_error(y_true, y_pred)

    return {"rmse": rmse, "mae": mae}


def save_metrics(metrics: dict, save_path=paths.METRICS_PATH):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    df = pd.DataFrame([metrics])
    df.to_csv(save_path, index=False)

    print(f"\n✅ Metrics saved to {save_path}")


def plot_model_performance(y_true, y_pred, save_dir=None):

    if save_dir is None:
        save_dir = paths.METRICS_DIR

    os.makedirs(save_dir, exist_ok=True)

    residuals = y_true - y_pred

    plt.figure(figsize=(15, 10))

    # 1. Actual vs Predicted (zoom)
    plt.subplot(2, 2, 1)
    n = 200
    plt.plot(y_true[:n], label="Actual")
    plt.plot(y_pred[:n], linestyle="--", label="Predicted")
    plt.title("Actual vs Predicted (Zoom)")
    plt.legend()
    plt.grid(True)

    # 2. Scatter plot
    plt.subplot(2, 2, 2)
    plt.scatter(y_true, y_pred, alpha=0.3)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Scatter: Actual vs Predicted")
    plt.grid(True)

    # 3. Residuals
    plt.subplot(2, 2, 3)
    plt.plot(residuals)
    plt.axhline(0, linestyle="--")
    plt.title("Residuals")
    plt.grid(True)

    # 4. Error distribution
    plt.subplot(2, 2, 4)
    sns.histplot(residuals, bins=50, kde=True)
    plt.title("Error Distribution")
    plt.grid(True)

    plt.tight_layout()

    plot_path = os.path.join(save_dir, "model_performance.png")
    # plt.savefig(plot_path)
    # plt.close()

    return plot_path


def save_top_errors(y_true, y_pred, save_dir=None, top_n=100):

    if save_dir is None:
        save_dir = paths.METRICS_DIR

    os.makedirs(save_dir, exist_ok=True)

    error_df = pd.DataFrame(
        {"actual": y_true, "pred": y_pred, "error": np.abs(y_true - y_pred)}
    )

    error_df = error_df.sort_values(by="error", ascending=False)

    file_path = os.path.join(save_dir, "top_errors.csv")
    error_df.head(top_n).to_csv(file_path, index=False)

    return file_path
