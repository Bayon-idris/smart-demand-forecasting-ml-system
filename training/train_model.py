import pandas as pd
import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

from eda.eda import load_data, prepare_features
from utils import constant
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def plot_model_performance(y_test, y_pred, save_path=constant.metric_base_path):

    os.makedirs(save_path, exist_ok=True)

    residuals = y_test - y_pred

    plt.figure(figsize=(15, 10))

    plt.subplot(2, 2, 1)
    n = 200
    plt.plot(y_test[:n], label="Actual")
    plt.plot(y_pred[:n], linestyle="--", label="Predicted")
    plt.title("Actual vs Predicted (Zoom)")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 2, 2)
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title("Scatter: Actual vs Predicted")
    plt.grid(True)

    plt.subplot(2, 2, 3)
    plt.plot(residuals)
    plt.axhline(0, linestyle="--")
    plt.title("Residuals (Errors)")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    sns.histplot(residuals, bins=50, kde=True)
    plt.title("Error Distribution")
    plt.grid(True)

    plt.tight_layout()

    plt.savefig(f"{save_path}/model_performance.png")
    plt.close()

    error_df = pd.DataFrame(
        {"actual": y_test, "pred": y_pred, "error": np.abs(y_test - y_pred)}
    )

    error_df_sorted = error_df.sort_values(by="error", ascending=False)

    error_df_sorted.to_csv(f"{save_path}/top_errors.csv", index=False)

    print(f"\n✅ Graph saved to {save_path}/model_performance.png")
    print(f"✅ Errors saved to {save_path}/top_errors.csv")


def train():

    train_df = load_data("data/train.csv")
    test_df = load_data("data/test.csv")

    dv = DictVectorizer()

    X_train = prepare_features(train_df, dv, fit=True)
    X_test = prepare_features(test_df, dv, fit=False)

    y_train = train_df["sales"].values
    y_test = test_df["sales"].values

    model = XGBRegressor(
        subsample=0.8,
        n_estimators=500,
        min_child_weight=3,
        max_depth=3,
        learning_rate=0.05,
        gamma=0,
        colsample_bytree=0.9,
        random_state=42,
    )

    print("\nTraining tuned XGBoost model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    print(f"RMSE (Tuned XGBoost): {rmse:.3f}")

    plot_model_performance(y_test, y_pred)


def save_model(dv: DictVectorizer, model: XGBRegressor):
    with open(constant.model_base_path, "wb") as f_out:
        pickle.dump((dv, model), f_out)

    print("\nModel saved to model/xgboost_sales_model.bin")
