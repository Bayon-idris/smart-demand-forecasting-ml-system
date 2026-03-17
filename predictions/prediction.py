import pandas as pd
import joblib
import numpy as np
import utils.constant as constant
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt

features = [
    "wm_yr_wk",
    "wday",
    "snap",
    "year",
    "month",
    "day",
    "lag_1",
    "lag_7",
    "lag_14",
    "lag_28",
    "rolling_mean_7",
    "rolling_mean_30",
    "rolling_std_7",
    "item_subcategory",
    "item_number",
    "sell_price",
    "price_flag",
    "snap_weekend",
    "event_count",
    "event_impact"
]
target = "sales"


def predict(filepath):

    df = pd.read_csv(filepath)

    valid_df = df[df["year"] == 2015]

    X_valid = valid_df[features]
    y_valid = valid_df[target]

    model = joblib.load(constant.model_base_path)

    preds = model.predict(X_valid)

    mae = mean_absolute_error(y_valid, preds)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))

    print("===== XGBOOST MODEL =====")
    print("MAE :", round(mae, 4))
    print("RMSE :", round(rmse, 4))

    plt.figure(figsize=(12, 6))

    plt.plot(y_valid.values[:100], label="Real sales")
    plt.plot(preds[:100], label="Predicted sales")

    plt.legend()
    plt.show()
