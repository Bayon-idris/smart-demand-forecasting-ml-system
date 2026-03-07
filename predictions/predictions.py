import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import matplotlib.pyplot as plt


features = ["lag_1", "lag_7", "sell_price", "snap", "is_weekend", "wday", "month"]

target = "sales"


def predict(filepath):

    df = pd.read_csv(filepath)
    train_df = df[df["year"] <= 2014]
    valid_df = df[df["year"] == 2015]

    X_train = train_df[features]
    y_train = train_df[target]

    X_valid = valid_df[features]
    y_valid = valid_df[target]

    model = XGBRegressor(
        n_estimators=200, max_depth=6, learning_rate=0.05, random_state=42, n_jobs=-1
    )

    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    mae = mean_absolute_error(y_valid, preds)
    rmse = np.sqrt(mean_squared_error(y_valid, preds))

    importance = model.feature_importances_

    feature_importance = pd.DataFrame({
        "feature": features,
        "importance": importance
    }).sort_values(by="importance", ascending=False)

    print(feature_importance)
    print("===== XGBOOST MODEL =====")
    print("MAE :", round(mae, 4))
    print("RMSE :", round(rmse, 4))

    print(feature_importance)
