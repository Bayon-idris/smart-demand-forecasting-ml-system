import pandas as pd
import joblib
import numpy as np
import utils.constant as constant 
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
features = ["lag_1", "lag_7", "sell_price", "snap", "is_weekend", "wday", "month"]
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
    
    plt.figure(figsize=(12,6))

    plt.plot(y_valid.values[:100], label="Real sales")
    plt.plot(preds[:100], label="Predicted sales")

    plt.legend()
    plt.show()