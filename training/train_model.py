import pandas as pd
import joblib
from xgboost import XGBRegressor

from utils import constant

features = ["lag_1", "lag_7", "sell_price", "snap", "is_weekend", "wday", "month"]
target = "sales"


def train(filepath):

    df = pd.read_csv(filepath)

    train_df = df[df["year"] <= 2014]

    X_train = train_df[features]
    y_train = train_df[target]

    # Best Paramters found with GridSearchCV by scikit-learn
    model = XGBRegressor(
        learning_rate=0.1,
        max_depth=5,
        n_estimators=100,
        subsample=1.0,
        colsample_bytree=1.0,
        random_state=42,
    )
    model.fit(X_train, y_train)

    joblib.dump(model, constant.model_base_path)

    print("Model saved successfully")
