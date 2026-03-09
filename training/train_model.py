import pandas as pd
import joblib
from xgboost import XGBRegressor
from sklearn.model_selection import RandomizedSearchCV

from utils import constant



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
def train(filepath):

    df = pd.read_csv(filepath)

    train_df = df[df["year"] <= 2014]

    X_train = train_df[features]
    y_train = train_df[target]

    
    #Find the best parameters with RandomizedSearchCV
    # model = XGBRegressor(random_state=42)
    # param_dist = {
    #     "n_estimators": [100, 200, 300, 400],
    #     "max_depth": [3, 4, 5, 6, 7],
    #     "learning_rate": [0.01, 0.03, 0.05, 0.1],
    #     "subsample": [0.7, 0.8, 0.9, 1.0],
    #     "colsample_bytree": [0.7, 0.8, 0.9, 1.0]
    # }

    # random_search = RandomizedSearchCV(
    #     model,
    #     param_distributions=param_dist,
    #     n_iter=20,  # nombre de combinaisons testées
    #     cv=3,
    #     scoring="neg_mean_absolute_error",
    #     verbose=1,
    #     n_jobs=-1,
    #     random_state=42
    # )

    # print("Starting hyperparameter tuning...")
    # random_search.fit(X_train, y_train)
    # print("Best parameters:", random_search.best_params_)

    #Feature importance finding

    # importance = model.feature_importances_

    # feature_importance = pd.DataFrame(
    #     {"feature": features, "importance": importance}
    # ).sort_values(by="importance", ascending=False)

    # print(feature_importance)
    
    # Best Paramters found with RandomizedSearchCV by scikit-learn
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
