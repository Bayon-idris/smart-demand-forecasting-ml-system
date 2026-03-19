from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from src.utils import paths
from src.features.build_features import prepare_features
from src.data.preprocessing import load_data
from src.evaluation.metrics import plot_model_performance
from src.utils.utils import save_model
from src.evaluation.metrics import (
    compute_metrics,
    save_metrics,
    plot_model_performance,
    save_top_errors,
)


def train():

    train_df = load_data(paths.TRAIN_PATH)
    test_df = load_data(paths.TEST_PATH)

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

    print("\nTraining model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    print(f"RMSE: {rmse:.3f}")

    metrics = compute_metrics(y_test, y_pred)

    print(metrics)

    save_metrics(metrics)

    plot_model_performance(y_test, y_pred)

    save_top_errors(y_test, y_pred)
    save_model(dv, model)
