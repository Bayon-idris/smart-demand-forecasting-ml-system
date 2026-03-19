from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from src.features.build_features import prepare_features
from src.evaluation.metrics import (
    compute_metrics,
    save_metrics,
    plot_model_performance,
    save_top_errors,
)
from src.utils.config import load_config
from src.utils.utils import save_model


def train(df):
    config = load_config()
    params = config["model_parameters"]
    print("🔹 Splitting dataset...")

    df = df.sort_values("date")

    split_index = int(len(df) * 0.8)

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    print(f"Train: {train_df.shape}, Test: {test_df.shape}")

    dv = DictVectorizer()

    X_train = prepare_features(train_df, dv, fit=True)
    X_test = prepare_features(test_df, dv, fit=False)

    y_train = train_df["sales"].values
    y_test = test_df["sales"].values

    model = XGBRegressor(**params)

    print("\n🚀 Training model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    print(f"\nRMSE: {rmse:.3f}")

    metrics = compute_metrics(y_test, y_pred)
    print("Metrics:", metrics)

    save_metrics(metrics)
    plot_model_performance(y_test, y_pred)
    save_top_errors(y_test, y_pred)

    save_model(dv, model)

    print("\n✅ Training pipeline completed")

    return model
