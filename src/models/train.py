from sklearn.feature_extraction import DictVectorizer
from xgboost import XGBRegressor

from src.evaluation.metrics import compute_metrics
from src.evaluation.offline_metrics import plot_model_performance, save_metrics
from src.features.build_features import prepare_features
from src.utils import paths
from src.utils.config import load_config
from src.utils.utils import save_model


def train(df):

    config = load_config()
    params = config["model_parameters"]

    df = df.sort_values("date")
    split_index = int(len(df) * 0.8)

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    dv = DictVectorizer()
    X_train = prepare_features(train_df, dv, fit=True)
    X_test = prepare_features(test_df, dv, fit=False)

    y_train = train_df["sales"].values
    y_test = test_df["sales"].values

    model = XGBRegressor(**params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = compute_metrics(y_test, y_pred)

    save_metrics(metrics)
    save_model(model, dv, paths.MODEL_PATH)
    plot_model_performance(y_test, y_pred)
    
    print("Model training completed. Metrics saved and model stored at:", paths.MODEL_PATH)

    return {"metrics": metrics, "model_path": paths.MODEL_PATH}
