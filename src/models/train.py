from matplotlib import pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from src.features.build_features import prepare_features
from src.evaluation.metrics import (
    compute_metrics,
    plot_model_performance,
    save_top_errors,
)
from src.utils.config import load_config
from src.utils.utils import save_model


from src.utils.utils import paths


def train(df):

    config = load_config()
    params = config["model_parameters"]

    print("🔹 Splitting dataset...")

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

    print("\n🚀 Training model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    from matplotlib import pyplot as plt
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

from src.features.build_features import prepare_features
from src.evaluation.metrics import (
    compute_metrics,
    plot_model_performance,
    save_top_errors,
    save_metrics,
)
from src.utils.config import load_config
from src.utils.utils import save_model
from src.utils import paths

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

    return model, dv, metrics