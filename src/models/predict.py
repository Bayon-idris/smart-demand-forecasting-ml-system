import pickle
import pandas as pd

from src.utils import paths
from src.features.build_features import prepare_features
from src.data.preprocessing import load_data
from src.utils.utils import load_model


def predict(data_path):

    df = load_data(data_path)

    dv, model = load_model()

    X = prepare_features(df, dv, fit=False)

    predictions = model.predict(X)

    df["predictions"] = predictions

    print("\n✅ Predictions generated")

    return df
