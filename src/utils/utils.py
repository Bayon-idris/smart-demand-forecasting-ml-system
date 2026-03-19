import pickle
from src.utils import paths


def save_model(dv, model):

    with open(paths.MODEL_PATH, "wb") as f_out:
        pickle.dump((dv, model), f_out)

    print(f"\n✅ Model saved to {paths.MODEL_PATH}")


def load_model():

    with open(paths.MODEL_PATH, "rb") as f_in:
        dv, model = pickle.load(f_in)

    return dv, model