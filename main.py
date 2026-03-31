from src.features.build_features import create_features
from src.data.preprocessing import load_data
from src.models.train import train
from src.utils import paths


def main():
    df = load_data(paths.STORE_PATH)

    df = create_features(df)

    train(df)

    print("✅ Pipeline completed")


if __name__ == "__main__":
    main()
