from src.models.train import train
from src.models.predict import predict
from src.utils import paths


def main():
    train()
    df_pred = predict(paths.TEST_PATH)
    print(df_pred.head())


if __name__ == "__main__":
    main()
