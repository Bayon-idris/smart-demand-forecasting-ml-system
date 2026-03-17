import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from scipy import sparse

import pandas as pd
import logging


def load_data(filepath):
    try:
        df = pd.read_csv(filepath)
        return df
    except FileNotFoundError:
        logging.error(f"Error: File not found at {filepath}")
        raise
    except pd.errors.EmptyDataError:
        logging.warning(f"The file at {filepath} exists but is empty.")
        return pd.DataFrame()


def basic_eda(df):

    print("\nDataset Info")
    print(df.info())

    print("\nDescribe")
    print(df.describe())

    print("\nMissing values")
    print(df.isnull().sum())

    print("\nDuplicates:", df.duplicated().sum())


def create_features(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    # Lags
    df["lag_1"] = df.groupby("store")["sales"].shift(1)
    # df["lag_3"] = df.groupby("store")["sales"].shift(3)  # ✅ NEW
    df["lag_7"] = df.groupby("store")["sales"].shift(7)
    df["lag_14"] = df.groupby("store")["sales"].shift(14)
    df["lag_21"] = df.groupby("store")["sales"].shift(21)  # ✅ NEW
    df["lag_28"] = df.groupby("store")["sales"].shift(28)

    # Rolling mean
    df["rolling_mean_7"] = df.groupby("store")["sales"].shift(1).rolling(7).mean()
    df["rolling_mean_14"] = df.groupby("store")["sales"].shift(1).rolling(14).mean()
    df["rolling_mean_28"] = df.groupby("store")["sales"].shift(1).rolling(28).mean()

    # Rolling std (très important pour les pics)
    # df["rolling_std_7"] = df.groupby("store")["sales"].shift(1).rolling(7).std()

    df = df.dropna()

    return df


def prepare_features(df, dv=None, fit=False):

    categorical = [
        "store",
        "promo",
        "holiday",
        "year",
        "month",
        "dayofweek",
        "is_weekend",
    ]

    numerical = [
        "lag_1",
        # 'lag_3',
        "lag_7",
        "lag_14",
        "lag_21",  
        "lag_28",
        "rolling_mean_7",
        "rolling_mean_14",
        "rolling_mean_28",
        # "rolling_std_7",
    ]

    dicts = df[categorical].to_dict(orient="records")

    if fit:
        X_cat = dv.fit_transform(dicts)
    else:
        X_cat = dv.transform(dicts)

    X_num = df[numerical].values

    X = sparse.hstack([X_cat, X_num])

    return X


def visualize_sales(df):

    sns.displot(df["sales"])
    plt.title("Sales Distribution")
    plt.show()

    df.groupby("date")["sales"].sum().plot(figsize=(12, 5))
    plt.title("Total Sales Over Time")
    plt.xlabel("Date")
    plt.ylabel("Sales")
    plt.grid(True)
    plt.show()


def split_dataset(df):

    df = df.sort_values("date")

    split_index = int(len(df) * 0.8)

    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print("\nTrain/Test split created")
    print("Train shape:", train_df.shape)
    print("Test shape:", test_df.shape)

    print("\nTrain period:", train_df["date"].min(), "→", train_df["date"].max())
    print("Test period:", test_df["date"].min(), "→", test_df["date"].max())


def main():

    df = load_data("data/store_sales.csv")
    df = create_features(df)

    split_dataset(df)
    # print(df["sales"].describe())


if __name__ == "__main__":
    main()
