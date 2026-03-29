from scipy import sparse
import pandas as pd

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


def create_features(df):
    df = df.copy()

    df["date"] = pd.to_datetime(df["date"])

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["dayofweek"] = df["date"].dt.dayofweek
    df["is_weekend"] = df["dayofweek"].isin([5, 6]).astype(int)

    df["lag_1"] = df.groupby("store")["sales"].shift(1)
    df["lag_7"] = df.groupby("store")["sales"].shift(7)
    df["lag_14"] = df.groupby("store")["sales"].shift(14)
    df["lag_21"] = df.groupby("store")["sales"].shift(21)
    df["lag_28"] = df.groupby("store")["sales"].shift(28)

    df["rolling_mean_7"] = df.groupby("store")["sales"].shift(1).rolling(7).mean()
    df["rolling_mean_14"] = df.groupby("store")["sales"].shift(1).rolling(14).mean()
    df["rolling_mean_28"] = df.groupby("store")["sales"].shift(1).rolling(28).mean()

    return df
