from scipy import sparse


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
