from src.features.build_features import create_features, prepare_features


def run_inference(df, model, dv, rmse):

    df = create_features(df)
    df = df.dropna()

    X = prepare_features(df, dv, fit=False)

    df["predicted_sales"] = model.predict(X)

    df["prediction_lower"] = df["predicted_sales"] - rmse
    df["prediction_upper"] = df["predicted_sales"] + rmse

    SERVICE_LEVEL_Z = 1.65
    df["safety_stock"] = SERVICE_LEVEL_Z * rmse

    df["recommended_stock"] = df["predicted_sales"] + df["safety_stock"]

    # decision
    df["decision"] = df.apply(
        lambda row: (
            "high_uncertainty"
            if row["recommended_stock"] > row["predicted_sales"] * 1.5
            else (
                "risk_stockout"
                if row["recommended_stock"] < row["predicted_sales"]
                else "ok"
            )
        ),
        axis=1,
    )
    df["date"] = df["date"].astype(str)

    return df


def build_summary(df, rmse):

    return {
        "total_predicted_sales": float(df["predicted_sales"].sum()),
        "total_recommended_stock": float(df["recommended_stock"].sum()),
        "average_uncertainty": float(rmse),
    }
