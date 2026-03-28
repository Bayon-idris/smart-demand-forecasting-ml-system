import pandas as pd
import matplotlib.pyplot as plt

from src.features.build_features import create_features, prepare_features
from src.utils.utils import load_model
from src.utils import paths


def load_input_data(input_path):
    print("📥 Loading data...")
    return pd.read_csv(input_path)


def build_features(df):
    print("Creating features...")
    df = create_features(df)
    df = df.dropna()
    return df


def load_trained_model():
    print("📦 Loading model...")
    model, dv = load_model(paths.MODEL_PATH)
    return model, dv


def predict_sales(df, model, dv):
    print("🔮 Predicting sales...")
    X = prepare_features(df, dv, fit=False)
    df["predicted_sales"] = model.predict(X)
    return df


def compute_uncertainty(df):
    print("📉 Computing uncertainty...")

    metrics = pd.read_csv(paths.METRICS_PATH)
    rmse = metrics["rmse"].iloc[0]

    df["prediction_lower"] = df["predicted_sales"] - rmse
    df["prediction_upper"] = df["predicted_sales"] + rmse

    return df, rmse


def compute_safety_stock(df, rmse):
    print("📦 Computing safety stock...")

    SERVICE_LEVEL_Z = 1.65  # ~95%
    df["safety_stock"] = SERVICE_LEVEL_Z * rmse

    return df


def recommend_stock(df):
    print("📊 Recommending stock...")
    df["recommended_stock"] = df["predicted_sales"] + df["safety_stock"]
    return df


def add_decision_insights(df):
    print("🧠 Generating decisions...")

    def decision(row):
        if row["recommended_stock"] > row["predicted_sales"] * 1.5:
            return "⚠️ High uncertainty"
        elif row["recommended_stock"] < row["predicted_sales"]:
            return "🚨 Risk of stockout"
        else:
            return "✅ Stock OK"

    df["decision"] = df.apply(decision, axis=1)
    return df


def generate_summary(df):
    print("\n📊 BUSINESS SUMMARY")

    total_predicted = df["predicted_sales"].sum()
    total_stock = df["recommended_stock"].sum()
    avg_uncertainty = (df["prediction_upper"] - df["prediction_lower"]).mean()

    print(f"🔹 Total predicted sales: {total_predicted:.2f}")
    print(f"🔹 Total recommended stock: {total_stock:.2f}")
    print(f"🔹 Avg uncertainty range: {avg_uncertainty:.2f}")

    risk_stockout = (df["decision"] == "🚨 Risk of stockout").sum()
    high_uncertainty = (df["decision"] == "⚠️ High uncertainty").sum()

    print("\n📦 Risk Analysis:")
    print(f"🚨 Stockout risks: {risk_stockout}")
    print(f"⚠️ High uncertainty cases: {high_uncertainty}")


def show_top_risks(df, top_n=10):
    print("\n🚨 TOP STOCK RISKS")

    risks = df[df["decision"] == "🚨 Risk of stockout"]

    if risks.empty:
        print("✅ No major stockout risks detected")
        return

    print(
        risks[["store", "predicted_sales", "recommended_stock"]]
        .head(top_n)
    )


def plot_forecast(df):
    print("📈 Plotting forecast...")

    df_sample = df.head(200)

    plt.figure(figsize=(10, 5))
    plt.plot(df_sample["predicted_sales"], label="Predicted Sales")
    plt.plot(df_sample["recommended_stock"], linestyle="--", label="Recommended Stock")

    plt.legend()
    plt.title("Forecast vs Recommended Stock")
    plt.grid(True)

    plt.savefig("forecast_plot.png")
    plt.close()


def save_clean_output(df, output_path):
    print("💾 Saving clean output...")

    cols = [
        "date",
        "store",
        "predicted_sales",
        "recommended_stock",
        "decision",
    ]

    df[cols].to_csv(output_path, index=False)

    print(f"✅ Clean predictions saved to {output_path}")


def predict(input_path, output_path="predictions.csv"):

    df = load_input_data(input_path)

    df = build_features(df)

    model, dv = load_trained_model()

    df = predict_sales(df, model, dv)

    df, rmse = compute_uncertainty(df)

    df = compute_safety_stock(df, rmse)

    df = recommend_stock(df)

    df = add_decision_insights(df)

    generate_summary(df)
    show_top_risks(df)
    plot_forecast(df)
    save_clean_output(df, output_path)

    return df


if __name__ == "__main__":
    predict(paths.DATASET_PATH)