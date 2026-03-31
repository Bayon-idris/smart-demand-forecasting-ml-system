import os
import pickle

from fastapi import HTTPException, UploadFile
from src.utils import paths


def save_model(model, dv, model_output_path):

    os.makedirs(os.path.dirname(model_output_path), exist_ok=True)
    with open(model_output_path, "wb") as f_out:
        pickle.dump((model, dv), f_out)

    print(f"✅ Model saved to {model_output_path}")


def load_model(model_input_path):
    if not os.path.exists(model_input_path):
        raise FileNotFoundError(
            f"No file found at: {os.path.abspath(model_input_path)}"
        )

    with open(model_input_path, "rb") as f_in:
        dv, model = pickle.load(f_in)

    return dv, model


def validate_input(df):

    required = ["date", "product_id", "sales"]

    missing_required = [c for c in required if c not in df.columns]

    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    return df


def validate_file(file: UploadFile):
    if not file:
        raise HTTPException(status_code=400, detail="no file uploaded")
    if not file.filename.endswith((".csv", ".xlsx")):
        raise ValueError(
            f"Invalid file type: {file.filename}. Only .csv and .xlsx files are allowed."
        )


def format_api_response(df):

    results = []

    for _, row in df.iterrows():
        results.append({
            "product_id": int(row["product_id"]),
            "date": str(row["date"]),

            "forecast": {
                "predicted_sales": float(row["predicted_sales"]),
                "uncertainty": {
                    "lower": float(row["prediction_lower"]),
                    "upper": float(row["prediction_upper"]),
                }
            },

            "inventory": {
                "recommended_stock": float(row["recommended_stock"]),
                "safety_stock": float(row["safety_stock"]),
            },

            "decision": row["decision"]
        })

    return results