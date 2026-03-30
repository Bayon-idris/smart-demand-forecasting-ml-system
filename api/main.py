import json

from fastapi import FastAPI, File, HTTPException, UploadFile
import pandas as pd

from src.pipeline.inference import build_summary, run_inference
from src.utils.paths import MODEL_PATH, METRICS_PATH
from src.utils.utils import (
    format_api_response,
    load_model,
    validate_file,
    validate_file,
    validate_input,
)

app = FastAPI()


@app.get("/")
def root():
    return {"message": "Smart Demand Forecasting API"}


@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    try:
        validate_file(file)
        df = pd.read_csv(file.file)
        validate_input(df)

        model, dv = load_model(MODEL_PATH)

        metrics = pd.read_csv(METRICS_PATH)
        rmse = metrics["rmse"].iloc[0]

        result_df = run_inference(df, model, dv, rmse)

        summary = build_summary(result_df, rmse)

        response = {
            "summary": summary,
            "predictions": format_api_response(result_df),
        }

        print(json.dumps(response, indent=2))

        return response

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
