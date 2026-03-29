import pandas as pd

from src.pipeline.inference import build_summary, run_inference
from src.data.preprocessing import load_data
import pandas as pd
import json
from src.utils.utils import load_model
from src.utils import paths


def main():

    df = pd.read_csv(paths.STORE_PATH)

    model, dv = load_model(paths.MODEL_PATH)

    metrics = pd.read_csv(paths.METRICS_PATH)
    rmse = metrics["rmse"].iloc[0]

    result_df = run_inference(df, model, dv, rmse)

    summary = build_summary(result_df, rmse)

    response = {"summary": summary, "predictions": result_df.to_dict(orient="records")}

    # 7. print JSON propre
    print(json.dumps(response, indent=2))


if __name__ == "__main__":
    main()
