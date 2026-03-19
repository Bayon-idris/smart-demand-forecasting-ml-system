from pathlib import Path


def verify_data_path(file_path: str):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Missing file: {file_path}")

    if not path.is_file():
        raise IsADirectoryError(f"Path is a directory, not a file: {file_path}")

    return {"status": "success", "message": f"File {file_path} is ready"}


from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODEL_PATH = BASE_DIR / "artifacts/models/xgboost_model.pkl"
STORE_PATH = BASE_DIR / "data/raw/store_sales.csv"
METRICS_PATH = BASE_DIR / "artifacts/metrics/metrics.csv"

METRICS_DIR = BASE_DIR / "artifacts/metrics"
METRICS_PATH = METRICS_DIR / "metrics.csv"
TRAIN_PATH = BASE_DIR / "data/processed/train.csv"
TEST_PATH = BASE_DIR / "data/processed/test.csv"
