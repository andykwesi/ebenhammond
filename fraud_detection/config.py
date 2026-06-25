from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = BASE_DIR / "artifacts"
LOG_DIR = BASE_DIR / "logs"

DEFAULT_DATASET = DATA_DIR / "creditcard.csv"
DEMO_DATASET = DATA_DIR / "demo_creditcard.csv"
MODEL_PATH = ARTIFACT_DIR / "best_model.joblib"
METRICS_JSON_PATH = ARTIFACT_DIR / "metrics.json"
METRICS_CSV_PATH = ARTIFACT_DIR / "metrics.csv"
PREDICTION_LOG_PATH = LOG_DIR / "predictions.csv"

FEATURE_COLUMNS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
TARGET_COLUMN = "Class"


def ensure_directories() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    ARTIFACT_DIR.mkdir(exist_ok=True)
    LOG_DIR.mkdir(exist_ok=True)

