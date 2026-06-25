from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd

from .config import FEATURE_COLUMNS, PREDICTION_LOG_PATH, ensure_directories
from .data import validate_transaction_frame
from .modeling import load_model_bundle


def predict_frame(df: pd.DataFrame) -> pd.DataFrame:
    bundle = load_model_bundle()
    cleaned = validate_transaction_frame(df, require_target=False)
    features = cleaned[bundle["feature_columns"]]
    probabilities = bundle["model"].predict_proba(features)[:, 1]
    labels = (probabilities >= 0.5).astype(int)

    results = cleaned.copy()
    results["fraud_probability"] = probabilities.round(6)
    results["prediction"] = labels
    results["prediction_label"] = results["prediction"].map({0: "Legitimate", 1: "Fraud"})
    return results


def predict_single(values: dict[str, float]) -> dict[str, object]:
    df = pd.DataFrame([{column: values[column] for column in FEATURE_COLUMNS}])
    result = predict_frame(df).iloc[0]
    log_prediction(int(result["prediction"]), float(result["fraud_probability"]), source="single")
    return {
        "prediction": int(result["prediction"]),
        "prediction_label": str(result["prediction_label"]),
        "fraud_probability": float(result["fraud_probability"]),
    }


def log_prediction(prediction: int, probability: float, source: str) -> None:
    ensure_directories()
    record = pd.DataFrame(
        [
            {
                "timestamp": datetime.utcnow().isoformat(timespec="seconds"),
                "source": source,
                "prediction": prediction,
                "prediction_label": "Fraud" if prediction == 1 else "Legitimate",
                "fraud_probability": probability,
            }
        ]
    )
    record.to_csv(PREDICTION_LOG_PATH, mode="a", index=False, header=not PREDICTION_LOG_PATH.exists())


def log_batch_predictions(results: pd.DataFrame) -> None:
    if results.empty:
        return
    for _, row in results.iterrows():
        log_prediction(int(row["prediction"]), float(row["fraud_probability"]), source="batch")


def prediction_summary(path: Path = PREDICTION_LOG_PATH) -> dict[str, int]:
    if not path.exists():
        return {"total": 0, "fraud": 0, "legitimate": 0}
    df = pd.read_csv(path)
    if df.empty:
        return {"total": 0, "fraud": 0, "legitimate": 0}
    fraud = int((df["prediction"] == 1).sum())
    legitimate = int((df["prediction"] == 0).sum())
    return {"total": int(len(df)), "fraud": fraud, "legitimate": legitimate}

