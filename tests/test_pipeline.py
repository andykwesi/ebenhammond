import pandas as pd
import pytest

from fraud_detection.config import FEATURE_COLUMNS
from fraud_detection.data import DatasetValidationError, create_demo_dataset, load_dataset, validate_transaction_frame
from fraud_detection.modeling import train_and_evaluate
from fraud_detection.prediction import predict_frame


def test_demo_dataset_matches_expected_schema(tmp_path):
    path = create_demo_dataset(tmp_path / "demo.csv", rows=200, fraud_rate=0.05)
    df = load_dataset(path)
    assert FEATURE_COLUMNS[0] == "Time"
    assert set(FEATURE_COLUMNS).issubset(df.columns)
    assert "Class" in df.columns


def test_validation_rejects_missing_columns():
    df = pd.DataFrame({"Amount": [10.0]})
    with pytest.raises(DatasetValidationError):
        validate_transaction_frame(df, require_target=False)


def test_training_and_prediction_smoke(tmp_path, monkeypatch):
    from fraud_detection import config, modeling

    data_path = create_demo_dataset(tmp_path / "demo.csv", rows=260, fraud_rate=0.08)
    monkeypatch.setattr(config, "ARTIFACT_DIR", tmp_path)
    monkeypatch.setattr(modeling, "MODEL_PATH", tmp_path / "best_model.joblib")
    monkeypatch.setattr(modeling, "METRICS_JSON_PATH", tmp_path / "metrics.json")
    monkeypatch.setattr(modeling, "METRICS_CSV_PATH", tmp_path / "metrics.csv")

    result = train_and_evaluate(data_path)
    assert result.model_path.exists()
    assert result.metrics_json_path.exists()

    df = load_dataset(data_path).drop(columns=["Class"]).head(3)
    predictions = predict_frame(df)
    assert "fraud_probability" in predictions.columns
    assert "prediction_label" in predictions.columns

