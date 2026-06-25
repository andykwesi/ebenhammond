from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import DEFAULT_DATASET, DEMO_DATASET, FEATURE_COLUMNS, TARGET_COLUMN, ensure_directories


class DatasetValidationError(ValueError):
    """Raised when a transaction dataset cannot be used by the system."""


def create_demo_dataset(path: Path = DEMO_DATASET, rows: int = 700, fraud_rate: float = 0.04) -> Path:
    """Create a small synthetic dataset that matches the public dataset schema."""
    ensure_directories()
    rng = np.random.default_rng(42)
    fraud_count = max(10, int(rows * fraud_rate))
    legitimate_count = rows - fraud_count

    legitimate = rng.normal(0, 1, size=(legitimate_count, 28))
    fraud = rng.normal(0.95, 1.25, size=(fraud_count, 28))

    features = np.vstack([legitimate, fraud])
    labels = np.array([0] * legitimate_count + [1] * fraud_count)

    amount_legitimate = rng.gamma(shape=2.0, scale=35.0, size=legitimate_count)
    amount_fraud = rng.gamma(shape=2.7, scale=85.0, size=fraud_count)
    amounts = np.concatenate([amount_legitimate, amount_fraud])
    times = rng.integers(0, 172800, size=rows)

    df = pd.DataFrame(features, columns=[f"V{i}" for i in range(1, 29)])
    df.insert(0, "Time", times)
    df["Amount"] = amounts.round(2)
    df[TARGET_COLUMN] = labels
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(path, index=False)
    return path


def resolve_dataset_path(path: str | Path | None = None) -> Path:
    if path:
        return Path(path)
    if DEFAULT_DATASET.exists():
        return DEFAULT_DATASET
    return create_demo_dataset()


def validate_transaction_frame(df: pd.DataFrame, require_target: bool) -> pd.DataFrame:
    if df.empty:
        raise DatasetValidationError("The uploaded file is empty.")

    required = FEATURE_COLUMNS + ([TARGET_COLUMN] if require_target else [])
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise DatasetValidationError(f"Missing required column(s): {', '.join(missing)}")

    cleaned = df.copy()
    for column in required:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")

    invalid_columns = [column for column in required if cleaned[column].isna().any()]
    if invalid_columns:
        raise DatasetValidationError(f"Column(s) contain non-numeric or missing values: {', '.join(invalid_columns)}")

    if require_target:
        target_values = set(cleaned[TARGET_COLUMN].unique().tolist())
        if not target_values.issubset({0, 1}):
            raise DatasetValidationError("The Class column must contain only 0 for legitimate or 1 for fraud.")

    return cleaned


def load_dataset(path: str | Path | None = None) -> pd.DataFrame:
    dataset_path = resolve_dataset_path(path)
    if not dataset_path.exists():
        raise DatasetValidationError(f"Dataset not found: {dataset_path}")
    try:
        df = pd.read_csv(dataset_path)
    except pd.errors.EmptyDataError as exc:
        raise DatasetValidationError("The dataset file is empty.") from exc
    return validate_transaction_frame(df, require_target=True)
