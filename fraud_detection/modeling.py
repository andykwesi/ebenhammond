from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import (
    FEATURE_COLUMNS,
    METRICS_CSV_PATH,
    METRICS_JSON_PATH,
    MODEL_PATH,
    TARGET_COLUMN,
    ensure_directories,
)
from .data import load_dataset


@dataclass(frozen=True)
class TrainingResult:
    best_model_name: str
    metrics: list[dict[str, Any]]
    model_path: Path
    metrics_json_path: Path
    metrics_csv_path: Path


def build_models(use_smote: bool = True) -> dict[str, Pipeline | ImbPipeline]:
    sampler = [("smote", SMOTE(random_state=42, k_neighbors=3))] if use_smote else []
    return {
        "Logistic Regression": ImbPipeline(
            sampler
            + [
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42)),
            ]
        ),
        "Random Forest": ImbPipeline(
            sampler
            + [
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=160,
                        max_depth=None,
                        min_samples_leaf=2,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=42,
                    ),
                )
            ]
        ),
        "Support Vector Machine": ImbPipeline(
            sampler
            + [
                ("scaler", StandardScaler()),
                ("model", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=42)),
            ]
        ),
        "Gradient Boosting": Pipeline(
            [
                ("model", GradientBoostingClassifier(random_state=42)),
            ]
        ),
    }


def _score_model(model: Pipeline | ImbPipeline, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    predictions = model.predict(x_test)
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(x_test)[:, 1]
    else:
        probabilities = predictions

    tn, fp, fn, tp = confusion_matrix(y_test, predictions, labels=[0, 1]).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "pr_auc": round(float(average_precision_score(y_test, probabilities)), 4),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def _ranking_key(metrics: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        metrics["recall"],
        metrics["f1_score"],
        metrics["pr_auc"],
        -float(metrics["false_positives"]),
    )


def train_and_evaluate(dataset_path: str | Path | None = None) -> TrainingResult:
    ensure_directories()
    df = load_dataset(dataset_path)
    x = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    fraud_count = int(y.sum())
    use_smote = fraud_count >= 4

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model_metrics: list[dict[str, Any]] = []
    fitted_models: dict[str, Pipeline | ImbPipeline] = {}
    for name, model in build_models(use_smote=use_smote).items():
        model.fit(x_train, y_train)
        metrics = _score_model(model, x_test, y_test)
        metrics["model_name"] = name
        model_metrics.append(metrics)
        fitted_models[name] = model

    best = max(model_metrics, key=_ranking_key)
    best_name = best["model_name"]
    bundle = {
        "model_name": best_name,
        "model": fitted_models[best_name],
        "feature_columns": FEATURE_COLUMNS,
        "ranking_metric": "recall, then f1_score, then pr_auc, then fewer false positives",
    }

    joblib.dump(bundle, MODEL_PATH)
    METRICS_JSON_PATH.write_text(json.dumps({"best_model": best_name, "models": model_metrics}, indent=2))
    pd.DataFrame(model_metrics).to_csv(METRICS_CSV_PATH, index=False)

    return TrainingResult(
        best_model_name=best_name,
        metrics=model_metrics,
        model_path=MODEL_PATH,
        metrics_json_path=METRICS_JSON_PATH,
        metrics_csv_path=METRICS_CSV_PATH,
    )


def load_model_bundle(path: Path | None = None) -> dict[str, Any]:
    path = path or MODEL_PATH
    if not path.exists():
        train_and_evaluate()
    return joblib.load(path)


def load_metrics() -> dict[str, Any]:
    if not METRICS_JSON_PATH.exists():
        train_and_evaluate()
    return json.loads(METRICS_JSON_PATH.read_text())
