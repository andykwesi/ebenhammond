from __future__ import annotations

from io import StringIO

import pandas as pd
from flask import Flask, Response, flash, redirect, render_template, request, session, url_for

from fraud_detection.config import FEATURE_COLUMNS, ensure_directories
from fraud_detection.data import DatasetValidationError
from fraud_detection.modeling import load_metrics, load_model_bundle, train_and_evaluate
from fraud_detection.prediction import log_batch_predictions, predict_frame, predict_single, prediction_summary


def create_app() -> Flask:
    ensure_directories()
    app = Flask(__name__)
    app.config["SECRET_KEY"] = "fraud-detection-thesis-prototype"

    @app.route("/")
    def dashboard():
        metrics = load_metrics()
        best_model = metrics["best_model"]
        best_metrics = next(model for model in metrics["models"] if model["model_name"] == best_model)
        return render_template(
            "dashboard.html",
            active_page="dashboard",
            summary=prediction_summary(),
            best_model=best_model,
            best_metrics=best_metrics,
        )

    @app.route("/train", methods=["POST"])
    def train_models():
        result = train_and_evaluate()
        flash(f"Training complete. Best model: {result.best_model_name}", "success")
        return redirect(url_for("metrics"))

    @app.route("/predict", methods=["GET", "POST"])
    def predict():
        result = None
        form_values = _default_form_values()
        if request.method == "POST":
            try:
                form_values = _parse_single_prediction_form(request.form)
                result = predict_single(form_values)
                flash("Transaction scored successfully.", "success")
            except (ValueError, KeyError, DatasetValidationError) as exc:
                flash(str(exc), "error")
        return render_template(
            "predict.html",
            active_page="predict",
            feature_columns=FEATURE_COLUMNS,
            form_values=form_values,
            result=result,
        )

    @app.route("/batch", methods=["GET", "POST"])
    def batch():
        preview = None
        if request.method == "POST":
            uploaded = request.files.get("file")
            if not uploaded or uploaded.filename == "":
                flash("Choose a CSV file to upload.", "error")
                return redirect(url_for("batch"))
            try:
                df = pd.read_csv(uploaded)
                results = predict_frame(df)
                log_batch_predictions(results)
                session["batch_predictions_csv"] = results.to_csv(index=False)
                preview = results.head(20).to_dict(orient="records")
                flash(f"Batch scored successfully: {len(results)} transaction(s).", "success")
            except (pd.errors.EmptyDataError, DatasetValidationError, ValueError) as exc:
                flash(str(exc), "error")
        return render_template(
            "batch.html",
            active_page="batch",
            feature_columns=FEATURE_COLUMNS,
            preview=preview,
        )

    @app.route("/batch/download")
    def download_batch():
        csv_data = session.get("batch_predictions_csv")
        if not csv_data:
            flash("No batch prediction file is available yet.", "error")
            return redirect(url_for("batch"))
        return Response(
            csv_data,
            mimetype="text/csv",
            headers={"Content-Disposition": "attachment; filename=fraud_predictions.csv"},
        )

    @app.route("/metrics")
    def metrics():
        model_bundle = load_model_bundle()
        metrics_data = load_metrics()
        return render_template(
            "metrics.html",
            active_page="metrics",
            metrics=metrics_data,
            model_name=model_bundle["model_name"],
        )

    return app


def _default_form_values() -> dict[str, float]:
    values = {column: 0.0 for column in FEATURE_COLUMNS}
    values["Amount"] = 100.0
    return values


def _parse_single_prediction_form(form: dict[str, str]) -> dict[str, float]:
    values = {}
    for column in FEATURE_COLUMNS:
        raw = form.get(column, "")
        if raw == "":
            raise ValueError(f"{column} is required.")
        try:
            values[column] = float(raw)
        except ValueError as exc:
            raise ValueError(f"{column} must be numeric.") from exc
    return values

