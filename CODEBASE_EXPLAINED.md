# Codebase Explained In Plain English

This project is a small credit card fraud detection system. It has two main jobs:

1. Train machine learning models that learn the difference between legitimate and fraudulent transactions.
2. Provide a web app where a person can view results, enter one transaction, upload many transactions, and see fraud predictions.

The code is written in Python. The web pages are made with HTML templates and CSS styling. The machine learning work is done with Python data science libraries.

## Big Picture

Think of the project like a small office:

- `fraud_detection/` is the analysis team. It prepares data, trains models, saves results, makes predictions, and logs prediction history.
- `app/` is the front desk. It shows web pages, receives form submissions and CSV uploads, and calls the analysis team when it needs predictions.
- `train.py` is a shortcut for training models from the command line.
- `run.py` is a shortcut for starting the website.
- `data/` is where input datasets live.
- `artifacts/` is where trained model files and training results are saved.
- `logs/` is where prediction history is saved.
- `tests/` checks that the important pieces still work.
- `docs/` contains project writeups for the final year project.

## Frameworks And Libraries Used

### Flask

Flask is the web framework. It lets the project define website routes such as:

- `/` for the dashboard
- `/predict` for single transaction prediction
- `/batch` for CSV upload prediction
- `/metrics` for model results

Flask connects Python functions to web pages. In this project, those Python functions live in `app/__init__.py`.

### Jinja Templates

Jinja is the template language used by Flask. The files in `app/templates/` are HTML pages with special placeholders like `{{ best_model }}` or `{% for model in metrics.models %}`.

Flask fills those placeholders with real Python data before sending the page to the browser.

### Pandas

Pandas handles spreadsheet-like data. It reads CSV files, checks columns, stores transaction rows, and writes results back to CSV.

### NumPy

NumPy helps create numeric data. This project uses it to create a fake demo dataset when the real dataset is missing.

### Scikit-learn

Scikit-learn provides the machine learning models and evaluation tools. The project uses it for:

- Logistic Regression
- Random Forest
- Support Vector Machine
- Gradient Boosting
- Train/test splitting
- Feature scaling
- Model scoring metrics

### Imbalanced-learn

Imbalanced-learn provides SMOTE, a technique that helps train models when fraud cases are much rarer than legitimate cases.

Credit card fraud data is usually very imbalanced because most transactions are not fraud.

### Joblib

Joblib saves and loads the trained model file. The saved model is stored at `artifacts/best_model.joblib`.

### Pytest

Pytest runs automated checks in `tests/test_pipeline.py`.

## How The Whole System Works

### Training Flow

This is what happens when someone runs:

```bash
python train.py
```

1. `train.py` calls `train_and_evaluate()` from `fraud_detection/modeling.py`.
2. `train_and_evaluate()` loads a dataset through `load_dataset()` in `fraud_detection/data.py`.
3. If `data/creditcard.csv` does not exist, the system creates a small demo dataset at `data/demo_creditcard.csv`.
4. The data is checked to make sure it has all required columns.
5. The data is split into training data and testing data.
6. Four different models are trained.
7. Each model is scored using fraud-focused measurements.
8. The best model is chosen.
9. The best model is saved to `artifacts/best_model.joblib`.
10. The metrics are saved to `artifacts/metrics.json` and `artifacts/metrics.csv`.

### Web App Flow

This is what happens when someone runs:

```bash
python run.py
```

1. `run.py` imports `create_app()` from `app/__init__.py`.
2. `create_app()` creates a Flask website.
3. Flask routes are registered for dashboard, prediction, batch upload, and metrics pages.
4. The website starts at `http://127.0.0.1:5000`.
5. When a user opens a page, Flask gathers the needed data and renders an HTML template.

### Prediction Flow

For one transaction:

1. The user opens `/predict`.
2. The user enters values for `Time`, `V1` through `V28`, and `Amount`.
3. Flask reads the form values.
4. Flask calls `predict_single()` in `fraud_detection/prediction.py`.
5. `predict_single()` turns the form values into a Pandas table.
6. `predict_frame()` loads the saved model and predicts fraud probability.
7. The prediction is shown on the page.
8. The prediction is logged in `logs/predictions.csv`.

For many transactions:

1. The user opens `/batch`.
2. The user uploads a CSV file.
3. Flask reads the CSV with Pandas.
4. Flask calls `predict_frame()`.
5. The result table gets new columns for fraud probability and prediction label.
6. The first 20 predictions are previewed on the page.
7. The full prediction CSV can be downloaded.
8. Predictions are logged in `logs/predictions.csv`.

## Important Data Columns

The expected transaction columns are:

- `Time`
- `V1` through `V28`
- `Amount`
- `Class` when training

`Class` is the answer column in the training data:

- `0` means legitimate
- `1` means fraud

When making predictions, the uploaded file does not need `Class`, because the model is trying to predict it.

## File And Folder Guide

### `README.md`

This is the main project introduction. It explains what the system does, how to install dependencies, where to place the dataset, how to train models, how to run the app, and how to run tests.

### `requirements.txt`

This lists the Python packages needed by the project.

Each line pins one package to a specific version, for example `Flask==3.1.2`.

### `pytest.ini`

This configures Pytest.

`pythonpath = .` tells Pytest to treat the project root as an import location. That allows tests to import modules like `fraud_detection.data`.

### `run.py`

This starts the web app.

Functions and variables:

- `app = create_app()` creates the Flask app by calling the factory function in `app/__init__.py`.
- `if __name__ == "__main__": app.run(debug=True)` starts the local development server when the file is run directly.

### `train.py`

This trains the machine learning models from the command line.

Functions:

- `main()` calls `train_and_evaluate()`, then prints the best model name and where the model and metrics were saved.
- `if __name__ == "__main__": main()` runs `main()` only when the file is executed directly.

### `data/`

This folder is for datasets.

Files:

- `.gitkeep` is an empty placeholder so Git keeps the folder even when there is no dataset.
- `creditcard.csv` is expected here if using the real public dataset.
- `demo_creditcard.csv` may be created automatically if the real dataset is missing.

### `artifacts/`

This folder stores generated training outputs.

Files:

- `.gitkeep` is an empty placeholder.
- `best_model.joblib` is created after training and contains the chosen model.
- `metrics.json` is created after training and contains model scores for the app.
- `metrics.csv` is created after training and contains the same model scores in spreadsheet form.

### `logs/`

This folder stores prediction history.

Files:

- `.gitkeep` is an empty placeholder.
- `predictions.csv` is created when predictions are made.

### `docs/literature_review.md`

This is a written literature review for the project. It supports the academic side of the fraud detection system.

### `docs/final_year_project_report.md`

This is the final year project report document. It explains the project background, design, implementation, and results in report form.

## The `fraud_detection` Package

The `fraud_detection/` folder contains the core business logic. It does not display web pages. It focuses on data, models, predictions, and logs.

### `fraud_detection/__init__.py`

This marks `fraud_detection/` as a Python package. A package is a folder that Python can import from.

It contains only a short package description.

### `fraud_detection/config.py`

This file keeps shared settings in one place.

Important variables:

- `BASE_DIR` is the main project folder.
- `DATA_DIR` points to `data/`.
- `ARTIFACT_DIR` points to `artifacts/`.
- `LOG_DIR` points to `logs/`.
- `DEFAULT_DATASET` points to `data/creditcard.csv`.
- `DEMO_DATASET` points to `data/demo_creditcard.csv`.
- `MODEL_PATH` points to `artifacts/best_model.joblib`.
- `METRICS_JSON_PATH` points to `artifacts/metrics.json`.
- `METRICS_CSV_PATH` points to `artifacts/metrics.csv`.
- `PREDICTION_LOG_PATH` points to `logs/predictions.csv`.
- `FEATURE_COLUMNS` lists the input columns used by the model.
- `TARGET_COLUMN` is `Class`, the column the model learns to predict.

Functions:

- `ensure_directories()` creates the `data/`, `artifacts/`, and `logs/` folders if they do not already exist.

### `fraud_detection/data.py`

This file handles datasets. It creates demo data, finds the right dataset file, checks data quality, and loads CSV files.

Classes:

- `DatasetValidationError` is a custom error. The project raises this when a dataset or upload cannot be used.

Functions:

- `create_demo_dataset(path=DEMO_DATASET, rows=700, fraud_rate=0.04)` creates a small fake credit card dataset. It uses random numbers to create legitimate and fraudulent-looking examples, then writes them to a CSV file.
- `resolve_dataset_path(path=None)` decides which dataset file to use. If a path is provided, it uses that. If `data/creditcard.csv` exists, it uses that. Otherwise, it creates and uses the demo dataset.
- `validate_transaction_frame(df, require_target)` checks a Pandas table. It makes sure the table is not empty, required columns exist, values are numeric, and `Class` contains only `0` or `1` when `require_target` is true.
- `load_dataset(path=None)` finds the dataset, reads it with Pandas, handles empty-file errors, validates it, and returns the clean table.

### `fraud_detection/modeling.py`

This file trains, compares, saves, and loads machine learning models.

Classes:

- `TrainingResult` is a small data container. It stores the best model name, all model metrics, and the paths where outputs were saved.

Functions:

- `build_models(use_smote=True)` creates the four candidate machine learning pipelines. A pipeline is a chain of steps, such as balancing data, scaling numbers, and training a model.
- `_score_model(model, x_test, y_test)` tests one trained model and returns scores such as accuracy, precision, recall, F1-score, ROC-AUC, PR-AUC, and confusion matrix values.
- `_ranking_key(metrics)` decides how models should be compared. This project prefers higher recall first, then higher F1-score, then higher PR-AUC, then fewer false positives.
- `train_and_evaluate(dataset_path=None)` runs the full training process. It loads data, splits it, trains all models, scores them, chooses the best model, saves the model bundle, saves metrics, and returns a `TrainingResult`.
- `load_model_bundle(path=None)` loads the saved model bundle. If no saved model exists yet, it trains models first.
- `load_metrics()` loads saved metrics from `artifacts/metrics.json`. If metrics do not exist yet, it trains models first.

Models created by `build_models()`:

- Logistic Regression: a simple model that estimates the chance of fraud using weighted input features.
- Random Forest: many decision trees combined together.
- Support Vector Machine: a model that tries to separate fraud and legitimate examples using a boundary.
- Gradient Boosting: a sequence of small models that improve on earlier mistakes.

### `fraud_detection/prediction.py`

This file uses the saved model to make predictions and record prediction history.

Functions:

- `predict_frame(df)` accepts a Pandas table of transaction rows, validates the input, loads the saved model, predicts fraud probabilities, adds prediction columns, and returns the result table.
- `predict_single(values)` accepts one transaction as a dictionary, turns it into a one-row table, predicts the result, logs it, and returns a simple dictionary for the web page.
- `log_prediction(prediction, probability, source)` adds one prediction record to `logs/predictions.csv`.
- `log_batch_predictions(results)` logs every row from a batch prediction result.
- `prediction_summary(path=PREDICTION_LOG_PATH)` reads the prediction log and returns totals for all predictions, fraud predictions, and legitimate predictions.

## The Flask Web App

The `app/` folder contains the website.

### `app/__init__.py`

This is the main Flask app file.

Functions:

- `create_app()` creates and configures the Flask app. It also defines all website routes.
- `_default_form_values()` creates starting values for the single prediction form. Most fields start at `0.0`, while `Amount` starts at `100.0`.
- `_parse_single_prediction_form(form)` reads the submitted form, makes sure every required field is present, converts each value to a number, and returns a dictionary.

Routes inside `create_app()`:

- `dashboard()` handles `/`. It loads metrics, finds the best model, loads prediction totals, and renders `dashboard.html`.
- `train_models()` handles `POST /train`. It retrains the models, shows a success message, and redirects to the metrics page.
- `predict()` handles `/predict`. On GET, it shows the form. On POST, it reads form values, makes one prediction, shows messages, and renders `predict.html`.
- `batch()` handles `/batch`. On GET, it shows the upload page. On POST, it reads the uploaded CSV, predicts every row, stores a downloadable CSV in the session, logs predictions, and renders `batch.html`.
- `download_batch()` handles `/batch/download`. It lets the user download the most recent batch prediction CSV.
- `metrics()` handles `/metrics`. It loads model metrics and renders `metrics.html`.

Important Flask features used here:

- `render_template()` sends data into an HTML template.
- `request` reads form submissions and uploaded files.
- `redirect()` sends the browser to another page.
- `url_for()` builds route URLs safely.
- `flash()` stores short success or error messages for the next page load.
- `session` temporarily stores the latest batch prediction CSV for downloading.
- `Response` returns a custom CSV download response.

### `app/templates/base.html`

This is the shared page layout.

It contains:

- The HTML document structure.
- The link to `app/static/css/styles.css`.
- The sidebar navigation.
- The train models button.
- The top header.
- The area where page-specific content is inserted.
- The flash message display.

Template blocks:

- `{% block page_title %}` lets each page set its header title.
- `{% block content %}` lets each page insert its main content.

Other templates use `{% extends "base.html" %}` so they inherit this shared layout.

### `app/templates/dashboard.html`

This is the monitoring dashboard page.

It displays:

- The best model name.
- Total predictions made by the app.
- How many predictions were fraud.
- How many predictions were legitimate.
- The best model's recall.
- Precision, recall, F1-score, and PR-AUC.
- Confusion matrix counts.

It receives data from the `dashboard()` route.

### `app/templates/predict.html`

This is the single transaction prediction page.

It displays:

- A form field for every feature column.
- A submit button to score the transaction.
- The result after prediction.

It receives data from the `predict()` route.

### `app/templates/batch.html`

This is the CSV upload page.

It displays:

- A file upload input.
- The required CSV columns.
- A preview of the first 20 prediction rows after upload.
- A download button when batch results are available.

It receives data from the `batch()` route.

### `app/templates/metrics.html`

This page displays the model comparison table.

It shows each model's:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- True negatives
- False positives
- False negatives
- True positives

The selected best model row is visually highlighted.

### `app/static/css/styles.css`

This controls how the website looks.

Main responsibilities:

- Defines reusable colors and shadow values using CSS variables in `:root`.
- Sets the general page font, background, and text color.
- Creates the two-column app layout with sidebar and main content.
- Styles the sidebar, navigation links, brand area, and training card.
- Styles buttons, status pills, cards, panels, tables, forms, upload controls, result messages, and prediction result bands.
- Adds responsive behavior so the app works on smaller screens.

Important CSS sections:

- `:root` stores design tokens such as colors.
- `.app-shell` creates the main layout.
- `.sidebar` styles the left navigation area.
- `.topbar` styles the header.
- `.stat-grid`, `.dashboard-grid`, and `.feature-grid` create grid layouts.
- `.panel` and `.stat-card` style reusable white content boxes.
- `.result-band.safe` and `.result-band.danger` style legitimate and fraud results differently.
- `@media (max-width: 1080px)` adjusts the layout for tablets and smaller screens.
- `@media (max-width: 680px)` stacks content for phone-sized screens.

## Tests

### `tests/test_pipeline.py`

This file contains automated checks for the project.

Functions:

- `test_demo_dataset_matches_expected_schema(tmp_path)` creates a demo dataset in a temporary folder, loads it, and checks that the expected columns are present.
- `test_validation_rejects_missing_columns()` creates an incomplete table and checks that validation raises `DatasetValidationError`.
- `test_training_and_prediction_smoke(tmp_path, monkeypatch)` creates demo data, trains models using temporary artifact paths, checks that model and metrics files are created, then runs predictions on a few rows.

Pytest helpers used:

- `tmp_path` gives the test a temporary folder.
- `monkeypatch` temporarily changes paths during the test so it does not overwrite normal project files.

## How Files Connect To Each Other

Here is the most important connection chain:

```text
run.py
  -> app.create_app()
    -> Flask routes
      -> fraud_detection.modeling.load_metrics()
      -> fraud_detection.modeling.load_model_bundle()
      -> fraud_detection.prediction.predict_single()
      -> fraud_detection.prediction.predict_frame()
      -> fraud_detection.prediction.prediction_summary()
```

Training connects like this:

```text
train.py
  -> fraud_detection.modeling.train_and_evaluate()
    -> fraud_detection.data.load_dataset()
      -> fraud_detection.data.resolve_dataset_path()
      -> fraud_detection.data.validate_transaction_frame()
    -> fraud_detection.modeling.build_models()
    -> fraud_detection.modeling._score_model()
    -> save files in artifacts/
```

Prediction connects like this:

```text
app form or CSV upload
  -> app route in app/__init__.py
    -> fraud_detection.prediction.predict_single() or predict_frame()
      -> fraud_detection.modeling.load_model_bundle()
      -> fraud_detection.data.validate_transaction_frame()
      -> saved model predicts fraud
      -> fraud_detection.prediction.log_prediction()
        -> logs/predictions.csv
```

Template rendering connects like this:

```text
Flask route
  -> render_template("page.html", data=value)
    -> page template extends base.html
      -> browser receives final HTML
      -> styles.css controls appearance
```

## What Happens If Files Are Missing

The project is designed to keep running for demonstrations:

- If `data/creditcard.csv` is missing, the code creates `data/demo_creditcard.csv`.
- If `artifacts/best_model.joblib` is missing, the code trains a model before predicting.
- If `artifacts/metrics.json` is missing, the code trains models before loading metrics.
- If `logs/predictions.csv` is missing, the dashboard shows zero prediction totals.

## What The Model Outputs Mean

The prediction result includes:

- `fraud_probability`: the model's estimated chance that a transaction is fraud.
- `prediction`: `0` for legitimate or `1` for fraud.
- `prediction_label`: the readable label, either `Legitimate` or `Fraud`.

The project uses a probability cutoff of `0.5`:

- Probability below `0.5` becomes `Legitimate`.
- Probability `0.5` or above becomes `Fraud`.

## What The Metrics Mean

- Accuracy: how often the model is correct overall.
- Precision: when the model says fraud, how often it is actually fraud.
- Recall: how many real fraud cases the model catches.
- F1-score: a balance between precision and recall.
- ROC-AUC: how well the model separates fraud and legitimate transactions across thresholds.
- PR-AUC: useful for rare fraud cases because it focuses on precision and recall.
- True negatives: legitimate transactions correctly predicted as legitimate.
- False positives: legitimate transactions incorrectly flagged as fraud.
- False negatives: fraud transactions missed by the model.
- True positives: fraud transactions correctly caught.

This project chooses the best model mainly by recall, because missing fraud can be costly.

## How To Run The Project

Install dependencies:

```bash
pip install -r requirements.txt
```

Train models:

```bash
python train.py
```

Run the web app:

```bash
python run.py
```

Run tests:

```bash
pytest
```

## Simple Mental Model

The easiest way to understand the project is:

1. `data.py` prepares trustworthy data.
2. `modeling.py` trains and saves the best model.
3. `prediction.py` uses the saved model to score new transactions.
4. `app/__init__.py` connects those Python abilities to web pages.
5. `templates/` show the pages.
6. `styles.css` makes the pages look polished.
7. `tests/` make sure the main workflow still works.
