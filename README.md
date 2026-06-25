# Credit Card Fraud Detection System

A thesis-ready Flask prototype for credit card fraud detection using classical machine learning techniques. The app trains multiple models, selects the best model using fraud-sensitive metrics, and provides dashboard pages for single and batch transaction prediction.

## Features

- Model comparison for Logistic Regression, Random Forest, Support Vector Machine, and Gradient Boosting
- Imbalanced-data handling with SMOTE when available for the training split
- Fraud-focused metrics: precision, recall, F1-score, ROC-AUC, PR-AUC, and confusion matrix
- Flask dashboard with monitoring summary, single prediction, batch CSV prediction, and metrics pages
- Local artifact storage for trained model bundle, metrics, and prediction logs
- Synthetic demo dataset fallback when `data/creditcard.csv` is not present

## Project Structure

```text
app/                  Flask application, templates, and static assets
fraud_detection/      Data validation, training, prediction, and logging logic
data/                 Place the public credit card fraud CSV here
artifacts/            Generated model and metrics artifacts
logs/                 Prediction logs
tests/                Pytest smoke tests
```

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset

For the public European credit card fraud dataset, place the CSV at:

```text
data/creditcard.csv
```

The expected columns are `Time`, `V1` through `V28`, `Amount`, and `Class`.

If the file is missing, the training script creates a small synthetic demo dataset at `data/demo_creditcard.csv`. This keeps the system runnable for demonstrations, but thesis results should use the public dataset.

## Train Models

```bash
python train.py
```

The best model bundle is saved to:

```text
artifacts/best_model.joblib
```

Metrics are saved to:

```text
artifacts/metrics.json
artifacts/metrics.csv
```

## Run The App

### Quick Run

```bash
python run.py
```

Open:

```text
http://127.0.0.1:5000
```

### Manual Run From A Fresh Terminal

If the server is not running, start it manually from the project folder:

```bash
cd "/Users/andyapenteng/Documents/Eben Hammond"
source .venv/bin/activate
python train.py
python run.py
```

Leave that terminal window open while using the web app. The Flask development server stops when you close the terminal or press `Ctrl+C`.

After `python run.py` starts successfully, you should see output similar to:

```text
* Running on http://127.0.0.1:5000
```

Then open this address in your browser:

```text
http://127.0.0.1:5000
```

### If The Server Does Not Start

- Make sure the virtual environment is active. Your terminal prompt may show `(.venv)`.
- If dependencies are missing, run:

```bash
pip install -r requirements.txt
```

- If port `5000` is already in use, run Flask on another port:

```bash
flask --app app run --debug --port 5001
```

Then open:

```text
http://127.0.0.1:5001
```

## Tests

```bash
pytest
```
