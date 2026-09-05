# Customer Support Ticket Classification

An end-to-end NLP portfolio project that cleans support tickets, compares three text classifiers, persists the selected pipeline, and serves predictions through a responsive Flask application and REST API.

## Overview

The repository covers data generation/loading, data-quality checks, EDA, leakage-safe TF-IDF feature extraction, stratified evaluation, model persistence, inference, tests, Docker, and a reproducible notebook.

## Business Problem

Manual triage slows first response and produces inconsistent routing. The classifier predicts one of seven support categories from a ticket subject and description so an operations team can route new work sooner.

## Features

- Reproducible cleaning with duplicate, null, whitespace, and invalid-row handling
- EDA for category, priority, channel, text length, resolution, and satisfaction fields
- Logistic Regression, LinearSVC, and Multinomial Naive Bayes comparison
- Macro/weighted evaluation, class reports, confusion matrix, and model coefficients
- Saved Scikit-learn pipeline and probability-based top alternatives when supported
- Validated JSON API, health check, responsive web UI, tests, and container config

## Dataset

This build uses **2,100 synthetic tickets** (300 per category), clearly marked by `is_synthetic=true`. It is a development fallback because the build environment could read the Kaggle page but could not download the file. It contains no real people or private data.

The preferred source is [Suraj520's Customer Support Ticket Dataset](https://www.kaggle.com/datasets/suraj520/customer-support-ticket-dataset), listed by Kaggle as CC0/Public Domain with 8,469 records and 17 fields. See `data/README.md` for substitution instructions. The target is `category`; features are only `subject + description`. Post-resolution text is excluded to prevent leakage.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, Matplotlib, Flask, JavaScript, pytest, Jupyter, joblib, Waitress, Docker.

## Project Architecture

`src/` owns data and ML workflows, `app/` owns HTTP/inference presentation, `models/` and `reports/` hold versioned results, and `tests/` protects preprocessing, loading, prediction, validation, and endpoints.

## Data Preprocessing

The loader removes exact duplicates, normalizes whitespace, converts null text to empty strings, rejects missing labels and fully blank tickets, and combines subject/description. Stratification occurs before TF-IDF fitting; vectorization remains inside each pipeline.

## Exploratory Data Analysis

Generated artifacts include category and text-length distributions plus the final confusion matrix under `reports/figures/`. The notebook additionally inspects dtypes, nulls, duplicates, priority, channel, and category-by-priority proportions.

## Machine Learning Pipeline

`TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1,2), min_df=2, max_df=.98, max_features=12000)` feeds each candidate model. The test split is 20%, stratified, with `random_state=42`. Selection uses macro F1 rather than accuracy alone.

## Models Compared

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Logistic Regression | 1.000 | 1.000 | 1.000 |
| LinearSVC | 1.000 | 1.000 | 1.000 |
| Multinomial Naive Bayes | 1.000 | 1.000 | 1.000 |

## Evaluation Results

Logistic Regression was selected by deterministic tie order and supports calibrated-looking class probabilities through `predict_proba` (these probabilities are model outputs, not a claim of empirical calibration). Accuracy, macro F1, and weighted F1 are **1.000 on 420 held-out synthetic records**. This perfect result reflects easily separable generated templates and **must not be presented as real-world performance**. Full precision, recall, F1, support, and class reports are in `reports/model_metrics.json`.

## Sample Prediction

Input: “Password reset failed — my account is locked and I cannot login.” Expected model route: `Account Access`; confidence is computed at request time.

## Web Application

The SaaS-style interface includes accessible labels, a sample ticket, loading/error states, character limits, responsive layout, the predicted category, confidence, and alternative categories.

## API

`POST /api/predict` with JSON `{"subject":"Unable to login","description":"My reset code expired."}`. Responses contain `prediction` and, for the selected model, `confidence` and `alternatives`. `GET /health` returns `{"status":"ok"}`. Invalid media types, JSON, fields, value types, blank input, and input over 5,000 characters receive 4xx responses.

## Installation and Local Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.train
python run.py
```

Open `http://127.0.0.1:5000`.

## Running the Notebook

Install Jupyter (`python -m pip install jupyter`), start `jupyter notebook`, and run `notebooks/customer_support_ticket_classification.ipynb` top to bottom from the repository root or notebook folder.

## Running Tests

```bash
python -m pytest -q
```

## Docker

```bash
docker build -t ticket-classifier .
docker run --rm -p 5000:5000 ticket-classifier
```

## Project Structure

```text
app/           Flask factory, predictor, templates, CSS, JavaScript
data/          synthetic fallback, processed data, source documentation
models/        serialized pipeline and metadata
notebooks/     reproducible analysis workflow
reports/       metrics JSON and figures
src/           cleaning, EDA, features, training, evaluation
tests/         preprocessing, model, and API tests
```

## Key Insights

- Balanced generation makes accuracy and macro F1 agree; production imbalance may not.
- Category-specific keywords dominate this synthetic benchmark, explaining perfect separation.
- Logistic Regression supplies useful class probabilities while matching the strongest measured macro F1.

## Limitations

The included tickets are templated and English-only, and the test split shares the generator's vocabulary. Metrics therefore validate implementation, not production generalization. Confidence has not undergone an independent calibration study. Real deployment requires representative labeled traffic, privacy review, drift monitoring, and human fallback.

## Future Improvements

Retrain on the documented CC0 data, deduplicate semantically before splitting, add temporal/out-of-domain evaluation, calibrate probabilities, monitor drift, and add human feedback for uncertain predictions.

## Deployment

`render.yaml`, `Procfile`, and `Dockerfile` are ready. On Render: create a Blueprint from this repository, accept the detected web service, and deploy. The `/health` route is configured as the health check.

## Author

Jatin Gohel — [GitHub](https://github.com/jatinit056)

## Resume Highlights

- Built an end-to-end NLP ticket-routing pipeline with Pandas, Scikit-learn, TF-IDF, and three classification algorithms across seven synthetic development categories.
- Implemented leakage-safe preprocessing, stratified evaluation, macro/weighted metrics, confusion matrices, and model explainability; achieved 1.000 held-out macro F1 on an explicitly synthetic 420-record test set.
- Developed a responsive Flask application and validated REST API with probability-based predictions, strict request validation, health monitoring, and persistent joblib artifacts.
- Added a reproducible Jupyter workflow, automated pytest coverage, Docker/Render deployment configuration, and transparent dataset/provenance documentation.
