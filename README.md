# Retail Fraud Classifier

Leakage-aware fraud detection for retail transactions, designed around the reality that fraud labels are imbalanced and a default probability threshold is rarely the operating point a business needs.

## Problem

The task is binary classification: identify fraudulent transactions from behavioural, device, payment, merchant, location, and time information. The pipeline prioritizes F1 because both missed fraud and unnecessary review queues are costly.

## What was achieved

The original benchmark compared LightGBM, XGBoost, Random Forest, and CatBoost under a time-aware, no-leakage workflow. The weighted ensemble achieved **F1 0.8168**, **ROC-AUC 0.9151**, and **PR-AUC 0.9135** on the labelled evaluation split. These are retained as the historical benchmark; every new run produces its own metrics artifact.

## Current production pipeline

- Loads Excel or CSV data and validates/infer the target column.
- Derives calendar features from timestamp-like columns without retaining raw timestamps as arbitrary strings.
- Imputes numeric and categorical variables inside a scikit-learn pipeline.
- Uses a stratified hold-out split, trains a gradient-boosted classifier, and tunes the decision threshold on validation F1.
- Saves a portable `model.joblib` and `metrics.json` with precision, recall, F1, PR-AUC, and ROC-AUC where valid.

## Reproduce

```bash
pip install -e ".[dev]"
retail-fraud train --train-path train.xlsx --target fraud_flag --output-dir artifacts
```

The canonical implementation is under `src/retail_fraud`. CI runs linting and tests on every push.
