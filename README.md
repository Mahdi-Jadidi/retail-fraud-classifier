# Retail Transaction Fraud Detection

A binary classification pipeline that predicts whether a retail transaction is fraudulent (`fraud_flag`), built around a strict **no-leakage, time-aware** workflow and a **4-model ensemble** refined with **semi-supervised pseudo-labeling**.

This project was built for a classification competition (Quera) where submissions are scored primarily on **F1**, with explicit penalties for data leakage.

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Models](#models)
- [Results](#results)
- [Feature Importance](#feature-importance)
- [Semi-Supervised Refinement](#semi-supervised-refinement)
- [Setup & Usage](#setup--usage)
- [Repository Structure](#repository-structure)
- [Design Decisions](#design-decisions)
- [Possible Improvements](#possible-improvements)

---

## Problem Statement

Given a dataset of retail transactions — including transaction details, customer activity patterns, payment method, device/location information, merchant category, and behavioral risk flags — the goal is to classify each transaction as **fraudulent (1)** or **legitimate (0)**.

Key constraints set by the competition brief:

- **F1 score** is the primary evaluation metric (not accuracy, due to class imbalance).
- **No data leakage**: features must only use information available *before* the transaction time; any preprocessing/feature engineering that peeks into the future or into validation/test data is disqualifying.
- **Classification threshold is not assumed to be 0.5** — it should be tuned on a validation set.
- A full suite of metrics is reported for transparency: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC, Specificity, MCC, and Cohen's Kappa.
- Final predictions are submitted as a CSV with two columns: `id`, `label`.

## Dataset

| Split | Rows | Columns | Notes |
|---|---|---|---|
| Labeled data | 80,000 | 17 raw → 45 engineered (LE) / 23 one-hot expanded | Loaded from `train.xlsx` |
| Unlabeled test data | 20,000 | 17 raw | Loaded from `student_test.xlsx`; predictions submitted as `submission.csv` |

Raw fields include: `customer_id`, `transaction_timestamp`, `transaction_amount`, `payment_method`, `device_type`, `merchant_category`, `location`, `transaction_frequency_24h`, `avg_transaction_amount_7d`, `failed_transaction_count_24h`, `account_age_days`, plus four boolean risk flags (`is_international`, `unusual_amount_flag`, `multiple_transactions_short_time`, `high_risk_device_flag`) and the target `fraud_flag` / `label`.

The labeled fraud rate is roughly **48%**, so while the classes aren't extremely imbalanced, F1-based threshold tuning still meaningfully outperforms a default 0.5 cutoff.

> **Note:** the notebook fetches `train.xlsx` and `student_test.xlsx` directly from Google Sheets export links at runtime (`wget`). To reproduce results, you'll need access to those source files (or point the notebook at your own copies with the same schema).

## Pipeline

### 1. Cleaning
- Free-text categorical fields (`payment_method`, `device_type`, `merchant_category`, `location`) are normalized through hand-written mapping functions that collapse typos and inconsistent casing (e.g. `"moblie"` → `Mobile`, `"untied kingdom"` → `UK`) into a fixed set of categories.
- Boolean risk flags are mapped from mixed string representations (`true/false`, `yes/no`, `1/0`) to numeric.
- Numeric fields are coerced to numeric and invalid negative values are nulled out before imputation.

### 2. Leakage-safe split & imputation
- Transactions are sorted by timestamp, then split **chronologically** into train / validation / test (60% / 20% / 20%) — not randomly — so the model is always evaluated on transactions that occur *after* what it was trained on.
- Imputation statistics (median for numeric, mode for categorical/boolean) are computed **on the training split only** and then applied to validation and test, preventing any information from leaking backward.

### 3. Feature engineering
All engineered features are computed causally (using only information available at or before the transaction):

| Feature | Description |
|---|---|
| `risk_score` | Sum of the four binary risk flags (0–4) |
| `intl_x_unusual`, `multi_x_failed`, `risk_device_x_intl` | Pairwise interaction terms between risk flags |
| `amount_to_avg_ratio` | Transaction amount relative to the customer's 7-day average |
| `failure_velocity_ratio` | Failed transactions relative to transaction frequency |
| `log_account_age` | Log-transformed account age |
| `seconds_since_last_txn` | Time since the customer's previous transaction, computed per-customer using only *prior* transactions (validation/test use the training set as historical reference, never their own future rows) |

### 4. Encoding
Two parallel feature representations are built:
- **Label-encoded (LE)** categorical features, used by LightGBM and CatBoost (both support native categorical handling).
- **One-hot encoded (OHE)** features, used by XGBoost and Random Forest.

## Models

Four classifiers are trained independently, each with early stopping on the validation set and an F1-optimal decision threshold (swept over `[0.1, 0.9]`):

- **LightGBM** (`LGBMClassifier`)
- **XGBoost** (`XGBClassifier`)
- **CatBoost** (`CatBoostClassifier`)
- **Random Forest** (`RandomForestClassifier`, `class_weight="balanced"`)

A **weighted ensemble** then combines all four, with weights proportional to each model's validation F1 score.

## Results

### Individual models (test set, threshold tuned on validation F1)

| Model | Val F1 | Test F1 | ROC-AUC | PR-AUC | MCC |
|---|---|---|---|---|---|
| **LightGBM** | 0.8166 | 0.8116 | 0.9148 | 0.9132 | 0.6347 |
| XGBoost | 0.8165 | 0.8112 | 0.9144 | 0.9124 | 0.6341 |
| Random Forest | 0.8164 | 0.8110 | 0.9150 | 0.9130 | 0.6339 |
| CatBoost | 0.7961 | 0.7910 | 0.9023 | 0.8994 | 0.5917 |

LightGBM, XGBoost, and Random Forest perform almost identically and clearly outperform CatBoost on this dataset under default-ish hyperparameters.

### Weighted ensemble

| | Value |
|---|---|
| Ensemble weights | LightGBM 0.252, XGBoost 0.252, Random Forest 0.252, CatBoost 0.245 |
| Validation F1 | **0.8168** |
| Tuned threshold | 0.409 |

The ensemble edges out the best single model (LightGBM) by a small margin, which is expected given how correlated the four models' predictions already are.

## Feature Importance

Top drivers of fraud predictions, by LightGBM information gain:

| Rank | Feature | Gain |
|---|---|---|
| 1 | `risk_score` | 506,043 |
| 2 | `failed_transaction_count_24h` | 115,820 |
| 3 | `is_international` | 62,053 |
| 4 | `multi_x_failed` | 59,302 |
| 5 | `failure_velocity_ratio` | 59,226 |
| 6 | `unusual_amount_flag` | 44,982 |
| 7 | `transaction_frequency_24h` | 35,372 |
| 8 | `intl_x_unusual` | 25,836 |
| 9 | `high_risk_device_flag` | 18,576 |
| 10 | `multiple_transactions_short_time` | 15,558 |

The aggregate `risk_score` feature dominates by a wide margin, followed by transaction-failure velocity and the international/device risk flags — suggesting the four raw risk flags (and their interactions) carry most of the predictive signal, with engineered ratios adding a meaningful secondary contribution.

## Semi-Supervised Refinement

To squeeze additional signal out of the unlabeled 20,000-row test set before generating the final submission, the pipeline applies a simple **self-training** step:

1. All four models are refit on the **full** 80,000-row labeled dataset using the iteration counts found during early stopping.
2. The weighted ensemble scores the unlabeled test set.
3. Rows where the ensemble is highly confident (`p ≥ 0.85` fraud or `p ≤ 0.05` legitimate) are pseudo-labeled — in practice this captures ~10,900 of the 20,000 rows.
4. A "student" LightGBM model is retrained on the original labeled data (full weight) plus the pseudo-labeled rows (down-weighted to 0.4), and its predictions — thresholded at the ensemble's tuned cutoff — produce the final `submission.csv`.

This is a pragmatic way to leverage the large pool of unlabeled test data without fully trusting it, but it's worth noting pseudo-labeling can reinforce the base ensemble's existing biases — see [Possible Improvements](#possible-improvements).

## Setup & Usage

### Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost catboost
```

### Running

1. Provide `train.xlsx` (80,000 labeled rows) and `student_test.xlsx` (20,000 unlabeled rows) — either let the notebook download them from the configured Google Sheets links, or place your own copies with a matching schema in the working directory and update `RAW_PATH` / `TEST_PATH`.
2. Run `Clasification_Model.ipynb` top to bottom.
3. The final cell writes predictions to `submission.csv` with columns `id` and `label`.

## Repository Structure

```
.
├── Clasification_Model.ipynb   # Full pipeline: cleaning → features → models → ensemble → submission
├── Project_Description.pdf     # Original competition brief
└── README.md
```

## Design Decisions

- **Chronological split over random split.** Fraud detection is inherently a forecasting problem — a random split would let the model implicitly learn from "future" transactions, inflating offline metrics in a way that wouldn't hold up in production.
- **Per-customer history computed causally.** `seconds_since_last_txn` for validation/test rows is computed against a reference set restricted to transactions known at that point in time, never against the row's own split.
- **Threshold tuned per model, not fixed at 0.5.** Given the F1-centric scoring and the class balance being close to (but not exactly) 50/50, the optimal threshold consistently lands well below 0.5 (~0.35–0.48) for all four models.
- **Two encodings, not one.** Native categorical support in LightGBM/CatBoost is more efficient and often more accurate than one-hot encoding for high-cardinality categories, while XGBoost/Random Forest here use the one-hot representation for compatibility.

## Possible Improvements

- Validate the pseudo-labeling step more rigorously (e.g. confirm it improves held-out F1 rather than just trusting ensemble confidence) before relying on it for the final submission.
- Try probability calibration (Platt scaling / isotonic regression) on top of the raw ensemble outputs.
- Tune hyperparameters (e.g. via Optuna) rather than using a single fixed configuration per model.
- Add SHAP-based explanations for individual predictions, not just global gain-based importance.
- Stress-test the categorical-cleaning functions against unseen typo patterns that weren't present in this particular dataset.
