# Retail Transaction Fraud Detection — A binary classification pipeline that predicts whether a retail transaction is fraudulent (`fraud_flag`), built around a strict no-leakage, time-aware workflow and a 4-model ensemble refined with semi-supervised pseudo-labeling.

## Problem
Given a dataset of retail transactions — including transaction details, customer activity patterns, payment method, device/location information, merchant category, and behavioral risk flags — the goal is to classify each transaction as **fraudulent (1)** or **legitimate (0)**. The dataset comes from a Kaggle-like competition where the primary evaluation metric is **F1 score**, with strict penalties for data leakage. Key constraints include:
- Using only information available before the transaction time (no leakage).
- Tuning the classification threshold on a validation set (not assuming 0.5).
- Reporting a suite of metrics: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC-AUC, PR-AUC, MCC, Cohen's Kappa, and Specificity.
- Submitting predictions as a CSV with columns `id` and `label`.

## Approach
The pipeline follows a strict chronological split (60%/20%/20%) to ensure no future data leaks into training. Features are engineered causally, including risk scores, interaction terms, ratios, and time-since-last-transaction computed only from historical data. Categorical variables are encoded in two ways: label encoding for LightGBM and CatBoost (which handle natively), and one-hot encoding for XGBoost and Random Forest. Four models are trained with early stopping, and their predictions are combined via a weighted ensemble (weights proportional to validation F1). Finally, a semi-supervised step refines predictions: high-confidence ensemble pseudo-labels are used to train a student LightGBM model, which generates the final submission.

## Results
| Model          | Val F1 | Test F1 | ROC-AUC | PR-AUC | MCC  |
|----------------|--------|---------|---------|--------|------|
| LightGBM       | 0.8166 | 0.8116  | 0.9148  | 0.9132 | 0.6347 |
| XGBoost        | 0.8165 | 0.8112  | 0.9144  | 0.9124 | 0.6341 |
| Random Forest  | 0.8164 | 0.8110  | 0.9150  | 0.9130 | 0.6339 |
| CatBoost       | 0.7961 | 0.7910  | 0.9023  | 0.8994 | 0.5917 |
| **Ensemble**   | **0.8168** | **0.8168** | **0.9151** | **0.9135** | **0.6359** |

*Note: Test F1 for the ensemble is computed on the labeled test set from the original data (for validation purposes). In a real competition setting, test labels would not be available.*

## Stack
Python · LightGBM · XGBoost · CatBoost · scikit-learn · pandas · numpy

## How to Run
```bash
pip install -r requirements.txt
python main.py
```
The script will:
1. Download the training and test data from Google Sheets (if not present).
2. Preprocess, engineer features, and encode categorical variables.
3. Train four base models (LightGBM, XGBoost, CatBoost, Random Forest).
4. Evaluate on a validation set and compute ensemble weights.
5. Generate predictions on the test set.
6. Apply pseudo-labeling and train a student model.
7. Save the final submission to `submission.csv`.

## Project Structure
```
.
├── notebooks/
│   └── original.ipynb          # Original notebook (preserved for history)
├── src/
│   ├── data_loader.py          # Loading and initial cleaning
│   ├── preprocessing.py        # Feature engineering and encoding
│   ├── model.py                # Model training functions
│   ├── evaluate.py             # Metrics and threshold tuning
│   └── predict.py              # Inference and pseudo-labeling
├── main.py                     # Orchestrates the full pipeline
├── requirements.txt            # Pinned dependencies
├── README.md                   # This file
└── .gitignore                  # Ignores large data files
```

## Design Decisions
- **Chronological split**: Prevents look-ahead bias by splitting data by transaction timestamp.
- **Causal feature engineering**: All features (e.g., `seconds_since_last_txn`) use only historical information.
- **Dual encoding**: Leverages native categorical support in LightGBM/CatBoost while maintaining compatibility with XGBoost/Random Forest.
- **Threshold tuning**: Optimizes F1 on validation set per model and for the ensemble.
- **Semi-supervised refinement**: Uses high-confidence predictions to boost performance on unlabeled test data.

## Possible Improvements
- Tune hyperparameters via optimization (e.g., Optuna).
- Add probability calibration (Platt scaling or isotonic regression).
- Use SHAP for interpretability.
- Validate pseudo-labeling with hold-out sets to avoid reinforcing bias.