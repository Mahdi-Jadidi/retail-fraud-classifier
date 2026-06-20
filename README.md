# Fraud Detection in Transactions

This project builds a fraud-detection classification pipeline for transaction data. It cleans categorical values, creates customer history features, preprocesses mixed numerical/categorical inputs, selects important features, and trains CatBoost-based classification ensembles.

## What the project does

- Loads transaction training and test data from Excel files.
- Cleans misspelled categorical values such as payment method, device type, location, and merchant category.
- Converts boolean-like fields into numeric values.
- Sorts transactions by timestamp.
- Builds customer history features:
  - number of previous locations
  - number of previous merchant categories
  - previous fraud count
  - average international transaction indicator
- Applies imputation, one-hot encoding, and polynomial features.
- Creates interaction features such as amount ratio and international large-amount flag.
- Trains a CatBoost classifier.
- Performs permutation feature importance.
- Selects the top 35 features.
- Trains a CatBoost stacking/ensemble model.
- Produces predicted fraud labels for the test set.

## Dataset

Training data:

- `train.xlsx` — 80,000 labeled transactions

Test data:

- `test.xlsx` — 20,000 unlabeled transactions

The notebook output shows the training data has 21 columns before preprocessing and expands to 98 engineered/preprocessed columns.

## How to run

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn catboost pyspark openpyxl
```

Then run:

```text
Classification-Model.ipynb
```

## Preprocessing

The pipeline includes:

- Categorical normalization and typo correction
- Boolean mapping for risk flags
- Numeric coercion for transaction amount, frequency, account age, and history fields
- KNN imputation
- One-hot encoding
- Polynomial feature expansion
- Customer-level history features

## Results

### CatBoost cross-validation

The notebook reports 5-fold F1 scores:

```text
CatBoost ROC-AUC: 0.7930660657047038
[0.79765005 0.79355353 0.79337316 0.7901718 0.79058179]
```

### Feature selection

The pipeline reduces the feature matrix from 98 columns to the top 35 features based on permutation importance.

### Ensemble result

A stacked CatBoost ensemble achieved:

```text
CatBoost Stacking F1: 0.7967675180391112
[0.79713547 0.79639957]
```

The final model predicts labels for **20,000** test transactions and writes them to `submission.csv` when the notebook is executed.

## Strongest signals

The correlation analysis indicates that the strongest fraud indicators include:

- `is_international`
- `multiple_transactions_short_time`
- `unusual_amount_flag`
- `high_risk_device_flag`
- transaction amount interactions
- failed transaction count interactions
- previous fraud history

## Suggested improvements

- Use time-based validation to reflect real fraud-detection deployment.
- Optimize the classification threshold for recall/F1 based on business cost.
- Add calibrated probabilities instead of hard labels.
- Compare CatBoost with XGBoost, LightGBM, and neural anomaly-detection approaches.

## Files

- `Classification-Model.ipynb` — complete fraud-detection pipeline
- `train.xlsx` — labeled training data
- `test.xlsx` — unlabeled test data
- `Project Description.pdf` — project brief
