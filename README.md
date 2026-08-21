# Retail Fraud Classifier

Production-style, leakage-aware fraud classification for retail transactions.

The pipeline loads CSV or Excel transactions, handles mixed numeric/categorical data, tunes the F1 threshold, and writes model.joblib plus metrics.json.

    pip install -e ".[dev]"
    retail-fraud train --train-path train.xlsx --target fraud_flag --output-dir artifacts

GitHub Actions runs linting and tests independently of any notebook.

