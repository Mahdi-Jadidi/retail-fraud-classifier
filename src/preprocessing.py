\"\"\"
\"\""
Preprocessing module for the fraud detection pipeline.

This module handles:
1. Chronological train/validation/test split (60%/20%/20%)
2. Imputation of missing values using statistics computed only on the training set
3. Feature engineering:
   - Risk score (sum of four binary risk flags)
   - Interaction terms between risk flags
   - Amount-to-average and failure-velocity ratios
   - Log-transformed account age
   - Seconds since last transaction (computed causally using only historical data)
4. Encoding of categorical features:
   - Label encoding for LightGBM and CatBoost (native categorical support)
   - One-hot encoding for XGBoost and Random Forest

The module provides functions to process raw cleaned dataframes into feature
matrices ready for model training and evaluation.
\"\"\"
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd

# Import from data_loader to avoid circular dependencies? We'll just define the constants here.
# These should match the ones in data_loader.py
BOOL_COLS = ["is_international", "unusual_amount_flag",
             "multiple_transactions_short_time", "high_risk_device_flag"]
NUMERIC_COLS = ["transaction_amount", "transaction_frequency_24h",
                "avg_transaction_amount_7d", "failed_transaction_count_24h",
                "account_age_days"]
CATEGORICAL_COLS = ["payment_method", "device_type", "merchant_category", "location"]
ENGINEERED = [
    "risk_score", "intl_x_unusual", "multi_x_failed", "risk_device_x_intl",
    "amount_to_avg_ratio", "failure_velocity_ratio", "log_account_age",
    "seconds_since_last_txn"
]
BASE_FEATURES = NUMERIC_COLS + BOOL_COLS + ["month", "day_of_week"] + ENGINEERED


def split_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    \"\"\"
    Split a dataframe chronologically into train, validation, and test sets (60%/20%/20%).

    Parameters
    ----------
    df : pd.DataFrame
        Input dataframe sorted by transaction timestamp.

    Returns
    -------
    train_df, val_df, test_df : tuple of pd.DataFrame
        The three splits, each as a dataframe.
    \"\"\"
    n = len(df)
    train_end = int(n * 0.60)
    val_end = int(n * 0.80)
    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()
    return train_df, val_df, test_df


def impute_data(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    \"\"\"
    Impute missing values using statistics computed only from the training set.

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        The three data splits (already cleaned and sorted).

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
        The imputed dataframes.
    impute_stats : dict
        Dictionary mapping column names to the imputation value (median for numeric,
        mode for boolean/categorical) learned from the training set.
    \"\"\"
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()
    impute_stats = {}

    # Impute numeric columns with median from training set
    for col in NUMERIC_COLS:
        if col in train_df.columns:
            med = train_df[col].median()
            impute_stats[col] = med
            for df in [train_df, val_df, test_df]:
                df[col] = df[col].fillna(med)

    # Impute boolean columns with mode from training set
    for col in BOOL_COLS:
        if col in train_df.columns:
            mode_val = float(train_df[col].mode()[0])
            impute_stats[col] = mode_val
            for df in [train_df, val_df, test_df]:
                df[col] = df[col].fillna(mode_val).astype(int)

    # Impute categorical columns with mode from training set
    for col in CATEGORICAL_COLS:
        if col in train_df.columns:
            mode_val = train_df[col].mode()[0]
            impute_stats[col] = mode_val
            for df in [train_df, val_df, test_df]:
                df[col] = df[col].fillna(mode_val)

    return train_df, val_df, test_df, impute_stats


def engineer_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    \"\"\"
    Engineer features causally (using only information available at or before each transaction).

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        The three data splits (already imputed).

    Returns
    -------
    train_df, val_df, test_df : pd.DataFrame
        The dataframes with additional engineered features.
    \"\"\"
    def _add_features(df: pd.DataFrame, train_ref: pd.DataFrame = None) -> pd.DataFrame:
        df = df.copy()

        # Risk score: sum of the four binary risk flags
        df[\"risk_score\"] = (
            df[\"is_international\"] + df[\"unusual_amount_flag\"] +
            df[\"multiple_transactions_short_time\"] + df[\"high_risk_device_flag\"]
        )

        # Interaction terms
        df[\"intl_x_unusual\"] = df[\"is_international\"] * df[\"unusual_amount_flag\"]
        df[\"multi_x_failed\"] = df[\"multiple_transactions_short_time\"] * df[\"failed_transaction_count_24h\"]
        df[\"risk_device_x_intl\"] = df[\"high_risk_device_flag\"] * df[\"is_international\"]

        # Ratios
        df[\"amount_to_avg_ratio\"] = df[\"transaction_amount\"] / (df[\"avg_transaction_amount_7d\"] + 1e-5)
        df[\"failure_velocity_ratio\"] = df[\"failed_transaction_count_24h\"] / (df[\"transaction_frequency_24h\"] + 1e-5)

        # Log-transformed account age
        df[\"log_account_age\"] = np.log1p(df[\"account_age_days\"])

        # Seconds since last transaction (causal)
        if train_ref is not None:
            # For validation/test, compute using historical reference (train+val for test, train for val)
            ref = train_ref[[\"customer_id\", \"transaction_timestamp\"]].copy()
            combined = pd.concat(
                [ref, df[\"customer_id\", \"transaction_timestamp\"]],
                ignore_index=False
            ).sort_values([\"customer_id\", \"transaction_timestamp\"])
            diffs = combined.groupby(\"customer_id\")[\"transaction_timestamp\"].diff().dt.total_seconds()
            df[\"seconds_since_last_txn\"] = diffs.reindex(df.index).fillna(9_999_999)
        else:
            # For training, compute using only its own past
            tmp = df.sort_values([\"customer_id\", \"transaction_timestamp\"])
            tmp[\"seconds_since_last_txn\"] = (
                tmp.groupby(\"customer_id\")[\"transaction_timestamp\"]
                .diff().dt.total_seconds().fillna(9_999_999)
            )
            df[\"seconds_since_last_txn\"] = tmp[\"seconds_since_last_txn\"].reindex(df.index)

        return df

    # Apply feature engineering to each set, passing the appropriate reference
    train_df = _add_features(train_df, train_ref=None)
    val_df = _add_features(val_df, train_ref=train_df)
    # For test, reference is train+val
    train_val = pd.concat([train_df, val_df])
    test_df = _add_features(test_df, train_ref=train_val)

    return train_df, val_df, test_df


def encode_features(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> tuple[
    pd.DataFrame, pd.DataFrame, pd.DataFrame,  # LE encoded features
    pd.DataFrame, pd.DataFrame, pd.DataFrame,  # OHE encoded features
    dict,  # Label encoders for each categorical column
    list,  # List of LE column names
    list   # List of OHE column names (in the order they appear in the OHE matrices)
]:
    \"\"\"
    Encode categorical features in two ways:
    1. Label encoding (for LightGBM and CatBoost)
    2. One-hot encoding (for XGBoost and Random Forest)

    Parameters
    ----------
    train_df, val_df, test_df : pd.DataFrame
        The three data splits (after feature engineering).

    Returns
    -------
    X_train_le, X_val_le, X_test_le : pd.DataFrame
        Features with label-encoded categorical columns (plus base features).
    X_train_ohe, X_val_ohe, X_test_ohe : pd.DataFrame
        Features with one-hot encoded categorical columns.
    le_dict : dict
        Mapping from column name to fitted LabelEncoder.
    le_cols : list
        List of the suffix '-le' column names (e.g., ['payment_method_le', ...]).
    ohe_cols : list
        List of column names in the OHE matrices (base features + one-hot encoded columns),
        in the same order as the columns appear in the returned DataFrames.
    \"\"\"
    # Make copies to avoid modifying originals
    train_df = train_df.copy()
    val_df = val_df.copy()
    test_df = test_df.copy()

    # Label encoding
    from sklearn import preprocessing
    le_dict = {}
    le_cols = []
    for col in CATEGORICAL_COLS:
        le_col = f\"{col}_le\"
        le = preprocessing.LabelEncoder()
        # Fit on training data
        train_df[le_col] = le.fit_transform(train_df[col].astype(str))
        # Transform validation and test, handling unseen labels
        def map_label(x, le=le):
            try:
                return le.transform([x])[0]
            except ValueError:
                # Unseen label: return -1 (or could use the most frequent class?)
                return -1
        val_df[le_col] = val_df[col].astype(str).apply(map_label)
        test_df[le_col] = test_df[col].astype(str).apply(map_label)
        le_dict[col] = le
        le_cols.append(le_col)

    # Base features (numeric, boolean, month/day_of_week, engineered)
    base_features = BASE_FEATURES.copy()
    # Ensure all base features exist in the dataframes
    for col in base_features:
        if col not in train_df.columns:
            # If missing, fill with 0 (should not happen if upstream steps are correct)
            train_df[col] = 0
            val_df[col] = 0
            test_df[col] = 0

    # LE feature set: base features + label-encoded columns
    le_feature_cols = base_features + le_cols
    X_train_le = train_df[le_feature_cols].copy()
    X_val_le = val_df[le_feature_cols].copy()
    X_test_le = test_df[le_feature_cols].copy()

    # OHE feature set: base features + one-hot encoded categorical columns
    # We'll use pandas get_dummies on the categorical columns only, then combine with base features
    def _ohe_df(df: pd.DataFrame) -> pd.DataFrame:
        # One-hot encode the categorical columns
        ohe_part = pd.get_dummies(
            df[CATEGORICAL_COLS],
            columns=CATEGORICAL_COLS,
            drop_first=False,
            dtype=np.uint8
        )
        # Combine with base features
        return pd.concat([df[base_features], ohe_part], axis=1)

    X_train_ohe = _ohe_df(train_df)
    # Ensure validation and test have the same columns as training (add missing, fill with 0)
    X_val_ohe = _ohe_df(val_df).reindex(columns=X_train_ohe.columns, fill_value=0)
    X_test_ohe = _ohe_df(test_df).reindex(columns=X_train_ohe.columns, fill_value=0)

    # The OHE columns are the columns of X_train_ohe (in order)
    ohe_cols = list(X_train_ohe.columns)

    return (X_train_le, X_val_le, X_test_le,
            X_train_ohe, X_val_ohe, X_test_ohe,
            dict(le_dict), le_cols, ohe_cols)