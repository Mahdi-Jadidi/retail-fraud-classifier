\"\"\"
\"\""
Data loading and initial cleaning module for the fraud detection pipeline.

This module handles downloading the raw Excel files (if needed), loading them into
pandas DataFrames, and performing initial cleaning steps:
- Boolean column mapping
- Numeric coercion and non-negative enforcement
- Categorical text cleaning (payment method, device, merchant category, location)
- Timestamp parsing and extraction of month/day-of-week features
- Chronological sorting by transaction timestamp

The cleaned DataFrames retain all original columns plus the engineered month and
day_of_week columns, ready for feature engineering and preprocessing.
\"\"\"
import os
from typing import Tuple
import numpy as np
import pandas as pd

# URLs for the raw data hosted on Google Sheets (exported as Excel)
TRAIN_URL = (
    "https://docs.google.com/spreadsheets/d/12Ob7YOCjxh3nzvvJB1OoCWoW9cmVNlK8/"
    "export?format=xlsx"
)
TEST_URL = (
    "https://docs.google.com/spreadsheets/d/1gJPC23TrfpKWxeK8OKXMrRqT8_hz17PF/"
    "export?format=xlsx"
)

# Expected local filenames
TRAIN_PATH = "train.xlsx"
TEST_PATH = "student_test.xlsx"

# Boolean columns and their mapping
BOOL_COLS = ["is_international", "unusual_amount_flag",
             "multiple_transactions_short_time", "high_risk_device_flag"]
BOOL_MAP = {"true": 1, "false": 0, "yes": 1, "no": 0,
            "y": 1, "n": 0, "1": 1, "0": 0}

# Numeric columns that should be non-negative
NUMERIC_COLS = ["transaction_amount", "transaction_frequency_24h",
                "avg_transaction_amount_7d", "failed_transaction_count_24h",
                "account_age_days"]

# Categorical columns to clean
CATEGORICAL_COLS = ["payment_method", "device_type", "merchant_category", "location"]

# Location mapping dictionary (lowercase keys)
LOCATION_MAP = {
    "usa": "USA", "us": "USA", "u.s.a.": "USA",
    "canada": "Canada", "ca": "Canada", "canadaa": "Canada",
    "uk": "UK", "united kingdom": "UK", "u.k.": "UK", "untied kingdom": "UK",
    "australia": "Australia", "au": "Australia", "austraila": "Australia",
    "india": "India", "in": "India", "indai": "India",
    "germany": "Germany", "de": "Germany",
    "france": "France", "brazil": "Brazil", "china": "China", "japan": "Japan",
}


def _download_if_missing(url: str, dest_path: str) -> None:
    \"\"\"Download a file from a URL if it does not already exist locally.\"\"\"
    if not os.path.exists(dest_path):
        # Use wget via subprocess for simplicity and to match notebook behavior
        import subprocess
        subprocess.run(
            ["wget", "--no-check-certificate", url, "-O", dest_path],
            check=False,  # Don't raise exception on non-zero exit
        )


def _clean_payment(s: str) -> str:
    \"\"\"Clean payment method strings.\"\"\"
    if pd.isna(s):
        return np.nan
    s = str(s).strip().lower().replace(" ", "").replace("_", "")
    if "paypal" in s:
        return "PayPal"
    if "applepay" in s:
        return "Apple Pay"
    if "googlepay" in s:
        return "Google Pay"
    if "debit" in s:
        return "Debit Card"
    if "credit" in s:
        return "Credit Card"
    return "Other"


def _clean_device(s: str) -> str:
    \"\"\"Clean device type strings.\"\"\"
    if pd.isna(s):
        return np.nan
    s = str(s).strip().lower().replace(" ", "").replace("_", "")
    if "mobile" in s or s == "moblie":
        return "Mobile"
    if "tablet" in s or s == "tabelt":
        return "Tablet"
    if "desktop" in s or s == "desktp":
        return "Desktop"
    if "console" in s:
        return "Console"
    if "tv" in s:
        return "Smart TV"
    if "wearable" in s:
        return "Wearable"
    return "Other"


def _clean_merchant(s: str) -> str:
    \"\"\"Clean merchant category strings.\"\"\"
    if pd.isna(s):
        return np.nan
    s = str(s).strip().lower()
    if "electron" in s:
        return "Electronics"
    if "fashion" in s or "fashoin" in s:
        return "Fashion"
    if "grocer" in s:
        return "Groceries"
    if "gaming" in s or "gameing" in s:
        return "Gaming"
    if "luxury" in s:
        return "Luxury"
    if "travel" in s:
        return "Travel"
    return "Other"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    \"\""
    Load and clean the training and test datasets.

    Returns
    -------
    train_df : pd.DataFrame
        Cleaned training data with shape (80000, 17 original + 2 engineered).
        Includes the target column 'label' (fraud_flag).
    test_df : pd.DataFrame
        Cleaned test data with shape (20000, 17 original + 2 engineered).
        Does NOT include a target column.

    Both DataFrames have the following processed columns:
        - Original columns cleaned as described above
        - 'transaction_timestamp' as datetime
        - 'month' (1-12) and 'day_of_week' (0-6) extracted from timestamp
        - Rows sorted chronologically by 'transaction_timestamp'
    \"\"\"
    # Ensure data files are present
    _download_if_missing(TRAIN_URL, TRAIN_PATH)
    _download_if_missing(TEST_URL, TEST_PATH)

    # Load raw data
    train_df = pd.read_excel(TRAIN_PATH)
    test_df = pd.read_excel(TEST_PATH)

    # Define a helper to apply cleaning steps to a DataFrame
    def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # Boolean columns: map strings to 0/1, fill missing with 0
        for col in BOOL_COLS:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .map(BOOL_MAP)
                    .astype("float32")  # Use float32 to match notebook
                )

        # Numeric columns: coerce to numeric, set negatives to NaN
        for col in NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df[col] = df[col].where(df[col] >= 0)

        # Categorical columns: apply cleaning functions
        df["payment_method"] = df["payment_method"].apply(_clean_payment)
        df["device_type"] = df["device_type"].apply(_clean_device)
        df["merchant_category"] = df["merchant_category"].apply(_clean_merchant)
        df["location"] = (
            df["location"]
            .astype(str)
            .str.strip()
            .str.lower()
            .map(LOCATION_MAP)
            .fillna("Other")
        )

        # Timestamp parsing
        if "transaction_timestamp" in df.columns:
            df["transaction_timestamp"] = pd.to_datetime(
                df["transaction_timestamp"], errors="coerce"
            )
            # Extract month and day-of-week
            df["month"] = df["transaction_timestamp"].dt.month
            df["day_of_week"] = df["transaction_timestamp"].dt.dayofweek

        # Sort by timestamp chronologically
        if "transaction_timestamp" in df.columns:
            df = df.sort_values("transaction_timestamp").reset_index(drop=True)

        return df

    # Clean both dataframes
    train_df = _clean_df(train_df)
    test_df = _clean_df(test_df)

    return train_df, test_df