\"\"\"
\"\""
Main pipeline script for the fraud detection project.

This script orchestrates the entire workflow:
1. Load and clean the data
2. Split into train/validation/test
3. Impute missing values
4. Engineer features
5. Encode categorical variables
6. Train base models
7. Evaluate models and compute ensemble weights
8. Generate base predictions on test set
9. Apply pseudo-labeling and train student model
10. Produce final submission.csv
\"\"\"
import pandas as pd
import numpy as np
from pathlib import Path

# Import our custom modules
from src import data_loader
from src import preprocessing
from src import model
from src import evaluate
from src import predict

def main():
    print("Loading data...")
    train_df, test_df_raw = data_loader.load_data()
    print(f"Training data shape: {train_df.shape}")
    print(f"Test data shape: {test_df_raw.shape}")

    # Step 1: Split data chronologically (60%/20%/20%)
    print("\nSplitting data into train/validation/test sets...")
    train_val_test = preprocessing.split_data(train_df)
    train_raw, val_raw, test_raw = train_val_test
    print(f"Train: {len(train_raw)}, Val: {len(val_raw)}, Test: {len(test_raw)}")

    # Step 2: Impute missing values (using statistics from train set only)
    print("\nImputing missing values...")
    train_imp, val_imp, test_imp, impute_stats = preprocessing.impute_data(
        train_raw, val_raw, test_raw
    )
    print("Imputation complete.")

    # Step 3: Feature engineering
    print("\nEngineering features...")
    train_feat, val_feat, test_feat = preprocessing.engineer_features(
        train_imp, val_imp, test_imp
    )
    print("Feature engineering complete.")

    # Step 4: Encode categorical features
    print("\nEncoding categorical features...")
    (X_train_le, X_val_le, X_test_le,
     X_train_ohe, X_val_ohe, X_test_ohe,
     le_dict, le_cols, ohe_cols) = preprocessing.encode_features(
        train_feat, val_feat, test_feat
    )
    print(f"LE features shape: {X_train_le.shape}")
    print(f"OHE features shape: {X_train_ohe.shape}")

    # Prepare target arrays
    y_train = train_feat['label'].values
    y_val = val_feat['label'].values
    y_test = test_feat['label'].values  # Note: test set has labels in the raw data, but in competition it wouldn't
    # We keep it here for evaluation only.

    # Define feature lists for later use
    base_features = preprocessing.BASE_FEATURES
    all_features_le = list(base_features) + le_cols
    numeric_cols = preprocessing.NUMERIC_COLS
    bool_cols = preprocessing.BOOL_COLS
    categorical_cols = preprocessing.CATEGORICAL_COLS

    # Step 5: Train base models
    print("\nTraining base models...")
    # LightGBM
    print("Training LightGBM...")
    lgb_model = model.train_lightgbm(
        X_train_le, y_train,
        X_val_le, y_val,
        categorical_feature=le_cols,
        random_state=42
    )
    # XGBoost
    print("Training XGBoost...")
    xgb_model = model.train_xgboost(
        X_train_ohe, y_train,
        X_val_ohe, y_val,
        random_state=42
    )
    # CatBoost
    print("Training CatBoost...")
    cat_model = model.train_catboost(
        X_train_le, y_train,
        X_val_le, y_val,
        cat_features=le_cols,
        random_state=42
    )
    # Random Forest
    print("Training Random Forest...")
    rf_model = model.train_random_forest(
        X_train_ohe, y_train,
        random_state=42
    )

    models = {
        'lightgbm': lgb_model,
        'xgboost': xgb_model,
        'catboost': cat_model,
        'random_forest': rf_model
    }

    # Step 6: Evaluate models on validation set and compute ensemble weights
    print("\nEvaluating models on validation set...")
    val_probs = {}
    val_f1_scores = {}

    # LightGBM
    lgb_val_probs = lgb_model.predict_proba(X_val_le)[:, 1]
    lgb_thresh, lgb_val_f1 = evaluate.tune_threshold_f1(y_val, lgb_val_probs)
    val_probs['lightgbm'] = lgb_val_probs
    val_f1_scores['lightgbm'] = lgb_val_f1
    print(f"LightGBM - Val F1: {lgb_val_f1:.4f} at threshold {lgb_thresh:.3f}")

    # XGBoost
    xgb_val_probs = xgb_model.predict_proba(X_val_ohe)[:, 1]
    xgb_thresh, xgb_val_f1 = evaluate.tune_threshold_f1(y_val, xgb_val_probs)
    val_probs['xgboost'] = xgb_val_probs
    val_f1_scores['xgboost'] = xgb_val_f1
    print(f"XGBoost - Val F1: {xgb_val_f1:.4f} at threshold {xgb_thresh:.3f}")

    # CatBoost
    cat_val_probs = cat_model.predict_proba(X_val_le)[:, 1]
    cat_thresh, cat_val_f1 = evaluate.tune_threshold_f1(y_val, cat_val_probs)
    val_probs['catboost'] = cat_val_probs
    val_f1_scores['catboost'] = cat_val_f1
    print(f"CatBoost - Val F1: {cat_val_f1:.4f} at threshold {cat_thresh:.3f}")

    # Random Forest
    rf_val_probs = rf_model.predict_proba(X_val_ohe)[:, 1]
    rf_thresh, rf_val_f1 = evaluate.tune_threshold_f1(y_val, rf_val_probs)
    val_probs['random_forest'] = rf_val_probs
    val_f1_scores['random_forest'] = rf_val_f1
    print(f"Random Forest - Val F1: {rf_val_f1:.4f} at threshold {rf_thresh:.3f}")

    # Compute ensemble weights (proportional to validation F1)
    weights = evaluate.compute_ensemble_weights(val_f1_scores)
    print("\nEnsemble weights:")
    for name, w in weights.items():
        print(f"  {name}: {w:.4f}")

    # Evaluate ensemble on validation set
    ens_val_probs = evaluate.ensemble_predictions(val_probs, weights)
    ens_thresh, ens_val_f1 = evaluate.tune_threshold_f1(y_val, ens_val_probs)
    print(f"\nEnsemble - Val F1: {ens_val_f1:.4f} at threshold {ens_thresh:.3f}")

    # Step 7: Generate base predictions on test set (using the raw test data for submission)
    # We need to preprocess the raw test data (test_df_raw) in the same way as the training data,
    # but without using the test labels (which we don't have in a real scenario).
    # We'll create a function to process raw data using the fitted preprocessors.

    print("\nPreprocessing test data for submission...")
    # We'll use the same preprocessing steps but on the raw test data.
    # We have the imputation statistics, label encoders, and we need to compute features
    # using the training data as reference (to avoid leakage).

    # First, we need to combine the training and validation data as reference for feature engineering
    # (as done in the notebook for the test set processing).
    train_val_ref = pd.concat([train_feat, val_feat], ignore_index=True)

    # Define a function to process raw data (similar to the notebook's preprocess_submission)
    def preprocess_submission(raw_df, impute_stats, le_dict, train_ref_df,
                              all_features_le, base_features, categorical_cols,
                              le_cols, numeric_cols, bool_cols):
        """Preprocess raw test data for prediction."""
        df = raw_df.copy()
        original_index = df.index
        # We'll reset index to avoid issues, then restore at the end
        df.index = [f"test_idx_{i}" for i in range(len(df))]

        # Boolean columns
        bool_map = {'true':1,'false':0,'yes':1,'no':0,'y':1,'n':0,'1':1,'0':0}
        for col in bool_cols:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .strip()
                    .lower()
                    .map(bool_map)
                    .fillna(impute_stats[col])
                    .astype(int)
                )

        # Numeric columns
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                df[col] = df[col].where(df[col] >= 0).fillna(impute_stats[col])

        # Categorical columns: cleaning (same functions as in data_loader)
        def clean_payment(s):
            if pd.isna(s): return np.nan
            s = str(s).strip().lower().replace(' ','').replace('_','')
            if 'paypal' in s: return 'PayPal'
            if 'applepay' in s: return 'Apple Pay'
            if 'googlepay' in s: return 'Google Pay'
            if 'debit' in s: return 'Debit Card'
            if 'credit' in s: return 'Credit Card'
            return 'Other'

        def clean_device(s):
            if pd.isna(s): return np.nan
            s = str(s).strip().lower().replace(' ','').replace('_','')
            if 'mobile' in s or s == 'moblie': return 'Mobile'
            if 'tablet' in s or s == 'tabelt': return 'Tablet'
            if 'desktop' in s or s == 'desktp': return 'Desktop'
            if 'console' in s: return 'Console'
            if 'tv' in s: return 'Smart TV'
            if 'wearable' in s: return 'Wearable'
            return 'Other'

        def clean_merchant(s):
            if pd.isna(s): return np.nan
            s = str(s).strip().lower()
            if 'electron' in s: return 'Electronics'
            if 'fashion' in s or 'fashoin' in s: return 'Fashion'
            if 'grocer' in s: return 'Groceries'
            if 'gaming' in s or 'gameing' in s: return 'Gaming'
            if 'luxury' in s: return 'Luxury'
            if 'travel' in s: return 'Travel'
            return 'Other'

        location_map = {
            'usa':'USA','us':'USA','u.s.a.':'USA',
            'canada':'Canada','ca':'Canada','canadaa':'Canada',
            'uk':'UK','united kingdom':'UK','u.k.':'UK','untied kingdom':'UK',
            'australia':'Australia','au':'Australia','austraila':'Australia',
            'india':'India','in':'India','indai':'India',
            'germany':'Germany','de':'Germany',
            'france':'France','brazil':'Brazil','china':'China','japan':'Japan',
        }

        df['payment_method'] = df['payment_method'].apply(clean_payment)
        df['device_type'] = df['device_type'].apply(clean_device)
        df['merchant_category'] = df['merchant_category'].apply(clean_merchant)
        df['location'] = (
            df['location']
            .astype(str)
            .strip()
            .str.lower()
            .map(location_map)
            .fillna('Other')
        )

        # Timestamp
        if 'transaction_timestamp' in df.columns:
            df['transaction_timestamp'] = pd.to_datetime(
                df['transaction_timestamp'], errors='coerce'
            )
            df.loc[:, 'month'] = df['transaction_timestamp'].dt.month
            df.loc[:, 'day_of_week'] = df['transaction_timestamp'].dt.dayofweek

        # Feature engineering (causal, using train_ref as historical reference)
        def add_features(df, train_ref):
            df = df.copy()
            # Risk score
            df['risk_score'] = (
                df['is_international'] + df['unusual_amount_flag'] +
                df['multiple_transactions_short_time'] + df['high_risk_device_flag']
            )
            # Interactions
            df['intl_x_unusual'] = df['is_international'] * df['unusual_amount_flag']
            df['multi_x_failed'] = df['multiple_transactions_short_time'] * df['failed_transaction_count_24h']
            df['risk_device_x_intl'] = df['high_risk_device_flag'] * df['is_international']
            # Ratios
            df['amount_to_avg_ratio'] = df['transaction_amount'] / (df['avg_transaction_amount_7d'] + 1e-5)
            df['failure_velocity_ratio'] = df['failed_transaction_count_24h'] / (df['transaction_frequency_24h'] + 1e-5)
            # Log account age
            df['log_account_age'] = np.log1p(df['account_age_days'])
            # Seconds since last transaction (causal)
            if train_ref is not None:
                ref = train_ref[['customer_id', 'transaction_timestamp']].copy()
                combined = pd.concat(
                    [ref, df[['customer_id', 'transaction_timestamp']]],
                    ignore_index=False
                ).sort_values(['customer_id', 'transaction_timestamp'])
                diffs = combined.groupby('customer_id')['transaction_timestamp'].diff().dt.total_seconds()
                df['seconds_since_last_txn'] = diffs.reindex(df.index).fillna(9_999_999)
            else:
                # Fallback: use the dataframe's own history (less ideal but functional)
                tmp = df.sort_values(['customer_id', 'transaction_timestamp'])
                tmp['seconds_since_last_txn'] = (
                    tmp.groupby('customer_id')['transaction_timestamp']
                    .diff().dt.total_seconds().fillna(9_999_999)
                )
                df['seconds_since_last_txn'] = tmp['seconds_since_last_txn'].reindex(df.index)
            return df

        df = add_features(df, train_ref=train_ref_df)

        # Label encoding
        for col in categorical_cols:
            if col in df.columns:
                le = le_dict[col]
                df[col+'_le'] = df[col].astype(str).map(
                    lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
                )

        # Feature arrays
        # LE features: base_features + le_cols
        X_le = df[all_features_le].copy()
        # OHE features: we need to one-hot encode the categorical columns and combine with base_features
        base_df = df[base_features].copy()
        ohe_part = pd.get_dummies(
            df[categorical_cols],
            columns=categorical_cols,
            drop_first=False,
            dtype=np.uint8
        )
        # Ensure the OHE columns match the training set (we assume ohe_cols is the list of columns from training)
        X_ohe = pd.concat([base_df, ohe_part], axis=1).reindex(columns=ohe_cols, fill_value=0)

        # Restore original index
        X_le.index = original_index
        X_ohe.index = original_index
        # Subset ID: we assume the raw data has an 'id' column (as in the test set)
        sub_id = df.get('id', df.get('transaction_id', pd.RangeIndex(len(df))))
        sub_id.index = original_index

        return X_le, X_ohe, sub_id

    # Preprocess the raw test data (test_df_raw) using the training+validation as reference
    X_sub_le, X_sub_ohe, sub_ids = preprocess_submission(
        test_df_raw, impute_stats, le_dict, train_val_ref,
        all_features_le, base_features, categorical_cols,
        le_cols, numeric_cols, bool_cols
    )

    # Step 8: Get base predictions from the models on the submission data
    sub_probs = {}
    sub_probs['lightgbm'] = models['lightgbm'].predict_proba(X_sub_le)[:, 1]
    sub_probs['xgboost'] = models['xgboost'].predict_proba(X_sub_ohe)[:, 1]
    sub_probs['catboost'] = models['catboost'].predict_proba(X_sub_le)[:, 1]
    sub_probs['random_forest'] = models['random_forest'].predict_proba(X_sub_ohe)[:, 1]

    # Ensemble the base predictions
    ensemble_sub_probs = predict.ensemble_predictions(sub_probs, weights)

    # Step 9: Apply pseudo-labeling and student model to the submission data
    # We'll use the predict.refine_with_pseudo_labeling function.
    # We need to provide:
    #   - Training data (X_train_le, X_train_ohe, y_train)
    #   - Test data (X_sub_le, X_sub_ohe)
    #   - Base model probabilities on test set (we have ensemble_sub_probs, but the function expects model probabilities?
    #     Actually, the function expects test_probs which are the base model probabilities (ensemble) for the test set.)
    #   - The trained base models (to retrain on full data for the student model)
    #   - Feature column lists
    #   - Ensemble weights
    #   - Thresholds
    #
    # Note: The refine_with_pseudo_labeling function in predict.py expects:
    #   test_probs: the base model probabilities on the test set (which we have as ensemble_sub_probs)
    #   base_models: the trained base models (we have the models dict)
    #   ... and so on.

    final_preds = predict.refine_with_pseudo_labeling(
        X_train_le=X_train_le,
        X_train_ohe=X_train_ohe,
        y_train=y_train,
        X_test_le=X_sub_le,
        X_test_ohe=X_sub_ohe,
        test_probs=ensemble_sub_probs,
        base_models=models,
        all_features_le=all_features_le,
        ohe_cols=ohe_cols,
        weights=weights,
        fraud_threshold=0.85,
        legit_threshold=0.05,
        student_model_params=None,  # use default
        random_state=42,
        final_threshold=ens_thresh  # use the ensemble threshold from validation
    )

    # Step 10: Create submission DataFrame and save
    submission = pd.DataFrame({
        'id': sub_ids.values,
        'label': final_preds
    })

    output_path = "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"\nSubmission saved to {output_path}")
    print(submission['label'].value_counts())

if __name__ == "__main__":
    main()