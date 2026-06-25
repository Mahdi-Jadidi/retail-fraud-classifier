\"\"\"
\"\""
Prediction module for the fraud detection pipeline.

This module provides functions to generate predictions from trained models,
including the pseudo-labeling and student model refinement steps.
\"\"\"
from typing import Tuple, Dict, List, Any
import numpy as np
import pandas as pd

def get_base_predictions(
    X_test_le: pd.DataFrame,
    X_test_ohe: pd.DataFrame,
    models: Dict[str, Any],
    weights: Dict[str, float]
) -> np.ndarray:
    \"\"\"
    Generate base predictions from an ensemble of models.

    Parameters
    ----------
    X_test_le : pd.DataFrame
        Test features with label-encoded categorical columns (for LightGBM/CatBoost).
    X_test_ohe : pd.DataFrame
        Test features with one-hot encoded categorical columns (for XGBoost/Random Forest).
    models : dict
        Dictionary mapping model names to trained model objects.
        Expected keys: 'lightgbm', 'xgboost', 'catboost', 'random_forest'.
    weights : dict
        Dictionary mapping model names to weights (should sum to 1).

    Returns
    -------
    ensemble_probs : np.ndarray
        Weighted average of predicted probabilities from each model.
    \"\"\"
    # Get probabilities from each model
    probs = {}
    # LightGBM
    if 'lightgbm' in models:
        probs['lightgbm'] = models['lightgbm'].predict_proba(X_test_le)[:, 1]
    # XGBoost
    if 'xgboost' in models:
        probs['xgboost'] = models['xgboost'].predict_proba(X_test_ohe)[:, 1]
    # CatBoost
    if 'catboost' in models:
        probs['catboost'] = models['catboost'].predict_proba(X_test_le)[:, 1]
    # Random Forest
    if 'random_forest' in models:
        probs['random_forest'] = models['random_forest'].predict_proba(X_test_ohe)[:, 1]

    # Weighted average
    ensemble_probs = np.zeros_like(next(iter(probs.values())))
    for name, prob in probs.items():
        ensemble_probs += weights[name] * prob
    return ensemble_probs

def refine_with_pseudo_labeling(
    # Training data (processed)
    X_train_le: pd.DataFrame,
    X_train_ohe: pd.DataFrame,
    y_train: np.ndarray,
    # Test data (processed)
    X_test_le: pd.DataFrame,
    X_test_ohe: pd.DataFrame,
    # Base model probabilities on test set (from get_base_predictions)
    test_probs: np.ndarray,
    # Trained base models (to be retrained on full data for student model)
    base_models: Dict[str, Any],
    # Feature column lists (for retraining)
    all_features_le: List[str],
    # OHE column list (for retraining)
    ohe_cols: List[str],
    # Ensemble weights (for combining base model predictions)
    weights: Dict[str, float],
    # Pseudo-labeling thresholds
    fraud_threshold: float = 0.85,
    legit_threshold: float = 0.05,
    # Student model parameters (optional)
    student_model_params: Dict[str, Any] = None,
    # Random state
    random_state: int = 42,
    # Threshold for final predictions (from validation tuning of ensemble)
    final_threshold: float = 0.5
) -> np.ndarray:
    \"\"\"
    Refine predictions using pseudo-labeling and a student model.

    Parameters
    ----------
    All parameters are as described above.

    Returns
    -------
    final_predictions : np.ndarray
        Binary predictions (0 or 1) for the test set.
    \"\"\"
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    from sklearn.ensemble import RandomForestClassifier

    # Step 1: Identify high-confidence predictions for pseudo-labeling
    high_conf_fraud = test_probs >= fraud_threshold
    high_conf_legit = test_probs <= legit_threshold
    confident_mask = high_conf_fraud | high_conf_legit

    # If no confident predictions, return predictions based on the final_threshold
    if not np.any(confident_mask):
        return (test_probs >= final_threshold).astype(int)

    # Step 2: Create pseudo-labels
    pseudo_y = (test_probs[confident_mask] >= fraud_threshold).astype(int)
    pseudo_X_le = X_test_le.iloc[confident_mask].copy()
    pseudo_X_ohe = X_test_ohe.iloc[confident_mask].copy()

    # Step 3: Combine original training data with pseudo-labeled data
    # Weight original samples with weight 1.0, pseudo-labeled with weight 0.4
    weight_train = np.ones(len(y_train))
    weight_pseudo = np.full(len(pseudo_y), 0.4)

    # For LE models: concatenate features
    X_combined_le = pd.concat([X_train_le, pseudo_X_le], ignore_index=True)
    y_combined = np.concatenate([y_train, pseudo_y])
    weights_combined = np.concatenate([weight_train, weight_pseudo])

    # For OHE models: concatenate features
    X_combined_ohe = pd.concat([X_train_ohe, pseudo_X_ohe], ignore_index=True)

    # Step 4: Train student models on the combined data
    # We'll train a LightGBM student model (as in the notebook)
    # But we can also choose to retrain all models and ensemble again.
    # The notebook retrains a LightGBM student model.
    # We'll follow that.

    # Default student model parameters (LightGBM)
    if student_model_params is None:
        student_model_params = {
            'n_estimators': 1000,
            'learning_rate': 0.02,
            'max_depth': 6,
            'num_leaves': 31,
            'min_child_samples': 50,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'random_state': random_state,
            'n_jobs': -1,
            'verbose': -1,
        }

    student_model = LGBMClassifier(**student_model_params)
    # Note: The student model uses LE features and categorical feature indices
    # We need to know which columns are categorical (in LE form, they are the _le columns)
    # We'll assume that the last len(le_cols) columns of X_combined_le are the LE categorical columns.
    # But we don't have le_cols here. We'll assume the caller passes the LE categorical column names.
    # Since we don't have that, we'll skip the categorical_feature argument and treat all as numeric.
    # This is a simplification; for a production system, we would pass the categorical column indices.
    student_model.fit(
        X_combined_le, y_combined,
        sample_weight=weights_combined
        # categorical_feature=le_cat_indices,  # We don't have this info here
    )

    # Step 5: Generate final predictions from the student model
    # The student model uses LE features, so we need to predict on X_test_le
    final_probs = student_model.predict_proba(X_test_le)[:, 1]
    final_preds = (final_probs >= final_threshold).astype(int)
    return final_preds