\"\"\"
\"\""
Model training module for the fraud detection pipeline.

This module defines functions to train each of the four base models:
- LightGBM
- XGBoost
- CatBoost
- Random Forest

Each training function returns the fitted model.
\"\"\"
from typing import Tuple, Dict, Any, List
import numpy as np
import pandas as pd

# LightGBM
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
# XGBoost
from xgboost import XGBClassifier
# CatBoost
from catboost import CatBoostClassifier
# Scikit-learn
from sklearn.ensemble import RandomForestClassifier

def train_lightgbm(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    categorical_feature: List[str],
    random_state: int = 42,
    verbose: int = -1
) -> LGBMClassifier:
    \"\"\"
    Train a LightGBM classifier with early stopping.

    Parameters
    ----------
    X_train, X_val : pd.DataFrame
        Feature matrices for training and validation.
    y_train, y_val : np.ndarray
        Target arrays for training and validation.
    categorical_feature : list
        List of column names indicating categorical features.
    random_state : int, default 42
        Random seed for reproducibility.
    verbose : int, default -1
        Verbosity level for training output.

    Returns
    -------
    model : LGBMClassifier
        Fitted LightGBM model.
    \"\"\"
    model = LGBMClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=6,
        num_leaves=31,
        min_child_samples=50,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=random_state,
        n_jobs=-1,
        verbose=verbose,
    )
    model.fit(
        X_train, y_train,
        categorical_feature=categorical_feature,
        eval_set=[(X_val, y_val)],
        callbacks=[
            early_stopping(stopping_rounds=50, verbose=False),
            log_evaluation(period=100)
        ]
    )
    return model

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    random_state: int = 42
) -> XGBClassifier:
    \"\"\"
    Train an XGBoost classifier with early stopping.

    Parameters
    ----------
    X_train, X_val : pd.DataFrame
        Feature matrices for training and validation.
    y_train, y_val : np.ndarray
        Target arrays for training and validation.
    random_state : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    model : XGBClassifier
        Fitted XGBoost model.
    \"\"\"
    model = XGBClassifier(
        n_estimators=1000,
        learning_rate=0.02,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric='logloss',
        early_stopping_rounds=50,
        random_state=random_state,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    return model

def train_catboost(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
    cat_features: List[str],
    random_state: int = 42,
    verbose: int = 100
) -> CatBoostClassifier:
    \"\"\"
    Train a CatBoost classifier with early stopping.

    Parameters
    ----------
    X_train, X_val : pd.DataFrame
        Feature matrices for training and validation.
    y_train, y_val : np.ndarray
        Target arrays for training and validation.
    cat_features : list
        List of column indices or names indicating categorical features.
    random_state : int, default 42
        Random seed for reproducibility.
    verbose : int, default 100
        Verbosity level for training output.

    Returns
    -------
    model : CatBoostClassifier
        Fitted CatBoost model.
    \"\"\"
    model = CatBoostClassifier(
        iterations=1000,
        learning_rate=0.02,
        depth=6,
        l2_leaf_reg=3,
        subsample=0.8,
        eval_metric='F1',
        early_stopping_rounds=50,
        random_seed=random_state,
        verbose=verbose,
        cat_features=cat_features,
    )
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        use_best_model=True
    )
    return model

def train_random_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    random_state: int = 42
) -> RandomForestClassifier:
    \"\"\"
    Train a Random Forest classifier.

    Parameters
    ----------
    X_train : pd.DataFrame
        Feature matrix for training.
    y_train : np.ndarray
        Target array for training.
    random_state : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    model : RandomForestClassifier
        Fitted Random Forest model.
    \"\"\"
    model = RandomForestClassifier(
        n_estimators=400,
        max_depth=14,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model