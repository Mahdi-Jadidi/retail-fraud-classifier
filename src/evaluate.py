\"\"\"
\"\""
Evaluation module for the fraud detection pipeline.

This module provides functions to:
- Threshold tuning based on validation F1-score
- Compute comprehensive classification metrics (accuracy, precision, recall, F1,
  ROC-AUC, PR-AUC, MCC, Cohen's kappa, specificity)
- Generate ensemble predictions using weighted averaging of model probabilities
- Compute weighted ensemble weights based on validation F1 scores
\"\"\"
from typing import Tuple, Dict, List
import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, auc,
    matthews_corrcoef, cohen_kappa_score, f1_score
)

def tune_threshold_f1(y_true: np.ndarray, y_probs: np.ndarray) -> tuple[float, float]:
    \"\"\"
    Find the probability threshold that maximizes F1-score on validation data.

    Parameters
    ----------
    y_true : np.ndarray
        True binary labels.
    y_probs : np.ndarray
        Predicted probabilities for the positive class.

    Returns
    -------
    best_threshold : float
        Threshold that yields the highest F1-score.
    best_f1 : float
        The highest F1-score achievable.
    \"\"\"
    best_t, best_f1 = 0.5, 0.0
    for t in np.linspace(0.1, 0.9, 400):
        f1 = f1_score(y_true, (y_probs >= t).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return best_t, best_f1

def full_report(
    name: str,
    y_true: np.ndarray,
    y_probs: np.ndarray,
    threshold: float
) -> dict:
    \"\"\"
    Compute and print a comprehensive classification report.

    Parameters
    ----------
    name : str
        Name of the model (for printing).
    y_true : np.ndarray
        True binary labels.
    y_probs : np.ndarray
        Predicted probabilities for the positive class.
    threshold : float
        Classification threshold to convert probabilities to binary predictions.

    Returns
    -------
    metrics : dict
        Dictionary containing:
        - f1, roc_auc, pr_auc, mcc
        (accuracy, precision, recall, etc. are printed but not returned
         to keep the dictionary compact; can be extended if needed).
    \"\"\"
    y_pred = (y_probs >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    prec_arr, rec_arr, _ = precision_recall_curve(y_true, y_probs)
    pr_auc = auc(rec_arr, prec_arr)
    spec = tn / (tn + fp + 1e-10)

    # Print the report (matching the notebook's output)
    print(f' {name} [threshold={threshold:.3f}]')
    print(classification_report(y_true, y_pred, target_names=['Legit','Fraud']))
    print(f'Confusion Matrix:\\n{cm}')
    print(f'\\nROC-AUC : {roc_auc_score(y_true, y_probs):.4f}')
    print(f'PR-AUC : {pr_auc:.4f}')
    print(f'MCC : {matthews_corrcoef(y_true, y_pred):.4f}')
    print(f\"Cohen's κ : {cohen_kappa_score(y_true, y_pred):.4f}\")
    print(f'Specificity : {spec:.4f}')

    return {
        'f1': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_probs),
        'pr_auc': pr_auc,
        'mcc': matthews_corrcoef(y_true, y_pred)
    }

def ensemble_predictions(
    model_probs: dict,
    weights: dict = None
) -> np.ndarray:
    \"\"\"
    Combine predicted probabilities from multiple models using weighted averaging.

    Parameters
    ----------
    model_probs : dict
        Mapping from model name to predicted probabilities (numpy array).
    weights : dict, optional
        Mapping from model name to weight. If None, uniform weights are used.

    Returns
    -------
    ensemble_probs : np.ndarray
        Weighted average of the input probabilities.
    \"\"\"
    if weights is None:
        # Uniform weights
        weights = {name: 1.0 for name in model_probs.keys()}
    # Normalize weights to sum to 1
    total_weight = sum(weights.values())
    normalized_weights = {name: w / total_weight for name, w in weights.items()}
    # Compute weighted sum
    ensemble_probs = np.zeros_like(next(iter(model_probs.values())))
    for name, probs in model_probs.items():
        ensemble_probs += normalized_weights[name] * probs
    return ensemble_probs

def compute_ensemble_weights(val_f1_scores: dict) -> dict:
    \"\"\"
    Compute ensemble weights proportional to validation F1 scores.

    Parameters
    ----------
    val_f1_scores : dict
        Mapping from model name to validation F1 score.

    Returns
    -------
    weights : dict
        Mapping from model name to weight (summing to 1).
    \"\"\"
    # Convert to numpy array for normalization
    names = list(val_f1_scores.keys())
    scores = np.array([val_f1_scores[name] for name in names])
    weights = scores / scores.sum()
    return dict(zip(names, weights))