import numpy as np
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score


def tune_threshold(y_true, probabilities) -> tuple[float, float]:
    thresholds = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, probabilities >= t, zero_division=0) for t in thresholds]
    index = int(np.argmax(scores))
    return float(thresholds[index]), float(scores[index])


def classification_report(y_true, probabilities, threshold: float) -> dict[str, float]:
    predictions = probabilities >= threshold
    result = {
        "threshold": threshold,
        "f1": f1_score(y_true, predictions, zero_division=0),
        "precision": precision_score(y_true, predictions, zero_division=0),
        "recall": recall_score(y_true, predictions, zero_division=0),
        "pr_auc": average_precision_score(y_true, probabilities),
    }
    if len(np.unique(y_true)) > 1:
        result["roc_auc"] = roc_auc_score(y_true, probabilities)
    return {key: float(value) for key, value in result.items()}
