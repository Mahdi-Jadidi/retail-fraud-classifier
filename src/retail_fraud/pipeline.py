from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from .data import infer_target, load_table
from .features import make_features, make_preprocessor
from .metrics import classification_report, tune_threshold


def train(train_path: str, output_dir: str = "artifacts", target: str | None = None) -> dict:
    raw = load_table(train_path)
    target = infer_target(raw, target)
    y = raw[target].astype(int)
    X = make_features(raw, target)
    stratify = y if y.nunique() > 1 else None
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=stratify)
    preprocessor = make_preprocessor(X)
    model = HistGradientBoostingClassifier(max_iter=150, learning_rate=0.08, random_state=42)
    pipeline = __import__("sklearn.pipeline", fromlist=["Pipeline"]).Pipeline([("preprocess", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)
    probabilities = pipeline.predict_proba(X_valid)[:, 1]
    threshold, _ = tune_threshold(y_valid.to_numpy(), probabilities)
    metrics = classification_report(y_valid.to_numpy(), probabilities, threshold)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, output / "model.joblib")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics
