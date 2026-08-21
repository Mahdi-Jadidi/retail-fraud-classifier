from pathlib import Path
import pandas as pd


def load_table(path: str | Path) -> pd.DataFrame:
    """Load CSV or Excel data with a clear error for unsupported files."""
    path = Path(path)
    if path.suffix.lower() in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported data format: {path.suffix}")


def infer_target(frame: pd.DataFrame, target: str | None = None) -> str:
    if target and target in frame:
        return target
    for candidate in ("fraud_flag", "label", "is_fraud", "target"):
        if candidate in frame:
            return candidate
    raise ValueError("Target column was not found; pass --target explicitly.")
