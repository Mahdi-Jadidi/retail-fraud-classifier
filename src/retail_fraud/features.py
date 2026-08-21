import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def make_features(frame: pd.DataFrame, target: str) -> pd.DataFrame:
    data = frame.drop(columns=[target], errors="ignore").copy()
    for column in list(data.columns):
        if "time" in column.lower() or column.lower() in {"timestamp", "date"}:
            parsed = pd.to_datetime(data[column], errors="coerce")
            if parsed.notna().any():
                data[f"{column}_hour"] = parsed.dt.hour
                data[f"{column}_dayofweek"] = parsed.dt.dayofweek
                data[f"{column}_ordinal"] = parsed.astype("int64") // 10**9
                data = data.drop(columns=[column])
    return data


def make_preprocessor(frame: pd.DataFrame) -> ColumnTransformer:
    numeric = frame.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in frame.columns if c not in numeric]
    return ColumnTransformer(
        transformers=[
            ("numeric", SimpleImputer(strategy="median"), numeric),
            (
                "categorical",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                categorical,
            ),
        ],
        remainder="drop",
    )
