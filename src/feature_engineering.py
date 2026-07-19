import numpy as np
import pandas as pd
from typing import Dict, Tuple


def find_target_column(df: pd.DataFrame) -> str:
    """Identify the target column based on common naming patterns."""
    priority_names = ["target", "label", "y", "price", "SalePrice", "outcome"]
    for name in priority_names:
        if name in df.columns:
            return name
    return df.columns[-1]


def determine_task_type(series: pd.Series) -> str:
    """Decide whether the problem is regression or classification."""
    if series.dtype.kind in "O" or series.nunique() <= 15:
        return "classification"
    return "regression"


def engineer_features(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """Add derived features to improve model performance."""
    feature_df = df.copy()
    numeric_cols = [col for col in feature_df.select_dtypes(include=[np.number]).columns if col != target_col]

    if len(numeric_cols) >= 2:
        feature_df["numeric_mean"] = feature_df[numeric_cols].mean(axis=1)
        feature_df["numeric_sum"] = feature_df[numeric_cols].sum(axis=1)
        feature_df["numeric_max"] = feature_df[numeric_cols].max(axis=1)

    for column in numeric_cols:
        if (feature_df[column] > 0).all():
            feature_df[f"log_{column}"] = np.log1p(feature_df[column])

    return feature_df


def build_feature_metadata(df: pd.DataFrame, target_col: str) -> Dict[str, str]:
    """Build a mapping of feature names to simple data types."""
    features = {}
    for column in df.columns:
        if column == target_col:
            continue
        dtype = df[column].dtype
        if dtype.kind in "biufc":
            features[column] = "numeric"
        else:
            features[column] = "categorical"
    return features
