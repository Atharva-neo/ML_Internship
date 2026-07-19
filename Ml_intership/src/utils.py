import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd


def get_first_csv_path(directory: str) -> Optional[str]:
    """Return the first CSV file path found under a directory."""
    path = Path(directory)
    if not path.exists():
        return None
    csv_files = list(path.glob("*.csv"))
    return str(csv_files[0]) if csv_files else None


def ensure_directories(paths: List[str]) -> None:
    """Create required directories if they do not exist."""
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def load_data(csv_path: str) -> pd.DataFrame:
    """Load a CSV dataset into a pandas DataFrame."""
    return pd.read_csv(csv_path)


def save_csv(df: pd.DataFrame, output_path: str) -> None:
    """Save a DataFrame to CSV."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def save_metrics(metrics: Dict[str, Any], output_path: str) -> None:
    """Save evaluation metrics to a text file."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        for key, value in metrics.items():
            file.write(f"{key}: {value}\n")


def save_model(model: Any, model_path: str, metadata: Dict[str, Any]) -> None:
    """Save the trained model and metadata using joblib."""
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "metadata": metadata}, model_path)


def load_model(model_path: str) -> Dict[str, Any]:
    """Load a model object and metadata from a joblib file."""
    return joblib.load(model_path)
