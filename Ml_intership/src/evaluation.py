from typing import Dict, Any, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    f1_score,
)


def compute_classification_metrics(y_true, y_pred) -> Dict[str, Any]:
    """Compute classification metrics for model evaluation."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_true, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    }


def compute_regression_metrics(y_true, y_pred) -> Dict[str, Any]:
    """Compute regression metrics for model evaluation."""
    return {
        "rmse": mean_squared_error(y_true, y_pred, squared=False),
        "mae": mean_absolute_error(y_true, y_pred),
        "r2_score": r2_score(y_true, y_pred),
    }


def format_metrics(metrics: Dict[str, Any]) -> str:
    """Format metric dictionary for display or file output."""
    lines = [f"{key}: {value:.4f}" for key, value in metrics.items() if value is not None]
    return "\n".join(lines)


def extract_best_model(model_scores: Dict[str, Dict[str, Any]], task_type: str) -> Tuple[str, Dict[str, Any]]:
    """Select the best model based on evaluation metrics."""
    best_model_name = None
    best_selection_value = -float("inf")

    for name, details in model_scores.items():
        if task_type == "classification":
            value = details["metrics"]["accuracy"]
        else:
            value = -details["metrics"]["rmse"]

        if best_model_name is None or value > best_selection_value:
            best_model_name = name
            best_selection_value = value

    return best_model_name, model_scores[best_model_name]
