from typing import Any, Dict, List, Tuple

import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing pipeline for numeric and categorical variables."""
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    transformers = []
    if numeric_cols:
        transformers.append(("num", StandardScaler(), numeric_cols))
    if categorical_cols:
        transformers.append(
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse=False),
                categorical_cols,
            )
        )

    return ColumnTransformer(transformers=transformers, remainder="drop")


def train_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    task_type: str,
    random_state: int = 42,
) -> Dict[str, Pipeline]:
    """Train a set of candidate models for the given task."""
    preprocessor = build_preprocessor(X_train)
    models: Dict[str, Any] = {}

    if task_type == "classification":
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=random_state),
            "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=random_state),
            "GradientBoostingClassifier": GradientBoostingClassifier(random_state=random_state),
        }
    else:
        models = {
            "LinearRegression": LinearRegression(),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=random_state),
            "GradientBoostingRegressor": GradientBoostingRegressor(random_state=random_state),
        }

    pipelines = {}
    for name, estimator in models.items():
        pipelines[name] = Pipeline(
            steps=[("preprocessor", preprocessor), ("estimator", estimator)]
        )
    return pipelines


def train_and_evaluate(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    task_type: str,
) -> Tuple[Dict[str, Any], str]:
    """Train candidate models, evaluate them, and return model summaries."""
    pipelines = train_models(X_train, y_train, task_type)
    model_scores: Dict[str, Dict[str, Any]] = {}
    best_model_name = ""
    best_score = float("-inf")

    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)

        if task_type == "classification":
            score = pipeline.score(X_test, y_test)
            metrics = {
                "accuracy": score,
                "precision": None,
                "recall": None,
                "f1_score": None,
            }
        else:
            rmse = mean_squared_error(y_test, predictions, squared=False)
            score = -rmse
            metrics = {
                "rmse": rmse,
                "mae": mean_absolute_error(y_test, predictions),
                "r2_score": r2_score(y_test, predictions),
            }

        model_scores[name] = {
            "pipeline": pipeline,
            "predictions": predictions,
            "metrics": metrics,
            "selection_score": score,
        }

        if score > best_score:
            best_score = score
            best_model_name = name

    return model_scores, best_model_name
