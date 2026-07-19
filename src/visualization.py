import os
from typing import Optional

import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature_distributions(df, output_dir: str) -> None:
    """Generate histograms for numeric features and save charts."""
    os.makedirs(output_dir, exist_ok=True)
    numeric_columns = df.select_dtypes(include=["number"]).columns.tolist()

    for column in numeric_columns:
        plt.figure(figsize=(8, 4))
        sns.histplot(df[column], kde=True, color="steelblue")
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"distribution_{column}.png"))
        plt.close()


def plot_correlation_matrix(df, output_path: str) -> None:
    """Save a correlation matrix heatmap for numeric features."""
    numeric_df = df.select_dtypes(include=["number"])
    plt.figure(figsize=(10, 8))
    correlation = numeric_df.corr()
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def plot_model_predictions(y_true, y_pred, output_path: str, title: Optional[str] = None) -> None:
    """Create a scatter plot comparing true and predicted values."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.7, color="darkorange", edgecolor="k")
    plt.plot([min(y_true), max(y_true)], [min(y_true), max(y_true)], color="navy", linestyle="--")
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.title(title or "Actual vs Predicted")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
