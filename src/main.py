import os
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from data_preprocessing import preprocess_data
from evaluation import extract_best_model, format_metrics
from feature_engineering import build_feature_metadata, engineer_features, determine_task_type, find_target_column
from model import train_and_evaluate
from utils import ensure_directories, get_first_csv_path, load_data, save_csv, save_metrics, save_model
from visualization import plot_correlation_matrix, plot_feature_distributions, plot_model_predictions


def main():
    root_dir = Path(__file__).resolve().parent.parent
    raw_data_dir = root_dir / "data" / "raw"
    processed_data_dir = root_dir / "data" / "processed"
    outputs_dir = root_dir / "outputs"
    figures_dir = outputs_dir / "figures"
    models_dir = root_dir / "models"

    ensure_directories([
        str(raw_data_dir),
        str(processed_data_dir),
        str(figures_dir),
        str(models_dir),
    ])

    csv_path = get_first_csv_path(str(raw_data_dir))
    if csv_path is None:
        raise FileNotFoundError("No CSV file found in data/raw. Please add your dataset.")

    df = load_data(csv_path)
    df = preprocess_data(df)

    target_col = find_target_column(df)
    task_type = determine_task_type(df[target_col])
    print(f"Detected target: {target_col}")
    print(f"Detected task type: {task_type}")

    df = engineer_features(df, target_col)
    feature_metadata = build_feature_metadata(df, target_col)

    plot_feature_distributions(df.drop(columns=[target_col]), str(figures_dir))
    plot_correlation_matrix(df, str(figures_dir / "correlation_matrix.png"))

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if task_type == "classification" else None
    )

    model_scores, best_model_name = train_and_evaluate(X_train, X_test, y_train, y_test, task_type)
    best_model_name, best_model_data = extract_best_model(model_scores, task_type)

    best_pipeline = best_model_data["pipeline"]
    predictions = best_model_data["predictions"]
    metrics = best_model_data["metrics"]

    results_df = pd.DataFrame({"actual": y_test.reset_index(drop=True), "predicted": predictions})
    save_csv(results_df, str(outputs_dir / "predictions.csv"))
    save_metrics(metrics, str(outputs_dir / "metrics.txt"))
    save_model(best_pipeline, str(models_dir / "trained_model.pkl"), {"model_name": best_model_name, "task_type": task_type})

    if task_type == "regression":
        plot_model_predictions(
            y_test.reset_index(drop=True), predictions, str(figures_dir / "prediction_scatter.png"), title=f"{best_model_name} Predictions"
        )

    print("Completed training and evaluation.")
    print(f"Best model: {best_model_name}")
    print(format_metrics(metrics))


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    main()
