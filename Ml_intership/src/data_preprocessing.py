import numpy as np
import pandas as pd


def inspect_dataset(df: pd.DataFrame) -> None:
    """Print a basic inspection summary for a dataset."""
    print("Dataset inspection")
    print("------------------")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")
    print(df.dtypes)
    print(df.head(5))


def handle_missing_values(df: pd.DataFrame, threshold: float = 0.4) -> pd.DataFrame:
    """Handle missing values by dropping sparse columns and imputing remaining values."""
    missing_ratio = df.isna().mean()
    drop_cols = missing_ratio[missing_ratio > threshold].index.tolist()
    if drop_cols:
        df = df.drop(columns=drop_cols)

    for column in df.columns:
        if df[column].isna().sum() == 0:
            continue
        if df[column].dtype.kind in "biufc":
            df[column] = df[column].fillna(df[column].median())
        else:
            df[column] = df[column].fillna(df[column].mode().iloc[0])

    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate rows from the dataset."""
    initial_count = df.shape[0]
    df = df.drop_duplicates(ignore_index=True)
    final_count = df.shape[0]
    print(f"Removed {initial_count - final_count} duplicate rows.")
    return df


def treat_outliers(df: pd.DataFrame, multiplier: float = 1.5) -> pd.DataFrame:
    """Cap extreme values on numeric columns using the IQR method."""
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    for column in numeric_columns:
        if df[column].nunique() < 10:
            continue
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        df[column] = np.where(df[column] < lower_bound, lower_bound, df[column])
        df[column] = np.where(df[column] > upper_bound, upper_bound, df[column])
    return df


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Run the full preprocessing pipeline on the dataset."""
    inspect_dataset(df)
    df = remove_duplicates(df)
    df = handle_missing_values(df)
    df = treat_outliers(df)
    return df
