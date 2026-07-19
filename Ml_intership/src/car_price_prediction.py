import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def run_car_price_prediction(csv_path: str):
    df = pd.read_csv(csv_path)

    print("Car Price Prediction")
    print("--------------------")
    print(f"Data shape: {df.shape}")
    print(df.head())
    print("\nMissing values by column:")
    print(df.isna().sum())

    df = df.dropna()

    target_col = "Price"
    if target_col not in df.columns:
        target_col = df.select_dtypes(include=["number"]).columns[-1]

    feature_cols = [c for c in df.columns if c != target_col]
    X = df[feature_cols]
    y = df[target_col]

    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse=False),
                categorical_cols,
            ),
        ]
    )

    model = Pipeline(
        steps=[("preprocessor", preprocessor), ("regressor", RandomForestRegressor(random_state=42))]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(f"Using target column: {target_col}")
    print(f"RMSE: {mean_squared_error(y_test, y_pred, squared=False):.4f}")
    print(f"R2 score: {r2_score(y_test, y_pred):.4f}")


if __name__ == "__main__":
    data_path = "../data/car_price_data.csv"
    run_car_price_prediction(data_path)
