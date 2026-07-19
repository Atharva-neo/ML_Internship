import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def run_unemployment_analysis(csv_path: str):
    df = pd.read_csv(csv_path)

    print("Unemployment Analysis")
    print("---------------------")
    print(f"Data shape: {df.shape}")
    print(df.head())
    print("\nMissing values by column:")
    print(df.isna().sum())

    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.sort_values("Date")
    elif "Year" in df.columns and "Month" in df.columns:
        df["Date"] = pd.to_datetime(df[["Year", "Month"]].assign(DAY=1))
        df = df.sort_values("Date")

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    print("\nNumeric columns:", numeric_cols)

    if "Unemployment Rate" in df.columns:
        target_col = "Unemployment Rate"
    elif "Unemployment_Rate" in df.columns:
        target_col = "Unemployment_Rate"
    else:
        target_col = numeric_cols[0]

    print(f"Using target column: {target_col}")

    plt.figure(figsize=(10, 5))
    sns.lineplot(data=df, x="Date", y=target_col)
    plt.title("Unemployment Rate Over Time")
    plt.xlabel("Date")
    plt.ylabel(target_col)
    plt.tight_layout()
    plt.savefig("unemployment_trend.png")
    plt.close()

    if "Covid" in df.columns or "Pandemic" in df.columns:
        print("Covid-related columns found; inspect relationships.")

    print("Saved plot: unemployment_trend.png")


if __name__ == "__main__":
    data_path = "../data/unemployment_data.csv"
    run_unemployment_analysis(data_path)
