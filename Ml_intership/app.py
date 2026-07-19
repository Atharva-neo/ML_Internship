import streamlit as st
import pandas as pd
from joblib import load


def load_trained_model(model_path: str):
    """Load the trained model from a joblib file."""
    return load(model_path)


def main():
    st.set_page_config(page_title="CodeAlpha ML App", layout="centered")
    st.title("CodeAlpha Internship ML Demo")
    st.write("Upload your dataset and get model predictions using the trained pipeline.")

    model_file = "models/trained_model.pkl"
    try:
        trained = load_trained_model(model_file)
        model = trained["model"]
        metadata = trained.get("metadata", {})
    except FileNotFoundError:
        st.error("Trained model not found. Run src/main.py first to train and save the model.")
        return

    st.sidebar.header("Prediction Inputs")
    uploaded_file = st.sidebar.file_uploader("Upload raw CSV data", type=["csv"])

    input_data = None
    if uploaded_file is not None:
        input_data = pd.read_csv(uploaded_file)
        st.sidebar.success("File uploaded successfully.")

    if input_data is not None:
        st.write("### Sample Input Data")
        st.dataframe(input_data.head())

        if st.sidebar.button("Run Prediction"):
            try:
                predictions = model.predict(input_data)
                st.success("Prediction complete")
                st.write("### Predictions")
                st.dataframe(pd.DataFrame({"prediction": predictions}))
                if hasattr(model, "predict_proba"):
                    probabilities = model.predict_proba(input_data)
                    st.write("### Confidence Scores")
                    st.dataframe(pd.DataFrame(probabilities))
            except Exception as exc:
                st.error(f"Prediction failed: {exc}")
    else:
        st.info("Upload a CSV file in the sidebar to start.")

    st.sidebar.write("### Model Info")
    st.sidebar.write(metadata)


if __name__ == "__main__":
    main()
