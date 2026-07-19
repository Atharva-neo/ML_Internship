# CodeAlpha Data Science Internship Project

## Project Overview

This repository is a production-ready data science project scaffold for the CodeAlpha internship program. It includes data ingestion, preprocessing, feature engineering, model training, evaluation, visualization, and a Streamlit app for inference.

## Dataset Description

The repository expects a CSV dataset placed in `data/raw/`. The code automatically detects the first CSV file and uses it for the full workflow. The dataset should contain a target variable and input features.

## Features

- Automatic dataset inspection
- Missing values handling and duplicate removal
- Outlier treatment using IQR capping
- Categorical encoding and numeric scaling
- Feature engineering with numeric transformations
- Train/test split and model selection
- Multiple models evaluated for best performance
- Predictions saved as `outputs/predictions.csv`
- Model saved as `models/trained_model.pkl`
- Visualizations saved under `outputs/figures/`
- Streamlit UI for online inference

## Installation

1. Create a virtual environment:

```bash
python -m venv venv
```

2. Activate the environment:

```bash
# Windows
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Project Structure

```text
CodeAlpha_ProjectName/
├── app.py
├── data/
│   ├── raw/
│   │   └── dataset.csv
│   └── processed/
├── models/
│   └── trained_model.pkl
├── notebooks/
│   └── EDA.ipynb
├── outputs/
│   ├── figures/
│   ├── metrics.txt
│   └── predictions.csv
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py
│   ├── evaluation.py
│   ├── feature_engineering.py
│   ├── main.py
│   ├── model.py
│   ├── utils.py
│   └── visualization.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Workflow

1. Place raw dataset in `data/raw/`
2. Run `python src/main.py`
3. Review model outputs in `outputs/`
4. Start the app with `streamlit run app.py`

## Exploratory Data Analysis

The notebook `notebooks/EDA.ipynb` contains sample exploratory data analysis steps, distribution plots, and correlation visualization.

## Model Building

The workflow trains multiple candidate models depending on the dataset task type (classification or regression) and chooses the best performer.

## Evaluation Metrics

- Regression: RMSE, MAE, R2 score
- Classification: Accuracy, Precision, Recall, F1-score

## Results

The best model is stored in `models/trained_model.pkl`. Predictions are exported to `outputs/predictions.csv`, and metrics are saved in `outputs/metrics.txt`.

## Visualizations

Plots are saved under `outputs/figures/` and include feature distributions, correlation heatmap, and model predictions.

## Future Improvements

- Add hyperparameter tuning with grid search
- Add cross-validation for more robust evaluation
- Support multiple raw datasets automatically
- Expand the Streamlit app with feature controls and classification probability display

## Screenshots

![Screenshot 1](screenshots/screenshot1.png)
![Screenshot 2](screenshots/screenshot2.png)

## GitHub Badges

![Python](https://img.shields.io/badge/python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)
