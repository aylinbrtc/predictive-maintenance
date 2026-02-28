# Predictive Maintenance — Industrial Machine Failure Prediction

TOBB ETU · BIL 476 Data Mining · Classification Project

## Overview

This project applies machine learning classification to predict industrial machine component failures using the Microsoft Azure Predictive Maintenance dataset. The goal is to answer:

> *Can component failures be predicted in advance using sensor data (voltage, rotation, pressure, vibration), error logs, and maintenance history? Which algorithm performs best, and which features are most indicative of failure?*

## Dataset

Microsoft Azure Predictive Maintenance — [Kaggle](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance)

Download the 5 CSV files and place them in `data/raw/`:
- `PdM_telemetry.csv` (876,099 rows — hourly sensor readings)
- `PdM_errors.csv` (3,919 rows — error events)
- `PdM_failures.csv` (761 rows — component failure events)
- `PdM_maint.csv` (3,286 rows — maintenance events)
- `PdM_machines.csv` (100 rows — machine metadata)

## Setup

```bash
# Clone the repo
git clone <repo-url>
cd predictive-maintenance

# Create a virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter lab
```

## Project Structure

```
predictive-maintenance/
├── notebooks/
│   ├── 01_eda.ipynb               # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb  # Data merging & feature creation
│   ├── 03_modeling.ipynb          # Model training & hyperparameter tuning
│   └── 04_evaluation.ipynb        # Evaluation, comparison, visualizations
├── src/
│   ├── data_loader.py             # Load and merge datasets
│   ├── feature_engineering.py     # Feature creation functions
│   ├── train.py                   # Model training pipeline
│   └── evaluate.py                # Evaluation metrics and plots
├── figures/                       # All saved plots (for IEEE report)
├── report/                        # IEEE LaTeX report
├── models/                        # Saved trained models (.joblib)
├── data/
│   ├── raw/                       # Kaggle CSVs (not in git)
│   └── processed/                 # Merged feature dataset
└── requirements.txt
```

## Notebooks — Run in Order

| Notebook | Description |
|----------|-------------|
| `01_eda.ipynb` | Load raw data, descriptive statistics, distributions, correlations |
| `02_feature_engineering.ipynb` | Merge 5 tables, rolling features, target variable creation |
| `03_modeling.ipynb` | Train Decision Tree, Random Forest, XGBoost, k-NN; hyperparameter tuning |
| `04_evaluation.ipynb` | Metrics, confusion matrices, ROC/PR curves, SMOTE experiment |

## Algorithms Compared

- Decision Tree
- Random Forest
- XGBoost
- k-Nearest Neighbors (k-NN)

## Key Results

*(To be filled after model training)*

## Reproducibility

All random operations use `random_state=42`. Python and library versions are pinned in `requirements.txt`.
