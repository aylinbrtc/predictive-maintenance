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
git clone https://github.com/aylinbrtc/predictive-maintenance
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
│   ├── 01_eda.ipynb                      # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb      # Data merging & feature creation
│   ├── 03_modeling.ipynb                 # Model training & hyperparameter tuning
│   ├── 04_evaluation.ipynb               # Evaluation, comparison, visualizations
│   ├── 05_advanced_analysis.ipynb        # Naive Bayes, McNemar, SHAP, ablation, temporal/horizon/cost experiments
│   ├── 06_walk_forward_validation.ipynb  # Monthly walk-forward validation
│   └── 07_multiclass_classification.ipynb  # Multi-class component identification
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
| `05_advanced_analysis.ipynb` | Temporal split, machine generalisation, horizon analysis, cost-sensitive threshold, Naive Bayes, McNemar test, SHAP, ablation study |
| `06_walk_forward_validation.ipynb` | Monthly walk-forward validation simulating real deployment cycle |
| `07_multiclass_classification.ipynb` | Multi-class component identification (direct + two-stage pipeline) |

## Algorithms Compared

- Decision Tree
- Random Forest
- XGBoost
- k-Nearest Neighbors (k-NN)
- Naïve Bayes (Gaussian)

## Key Results

Binary classification target: will a failure occur within the next 24 hours?
Dataset after feature engineering: 36,600 machine-days, 62 features, 47:1 class imbalance.

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|----|-----|
| Decision Tree | 0.9980 | 0.9388 | 0.9583 | 0.9485 | 0.9785 |
| k-NN (k=11) | 0.9980 | 0.9108 | 0.9931 | 0.9502 | 0.9997 |
| XGBoost (baseline) | 0.9988 | 0.9470 | 0.9931 | 0.9695 | 0.9999 |
| XGBoost (tuned) | 0.9986 | 0.9408 | 0.9931 | 0.9662 | 1.0000 |
| Random Forest (baseline) | 0.9988 | 0.9592 | 0.9792 | 0.9691 | 1.0000 |
| **Random Forest (tuned)** | **0.9989** | **0.9658** | **0.9792** | **0.9724** | **0.9999** |

**Best model: Random Forest (tuned) — F1 = 0.9724, AUC = 0.9999**

Additional experiments:
- Temporal split validation (train Jan–Sep, test Oct–Dec): F1 = 0.9702
- Machine generalisation (train on machines 1–80, test on 81–100): F1 = 0.9672
- Prediction horizon: F1 drops from 0.97 (24h) to 0.36 (7 days); AUC stays above 0.93
- Cost-sensitive threshold: shifting threshold from 0.50 → 0.40 reduces operational cost by 40% at a 5:1 FN/FP cost ratio
- Walk-forward validation (Apr–Dec 2015, monthly): mean F1 = 0.9712, AUC = 0.9999 every month
- Multi-class component identification: XGBoost macro F1 = 0.9781; two-stage pipeline macro F1 = 0.9716

## Reproducibility

All random operations use `random_state=42`. Python and library versions are pinned in `requirements.txt`.
