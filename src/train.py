"""
train.py
--------
Model training pipeline for the predictive maintenance classification task.
Trains Decision Tree, Random Forest, XGBoost, and k-NN with baseline defaults,
then optionally performs hyperparameter search on the top models.
"""

import joblib
import os
import numpy as np
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier


MODELS_DIR = os.path.join(os.path.dirname(__file__), '..', 'models')
RANDOM_STATE = 42


def get_baseline_models(scale_pos_weight: float = 1.0) -> dict:
    """
    Return a dict of baseline classifiers with default parameters.

    Parameters
    ----------
    scale_pos_weight : ratio of negative to positive samples (for XGBoost)

    Returns
    -------
    dict mapping model name -> sklearn-compatible estimator
    """
    return {
        'DecisionTree': DecisionTreeClassifier(random_state=RANDOM_STATE),
        'RandomForest': RandomForestClassifier(
            n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            eval_metric='logloss',
            n_jobs=-1,
        ),
        'kNN': KNeighborsClassifier(n_neighbors=5),
    }


def train_all(X_train: pd.DataFrame, y_train: pd.Series,
              scale_pos_weight: float = 1.0) -> dict:
    """
    Train all baseline models on (X_train, y_train).

    Returns
    -------
    dict mapping model name -> fitted estimator
    """
    models = get_baseline_models(scale_pos_weight)
    trained = {}
    for name, clf in models.items():
        print(f'Training {name}...')
        clf.fit(X_train, y_train)
        trained[name] = clf
    return trained


def hyperparameter_search(X_train: pd.DataFrame, y_train: pd.Series,
                          model_name: str, n_iter: int = 30) -> object:
    """
    Run RandomizedSearchCV on the specified model.

    Parameters
    ----------
    model_name : one of 'RandomForest', 'XGBoost'
    n_iter     : number of parameter combinations to try

    Returns
    -------
    Best fitted estimator.
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    param_grids = {
        'RandomForest': {
            'n_estimators': [100, 200, 300, 500],
            'max_depth': [None, 5, 10, 20, 30],
            'max_features': ['sqrt', 'log2', 0.5],
            'min_samples_split': [2, 5, 10],
            'class_weight': ['balanced', None],
        },
        'XGBoost': {
            'n_estimators': [100, 200, 300],
            'max_depth': [3, 5, 7, 10],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'subsample': [0.7, 0.8, 1.0],
            'colsample_bytree': [0.7, 0.8, 1.0],
        },
    }

    base_models = {
        'RandomForest': RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        'XGBoost': XGBClassifier(random_state=RANDOM_STATE, eval_metric='logloss', n_jobs=-1),
    }

    if model_name not in base_models:
        raise ValueError(f"Hyperparameter search not configured for '{model_name}'.")

    search = RandomizedSearchCV(
        estimator=base_models[model_name],
        param_distributions=param_grids[model_name],
        n_iter=n_iter,
        scoring='f1',
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print(f'Best params for {model_name}: {search.best_params_}')
    print(f'Best CV F1: {search.best_score_:.4f}')
    return search.best_estimator_


def save_model(model, name: str, models_dir: str = MODELS_DIR) -> str:
    """Save a trained model to disk using joblib."""
    os.makedirs(models_dir, exist_ok=True)
    path = os.path.join(models_dir, f'{name}.joblib')
    joblib.dump(model, path)
    print(f'Model saved: {path}')
    return path


def load_model(name: str, models_dir: str = MODELS_DIR) -> object:
    """Load a previously saved model from disk."""
    path = os.path.join(models_dir, f'{name}.joblib')
    return joblib.load(path)
