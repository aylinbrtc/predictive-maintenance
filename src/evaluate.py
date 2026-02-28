"""
evaluate.py
-----------
Evaluation metrics and publication-quality visualizations for model comparison.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score, classification_report,
)


FIGURES_DIR = os.path.join(os.path.dirname(__file__), '..', 'figures')
sns.set_theme(style='whitegrid', palette='muted', font_scale=1.1)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                    y_prob: np.ndarray = None) -> dict:
    """
    Compute classification metrics for a single model.

    Parameters
    ----------
    y_true : true labels
    y_pred : predicted labels
    y_prob : predicted probabilities for the positive class (optional, needed for AUC)

    Returns
    -------
    dict with keys: accuracy, precision, recall, f1, roc_auc (if y_prob provided)
    """
    metrics = {
        'accuracy':  accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall':    recall_score(y_true, y_pred, zero_division=0),
        'f1':        f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
        metrics['avg_precision'] = average_precision_score(y_true, y_prob)
    return metrics


def build_comparison_table(results: dict) -> pd.DataFrame:
    """
    Build a summary DataFrame comparing all models.

    Parameters
    ----------
    results : dict mapping model name -> metrics dict (from compute_metrics)

    Returns
    -------
    DataFrame with one row per model.
    """
    rows = []
    for name, metrics in results.items():
        row = {'Model': name}
        row.update(metrics)
        rows.append(row)
    df = pd.DataFrame(rows).set_index('Model')
    return df.round(4)


def plot_confusion_matrices(models_preds: dict, y_true: np.ndarray,
                            save: bool = True) -> None:
    """
    Plot confusion matrices for all models side by side.

    Parameters
    ----------
    models_preds : dict mapping model name -> predicted labels
    y_true       : true labels
    save         : whether to save to figures/
    """
    n = len(models_preds)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, (name, y_pred) in zip(axes, models_preds.items()):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['No Failure', 'Failure'],
                    yticklabels=['No Failure', 'Failure'])
        ax.set_title(f'{name}', fontweight='bold')
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')

    plt.suptitle('Confusion Matrices — All Models', fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_roc_curves(models_probs: dict, y_true: np.ndarray, save: bool = True) -> None:
    """
    Plot ROC curves for all models on a single figure.

    Parameters
    ----------
    models_probs : dict mapping model name -> predicted probabilities (positive class)
    y_true       : true labels
    save         : whether to save to figures/
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.50)')

    for name, y_prob in models_probs.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        ax.plot(fpr, tpr, linewidth=2, label=f'{name} (AUC = {auc:.3f})')

    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve Comparison', fontweight='bold')
    ax.legend(loc='lower right')
    plt.tight_layout()
    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'roc_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_precision_recall_curves(models_probs: dict, y_true: np.ndarray,
                                 save: bool = True) -> None:
    """
    Plot Precision-Recall curves for all models (more informative for imbalanced data).
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    for name, y_prob in models_probs.items():
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        ax.plot(recall, precision, linewidth=2, label=f'{name} (AP = {ap:.3f})')

    baseline = y_true.mean()
    ax.axhline(baseline, color='k', linestyle='--', label=f'Baseline (AP = {baseline:.3f})')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve Comparison', fontweight='bold')
    ax.legend(loc='upper right')
    plt.tight_layout()
    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'pr_curves.png'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_feature_importance(importances: dict, top_n: int = 10, save: bool = True) -> None:
    """
    Plot top-N feature importances for one or more tree-based models.

    Parameters
    ----------
    importances : dict mapping model name -> pd.Series (feature name -> importance score)
    top_n       : number of top features to display
    save        : whether to save to figures/
    """
    n = len(importances)
    fig, axes = plt.subplots(1, n, figsize=(7 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, imp_series) in zip(axes, importances.items()):
        top = imp_series.nlargest(top_n).sort_values()
        ax.barh(top.index, top.values, color=sns.color_palette('muted')[0])
        ax.set_title(f'{name} — Top {top_n} Features', fontweight='bold')
        ax.set_xlabel('Importance')

    plt.tight_layout()
    if save:
        os.makedirs(FIGURES_DIR, exist_ok=True)
        plt.savefig(os.path.join(FIGURES_DIR, 'feature_importance.png'), dpi=300, bbox_inches='tight')
    plt.show()
