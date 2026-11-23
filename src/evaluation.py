"""
evaluation.py
-------------
Compute genuine evaluation metrics from actual model predictions.

No hardcoded / invented metric values are used.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.train_model import TrainedModelBundle

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHARTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "charts")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "outputs", "reports")


def ensure_output_dirs() -> None:
    os.makedirs(CHARTS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)


def compute_metrics(
    y_true,
    y_pred,
    y_score=None,
) -> Dict[str, Any]:
    """
    Compute classification metrics.

    For Isolation Forest, y_score is an anomaly score (higher = more anomalous).
    ROC-AUC / PR-AUC still work with ranking scores.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    metrics: Dict[str, Any] = {
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-Score": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": None,
        "PR-AUC": None,
    }

    if y_score is not None:
        try:
            metrics["ROC-AUC"] = float(roc_auc_score(y_true, y_score))
        except ValueError:
            metrics["ROC-AUC"] = None
        try:
            metrics["PR-AUC"] = float(average_precision_score(y_true, y_score))
        except ValueError:
            metrics["PR-AUC"] = None

    metrics["Confusion Matrix"] = confusion_matrix(y_true, y_pred).tolist()
    metrics["Classification Report"] = classification_report(
        y_true, y_pred, target_names=["Legitimate (0)", "Fraud (1)"], zero_division=0
    )
    return metrics


def evaluate_bundle(bundle: TrainedModelBundle) -> Dict[str, Any]:
    """Evaluate a trained model bundle on its held-out test predictions."""
    if bundle.y_test is None or bundle.y_pred is None:
        raise ValueError(
            f"Model '{bundle.model_name}' has no test predictions. Train the model first."
        )
    metrics = compute_metrics(bundle.y_test, bundle.y_pred, bundle.y_score)
    metrics["Model"] = bundle.model_name
    metrics["Trained At"] = bundle.trained_at
    metrics["Uses SMOTE"] = bundle.uses_smote
    return metrics


def build_comparison_table(metrics_list: List[Dict[str, Any]]) -> pd.DataFrame:
    """Create a side-by-side comparison DataFrame."""
    rows = []
    for m in metrics_list:
        rows.append(
            {
                "Model": m.get("Model", "Unknown"),
                "Accuracy": m.get("Accuracy"),
                "Precision": m.get("Precision"),
                "Recall": m.get("Recall"),
                "F1-Score": m.get("F1-Score"),
                "ROC-AUC": m.get("ROC-AUC"),
                "PR-AUC": m.get("PR-AUC"),
                "Uses SMOTE": m.get("Uses SMOTE"),
            }
        )
    df = pd.DataFrame(rows)
    return df


def select_best_model(comparison_df: pd.DataFrame, primary_metric: str = "F1-Score") -> str:
    """
    Choose the best model using a fraud-relevant primary metric.

    Default: F1-Score (balances precision and recall under imbalance).
    Ties broken by higher Recall, then PR-AUC.
    """
    if comparison_df.empty:
        return "No models available"

    df = comparison_df.copy()
    sort_cols = [primary_metric]
    for col in ["Recall", "PR-AUC", "ROC-AUC"]:
        if col in df.columns and col not in sort_cols:
            sort_cols.append(col)

    df_sorted = df.sort_values(by=sort_cols, ascending=False, na_position="last")
    return str(df_sorted.iloc[0]["Model"])


def plot_confusion_matrix_plotly(cm: list, model_name: str):
    """Interactive confusion matrix heatmap."""
    z = np.array(cm)
    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=["Pred Legit (0)", "Pred Fraud (1)"],
            y=["Actual Legit (0)", "Actual Fraud (1)"],
            colorscale="Blues",
            text=z,
            texttemplate="%{text}",
            showscale=True,
        )
    )
    fig.update_layout(
        title=f"Confusion Matrix — {model_name}",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=420,
    )
    return fig


def plot_roc_curves(bundles: List[TrainedModelBundle]):
    """Overlay ROC curves for multiple models."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random baseline",
            line=dict(dash="dash", color="gray"),
        )
    )
    for b in bundles:
        if b.y_test is None or b.y_score is None:
            continue
        try:
            fpr, tpr, _ = roc_curve(b.y_test, b.y_score)
            auc = roc_auc_score(b.y_test, b.y_score)
            fig.add_trace(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    mode="lines",
                    name=f"{b.model_name} (AUC={auc:.3f})",
                )
            )
        except ValueError:
            continue

    fig.update_layout(
        title="ROC Curves",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_pr_curves(bundles: List[TrainedModelBundle]):
    """Overlay Precision-Recall curves for multiple models."""
    fig = go.Figure()
    for b in bundles:
        if b.y_test is None or b.y_score is None:
            continue
        try:
            precision, recall, _ = precision_recall_curve(b.y_test, b.y_score)
            ap = average_precision_score(b.y_test, b.y_score)
            fig.add_trace(
                go.Scatter(
                    x=recall,
                    y=precision,
                    mode="lines",
                    name=f"{b.model_name} (AP={ap:.3f})",
                )
            )
        except ValueError:
            continue

    fig.update_layout(
        title="Precision-Recall Curves",
        xaxis_title="Recall",
        yaxis_title="Precision",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    return fig


def plot_class_distribution(n_legit: int, n_fraud: int):
    """Bar chart of legitimate vs fraudulent transactions."""
    df = pd.DataFrame(
        {
            "Class": ["Legitimate (0)", "Fraudulent (1)"],
            "Count": [n_legit, n_fraud],
        }
    )
    fig = px.bar(
        df,
        x="Class",
        y="Count",
        color="Class",
        color_discrete_map={
            "Legitimate (0)": "#2E86AB",
            "Fraudulent (1)": "#E94F37",
        },
        title="Class Distribution (Highly Imbalanced)",
        text="Count",
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(showlegend=False, height=420)
    return fig


def plot_feature_distribution(df: pd.DataFrame, feature: str, sample_n: int = 5000):
    """Histogram of a selected feature, colored by Class."""
    plot_df = df
    if len(df) > sample_n:
        plot_df = df.sample(sample_n, random_state=42)
    fig = px.histogram(
        plot_df,
        x=feature,
        color="Class",
        barmode="overlay",
        opacity=0.65,
        title=f"Distribution of {feature} by Class",
        color_discrete_map={0: "#2E86AB", 1: "#E94F37"},
    )
    fig.update_layout(height=420)
    return fig


def plot_feature_importance(importances: Dict[str, float], model_name: str, top_n: int = 15):
    """Horizontal bar chart of feature importances / coefficient magnitudes."""
    s = pd.Series(importances).sort_values(ascending=False).head(top_n)
    fig = px.bar(
        x=s.values,
        y=s.index,
        orientation="h",
        title=f"Top {top_n} Feature Importances — {model_name}",
        labels={"x": "Importance", "y": "Feature"},
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=480)
    return fig


def save_classification_report(text: str, model_name: str) -> str:
    """Write classification report text to outputs/reports/."""
    ensure_output_dirs()
    safe = model_name.lower().replace(" ", "_")
    path = os.path.join(REPORTS_DIR, f"{safe}_classification_report.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    return path


def imbalance_accuracy_explanation() -> str:
    """Plain-language explanation for viva / dashboard."""
    return (
        "Accuracy alone is not enough for fraud detection because the dataset is "
        "highly imbalanced. Typically more than 99% of transactions are legitimate. "
        "A naive model that always predicts 'legitimate' would achieve ~99% accuracy "
        "while catching zero frauds. Precision, Recall, F1-score, ROC-AUC, and "
        "especially Precision-Recall AUC better reflect real fraud-catching performance."
    )
