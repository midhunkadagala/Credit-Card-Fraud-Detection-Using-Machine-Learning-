"""
explainability.py
-----------------
SHAP-based explanations for model predictions.

The Kaggle dataset uses anonymized PCA features (V1–V28), so explanations
refer to these feature names — not merchant/location fields.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import shap

from src.train_model import (
    MODEL_ISOLATION_FOREST,
    MODEL_LOGISTIC,
    MODEL_RANDOM_FOREST,
    TrainedModelBundle,
)

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SHAP_DIR = os.path.join(PROJECT_ROOT, "outputs", "shap_plots")


def ensure_shap_dir() -> str:
    os.makedirs(SHAP_DIR, exist_ok=True)
    return SHAP_DIR


def _background_sample(X: pd.DataFrame, max_rows: int = 200, random_state: int = 42) -> pd.DataFrame:
    """Small background set for KernelExplainer / TreeExplainer speed."""
    if len(X) <= max_rows:
        return X
    return X.sample(max_rows, random_state=random_state)


def compute_shap_values(
    bundle: TrainedModelBundle,
    X_explain: pd.DataFrame,
    max_explain_rows: int = 100,
    background_rows: int = 100,
) -> Tuple[Any, pd.DataFrame, str]:
    """
    Compute SHAP values for a trained model.

    Returns
    -------
    (shap_values, X_used, notes)
    """
    if bundle.model_name == MODEL_ISOLATION_FOREST:
        raise ValueError(
            "SHAP Tree/Kernel explanations for Isolation Forest are limited in this "
            "project demo. Please use Logistic Regression or Random Forest for "
            "Explainable AI pages."
        )

    X_used = _background_sample(X_explain, max_rows=max_explain_rows)
    background = _background_sample(X_explain, max_rows=background_rows)

    notes = ""
    if bundle.model_name == MODEL_RANDOM_FOREST:
        explainer = shap.TreeExplainer(bundle.model)
        shap_values = explainer.shap_values(X_used)
        # For binary classifiers, shap_values may be a list [class0, class1]
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # fraud class
        notes = "TreeExplainer used for Random Forest (fast and exact for trees)."
    elif bundle.model_name == MODEL_LOGISTIC:
        # Linear models: prefer LinearExplainer when possible
        try:
            explainer = shap.LinearExplainer(bundle.model, background)
            shap_values = explainer.shap_values(X_used)
            notes = "LinearExplainer used for Logistic Regression."
        except Exception:
            # Fallback
            predict_fn = lambda data: bundle.model.predict_proba(data)[:, 1]
            explainer = shap.KernelExplainer(predict_fn, background)
            shap_values = explainer.shap_values(X_used, nsamples=100)
            notes = "KernelExplainer fallback used for Logistic Regression."
    else:
        raise ValueError(f"SHAP not configured for model: {bundle.model_name}")

    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    shap_values = np.asarray(shap_values)
    return shap_values, X_used, notes


def global_mean_abs_shap(shap_values: np.ndarray, feature_names: List[str]) -> pd.DataFrame:
    """Mean |SHAP| per feature — global importance ranking."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({"Feature": feature_names, "Mean |SHAP|": mean_abs})
    return df.sort_values("Mean |SHAP|", ascending=False).reset_index(drop=True)


def plot_global_importance(importance_df: pd.DataFrame, top_n: int = 15):
    """Plotly bar chart of global SHAP importance."""
    top = importance_df.head(top_n).iloc[::-1]
    fig = px.bar(
        top,
        x="Mean |SHAP|",
        y="Feature",
        orientation="h",
        title=f"Global Feature Importance (Top {top_n} by Mean |SHAP|)",
    )
    fig.update_layout(height=480)
    return fig


def create_summary_plot_figure(
    shap_values: np.ndarray,
    X_used: pd.DataFrame,
    max_display: int = 15,
):
    """
    Create a matplotlib SHAP summary (beeswarm) figure for Streamlit.

    Returns a matplotlib Figure.
    """
    ensure_shap_dir()
    fig = plt.figure(figsize=(10, 6))
    shap.summary_plot(
        shap_values,
        X_used,
        show=False,
        max_display=max_display,
        plot_size=None,
    )
    plt.tight_layout()
    save_path = os.path.join(SHAP_DIR, "shap_summary.png")
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    return fig, save_path


def explain_single_prediction(
    bundle: TrainedModelBundle,
    row: pd.Series,
    background: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Explain one transaction.

    Returns feature contributions that push toward fraud vs away from fraud.
    """
    if bundle.model_name == MODEL_ISOLATION_FOREST:
        raise ValueError(
            "Individual SHAP explanations are supported for Logistic Regression "
            "and Random Forest in this project."
        )

    X_row = pd.DataFrame([row[bundle.feature_names]])
    bg = _background_sample(background[bundle.feature_names], max_rows=80)

    if bundle.model_name == MODEL_RANDOM_FOREST:
        explainer = shap.TreeExplainer(bundle.model)
        sv = explainer.shap_values(X_row)
        if isinstance(sv, list):
            sv = sv[1]
        values = np.asarray(sv).reshape(-1)
        base = float(
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        )
    else:
        try:
            explainer = shap.LinearExplainer(bundle.model, bg)
            sv = explainer.shap_values(X_row)
            values = np.asarray(sv).reshape(-1)
            base = float(explainer.expected_value)
        except Exception:
            predict_fn = lambda data: bundle.model.predict_proba(data)[:, 1]
            explainer = shap.KernelExplainer(predict_fn, bg)
            sv = explainer.shap_values(X_row, nsamples=100)
            values = np.asarray(sv).reshape(-1)
            base = float(explainer.expected_value)

    contrib = pd.DataFrame(
        {
            "Feature": bundle.feature_names,
            "Feature Value": X_row.iloc[0].values,
            "SHAP Value": values,
        }
    )
    contrib["Direction"] = np.where(
        contrib["SHAP Value"] > 0, "Increases fraud score", "Decreases fraud score"
    )
    contrib = contrib.sort_values("SHAP Value", key=abs, ascending=False)

    toward_fraud = contrib[contrib["SHAP Value"] > 0].head(8)
    away_fraud = contrib[contrib["SHAP Value"] < 0].head(8)

    simple_text = build_simple_explanation(toward_fraud, away_fraud, base)

    return {
        "base_value": base,
        "contributions": contrib,
        "toward_fraud": toward_fraud,
        "away_fraud": away_fraud,
        "simple_explanation": simple_text,
    }


def build_simple_explanation(
    toward_fraud: pd.DataFrame,
    away_fraud: pd.DataFrame,
    base_value: float,
) -> str:
    """Beginner-friendly paragraph describing SHAP contributions."""
    lines = [
        "How to read this explanation:",
        f"- The model's baseline (average) score starts near {base_value:.4f}.",
        "- Features with positive SHAP values push the prediction toward FRAUD.",
        "- Features with negative SHAP values push the prediction toward LEGITIMATE.",
        "",
        "Top features increasing the fraud score:",
    ]
    if toward_fraud.empty:
        lines.append("  (none with positive contribution in this sample)")
    else:
        for _, r in toward_fraud.head(5).iterrows():
            lines.append(
                f"  • {r['Feature']} = {r['Feature Value']:.4f} "
                f"(SHAP {r['SHAP Value']:+.4f})"
            )

    lines.append("")
    lines.append("Top features decreasing the fraud score:")
    if away_fraud.empty:
        lines.append("  (none with negative contribution in this sample)")
    else:
        for _, r in away_fraud.head(5).iterrows():
            lines.append(
                f"  • {r['Feature']} = {r['Feature Value']:.4f} "
                f"(SHAP {r['SHAP Value']:+.4f})"
            )

    lines.append("")
    lines.append(
        "Note: V1–V28 are anonymized PCA components from the original dataset, "
        "so they do not map to readable fields like merchant name or city. "
        "SHAP still shows which of these components most influenced the decision."
    )
    return "\n".join(lines)


def plot_waterfall_like(contrib: pd.DataFrame, top_n: int = 12):
    """Simple Plotly bar of SHAP contributions for one row."""
    top = contrib.head(top_n).copy()
    top["Color"] = np.where(top["SHAP Value"] > 0, "Toward fraud", "Toward legitimate")
    fig = px.bar(
        top.iloc[::-1],
        x="SHAP Value",
        y="Feature",
        color="Color",
        color_discrete_map={
            "Toward fraud": "#E94F37",
            "Toward legitimate": "#2E86AB",
        },
        title="Individual Prediction — Feature Contributions (SHAP)",
    )
    fig.update_layout(height=480)
    return fig
