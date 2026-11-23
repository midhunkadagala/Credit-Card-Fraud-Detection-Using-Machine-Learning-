"""
Explainable Credit Card Fraud Detection — Streamlit Dashboard
=============================================================
Run locally:
    streamlit run app.py

DATASET RULE:
- The dataset is NEVER bundled or auto-downloaded.
- The user must manually upload a CSV at runtime.
"""

from __future__ import annotations

import os
import sys
import traceback
from typing import Optional

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Ensure project root is on sys.path when launched via Streamlit
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import (  # noqa: E402
    REQUIRED_FEATURES,
    get_dataset_info,
    load_uploaded_csv,
    validate_prediction_csv,
)
from src.evaluation import (  # noqa: E402
    build_comparison_table,
    evaluate_bundle,
    imbalance_accuracy_explanation,
    plot_class_distribution,
    plot_confusion_matrix_plotly,
    plot_feature_distribution,
    plot_feature_importance,
    plot_pr_curves,
    plot_roc_curves,
    save_classification_report,
    select_best_model,
)
from src.explainability import (  # noqa: E402
    compute_shap_values,
    create_summary_plot_figure,
    explain_single_prediction,
    global_mean_abs_shap,
    plot_global_importance,
    plot_waterfall_like,
)
from src.prediction import build_downloadable_report, predict_dataframe  # noqa: E402
from src.preprocessing import (  # noqa: E402
    drop_duplicates_and_report,
    handle_missing_values,
    prepare_data,
    prepare_data_for_isolation_forest,
)
from src.train_model import (  # noqa: E402
    MODEL_ISOLATION_FOREST,
    MODEL_LOGISTIC,
    MODEL_RANDOM_FOREST,
    SUPPORTED_MODELS,
    list_saved_models,
    load_model_bundle,
    save_model_bundle,
    train_model,
)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Explainable Credit Card Fraud Detection",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

PAGES = [
    "Home Dashboard",
    "Data Analysis",
    "Model Training",
    "Model Comparison",
    "Fraud Prediction",
    "Explainable AI",
]


def init_session_state() -> None:
    defaults = {
        "dataset": None,
        "dataset_filename": None,
        "dataset_info": None,
        "validation_message": None,
        "trained_models": {},  # name -> TrainedModelBundle
        "metrics": {},  # name -> metrics dict
        "prepared_cache": {},
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def require_dataset() -> bool:
    """Return True if a valid dataset is loaded; otherwise show a blocking message."""
    if st.session_state["dataset"] is None:
        st.warning("Please upload your credit card fraud detection dataset to continue.")
        st.info(
            "Go to **Home Dashboard**, upload a CSV matching the Kaggle Credit Card "
            "Fraud Detection format (`Time`, `V1`–`V28`, `Amount`, `Class`), then return here."
        )
        return False
    return True


def ensure_models_loaded_from_disk() -> list:
    """
    Load every saved model from models/ into session state.

    Always merges from disk (not only when session is empty), so Fraud Prediction
    shows Logistic Regression, Random Forest, and Isolation Forest together.
    """
    for name in list_saved_models():
        if name not in st.session_state["trained_models"]:
            try:
                bundle = load_model_bundle(name)
                st.session_state["trained_models"][name] = bundle
                st.session_state["metrics"][name] = evaluate_bundle(bundle)
            except Exception:
                pass
    return list(st.session_state["trained_models"].keys())


def _commit_loaded_dataset(
    df: pd.DataFrame,
    filename: str,
    validation_message: str,
    refresh_sidebar: bool = False,
) -> None:
    """Store a validated dataset in session state and show a short summary."""
    cleaned, n_missing_rows = handle_missing_values(df, strategy="drop")
    cleaned, n_dupes = drop_duplicates_and_report(cleaned)
    info = get_dataset_info(cleaned, filename)

    st.session_state["dataset"] = cleaned
    st.session_state["dataset_filename"] = filename
    st.session_state["dataset_info"] = info
    st.session_state["validation_message"] = validation_message
    # Keep already-trained models on disk; clear only in-session caches tied to old data
    st.session_state["trained_models"] = {}
    st.session_state["metrics"] = {}
    st.session_state["prepared_cache"] = {}

    st.success(validation_message)
    if n_missing_rows:
        st.warning(f"Dropped {n_missing_rows} row(s) containing missing values.")
    if n_dupes:
        st.warning(f"Removed {n_dupes} duplicate row(s).")

    st.write(f"**Filename:** `{info.filename}`")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{info.n_rows:,}")
    c2.metric("Columns", info.n_columns)
    c3.metric("Fraud", f"{info.n_fraud:,}")
    c4.metric("Legitimate", f"{info.n_legit:,}")
    st.dataframe(cleaned.head(10), width="stretch")
    if refresh_sidebar:
        st.rerun()


def upload_dataset_widget(key: str = "main_uploader") -> None:
    """Shared uploader used on Home (and optionally elsewhere)."""
    st.subheader("Manual Dataset Upload")
    st.markdown(
        "**Please upload your credit card fraud detection dataset to continue.**"
    )
    st.caption(
        "The dataset is not included in this project. Select a CSV from your computer, "
        "or paste a full local file path. Nothing is downloaded from Kaggle or any other website."
    )

    uploaded = st.file_uploader(
        "Select CSV file",
        type=["csv"],
        key=key,
        help="Expected columns: Time, V1–V28, Amount, Class",
    )

    if uploaded is not None:
        df, validation = load_uploaded_csv(uploaded)
        if not validation.is_valid:
            st.error(validation.message)
            if validation.missing_columns:
                st.error("Missing columns: " + ", ".join(validation.missing_columns))
            if validation.extra_info:
                st.caption(validation.extra_info)
        else:
            _commit_loaded_dataset(df, uploaded.name, validation.message)
            return

    st.markdown("**Or load from a local file path on this PC** (useful for large CSVs):")
    with st.form(key=f"{key}_path_form", clear_on_submit=False):
        local_path = st.text_input(
            "Full path to CSV",
            value="",
            placeholder=r"C:\Users\YourName\Downloads\creditcard.csv",
        )
        load_path = st.form_submit_button("Load path", type="primary")

    if load_path:
        path = (local_path or "").strip().strip('"')
        if not path:
            st.error("Please paste the full path to your CSV file.")
        elif not os.path.isfile(path):
            st.error(f"File not found: {path}")
        elif not path.lower().endswith(".csv"):
            st.error("Please provide a .csv file.")
        else:
            with st.spinner("Loading CSV from disk…"):
                try:
                    raw = pd.read_csv(path)
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Could not read the CSV file. Error: {exc}")
                    return
                from src.data_loader import validate_dataset_schema

                validation = validate_dataset_schema(raw)
                if not validation.is_valid:
                    st.error(validation.message)
                    if validation.missing_columns:
                        st.error("Missing columns: " + ", ".join(validation.missing_columns))
                    if validation.extra_info:
                        st.caption(validation.extra_info)
                    return
                raw["Class"] = raw["Class"].astype(int)
                _commit_loaded_dataset(
                    raw,
                    os.path.basename(path),
                    validation.message,
                    refresh_sidebar=True,
                )
                return

    if st.session_state["dataset"] is None:
        st.info("Waiting for CSV upload…")


# ===========================================================================
# PAGE 1 — Home
# ===========================================================================
def page_home() -> None:
    st.title("Explainable Credit Card Fraud Detection")
    st.markdown(
        """
        ### Project Introduction
        This M.Tech demonstration system detects potentially fraudulent credit card
        transactions using machine learning, compares multiple algorithms, handles
        class imbalance with **SMOTE**, and explains model decisions with **SHAP**.

        **Models:** Logistic Regression · Random Forest · Isolation Forest

        > The original dataset is **not** shipped with this repository.
        > You must upload it manually every time you want to train or analyse.
        """
    )
    upload_dataset_widget(key="home_uploader")

    info = st.session_state.get("dataset_info")
    if info is not None:
        st.markdown("---")
        st.subheader("Dataset Summary")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Total Transactions", f"{info.n_rows:,}")
        m2.metric("Fraudulent", f"{info.n_fraud:,}")
        m3.metric("Legitimate", f"{info.n_legit:,}")
        m4.metric("Fraud %", f"{info.fraud_pct:.4f}%")
        m5.metric("Missing Values", info.n_missing)
        st.plotly_chart(
            plot_class_distribution(info.n_legit, info.n_fraud),
            use_container_width=True,
        )


# ===========================================================================
# PAGE 2 — Data Analysis
# ===========================================================================
def page_data_analysis() -> None:
    st.title("Data Analysis")
    if not require_dataset():
        return

    df = st.session_state["dataset"]
    info = st.session_state["dataset_info"]

    st.subheader("Dataset Preview")
    st.dataframe(df.head(50), use_container_width=True)

    st.subheader("Missing-Value Analysis")
    missing = df.isna().sum()
    missing_df = pd.DataFrame({"Column": missing.index, "Missing Count": missing.values})
    st.dataframe(missing_df, use_container_width=True)
    if info.n_missing == 0:
        st.success("No missing values in the loaded dataset.")
    else:
        st.warning(f"Total missing cells: {info.n_missing}")

    st.subheader("Duplicate Analysis")
    st.write(f"Duplicate rows detected before cleaning were removed on upload. Current rows: **{len(df):,}**.")

    st.subheader("Fraud Distribution")
    st.plotly_chart(
        plot_class_distribution(info.n_legit, info.n_fraud),
        use_container_width=True,
    )
    st.info(imbalance_accuracy_explanation())

    st.subheader("Feature Distribution Charts")
    feature = st.selectbox("Select feature", REQUIRED_FEATURES, index=REQUIRED_FEATURES.index("Amount"))
    st.plotly_chart(plot_feature_distribution(df, feature), use_container_width=True)


# ===========================================================================
# PAGE 3 — Model Training
# ===========================================================================
def page_model_training() -> None:
    st.title("Model Training")
    if not require_dataset():
        return

    df = st.session_state["dataset"]

    st.markdown(
        "Train one model at a time. SMOTE is applied **only** to training data for "
        "Logistic Regression and Random Forest. Isolation Forest does **not** use SMOTE."
    )

    model_name = st.selectbox("Select model", SUPPORTED_MODELS)
    test_size = st.slider("Test size", 0.1, 0.4, 0.2, 0.05)
    random_state = st.number_input("Random state", min_value=0, value=42, step=1)

    kwargs = {}
    if model_name == MODEL_LOGISTIC:
        kwargs["C"] = st.slider("Regularization C", 0.01, 10.0, 1.0)
        kwargs["max_iter"] = st.number_input("Max iterations", 100, 5000, 1000, 100)
        apply_smote = st.checkbox("Apply SMOTE (train only)", value=True)
    elif model_name == MODEL_RANDOM_FOREST:
        kwargs["n_estimators"] = st.slider("n_estimators", 50, 300, 100, 10)
        kwargs["max_depth"] = st.slider("max_depth", 5, 40, 20, 1)
        apply_smote = st.checkbox("Apply SMOTE (train only)", value=True)
    else:
        kwargs["n_estimators"] = st.slider("n_estimators", 50, 300, 100, 10)
        cont = st.number_input(
            "Contamination (leave ~0 to auto-estimate from train fraud rate)",
            min_value=0.0,
            max_value=0.5,
            value=0.0,
            step=0.0001,
            format="%.4f",
        )
        if cont > 0:
            kwargs["contamination"] = float(cont)
        apply_smote = False
        st.caption("Isolation Forest is unsupervised — SMOTE is not applied.")

    if st.button("Train Selected Model", type="primary"):
        progress = st.progress(0, text="Preparing data…")
        try:
            if model_name == MODEL_ISOLATION_FOREST:
                prepared = prepare_data_for_isolation_forest(
                    df, test_size=test_size, random_state=int(random_state)
                )
            else:
                prepared = prepare_data(
                    df,
                    test_size=test_size,
                    random_state=int(random_state),
                    apply_smote=apply_smote,
                    scale_features=True,
                )
            progress.progress(40, text=f"Training {model_name}…")
            bundle = train_model(model_name, prepared, **kwargs)
            progress.progress(75, text="Evaluating…")
            metrics = evaluate_bundle(bundle)
            path = save_model_bundle(bundle)
            save_classification_report(metrics["Classification Report"], model_name)
            progress.progress(100, text="Done")

            st.session_state["trained_models"][model_name] = bundle
            st.session_state["metrics"][model_name] = metrics
            st.session_state["prepared_cache"][model_name] = prepared

            st.success(f"Training complete. Model saved to `{path}`")
            st.write(bundle.training_notes)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{metrics['Accuracy']:.4f}")
            c2.metric("Precision", f"{metrics['Precision']:.4f}")
            c3.metric("Recall", f"{metrics['Recall']:.4f}")
            c4.metric("F1-Score", f"{metrics['F1-Score']:.4f}")
            c5, c6 = st.columns(2)
            c5.metric("ROC-AUC", f"{metrics['ROC-AUC']:.4f}" if metrics["ROC-AUC"] is not None else "N/A")
            c6.metric("PR-AUC", f"{metrics['PR-AUC']:.4f}" if metrics["PR-AUC"] is not None else "N/A")

            st.plotly_chart(
                plot_confusion_matrix_plotly(metrics["Confusion Matrix"], model_name),
                use_container_width=True,
            )
            st.text(metrics["Classification Report"])

            if bundle.feature_importances:
                st.plotly_chart(
                    plot_feature_importance(bundle.feature_importances, model_name),
                    use_container_width=True,
                )
        except Exception as exc:  # noqa: BLE001
            progress.empty()
            st.error(f"Training failed: {exc}")
            st.caption("Details (for debugging):")
            st.code(traceback.format_exc())

    # Show already-trained models in this session
    if st.session_state["trained_models"]:
        st.markdown("---")
        st.write("Trained in this session:", ", ".join(st.session_state["trained_models"].keys()))


# ===========================================================================
# PAGE 4 — Model Comparison
# ===========================================================================
def page_model_comparison() -> None:
    st.title("Model Comparison")
    if not require_dataset():
        return

    st.markdown(
        "Train all three models on the Model Training page first, or load saved models "
        "from the local `models/` folder."
    )

    if st.button("Load saved models from disk"):
        loaded = 0
        for name in list_saved_models():
            try:
                bundle = load_model_bundle(name)
                st.session_state["trained_models"][name] = bundle
                st.session_state["metrics"][name] = evaluate_bundle(bundle)
                loaded += 1
            except Exception as exc:  # noqa: BLE001
                st.warning(f"Could not load {name}: {exc}")
        st.success(f"Loaded {loaded} model(s).")

    bundles = list(st.session_state["trained_models"].values())
    if len(bundles) < 1:
        st.warning("No trained models available yet. Train models on the Model Training page.")
        return

    metrics_list = [st.session_state["metrics"][b.model_name] for b in bundles]
    comparison = build_comparison_table(metrics_list)
    st.subheader("Performance Comparison Table")
    st.dataframe(
        comparison.style.format(
            {
                "Accuracy": "{:.4f}",
                "Precision": "{:.4f}",
                "Recall": "{:.4f}",
                "F1-Score": "{:.4f}",
                "ROC-AUC": "{:.4f}",
                "PR-AUC": "{:.4f}",
            },
            na_rep="N/A",
        ),
        use_container_width=True,
    )

    best = select_best_model(comparison, primary_metric="F1-Score")
    st.success(
        f"Best-performing model (by F1-Score, then Recall / PR-AUC): **{best}**"
    )
    st.info(imbalance_accuracy_explanation())

    st.subheader("Confusion Matrices")
    cols = st.columns(len(bundles))
    for col, b in zip(cols, bundles):
        m = st.session_state["metrics"][b.model_name]
        with col:
            st.plotly_chart(
                plot_confusion_matrix_plotly(m["Confusion Matrix"], b.model_name),
                use_container_width=True,
            )

    st.subheader("ROC Curves")
    st.plotly_chart(plot_roc_curves(bundles), use_container_width=True)

    st.subheader("Precision-Recall Curves")
    st.plotly_chart(plot_pr_curves(bundles), use_container_width=True)

    for b in bundles:
        m = st.session_state["metrics"][b.model_name]
        with st.expander(f"Classification Report — {b.model_name}"):
            st.text(m["Classification Report"])


# ===========================================================================
# PAGE 5 — Fraud Prediction
# ===========================================================================
def page_fraud_prediction() -> None:
    st.title("Fraud Prediction")
    if not require_dataset():
        return

    available = ensure_models_loaded_from_disk()
    st.caption(
        "Available algorithms: Logistic Regression, Random Forest, Isolation Forest "
        "(only models already trained/saved will appear below)."
    )

    if not available:
        st.error("No trained model found. Please train a model on the Model Training page first.")
        return

    model_name = st.selectbox("Model for prediction", available)
    bundle = st.session_state["trained_models"][model_name]

    st.warning(
        "Predictions are **model outputs**, not confirmed fraud. Flagged transactions "
        "should be reviewed by a human analyst."
    )

    tab1, tab2 = st.tabs(["Select row from uploaded dataset", "Upload prediction CSV"])

    with tab1:
        df = st.session_state["dataset"]
        max_idx = len(df) - 1
        idx = st.number_input("Row index", min_value=0, max_value=max_idx, value=0, step=1)
        st.dataframe(df.iloc[[idx]][REQUIRED_FEATURES + (["Class"] if "Class" in df.columns else [])])
        if st.button("Predict selected row"):
            try:
                result, disclaimer = predict_dataframe(bundle, df.iloc[[idx]])
                st.dataframe(result, use_container_width=True)
                st.caption(disclaimer)
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

    with tab2:
        pred_file = st.file_uploader(
            "Upload CSV with columns Time, V1–V28, Amount (Class optional)",
            type=["csv"],
            key="pred_uploader",
        )
        if pred_file is not None:
            try:
                pred_df = pd.read_csv(pred_file)
                validation = validate_prediction_csv(pred_df)
                if not validation.is_valid:
                    st.error(validation.message)
                    if validation.missing_columns:
                        st.error("Missing: " + ", ".join(validation.missing_columns))
                else:
                    result, disclaimer = predict_dataframe(bundle, pred_df)
                    st.success(f"Predicted {len(result)} transaction(s).")
                    st.dataframe(result.head(100), use_container_width=True)
                    st.caption(disclaimer)
                    st.download_button(
                        "Download prediction report (CSV)",
                        data=build_downloadable_report(result),
                        file_name="fraud_predictions.csv",
                        mime="text/csv",
                    )
            except Exception as exc:  # noqa: BLE001
                st.error(f"Prediction failed: {exc}")


# ===========================================================================
# PAGE 6 — Explainable AI
# ===========================================================================
def page_explainable_ai() -> None:
    st.title("Explainable AI (SHAP)")
    if not require_dataset():
        return

    ensure_models_loaded_from_disk()
    # Prefer models that support SHAP well
    shap_models = [
        n
        for n in st.session_state["trained_models"]
        if n in (MODEL_LOGISTIC, MODEL_RANDOM_FOREST)
    ]

    if not shap_models:
        st.error(
            "Train Logistic Regression or Random Forest first. "
            "SHAP explanations in this demo focus on those two models."
        )
        return

    model_name = st.selectbox("Model to explain", shap_models)
    bundle = st.session_state["trained_models"][model_name]

    if bundle.X_test is None:
        st.error("Test features are missing from the model bundle. Please retrain the model.")
        return

    st.markdown(
        """
        **What is SHAP?**  
        SHAP (SHapley Additive exPlanations) shows how much each feature pushed a
        prediction toward fraud or toward legitimate. Positive SHAP → more fraud-like;
        negative SHAP → more legitimate-like.
        """
    )

    max_rows = st.slider("Rows for global SHAP (larger = slower)", 20, 200, 80, 10)

    if st.button("Compute Global SHAP", type="primary"):
        with st.spinner("Computing SHAP values… this may take a minute"):
            try:
                shap_values, X_used, notes = compute_shap_values(
                    bundle, bundle.X_test, max_explain_rows=max_rows
                )
                st.session_state["shap_cache"] = {
                    "model": model_name,
                    "shap_values": shap_values,
                    "X_used": X_used,
                    "notes": notes,
                }
                st.success(notes)
            except Exception as exc:  # noqa: BLE001
                st.error(f"SHAP computation failed: {exc}")
                return

    cache = st.session_state.get("shap_cache")
    if cache and cache.get("model") == model_name:
        importance = global_mean_abs_shap(cache["shap_values"], bundle.feature_names)
        st.subheader("Global Feature Importance")
        st.plotly_chart(plot_global_importance(importance), use_container_width=True)

        st.subheader("SHAP Summary Plot")
        try:
            fig, path = create_summary_plot_figure(cache["shap_values"], cache["X_used"])
            st.pyplot(fig)
            plt.close(fig)
            st.caption(f"Also saved to `{path}`")
        except Exception as exc:  # noqa: BLE001
            st.warning(f"Could not render summary plot: {exc}")

        st.subheader("Individual Transaction Explanation")
        idx = st.number_input(
            "Index within SHAP sample",
            min_value=0,
            max_value=len(cache["X_used"]) - 1,
            value=0,
            step=1,
        )
        if st.button("Explain this transaction"):
            try:
                row = cache["X_used"].iloc[idx]
                explanation = explain_single_prediction(bundle, row, cache["X_used"])
                st.plotly_chart(
                    plot_waterfall_like(explanation["contributions"]),
                    use_container_width=True,
                )
                st.markdown("#### Features increasing fraud score")
                st.dataframe(explanation["toward_fraud"], use_container_width=True)
                st.markdown("#### Features decreasing fraud score")
                st.dataframe(explanation["away_fraud"], use_container_width=True)
                st.markdown("#### Simple interpretation")
                st.text(explanation["simple_explanation"])
            except Exception as exc:  # noqa: BLE001
                st.error(f"Individual explanation failed: {exc}")
    else:
        st.info("Click **Compute Global SHAP** to generate explanations.")


# ===========================================================================
# Main
# ===========================================================================
def main() -> None:
    init_session_state()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", PAGES)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Dataset Status")
    if st.session_state["dataset"] is None:
        st.sidebar.error("No dataset uploaded")
        st.sidebar.caption("Please upload your credit card fraud detection dataset to continue.")
    else:
        info = st.session_state["dataset_info"]
        st.sidebar.success(f"Loaded: {info.filename}")
        st.sidebar.caption(f"{info.n_rows:,} rows · {info.n_fraud:,} fraud")

    st.sidebar.markdown("---")
    st.sidebar.caption("Local demo · No cloud · Manual CSV upload only")

    if page == "Home Dashboard":
        page_home()
    elif page == "Data Analysis":
        page_data_analysis()
    elif page == "Model Training":
        page_model_training()
    elif page == "Model Comparison":
        page_model_comparison()
    elif page == "Fraud Prediction":
        page_fraud_prediction()
    elif page == "Explainable AI":
        page_explainable_ai()


if __name__ == "__main__":
    main()
