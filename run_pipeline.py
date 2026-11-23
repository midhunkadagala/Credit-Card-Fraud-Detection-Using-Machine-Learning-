"""
One-shot pipeline runner for local verification.
Uses a CSV path passed on the command line (or CREDITCARD_CSV env var).
Does NOT bundle or download the dataset.
"""

from __future__ import annotations

import os
import sys
import time

import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from src.data_loader import get_dataset_info, validate_dataset_schema
from src.evaluation import (
    build_comparison_table,
    evaluate_bundle,
    save_classification_report,
    select_best_model,
)
from src.preprocessing import (
    drop_duplicates_and_report,
    handle_missing_values,
    prepare_data,
    prepare_data_for_isolation_forest,
)
from src.train_model import (
    MODEL_ISOLATION_FOREST,
    MODEL_LOGISTIC,
    MODEL_RANDOM_FOREST,
    save_model_bundle,
    train_model,
)


def main() -> int:
    csv_path = None
    if len(sys.argv) > 1:
        csv_path = sys.argv[1]
    else:
        csv_path = os.environ.get("CREDITCARD_CSV")

    if not csv_path:
        print("ERROR: Provide dataset path.")
        print('Usage: python run_pipeline.py "C:\\Users\\...\\creditcard.csv"')
        return 1

    if not os.path.isfile(csv_path):
        print(f"ERROR: File not found: {csv_path}")
        return 1

    print(f"Loading: {csv_path}")
    t0 = time.time()
    df = pd.read_csv(csv_path)
    print(f"Raw shape: {df.shape}  ({time.time() - t0:.1f}s)")

    validation = validate_dataset_schema(df)
    if not validation.is_valid:
        print(f"VALIDATION FAILED: {validation.message}")
        if validation.missing_columns:
            print("Missing:", ", ".join(validation.missing_columns))
        if validation.extra_info:
            print(validation.extra_info)
        return 1
    print(validation.message)

    df, n_miss = handle_missing_values(df, strategy="drop")
    df, n_dup = drop_duplicates_and_report(df)
    print(f"After clean: shape={df.shape}, missing_rows_dropped={n_miss}, duplicates_removed={n_dup}")

    info = get_dataset_info(df, os.path.basename(csv_path))
    print(
        f"Filename={info.filename} | rows={info.n_rows:,} | cols={info.n_columns} | "
        f"fraud={info.n_fraud:,} | legit={info.n_legit:,} | fraud%={info.fraud_pct:.4f}"
    )

    metrics_list = []
    bundles = []

    print("\n=== Training Logistic Regression ===")
    t1 = time.time()
    prepared_lr = prepare_data(df, test_size=0.2, random_state=42, apply_smote=True)
    print(
        f"Train={len(prepared_lr.X_train):,} Test={len(prepared_lr.X_test):,} "
        f"SMOTE train={len(prepared_lr.X_train_resampled):,}"
    )
    bundle_lr = train_model(MODEL_LOGISTIC, prepared_lr, max_iter=1000, C=1.0)
    path_lr = save_model_bundle(bundle_lr)
    m_lr = evaluate_bundle(bundle_lr)
    save_classification_report(m_lr["Classification Report"], MODEL_LOGISTIC)
    metrics_list.append(m_lr)
    bundles.append(bundle_lr)
    print(
        f"Saved {path_lr} | Acc={m_lr['Accuracy']:.4f} Prec={m_lr['Precision']:.4f} "
        f"Rec={m_lr['Recall']:.4f} F1={m_lr['F1-Score']:.4f} "
        f"ROC-AUC={m_lr['ROC-AUC']:.4f} PR-AUC={m_lr['PR-AUC']:.4f} "
        f"({time.time() - t1:.1f}s)"
    )

    print("\n=== Training Random Forest ===")
    t2 = time.time()
    prepared_rf = prepare_data(df, test_size=0.2, random_state=42, apply_smote=True)
    bundle_rf = train_model(
        MODEL_RANDOM_FOREST,
        prepared_rf,
        n_estimators=100,
        max_depth=20,
    )
    path_rf = save_model_bundle(bundle_rf)
    m_rf = evaluate_bundle(bundle_rf)
    save_classification_report(m_rf["Classification Report"], MODEL_RANDOM_FOREST)
    metrics_list.append(m_rf)
    bundles.append(bundle_rf)
    print(
        f"Saved {path_rf} | Acc={m_rf['Accuracy']:.4f} Prec={m_rf['Precision']:.4f} "
        f"Rec={m_rf['Recall']:.4f} F1={m_rf['F1-Score']:.4f} "
        f"ROC-AUC={m_rf['ROC-AUC']:.4f} PR-AUC={m_rf['PR-AUC']:.4f} "
        f"({time.time() - t2:.1f}s)"
    )

    print("\n=== Training Isolation Forest (no SMOTE) ===")
    t3 = time.time()
    prepared_if = prepare_data_for_isolation_forest(df, test_size=0.2, random_state=42)
    bundle_if = train_model(MODEL_ISOLATION_FOREST, prepared_if, n_estimators=100)
    path_if = save_model_bundle(bundle_if)
    m_if = evaluate_bundle(bundle_if)
    save_classification_report(m_if["Classification Report"], MODEL_ISOLATION_FOREST)
    metrics_list.append(m_if)
    bundles.append(bundle_if)
    print(
        f"Saved {path_if} | Acc={m_if['Accuracy']:.4f} Prec={m_if['Precision']:.4f} "
        f"Rec={m_if['Recall']:.4f} F1={m_if['F1-Score']:.4f} "
        f"ROC-AUC={m_if['ROC-AUC']} PR-AUC={m_if['PR-AUC']} "
        f"({time.time() - t3:.1f}s)"
    )

    comparison = build_comparison_table(metrics_list)
    print("\n=== Model Comparison ===")
    print(comparison.to_string(index=False))
    best = select_best_model(comparison, primary_metric="F1-Score")
    print(f"\nBest model by F1-Score: {best}")
    print(f"\nTotal pipeline time: {time.time() - t0:.1f}s")
    print("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
