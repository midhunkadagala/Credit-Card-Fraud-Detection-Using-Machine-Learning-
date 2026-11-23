"""Quick check that saved models load and metrics are available."""

from src.evaluation import build_comparison_table, evaluate_bundle, select_best_model
from src.train_model import list_saved_models, load_model_bundle


def main() -> None:
    models = list_saved_models()
    print("Saved models:", models)
    metrics = []
    for name in models:
        bundle = load_model_bundle(name)
        m = evaluate_bundle(bundle)
        metrics.append(m)
        print(
            f"{name}: Acc={m['Accuracy']:.4f} F1={m['F1-Score']:.4f} "
            f"Recall={m['Recall']:.4f} ROC-AUC={m['ROC-AUC']:.4f}"
        )
    if metrics:
        cmp = build_comparison_table(metrics)
        print(cmp.to_string(index=False))
        print("Best:", select_best_model(cmp))
    print("VERIFY_OK")


if __name__ == "__main__":
    main()
