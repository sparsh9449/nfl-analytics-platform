"""
Win Probability — final test-set evaluation.

!!!  Run this script exactly once, after all training and tuning is done.  !!!
!!!  Looking at test metrics mid-project and then re-tuning is data leakage. !!!

What this does
──────────────
1. Loads the trained model artifacts from artifacts/.
2. Runs evaluate_model (ROC-AUC, Brier, calibration table + plot) on
   validation (2023) and test (2024–2025) for both models.
3. Prints a side-by-side comparison so it is immediately clear whether
   XGBoost earns its complexity over the logistic regression baseline.
4. Writes the complete results to artifacts/metrics_summary.json.

Usage (from project root):
    python models/win_probability/final_eval.py
"""

import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.win_probability.data_prep import load_and_split, VAL_SEASONS, TEST_SEASONS
from models.win_probability.evaluate import evaluate_model
from models.win_probability.train import drop_ties

ARTIFACTS = ROOT / "models" / "win_probability" / "artifacts"


def load_models() -> dict:
    return {
        "logistic_regression": joblib.load(ARTIFACTS / "lr_model.pkl"),
        "xgboost":             joblib.load(ARTIFACTS / "xgb_model.pkl"),
    }


def print_comparison(all_metrics: dict) -> None:
    """Side-by-side val vs test table — the deliverable for interview / review."""
    models = list(all_metrics.keys())

    print("\n" + "=" * 72)
    print("  Final comparison — Validation (2023) vs Test (2024–2025)")
    print("=" * 72)
    print(f"  {'Model':<25} {'Val AUC':>9} {'Val Brier':>11} {'Test AUC':>10} {'Test Brier':>11}")
    print(f"  {'-'*68}")

    for name in models:
        v = all_metrics[name]["val_2023"]
        t = all_metrics[name]["test_2024_2025"]
        print(
            f"  {name:<25} {v['roc_auc']:>9.4f} {v['brier']:>11.4f}"
            f" {t['roc_auc']:>10.4f} {t['brier']:>11.4f}"
        )

    print()

    # Verdict — does XGBoost earn its complexity on test?
    lr_auc  = all_metrics["logistic_regression"]["test_2024_2025"]["roc_auc"]
    xgb_auc = all_metrics["xgboost"]["test_2024_2025"]["roc_auc"]
    lr_brier  = all_metrics["logistic_regression"]["test_2024_2025"]["brier"]
    xgb_brier = all_metrics["xgboost"]["test_2024_2025"]["brier"]

    auc_gap   = xgb_auc   - lr_auc
    brier_gap = xgb_brier - lr_brier   # negative = XGBoost is better

    winner = "XGBoost" if auc_gap > 0 else "Logistic Regression"
    print(f"  Verdict  : {winner} wins on test-set AUC")
    print(f"  AUC gap  : {auc_gap:+.4f}  (XGBoost minus LR)")
    print(f"  Brier gap: {brier_gap:+.4f}  (XGBoost minus LR, negative = XGBoost better)")
    print("=" * 72)


if __name__ == "__main__":
    print("=" * 72)
    print("  Win Probability — Final Evaluation")
    print("=" * 72)

    splits = load_and_split()

    X_val,  y_val  = drop_ties(splits.X_val,  splits.y_val)
    X_test, y_test = drop_ties(splits.X_test, splits.y_test)

    print(f"\n  Val  : {len(X_val):,} plays  (seasons {VAL_SEASONS})")
    print(f"  Test : {len(X_test):,} plays  (seasons {TEST_SEASONS})")

    models = load_models()
    all_metrics: dict = {}

    for name, model in models.items():
        all_metrics[name] = {}

        print(f"\n{'━' * 72}")
        print(f"  {name.upper()}")
        print(f"{'━' * 72}")

        all_metrics[name]["val_2023"] = evaluate_model(
            model, X_val, y_val,
            label=f"{name} — val (2023)",
            save_dir=ARTIFACTS,
        )
        all_metrics[name]["test_2024_2025"] = evaluate_model(
            model, X_test, y_test,
            label=f"{name} — test (2024–2025)",
            save_dir=ARTIFACTS,
        )

    # Side-by-side comparison
    print_comparison(all_metrics)

    # Persist full results (calibration included) to JSON
    summary_path = ARTIFACTS / "metrics_summary.json"
    summary_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"\n  Full results saved -> {summary_path.name}")
