"""
Post-hoc calibration of the trained XGBoost win-probability model.

The raw XGBoost model is over-confident in the low-probability tail
(predicted ~6%, actual ~2.5%; predicted ~15%, actual ~11%). Isotonic
regression learns a monotone mapping from raw scores to calibrated
probabilities without touching the underlying model weights.

Method
------
CalibratedClassifierCV(cv='prefit', method='isotonic'):
  - cv='prefit' means the XGBoost model is already trained and frozen.
  - Isotonic regression is fit on the 2023 val set — the same set used
    for all other tuning decisions, so no additional data leakage.

Output
------
  artifacts/xgb_calibrated.pkl                           — wrapped model
  artifacts/calibration_xgb_calibrated_—_test.png       — reliability diagram
  artifacts/metrics_summary.json                         — updated with cal metrics
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from models.win_probability.calibrated_model import IsotonicCalibratedXGB

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.win_probability.data_prep import load_and_split, FEATURES
from models.win_probability.evaluate import evaluate_model
from models.win_probability.train import drop_ties

ARTIFACTS = ROOT / "models" / "win_probability" / "artifacts"



def _load_splits():
    splits = load_and_split()
    X_val,  y_val  = drop_ties(splits.X_val,  splits.y_val)
    X_test, y_test = drop_ties(splits.X_test, splits.y_test)
    return X_val, y_val, X_test, y_test


def calibrate_xgboost(X_val, y_val) -> IsotonicCalibratedXGB:
    """Fit isotonic regression on val-set raw scores and return a wrapped model."""
    raw_model = joblib.load(ARTIFACTS / "xgb_model.pkl")
    raw_prob  = raw_model.predict_proba(X_val)[:, 1]
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(raw_prob, y_val.to_numpy())
    return IsotonicCalibratedXGB(raw_model, iso)


def _comparison_plot(
    raw_prob: np.ndarray,
    cal_prob: np.ndarray,
    y_true: np.ndarray,
    save_path: Path,
    n_bins: int = 10,
) -> None:
    """Overlay raw and calibrated reliability curves on one plot."""
    frac_raw,  mean_raw  = calibration_curve(y_true, raw_prob, n_bins=n_bins, strategy="uniform")
    frac_cal,  mean_cal  = calibration_curve(y_true, cal_prob, n_bins=n_bins, strategy="uniform")

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Perfect calibration")
    ax.plot(mean_raw, frac_raw, "o--", color="#4C72B0", alpha=0.7, label="XGBoost (raw)")
    ax.plot(mean_cal, frac_cal, "o-",  color="#DD8452",             label="XGBoost (calibrated)")
    ax.set_xlabel("Mean predicted win probability")
    ax.set_ylabel("Actual win rate")
    ax.set_title("Calibration — raw vs isotonic  (test 2024–2025)")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  comparison plot saved -> {save_path.name}")


if __name__ == "__main__":
    print("=" * 60)
    print("XGBoost — Isotonic Calibration")
    print("=" * 60)

    X_val, y_val, X_test, y_test = _load_splits()
    y_test_arr = y_test.to_numpy()

    # -------------------------------------------------------------------------
    # 1. Fit calibrated model on val set
    # -------------------------------------------------------------------------
    print("\n[1/3] Fitting isotonic calibration on val (2023)...")
    cal_model = calibrate_xgboost(X_val, y_val)
    joblib.dump(cal_model, ARTIFACTS / "xgb_calibrated.pkl")
    print("  saved -> artifacts/xgb_calibrated.pkl")

    # -------------------------------------------------------------------------
    # 2. Evaluate raw vs calibrated on test set
    # -------------------------------------------------------------------------
    print("\n[2/3] Evaluating on test set (2024–2025)...")

    raw_model  = joblib.load(ARTIFACTS / "xgb_model.pkl")
    raw_prob   = raw_model.predict_proba(X_test)[:, 1]
    cal_prob   = cal_model.predict_proba(X_test)[:, 1]

    print("\n--- Raw XGBoost ---")
    raw_metrics = evaluate_model(
        raw_model, X_test, y_test,
        label="XGBoost raw — test 2024–2025",
        save_dir=ARTIFACTS,
    )

    print("\n--- Calibrated XGBoost ---")
    cal_metrics = evaluate_model(
        cal_model, X_test, y_test,
        label="XGBoost calibrated — test 2024–2025",
        save_dir=ARTIFACTS,
    )

    # -------------------------------------------------------------------------
    # 3. Comparison plot + metrics update
    # -------------------------------------------------------------------------
    print("\n[3/3] Saving comparison plot and updating metrics...")

    _comparison_plot(
        raw_prob, cal_prob, y_test_arr,
        save_path=ARTIFACTS / "calibration_xgb_raw_vs_calibrated_—_test.png",
    )

    # Merge into existing metrics_summary.json
    summary_path = ARTIFACTS / "metrics_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["xgboost"]["test_2024_2025_calibrated"] = cal_metrics
    summary_path.write_text(json.dumps(summary, indent=2))
    print("  metrics_summary.json updated")

    # -------------------------------------------------------------------------
    # Summary table
    # -------------------------------------------------------------------------
    print(f"\n{'Model':<35} {'ROC-AUC':>9} {'Brier':>8}")
    print("-" * 55)
    print(f"{'XGBoost raw':<35} {raw_metrics['roc_auc']:>9.4f} {raw_metrics['brier']:>8.4f}")
    print(f"{'XGBoost calibrated':<35} {cal_metrics['roc_auc']:>9.4f} {cal_metrics['brier']:>8.4f}")
