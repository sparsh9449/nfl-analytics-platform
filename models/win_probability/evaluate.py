"""
Reusable evaluation functions for the Win Probability model.

Three metrics
─────────────
ROC-AUC       Discriminative power.  Higher = better.  0.5 = coin flip.
Brier score   Mean squared error between predicted prob and outcome.
              Lower = better.  0.25 = always predicting 50%.
Calibration   Predicted probability bucket vs actual win rate in that bucket.
              A well-calibrated model has predicted ≈ actual across all bins.

Usage
─────
    from models.win_probability.evaluate import evaluate_model

    metrics = evaluate_model(model, X_val, y_val, label="LR val (2023)")
    # or with a saved plot:
    metrics = evaluate_model(model, X_val, y_val, label="LR val (2023)",
                             save_dir=Path("models/win_probability/artifacts"))
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe in scripts and notebooks
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss, roc_auc_score


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def compute_metrics(prob: np.ndarray, y_true: np.ndarray) -> dict:
    """Return ROC-AUC and Brier score as a plain dict."""
    return {
        "roc_auc": round(float(roc_auc_score(y_true, prob)), 4),
        "brier":   round(float(brier_score_loss(y_true, prob)), 4),
    }


def calibration_table(prob: np.ndarray, y_true: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """
    Bin predicted probabilities into n_bins equal-width buckets and compare
    the mean predicted probability to the actual win rate in each bucket.

    Rows flagged with '!' have a gap > 0.05 — those are the bins where the
    model is meaningfully over- or under-confident.

    Returns a DataFrame so the caller can include it in the metrics JSON.
    """
    frac_pos, mean_pred = calibration_curve(
        y_true, prob, n_bins=n_bins, strategy="uniform"
    )
    df = pd.DataFrame({
        "predicted": mean_pred.round(3),
        "actual":    frac_pos.round(3),
        "diff":      (frac_pos - mean_pred).round(3),
    })

    print(f"  {'predicted':>10} {'actual':>10} {'diff':>8}")
    print(f"  {'-'*32}")
    for _, row in df.iterrows():
        flag = "  !" if abs(row["diff"]) > 0.05 else ""
        print(f"  {row['predicted']:>10.3f} {row['actual']:>10.3f} {row['diff']:>+8.3f}{flag}")

    return df


def calibration_plot(
    prob: np.ndarray,
    y_true: np.ndarray,
    label: str,
    n_bins: int = 10,
    save_path: Path | None = None,
) -> None:
    """
    Reliability diagram: predicted probability on x-axis, actual win rate on
    y-axis, diagonal = perfect calibration.  Saved to save_path if provided.
    """
    frac_pos, mean_pred = calibration_curve(
        y_true, prob, n_bins=n_bins, strategy="uniform"
    )

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="Perfect calibration")
    ax.plot(mean_pred, frac_pos, marker="o", label=label)
    ax.set_xlabel("Mean predicted win probability")
    ax.set_ylabel("Actual win rate")
    ax.set_title(f"Calibration curve — {label}")
    ax.legend()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=120)
        print(f"  plot saved -> {save_path.name}")

    plt.close(fig)


# ---------------------------------------------------------------------------
# Main entry point — runs all three metrics together
# ---------------------------------------------------------------------------

def evaluate_model(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    label: str,
    n_bins: int = 10,
    save_dir: Path | None = None,
) -> dict:
    """
    Full evaluation pass: ROC-AUC, Brier score, calibration table, and
    (optionally) a saved calibration plot.

    Returns a metrics dict suitable for inclusion in metrics_summary.json.
    """
    prob  = model.predict_proba(X)[:, 1]
    y_arr = np.asarray(y)

    metrics = compute_metrics(prob, y_arr)

    print(f"\n{'─' * 52}")
    print(f"  {label}")
    print(f"{'─' * 52}")
    print(f"  ROC-AUC : {metrics['roc_auc']:.4f}")
    print(f"  Brier   : {metrics['brier']:.4f}")
    print(f"\n  Calibration ({n_bins} uniform bins)  [! = |gap| > 0.05]")

    cal_df = calibration_table(prob, y_arr, n_bins=n_bins)
    metrics["calibration"] = cal_df.to_dict(orient="records")

    if save_dir is not None:
        slug = label.lower().replace(" ", "_").replace("(", "").replace(")", "")
        plot_path = save_dir / f"calibration_{slug}.png"
        calibration_plot(prob, y_arr, label, n_bins=n_bins, save_path=plot_path)

    return metrics
