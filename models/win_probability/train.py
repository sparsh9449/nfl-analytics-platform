"""
Win Probability — model training.

Trains two models in sequence:
  1. Logistic regression baseline (sklearn)
  2. XGBoost (added in step 3)

Metrics on validation (2023) are printed after each model.
Final test-set evaluation lives in evaluate.py and is intentionally kept
separate so the test set is only touched once.

Usage (from project root):
    python models/win_probability/train.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.win_probability.data_prep import load_and_split, FEATURES

ARTIFACTS = ROOT / "models" / "win_probability" / "artifacts"
ARTIFACTS.mkdir(exist_ok=True)

# "down" is categorical (1, 2, 3, 4); the other features are numeric.
# The model must not assume that 4th-down is linearly "more" than 3rd-down
# in a single-coefficient sense — hence one-hot rather than passthrough.
CATEGORICAL_FEATURES = ["down"]
NUMERIC_FEATURES     = [f for f in FEATURES if f not in CATEGORICAL_FEATURES]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def drop_ties(X, y):
    """
    Remove the ~0.5% of plays that ended in a tie (posteam_win == 0.5).
    Sklearn classifiers expect binary {0, 1} labels, not 0.5.
    """
    keep = y != 0.5
    return X[keep].reset_index(drop=True), y[keep].astype(int).reset_index(drop=True)


def val_metrics(model, X_val, y_val, name: str) -> dict:
    """
    Compute ROC-AUC and Brier score on a held-out split.
    Returns a dict so the caller can accumulate results for the summary.
    """
    prob  = model.predict_proba(X_val)[:, 1]
    auc   = roc_auc_score(y_val, prob)
    brier = brier_score_loss(y_val, prob)
    print(f"  {name:<35}  ROC-AUC={auc:.4f}  Brier={brier:.4f}")
    return {"roc_auc": round(auc, 4), "brier": round(brier, 4)}


# ---------------------------------------------------------------------------
# Model 1 — Logistic Regression baseline
# ---------------------------------------------------------------------------

def train_logistic_regression(X_train, y_train) -> Pipeline:
    """
    sklearn Pipeline:
      numeric  → SimpleImputer(median) → StandardScaler
      "down"   → OneHotEncoder(drop first category)
    Imputation covers the ~2 400 week-1 plays per season where rolling
    stats are null (no prior games in that season to average).
    """
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("impute", SimpleImputer(strategy="median")),
                    ("scale",  StandardScaler()),
                ]),
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                OneHotEncoder(drop="first", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
        ]
    )
    pipe = Pipeline([
        ("prep", preprocessor),
        ("clf",  LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs")),
    ])
    pipe.fit(X_train, y_train)
    return pipe


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Win Probability — Training")
    print("=" * 60)

    splits = load_and_split()

    # Drop the tiny fraction of tie games for binary classification
    X_train, y_train = drop_ties(splits.X_train, splits.y_train)
    X_val,   y_val   = drop_ties(splits.X_val,   splits.y_val)

    n_ties = len(splits.y_train) - len(y_train)
    print(f"\nTies removed from train: {n_ties:,}  "
          f"({n_ties / len(splits.y_train):.2%} of train plays)")
    print(f"Train: {len(X_train):,} plays  —  {y_train.mean():.1%} wins")
    print(f"Val  : {len(X_val):,} plays  —  {y_val.mean():.1%} wins")

    all_metrics = {}

    # -------------------------------------------------------------------------
    # 1. Logistic Regression
    # -------------------------------------------------------------------------
    print("\n[1/1] Logistic Regression baseline")
    lr = train_logistic_regression(X_train, y_train)
    all_metrics["logistic_regression"] = {
        "val_2023": val_metrics(lr, X_val, y_val, "Logistic Regression  val (2023)"),
    }

    joblib.dump(lr, ARTIFACTS / "lr_model.pkl")
    print(f"  saved -> artifacts/lr_model.pkl")

    # Persist metrics so results survive without re-running training
    summary_path = ARTIFACTS / "metrics_summary.json"
    summary_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"\nMetrics saved -> artifacts/metrics_summary.json")
    print(json.dumps(all_metrics, indent=2))
