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
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.win_probability.data_prep import load_and_split, FEATURES
from models.win_probability.evaluate import compute_metrics

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
    One-line ROC-AUC + Brier score for quick feedback during the training loop.
    Full evaluation (+ calibration table and plot) lives in evaluate.py.
    """
    metrics = compute_metrics(model.predict_proba(X_val)[:, 1], y_val.to_numpy())
    print(f"  {name:<35}  ROC-AUC={metrics['roc_auc']:.4f}  Brier={metrics['brier']:.4f}")
    return metrics


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
# Model 2 — XGBoost
# ---------------------------------------------------------------------------

def train_xgboost(X_train, y_train) -> xgb.XGBClassifier:
    """
    Gradient-boosted trees via XGBoost.

    No preprocessing pipeline is needed:
      - Trees are scale-invariant, so StandardScaler adds no value.
      - XGBoost handles missing values natively by learning the best
        direction to send null-feature rows at each split — so the week-1
        null rolling stats need no imputation.
      - 'down' is passed as numeric 1-4; trees find the right thresholds
        without one-hot encoding.

    Early stopping: we hold out the last 10% of training plays (still within
    train seasons, no val/test leakage) as an internal eval set.  XGBoost
    stops adding trees once logloss on that set hasn't improved for 30 rounds,
    preventing the model from memorising the training data.
    """
    # Chronological 90/10 split within the training data for early stopping.
    # iloc preserves time order because data_prep returns plays sorted by season.
    split_at = int(len(X_train) * 0.9)
    X_tr, X_es = X_train.iloc[:split_at], X_train.iloc[split_at:]
    y_tr, y_es = y_train.iloc[:split_at], y_train.iloc[split_at:]

    model = xgb.XGBClassifier(
        n_estimators=1000,           # upper bound; early stopping cuts this short
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,         # require ≥10 plays in each leaf
        tree_method="hist",
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        verbosity=0,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_es, y_es)], verbose=False)
    print(f"  best iteration: {model.best_iteration}  "
          f"(stopped before {model.n_estimators})")
    return model


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
    print("\n[1/2] Logistic Regression baseline")
    lr = train_logistic_regression(X_train, y_train)
    all_metrics["logistic_regression"] = {
        "val_2023": val_metrics(lr, X_val, y_val, "Logistic Regression  val (2023)"),
    }
    joblib.dump(lr, ARTIFACTS / "lr_model.pkl")
    print(f"  saved -> artifacts/lr_model.pkl")

    # -------------------------------------------------------------------------
    # 2. XGBoost
    # -------------------------------------------------------------------------
    print("\n[2/2] XGBoost")
    xgb_model = train_xgboost(X_train, y_train)
    all_metrics["xgboost"] = {
        "val_2023": val_metrics(xgb_model, X_val, y_val, "XGBoost              val (2023)"),
    }
    joblib.dump(xgb_model, ARTIFACTS / "xgb_model.pkl")
    print(f"  saved -> artifacts/xgb_model.pkl")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Validation summary (2023)")
    print("=" * 60)
    print(f"{'Model':<25} {'ROC-AUC':>9} {'Brier':>8}")
    print("-" * 45)
    for model_name, results in all_metrics.items():
        m = results["val_2023"]
        print(f"{model_name:<25} {m['roc_auc']:>9.4f} {m['brier']:>8.4f}")

    # Persist metrics so results survive without re-running training
    summary_path = ARTIFACTS / "metrics_summary.json"
    summary_path.write_text(json.dumps(all_metrics, indent=2))
    print(f"\nMetrics saved -> artifacts/metrics_summary.json")
