"""
Train and evaluate the pre-game prediction model.

This is a game-level model (one row per game, not per play).
Unlike the in-game WP model, it uses only pre-game knowable features:
market implied WP from the spread, rolling team stats, rest, week.

The model's output is P(home team wins) before kickoff.

The in-game model (xgb_calibrated.pkl) then takes this as its anchor:
at kickoff it starts at pre_game_wp and drifts play-by-play.

Output:
  models/pregame/artifacts/xgb_pregame_calibrated.pkl
  models/pregame/artifacts/pregame_metrics.json
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from xgboost import XGBClassifier

ROOT         = Path(__file__).resolve().parents[2]
GOLD_PATH    = ROOT / "data" / "gold" / "pregame_features.parquet"
ARTIFACT_DIR = ROOT / "models" / "pregame" / "artifacts"

TRAIN_SEASONS = list(range(2016, 2023))
VAL_SEASONS   = [2023]
TEST_SEASONS  = [2024, 2025]

FEATURES = [
    "market_wp",              # Vegas implied P(home wins) — strongest signal
    # spread_line omitted: it's a monotone transform of market_wp (collinear)
    "total_line",             # over/under
    "week",                   # week of season
    "home_roll4_pts_scored",
    "home_roll4_pts_allowed",
    "home_roll4_epa_scored",
    "home_roll4_epa_allowed",
    "home_roll4_win_pct",
    "home_rest_days",
    "away_roll4_pts_scored",
    "away_roll4_pts_allowed",
    "away_roll4_epa_scored",
    "away_roll4_epa_allowed",
    "away_roll4_win_pct",
    "away_rest_days",
]

TARGET = "home_win"


def _split(df: pd.DataFrame):
    train = df[df["season"].isin(TRAIN_SEASONS)]
    val   = df[df["season"].isin(VAL_SEASONS)]
    test  = df[df["season"].isin(TEST_SEASONS)]
    return train, val, test


def _metrics(y_true, y_prob, name: str) -> dict:
    y_arr = np.array(y_true, dtype=float)
    p_arr = np.array(y_prob, dtype=float)

    auc      = roc_auc_score(y_arr, p_arr)
    brier    = brier_score_loss(y_arr, p_arr)
    acc      = float(((p_arr > 0.5) == (y_arr == 1)).mean())
    # Market baseline accuracy
    mkt_acc  = float(((p_arr > 0.5) == (y_arr == 1)).mean())  # placeholder; computed separately

    print(f"  [{name}]  AUC={auc:.4f}  Brier={brier:.4f}  Acc={acc:.3f}")
    return {"split": name, "auc": round(auc, 4), "brier": round(brier, 4),
            "accuracy": round(acc, 4)}


def train():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    print("[pregame model] loading features...")
    df = pd.read_parquet(GOLD_PATH)
    print(f"  {len(df):,} games  ({df['season'].min()}–{df['season'].max()})")

    train_df, val_df, test_df = _split(df)
    print(f"  train={len(train_df):,}  val={len(val_df):,}  test={len(test_df):,}")

    X_train = train_df[FEATURES]
    y_train = train_df[TARGET]
    X_val   = val_df[FEATURES]
    y_val   = val_df[TARGET]
    X_test  = test_df[FEATURES]
    y_test  = test_df[TARGET]

    # --- Train XGBoost on train set ---
    print("\n[pregame model] training XGBoost...")
    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,   # prevents overfitting on ~2500 games
        eval_metric="logloss",
        early_stopping_rounds=30,
        random_state=42,
        verbosity=0,
    )
    xgb.fit(X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False)

    print(f"  best iteration: {xgb.best_iteration}")

    # --- Platt scaling (sigmoid) calibration on val set ---
    # Isotonic regression overfits badly with only ~285 val games,
    # producing a step-function that collapses wide probability ranges.
    # Platt scaling fits just 2 parameters (slope + bias) and stays monotone.
    print("[pregame model] calibrating on val set (Platt scaling)...")
    raw_val_proba = xgb.predict_proba(X_val)[:, 1]
    platt = LogisticRegression(C=1.0, max_iter=1000)
    platt.fit(raw_val_proba.reshape(-1, 1), y_val.values)
    iso = platt  # kept as 'iso' so save/load code is unchanged

    # --- Evaluate ---
    print("\n[pregame model] evaluation:")
    results = []

    def _apply_platt(raw_p: np.ndarray) -> np.ndarray:
        return platt.predict_proba(raw_p.reshape(-1, 1))[:, 1]

    for split_name, X, y in [
        ("val",  X_val,  y_val),
        ("test", X_test, y_test),
    ]:
        raw_p  = xgb.predict_proba(X)[:, 1]
        cal_p  = _apply_platt(raw_p)
        mkt_p  = X["market_wp"].values

        raw_m = _metrics(y, raw_p,  f"{split_name}/raw")
        cal_m = _metrics(y, cal_p,  f"{split_name}/calibrated")
        mkt_m = _metrics(y, mkt_p,  f"{split_name}/market-baseline")

        # ATS accuracy: did the team our model likes more actually win?
        model_pick_correct = float(((cal_p > 0.5) == (y.values == 1)).mean())
        mkt_pick_correct   = float(((mkt_p > 0.5) == (y.values == 1)).mean())
        print(f"  [{split_name}] model_pick_acc={model_pick_correct:.3f}  market_pick_acc={mkt_pick_correct:.3f}")

        results.append({
            **cal_m,
            "model_pick_acc": round(model_pick_correct, 4),
            "market_pick_acc": round(mkt_pick_correct, 4),
        })

    # --- Save ---
    artifact = {"xgb": xgb, "iso": iso, "features": FEATURES}
    out_path = ARTIFACT_DIR / "xgb_pregame_calibrated.pkl"
    joblib.dump(artifact, out_path)
    print(f"\n[pregame model] saved -> {out_path.name}")

    metrics_path = ARTIFACT_DIR / "pregame_metrics.json"
    metrics_path.write_text(json.dumps({"splits": results}, indent=2))
    print(f"[pregame model] metrics -> {metrics_path.name}")

    return artifact


if __name__ == "__main__":
    train()
