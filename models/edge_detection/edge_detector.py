"""
Pre-game edge detection: compare XGBoost win-probability to the market-implied
win probability at the first play of every game.

Methodology
-----------
1. Slice to the first scrimmage play of each game (highest game_seconds_remaining
   among plays with a valid down, posteam_win, and vegas_wp).  At that moment the
   score is 0-0 and vegas_wp captures the pre-game market consensus.
2. Convert vegas_wp (home-team perspective in nflfastR) to the possessing team's
   perspective so both the model and market are on the same footing.
3. Run the trained XGBoost model to get model_wp.
4. edge = model_wp - market_wp.
5. Bucket edges into five bins and report actual win rate vs market-implied rate
   for each bucket to validate whether the model's edge is real.

Output
------
  artifacts/edge_summary.json   — per-bucket stats + overall summary
  artifacts/edge_by_bucket.png  — bar chart of actual vs market win rate by bucket
"""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.win_probability.data_prep import FEATURES

GOLD_PATH = ROOT / "data" / "gold" / "wp_features.parquet"
XGB_PATH  = ROOT / "models" / "win_probability" / "artifacts" / "xgb_model.pkl"
ARTIFACTS = ROOT / "models" / "edge_detection" / "artifacts"

# Five symmetric buckets around zero; "near-fair" is the [-5%, +5%] center band
EDGE_BINS   = [-np.inf, -0.10, -0.05, 0.05, 0.10, np.inf]
EDGE_LABELS = ["< -10%", "-10% to -5%", "-5% to +5%", "+5% to +10%", "> +10%"]

TEST_SEASONS = [2024, 2025]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _first_play_per_game(df: pd.DataFrame) -> pd.DataFrame:
    """First valid scrimmage play of each game by highest game_seconds_remaining."""
    valid = df[
        df["down"].notna() &
        df["posteam_win"].notna() &
        df["vegas_wp"].notna()
    ].copy()
    return (
        valid
        .sort_values("game_seconds_remaining", ascending=False)
        .groupby("game_id", sort=False)
        .first()
        .reset_index()
    )


def _market_wp_posteam(df: pd.DataFrame) -> np.ndarray:
    """
    Convert vegas_wp (home-team WP) to the possessing team's perspective.
    Possession team == home team  →  market_wp = vegas_wp
    Possession team == away team  →  market_wp = 1 - vegas_wp
    """
    is_home = (df["posteam"] == df["home_team"]).to_numpy()
    return np.where(is_home, df["vegas_wp"].to_numpy(), 1.0 - df["vegas_wp"].to_numpy())


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------

def run_edge_detection(seasons: list[int] = TEST_SEASONS) -> dict:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print(f"[edge detection] seasons: {seasons}")
    df = pd.read_parquet(GOLD_PATH)
    df = df[df["season"].isin(seasons)]
    print(f"  {len(df):,} plays loaded")

    first_plays = _first_play_per_game(df)
    print(f"  {len(first_plays):,} games with valid first play")

    xgb_model = joblib.load(XGB_PATH)

    X = first_plays[FEATURES]
    fp = first_plays.copy()
    fp["model_wp"]  = xgb_model.predict_proba(X)[:, 1]
    fp["market_wp"] = _market_wp_posteam(fp)
    fp["edge"]      = fp["model_wp"] - fp["market_wp"]
    fp["actual_win"] = (fp["posteam_win"] == 1.0).astype(float)

    fp["edge_bucket"] = pd.cut(
        fp["edge"],
        bins=EDGE_BINS,
        labels=EDGE_LABELS,
        right=True,
    )

    summary = _bucket_stats(fp)
    result  = _build_output(seasons, fp, summary)

    json_path = ARTIFACTS / "edge_summary.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"  saved -> {json_path.name}")

    _plot(summary)
    _print_table(summary)

    return result


def _bucket_stats(fp: pd.DataFrame) -> list[dict]:
    rows = []
    for label in EDGE_LABELS:
        bucket = fp[fp["edge_bucket"] == label]
        n = len(bucket)
        rows.append({
            "bucket":          label,
            "count":           n,
            "market_wp_mean":  round(float(bucket["market_wp"].mean()), 4) if n else None,
            "model_wp_mean":   round(float(bucket["model_wp"].mean()),  4) if n else None,
            "actual_win_rate": round(float(bucket["actual_win"].mean()), 4) if n else None,
            "edge_mean":       round(float(bucket["edge"].mean()), 4) if n else None,
        })
    return rows


def _build_output(seasons: list[int], fp: pd.DataFrame, summary: list[dict]) -> dict:
    # Overall stats: does the model see net positive edge on any side?
    pos_edge = fp[fp["edge"] > 0.05]
    neg_edge = fp[fp["edge"] < -0.05]
    return {
        "seasons":      seasons,
        "n_games":      len(fp),
        "model":        "xgboost",
        "overall": {
            "mean_edge":       round(float(fp["edge"].mean()), 4),
            "std_edge":        round(float(fp["edge"].std()),  4),
            "pct_pos_edge_5":  round(float((fp["edge"] > 0.05).mean()), 4),
            "pct_neg_edge_5":  round(float((fp["edge"] < -0.05).mean()), 4),
            "pos_edge_actual_win_rate": round(float(pos_edge["actual_win"].mean()), 4) if len(pos_edge) else None,
            "neg_edge_actual_win_rate": round(float(neg_edge["actual_win"].mean()), 4) if len(neg_edge) else None,
        },
        "buckets": summary,
    }


def _plot(summary: list[dict]) -> None:
    active = [r for r in summary if r["count"] > 0]
    labels  = [r["bucket"] for r in active]
    market  = [r["market_wp_mean"] for r in active]
    actual  = [r["actual_win_rate"] for r in active]
    counts  = [r["count"] for r in active]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_m = ax.bar(x - width / 2, market, width, label="Market implied WP", color="#4C72B0", alpha=0.85)
    bars_a = ax.bar(x + width / 2, actual, width, label="Actual win rate",   color="#DD8452", alpha=0.85)

    ax.set_xlabel("Edge bucket  (model WP − market WP)")
    ax.set_ylabel("Win rate")
    ax.set_title("Pre-game Edge Backtest — XGBoost vs Market (2024–2025 test seasons)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_ylim(0, 1.05)
    ax.legend()

    for bar, n in zip(bars_a, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.015,
            f"n={n}",
            ha="center", va="bottom", fontsize=8, color="dimgray",
        )

    fig.tight_layout()
    plot_path = ARTIFACTS / "edge_by_bucket.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {plot_path.name}")


def _print_table(summary: list[dict]) -> None:
    print(f"\n{'Bucket':<18} {'n':>5} {'Market WP':>10} {'Model WP':>9} {'Actual WR':>10} {'Edge':>7}")
    print("-" * 65)
    for r in summary:
        if r["count"] == 0:
            print(f"{r['bucket']:<18} {'0':>5}")
            continue
        print(
            f"{r['bucket']:<18} {r['count']:>5} "
            f"{r['market_wp_mean']:>10.1%} "
            f"{r['model_wp_mean']:>9.1%} "
            f"{r['actual_win_rate']:>10.1%} "
            f"{r['edge_mean']:>+7.1%}"
        )


if __name__ == "__main__":
    result = run_edge_detection(TEST_SEASONS)
    print(f"\nOverall edge mean: {result['overall']['mean_edge']:+.4f}")
    print(f">+5% edge games:   {result['overall']['pct_pos_edge_5']:.1%}  "
          f"actual win rate: {result['overall']['pos_edge_actual_win_rate']:.1%}")
    print(f"<-5% edge games:   {result['overall']['pct_neg_edge_5']:.1%}  "
          f"actual win rate: {result['overall']['neg_edge_actual_win_rate']:.1%}")
