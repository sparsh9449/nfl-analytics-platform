"""
Week-of-season breakdown for the pre-game edge signal.

Question: is the >+10% edge concentrated in weeks 1–4 (when 4-game rolling
stats are thin or null) or does it persist into weeks 5+?

If the signal collapses after week 4, it means the model is finding edge
only because rolling stats are sparse early in the season — not because it
has a genuine read on team quality relative to the market. If it persists,
the recent-form signal is adding real value the market misses.

Groups
------
  Early  : weeks  1–4   (0–3 prior games; rolling stats sparse/null)
  Mid    : weeks  5–10  (full 4-game window, first half of season)
  Late   : weeks 11–18  (full window, second half + playoff lead-in)

Output
------
  artifacts/week_breakdown.json         — full stats table
  artifacts/week_breakdown_edge.png     — actual vs market win rate by week group,
                                          for the >+10% edge bucket
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
from models.edge_detection.edge_detector import (
    _first_play_per_game,
    _market_wp_posteam,
    GOLD_PATH,
    XGB_PATH,
    ARTIFACTS,
    EDGE_BINS,
    EDGE_LABELS,
    TEST_SEASONS,
)
from models.win_probability.calibrated_model import IsotonicCalibratedXGB  # noqa: F401

WEEK_GROUPS = [
    ("Early (wk 1–4)",  (1,  4)),
    ("Mid   (wk 5–10)", (5,  10)),
    ("Late  (wk 11+)",  (11, 99)),
]


def _build_first_plays(seasons: list[int]) -> pd.DataFrame:
    df = pd.read_parquet(GOLD_PATH)
    df = df[df["season"].isin(seasons)]
    fp = _first_play_per_game(df)

    model = joblib.load(XGB_PATH)
    fp = fp.copy()
    fp["model_wp"]   = model.predict_proba(fp[FEATURES])[:, 1]
    fp["market_wp"]  = _market_wp_posteam(fp)
    fp["edge"]       = fp["model_wp"] - fp["market_wp"]
    fp["actual_win"] = (fp["posteam_win"] == 1.0).astype(float)
    fp["roll_null"]  = fp[["posteam_roll4_pts_scored", "posteam_roll4_pts_allowed",
                            "defteam_roll4_pts_scored",  "defteam_roll4_pts_allowed"]].isnull().any(axis=1)
    fp["edge_bucket"] = pd.cut(fp["edge"], bins=EDGE_BINS, labels=EDGE_LABELS, right=True)
    return fp


def _week_group(week: int) -> str:
    for label, (lo, hi) in WEEK_GROUPS:
        if lo <= week <= hi:
            return label
    return "Unknown"


def _stats(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"n": 0, "market_wp": None, "model_wp": None, "actual_wr": None,
                "edge_mean": None, "pct_null_roll": None}
    return {
        "n":             len(df),
        "market_wp":     round(float(df["market_wp"].mean()), 4),
        "model_wp":      round(float(df["model_wp"].mean()),  4),
        "actual_wr":     round(float(df["actual_win"].mean()), 4),
        "edge_mean":     round(float(df["edge"].mean()), 4),
        "pct_null_roll": round(float(df["roll_null"].mean()), 4),
    }


def run_week_breakdown(seasons: list[int] = TEST_SEASONS) -> dict:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print(f"[week breakdown] seasons: {seasons}")
    fp = _build_first_plays(seasons)
    fp["week_group"] = fp["week"].apply(_week_group)

    # -------------------------------------------------------------------------
    # 1. Full breakdown: every week group × every edge bucket
    # -------------------------------------------------------------------------
    rows = []
    for grp_label, _ in WEEK_GROUPS:
        grp = fp[fp["week_group"] == grp_label]
        for bucket in EDGE_LABELS:
            sub = grp[grp["edge_bucket"] == bucket]
            row = {"week_group": grp_label, "bucket": bucket}
            row.update(_stats(sub))
            rows.append(row)

    # -------------------------------------------------------------------------
    # 2. >+10% edge slice — the signal we care about most
    # -------------------------------------------------------------------------
    pos_edge = fp[fp["edge"] > 0.10]
    pos_by_group = []
    for grp_label, _ in WEEK_GROUPS:
        sub = pos_edge[pos_edge["week_group"] == grp_label]
        row = {"week_group": grp_label}
        row.update(_stats(sub))
        pos_by_group.append(row)

    result = {
        "seasons":            seasons,
        "n_games":            len(fp),
        "full_breakdown":     rows,
        "pos_edge_gt10_by_week_group": pos_by_group,
    }

    json_path = ARTIFACTS / "week_breakdown.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"  saved -> {json_path.name}")

    _plot(pos_by_group)
    _print_tables(rows, pos_by_group)

    return result


def _plot(pos_by_group: list[dict]) -> None:
    active = [r for r in pos_by_group if r["n"] > 0]
    if not active:
        return

    labels  = [r["week_group"].strip() for r in active]
    market  = [r["market_wp"]  for r in active]
    actual  = [r["actual_wr"]  for r in active]
    counts  = [r["n"]          for r in active]

    x     = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars_m = ax.bar(x - width / 2, market, width, label="Market implied WP", color="#4C72B0", alpha=0.85)
    bars_a = ax.bar(x + width / 2, actual, width, label="Actual win rate",   color="#DD8452", alpha=0.85)

    ax.set_xlabel("Week group")
    ax.set_ylabel("Win rate")
    ax.set_title(">+10% edge games: actual win rate vs market by week group\n(2024–2025 test seasons, calibrated XGBoost)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_ylim(0, 0.85)
    ax.legend()

    for bar, n in zip(bars_a, counts):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.012,
            f"n={n}",
            ha="center", va="bottom", fontsize=9, color="dimgray",
        )

    fig.tight_layout()
    plot_path = ARTIFACTS / "week_breakdown_edge.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {plot_path.name}")


def _print_tables(rows: list[dict], pos_by_group: list[dict]) -> None:
    print(f"\n--- >+10% edge games by week group ---")
    print(f"{'Week group':<20} {'n':>5} {'Market WP':>10} {'Model WP':>9} {'Actual WR':>10} {'Null roll%':>11}")
    print("-" * 72)
    for r in pos_by_group:
        if r["n"] == 0:
            print(f"{r['week_group']:<20} {'0':>5}")
            continue
        print(
            f"{r['week_group']:<20} {r['n']:>5} "
            f"{r['market_wp']:>10.1%} "
            f"{r['model_wp']:>9.1%} "
            f"{r['actual_wr']:>10.1%} "
            f"{r['pct_null_roll']:>10.1%}"
        )

    print(f"\n--- All buckets by week group ---")
    hdr = f"{'Week group':<20} {'Bucket':<18} {'n':>5} {'Market WP':>10} {'Actual WR':>10} {'Null roll%':>11}"
    print(hdr)
    print("-" * 80)
    for r in rows:
        if r["n"] == 0:
            continue
        print(
            f"{r['week_group']:<20} {r['bucket']:<18} {r['n']:>5} "
            f"{r['market_wp']:>10.1%} "
            f"{r['actual_wr']:>10.1%} "
            f"{r['pct_null_roll']:>10.1%}"
        )


if __name__ == "__main__":
    run_week_breakdown(TEST_SEASONS)
