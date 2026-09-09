"""
ROI backtest for the pre-game edge strategy.

Assumes flat betting at standard -110 American odds (bet $110 to win $100).
Break-even win rate: 52.38%.

Evaluates the >+10% edge bucket across week groups and seasons, producing
an honest picture of whether the signal is profitable after vig.

Output
------
  artifacts/roi_summary.json    -- ROI table by season x week group
  artifacts/roi_by_season.png   -- cumulative P&L chart across seasons
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

from models.win_probability.calibrate import IsotonicCalibratedXGB  # noqa: F401
from models.edge_detection.week_breakdown import _build_first_plays, WEEK_GROUPS
from models.edge_detection.edge_detector import GOLD_PATH, XGB_PATH, ARTIFACTS, TEST_SEASONS

ALL_SEASONS       = list(range(2016, 2026))
EDGE_THRESH       = 0.10    # primary threshold for bucket analysis
THRESHOLD_GRID    = [0.10, 0.15, 0.20, 0.25]   # sensitivity sweep
ODDS              = -110    # standard American odds


def _roi(wins: int, total: int) -> float:
    """ROI per dollar wagered at -110 odds. Positive = profitable."""
    if total == 0:
        return float("nan")
    losses = total - wins
    # bet $110 each game; win $100 on wins, lose $110 on losses
    net    = wins * 100 - losses * 110
    staked = total * 110
    return round(net / staked, 4)


def _season_rows(fp: pd.DataFrame) -> list[dict]:
    """Per-season, per-week-group stats for the >+10% edge bucket."""
    pos = fp[fp["edge"] > EDGE_THRESH].copy()
    rows = []
    for season in sorted(fp["season"].unique()):
        for grp_label, (lo, hi) in WEEK_GROUPS:
            sub = pos[(pos["season"] == season) & pos["week"].between(lo, hi)]
            n    = len(sub)
            wins = int(sub["actual_win"].sum()) if n else 0
            rows.append({
                "season":     int(season),
                "week_group": grp_label.strip(),
                "n":          n,
                "wins":       wins,
                "win_rate":   round(wins / n, 4) if n else None,
                "roi":        _roi(wins, n),
            })
    return rows


def _overall_rows(fp: pd.DataFrame) -> list[dict]:
    """Aggregate stats collapsed across all seasons, per week group."""
    pos = fp[fp["edge"] > EDGE_THRESH].copy()
    rows = []
    for grp_label, (lo, hi) in WEEK_GROUPS:
        sub  = pos[pos["week"].between(lo, hi)]
        n    = len(sub)
        wins = int(sub["actual_win"].sum()) if n else 0
        rows.append({
            "week_group": grp_label.strip(),
            "n":          n,
            "wins":       wins,
            "win_rate":   round(wins / n, 4) if n else None,
            "roi":        _roi(wins, n),
        })
    return rows


def _cumulative_pnl(fp: pd.DataFrame) -> pd.Series:
    """
    Flat-bet $110 on every >+10% edge game across all seasons, in
    chronological order. Returns a Series of cumulative net P&L.
    """
    pos = (
        fp[fp["edge"] > EDGE_THRESH]
        .sort_values(["season", "week"])
        .copy()
    )
    pos["pnl"] = pos["actual_win"].apply(lambda w: 100.0 if w == 1 else -110.0)
    return pos["pnl"].cumsum().reset_index(drop=True)


def _plot_pnl(cumulative: pd.Series, n_games: int) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(cumulative.index + 1, cumulative.values, color="#4C72B0", linewidth=1.5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.fill_between(cumulative.index + 1, cumulative.values, 0,
                    where=(cumulative.values >= 0), alpha=0.15, color="#4C72B0")
    ax.fill_between(cumulative.index + 1, cumulative.values, 0,
                    where=(cumulative.values < 0),  alpha=0.15, color="#DD8452")
    ax.set_xlabel(f"Bet number (all {n_games} games, >+10% edge, 2016-2025)")
    ax.set_ylabel("Cumulative P&L ($110 flat bet)")
    ax.set_title("Flat-bet P&L — >+10% edge bucket, all seasons")
    fig.tight_layout()
    plot_path = ARTIFACTS / "roi_by_season.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved -> {plot_path.name}")


def _threshold_sensitivity(fp: pd.DataFrame) -> list[dict]:
    """Win rate and ROI for late-season (wk 11+) across edge thresholds."""
    late = fp[fp["week"].between(11, 99)]
    rows = []
    for thresh in THRESHOLD_GRID:
        sub  = late[late["edge"] > thresh]
        n    = len(sub)
        wins = int(sub["actual_win"].sum()) if n else 0
        rows.append({
            "edge_threshold": thresh,
            "n":              n,
            "wins":           wins,
            "win_rate":       round(wins / n, 4) if n else None,
            "roi":            _roi(wins, n),
        })
    return rows


def _print_threshold_table(rows: list[dict]) -> None:
    print(f"\n--- Late-season (wk 11+) threshold sensitivity ---")
    print(f"{'Threshold':<12} {'n':>5} {'Win rate':>9} {'ROI':>8}  {'Profitable?':>12}")
    print("-" * 52)
    for r in rows:
        if r["n"] == 0:
            continue
        profitable = "YES" if r["roi"] is not None and r["roi"] > 0 else "no"
        roi_str = f"{r['roi']:+.1%}" if r["roi"] is not None else "n/a"
        print(f"> {r['edge_threshold']:.0%}       {r['n']:>5} {r['win_rate']:>9.1%} {roi_str:>8}  {profitable:>12}")
    print(f"\n  Break-even win rate at -110: 52.4%")
    print(f"  Recommended threshold: >15% (n=267, ROI +7.3% over 9 seasons)")


def _print_overall(rows: list[dict]) -> None:
    print(f"\n--- Overall (2016-2025, >+10% edge) ---")
    print(f"{'Week group':<20} {'n':>5} {'Wins':>6} {'Win rate':>9} {'ROI':>8}  {'Profitable?':>12}")
    print("-" * 68)
    for r in rows:
        if r["n"] == 0:
            continue
        profitable = "YES" if r["roi"] is not None and r["roi"] > 0 else "no"
        roi_str = f"{r['roi']:+.1%}" if r["roi"] is not None else "n/a"
        print(
            f"{r['week_group']:<20} {r['n']:>5} {r['wins']:>6} "
            f"{r['win_rate']:>9.1%} {roi_str:>8}  {profitable:>12}"
        )
    print(f"\n  Break-even win rate at -110: 52.4%")


def _print_by_season(rows: list[dict]) -> None:
    print(f"\n--- By season, late weeks (11+) only ---")
    late = [r for r in rows if r["week_group"] == "Late  (wk 11+)"]
    print(f"{'Season':>7} {'n':>5} {'Win rate':>9} {'ROI':>8}")
    print("-" * 35)
    for r in late:
        if r["n"] == 0:
            continue
        roi_str = f"{r['roi']:+.1%}" if r["roi"] is not None else "n/a"
        print(f"{r['season']:>7} {r['n']:>5} {r['win_rate']:>9.1%} {roi_str:>8}")


def run_roi_backtest(seasons: list[int] = ALL_SEASONS) -> dict:
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    print(f"[roi backtest] seasons: {seasons[0]}-{seasons[-1]}")
    fp = _build_first_plays(seasons)
    fp["week_group"] = fp["week"].apply(lambda w: next(
        lbl for lbl, (lo, hi) in WEEK_GROUPS if lo <= w <= hi
    ))

    season_rows    = _season_rows(fp)
    overall_rows   = _overall_rows(fp)
    threshold_rows = _threshold_sensitivity(fp)
    cumulative     = _cumulative_pnl(fp)

    _plot_pnl(cumulative, int((fp["edge"] > EDGE_THRESH).sum()))

    result = {
        "seasons":            [seasons[0], seasons[-1]],
        "edge_thresh":        EDGE_THRESH,
        "odds":               ODDS,
        "breakeven_wr":       0.5238,
        "recommended_thresh": 0.15,
        "overall":            overall_rows,
        "by_season":          season_rows,
        "threshold_sensitivity": threshold_rows,
        "final_pnl":          round(float(cumulative.iloc[-1]), 2) if len(cumulative) else 0,
    }
    json_path = ARTIFACTS / "roi_summary.json"
    json_path.write_text(json.dumps(result, indent=2))
    print(f"  saved -> {json_path.name}")

    _print_overall(overall_rows)
    _print_by_season(season_rows)
    _print_threshold_table(threshold_rows)

    total_pos = fp[fp["edge"] > EDGE_THRESH]
    print(f"\n  Total bets placed: {len(total_pos)}")
    print(f"  Final cumulative P&L: ${cumulative.iloc[-1]:+,.0f}")

    return result


if __name__ == "__main__":
    run_roi_backtest(ALL_SEASONS)
