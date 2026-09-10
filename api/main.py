"""
NFL Analytics Platform - FastAPI backend.

Endpoints
---------
GET /                           health check
GET /seasons                    available seasons and their week ranges
GET /games/{season}/{week}      model WP vs market WP for each game
GET /history/{season}/{week}    same with actual outcomes (past weeks)
GET /model/metrics              AUC, Brier, calibration data
GET /model/edge-summary         edge bucket backtest results
GET /model/week-breakdown       edge signal by week group
GET /model/roi                  flat-bet ROI analysis
GET /model/season-accuracy      per-season model accuracy breakdown
"""

import json
import math
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.win_probability.calibrated_model import IsotonicCalibratedXGB  # noqa: F401
from models.inference.weekly_picks import (
    _games_for_week,
    _compute_rolling_stats,
    _build_features,
    MODEL_PATH,
    NFL_SPREAD_SIGMA,
)
from models.win_probability.data_prep import FEATURES

# ---------------------------------------------------------------------------
# Startup: load artifacts once into module-level variables
# ---------------------------------------------------------------------------

ARTIFACTS  = ROOT / "models" / "win_probability" / "artifacts"
EDGE_DIR   = ROOT / "models" / "edge_detection" / "artifacts"
GOLD_PATH  = ROOT / "data" / "gold" / "wp_features.parquet"
SILVER_DIR = ROOT / "data" / "silver"

_model      = joblib.load(MODEL_PATH)
_metrics    = json.loads((ARTIFACTS / "metrics_summary.json").read_text())
_edge_sum   = json.loads((EDGE_DIR / "edge_summary.json").read_text())
_week_break = json.loads((EDGE_DIR / "week_breakdown.json").read_text())
_roi_sum    = json.loads((EDGE_DIR / "roi_summary.json").read_text())

_seasons = sorted(
    int(p.stem.replace("pbp_", ""))
    for p in SILVER_DIR.glob("pbp_*.parquet")
)

# Cache season/week metadata so /seasons doesn't re-scan parquet on every call
_season_meta: list[dict] | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_records(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records, replacing NaN/inf with None."""
    out = []
    for row in df.to_dict(orient="records"):
        clean = {}
        for k, v in row.items():
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                clean[k] = None
            elif isinstance(v, (np.integer,)):
                clean[k] = int(v)
            elif isinstance(v, (np.floating,)):
                clean[k] = None if math.isnan(float(v)) else round(float(v), 4)
            elif isinstance(v, (np.bool_,)):
                clean[k] = bool(v)
            else:
                clean[k] = v
        out.append(clean)
    return out


def _predict(fp: pd.DataFrame) -> np.ndarray:
    """Run model on feature DataFrame, returning P(home wins)."""
    X = fp[FEATURES].copy()
    for col in FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    return _model.predict_proba(X)[:, 1]


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="NFL Analytics Platform",
    description="Win probability model and pre-game edge detection API",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/")
def root():
    return {"status": "ok", "model": MODEL_PATH.name, "seasons": _seasons}


@app.get("/seasons")
def get_seasons():
    """Available seasons and their week ranges (cached after first call)."""
    global _season_meta
    if _season_meta is None:
        df = pd.read_parquet(GOLD_PATH, columns=["season", "week"])
        _season_meta = (
            df.groupby("season")["week"]
            .agg(min_week="min", max_week="max")
            .reset_index()
            .to_dict(orient="records")
        )
    return {"seasons": _season_meta}


@app.get("/games/{season}/{week}")
def get_games(season: int, week: int, threshold: Optional[float] = None):
    """
    Model WP vs market WP for every game in a given week.
    Pass threshold=0.10 to filter to |edge| >= 10%.
    """
    if season not in _seasons:
        raise HTTPException(404, f"Season {season} not available")

    games = _games_for_week(season, week)
    if games.empty:
        raise HTTPException(404, f"No games for season {season} week {week}")

    roll = _compute_rolling_stats(season, before_week=week)
    fp   = _build_features(games, roll)

    fp["model_wp"]  = _predict(fp)
    fp["market_wp"] = norm.cdf(fp["spread_line"] / NFL_SPREAD_SIGMA)
    fp["edge"]      = fp["model_wp"] - fp["market_wp"]
    fp["bet_team"]  = fp.apply(
        lambda r: r["home_team"] if r["edge"] > 0 else r["away_team"], axis=1
    )
    fp["is_late_season"] = (fp["week"] >= 11)

    if threshold is not None:
        fp = fp[fp["edge"].abs() >= threshold]

    fp = fp.sort_values("edge", key=abs, ascending=False)

    cols = ["game_id", "home_team", "away_team", "spread_line", "total_line",
            "market_wp", "model_wp", "edge", "bet_team", "is_late_season", "week"]
    return {"season": season, "week": week, "n_games": len(fp),
            "games": _to_records(fp[cols])}


@app.get("/history/{season}/{week}")
def get_history(season: int, week: int):
    """
    Game results for a completed week with model predictions and accuracy.
    """
    if season not in _seasons:
        raise HTTPException(404, f"Season {season} not available")

    path = SILVER_DIR / f"pbp_{season}.parquet"
    cols = ["game_id", "week", "home_team", "away_team",
            "spread_line", "total_line",
            "total_home_score", "total_away_score", "result"]
    df = pd.read_parquet(path, columns=cols)

    week_df = df[df["week"] == week]
    if week_df.empty:
        raise HTTPException(404, f"No data for season {season} week {week}")

    game_sum = (
        week_df.groupby("game_id")
        .agg(
            home_team   = ("home_team",        "first"),
            away_team   = ("away_team",        "first"),
            spread_line = ("spread_line",      "first"),
            total_line  = ("total_line",       "first"),
            home_score  = ("total_home_score", "max"),
            away_score  = ("total_away_score", "max"),
            result      = ("result",           "first"),
        )
        .reset_index()
    )
    game_sum["home_won"]  = (game_sum["result"].astype(float) > 0).astype(int)
    game_sum["market_wp"] = norm.cdf(game_sum["spread_line"] / NFL_SPREAD_SIGMA)

    roll = _compute_rolling_stats(season, before_week=week)
    meta = game_sum[["game_id", "home_team", "away_team",
                     "spread_line", "total_line"]].copy()
    meta["vegas_wp"] = 0.5
    meta["week"]     = week
    fp = _build_features(meta, roll)
    fp["model_wp"] = _predict(fp)

    game_sum = game_sum.merge(fp[["game_id", "model_wp"]], on="game_id", how="left")
    game_sum["edge"] = game_sum["model_wp"] - game_sum["market_wp"]
    game_sum["model_correct"] = (
        ((game_sum["model_wp"] > 0.5) & (game_sum["home_won"] == 1)) |
        ((game_sum["model_wp"] < 0.5) & (game_sum["home_won"] == 0))
    ).astype(int)

    cols = ["game_id", "home_team", "away_team", "spread_line",
            "home_score", "away_score", "home_won",
            "market_wp", "model_wp", "edge", "model_correct"]
    return {"season": season, "week": week, "n_games": len(game_sum),
            "games": _to_records(game_sum[cols])}


@app.get("/model/metrics")
def get_metrics():
    """AUC, Brier, and calibration data for all trained models."""
    return _metrics


@app.get("/model/edge-summary")
def get_edge_summary():
    """Pre-game edge bucket backtest (test seasons)."""
    return _edge_sum


@app.get("/model/week-breakdown")
def get_week_breakdown():
    """Edge signal breakdown by week group (early / mid / late season)."""
    return _week_break


@app.get("/model/roi")
def get_roi():
    """Flat-bet ROI analysis by season and edge threshold."""
    return _roi_sum


@app.get("/model/season-accuracy")
def get_season_accuracy():
    """
    Per-season model accuracy and Brier score, computed from the gold table.
    Uses the first valid scrimmage play per game (pre-game state) and
    compares the model's predicted P(home wins) to the actual outcome.
    Only evaluates seasons in the test set (2024-2025) for unbiased results;
    returns all seasons for the historical chart.
    """
    df = pd.read_parquet(GOLD_PATH)

    # First valid play per game
    valid = df[df["down"].notna() & df["posteam_win"].notna()].copy()
    first = (valid.sort_values("game_seconds_remaining", ascending=False)
             .groupby("game_id").first().reset_index())

    # Build home-team-perspective features
    first["is_home_first"] = (first["posteam"] == first["home_team"])
    first["home_win"] = first.apply(
        lambda r: r["posteam_win"] if r["is_home_first"] else 1.0 - r["posteam_win"],
        axis=1,
    )
    first = first[first["home_win"].isin([0.0, 1.0])]

    results = []
    for season in sorted(first["season"].unique()):
        sg = first[first["season"] == season]
        if len(sg) < 10:
            continue

        X = sg[FEATURES].copy()
        for col in FEATURES:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        model_wp = _model.predict_proba(X)[:, 1]
        actual   = sg["home_win"].to_numpy()
        correct  = ((model_wp > 0.5) == (actual == 1)).mean()
        brier    = float(np.mean((model_wp - actual) ** 2))
        results.append({
            "season":   int(season),
            "n_games":  int(len(sg)),
            "accuracy": round(float(correct), 4),
            "brier":    round(brier, 4),
        })

    return {"seasons": results}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True)
