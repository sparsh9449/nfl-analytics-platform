"""
NFL Analytics Platform — FastAPI backend.

Serves pre-computed artifacts and live model inference to the Next.js frontend.
All heavy computation (model loading, parquet reads) is done at startup and
cached in module-level variables to keep request latency low.
"""

import json
import sys
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from models.win_probability.calibrated_model import IsotonicCalibratedXGB  # noqa: F401
from models.inference.weekly_picks import (
    run_weekly_picks,
    _games_for_week,
    _compute_rolling_stats,
    _build_features,
    MODEL_PATH,
    NFL_SPREAD_SIGMA,
)
from models.win_probability.data_prep import FEATURES
from scipy.stats import norm

# ---------------------------------------------------------------------------
# Startup: load artifacts once
# ---------------------------------------------------------------------------

ARTIFACTS   = ROOT / "models" / "win_probability" / "artifacts"
EDGE_DIR    = ROOT / "models" / "edge_detection" / "artifacts"
GOLD_PATH   = ROOT / "data" / "gold" / "wp_features.parquet"
SILVER_DIR  = ROOT / "data" / "silver"

_model      = joblib.load(MODEL_PATH)
_metrics    = json.loads((ARTIFACTS / "metrics_summary.json").read_text())
_edge_sum   = json.loads((EDGE_DIR / "edge_summary.json").read_text())
_week_break = json.loads((EDGE_DIR / "week_breakdown.json").read_text())
_roi_sum    = json.loads((EDGE_DIR / "roi_summary.json").read_text())

# Available seasons = silver files present
_seasons = sorted(
    int(p.stem.replace("pbp_", ""))
    for p in SILVER_DIR.glob("pbp_*.parquet")
)

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
    """Available seasons and weeks in the dataset."""
    df = pd.read_parquet(GOLD_PATH, columns=["season", "week"])
    by_season = (
        df.groupby("season")["week"]
        .agg(min_week="min", max_week="max")
        .reset_index()
        .to_dict(orient="records")
    )
    return {"seasons": by_season}


@app.get("/games/{season}/{week}")
def get_games(season: int, week: int, threshold: Optional[float] = None):
    """
    Model WP vs market WP for every game in a given week.
    Optionally filter to games where |edge| >= threshold.
    """
    if season not in _seasons:
        raise HTTPException(404, f"Season {season} not available")

    games = _games_for_week(season, week)
    if games.empty:
        raise HTTPException(404, f"No games found for season {season} week {week}")

    roll = _compute_rolling_stats(season, before_week=week)
    fp   = _build_features(games, roll)

    X = fp[FEATURES]
    fp["model_wp"]  = _model.predict_proba(X)[:, 1]
    fp["market_wp"] = norm.cdf(fp["spread_line"] / NFL_SPREAD_SIGMA)
    fp["edge"]      = (fp["model_wp"] - fp["market_wp"]).round(4)

    fp["bet_team"] = fp.apply(
        lambda r: r["home_team"] if r["edge"] > 0 else r["away_team"], axis=1
    )
    fp["is_late_season"] = week >= 11

    if threshold is not None:
        fp = fp[fp["edge"].abs() >= threshold]

    fp = fp.sort_values("edge", key=abs, ascending=False)

    records = fp[[
        "game_id", "home_team", "away_team",
        "spread_line", "total_line",
        "market_wp", "model_wp", "edge",
        "bet_team", "is_late_season",
    ]].round(4).to_dict(orient="records")

    return {
        "season":    season,
        "week":      week,
        "n_games":   len(records),
        "games":     records,
    }


@app.get("/model/metrics")
def get_metrics():
    """AUC, Brier, and calibration data for all trained models."""
    return _metrics


@app.get("/model/edge-summary")
def get_edge_summary():
    """Pre-game edge bucket backtest results (test seasons)."""
    return _edge_sum


@app.get("/model/week-breakdown")
def get_week_breakdown():
    """Edge signal breakdown by week group (early / mid / late season)."""
    return _week_break


@app.get("/model/roi")
def get_roi():
    """Flat-bet ROI analysis by season and edge threshold."""
    return _roi_sum


@app.get("/history/{season}/{week}")
def get_history(season: int, week: int):
    """
    Historical game results for a past week, including actual outcomes
    so the frontend can show model accuracy on completed games.
    """
    if season not in _seasons:
        raise HTTPException(404, f"Season {season} not available")

    try:
        path = SILVER_DIR / f"pbp_{season}.parquet"
        cols = ["game_id", "week", "home_team", "away_team",
                "spread_line", "total_line", "vegas_wp",
                "total_home_score", "total_away_score", "result"]
        df = pd.read_parquet(path, columns=cols)
    except Exception as e:
        raise HTTPException(500, str(e))

    week_df = df[df["week"] == week]
    if week_df.empty:
        raise HTTPException(404, f"No data for season {season} week {week}")

    game_sum = (
        week_df.groupby("game_id")
        .agg(
            home_team    = ("home_team",        "first"),
            away_team    = ("away_team",        "first"),
            spread_line  = ("spread_line",      "first"),
            total_line   = ("total_line",       "first"),
            home_score   = ("total_home_score", "max"),
            away_score   = ("total_away_score", "max"),
            result       = ("result",           "first"),
        )
        .reset_index()
    )
    game_sum["home_won"] = (game_sum["result"] > 0).astype(int)
    game_sum["market_wp"] = norm.cdf(
        game_sum["spread_line"] / NFL_SPREAD_SIGMA
    ).round(4)

    # Get model WP for this week
    roll = _compute_rolling_stats(season, before_week=week)
    games_meta = game_sum[["game_id","home_team","away_team","spread_line","total_line"]].copy()
    games_meta["vegas_wp"] = 0.5
    games_meta["week"] = week
    fp = _build_features(games_meta, roll)
    fp["model_wp"] = _model.predict_proba(fp[FEATURES])[:, 1].round(4)
    game_sum = game_sum.merge(fp[["game_id","model_wp"]], on="game_id", how="left")
    game_sum["edge"]  = (game_sum["model_wp"] - game_sum["market_wp"]).round(4)
    game_sum["model_correct"] = (
        ((game_sum["model_wp"] > 0.5) & (game_sum["home_won"] == 1)) |
        ((game_sum["model_wp"] < 0.5) & (game_sum["home_won"] == 0))
    ).astype(int)

    return {
        "season":  season,
        "week":    week,
        "n_games": len(game_sum),
        "games":   game_sum.round(4).to_dict(orient="records"),
    }
