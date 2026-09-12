"""
Pre-game model inference for upcoming NFL games.

Fetches live game data from ESPN (matchups + DraftKings spreads),
computes rolling team stats from historical silver data as a warm-start,
and runs the calibrated pre-game XGBoost model.

ESPN spread convention: negative = home favored.
nflfastR convention (what our model was trained on): positive = home favored.
We convert: spread_line = -1 * espn_spread.
"""

from pathlib import Path
import math
import urllib.request
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT         = Path(__file__).resolve().parents[2]
SILVER_DIR   = ROOT / "data" / "silver"
ARTIFACT_DIR = ROOT / "models" / "pregame" / "artifacts"

NFL_SPREAD_SIGMA = 13.86
ROLL_N           = 4

PREGAME_FEATURES = [
    "market_wp",
    "total_line",
    "week",
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

_artifact_cache: dict | None = None


def _load_model() -> dict:
    global _artifact_cache
    if _artifact_cache is None:
        _artifact_cache = joblib.load(ARTIFACT_DIR / "xgb_pregame_calibrated.pkl")
    return _artifact_cache


def _fetch_espn_scoreboard() -> list[dict]:
    """Fetch current week's games from ESPN scoreboard including odds."""
    url = "https://site.api.espn.com/apis/site/v2/sports/football/nfl/scoreboard"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read())

    games = []
    for event in data.get("events", []):
        comp = event["competitions"][0]
        status = comp["status"]["type"]["name"]

        competitors = comp.get("competitors", [])
        home = next((c for c in competitors if c.get("homeAway") == "home"), None)
        away = next((c for c in competitors if c.get("homeAway") == "away"), None)
        if not home or not away:
            continue

        home_abbr = home["team"]["abbreviation"]
        away_abbr = away["team"]["abbreviation"]

        odds = comp.get("odds", [])
        spread_line = None   # nflfastR convention: positive = home favored
        total_line  = None
        if odds:
            o = odds[0]
            espn_spread = o.get("spread")
            if espn_spread is not None:
                # ESPN: negative spread = home favored → flip to nflfastR
                spread_line = -float(espn_spread)
            total_line = o.get("overUnder")

        games.append({
            "event_id":   event.get("id"),
            "name":       event["name"],
            "date":       event["date"],
            "status":     status,
            "home_team":  home_abbr,
            "away_team":  away_abbr,
            "home_score": home.get("score"),
            "away_score": away.get("score"),
            "spread_line": spread_line,
            "total_line":  total_line,
        })

    return games


def _team_abbr_map() -> dict[str, str]:
    """Map common ESPN abbreviations to nflfastR abbreviations."""
    return {
        "WSH": "WAS",
        "JAX": "JAC",
        "ARI": "ARI",
    }


def _warm_start_stats(warmup_season: int = 2025) -> dict[str, dict]:
    """
    Build per-team rolling stats from the most recent ROLL_N games in
    the warm-up season.  Used as the pre-2026 baseline.
    Returns {team: {roll4_pts_scored, roll4_pts_allowed, ...}}
    """
    path = SILVER_DIR / f"pbp_{warmup_season}.parquet"
    if not path.exists():
        return {}

    cols = ["game_id", "week", "home_team", "away_team", "game_date",
            "total_home_score", "total_away_score",
            "total_home_epa", "total_away_epa"]
    df = pd.read_parquet(path, columns=cols)

    gl = (df.groupby("game_id")
          .agg(week=("week","first"), home_team=("home_team","first"),
               away_team=("away_team","first"), game_date=("game_date","first"),
               home_score=("total_home_score","max"), away_score=("total_away_score","max"),
               home_epa=("total_home_epa","max"), away_epa=("total_away_epa","max"))
          .reset_index())
    gl["game_date"] = pd.to_datetime(gl["game_date"])

    home = gl.rename(columns={"home_team":"team","home_score":"pts_scored",
                               "away_score":"pts_allowed","home_epa":"epa_scored",
                               "away_epa":"epa_allowed"})
    home["win"] = (home["pts_scored"] > home["pts_allowed"]).astype(float)
    away = gl.rename(columns={"away_team":"team","away_score":"pts_scored",
                               "home_score":"pts_allowed","away_epa":"epa_scored",
                               "home_epa":"epa_allowed"})
    away["win"] = (away["pts_scored"] > away["pts_allowed"]).astype(float)

    tg = (pd.concat([home[["team","week","game_date","pts_scored","pts_allowed",
                            "epa_scored","epa_allowed","win"]],
                     away[["team","week","game_date","pts_scored","pts_allowed",
                            "epa_scored","epa_allowed","win"]]], ignore_index=True)
          .sort_values(["team","game_date"]).reset_index(drop=True))

    results = {}
    for team, grp in tg.groupby("team"):
        last = grp.tail(ROLL_N)
        last_game_date = grp["game_date"].max()
        rest = float((pd.Timestamp("2026-09-11") - last_game_date).days)
        results[team] = {
            "roll4_pts_scored":  last["pts_scored"].mean(),
            "roll4_pts_allowed": last["pts_allowed"].mean(),
            "roll4_epa_scored":  last["epa_scored"].mean(),
            "roll4_epa_allowed": last["epa_allowed"].mean(),
            "roll4_win_pct":     last["win"].mean(),
            "rest_days":         rest,
        }
    return results


def get_picks(week: int | None = None) -> list[dict]:
    """
    Return pre-game picks for all scheduled games this week.
    Each pick includes: teams, spread, market_wp, model_wp, edge, recommendation.
    """
    artifact = _load_model()
    xgb_model = artifact["xgb"]
    iso        = artifact["iso"]
    abbr_map   = _team_abbr_map()

    games = _fetch_espn_scoreboard()
    scheduled = [g for g in games if g["status"] == "STATUS_SCHEDULED"]
    if not scheduled:
        # Fall back to all games if nothing is scheduled (e.g. mid-week call)
        scheduled = games

    warm = _warm_start_stats(warmup_season=2025)

    picks = []
    for g in scheduled:
        home = abbr_map.get(g["home_team"], g["home_team"])
        away = abbr_map.get(g["away_team"], g["away_team"])
        spread_line = g["spread_line"]
        total_line  = g["total_line"]

        market_wp = norm.cdf(float(spread_line) / NFL_SPREAD_SIGMA) if spread_line is not None else 0.5

        home_stats = warm.get(home, {})
        away_stats = warm.get(away, {})

        row = {
            "market_wp":              market_wp,
            "total_line":             total_line,
            "week":                   week or 1,
            "home_roll4_pts_scored":  home_stats.get("roll4_pts_scored"),
            "home_roll4_pts_allowed": home_stats.get("roll4_pts_allowed"),
            "home_roll4_epa_scored":  home_stats.get("roll4_epa_scored"),
            "home_roll4_epa_allowed": home_stats.get("roll4_epa_allowed"),
            "home_roll4_win_pct":     home_stats.get("roll4_win_pct"),
            "home_rest_days":         home_stats.get("rest_days"),
            "away_roll4_pts_scored":  away_stats.get("roll4_pts_scored"),
            "away_roll4_pts_allowed": away_stats.get("roll4_pts_allowed"),
            "away_roll4_epa_scored":  away_stats.get("roll4_epa_scored"),
            "away_roll4_epa_allowed": away_stats.get("roll4_epa_allowed"),
            "away_roll4_win_pct":     away_stats.get("roll4_win_pct"),
            "away_rest_days":         away_stats.get("rest_days"),
        }

        X = pd.DataFrame([row])[PREGAME_FEATURES]
        for col in PREGAME_FEATURES:
            X[col] = pd.to_numeric(X[col], errors="coerce")

        raw_p    = xgb_model.predict_proba(X)[:, 1][0]
        # Platt scaler: predict_proba([[raw]])[:, 1]
        model_wp = float(np.clip(iso.predict_proba([[raw_p]])[0, 1], 0.0, 1.0))

        # edge from HOME team perspective (positive = model likes home more than market)
        home_edge = model_wp - market_wp
        # pick_edge: always positive, from the RECOMMENDED team's perspective
        pick_edge = abs(home_edge)

        # Recommended pick: whichever team the model prefers over the market
        # Use pick_edge (disagreement with market) for the pick direction
        pick_team = home if home_edge > 0 else away
        pick_side = "home" if home_edge > 0 else "away"

        # Confidence based on how much the model disagrees with the market
        confidence = "HIGH" if pick_edge >= 0.12 else "MEDIUM" if pick_edge >= 0.06 else "LOW"

        def _clean(v):
            if v is None: return None
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)): return None
            return round(v, 4) if isinstance(v, float) else v

        picks.append({
            "game_id":     g.get("event_id"),
            "name":        g["name"],
            "date":        g["date"],
            "home_team":   home,
            "away_team":   away,
            "spread_line": _clean(spread_line),
            "total_line":  _clean(total_line),
            "market_wp":   _clean(market_wp),
            "model_wp":    _clean(model_wp),
            "home_edge":   _clean(home_edge),   # signed, home perspective
            "edge":        _clean(pick_edge),   # always positive, pick's perspective
            "pick_team":   pick_team,
            "pick_side":   pick_side,
            "confidence":  confidence,
            # Context for UI
            "home_stats":  {k: _clean(v) for k, v in home_stats.items()},
            "away_stats":  {k: _clean(v) for k, v in away_stats.items()},
        })

    picks.sort(key=lambda x: abs(x["edge"] or 0), reverse=True)
    return picks
