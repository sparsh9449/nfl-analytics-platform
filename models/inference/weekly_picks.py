"""
Weekly picks: apply the calibrated model to a given season/week and print
games where the model's pre-game win probability disagrees with the market.

Feature construction
--------------------
All pre-game game-state features are fixed at their start-of-game values
(score 0-0, 3600 seconds, 1st-and-10 from the 65). The only differentiating
features are the rolling 4-game team stats, computed from games played in
weeks < target_week within the same season.

The prediction is always from the HOME team's perspective, which matches
how vegas_wp is defined in nflfastR (market-implied P(home wins)).
Positive edge = model likes the home team more than the market does.
Negative edge = model likes the away team.

Market WP derivation
--------------------
Market-implied P(home wins) = Phi(spread_line / 13.86), where positive
spread_line = home favored (nflfastR convention, opposite of standard American
odds notation). This is more reliable than using vegas_wp from the first play,
which is a game-state-adjusted figure that varies with field position.

Usage
-----
  python -m models.inference.weekly_picks --season 2024 --week 14
  python -m models.inference.weekly_picks --season 2024 --week 14 --threshold 0.15
  python -m models.inference.weekly_picks --season 2024 --week 14 --all   # show all games
"""

import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from models.win_probability.calibrated_model import IsotonicCalibratedXGB  # noqa: F401
from models.win_probability.data_prep import FEATURES

SILVER_DIR = ROOT / "data" / "silver"
_CAL_PATH  = ROOT / "models" / "win_probability" / "artifacts" / "xgb_calibrated.pkl"
_RAW_PATH  = ROOT / "models" / "win_probability" / "artifacts" / "xgb_model.pkl"
MODEL_PATH = _CAL_PATH if _CAL_PATH.exists() else _RAW_PATH

ROLL_N             = 4
LATE_SEASON_WEEK   = 11
DEFAULT_THRESHOLD  = 0.10
BREAKEVEN_WR       = 0.5238   # at -110 odds
NFL_SPREAD_SIGMA   = 13.86    # std dev of NFL point differentials

# Fixed pre-game game-state values (same for every game)
# Fixed pre-game game-state values.
# is_home = 1 always because inference always uses home team as posteam.
# week is added dynamically per game in _build_features.
PREGAME_STATE = {
    "score_differential":        0,
    "game_seconds_remaining":    3600,
    "yardline_100":              65,
    "down":                      1,
    "ydstogo":                   10,
    "posteam_timeouts_remaining": 3,
    "defteam_timeouts_remaining": 3,
    "is_home":                   1,
}


# ---------------------------------------------------------------------------
# Rolling stats
# ---------------------------------------------------------------------------

def _compute_rolling_stats(season: int, before_week: int) -> pd.DataFrame:
    """
    Rolling ROLL_N-game averages of points and EPA for each team,
    using only games from weeks strictly before `before_week` in `season`.
    Also computes rest_days (days since each team's previous game).
    """
    path = SILVER_DIR / f"pbp_{season}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Silver data not found: {path}")

    cols = ["game_id", "week", "home_team", "away_team", "game_date",
            "total_home_score", "total_away_score",
            "total_home_epa", "total_away_epa"]
    df = pd.read_parquet(path, columns=cols)

    prior = df[df["week"] < before_week]
    if prior.empty:
        return pd.DataFrame(columns=["game_id", "team",
                                     f"roll{ROLL_N}_pts_scored",  f"roll{ROLL_N}_pts_allowed",
                                     f"roll{ROLL_N}_epa_scored",  f"roll{ROLL_N}_epa_allowed",
                                     "rest_days"])

    gl = (prior.groupby("game_id")
          .agg(week      =("week",             "first"),
               home_team =("home_team",        "first"),
               away_team =("away_team",        "first"),
               game_date =("game_date",        "first"),
               home_final=("total_home_score", "max"),
               away_final=("total_away_score", "max"),
               home_epa  =("total_home_epa",   "max"),
               away_epa  =("total_away_epa",   "max"))
          .reset_index())

    home = gl[["game_id","week","game_date","home_team","home_final","away_final","home_epa","away_epa"]].rename(
        columns={"home_team":"team","home_final":"pts_scored","away_final":"pts_allowed",
                 "home_epa":"epa_scored","away_epa":"epa_allowed"})
    away = gl[["game_id","week","game_date","away_team","away_final","home_final","away_epa","home_epa"]].rename(
        columns={"away_team":"team","away_final":"pts_scored","home_final":"pts_allowed",
                 "away_epa":"epa_scored","home_epa":"epa_allowed"})

    tg = (pd.concat([home, away], ignore_index=True)
          .sort_values(["team","week"]).reset_index(drop=True))

    for raw, col in [("pts_scored",  f"roll{ROLL_N}_pts_scored"),
                     ("pts_allowed", f"roll{ROLL_N}_pts_allowed"),
                     ("epa_scored",  f"roll{ROLL_N}_epa_scored"),
                     ("epa_allowed", f"roll{ROLL_N}_epa_allowed")]:
        tg[col] = (tg.groupby("team")[raw]
                   .transform(lambda s: s.shift(1).rolling(ROLL_N, min_periods=1).mean()))

    tg["game_date"] = pd.to_datetime(tg["game_date"])
    tg["rest_days"] = tg.groupby("team")["game_date"].transform(lambda s: s.diff().dt.days)

    return tg[["game_id","team",
               f"roll{ROLL_N}_pts_scored", f"roll{ROLL_N}_pts_allowed",
               f"roll{ROLL_N}_epa_scored", f"roll{ROLL_N}_epa_allowed",
               "rest_days"]]


# ---------------------------------------------------------------------------
# Game list for the target week
# ---------------------------------------------------------------------------

def _games_for_week(season: int, week: int) -> pd.DataFrame:
    """
    One row per game in target week with home_team, away_team,
    spread_line, and vegas_wp sourced from the silver data.
    """
    path = SILVER_DIR / f"pbp_{season}.parquet"
    cols = ["game_id", "week", "home_team", "away_team",
            "spread_line", "total_line", "vegas_wp"]
    df = pd.read_parquet(path, columns=cols)
    week_df = df[df["week"] == week]
    if week_df.empty:
        return pd.DataFrame()

    # Take the first available row per game (spread/vegas_wp are game-level constants)
    return (
        week_df.sort_values("game_id")
        .groupby("game_id", sort=False)
        .first()
        .reset_index()
        [["game_id", "week", "home_team", "away_team", "spread_line", "total_line", "vegas_wp"]]
        .dropna(subset=["vegas_wp"])
    )


# ---------------------------------------------------------------------------
# Feature construction
# ---------------------------------------------------------------------------

def _build_features(games: pd.DataFrame, roll_stats: pd.DataFrame) -> pd.DataFrame:
    """
    Build one feature row per game from the home team's perspective.
    Joins rolling stats for home (posteam) and away (defteam) teams.
    """
    # Rolling stats keyed by (game_id, team) — but here we want
    # the stats each team carries into THIS game, not the prior game_id.
    # Use team-level lookup: take the rolling stats from the most recent
    # prior game for each team.
    stat_cols = ["team",
                 f"roll{ROLL_N}_pts_scored", f"roll{ROLL_N}_pts_allowed",
                 f"roll{ROLL_N}_epa_scored", f"roll{ROLL_N}_epa_allowed",
                 "rest_days"]
    available = [c for c in stat_cols if c in roll_stats.columns]
    latest = (
        roll_stats
        .sort_values("game_id")
        .groupby("team")
        .last()
        .reset_index()
        [available]
    )
    latest_dict = latest.set_index("team").to_dict("index")

    rows = []
    for _, g in games.iterrows():
        home_stats = latest_dict.get(g["home_team"], {})
        away_stats = latest_dict.get(g["away_team"], {})

        row = {
            **PREGAME_STATE,
            "week": int(g["week"]),
            "posteam_roll4_pts_scored":  home_stats.get(f"roll{ROLL_N}_pts_scored"),
            "posteam_roll4_pts_allowed": home_stats.get(f"roll{ROLL_N}_pts_allowed"),
            "posteam_roll4_epa_scored":  home_stats.get(f"roll{ROLL_N}_epa_scored"),
            "posteam_roll4_epa_allowed": home_stats.get(f"roll{ROLL_N}_epa_allowed"),
            "posteam_rest_days":         home_stats.get("rest_days"),
            "defteam_roll4_pts_scored":  away_stats.get(f"roll{ROLL_N}_pts_scored"),
            "defteam_roll4_pts_allowed": away_stats.get(f"roll{ROLL_N}_pts_allowed"),
            "defteam_roll4_epa_scored":  away_stats.get(f"roll{ROLL_N}_epa_scored"),
            "defteam_roll4_epa_allowed": away_stats.get(f"roll{ROLL_N}_epa_allowed"),
            "defteam_rest_days":         away_stats.get("rest_days"),
            # metadata (not model features)
            "game_id":     g["game_id"],
            "home_team":   g["home_team"],
            "away_team":   g["away_team"],
            "spread_line": g["spread_line"],
            "total_line":  g["total_line"],
            "vegas_wp":    g["vegas_wp"],
        }
        rows.append(row)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_weekly_picks(
    season: int,
    week: int,
    threshold: float = DEFAULT_THRESHOLD,
    show_all: bool = False,
) -> pd.DataFrame:

    print(f"\n[weekly picks] season {season}, week {week}")
    print(f"  model: {MODEL_PATH.name}  |  threshold: {threshold:+.0%}  |  odds: -110")
    is_late = week >= LATE_SEASON_WEEK
    if not is_late:
        print(f"  note: week {week} is early/mid season (< wk {LATE_SEASON_WEEK})")
        print(f"        signal is validated for wk {LATE_SEASON_WEEK}+ only; use with caution")

    games = _games_for_week(season, week)
    if games.empty:
        print(f"  no games found for season {season} week {week}")
        return pd.DataFrame()
    print(f"  {len(games)} games found")

    roll_stats = _compute_rolling_stats(season, before_week=week)
    print(f"  rolling stats computed from {len(roll_stats)//2 if len(roll_stats) else 0} prior games")

    fp = _build_features(games, roll_stats)

    model = joblib.load(MODEL_PATH)
    X = fp[FEATURES]
    fp["model_wp"]  = model.predict_proba(X)[:, 1]   # P(home wins)
    # Convert spread_line to market-implied P(home wins).
    # nflfastR: positive spread_line = home favored. P = Φ(spread / sigma).
    fp["market_wp"] = norm.cdf(fp["spread_line"] / NFL_SPREAD_SIGMA)
    fp["edge"]      = fp["model_wp"] - fp["market_wp"]

    # Determine bet side and label
    fp["bet_team"] = fp.apply(
        lambda r: r["home_team"] if r["edge"] > 0 else r["away_team"], axis=1
    )
    fp["bet_side"] = fp["edge"].apply(lambda e: "home" if e > 0 else "away")

    picks = fp.copy()
    if not show_all:
        picks = picks[picks["edge"].abs() >= threshold]

    picks = picks.reindex(picks["edge"].abs().sort_values(ascending=False).index)

    _print_table(picks, season, week, threshold, show_all, is_late)

    return picks[[
        "game_id", "home_team", "away_team", "spread_line",
        "market_wp", "model_wp", "edge", "bet_team", "bet_side",
    ]]


def _print_table(
    picks: pd.DataFrame,
    season: int,
    week: int,
    threshold: float,
    show_all: bool,
    is_late: bool,
) -> None:
    label = "ALL GAMES" if show_all else f"PICKS (|edge| >= {threshold:.0%})"
    late_tag = "[LATE SEASON - signal active]" if is_late else "[early/mid season - use with caution]"

    print(f"\n  {late_tag}")
    print(f"\n  Season {season} | Week {week} | {label}")
    print(f"  {'Matchup':<22} {'Spread':>7} {'Market WP':>10} {'Model WP':>9} {'Edge':>7}  {'Pick'}")
    print(f"  {'-'*75}")

    if picks.empty:
        print(f"  no games meet the {threshold:.0%} threshold this week")
        return

    for _, r in picks.iterrows():
        spread_str = f"{r['home_team']} {r['spread_line']:+.1f}"
        matchup    = f"{r['away_team']} @ {r['home_team']}"
        edge_str   = f"{r['edge']:+.1%}"
        pick_str   = f"BET {r['bet_team']} ({r['bet_side']})"
        print(
            f"  {matchup:<22} {spread_str:>7} "
            f"{r['market_wp']:>10.1%} {r['model_wp']:>9.1%} "
            f"{edge_str:>7}  {pick_str}"
        )

    if not show_all:
        print(f"\n  {len(picks)} pick(s) | break-even at -110: {BREAKEVEN_WR:.1%}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NFL pre-game edge picks")
    parser.add_argument("--season",    type=int, required=True, help="e.g. 2024")
    parser.add_argument("--week",      type=int, required=True, help="e.g. 14")
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD,
                        help="minimum |edge| to show (default 0.15)")
    parser.add_argument("--all",       action="store_true",
                        help="show all games regardless of edge size")
    args = parser.parse_args()

    run_weekly_picks(
        season=args.season,
        week=args.week,
        threshold=args.threshold,
        show_all=args.all,
    )
