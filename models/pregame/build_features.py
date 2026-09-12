"""
Build game-level pre-game features from silver play-by-play data.

One row per game.  All features are knowable before kickoff:
  - spread_line, total_line  (Vegas lines)
  - market_wp                (implied P(home wins) = Φ(spread / 13.86))
  - rolling 4-game team stats: pts scored/allowed, EPA scored/allowed, win pct
  - rest_days for home and away teams
  - week, season

Target: home_win  (1 = home team won, 0 = away won; ties excluded)

Output: data/gold/pregame_features.parquet
"""

from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT       = Path(__file__).resolve().parents[2]
SILVER_DIR = ROOT / "data" / "silver"
GOLD_DIR   = ROOT / "data" / "gold"

ROLL_N           = 4
NFL_SPREAD_SIGMA = 13.86
SEASONS          = list(range(2016, 2026))


def _load_game_level(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for s in seasons:
        path = SILVER_DIR / f"pbp_{s}.parquet"
        if not path.exists():
            print(f"  skip {s}: file not found")
            continue
        cols = ["game_id", "season", "week", "home_team", "away_team",
                "game_date", "spread_line", "total_line",
                "total_home_score", "total_away_score", "result"]
        df = pd.read_parquet(path, columns=cols)
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)

    gl = (
        raw.sort_values("game_id")
        .groupby("game_id")
        .agg(
            season      = ("season",           "first"),
            week        = ("week",             "first"),
            home_team   = ("home_team",        "first"),
            away_team   = ("away_team",        "first"),
            game_date   = ("game_date",        "first"),
            spread_line = ("spread_line",      "first"),
            total_line  = ("total_line",       "first"),
            home_score  = ("total_home_score", "max"),
            away_score  = ("total_away_score", "max"),
            result      = ("result",           "first"),
        )
        .reset_index()
    )
    gl["game_date"] = pd.to_datetime(gl["game_date"])
    return gl


def _rolling_stats(gl: pd.DataFrame) -> pd.DataFrame:
    """
    (game_id, team) table with rolling ROLL_N-game stats using shift(1)
    so no future information leaks into pre-game features.
    Resets each season.
    """
    home = gl[["game_id", "season", "week", "home_team",
               "home_score", "away_score", "result"]].rename(
        columns={"home_team": "team",
                 "home_score": "pts_scored",
                 "away_score": "pts_allowed"})
    home["win"] = (home["result"].astype(float) > 0).astype(float)

    away = gl[["game_id", "season", "week", "away_team",
               "away_score", "home_score", "result"]].rename(
        columns={"away_team": "team",
                 "away_score": "pts_scored",
                 "home_score": "pts_allowed"})
    away["win"] = (away["result"].astype(float) < 0).astype(float)

    # EPA per game from silver — derive from season totals available in game-level
    # (not available here directly; use pts as proxy; EPA added below if needed)
    tg = (pd.concat([home, away], ignore_index=True)
          .sort_values(["team", "season", "week"])
          .reset_index(drop=True))

    for raw_col, roll_col in [
        ("pts_scored",  f"roll{ROLL_N}_pts_scored"),
        ("pts_allowed", f"roll{ROLL_N}_pts_allowed"),
        ("win",         f"roll{ROLL_N}_win_pct"),
    ]:
        tg[roll_col] = (
            tg.groupby(["team", "season"])[raw_col]
            .transform(lambda s: s.shift(1).rolling(ROLL_N, min_periods=1).mean())
        )

    return tg[["game_id", "team",
               f"roll{ROLL_N}_pts_scored",
               f"roll{ROLL_N}_pts_allowed",
               f"roll{ROLL_N}_win_pct"]]


def _epa_stats(seasons: list[int]) -> pd.DataFrame:
    """Rolling EPA stats — requires loading EPA columns from silver."""
    frames = []
    for s in seasons:
        path = SILVER_DIR / f"pbp_{s}.parquet"
        if not path.exists():
            continue
        cols = ["game_id", "season", "week", "home_team", "away_team",
                "total_home_epa", "total_away_epa"]
        df = pd.read_parquet(path, columns=cols)
        frames.append(df)

    raw = pd.concat(frames, ignore_index=True)
    gl = (raw.groupby("game_id")
          .agg(season=("season","first"), week=("week","first"),
               home_team=("home_team","first"), away_team=("away_team","first"),
               home_epa=("total_home_epa","max"), away_epa=("total_away_epa","max"))
          .reset_index())

    home = gl[["game_id","season","week","home_team","home_epa","away_epa"]].rename(
        columns={"home_team":"team","home_epa":"epa_scored","away_epa":"epa_allowed"})
    away = gl[["game_id","season","week","away_team","away_epa","home_epa"]].rename(
        columns={"away_team":"team","away_epa":"epa_scored","home_epa":"epa_allowed"})

    tg = (pd.concat([home, away], ignore_index=True)
          .sort_values(["team","season","week"]).reset_index(drop=True))

    for raw_col, roll_col in [("epa_scored", f"roll{ROLL_N}_epa_scored"),
                               ("epa_allowed", f"roll{ROLL_N}_epa_allowed")]:
        tg[roll_col] = (
            tg.groupby(["team","season"])[raw_col]
            .transform(lambda s: s.shift(1).rolling(ROLL_N, min_periods=1).mean())
        )

    return tg[["game_id","team",
               f"roll{ROLL_N}_epa_scored",
               f"roll{ROLL_N}_epa_allowed"]]


def _rest_days(gl: pd.DataFrame) -> pd.DataFrame:
    home = gl[["game_id","game_date","home_team"]].rename(columns={"home_team":"team"})
    away = gl[["game_id","game_date","away_team"]].rename(columns={"away_team":"team"})
    tg = (pd.concat([home, away], ignore_index=True)
          .sort_values(["team","game_date"]).reset_index(drop=True))
    tg["rest_days"] = tg.groupby("team")["game_date"].transform(
        lambda s: s.diff().dt.days)
    return tg[["game_id","team","rest_days"]]


def build_pregame_features(seasons: list[int] = SEASONS) -> pd.DataFrame:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    out_path = GOLD_DIR / "pregame_features.parquet"

    print(f"[pregame features] loading {len(seasons)} seasons...")
    gl = _load_game_level(seasons)
    print(f"  {len(gl):,} games loaded")

    print("  computing rolling pts + win_pct stats...")
    roll = _rolling_stats(gl)

    print("  computing rolling EPA stats...")
    epa = _epa_stats(seasons)

    print("  computing rest days...")
    rest = _rest_days(gl)

    print("  joining features...")
    stats = (roll
             .merge(epa, on=["game_id","team"], how="left")
             .merge(rest, on=["game_id","team"], how="left"))

    rename_home = {
        "team":                         "home_team_check",
        f"roll{ROLL_N}_pts_scored":     "home_roll4_pts_scored",
        f"roll{ROLL_N}_pts_allowed":    "home_roll4_pts_allowed",
        f"roll{ROLL_N}_win_pct":        "home_roll4_win_pct",
        f"roll{ROLL_N}_epa_scored":     "home_roll4_epa_scored",
        f"roll{ROLL_N}_epa_allowed":    "home_roll4_epa_allowed",
        "rest_days":                    "home_rest_days",
    }
    rename_away = {
        "team":                         "away_team_check",
        f"roll{ROLL_N}_pts_scored":     "away_roll4_pts_scored",
        f"roll{ROLL_N}_pts_allowed":    "away_roll4_pts_allowed",
        f"roll{ROLL_N}_win_pct":        "away_roll4_win_pct",
        f"roll{ROLL_N}_epa_scored":     "away_roll4_epa_scored",
        f"roll{ROLL_N}_epa_allowed":    "away_roll4_epa_allowed",
        "rest_days":                    "away_rest_days",
    }

    home_stats = stats.rename(columns=rename_home)
    away_stats = stats.rename(columns=rename_away)

    df = (gl
          .merge(home_stats, left_on=["game_id","home_team"],
                 right_on=["game_id","home_team_check"], how="left")
          .merge(away_stats, left_on=["game_id","away_team"],
                 right_on=["game_id","away_team_check"], how="left"))

    df = df.drop(columns=["home_team_check","away_team_check"], errors="ignore")

    # Market-implied WP (positive spread_line = home favored)
    df["market_wp"] = norm.cdf(df["spread_line"].astype(float) / NFL_SPREAD_SIGMA)

    # Target: 1 = home won, 0 = away won (exclude ties)
    result_f = df["result"].astype(float)
    df["home_win"] = np.where(result_f > 0, 1.0,
                     np.where(result_f < 0, 0.0, np.nan))

    before = len(df)
    df = df[df["home_win"].notna() & df["spread_line"].notna()].copy()
    print(f"  dropped {before - len(df)} ties/no-line rows; {len(df):,} usable games")

    df.to_parquet(out_path, index=False)
    print(f"  saved -> {out_path.name}  ({len(df.columns)} columns)")
    return df


if __name__ == "__main__":
    build_pregame_features()
