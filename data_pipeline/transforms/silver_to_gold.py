"""
Gold layer: build the win-probability feature table from silver data.

Features added on top of raw game-state columns:

  Rolling team performance (4-game, within-season, shift(1)):
    - points scored / allowed
    - total EPA earned / conceded

  Rest days: calendar days since the team's previous game.
    First game of a season gets NaN (XGBoost handles natively).

  is_home: 1 if posteam is the home team, 0 otherwise.

  posteam_win label: derived from nflfastR's `result` column.

Output: data/gold/wp_features.parquet  (one row per play, all seasons combined)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SILVER_DIR = ROOT / "data" / "silver"
GOLD_DIR   = ROOT / "data" / "gold"

ROLL_N = 4

GAME_STATE_COLS = [
    "play_id",
    "game_id",
    "season",
    "week",
    "posteam",
    "defteam",
    "home_team",
    "away_team",
    "score_differential",
    "game_seconds_remaining",
    "yardline_100",
    "down",
    "ydstogo",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    "spread_line",
    "total_line",
    "vegas_wp",
]

HELPER_COLS = [
    "game_date",
    "total_home_score",
    "total_away_score",
    "total_home_epa",
    "total_away_epa",
    "result",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_silver(seasons: list[int]) -> pd.DataFrame:
    frames = []
    for season in seasons:
        path = SILVER_DIR / f"pbp_{season}.parquet"
        if not path.exists():
            print(f"  warning: {path.name} not found, skipping season {season}")
            continue
        frames.append(pd.read_parquet(path))
    return pd.concat(frames, ignore_index=True)


def _game_level(df: pd.DataFrame) -> pd.DataFrame:
    """One row per game: final scores, cumulative EPA, and game date."""
    return (
        df.groupby("game_id")
        .agg(
            season     = ("season",          "first"),
            week       = ("week",            "first"),
            home_team  = ("home_team",       "first"),
            away_team  = ("away_team",       "first"),
            game_date  = ("game_date",       "first"),
            home_final = ("total_home_score","max"),
            away_final = ("total_away_score","max"),
            home_epa   = ("total_home_epa",  "max"),
            away_epa   = ("total_away_epa",  "max"),
        )
        .reset_index()
    )


def _rolling_team_stats(game_level: pd.DataFrame) -> pd.DataFrame:
    """
    (game_id, team) table with rolling ROLL_N-game averages of pts and EPA,
    using only games that came before the current one (shift(1)).
    Window resets each season so week-1 stats are always NaN.
    """
    home = game_level[["game_id","season","week","home_team",
                        "home_final","away_final","home_epa","away_epa"]].rename(
        columns={"home_team":"team","home_final":"pts_scored",
                 "away_final":"pts_allowed","home_epa":"epa_scored","away_epa":"epa_allowed"})
    away = game_level[["game_id","season","week","away_team",
                        "away_final","home_final","away_epa","home_epa"]].rename(
        columns={"away_team":"team","away_final":"pts_scored",
                 "home_final":"pts_allowed","away_epa":"epa_scored","home_epa":"epa_allowed"})

    tg = (pd.concat([home, away], ignore_index=True)
          .sort_values(["team","season","week"])
          .reset_index(drop=True))

    for raw, col in [("pts_scored",  f"roll{ROLL_N}_pts_scored"),
                     ("pts_allowed", f"roll{ROLL_N}_pts_allowed"),
                     ("epa_scored",  f"roll{ROLL_N}_epa_scored"),
                     ("epa_allowed", f"roll{ROLL_N}_epa_allowed")]:
        tg[col] = (tg.groupby(["team","season"])[raw]
                   .transform(lambda s: s.shift(1).rolling(ROLL_N, min_periods=1).mean()))

    return tg[["game_id","team",
               f"roll{ROLL_N}_pts_scored", f"roll{ROLL_N}_pts_allowed",
               f"roll{ROLL_N}_epa_scored", f"roll{ROLL_N}_epa_allowed"]]


def _rest_days(game_level: pd.DataFrame) -> pd.DataFrame:
    """
    (game_id, team) table with calendar days since the team's previous game.
    Computed across the full dataset (no season reset) so the first game of
    a new season reflects true offseason rest. First-ever game per team = NaN.
    """
    home = game_level[["game_id","game_date","home_team"]].rename(
        columns={"home_team":"team"})
    away = game_level[["game_id","game_date","away_team"]].rename(
        columns={"away_team":"team"})

    tg = (pd.concat([home, away], ignore_index=True)
          .sort_values(["team","game_date"])
          .reset_index(drop=True))

    tg["game_date"] = pd.to_datetime(tg["game_date"])
    tg["rest_days"] = (tg.groupby("team")["game_date"]
                       .transform(lambda s: s.diff().dt.days))

    return tg[["game_id","team","rest_days"]]


def _join_team_features(df: pd.DataFrame,
                        team_stats: pd.DataFrame,
                        rest: pd.DataFrame) -> pd.DataFrame:
    """Join rolling stats and rest days for posteam and defteam."""
    combined = team_stats.merge(rest, on=["game_id","team"])

    for side in ("posteam", "defteam"):
        rename = {
            "team":                         side,
            f"roll{ROLL_N}_pts_scored":     f"{side}_roll4_pts_scored",
            f"roll{ROLL_N}_pts_allowed":    f"{side}_roll4_pts_allowed",
            f"roll{ROLL_N}_epa_scored":     f"{side}_roll4_epa_scored",
            f"roll{ROLL_N}_epa_allowed":    f"{side}_roll4_epa_allowed",
            "rest_days":                    f"{side}_rest_days",
        }
        df = df.merge(combined.rename(columns=rename),
                      on=["game_id", side], how="left")
    return df


def _add_is_home(df: pd.DataFrame) -> pd.DataFrame:
    df["is_home"] = (df["posteam"] == df["home_team"]).astype("float32")
    df.loc[df["posteam"].isna(), "is_home"] = np.nan
    return df


def _add_posteam_win(df: pd.DataFrame) -> pd.DataFrame:
    if "result" not in df.columns:
        print("  warning: 'result' column not found")
        return df

    result_f   = df["result"].astype("float64")
    home_mask  = (df["posteam"] == df["home_team"]).to_numpy()
    posteam_margin = np.where(home_mask, result_f.to_numpy(), -result_f.to_numpy())
    win = np.where(posteam_margin > 0, 1.0,
           np.where(posteam_margin < 0, 0.0, 0.5))
    df["posteam_win"] = win
    df.loc[df["posteam"].isna(), "posteam_win"] = np.nan

    print(f"    posteam_win: {(df['posteam_win']==1.0).sum():,} wins, "
          f"{(df['posteam_win']==0.0).sum():,} losses, "
          f"{(df['posteam_win']==0.5).sum():,} ties, "
          f"{df['posteam_win'].isna().sum():,} nulls")
    return df


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def silver_to_gold(seasons: list[int]) -> None:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    gold_path = GOLD_DIR / "wp_features.parquet"

    print(f"[silver -> gold] loading {len(seasons)} seasons...")
    df = _load_silver(seasons)
    print(f"  {len(df):,} total rows loaded")

    keep = [c for c in GAME_STATE_COLS + HELPER_COLS if c in df.columns]
    df = df[keep].copy()

    print("  computing game-level stats...")
    gl = _game_level(df)

    print(f"  computing rolling {ROLL_N}-game team stats (pts + EPA)...")
    team_stats = _rolling_team_stats(gl)

    print("  computing rest days...")
    rest = _rest_days(gl)

    print("  joining features to play level...")
    df = _join_team_features(df, team_stats, rest)

    print("  adding is_home and posteam_win...")
    df = _add_is_home(df)
    df = _add_posteam_win(df)

    df = df.drop(columns=[c for c in HELPER_COLS if c in df.columns], errors="ignore")

    df.to_parquet(gold_path, index=False)
    print(f"  {len(df):,} rows, {len(df.columns)} columns  ->  {gold_path.name}")


if __name__ == "__main__":
    silver_to_gold(list(range(2016, 2026)))
