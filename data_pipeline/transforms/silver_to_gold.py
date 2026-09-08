"""
Gold layer: build the win-probability feature table from silver data.

Two kinds of features are added on top of the raw game-state columns:

  Rolling team performance  —  for each team going into a game, the average
      points scored and allowed over their previous ROLL_N games. Computed at
      game level (not play level) and then joined back to plays. shift(1) is
      applied before the rolling window so that the current game's score is
      never included in its own look-back average.

  posteam_win label  —  derived from nflfastR's `result` column, which is
      home_final_score minus away_final_score. A positive result is a home win.

Output: data/gold/wp_features.parquet  (one row per play, all seasons combined)
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SILVER_DIR = ROOT / "data" / "silver"
GOLD_DIR   = ROOT / "data" / "gold"

ROLL_N = 4  # look-back window for rolling team stats

# Columns carried through to the final feature table
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
    # Betting market data — game-level constants repeated on every play
    "spread_line",   # home_team spread (negative = home favored)
    "total_line",    # over/under total
    "vegas_wp",      # nflfastR market-implied WP for the home team, per play
]

# Needed temporarily to build rolling stats and the win label; dropped at the end
HELPER_COLS = ["total_home_score", "total_away_score", "result"]


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


def _game_final_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse play-level data to one row per game with the final score for
    each side. max() on total_home/away_score gives the final tally because
    scores are monotonically non-decreasing through a game.
    """
    return (
        df.groupby("game_id")
        .agg(
            season    = ("season",           "first"),
            week      = ("week",             "first"),
            home_team = ("home_team",        "first"),
            away_team = ("away_team",        "first"),
            home_final= ("total_home_score", "max"),
            away_final= ("total_away_score", "max"),
        )
        .reset_index()
    )


def _rolling_team_stats(game_scores: pd.DataFrame) -> pd.DataFrame:
    """
    Return a (game_id, team) table with rolling ROLL_N-game averages of
    points scored and allowed, using only games that came *before* the
    current one (shift(1) on the sorted sequence).

    Grouping by (team, season) means the window resets at the start of each
    season, so week-1 rolling stats are always NaN (no prior games that year).
    Weeks 2-4 use 1-3 games; week 5+ uses the full ROLL_N-game window.
    """
    # One row per team per game from the home perspective
    home = game_scores[["game_id", "season", "week", "home_team", "home_final", "away_final"]].rename(
        columns={"home_team": "team", "home_final": "pts_scored", "away_final": "pts_allowed"}
    )
    # One row per team per game from the away perspective
    away = game_scores[["game_id", "season", "week", "away_team", "away_final", "home_final"]].rename(
        columns={"away_team": "team", "away_final": "pts_scored", "home_final": "pts_allowed"}
    )

    team_games = (
        pd.concat([home, away], ignore_index=True)
        .sort_values(["team", "season", "week"])
        .reset_index(drop=True)
    )

    for raw_col, roll_col in [("pts_scored", f"roll{ROLL_N}_pts_scored"),
                               ("pts_allowed", f"roll{ROLL_N}_pts_allowed")]:
        team_games[roll_col] = (
            team_games
            .groupby(["team", "season"])[raw_col]
            .transform(lambda s: s.shift(1).rolling(ROLL_N, min_periods=1).mean())
        )

    return team_games[["game_id", "team", f"roll{ROLL_N}_pts_scored", f"roll{ROLL_N}_pts_allowed"]]


def _join_rolling_stats(df: pd.DataFrame, team_stats: pd.DataFrame) -> pd.DataFrame:
    """Join rolling stats for the possession team and then the defensive team."""
    # posteam stats
    df = df.merge(
        team_stats.rename(columns={
            "team":                     "posteam",
            f"roll{ROLL_N}_pts_scored": "posteam_roll4_pts_scored",
            f"roll{ROLL_N}_pts_allowed":"posteam_roll4_pts_allowed",
        }),
        on=["game_id", "posteam"],
        how="left",
    )
    # defteam stats
    df = df.merge(
        team_stats.rename(columns={
            "team":                     "defteam",
            f"roll{ROLL_N}_pts_scored": "defteam_roll4_pts_scored",
            f"roll{ROLL_N}_pts_allowed":"defteam_roll4_pts_allowed",
        }),
        on=["game_id", "defteam"],
        how="left",
    )
    return df


def _add_posteam_win(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute a binary win label for the possession team.
    `result` = home_final - away_final (nflfastR convention, int32, no nulls).
    Ties are assigned 0.5 so the label can be used for regression or
    rounded to 0/1 for binary classification.
    Plays with no possession team (e.g. kickoffs) get NaN.
    """
    if "result" not in df.columns:
        print("  warning: 'result' column not found — posteam_win not computed")
        return df

    # Cast to float64 explicitly: parquet round-trips or multi-season concat can
    # silently change int32 to pandas Int32 (nullable), which makes np.where
    # output NaN wherever the nullable mask has <NA>.
    result_f = df["result"].astype("float64")

    home_mask = (df["posteam"] == df["home_team"]).to_numpy()
    posteam_margin = np.where(home_mask, result_f.to_numpy(), -result_f.to_numpy())

    win = np.where(posteam_margin > 0, 1.0, np.where(posteam_margin < 0, 0.0, 0.5))
    df["posteam_win"] = win

    # Plays without a possession team are not real scrimmage plays; null their label.
    df.loc[df["posteam"].isna(), "posteam_win"] = np.nan

    null_count = df["posteam_win"].isna().sum()
    print(f"    posteam_win: {(df['posteam_win'] == 1.0).sum():,} wins, "
          f"{(df['posteam_win'] == 0.0).sum():,} losses, "
          f"{(df['posteam_win'] == 0.5).sum():,} ties, "
          f"{null_count:,} nulls (no-possession plays)")
    return df


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def silver_to_gold(seasons: list[int]) -> None:
    GOLD_DIR.mkdir(parents=True, exist_ok=True)
    gold_path = GOLD_DIR / "wp_features.parquet"

    print(f"[silver → gold] loading {len(seasons)} seasons...")
    df = _load_silver(seasons)
    print(f"  {len(df):,} total rows loaded")

    # Narrow to only the columns we need (game-state + helpers for rolling/label)
    keep = [c for c in GAME_STATE_COLS + HELPER_COLS if c in df.columns]
    df = df[keep].copy()

    print("  computing game-level final scores...")
    game_scores = _game_final_scores(df)

    print(f"  computing rolling {ROLL_N}-game team stats...")
    team_stats = _rolling_team_stats(game_scores)

    print("  joining rolling stats to play level...")
    df = _join_rolling_stats(df, team_stats)

    print("  computing posteam_win label...")
    df = _add_posteam_win(df)

    # Drop the temporary helper columns
    df = df.drop(columns=[c for c in HELPER_COLS if c in df.columns], errors="ignore")

    df.to_parquet(gold_path, index=False)
    print(f"  {len(df):,} rows, {len(df.columns)} columns  ->  {gold_path.name}")


if __name__ == "__main__":
    silver_to_gold(list(range(2016, 2026)))
