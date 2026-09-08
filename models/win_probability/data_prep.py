"""
Load, filter, and split the Gold feature table for the Win Probability model.

Split strategy
--------------
Train : seasons 2016–2022  (7 seasons)
Val   : season  2023       (1 season — used for all tuning decisions)
Test  : seasons 2024–2025  (2 seasons — touched ONLY for final evaluation)

The split is strictly time-ordered and never shuffled, which prevents leakage
and reflects how the model would actually be deployed mid-season.

Filtering
---------
Rows where `down` is null are non-scrimmage plays (kickoffs, extra points,
administrative rows).  They carry no meaningful game-state context for a WP
model, so they are removed before any split or feature selection.

Rows where `posteam_win` is null are also removed (plays with no possession
team that survived the down filter — e.g. some two-point attempts).
"""

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
GOLD_PATH = ROOT / "data" / "gold" / "wp_features.parquet"

# ---------------------------------------------------------------------------
# Feature and target definitions  (single source of truth — imported by
# train.py and evaluate.py so they never drift out of sync)
# ---------------------------------------------------------------------------

FEATURES = [
    # Game state — the core inputs a human analyst would reach for
    "score_differential",
    "game_seconds_remaining",
    "yardline_100",
    "down",
    "ydstogo",
    "posteam_timeouts_remaining",
    "defteam_timeouts_remaining",
    # Rolling team performance — last-4-game averages entering this game
    "posteam_roll4_pts_scored",
    "posteam_roll4_pts_allowed",
    "defteam_roll4_pts_scored",
    "defteam_roll4_pts_allowed",
]

TARGET = "posteam_win"   # 1.0 = posteam wins, 0.0 = loses, 0.5 = tie

TRAIN_SEASONS = list(range(2016, 2023))   # 2016 – 2022
VAL_SEASONS   = [2023]
TEST_SEASONS  = [2024, 2025]


# ---------------------------------------------------------------------------
# Split container
# ---------------------------------------------------------------------------

@dataclass
class DataSplit:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val:   pd.DataFrame
    y_val:   pd.Series
    X_test:  pd.DataFrame
    y_test:  pd.Series


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_and_split(gold_path: Path = GOLD_PATH) -> DataSplit:
    """
    Load the Gold parquet, apply filters, and return the time-based split.
    Test data is included in the returned object but should not be inspected
    until final evaluation (see evaluate.py).
    """
    df = pd.read_parquet(gold_path)
    print(f"Loaded: {len(df):,} rows, {len(df.columns)} columns")

    # --- filter ---
    before = len(df)
    df = df[df["down"].notna() & df[TARGET].notna()].copy()
    print(f"After filter (non-null down + {TARGET}): {len(df):,} rows "
          f"({before - len(df):,} removed)")

    # Sanity-check that every requested feature is present
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Features missing from Gold table: {missing}")

    # --- split ---
    train = df[df["season"].isin(TRAIN_SEASONS)]
    val   = df[df["season"].isin(VAL_SEASONS)]
    test  = df[df["season"].isin(TEST_SEASONS)]

    _print_split_stats(train, val, test)

    return DataSplit(
        X_train = train[FEATURES].reset_index(drop=True),
        y_train = train[TARGET].reset_index(drop=True),
        X_val   = val[FEATURES].reset_index(drop=True),
        y_val   = val[TARGET].reset_index(drop=True),
        X_test  = test[FEATURES].reset_index(drop=True),
        y_test  = test[TARGET].reset_index(drop=True),
    )


def _print_split_stats(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> None:
    rows = [
        ("train", train),
        ("val",   val),
        ("test",  test),
    ]
    print(f"\n{'split':<8} {'seasons':<20} {'rows':>8} {'win_rate':>9} {'null_features':>14}")
    print("-" * 65)
    for name, split in rows:
        if split.empty:
            seasons_str = "—"
            print(f"{name:<8} {seasons_str:<20} {'(empty)':>8}")
            continue
        seasons_str = f"{split['season'].min()}–{split['season'].max()}"
        n_rows      = len(split)
        win_rate    = split[TARGET].mean()
        # Count rows where ANY feature is null (worth knowing before training)
        null_rows   = split[FEATURES].isnull().any(axis=1).sum()
        print(f"{name:<8} {seasons_str:<20} {n_rows:>8,} {win_rate:>9.3f} {null_rows:>14,}")
    print()
