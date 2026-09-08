"""
Silver layer: run quality checks on each bronze file and write the result to
data/silver/. All rows are written regardless of check outcomes — failures are
recorded in the DQ log for visibility, not silently removed here.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from data_pipeline.quality_checks.checks import run_all_checks

BRONZE_DIR = ROOT / "data" / "bronze"
SILVER_DIR = ROOT / "data" / "silver"


def bronze_to_silver(season: int) -> None:
    bronze_path = BRONZE_DIR / f"pbp_{season}.parquet"
    silver_path = SILVER_DIR / f"pbp_{season}.parquet"

    if not bronze_path.exists():
        print(f"  {season}: bronze file not found, skipping")
        return

    if silver_path.exists():
        print(f"  {season}: silver already exists, skipping")
        return

    print(f"  {season}: loading bronze ({bronze_path.name})...")
    df = pd.read_parquet(bronze_path)

    print(f"  {season}: running quality checks...")
    run_all_checks(df, season)

    SILVER_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(silver_path, index=False)
    print(f"  {season}: {len(df):,} rows  ->  {silver_path.name}")


def bronze_to_silver_all(seasons: list[int]) -> None:
    print(f"[bronze → silver] seasons {seasons[0]}–{seasons[-1]}")
    for season in seasons:
        bronze_to_silver(season)


if __name__ == "__main__":
    bronze_to_silver_all(list(range(2016, 2026)))
