"""
Bronze layer: download raw PBP data from nflreadpy and write one Parquet
file per season to data/bronze/. Files are written exactly as received —
no cleaning, no transforms.
"""

import sys
from pathlib import Path

import nflreadpy

ROOT = Path(__file__).resolve().parents[2]
BRONZE_DIR = ROOT / "data" / "bronze"


def ingest_season(season: int) -> None:
    """Download one season and write to bronze. Skips if file already exists."""
    BRONZE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = BRONZE_DIR / f"pbp_{season}.parquet"

    if out_path.exists():
        print(f"  {season}: already in bronze, skipping")
        return

    print(f"  {season}: downloading...")
    df = nflreadpy.load_pbp([season]).to_pandas()
    df.to_parquet(out_path, index=False)
    print(f"  {season}: {len(df):,} rows  ->  {out_path.name}")


def ingest_all(seasons: list[int]) -> None:
    print(f"[ingest] seasons {seasons[0]}–{seasons[-1]}")
    for season in seasons:
        ingest_season(season)


if __name__ == "__main__":
    ingest_all(list(range(2016, 2026)))
