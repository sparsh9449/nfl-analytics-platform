"""
Run the full Bronze → Silver → Gold pipeline for NFL play-by-play data.

Usage (from the project root):
    python data_pipeline/run_pipeline.py

Layer summary
─────────────
Bronze  data/bronze/pbp_<season>.parquet  — raw download, untouched
Silver  data/silver/pbp_<season>.parquet  — same rows, quality-checked
Gold    data/gold/wp_features.parquet     — game-state + rolling features, all seasons

Quality log: data/quality_log/dq_results.parquet  (one row per check per season)
Schema ref:  data/schemas/pbp_reference_schema.json  (saved on first 2025 run)
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from data_pipeline.ingest.ingest_pbp import ingest_all
from data_pipeline.transforms.bronze_to_silver import bronze_to_silver_all
from data_pipeline.transforms.silver_to_gold import silver_to_gold

SEASONS = list(range(2016, 2026))


def main() -> None:
    print("=" * 60)
    print("NFL PBP Pipeline  —  Bronze → Silver → Gold")
    print(f"Seasons: {SEASONS[0]}–{SEASONS[-1]}")
    print("=" * 60)

    print("\n[1/3] Ingest — raw download to bronze")
    ingest_all(SEASONS)

    print("\n[2/3] Quality checks + promote to silver")
    # 2025 must run before earlier seasons so the reference schema is saved first.
    ordered = sorted(SEASONS, reverse=True)
    bronze_to_silver_all(ordered)

    print("\n[3/3] Feature engineering — silver to gold")
    silver_to_gold(SEASONS)

    print("\n" + "=" * 60)
    print("Pipeline complete. Output locations:")
    print("  Bronze:      data/bronze/pbp_<season>.parquet")
    print("  Silver:      data/silver/pbp_<season>.parquet")
    print("  Gold:        data/gold/wp_features.parquet")
    print("  Quality log: data/quality_log/dq_results.parquet")
    print("  Schema ref:  data/schemas/pbp_reference_schema.json")
    print("=" * 60)


if __name__ == "__main__":
    main()
