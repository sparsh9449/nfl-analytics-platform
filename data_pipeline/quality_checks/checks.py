"""
Quality checks for NFL PBP data.

Each check returns a CheckResult (pass/fail + details).
run_all_checks() runs all three and logs every result to the DQ parquet log.

Reference schema is the 2025 season — saved automatically the first time
that season is processed, then used for drift comparisons on all other years.
"""

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DQ_LOG_PATH  = ROOT / "data" / "quality_log" / "dq_results.parquet"
SCHEMA_PATH  = ROOT / "data" / "schemas" / "pbp_reference_schema.json"
REFERENCE_SEASON = 2025


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class CheckResult:
    check_name: str
    passed: bool
    affected_row_count: int
    details: str


# ---------------------------------------------------------------------------
# Check 1 — Completeness
# ---------------------------------------------------------------------------

def run_completeness_check(df: pd.DataFrame, season: int) -> CheckResult:
    """
    Within each game_id, sort play_ids and look for gaps (diff > 1).
    A gap means at least one play_id was skipped, suggesting missing rows.
    """
    games_with_gaps = []
    total_gaps = 0

    for game_id, group in df.groupby("game_id"):
        ids = group["play_id"].dropna().sort_values().astype(int)
        if len(ids) < 2:
            continue
        gaps = ids.diff().dropna()
        gaps = gaps[gaps > 1]
        if not gaps.empty:
            games_with_gaps.append(game_id)
            total_gaps += len(gaps)

    passed = len(games_with_gaps) == 0
    details = (
        "no play_id gaps found"
        if passed
        else (
            f"{len(games_with_gaps)} games have play_id gaps "
            f"({total_gaps} gap points total)"
        )
    )
    return CheckResult(
        check_name="completeness",
        passed=passed,
        affected_row_count=len(games_with_gaps),
        details=details,
    )


# ---------------------------------------------------------------------------
# Check 2 — Validity
# ---------------------------------------------------------------------------

def run_validity_check(df: pd.DataFrame, season: int) -> CheckResult:
    """
    Flag rows with field values that are physically impossible:
      - down not in {1, 2, 3, 4} (when not null)
      - ydstogo < 0
      - yardline_100 outside [0, 100]
      - score_differential inconsistent with total_home_score / total_away_score
    """
    bad = pd.Series(False, index=df.index)

    if "down" in df.columns:
        bad |= df["down"].notna() & ~df["down"].isin([1, 2, 3, 4])

    if "ydstogo" in df.columns:
        bad |= df["ydstogo"].notna() & (df["ydstogo"] < 0)

    if "yardline_100" in df.columns:
        bad |= df["yardline_100"].notna() & (
            (df["yardline_100"] < 0) | (df["yardline_100"] > 100)
        )

    score_cols = {"score_differential", "total_home_score", "total_away_score",
                  "posteam", "home_team"}
    if score_cols.issubset(df.columns):
        # score_differential = posteam_score - defteam_score
        # when posteam is home: expected = total_home - total_away, and vice-versa
        home_mask = df["posteam"] == df["home_team"]
        expected = pd.Series(np.nan, index=df.index, dtype="float64")
        expected[home_mask]  = df.loc[home_mask,  "total_home_score"] - df.loc[home_mask,  "total_away_score"]
        expected[~home_mask] = df.loc[~home_mask, "total_away_score"] - df.loc[~home_mask, "total_home_score"]
        bad |= df["score_differential"].notna() & expected.notna() & (df["score_differential"] != expected)

    n_bad = int(bad.sum())
    passed = n_bad == 0
    details = "no invalid rows found" if passed else f"{n_bad} rows with impossible field values"
    return CheckResult(
        check_name="validity",
        passed=passed,
        affected_row_count=n_bad,
        details=details,
    )


# ---------------------------------------------------------------------------
# Check 3 — Schema drift
# ---------------------------------------------------------------------------

def _save_reference_schema(df: pd.DataFrame) -> None:
    SCHEMA_PATH.parent.mkdir(parents=True, exist_ok=True)
    schema = {
        "reference_season": REFERENCE_SEASON,
        "columns": list(df.columns),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }
    SCHEMA_PATH.write_text(json.dumps(schema, indent=2))
    print(f"  saved reference schema ({len(schema['columns'])} columns) -> {SCHEMA_PATH.name}")


def run_schema_drift_check(df: pd.DataFrame, season: int) -> CheckResult:
    """
    Compare the current season's columns and dtypes against the saved 2025
    reference. Creates the reference file on the first run of season 2025.
    """
    if season == REFERENCE_SEASON:
        _save_reference_schema(df)
        return CheckResult(
            check_name="schema_drift",
            passed=True,
            affected_row_count=0,
            details=f"season {REFERENCE_SEASON} is the reference — schema saved",
        )

    if not SCHEMA_PATH.exists():
        return CheckResult(
            check_name="schema_drift",
            passed=False,
            affected_row_count=0,
            details="reference schema not found; process season 2025 first",
        )

    ref = json.loads(SCHEMA_PATH.read_text())
    ref_cols = set(ref["columns"])
    cur_cols = set(df.columns)

    added        = sorted(cur_cols - ref_cols)
    dropped      = sorted(ref_cols - cur_cols)
    dtype_changes = [
        f"{col}: {ref['dtypes'][col]} -> {df.dtypes[col]}"
        for col in sorted(ref_cols & cur_cols)
        if ref["dtypes"].get(col) != str(df.dtypes[col])
    ]

    passed = not added and not dropped and not dtype_changes
    parts = []
    if added:
        parts.append(f"added columns: {added}")
    if dropped:
        parts.append(f"dropped columns: {dropped}")
    if dtype_changes:
        parts.append(f"dtype changes: {dtype_changes}")

    return CheckResult(
        check_name="schema_drift",
        passed=passed,
        affected_row_count=len(added) + len(dropped) + len(dtype_changes),
        details="schema matches reference" if passed else "; ".join(parts),
    )


# ---------------------------------------------------------------------------
# DQ log
# ---------------------------------------------------------------------------

def _log_result(result: CheckResult, season: int) -> None:
    """Append one row to the persistent DQ log parquet."""
    DQ_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_row = pd.DataFrame([{
        "timestamp":          datetime.now(timezone.utc),
        "season":             season,
        "check_name":         result.check_name,
        "passed":             result.passed,
        "affected_row_count": result.affected_row_count,
        "details":            result.details,
    }])

    if DQ_LOG_PATH.exists():
        combined = pd.concat([pd.read_parquet(DQ_LOG_PATH), new_row], ignore_index=True)
    else:
        combined = new_row

    combined.to_parquet(DQ_LOG_PATH, index=False)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_all_checks(df: pd.DataFrame, season: int) -> list[CheckResult]:
    """Run all three checks, print results, and append each to the DQ log."""
    results = [
        run_completeness_check(df, season),
        run_validity_check(df, season),
        run_schema_drift_check(df, season),
    ]
    for r in results:
        _log_result(r, season)
        status = "PASS" if r.passed else "FAIL"
        print(f"    [{status}] {r.check_name}: {r.details}")
    return results
