"""Load CSVs and normalize rows for the risk scoring engine."""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd

from .paths import DATA_DIR, ENGINE_DIR

# Import risk engine from sibling package (not installed as pip package)
if str(ENGINE_DIR) not in sys.path:
    sys.path.insert(0, str(ENGINE_DIR))

from risk_scoring_engine import (  # noqa: E402
    compute_risk_from_dataframe_row,
    explain_components_for_display,
)


def _as_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return False
    s = str(v).strip().lower()
    return s in ("true", "1", "yes", "t")


def row_dict_for_engine(row: pd.Series) -> dict[str, Any]:
    d = row.to_dict()
    for k in (
        "is_commuter",
        "works_part_time",
        "has_declared_disability",
        "is_international",
        "accessed_upcoming_assessment_brief",
    ):
        if k in d:
            d[k] = _as_bool(d[k])
    return d


@lru_cache(maxsize=1)
def load_students(path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path) if path else DATA_DIR / "students_data.csv"
    df = pd.read_csv(p)
    return df


@lru_cache(maxsize=1)
def load_interventions(path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path) if path else DATA_DIR / "interventions_data.csv"
    return pd.read_csv(p)


@lru_cache(maxsize=1)
def load_assessments(path: str | Path | None = None) -> pd.DataFrame:
    p = Path(path) if path else DATA_DIR / "assessments_data.csv"
    return pd.read_csv(p)


def enrich_with_recomputed_risk(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns aligned with the live engine for verification and display."""
    levels: list[str] = []
    pre: list[int] = []
    recalc: list[int] = []
    for _, row in df.iterrows():
        rd = row_dict_for_engine(row)
        out = compute_risk_from_dataframe_row(pd.Series(rd))
        recalc.append(int(out["risk_score"]))
        levels.append(str(out["risk_level"]))
        pre.append(int(out["risk_score_pre_intervention"]))
    out_df = df.copy()
    out_df["risk_recomputed"] = recalc
    out_df["risk_level_engine"] = levels
    out_df["risk_pre_intervention_display"] = pre
    return out_df


def component_explanations_for_row(row: pd.Series) -> list[tuple[str, float]]:
    return explain_components_for_display(row_dict_for_engine(row))
