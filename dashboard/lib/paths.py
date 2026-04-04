"""Resolve project paths regardless of cwd."""

from __future__ import annotations

from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = DASHBOARD_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
ENGINE_DIR = PROJECT_ROOT / "risk-scoring-engine"
