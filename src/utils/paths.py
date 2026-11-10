"""Filesystem locations used by this project."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FEATURES_DIR = DATA_DIR / "features"
ADDITIONAL_DIR = DATA_DIR / "additional"

# CNN Fear & Greed index (daily, 2011+), supplied alongside the raw data.
FEAR_GREED_PATH = ADDITIONAL_DIR / "fear-greed.csv"

# Backtest results (committed): one folder per run tag.
RESULTS_DIR = PROJECT_ROOT / "results"

# Validation study (committed): the joint-ranking outputs and the walk-forward
# prediction cache that the ranking replays instead of refitting.
STUDY_DIR = RESULTS_DIR / "_study"
PREDICTIONS_DIR = STUDY_DIR / "predictions"


def raw_data_dir(frequency: str) -> Path:
    """Return the folder holding the raw ticker CSVs for one frequency.

    Args:
        frequency: ``"daily"`` or ``"weekly"``.

    Returns:
        Path to ``data/raw/<frequency>``.
    """
    return RAW_DIR / frequency
