"""Filesystem locations used by every module."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "config.yaml"
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
FEATURES_DIR = DATA_DIR / "features"
ALTERNATIVE_DIR = DATA_DIR / "alternative"
RESULTS_DIR = PROJECT_ROOT / "results"
FEAR_GREED_PATH = ALTERNATIVE_DIR / "fear-greed.csv"


def raw_data_dir(frequency: str) -> Path:
    """Return the folder holding the raw ticker CSVs for one frequency.

    Args:
        frequency: ``"daily"`` or ``"weekly"``.

    Returns:
        Path to ``data/raw/<frequency>``.
    """
    return RAW_DIR / frequency
