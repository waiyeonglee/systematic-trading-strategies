"""Input validation shared by the data and feature modules."""

from __future__ import annotations

from typing import Iterable

import pandas as pd


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: str = "dataframe") -> None:
    """Raise if any of ``columns`` is missing from ``frame``.

    Raises:
        ValueError: listing the missing columns and their source.
    """
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def require_panel(frame: pd.DataFrame, source: str = "dataframe") -> None:
    """Raise unless ``frame`` is a panel indexed by ``(date, ticker)``.

    Raises:
        ValueError: if either index level is missing or the index is not unique.
    """
    missing = sorted({"date", "ticker"} - set(frame.index.names))
    if missing:
        raise ValueError(f"{source} must be a panel indexed by (date, ticker); missing {missing}")
    if not frame.index.is_unique:
        raise ValueError(f"{source} has duplicate (date, ticker) rows")

