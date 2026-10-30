"""Input validation shared by the data, feature and pipeline modules."""

from __future__ import annotations

from typing import Iterable

import pandas as pd

from utils.config import FREQUENCIES


def require_frequency(frequency: str, choices: Iterable[str] = FREQUENCIES) -> str:
    """Return ``frequency`` if supported, otherwise raise.

    Raises:
        ValueError: if ``frequency`` is not one of ``choices``.
    """
    choices = tuple(choices)
    if frequency not in choices:
        raise ValueError(f"frequency must be one of {choices}, got {frequency!r}")
    return frequency


def require_columns(frame: pd.DataFrame, columns: Iterable[str], source: str = "dataframe") -> None:
    """Raise if any of ``columns`` is missing from ``frame``.

    Raises:
        ValueError: listing the missing columns and their source.
    """
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing columns: {missing}")


def require_panel(frame: pd.DataFrame, levels: tuple[str, str] = ("date", "ticker")) -> None:
    """Raise unless ``frame`` is a MultiIndex panel with the expected level names.

    Raises:
        ValueError: if the index is not a MultiIndex with ``levels``.
    """
    names = tuple(frame.index.names) if isinstance(frame.index, pd.MultiIndex) else ()
    if names != levels:
        raise ValueError(f"expected a MultiIndex named {levels}, got {list(names)}")
