"""Train / validation / test splits, by calendar period or by fixed date range."""

from __future__ import annotations

from datetime import timedelta

import pandas as pd

from utils.config import duration_to_offset


def build_splits(panel: pd.DataFrame, test_start: str, test_end: str,
                 window_val: str = "3y") -> dict[str, tuple[str, str]]:
    """Derive ``train`` / ``val`` / ``test`` date ranges.

    ``val`` is the ``window_val`` span immediately before ``test_start`` and ``train`` is
    everything earlier; the rolling training window caps how much of it each refit uses.

    Args:
        panel: Any panel with a ``date`` index level (only its first date is used).
        test_start: First reported date, e.g. ``"2023-01-01"``.
        test_end: Last reported date, e.g. ``"2025-12-31"``.
        window_val: Validation window, e.g. ``"3y"``.

    Returns:
        ``{"train"|"val"|"test": (start, end)}`` as ISO date strings.
    """
    start = pd.Timestamp(test_start)
    val_start = start - duration_to_offset(window_val)
    one_day = timedelta(days=1)
    first = panel.index.get_level_values("date").min()
    return {
        "train": (f"{first:%Y-%m-%d}", f"{val_start - one_day:%Y-%m-%d}"),
        "val": (f"{val_start:%Y-%m-%d}", f"{start - one_day:%Y-%m-%d}"),
        "test": (f"{start:%Y-%m-%d}", f"{pd.Timestamp(test_end):%Y-%m-%d}"),
    }


def split_panel(panel: pd.DataFrame,
                windows: dict[str, tuple[str, str]]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Cut the panel into non-overlapping train / validation / test frames.

    Args:
        panel: Panel indexed by ``(date, ticker)``.
        windows: Output of :func:`build_splits`.

    Returns:
        Tuple ``(train, val, test)`` with the same columns.

    Raises:
        ValueError: if any window selects no rows.
    """
    dates = panel.index.get_level_values("date")
    parts = []
    for name in ("train", "val", "test"):
        start, end = windows[name]
        part = panel[(dates >= start) & (dates <= end)]
        if part.empty:
            raise ValueError(f"no rows in the {name} window {start}..{end}; widen the data range")
        parts.append(part)
    return tuple(parts)
