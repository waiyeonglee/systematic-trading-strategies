"""Train / validation / test splits, by calendar period or by fixed date range."""

from __future__ import annotations

import re
from datetime import timedelta
from typing import Any

import numpy as np
import pandas as pd

from utils.config import duration_to_offset


def build_splits(panel: pd.DataFrame, test_start: str, test_end: str,
                 window_val: str = "3y") -> dict[str, tuple[str, str]]:
    """Derive ``train`` / ``val`` / ``test`` date ranges.

    ``val`` is the ``window_val`` span immediately before ``test_start`` and ``train`` is
    everything earlier; the rolling training window caps how much of it each refit uses.

    Args:
        panel: Any panel with a ``date`` index level (only its first date is used).
        test_start: First reported date, e.g. ``"2025-01-01"``.
        test_end: Last reported date, e.g. ``"2026-09-30"``.
        window_val: Validation window, e.g. ``"3y"``, ``"9m"``, ``"2y 3m 4w"``.

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


def _as_timestamp(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    """A naive timestamp, so it can be compared with a series whose tz was dropped."""
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    return stamp.tz_localize(None) if stamp.tz is not None else stamp


def _label(period: pd.Period, freq: str) -> str:
    """A short, filename-safe name for one period; weeks use ISO year and week."""
    return period.strftime("%G-W%V") if freq.upper().startswith("W") else str(period)


def make_windows(
    dates: Any,
    *,
    freq: str,
    train_periods: int,
    val_periods: int,
    test_periods: int,
    horizon: int,
    embargo: int,
    first_test: str | pd.Timestamp | None = None,
    last_test: str | pd.Timestamp | None = None,
    max_windows: int | None = None,
) -> list[dict]:
    """Cut dated rows into sliding ``[train) [val) [test)`` windows.

    Every window trains on exactly ``train_periods`` calendar periods, so a difference
    between windows is not confounded by sample size. The last ``embargo`` bars of the
    train and val blocks are dropped, so no label there is realised inside the block
    that follows.

    Args:
        dates: Datetimes, ``DatetimeIndex`` or Series, one per row, sorted ascending.
        freq: pandas period alias, e.g. ``"D"``, ``"W"``, ``"M"``, ``"Q"``, ``"Y"``.
        train_periods: Training block length in periods.
        val_periods: Validation block length in periods.
        test_periods: Test block length in periods; consecutive windows slide by this much.
        horizon: Forecast horizon in bars; the label at ``t`` is realised at ``t + horizon``.
        embargo: Bars dropped from the end of train and val; must be at least ``horizon``.
        first_test: Earliest allowed test start, or ``None``.
        last_test: Latest allowed test end, or ``None``.
        max_windows: Keep only the first ``max_windows`` windows.

    Returns:
        One dict per window with ``window``, ``slug``, ``train``/``val``/``test`` slices
        and ``bounds`` mapping each split to its first and last timestamp.

    Raises:
        ValueError: on non-positive period counts, an embargo shorter than the horizon,
            reversed date bounds, or unsorted input.
    """
    for name, value in (("train_periods", train_periods), ("val_periods", val_periods),
                        ("test_periods", test_periods)):
        if value < 1:
            raise ValueError(f"{name} must be >= 1")
    if horizon < 1:
        raise ValueError("horizon must be >= 1 bar")
    if embargo < horizon:
        raise ValueError(
            f"embargo={embargo} is shorter than horizon={horizon}: "
            "the last training labels would be realised inside the validation block"
        )
    if max_windows is not None and max_windows < 1:
        raise ValueError("max_windows must be >= 1 or None")
    lower, upper = _as_timestamp(first_test), _as_timestamp(last_test)
    if lower is not None and upper is not None and lower > upper:
        raise ValueError("first_test must not be after last_test")

    stamps = pd.DatetimeIndex(pd.to_datetime(dates))
    if len(stamps) == 0:
        return []
    if not stamps.is_monotonic_increasing:
        raise ValueError("dates must be sorted ascending (oldest first)")
    if stamps.tz is not None:
        stamps = stamps.tz_localize(None)

    try:
        periods = stamps.to_period(freq)
    except (ValueError, KeyError) as exc:
        raise ValueError(
            f"unsupported freq {freq!r}; use a pandas period alias such as 'D', 'W', 'M', 'Q' or 'Y'"
        ) from exc

    unique = periods.unique().sort_values()
    rank = unique.get_indexer(periods)
    rows = np.arange(len(stamps))

    windows = []
    for start in range(train_periods + val_periods, len(unique) - test_periods + 1, test_periods):
        train = rows[(rank >= start - val_periods - train_periods) & (rank <= start - val_periods - 1)]
        val = rows[(rank >= start - val_periods) & (rank <= start - 1)]
        test = rows[(rank >= start) & (rank <= start + test_periods - 1)]
        if min(len(train), len(val)) <= embargo or len(test) == 0:
            continue
        if lower is not None and stamps[test[0]] < lower:
            continue
        if upper is not None and stamps[test[-1]] > upper:
            continue

        train, val = train[: len(train) - embargo], val[: len(val) - embargo]
        first_label = _label(unique[start], freq)
        name = first_label if test_periods == 1 else f"{first_label}_{_label(unique[start + test_periods - 1], freq)}"
        windows.append(
            {
                "window": name,
                "slug": re.sub(r"[^0-9A-Za-z]+", "-", name).strip("-"),
                "train": slice(int(train[0]), int(train[-1]) + 1),
                "val": slice(int(val[0]), int(val[-1]) + 1),
                "test": slice(int(test[0]), int(test[-1]) + 1),
                "bounds": {
                    "train": (stamps[train[0]], stamps[train[-1]]),
                    "val": (stamps[val[0]], stamps[val[-1]]),
                    "test": (stamps[test[0]], stamps[test[-1]]),
                },
            }
        )

    return windows[:max_windows] if max_windows is not None else windows


def describe_windows(windows: list[dict]) -> pd.DataFrame:
    """Return one summary row per window, for logging or a report header."""
    return pd.DataFrame(
        [
            {
                "window": window["window"],
                "train_rows": window["train"].stop - window["train"].start,
                "val_rows": window["val"].stop - window["val"].start,
                "test_rows": window["test"].stop - window["test"].start,
                "train_end": window["bounds"]["train"][1],
                "test_start": window["bounds"]["test"][0],
                "test_end": window["bounds"]["test"][1],
            }
            for window in windows
        ]
    )
