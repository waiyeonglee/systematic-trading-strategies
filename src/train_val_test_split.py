"""Sliding-window (rolling-origin) train / validation / test splits for time series.

A single chronological holdout leaks information whenever the observations are
autocorrelated, which financial bars always are, so the sample is cut into consecutive
blocks instead: the model trains on one block, is validated on the block that follows it,
and is scored on the block after that.  The pattern then slides forward and re-fits, and
the window never grows: every window trains on exactly ``train_periods`` periods, so a
difference in error between windows is not confounded by a difference in how much history
the model had.

Blocks are counted in *periods*, not rows, so one rule covers every frequency:

    freq="Y"  train=3,   val=1,  test=1    yearly
    freq="Q"  train=8,   val=2,  test=1    quarterly
    freq="M"  train=36,  val=6,  test=1    monthly
    freq="W"  train=156, val=4,  test=1    weekly
    freq="D"  train=756, val=21, test=5    daily bars (756 trading days ~ 3 years)

Only periods that actually contain bars are counted, so on a trading-day series "D" behaves
like a bar count and a holiday week does not consume one of the weekly periods.  Every period
is scored exactly once, and each window is trained only on bars that closed before its test
block opens.  Periods are calendar periods, so a series that begins or ends mid-period gives
that single boundary block a shorter span; align the data to period boundaries first if that
matters.

Usage::

    from train_val_test_split import make_windows

    windows = make_windows(dataset["date"], freq="W", train_periods=156, val_periods=4,
                           test_periods=1, horizon=1, embargo=1)
    for win in windows:
        X_train, y_train = X[win["train"]], y[win["train"]]
        X_val, y_val = X[win["val"]], y[win["val"]]
        X_test, y_test = X[win["test"]], y[win["test"]]
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd


def _as_timestamp(value: str | pd.Timestamp | None) -> pd.Timestamp | None:
    """A naive timestamp, so it can be compared with a series whose tz was dropped."""
    if value is None:
        return None
    stamp = pd.Timestamp(value)
    return stamp.tz_localize(None) if stamp.tz is not None else stamp


def _label(period: pd.Period, freq: str) -> str:
    """A short, filename-safe name for one period; a week spells out its own date range."""
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
    """Cut ``dates`` into sliding ``[train) [embargo] [val) [embargo] [test)`` windows.

    Everything after ``dates`` is keyword-only: at this many knobs a positional call is a
    guarantee of swapped ``train_periods`` and ``val_periods``.  Nothing is defaulted
    either, so no window is ever produced by a rule nobody chose.

    ``dates`` is any sequence of datetimes, a ``DatetimeIndex`` or a Series, one entry per
    row, already sorted ascending.  A timezone, if present, is dropped: the periods are
    calendar periods, so a tz-aware input would only make the labels ambiguous.

    ``freq`` is any pandas period alias: "D" day, "W" week, "M" month, "Q" quarter,
    "Y" year.  ``*_periods`` count periods of that frequency, so train/val/test cover the
    same *span of time* in every window; the bar counts differ slightly because holidays
    and half-days are not evenly distributed.

    ``horizon`` is the forecast horizon in bars: the label at bar ``t`` is realised at
    ``t + horizon``.  It is the length of the future a training row already commits to,
    which is what makes the embargo necessary.

    ``embargo`` is measured in bars, not periods, and must be at least ``horizon``: the
    last ``embargo`` bars are dropped from the train and val blocks so that no label there
    is realised inside the block that follows.

    ``first_test`` / ``last_test`` bound the study period: the first test block must start
    no earlier than ``first_test`` and the last must end no later than ``last_test``.  Both
    accept anything ``pd.Timestamp`` accepts, e.g. ``"2011-01-01"`` or ``"2020-12-31"``;
    leave them ``None`` to use every window the data supports.

    Each window's test block is ``test_periods`` periods long and consecutive windows slide
    by exactly that much, so every period is scored exactly once - never twice, never
    skipped.  The train block is always ``train_periods`` periods long, which keeps the
    amount of history constant across windows; an expanding rule would confound a change in
    error with a change in sample size.

    Returns one dict per window:

    ``window`` / ``slug``   label of the test block, e.g. ``"2014"``, ``"2014-07"``,
                            ``"2014-W07"``, ``"2014-07-01"``
    ``train``/``val``/``test``  positional slices, usable as ``X[win["train"]]``
    ``bounds``              ``split -> (first timestamp, last timestamp)``

    A candidate window is dropped when ``first_test``/``last_test`` rules it out, when its
    history does not reach back far enough, when its embargo would eat a whole block, or
    when its test block contains no bars.
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

    ts = pd.DatetimeIndex(pd.to_datetime(dates))
    if len(ts) == 0:
        return []
    if not ts.is_monotonic_increasing:
        raise ValueError("dates must be sorted ascending (oldest first)")
    if ts.tz is not None:
        ts = ts.tz_localize(None)

    try:
        periods = ts.to_period(freq)
    except (ValueError, KeyError) as exc:
        raise ValueError(
            f"unsupported freq {freq!r}; use a pandas period alias such as "
            "'D', 'W', 'M', 'Q' or 'Y'"
        ) from exc

    unique = periods.unique().sort_values()
    rank = unique.get_indexer(periods)      # row -> position of its period in `unique`
    rows = np.arange(len(ts))

    windows = []
    first_start = train_periods + val_periods
    for start in range(first_start, len(unique) - test_periods + 1, test_periods):
        train_lo = start - val_periods - train_periods
        train_hi = start - val_periods - 1
        val_lo, val_hi = start - val_periods, start - 1
        test_lo, test_hi = start, start + test_periods - 1

        train = rows[(rank >= train_lo) & (rank <= train_hi)]
        val = rows[(rank >= val_lo) & (rank <= val_hi)]
        test = rows[(rank >= test_lo) & (rank <= test_hi)]
        if min(len(train), len(val)) <= embargo or len(test) == 0:
            continue
        if lower is not None and ts[test[0]] < lower:
            continue
        if upper is not None and ts[test[-1]] > upper:
            continue

        train = train[: len(train) - embargo]
        val = val[: len(val) - embargo]

        first_label = _label(unique[test_lo], freq)
        label = first_label if test_periods == 1 \
            else f"{first_label}_{_label(unique[test_hi], freq)}"
        windows.append(
            {
                "window": label,
                "slug": re.sub(r"[^0-9A-Za-z]+", "-", label).strip("-"),
                "train": slice(int(train[0]), int(train[-1]) + 1),
                "val": slice(int(val[0]), int(val[-1]) + 1),
                "test": slice(int(test[0]), int(test[-1]) + 1),
                "bounds": {
                    "train": (ts[train[0]], ts[train[-1]]),
                    "val": (ts[val[0]], ts[val[-1]]),
                    "test": (ts[test[0]], ts[test[-1]]),
                },
            }
        )

    if max_windows is not None:
        windows = windows[:max_windows]
    return windows


def describe_windows(windows: list[dict]) -> pd.DataFrame:
    """One row per window, for a log line or the top of a report."""
    return pd.DataFrame(
        [
            {
                "window": w["window"],
                "train_rows": w["train"].stop - w["train"].start,
                "val_rows": w["val"].stop - w["val"].start,
                "test_rows": w["test"].stop - w["test"].start,
                "train_end": w["bounds"]["train"][1],
                "test_start": w["bounds"]["test"][0],
                "test_end": w["bounds"]["test"][1],
            }
            for w in windows
        ]
    )
