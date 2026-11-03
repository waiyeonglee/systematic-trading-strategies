"""Project constants: the universe, frequencies and the fixed dataset window."""

from __future__ import annotations

import random
import re

import numpy as np
import pandas as pd

DEFAULT_SEED = 5205


def seed_everything(seed: int = DEFAULT_SEED) -> int:
    """Seed the global RNGs so a study run is reproducible end to end.

    Model randomness is already pinned by ``random_state=seed`` on every estimator;
    this also fixes the interpreter-level generators in case a helper samples.
    Returns ``seed`` so callers can echo it.
    """
    random.seed(seed)
    np.random.seed(seed)
    return seed


FREQUENCIES = ("daily", "weekly")
PERIODS_PER_YEAR = {"daily": 252, "weekly": 52}

# Fixed dataset window (v2 convention — recorded in agents/current_state.md):
# data ends on or before 2026-01-01; the fetch bound is exclusive 2026-01-02.
START_DATE = "2004-01-01"         # inclusive; pre-2011 history is feature warmup only
END_DATE_INCLUSIVE = "2026-01-01"  # fixed end of the dataset
END_DATE_EXCLUSIVE = "2026-01-02"  # yfinance exclusive end (inclusive + 1 day)

# Universe (v2, identical to the reference project):
# 9 sector ETFs + 7 diversifiers = 16 tradables; SPY is benchmark only.
CORE_ETFS = ("XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY")
CROSS_ASSET_ETFS = ("TLT", "IEF", "GLD", "DBC", "UUP", "EFA", "EEM")
TRADABLE_ETFS = CORE_ETFS + CROSS_ASSET_ETFS
BENCHMARKS = ("SPY",)
TICKERS = TRADABLE_ETFS + BENCHMARKS


def periods_per_year(frequency: str) -> int:
    """Return 252 for daily bars and 52 for weekly bars.

    Raises:
        ValueError: if ``frequency`` is not a supported bar frequency.
    """
    if frequency not in FREQUENCIES:
        raise ValueError(f"frequency must be one of {FREQUENCIES}, got {frequency!r}")
    return PERIODS_PER_YEAR[frequency]


_DURATION = re.compile(r"^\s*(?:(\d+)\s*y)?\s*(?:(\d+)\s*m)?\s*(?:(\d+)\s*w)?\s*$",
                       re.IGNORECASE)


def parse_duration(text: str) -> tuple[int, int, int]:
    """Parse a window such as ``"2y 3m 4w"`` into ``(years, months, weeks)``.

    Raises:
        ValueError: if the text is empty, uses an unknown unit, or adds up to zero.
    """
    match = _DURATION.match(str(text))
    if not match or not any(match.groups()):
        raise ValueError(
            f"invalid window {text!r}: use years, months and weeks in that order, "
            "e.g. '5y', '18m', '2y 3m 4w'"
        )
    years, months, weeks = (int(value or 0) for value in match.groups())
    if years + months + weeks == 0:
        raise ValueError(f"window {text!r} must be longer than zero")
    return years, months, weeks


def duration_to_offset(text: str) -> pd.DateOffset:
    """Convert a window such as ``"2y 3m 4w"`` into a calendar offset."""
    years, months, weeks = parse_duration(text)
    return pd.DateOffset(years=years, months=months, weeks=weeks)


def duration_to_bars(text: str, frequency: str) -> int:
    """Convert a window into a number of bars at ``frequency``, at least one."""
    years, months, weeks = parse_duration(text)
    span = years + months / 12 + weeks / 52
    return max(1, round(span * periods_per_year(frequency)))


def duration_slug(text: str) -> str:
    """Compact filename-safe form of a window, ``"2y 3m 4w"`` -> ``"2y3m4w"``."""
    return str(text).replace(" ", "").lower()
