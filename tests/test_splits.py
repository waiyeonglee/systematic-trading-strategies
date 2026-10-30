"""Tests for :mod:`splits`.

Only the behaviours that would be bugs if they broke are covered: that no window can look
ahead or double-score a bar, that the study-period bounds bite, and that bad input is
rejected instead of silently mis-split.
"""

from __future__ import annotations

import pandas as pd
import pytest

from splits import make_windows

DATES = pd.bdate_range("2005-01-03", "2021-12-31")     # trading-day calendar

YEARLY = dict(freq="Y", train_periods=3, val_periods=1, test_periods=1, horizon=1, embargo=1)
MONTHLY = dict(freq="M", train_periods=36, val_periods=6, test_periods=1, horizon=1, embargo=1)
WEEKLY = dict(freq="W", train_periods=156, val_periods=4, test_periods=1, horizon=1, embargo=1)
DAILY = dict(freq="D", train_periods=756, val_periods=21, test_periods=5, horizon=1, embargo=1)

BASE = dict(YEARLY)

CONFIGS = pytest.mark.parametrize("cfg", [YEARLY, MONTHLY, WEEKLY, DAILY],
                                  ids=["Y", "M", "W", "D"])


def rows(sl: slice) -> list[int]:
    return list(range(sl.start, sl.stop))


# ---------------------------------------------------------------------------
# No look-ahead, no double scoring
# ---------------------------------------------------------------------------
@CONFIGS
def test_blocks_are_ordered_with_an_embargo_between_them(cfg):
    for win in make_windows(DATES, **cfg):
        train, val, test = rows(win["train"]), rows(win["val"]), rows(win["test"])
        assert train and val and test
        assert train[-1] < val[0] < val[-1] < test[0] <= test[-1]
        assert win["val"].start - win["train"].stop == cfg["embargo"]
        assert win["test"].start - win["val"].stop == cfg["embargo"]


@CONFIGS
def test_every_bar_is_scored_once_under_a_unique_label(cfg):
    windows = make_windows(DATES, **cfg)
    scored = [i for win in windows for i in rows(win["test"])]

    assert scored == sorted(scored), "test blocks are not in time order"
    assert len(scored) == len(set(scored)), "a bar was scored twice"
    assert len({win["slug"] for win in windows}) == len(windows), "duplicate cache filename"


@CONFIGS
def test_every_window_has_the_same_train_length(cfg):
    sizes = {
        pd.DatetimeIndex(DATES[win["train"]]).to_period(cfg["freq"]).nunique()
        for win in make_windows(DATES, **cfg)
    }
    # one distinct size: the window slides instead of expanding as the sample grows
    assert len(sizes) == 1, f"train block size varies across windows: {sorted(sizes)}"


def test_partial_first_and_last_periods_are_used_as_they_are():
    # neither end of this series is a whole calendar year, so the boundary blocks are short
    dates = pd.bdate_range("2015-07-01", "2021-12-31")
    windows = make_windows(dates, **YEARLY)

    assert [win["window"] for win in windows] == ["2019", "2020", "2021"]
    assert windows[0]["bounds"]["train"][0] == pd.Timestamp("2015-07-01")


# ---------------------------------------------------------------------------
# Study-period bounds
# ---------------------------------------------------------------------------
def test_bounds_restrict_the_windows_on_both_ends():
    windows = make_windows(DATES, freq="M", train_periods=36, val_periods=6, test_periods=1,
                           horizon=1, embargo=1,
                           first_test="2012-01-01", last_test="2015-12-31")
    assert windows[0]["window"] == "2012-01"
    assert windows[-1]["window"] == "2015-12"
    assert len(windows) == 48


# ---------------------------------------------------------------------------
# Cases that used to crash or silently mis-split
# ---------------------------------------------------------------------------
def test_history_that_cannot_fill_a_window_returns_nothing():
    assert make_windows(pd.bdate_range("2021-01-01", periods=100), **YEARLY) == []


def test_unsorted_dates_are_rejected():
    with pytest.raises(ValueError, match="sorted ascending"):
        make_windows(DATES[::-1], **YEARLY)


# ---------------------------------------------------------------------------
# The horizon drives the embargo
# ---------------------------------------------------------------------------
def test_embargo_shorter_than_horizon_is_rejected():
    with pytest.raises(ValueError, match="shorter than horizon"):
        make_windows(DATES, **{**BASE, "horizon": 5, "embargo": 4})


@pytest.mark.parametrize(
    "kwargs",
    [{"train_periods": 0}, {"horizon": 0}, {"max_windows": 0},
     {"first_test": "2020-01-01", "last_test": "2019-01-01"}],
)
def test_invalid_arguments_are_rejected(kwargs):
    with pytest.raises(ValueError):
        make_windows(DATES, **{**BASE, **kwargs})
