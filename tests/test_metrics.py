"""Per-year OOS R²: a year with no predictions is marked 0 and flagged, not dropped."""

from __future__ import annotations

import numpy as np
import pandas as pd

from metrics import oos_r2_by_period, used_years


def _preds(dates) -> pd.DataFrame:
    """Predictions that are clearly wrong, so a scored year has a negative R²."""
    index = pd.MultiIndex.from_product([pd.DatetimeIndex(dates), ["AAA", "BBB"]],
                                       names=["date", "ticker"])
    actual = np.resize([0.02, -0.03], len(index))
    return pd.DataFrame({"fwd_ret_4bar": actual, "pred": np.full(len(index), 0.05)},
                        index=index)


def test_missing_expected_year_is_zero_and_flagged():
    preds = _preds(pd.date_range("2023-01-06", "2024-12-27", freq="W-FRI"))
    by_year = oos_r2_by_period(preds, "fwd_ret_4bar", expected_years=range(2023, 2026))

    assert list(by_year.index) == [2023, 2024, 2025]
    assert by_year.loc[2025] == 0.0
    assert by_year.attrs["unscored"] == {"2025": "no predictions"}
    # the scored years keep their own R²: a zero forecast is worse than the actual values
    assert (by_year.loc[[2023, 2024]] < 0).all()


def test_years_outside_the_expected_span_are_not_invented():
    preds = _preds(pd.date_range("2023-01-06", "2024-12-27", freq="W-FRI"))
    by_year = oos_r2_by_period(preds, "fwd_ret_4bar", expected_years=range(2023, 2025))

    assert list(by_year.index) == [2023, 2024]
    assert by_year.attrs["unscored"] == {}


def test_unrevealed_labels_at_the_window_end_drop_out():
    """Trailing bars whose label has not realised must not turn a year into NaN."""
    preds = _preds(pd.date_range("2023-01-06", "2024-01-05", freq="W-FRI"))
    last_date = preds.index.get_level_values("date").max()
    preds.loc[preds.index.get_level_values("date") == last_date, "fwd_ret_4bar"] = np.nan

    by_year = oos_r2_by_period(preds, "fwd_ret_4bar")
    assert np.isfinite(by_year.loc[2023])
    # 2024 holds only that unrevealed bar, so it has nothing to score
    assert np.isnan(by_year.loc[2024])


def test_used_years_spans_the_index():
    book = pd.DataFrame({"net_ret": 0.0},
                        index=pd.DatetimeIndex(["2023-01-06", "2025-12-19"]))
    assert list(used_years(book)) == [2023, 2024, 2025]
