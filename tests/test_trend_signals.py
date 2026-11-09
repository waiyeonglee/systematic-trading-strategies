"""The rule path's decision step: trend feature -> signed long/flat signal."""

from __future__ import annotations

import pandas as pd
import pytest

from features import trend_rule_signals


def _panel() -> pd.DataFrame:
    """Two weekly dates, two tickers, one positive and one negative trend value."""
    index = pd.MultiIndex.from_product(
        [pd.to_datetime(["2020-01-03", "2020-01-10"]), ["AAA", "BBB"]],
        names=["date", "ticker"],
    )
    return pd.DataFrame(
        {"trend_sma40_ratio": [0.02, -0.01, 0.0, 0.05],
         "trend_mom52": [-0.10, 0.04, 0.03, -0.02]},
        index=index,
    )


def test_sma40_signal_is_strictly_positive_mapping():
    signals = trend_rule_signals(_panel(), "sma40")
    assert signals["pred"].tolist() == [1.0, 0.0, 0.0, 1.0]


def test_mom52_uses_its_own_column():
    signals = trend_rule_signals(_panel(), "mom52")
    assert signals["pred"].tolist() == [0.0, 1.0, 1.0, 0.0]


def test_shorting_variant_maps_off_signal_to_minus_one():
    signals = trend_rule_signals(_panel(), "sma40", shorting=True)
    assert signals["pred"].tolist() == [1.0, -1.0, -1.0, 1.0]


def test_unknown_rule_is_rejected():
    with pytest.raises(ValueError, match="unknown trend rule"):
        trend_rule_signals(_panel(), "sma200")


def test_missing_trend_column_is_rejected():
    with pytest.raises(ValueError, match="missing columns"):
        trend_rule_signals(_panel().drop(columns=["trend_mom52"]), "mom52")
