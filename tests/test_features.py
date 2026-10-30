"""Checks for weekly feature construction and correlation calculation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import talib

from data import resample_weekly
from features import FEATURES, build_feature_frame, feature_label_correlation
from labels import LABEL


def make_prices(weeks: int = 80):
    rng = np.random.default_rng(5205)
    dates = pd.bdate_range("2020-01-06", periods=weeks * 5)
    prices = {}
    for ticker, drift in (("XLB", 0.0006), ("XLE", 0.0001), ("XLF", 0.0004)):
        returns = rng.normal(drift, 0.025, size=len(dates))
        close = 100 * np.cumprod(1 + returns)
        prices[ticker] = pd.DataFrame({
            "Open": close * (1 - 0.002),
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": rng.integers(100_000, 500_000, size=len(dates)),
        }, index=dates)
    weekly_prices = {ticker: resample_weekly(bars) for ticker, bars in prices.items()}
    weekly_dates = weekly_prices["XLB"].index
    return prices, weekly_prices, dates, weekly_dates


def test_feature_frame_has_one_row_per_asset_date_and_uses_past_bars_only():
    prices, weekly_prices, daily_dates, weekly_dates = make_prices()
    features = build_feature_frame(weekly_prices, frequency="weekly", sectors=("XLB", "XLE", "XLF"))
    assert features.shape == (len(weekly_dates) * 3, len(FEATURES) + 1)
    assert features.columns.tolist() == FEATURES + [LABEL]

    key = (weekly_dates[60], "XLB")
    modified = {ticker: bars.copy() for ticker, bars in prices.items()}
    modified["XLB"].loc[daily_dates[350]:, "Close"] *= 1.5
    modified_weekly = {ticker: resample_weekly(bars) for ticker, bars in modified.items()}
    changed_features = build_feature_frame(modified_weekly, frequency="weekly", sectors=("XLB", "XLE", "XLF"))
    pd.testing.assert_series_equal(features.loc[key], changed_features.loc[key])

    daily_features = build_feature_frame(prices, frequency="daily", sectors=("XLB", "XLE", "XLF"))
    assert daily_features.shape == (len(daily_dates) * 3, len(FEATURES) + 1)
    daily_key = (daily_dates[300], "XLB")
    changed_daily_features = build_feature_frame(modified, frequency="daily", sectors=("XLB", "XLE", "XLF"))
    pd.testing.assert_series_equal(daily_features.loc[daily_key], changed_daily_features.loc[daily_key])


def test_indicators_use_bar_lookbacks_at_each_frequency():
    daily_prices, weekly_prices, daily_dates, weekly_dates = make_prices()
    daily_features = build_feature_frame(daily_prices, frequency="daily", sectors=("XLB",))
    weekly_features = build_feature_frame(weekly_prices, frequency="weekly", sectors=("XLB",))

    for features, dates in ((daily_features, daily_dates), (weekly_features, weekly_dates)):
        assert pd.isna(features.loc[(dates[13], "XLB"), "rsi_14bar"])
        assert np.isfinite(features.loc[(dates[14], "XLB"), "rsi_14bar"])
        assert pd.isna(features.loc[(dates[51], "XLB"), "return_52bar"])
        assert np.isfinite(features.loc[(dates[52], "XLB"), "return_52bar"])


def test_talib_indicators_match_library_outputs():
    prices, _, daily_dates, _ = make_prices()
    close = prices["XLB"]["Close"].to_numpy(dtype=float)
    features = build_feature_frame(prices, frequency="daily", sectors=("XLB",))
    expected_rsi = talib.RSI(close, timeperiod=14)[-1] / 100
    expected_macd, expected_signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
    last = (daily_dates[-1], "XLB")

    assert np.isclose(features.loc[last, "rsi_14bar"], expected_rsi)
    assert np.isclose(features.loc[last, "macd"], expected_macd[-1] / close[-1])
    assert np.isclose(features.loc[last, "macd_signal"], expected_signal[-1] / close[-1])


def test_fear_greed_index_aligns_as_of_bar_date_without_lookahead():
    daily_prices, weekly_prices, daily_dates, weekly_dates = make_prices(weeks=4)
    sentiment = pd.Series(
        [25.0, 60.0, 80.0],
        index=pd.to_datetime(["2020-01-07", "2020-01-13", "2020-01-24"]),
    )

    daily = build_feature_frame(
        daily_prices, frequency="daily", sectors=("XLB",), fear_greed_index=sentiment
    )
    weekly = build_feature_frame(
        weekly_prices, frequency="weekly", sectors=("XLB",), fear_greed_index=sentiment
    )

    assert pd.isna(daily.loc[(daily_dates[0], "XLB"), "fear_greed_index"])
    assert daily.loc[(daily_dates[1], "XLB"), "fear_greed_index"] == 25.0
    assert daily.loc[(daily_dates[4], "XLB"), "fear_greed_index"] == 25.0
    assert daily.loc[(daily_dates[5], "XLB"), "fear_greed_index"] == 60.0
    assert weekly.loc[(weekly_dates[0], "XLB"), "fear_greed_index"] == 25.0
    assert weekly.loc[(weekly_dates[1], "XLB"), "fear_greed_index"] == 60.0
    assert weekly.loc[(weekly_dates[2], "XLB"), "fear_greed_index"] == 80.0


def test_spy_return_uses_matching_bar_without_lookahead():
    daily_prices, _, daily_dates, _ = make_prices(weeks=16)
    spy = daily_prices["XLB"].copy()
    features = build_feature_frame(
        daily_prices, frequency="daily", sectors=("XLE",), benchmark_prices=spy
    )
    feature_date = daily_dates[51]
    expected = spy["Close"].pct_change(fill_method=None).loc[feature_date]
    assert pd.isna(features.loc[(daily_dates[0], "XLE"), "spy_return_1bar"])
    assert np.isclose(features.loc[(feature_date, "XLE"), "spy_return_1bar"], expected)

    modified_spy = spy.copy()
    modified_spy.loc[daily_dates[52]:, "Close"] *= 2
    changed = build_feature_frame(
        daily_prices, frequency="daily", sectors=("XLE",), benchmark_prices=modified_spy
    )
    pd.testing.assert_series_equal(
        features.loc[(feature_date, "XLE")], changed.loc[(feature_date, "XLE")]
    )


def test_forward_return_label_is_next_bar_return_not_a_predictor():
    prices, _, dates, _ = make_prices(weeks=4)
    features = build_feature_frame(prices, frequency="daily", sectors=("XLB",))
    key = (dates[10], "XLB")
    expected = prices["XLB"]["Close"].iloc[11] / prices["XLB"]["Close"].iloc[10] - 1

    assert LABEL == "label_forward_return_1bar"
    assert LABEL not in FEATURES
    assert np.isclose(features.loc[key, LABEL], expected)
    assert pd.isna(features.loc[(dates[-1], "XLB"), LABEL])


def test_feature_label_correlation_uses_all_dates_with_labels():
    _, weekly_prices, _, weekly_dates = make_prices()
    sentiment = pd.Series(np.linspace(10, 90, len(weekly_dates)), index=weekly_dates)
    dataset = build_feature_frame(
        weekly_prices, frequency="weekly", sectors=("XLB",),
        fear_greed_index=sentiment, benchmark_prices=weekly_prices["XLE"],
    )
    corr, sample_end = feature_label_correlation(dataset)
    assert corr.shape == (len(FEATURES) + 1, len(FEATURES) + 1)
    assert corr.index.tolist() == FEATURES + [LABEL]
    assert np.allclose(corr.to_numpy(), corr.to_numpy().T, equal_nan=True)
    finite = corr.to_numpy()[np.isfinite(corr.to_numpy())]
    assert ((finite >= -1) & (finite <= 1)).all()
    assert np.allclose(np.diag(corr), 1.0, equal_nan=True)
    assert sample_end == weekly_dates[-2]
    expected = dataset.loc[dataset[LABEL].notna(), FEATURES + [LABEL]].corr(
        method="pearson", min_periods=2
    )
    pd.testing.assert_frame_equal(corr, expected)


def test_weekly_features_use_pre_resampled_weekly_bars():
    dates = pd.bdate_range("2022-01-03", periods=300)
    close = pd.Series(np.linspace(100, 160, len(dates)), index=dates)
    prices = {"XLB": pd.DataFrame({
        "Open": close, "High": close * 1.01, "Low": close * 0.99,
        "Close": close, "Volume": np.arange(len(dates)) + 1000,
    })}
    weekly_prices = {"XLB": resample_weekly(prices["XLB"])}
    features = build_feature_frame(weekly_prices, frequency="weekly", sectors=("XLB",))
    weekly_dates = features.index.get_level_values("date").unique()
    assert len(weekly_dates) == 60
    weekly_close = weekly_prices["XLB"]["Close"]
    assert np.isclose(features.loc[(weekly_dates[5], "XLB"), "return_1bar"], weekly_close.iloc[5] / weekly_close.iloc[4] - 1)
    assert pd.isna(features.loc[(weekly_dates[51], "XLB"), "return_52bar"])
    assert np.isfinite(features.loc[(weekly_dates[52], "XLB"), "return_52bar"])
