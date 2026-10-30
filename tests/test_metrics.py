"""Checks for the performance and forecast metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd

from labels import TARGET_COL
from metrics import (capm_alpha, comparison_table, equal_weight_universe, forecast_metrics,
                     max_drawdown, oos_r2, perf_stats, spy_buy_and_hold)


def test_max_drawdown_reports_the_largest_peak_to_trough_fall():
    assert np.isclose(max_drawdown(pd.Series([0.10, -0.50, 0.20])), -0.5)


def test_perf_stats_annualises_a_constant_return():
    stats = perf_stats(pd.Series([0.01] * 252), ppy=252)

    assert np.isclose(stats["ann_return"], (1.01**252) - 1)
    assert np.isclose(stats["ann_vol"], 0.0, atol=1e-12)
    assert np.isnan(stats["sharpe"])
    assert stats["hit_rate"] == 1.0


def test_comparison_table_has_one_row_per_series():
    returns = pd.Series(np.random.default_rng(0).normal(0, 0.01, 300))
    table = comparison_table({"A": returns, "B": returns * 0.5})

    assert list(table.index) == ["A", "B"]
    assert {"ann_return", "sharpe", "max_drawdown"} <= set(table.columns)


def test_capm_alpha_recovers_a_known_beta():
    rng = np.random.default_rng(1)
    market = pd.Series(rng.normal(0, 0.01, 2000))
    strategy = 0.0001 + 0.5 * market

    result = capm_alpha(strategy, market)

    assert np.isclose(result["beta"], 0.5, atol=1e-6)
    assert np.isclose(result["alpha_ann"], 0.0001 * 252, atol=1e-6)


def test_spy_buy_and_hold_uses_the_next_bar_return():
    close = pd.Series([100.0, 101.0, 99.99], index=pd.bdate_range("2023-01-02", periods=3))

    held = spy_buy_and_hold(close, close.index[:2])

    assert np.allclose(held.to_numpy(), [0.01, -0.01])


def test_equal_weight_universe_averages_the_forward_return_across_tickers():
    dates = pd.bdate_range("2023-01-02", periods=2)
    preds = pd.DataFrame(
        {"fwd_ret": [0.02, 0.00, 0.01, 0.03], "pred": [1.0, 0.0, 1.0, 0.0]},
        index=pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "ticker"]),
    )

    equal_weight = equal_weight_universe(preds)

    assert np.allclose(equal_weight.to_numpy(), [0.01, 0.02])


def test_oos_r2_is_positive_only_when_the_forecast_beats_zero():
    actual = np.array([0.01, -0.01])

    assert np.isclose(oos_r2(actual, actual / 2), 0.75)
    assert oos_r2(actual, np.zeros_like(actual)) == 0.0


def test_forecast_metrics_include_rank_information_coefficient():
    dates = pd.bdate_range("2023-01-02", periods=3)
    preds = pd.DataFrame(
        {"pred": [0.2, 0.1, 0.3, 0.1, 0.2, 0.4], "fwd_ret": [0.02, 0.01, 0.03, 0.01, 0.02, 0.04],
         TARGET_COL: [0.01, 0.00, 0.02, 0.00, 0.01, 0.03]},
        index=pd.MultiIndex.from_product([dates, ["A", "B"]], names=["date", "ticker"]),
    )

    result = forecast_metrics(preds)

    assert np.isclose(result["rmse"], np.sqrt(np.mean((preds[TARGET_COL] - preds["pred"]) ** 2)))
    assert result["rank_ic_mean"] > 0.99
    assert result["direction_acc"] == 1.0
