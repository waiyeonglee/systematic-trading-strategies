"""Checks for the forecast-volatility-scaled portfolio and its broker cost models."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from portfolio import (ForecastVolParams, build_forecast_vol_portfolio, build_positions,
                       historical_forecast_vol, ibkr_commission, mark_to_market_weights,
                       moomoo_commission, order_cost_usd, orders_for_target, rebalance_dates,
                       target_weights)

DATES = pd.bdate_range("2023-01-02", periods=4)


def make_panel(predictions: list[tuple[float, float]],
               returns: list[tuple[float, float]] | None = None) -> pd.DataFrame:
    """Prediction panel for tickers A and B, one tuple per date."""
    index = pd.MultiIndex.from_product([DATES[:len(predictions)], ["A", "B"]],
                                      names=["date", "ticker"])
    forward = np.zeros(len(predictions) * 2) if returns is None else np.ravel(returns)
    return pd.DataFrame({"pred": np.ravel(predictions), "fwd_ret": forward}, index=index)


def test_forecast_vol_uses_only_predictions_before_the_decision_date():
    params = ForecastVolParams(vol_window=2, vol_min_periods=2)
    wide = pd.DataFrame({"A": [0.010, 0.020, 0.030]}, index=DATES[:3])

    forecast = historical_forecast_vol(wide, params)

    expected = np.std([0.010, 0.020], ddof=1)  # the first two predictions only
    assert np.isnan(forecast["A"].iloc[0]) and np.isnan(forecast["A"].iloc[1])
    assert forecast["A"].iloc[2] == pytest.approx(expected)


def test_forecast_vol_ignores_the_prediction_of_the_same_date():
    params = ForecastVolParams(vol_window=2, vol_min_periods=2)
    quiet = pd.DataFrame({"A": [0.010, 0.020, 0.030]}, index=DATES[:3])
    spiked = quiet.copy()
    spiked.iloc[-1, 0] = 50.0

    assert (historical_forecast_vol(quiet, params)["A"].iloc[2]
            == pytest.approx(historical_forecast_vol(spiked, params)["A"].iloc[2]))


def test_forecast_vol_waits_for_the_minimum_number_of_predictions():
    params = ForecastVolParams(vol_window=10, vol_min_periods=5)
    wide = pd.DataFrame({"A": np.linspace(0.001, 0.020, 12)}, index=pd.bdate_range("2023-01-02", periods=12))

    forecast = historical_forecast_vol(wide, params)

    assert forecast["A"].iloc[:5].isna().all()
    assert forecast["A"].iloc[5:].notna().all()


def test_forecast_vol_survives_a_monthly_refit():
    dates = pd.bdate_range("2023-01-02", periods=30)
    predictions = pd.Series(np.linspace(-0.02, 0.02, 30), index=dates)
    predictions.iloc[20:] += 0.05  # a refit shifts the level of every later prediction

    forecast = historical_forecast_vol(predictions.to_frame("A"),
                                       ForecastVolParams(vol_window=10, vol_min_periods=5))

    assert forecast["A"].iloc[20:].notna().all()


def test_target_weights_scale_each_signal_by_its_forecast_volatility():
    params = ForecastVolParams(vol_window=2, vol_min_periods=2)
    wide = pd.DataFrame({"A": [0.010, 0.020, 0.030], "B": [0.020, 0.015, 0.030]},
                        index=DATES[:3])
    forecast = historical_forecast_vol(wide, params)

    weights = target_weights(wide.loc[DATES[2]], forecast.loc[DATES[2]])

    assert weights["A"] == pytest.approx(1 / 3)
    assert weights["B"] == pytest.approx(2 / 3)
    assert weights.abs().sum() == pytest.approx(1.0)


def test_target_weights_keep_both_signs_without_forcing_dollar_neutrality():
    predictions = pd.Series({"A": 0.05, "B": 0.01, "C": -0.01})
    volatility = pd.Series(0.01, index=predictions.index)

    weights = target_weights(predictions, volatility)

    assert weights["C"] < 0 < weights["A"]
    assert weights.abs().sum() == pytest.approx(1.0)
    assert weights.sum() == pytest.approx(5 / 7)  # not zero: gross normalisation only


def test_target_weights_drop_assets_without_a_usable_forecast_or_volatility():
    predictions = pd.Series({"A": 0.02, "B": 0.02, "C": np.nan, "D": 0.02})
    volatility = pd.Series({"A": 0.01, "B": 0.0, "C": 0.01, "D": np.nan})

    weights = target_weights(predictions, volatility)

    assert weights["A"] == pytest.approx(1.0)
    assert weights[["B", "C", "D"]].eq(0.0).all()


def test_target_weights_apply_the_volatility_floor():
    predictions = pd.Series({"A": 0.02, "B": 0.02})
    volatility = pd.Series({"A": 1e-9, "B": 1e-5})

    weights = target_weights(predictions, volatility)

    assert weights["A"] == pytest.approx(0.5)  # 1e-9 is floored to 1e-5
    assert weights["B"] == pytest.approx(0.5)


def test_moomoo_commission_applies_the_floor_and_the_gst():
    assert moomoo_commission(100.0) == pytest.approx(0.77 * 1.09)
    assert moomoo_commission(10_000.0) == pytest.approx(3.0 * 1.09)
    assert moomoo_commission(0.0) == 0.0


def test_ibkr_commission_takes_the_smaller_of_one_dollar_and_one_percent():
    assert ibkr_commission(50.0) == pytest.approx(0.5)
    assert ibkr_commission(1_000.0) == pytest.approx(1.0)


@pytest.mark.parametrize("broker", ["moomoo", "ibkr"])
def test_order_cost_charges_the_proportional_cost_exactly_once(broker):
    params = ForecastVolParams(broker=broker, cost_bps=5.0)
    value = 1_000_000.0
    commission = moomoo_commission(value) if broker == "moomoo" else ibkr_commission(value)

    assert order_cost_usd(value, params) == pytest.approx(commission + 5e-4 * value)


def test_order_cost_is_charged_per_order_and_never_for_zero_value():
    params = ForecastVolParams(broker="moomoo", cost_bps=5.0)

    assert order_cost_usd(0.0, params) == 0.0
    assert (order_cost_usd(100.0, params) + order_cost_usd(100.0, params)
            > order_cost_usd(200.0, params))


def test_orders_for_target_net_a_flip_from_long_to_short_into_one_order():
    orders = orders_for_target(pd.Series({"A": 0.5}), pd.Series({"A": -0.5}), 1_000.0)

    assert orders["A"] == pytest.approx(-1_000.0)


def test_mark_to_market_drifts_weights_towards_the_winner():
    drifted = mark_to_market_weights(pd.Series({"A": 0.5, "B": 0.5}),
                                     pd.Series({"A": 0.10, "B": 0.0}))

    assert drifted["A"] == pytest.approx(0.55 / 1.05)
    assert drifted.abs().sum() == pytest.approx(1.0)


def daily_params(**overrides) -> ForecastVolParams:
    """Two-observation volatility and a rebalance on every bar, for the short panels below."""
    return ForecastVolParams(vol_window=2, vol_min_periods=2, rebalance="daily", **overrides)


def test_portfolio_holds_cash_until_enough_prediction_history_exists():
    panel = make_panel([(0.01, 0.02), (0.02, 0.015), (0.03, 0.03)])

    port = build_forecast_vol_portfolio(panel, daily_params())

    assert port["gross_ret"].iloc[:2].eq(0.0).all()
    assert port["cost"].iloc[:2].eq(0.0).all()
    assert port["n_holdings"].iloc[2] == 2


def test_portfolio_charges_each_order_and_nets_the_cost():
    panel = make_panel([(0.010, 0.020), (0.020, 0.015), (0.030, 0.030)],
                       [(0.0, 0.0), (0.0, 0.0), (0.10, -0.05)])

    port = build_forecast_vol_portfolio(panel, daily_params(broker="moomoo", cost_bps=5.0),
                                        initial_capital_usd=100_000.0)

    # weights 1/3 and 2/3 of 100k: 10.9 + 16.67 on A, 21.8 + 33.33 on B
    expected_cost = 10.9 + 16.666667 + 21.8 + 33.333333
    assert port["turnover"].iloc[2] == pytest.approx(1.0)
    assert port["gross_ret"].iloc[2] == pytest.approx(0.0, abs=1e-12)
    assert port["cost"].iloc[2] == pytest.approx(expected_cost / 100_000)
    assert port["net_ret"].iloc[2] == pytest.approx(-expected_cost / 100_000)
    assert np.allclose(port["net_ret"], port["gross_ret"] - port["cost"])


def test_portfolio_marks_holdings_to_market_before_rebalancing():
    panel = make_panel([(0.010, 0.020), (0.020, 0.015), (0.030, 0.030), (0.030, 0.030)],
                       [(0.0, 0.0), (0.0, 0.0), (0.20, 0.0), (0.0, 0.0)])

    port = build_forecast_vol_portfolio(panel, daily_params(cost_bps=0.0))

    # 1/3 and 2/3 drift to 0.375 and 0.625 after A returns 20%, target is 0.6 and 0.4
    assert port["turnover"].iloc[2] == pytest.approx(1.0)
    assert port["turnover"].iloc[3] == pytest.approx(0.45)


def test_rebalance_dates_offer_daily_weekly_and_monthly_cadences():
    days = pd.bdate_range("2024-01-01", "2024-02-09")

    assert rebalance_dates(days, "daily").equals(days)
    assert list(rebalance_dates(days, "weekly")) == [
        pd.Timestamp(day) for day in ["2024-01-01", "2024-01-08", "2024-01-15",
                                      "2024-01-22", "2024-01-29", "2024-02-05"]]
    assert list(rebalance_dates(days, "monthly")) == [pd.Timestamp("2024-01-01"),
                                                      pd.Timestamp("2024-02-01")]


def test_weekly_bars_rebalance_every_bar_when_weekly_and_monthly_by_month():
    fridays = pd.bdate_range("2024-01-05", "2024-02-09", freq="W-FRI")

    assert rebalance_dates(fridays, "weekly").equals(fridays)
    assert len(rebalance_dates(fridays, "monthly")) == 2


def test_portfolio_rebalances_monthly_and_lets_the_weights_drift_between():
    days = pd.bdate_range("2022-12-01", "2023-02-28")
    index = pd.MultiIndex.from_product([days, ["A", "B"]], names=["date", "ticker"])
    alternate = np.arange(len(days)) % 2 == 0
    predictions = np.empty(len(days) * 2)
    predictions[0::2] = np.where(alternate, 0.030, 0.010)   # A
    predictions[1::2] = np.where(alternate, 0.010, 0.030)   # B mirrors A
    returns = np.empty(len(days) * 2)
    returns[0::2] = 0.01                                    # A drifts up every bar
    returns[1::2] = -0.01                                   # B drifts down every bar
    panel = pd.DataFrame({"pred": predictions, "fwd_ret": returns}, index=index)

    port = build_forecast_vol_portfolio(panel, ForecastVolParams(vol_window=5, vol_min_periods=3))

    assert port["rebalanced"].sum() == 3                      # Dec 1, Jan 2, Feb 1
    assert port.loc[port["rebalanced"] == 0, "turnover"].eq(0.0).all()
    assert port["turnover"].loc["2023-01-02"] == pytest.approx(1.0)   # first build from cash
    assert port["gross_exposure"].loc["2023-01-31"] == pytest.approx(1.0)
    assert port["turnover"].loc["2023-02-01"] > 0.1           # January's drift, traded away


def test_portfolio_scales_commissions_with_the_account_size():
    panel = make_panel([(0.010, 0.020), (0.020, 0.015), (0.030, 0.030)])

    large = build_forecast_vol_portfolio(panel, daily_params(cost_bps=0.0),
                                         initial_capital_usd=100_000.0)
    small = build_forecast_vol_portfolio(panel, daily_params(cost_bps=0.0),
                                         initial_capital_usd=100.0)

    # a 33 USD order pays the 0.77 USD commission floor, so the drag is far larger when small
    assert small["cost"].iloc[2] > 20 * large["cost"].iloc[2]


def test_portfolio_ignores_a_missing_prediction():
    panel = make_panel([(0.010, 0.020), (0.020, 0.015), (np.nan, 0.030)])

    port = build_forecast_vol_portfolio(panel, daily_params())

    assert port["holdings"].iloc[2] == "B"


def test_portfolio_stays_in_cash_when_no_asset_is_eligible():
    panel = make_panel([(0.01, 0.01)] * 4)  # constant predictions: zero forecast volatility

    port = build_forecast_vol_portfolio(panel, daily_params())

    assert (port["n_holdings"] == 0).all()
    assert port["equity_usd"].eq(100_000.0).all()


def test_build_positions_dispatches_to_the_forecast_vol_scaled_strategy():
    config = SimpleNamespace(strategy="forecast_vol_scaled", forecast_vol=daily_params(),
                             initial_capital_usd=100_000.0)
    panel = make_panel([(0.010, 0.020), (0.020, 0.015), (0.030, 0.030)])

    port = build_positions(panel, config)

    assert set(port.columns) >= {"gross_ret", "net_ret", "turnover", "cost", "n_holdings"}


def test_invalid_settings_are_rejected():
    with pytest.raises(ValueError):
        ForecastVolParams(broker="schwab")
    with pytest.raises(ValueError):
        ForecastVolParams(rebalance="quarterly")
    with pytest.raises(ValueError):
        ForecastVolParams(vol_window=10, vol_min_periods=20)
    with pytest.raises(ValueError):
        ForecastVolParams(cost_bps=-1.0)
    with pytest.raises(ValueError):
        ForecastVolParams.from_mapping({"vol_window": 30, "commision_rate": 0.001})
    with pytest.raises(ValueError):
        ForecastVolParams.from_mapping({"gross_exposure": 1.0})  # fixed, not configurable
