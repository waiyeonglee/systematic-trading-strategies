"""The AR(1) forecast baseline: fit, no look-ahead, and the mean fallback."""

from __future__ import annotations

import numpy as np
import pandas as pd

from baselines import ar1_book, ar1_forecast, ar1_warmup, r2_against
from portfolio import ForecastVolParams


def _panel(scale_after: pd.Timestamp | None = None, scale: float = 1.0) -> pd.DataFrame:
    """Two tickers on weekly dates; target is an exact linear function of the lag.

    ``y = 0.01 + 0.5 * x`` with ``x`` = trailing 4-bar return, so an exact OLS recovers
    ``(intercept, slope) = (0.01, 0.5)`` and the prediction equals the target.
    """
    dates = pd.date_range("2015-01-02", "2018-12-28", freq="W-FRI")
    index = pd.MultiIndex.from_product([dates, ["AAA", "BBB"]], names=["date", "ticker"])
    step = np.tile(np.arange(len(dates)), 2)
    x = 0.002 * np.sin(step / 7.0) + 0.001
    y = 0.01 + 0.5 * x
    if scale_after is not None:
        later = index.get_level_values("date") >= scale_after
        y = np.where(later, y * scale, y)
    return pd.DataFrame({
        "return_4bar": x,
        "fwd_ret_4bar": y,
        "fwd_ret": x,
    }, index=index)


def test_exact_ar1_is_recovered():
    preds, coefficients = ar1_forecast(_panel(), "fwd_ret_4bar", window=60, embargo=4,
                                       min_obs=5)
    assert np.allclose(coefficients["slope"], 0.5)
    assert np.allclose(coefficients["intercept"], 0.01)
    fitted = preds.dropna(subset=["pred"])
    assert np.allclose(fitted["pred"], fitted["fwd_ret_4bar"], atol=1e-12)


def test_predictions_do_not_use_later_targets():
    cutoff = pd.Timestamp("2017-06-30")
    base, _ = ar1_forecast(_panel(), "fwd_ret_4bar", window=60, embargo=4)
    shocked, _ = ar1_forecast(_panel(scale_after=cutoff, scale=10.0), "fwd_ret_4bar",
                              window=60, embargo=4)
    dates = base.index.get_level_values("date")
    early = dates < cutoff
    assert np.allclose(base.loc[early, "pred"], shocked.loc[early, "pred"], equal_nan=True)


def test_below_min_obs_falls_back_to_the_historical_mean():
    preds, coefficients = ar1_forecast(_panel(), "fwd_ret_4bar", window=60, embargo=4,
                                       min_obs=10_000)
    assert (coefficients["slope"] == 0.0).all()
    fitted = preds.dropna(subset=["pred"])
    assert np.allclose(fitted["pred"], fitted["pred_mean"])


def test_r2_against_mean_is_positive_when_the_model_is_closer():
    preds = pd.DataFrame({
        "fwd_ret_4bar": [1.0, -1.0, 0.5],
        "pred": [1.0, -1.0, 0.5],
        "pred_mean": [0.0, 0.0, 0.0],
    })
    assert r2_against(preds, "pred_mean", "fwd_ret_4bar") == 1.0


def test_warmup_predictions_precede_the_evaluation_window():
    """Used as a vol seed: the history ends where the traded window begins."""
    panel = _panel()
    dates = panel.index.get_level_values("date").unique().sort_values()
    eval_start = dates[150]
    history = panel[panel.index.get_level_values("date") < eval_start]
    warm = ar1_warmup(history, "fwd_ret_4bar", bars=20, window=60, embargo=4,
                      min_obs=5)
    assert not warm.empty
    assert warm.index.get_level_values("date").max() < eval_start
    assert warm.index.get_level_values("date").nunique() == 20


def test_ar1_book_charges_costs_on_the_predictions():
    """The baseline is traded by the study's book, so its returns are net of costs."""
    panel = _panel()
    preds, _ = ar1_forecast(panel, "fwd_ret_4bar", window=60, embargo=4, min_obs=5)
    preds = preds.assign(fwd_ret=preds["fwd_ret_4bar"])
    book = ar1_book(preds, ForecastVolParams(vol_window=8, vol_min_periods=4,
                                             rebalance="4w"), ppy=52)
    assert {"net_ret", "gross_ret", "cost"} <= set(book.columns)
    assert (book["cost"] >= 0).all()
    assert book["net_ret"].abs().sum() > 0
