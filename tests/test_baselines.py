"""The AR(1) forecast baseline: fit, no look-ahead, and the mean fallback."""

from __future__ import annotations

import numpy as np
import pandas as pd

from baselines import ar1_forecast, r2_against


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
