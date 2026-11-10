"""Naive forecast baselines for the prediction study.

``ar1_forecast`` fits, per ticker, a one-lag autoregression of the H-bar forward
return on the trailing H-bar return::

    y_t = a + b * r_{t-H..t},      y_t = fwd_ret_Hbar (price[t+H+1] / price[t+1] - 1)

Both sides of the regression are known at the right time: the regressor is the
trailing return at the close of bar ``t`` (the decision date), and the target is only
used for estimation on rows whose label has fully realised before the refit date
(embargo = H). It is therefore a genuine out-of-sample forecast, not a lagged copy of
the target. For ``H = 1`` this is the textbook AR(1) on weekly returns; for ``H > 1``
it is the same one-lag model on the H-bar return series.

The refit schedule mirrors ``models.walk_forward_predict``: block-by-block over
``refit_every`` calendar periods, trained on dates strictly before the block minus
``embargo`` dates, optionally trimmed to the most recent ``window`` dates. Blocks
without enough training history are skipped, so the returned panel can carry fewer
dates than the input.

``pred_mean`` is the AR(0) comparison forecast — each ticker's training-window mean —
so the caller can report the standard "does the model beat the historical average?"
OOS R² as well as the against-zero R² used elsewhere in this project.

``ar1_book`` turns those forecasts into a tradable book with the study's own
``forecast_vol_scaled`` rule (same costs, same rebalance cadence), so the baseline can
be compared with the fitted models on Sharpe and drawdown, not only on R².
``ar1_baseline`` does both for one window of a split, which is what the report figures
and the arm comparison consume.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from labels import TARGET_COL, target_horizon
from utils.checks import require_columns, require_panel


def ar1_forecast(panel: pd.DataFrame, target_col: str = TARGET_COL,
                 refit_every: str = "QE", window: int | None = 416,
                 embargo: int | None = None,
                 min_obs: int = 40) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Walk-forward one-lag forecasts of the H-bar forward return.

    Args:
        panel: Feature panel indexed by ``(date, ticker)`` with ``fwd_ret``, the
            target column and the matching ``return_Hbar`` regressor.
        target_col: Label to forecast, ``fwd_ret_1bar`` / ``_4bar`` / ``_12bar``.
        refit_every: Refit block as a pandas offset alias, e.g. ``"QE"``.
        window: Rolling training window in dates; ``None`` uses all past dates.
        embargo: Most recent training dates dropped before each refit; defaults to
            the target horizon ``H``, which leaves the last kept label realised exactly
            at the first prediction date.
        min_obs: Minimum training pairs before the OLS slope is used; below it the
            ticker's training mean is the forecast (slope fixed at zero).

    Returns:
        ``(preds, coefficients)``. ``preds`` is the evaluated part of the panel with
        ``pred`` (the AR(1) forecast) and ``pred_mean`` (the AR(0) forecast), indexed
        by ``(date, ticker)``. ``coefficients`` has one row per refit block and ticker
        with ``intercept``, ``slope`` and ``n_obs``.

    Raises:
        ValueError: if the panel lacks the target or its trailing-return regressor.
    """
    require_panel(panel)
    horizon = target_horizon(target_col)
    lag_col = f"return_{horizon}bar"
    require_columns(panel, [target_col, lag_col, "fwd_ret"], "forecast panel")
    embargo = horizon if embargo is None else embargo
    if embargo < 0:
        raise ValueError("embargo cannot be negative")

    full = panel.sort_index()
    dates = full.index.get_level_values("date").unique().sort_values()
    frames, coefficient_rows = [], []
    for _, block_dates in pd.Series(dates, index=dates).groupby(pd.Grouper(freq=refit_every)):
        if block_dates.empty:
            continue
        past = dates[dates < block_dates.iloc[0]]
        if embargo:
            past = past[:-embargo]
        if window is not None:
            past = past[-window:]
        if past.empty:
            continue

        train = full.loc[full.index.get_level_values("date").isin(past)]
        block = full.loc[full.index.get_level_values("date").isin(block_dates)]
        pred = pd.Series(np.nan, index=block.index, dtype=float)
        mean_forecast = pd.Series(np.nan, index=block.index, dtype=float)

        for ticker, group in train.groupby(level="ticker", observed=True):
            y = group[target_col].to_numpy(dtype=float)
            x = group[lag_col].to_numpy(dtype=float)
            usable = np.isfinite(y) & np.isfinite(x)
            if not usable.any():
                continue
            y, x = y[usable], x[usable]
            mean_value = float(y.mean())
            if len(y) >= min_obs and np.var(x, ddof=1) > 0:
                slope = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
                intercept = mean_value - slope * float(x.mean())
            else:
                slope, intercept = 0.0, mean_value

            mask = block.index.get_level_values("ticker") == ticker
            x_block = block.loc[mask, lag_col].to_numpy(dtype=float)
            pred.loc[mask] = intercept + slope * x_block
            mean_forecast.loc[mask] = mean_value
            coefficient_rows.append(dict(refit=block_dates.iloc[0], ticker=ticker,
                                         intercept=intercept, slope=slope, n_obs=len(y)))

        frame = block[["fwd_ret", target_col]].copy()
        frame["pred"] = pred
        frame["pred_mean"] = mean_forecast
        frames.append(frame)

    if not frames:
        raise ValueError("no refit block had usable training history")
    predictions = pd.concat(frames).sort_index()
    coefficients = pd.DataFrame(coefficient_rows).set_index(["refit", "ticker"])
    return predictions, coefficients


def ar1_book(preds: pd.DataFrame, params, ppy: int = 52,
             initial_capital_usd: float = 100_000.0,
             vol_history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Trade the AR(1) forecasts with the study's own forecast-vol book.

    The baseline is only meaningful next to the fitted models if it faces the same
    costs and the same book, so the predictions from :func:`ar1_forecast` are handed
    to the identical rule the study trades (``forecast_vol_scaled``): weights scale
    each forecast by its own rolling volatility, positions rebalance on the same
    cadence, and the same broker commissions are charged.

    Args:
        preds: Output of :func:`ar1_forecast` (``pred`` and ``fwd_ret``).
        params: ``ForecastVolParams`` of the book (vol window, rebalance, broker).
        ppy: Periods per year of the panel frequency.
        initial_capital_usd: Starting equity.
        vol_history: Earlier out-of-sample AR(1) predictions used only to seed the
            rolling volatility estimate, so the first bar already has a history.

    Returns:
        Portfolio frame from ``build_forecast_vol_portfolio``.
    """
    from portfolio import build_forecast_vol_portfolio  # deferred: keeps the module leaf
    return build_forecast_vol_portfolio(preds, params, initial_capital_usd,
                                        ppy=ppy, vol_history=vol_history)


def ar1_warmup(panel: pd.DataFrame, target_col: str, bars: int, *,
               refit_every: str = "QE", window: int | None = 416,
               embargo: int | None = None, min_obs: int = 40) -> pd.DataFrame:
    """AR(1) predictions on the trailing ``bars`` dates of ``panel`` (vol seed only).

    The caller passes the history ending where the evaluation window begins; the
    returned rows are never traded — they only seed the book's rolling volatility, so
    the first evaluated bar already has a history. An empty frame is returned when
    there is not enough history to fit.
    """
    dates = panel.index.get_level_values("date")
    unique_dates = dates.unique().sort_values()
    embargo = target_horizon(target_col) if embargo is None else embargo
    if len(unique_dates) <= bars + embargo:
        return panel.iloc[0:0].copy().assign(pred=pd.Series(dtype=float))
    # Fit once on everything up to the evaluation window, then keep the trailing bars
    # that actually received a prediction: a refit block boundary can fall after the
    # cut, so slicing the input dates by position is not reliable.
    preds, _ = ar1_forecast(panel, target_col, refit_every=refit_every,
                            window=window, embargo=embargo, min_obs=min_obs)
    warm_dates = preds.index.get_level_values("date").unique().sort_values()[-bars:]
    return preds[preds.index.get_level_values("date").isin(warm_dates)]


def ar1_baseline(history: pd.DataFrame, eval_panel: pd.DataFrame, config,
                 vol_bars: int = 48) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit and trade the AR(1) baseline on one window of a split.

    One call produces both the forecasts and the book, so a caller that wants the
    baseline in its figures and tables does not have to repeat the split plumbing.
    The forecast is fitted on ``history`` (strictly earlier bars) and evaluated on
    ``eval_panel``; the book uses the ``config``'s own ``forecast_vol`` settings, so
    it is the same rule, costs and cadence the fitted models trade. The rolling
    volatility is seeded from the trailing ``vol_bars`` AR(1) predictions of
    ``history`` only, never from the evaluated window.

    Args:
        history: Panel of bars strictly before the evaluated window.
        eval_panel: Panel rows to forecast and trade.
        config: Pipeline config; supplies ``forecast_vol``, ``refit_every``,
            ``window_train_bars``, ``target_col``, ``embargo`` and ``ppy``.
        vol_bars: Trailing bars of ``history`` used to seed the volatility estimate.

    Returns:
        ``(preds, book)`` — the evaluated panel with ``pred`` / ``pred_mean``, and
        the ``forecast_vol_scaled`` book frame.
    """
    kwargs = dict(refit_every=config.refit_every, window=config.window_train_bars,
                  embargo=config.embargo)
    panel = pd.concat([history, eval_panel])
    preds, _ = ar1_forecast(panel, config.target_col, **kwargs)
    start, end = eval_panel.index.get_level_values("date").min(), \
        eval_panel.index.get_level_values("date").max()
    dates = preds.index.get_level_values("date")
    preds = preds[(dates >= start) & (dates <= end)]
    if preds.empty:
        raise ValueError("the AR(1) baseline produced no predictions in the window")

    warmup = ar1_warmup(history, config.target_col, vol_bars, **kwargs)
    book = ar1_book(preds, config.forecast_vol, ppy=config.ppy,
                    vol_history=None if warmup.empty else warmup)
    return preds, book


def r2_against(preds: pd.DataFrame, benchmark: str = "pred_mean",
               target_col: str = TARGET_COL) -> float:
    """OOS R² of ``pred`` relative to another forecast column on the same rows.

    ``1 - sum((y - pred)^2) / sum((y - benchmark)^2)``: positive means the AR(1)
    forecast has lower squared error than the benchmark forecast. Rows where either
    side is missing are dropped.
    """
    require_columns(preds, ["pred", benchmark, target_col], "predictions panel")
    actual = preds[target_col].to_numpy(dtype=float)
    model = preds["pred"].to_numpy(dtype=float)
    bench = preds[benchmark].to_numpy(dtype=float)
    finite = np.isfinite(actual) & np.isfinite(model) & np.isfinite(bench)
    sse_model = float(np.sum((actual[finite] - model[finite]) ** 2))
    sse_bench = float(np.sum((actual[finite] - bench[finite]) ** 2))
    return float(1 - sse_model / sse_bench) if sse_bench > 0 else float("nan")


def r2_against_by_year(preds: pd.DataFrame, benchmark: str = "pred_mean",
                       target_col: str = TARGET_COL) -> pd.Series:
    """``r2_against`` split by calendar year."""
    years = preds.index.get_level_values("date").year
    return preds.groupby(years).apply(
        lambda group: r2_against(group, benchmark=benchmark, target_col=target_col)
    )
