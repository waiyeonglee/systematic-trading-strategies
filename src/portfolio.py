"""Portfolio construction: predicted returns / trend signals in, holdings and returns out.

Two rules are implemented and picked by ``portfolio.strategy``:

* ``forecast_vol_scaled`` divides every predicted return by its own forecast standard
  deviation, normalises the signals to a gross exposure of 1.0 and trades them long and
  short (Equation 15 of Chinco, Clark-Joseph & Ye, 2019). Each rebalance is only executed
  when its net expected gain beats its total trading cost.
* ``trend_long_flat`` longs every asset whose trend rule is on, weights them by inverse
  realized volatility to a gross exposure of 1.0, holds cash when nothing is on-signal
  and rebalances on a fixed cadence with no cost gate (documented reference).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ForecastVolParams:
    """Settings of the ``forecast_vol_scaled`` rule.

    Attributes:
        vol_window: Bars of past predictions in the forecast-volatility window.
        vol_min_periods: Past predictions needed before any volatility is usable.
        rebalance: How often the holdings are traded back to the target weights:
            ``"daily"``, ``"weekly"``, ``"4w"`` or ``"12w"``.
        broker: ``"moomoo"``; selects the commission model.
        cost_bps: Extra proportional cost charged on every executed order, in basis points.
        cost_gate_multiple: An order leg is only executed when its expected gain exceeds
            this multiple of the order's full cost. ``1.0`` means the trade must pay for
            itself; higher values demand a larger edge.
        vol_target: Annualized volatility target for the vol-targeting overlay, e.g.
            ``0.10`` = 10% per year; ``None`` disables the overlay. When enabled, each
            rebalance scales the target weights by ``min(vol_target_cap,
            vol_target / realized_vol_book)``.
        vol_target_cap: Maximum gross-exposure scale. ``1.0`` means the overlay only ever
            de-risks; values above 1.0 allow leverage in low-volatility periods.
        vol_target_rf: Annualized risk-free rate credited to the de-risked (cash) sleeve,
            simulated as a constant, e.g. ``0.03`` = 3% per year.
        vol_target_lev_fee: Annualized holding fee charged on the borrowed amount only
            (gross exposure above 100%), e.g. ``0.068`` = 6.8%/yr.
    """

    vol_window: int = 24
    vol_min_periods: int = 20
    rebalance: str = "4w"
    broker: str = "moomoo"
    cost_bps: float = 0.0
    cost_gate_multiple: float = 1.0
    vol_target: float | None = None
    vol_target_cap: float = 1.0
    vol_target_rf: float = 0.0
    vol_target_lev_fee: float = 0.0

    def __post_init__(self) -> None:
        if self.rebalance not in ("daily", "weekly", "4w", "12w"):
            raise ValueError(
                f"rebalance must be 'daily', 'weekly', '4w' or '12w', got {self.rebalance!r}"
            )
        if self.broker != "moomoo":
            raise ValueError(f"broker must be 'moomoo', got {self.broker!r}")
        if self.vol_window < 2 or not 2 <= self.vol_min_periods <= self.vol_window:
            raise ValueError(
                "volatility needs 2 <= vol_min_periods <= vol_window, got "
                f"vol_min_periods={self.vol_min_periods}, vol_window={self.vol_window}"
            )
        if self.cost_bps < 0:
            raise ValueError("cost_bps cannot be negative")
        if self.cost_gate_multiple < 0:
            raise ValueError("cost_gate_multiple cannot be negative")
        if self.vol_target is not None and self.vol_target <= 0:
            raise ValueError("vol_target must be positive (annualized), or None to disable")
        if self.vol_target_cap <= 0:
            raise ValueError("vol_target_cap must be positive")
        if self.vol_target_rf < 0:
            raise ValueError("vol_target_rf cannot be negative (annualized risk-free rate)")
        if self.vol_target_lev_fee < 0:
            raise ValueError("vol_target_lev_fee cannot be negative (annualized leverage fee)")

    @classmethod
    def from_mapping(cls, block: Mapping[str, Any] | None) -> ForecastVolParams:
        """Build the settings from a YAML block, ignoring keys set to ``None``.

        Raises:
            ValueError: if the block contains a key this dataclass does not define.
        """
        block = dict(block or {})
        unknown = set(block) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"unknown forecast_vol_scaled setting(s): {sorted(unknown)}")
        return cls(**{key: value for key, value in block.items() if value is not None})


@dataclass(frozen=True)
class TrendParams:
    """Settings of the ``trend_long_flat`` rule (documented reference)."""

    vol_window: int = 24
    vol_min_periods: int = 20
    rebalance: str = "4w"
    broker: str = "moomoo"
    cost_bps: float = 0.0

    def __post_init__(self) -> None:
        if self.rebalance not in ("daily", "weekly", "4w", "12w"):
            raise ValueError(
                f"rebalance must be 'daily', 'weekly', '4w' or '12w', got {self.rebalance!r}"
            )
        if self.broker != "moomoo":
            raise ValueError(f"broker must be 'moomoo', got {self.broker!r}")
        if self.vol_window < 2 or not 2 <= self.vol_min_periods <= self.vol_window:
            raise ValueError(
                "volatility needs 2 <= vol_min_periods <= vol_window, got "
                f"vol_min_periods={self.vol_min_periods}, vol_window={self.vol_window}"
            )
        if self.cost_bps < 0:
            raise ValueError("cost_bps cannot be negative")

    @classmethod
    def from_mapping(cls, block: Mapping[str, Any] | None) -> TrendParams:
        """Build the settings from a YAML block, ignoring keys set to ``None``.

        Raises:
            ValueError: if the block contains a key this dataclass does not define.
        """
        block = dict(block or {})
        unknown = set(block) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"unknown trend_long_flat setting(s): {sorted(unknown)}")
        return cls(**{key: value for key, value in block.items() if value is not None})


def moomoo_commission(trade_value_usd: float) -> float:
    """Commission for one Moomoo order: 0.03% of the trade with a USD 0.77 floor, plus 9% GST."""
    value = abs(float(trade_value_usd))
    if value <= 0:
        return 0.0
    return max(0.0003 * value, 0.77) * 1.09


def order_cost_usd(trade_value_usd: float, params: ForecastVolParams) -> float:
    """Cost of one executed order: broker commission plus ``cost_bps`` on the trade value.

    An order of zero is not executed and costs nothing, so no commission floor applies.
    """
    value = abs(float(trade_value_usd))
    if value <= 0:
        return 0.0
    return moomoo_commission(value) + params.cost_bps / 1e4 * value


def historical_forecast_vol(predictions: pd.DataFrame,
                            params: ForecastVolParams) -> pd.DataFrame:
    """Rolling standard deviation of each asset's past predictions.

    The value dated ``t`` is computed from predictions dated strictly before ``t``
    (a one-bar shift), so the forecast used to trade on ``t`` never enters its own
    volatility estimate, and it stays NaN until ``vol_min_periods`` past predictions
    exist for that asset.

    Raises:
        ValueError: if the timestamps are not unique.
    """
    wide = predictions.sort_index().astype(float)
    if not wide.index.is_unique:
        raise ValueError("prediction timestamps must be unique")
    return wide.shift(1).rolling(params.vol_window, min_periods=params.vol_min_periods).std(ddof=1)


def historical_realized_vol(same_bar_returns: pd.DataFrame,
                            params: TrendParams) -> pd.DataFrame:
    """Rolling standard deviation of each asset's past same-bar returns (one-bar shift).

    Raises:
        ValueError: if the timestamps are not unique.
    """
    wide = same_bar_returns.sort_index().astype(float)
    if not wide.index.is_unique:
        raise ValueError("return timestamps must be unique")
    return wide.shift(1).rolling(params.vol_window, min_periods=params.vol_min_periods).std(ddof=1)


def trend_target_weights(signals: pd.Series,
                         realized_vol: pd.Series) -> pd.Series:
    """Long/flat target weights for one decision date.

    Every asset with a valid trend signal and finite positive realized volatility is
    weighted by ``1 / vol``; the weights are normalised so the absolute weights sum to
    1.0 (100% gross, no leverage).

    Only on-signal assets (``signals > 0.5``) get weight; off-signal assets are zero
    and when nothing is on the target is all cash.
    """
    vol_floor = 1e-5
    sig = pd.to_numeric(signals, errors="coerce").astype(float)
    vols = pd.to_numeric(realized_vol.reindex(sig.index), errors="coerce").astype(float)
    valid = sig.notna() & vols.notna() & np.isfinite(vols) & (vols > 0)
    weights = pd.Series(0.0, index=sig.index)
    on = valid & (sig > 0.5)
    weights[on] = 1.0 / vols[on].clip(lower=vol_floor)
    gross = weights.abs().sum()
    return weights / gross if gross > 0 else weights


def mark_to_market_weights(weights: pd.Series, returns: pd.Series) -> pd.Series:
    """Drift weights to just before the next rebalance; uninvested cash earns zero."""
    realised = returns.reindex(weights.index).fillna(0.0).astype(float)
    portfolio_return = float((weights * realised).sum())
    if 1.0 + portfolio_return <= 0:
        raise ValueError("portfolio is insolvent: weights and returns wiped out the equity")
    return weights * (1.0 + realised) / (1.0 + portfolio_return)


def orders_for_target(current_weights: pd.Series, desired_weights: pd.Series,
                      equity_usd: float) -> pd.Series:
    """Order values in USD that move the current book to ``desired_weights``.

    A flip from long to short is a single order whose size covers both the closing and the
    opening leg. Whether the rebalance is executed at all is decided by
    :func:`cost_gated_orders`; until then every leg is returned as a candidate.
    """
    tickers = current_weights.index.union(desired_weights.index)
    return (desired_weights.reindex(tickers, fill_value=0.0)
            - current_weights.reindex(tickers, fill_value=0.0)) * equity_usd


def rebalance_dates(dates: pd.DatetimeIndex, every: str = "4w") -> pd.DatetimeIndex:
    """Dates on which the targets are refreshed and the weight drift is traded away.

    ``"daily"`` (every bar), ``"weekly"`` (first bar of each week), ``"4w"`` (first
    bar, then every 4th bar — every 4 weeks on the weekly panel) or ``"12w"``
    (first bar, then every 12th bar). The n-week cadences are horizon-aligned:
    each position is held exactly n bars, matching the n-week look-ahead labels.
    """
    dates = pd.DatetimeIndex(dates)
    if every == "daily":
        return dates
    if every in ("4w", "12w"):  # every 4th / 12th bar (horizon-aligned cadence)
        return dates[::4 if every == "4w" else 12]
    period = {"weekly": "W"}[every]
    first_of_period = pd.Series(dates, index=dates).groupby(dates.to_period(period)).first()
    return pd.DatetimeIndex(first_of_period.to_numpy())


def _vol_target_scale(book_rets: list[float], params: ForecastVolParams, ppy: int) -> float:
    """Vol-targeting scale factor from the book's trailing realized volatility.

    ``scale = min(vol_target_cap, vol_target / vol_ann)`` with ``vol_ann`` the annualized
    std of the trailing weekly net returns. Returns 1.0 (no de-risking) until at least
    ``vol_min_periods`` bars of book history exist. The cap is applied last, so with
    ``vol_target_cap = 1.0`` the overlay never leverages.
    """
    if params.vol_target is None:
        return 1.0
    window = book_rets[-params.vol_window:]
    if len(window) < params.vol_min_periods or len(set(window)) < 2:
        return 1.0
    vol_ann = float(np.std(window, ddof=1)) * (ppy ** 0.5)
    if not np.isfinite(vol_ann) or vol_ann <= 0:
        return 1.0
    return min(params.vol_target_cap, params.vol_target / vol_ann)


def forecast_vol_frame(preds: pd.DataFrame, params: ForecastVolParams,
                       vol_history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Rolling forecast volatility aligned to the prediction dates.

    Depends only on the predictions, the warmup history and the volatility window, so
    a study can compute it once per (candidate, vol_window) and reuse it across every
    book row with the same window.

    Args:
        preds: Panel with ``pred``.
        params: Volatility settings (``vol_window``, ``vol_min_periods``).
        vol_history: Optional earlier out-of-sample predictions used only to seed it.

    Returns:
        Wide frame of rolling standard deviations, indexed like ``preds``.

    Raises:
        ValueError: if ``vol_history`` lacks ``pred`` or overlaps the portfolio dates.
    """
    predictions = preds["pred"].unstack("ticker").sort_index()
    if vol_history is not None and not vol_history.empty:
        if "pred" not in vol_history:
            raise ValueError("vol_history must contain a 'pred' column")
        historical = vol_history["pred"].unstack("ticker").sort_index()
        if historical.index.intersection(predictions.index).size:
            raise ValueError("vol_history dates must precede portfolio dates")
        vol_inputs = pd.concat([historical, predictions]).sort_index()
    else:
        vol_inputs = predictions
    return historical_forecast_vol(vol_inputs, params).reindex(predictions.index)


def _commission_usd(values: "np.ndarray", params: ForecastVolParams) -> float:
    """Vectorised :func:`order_cost_usd` over one bar's order vector."""
    magnitude = np.abs(np.asarray(values, dtype=float))
    mask = magnitude > 0
    if not mask.any():
        return 0.0
    live = magnitude[mask]
    per_order = np.maximum(0.0003 * live, 0.77) * 1.09
    return float(per_order.sum() + params.cost_bps / 1e4 * live.sum())


def build_forecast_vol_portfolio(preds: pd.DataFrame, params: ForecastVolParams,
                                 initial_capital_usd: float = 100_000.0,
                                 ppy: int = 52,
                                 apply_vol_target: bool = True,
                                 vol_history: pd.DataFrame | None = None,
                                 forecast_vol: pd.DataFrame | None = None) -> pd.DataFrame:
    """Trade forecast-volatility-scaled weights, charging broker costs per order.

    Decisions lag the market by one bar: on a rebalance bar ``d`` the target weights are
    built from the prediction dated ``d-1``, which matches the training label
    ``price[d+1] / price[d] - 1`` that the model fitted at ``d-1``. The book earns each
    bar's ``fwd_ret``, so nothing from the future is used. Between rebalances positions
    drift; on each rebalance date the book is marked to market and traded back to the
    target, which restores gross exposure to 100%. The vectorized cost gate keeps the
    full set of legs only when the reallocation's net expected gain beats its total cost.
    The first build from cash is exempt: deploying forgoes nothing and always executes.

    With ``params.vol_target`` set, the target weights are additionally scaled at every
    rebalance by :func:`_vol_target_scale`. The difference between the scaled gross and
    100% sits in a cash sleeve earning ``vol_target_rf``; with the default cap 1.0 the
    overlay only ever de-risks. ``apply_vol_target=False`` (used for model selection on
    the validation window) reproduces the plain ADR-07 book exactly.

    The simulation is a numpy state machine over the ``(date, ticker)`` matrices (the
    weights path is a drift recurrence, so only the bars need a loop); everything that
    can be precomputed — the target weights of every bar, the rebalance calendar, the
    commissions — is computed up front, which makes one book O(bars) rather than
    O(bars x pandas calls).

    Args:
        preds: Panel with ``pred`` and ``fwd_ret``, as returned by
            :func:`models.walk_forward_predict`, indexed by ``(date, ticker)``.
        params: Volatility, rebalance cadence, broker, cost and vol-target settings.
        initial_capital_usd: Starting equity; the broker minimums are absolute dollars.
        ppy: Periods per year, used to annualize the book's realized volatility.
        apply_vol_target: Whether the vol-target overlay is applied. Selection scoring on
            the validation window passes ``False``; the traded test book uses ``True``.
        vol_history: Optional earlier out-of-sample prediction panel used only to seed
            forecast volatility. Its returns are never traded or included in this book.
        forecast_vol: Optional precomputed :func:`forecast_vol_frame` output, so a study
            can share it across every book row with the same volatility window.

    Returns:
        Frame indexed by date with ``gross_ret``, ``turnover``, ``cost``, ``net_ret``,
        ``n_holdings``, ``holdings`` plus ``n_long``, ``n_short``, ``gross_exposure``,
        ``net_exposure``, ``equity_usd``, ``vol_scale``, ``leverage_blocked`` and
        ``rebalanced``. The frame also carries two per-date, per-ticker attachments
        (set with ``object.__setattr__`` so they never collide with column access):
        ``weights_wide`` (the portfolio weights held into the next bar) and
        ``orders_wide`` (executed order value in USD, signed: positive buys,
        negative sells, zero on non-rebalance bars).

    Raises:
        ValueError: if the panel lacks ``pred`` or ``fwd_ret``, the starting equity is not
            positive, or the portfolio is wiped out.
    """
    if initial_capital_usd <= 0:
        raise ValueError("initial_capital_usd must be positive")
    missing = sorted({"pred", "fwd_ret"} - set(preds.columns))
    if missing:
        raise ValueError(f"predictions panel is missing columns: {missing}")

    predictions = preds["pred"].unstack("ticker").sort_index()
    returns = preds["fwd_ret"].unstack("ticker").reindex(predictions.index)
    dates = predictions.index
    tickers = predictions.columns
    bars, width = len(dates), len(tickers)

    decisions = predictions.shift(1).to_numpy(dtype=float)
    realised = returns.to_numpy(dtype=float)
    realised = np.nan_to_num(realised, nan=0.0)          # returns.loc[date].fillna(0.0)
    if forecast_vol is None:
        forecast_vol = forecast_vol_frame(preds, params, vol_history)
    vols = forecast_vol.reindex(dates).to_numpy(dtype=float)

    # Precompute the unscaled target weights of every bar: predictions are divided by
    # their forecast volatility, then normalised to gross 1.0 (target_weights).
    eligible = (np.isfinite(decisions) & np.isfinite(vols) & (vols > 0)
                & (np.abs(decisions) > 0.0))
    ratio = np.where(eligible, decisions / np.clip(vols, 1e-5, None), 0.0)
    gross_ratio = np.abs(ratio).sum(axis=1, keepdims=True)
    targets = np.divide(ratio, gross_ratio, out=np.zeros_like(ratio),
                        where=gross_ratio > 0)

    is_rebalance = np.zeros(bars, dtype=bool)
    rebalance_set = set(rebalance_dates(dates, params.rebalance))
    is_rebalance[:] = [date in rebalance_set for date in dates]

    with_overlay = apply_vol_target and params.vol_target is not None
    rf_week = (1.0 + (params.vol_target_rf or 0.0)) ** (1.0 / ppy) - 1.0  # cash sleeve
    lev_week = (1.0 + (params.vol_target_lev_fee or 0.0)) ** (1.0 / ppy) - 1.0
    fee_horizon = (1.0 + (params.vol_target_lev_fee or 0.0)) ** (12.0 / ppy) - 1.0

    weights = np.zeros(width)
    equity = float(initial_capital_usd)
    book_rets: list[float] = []          # trailing net returns for the vol-target overlay

    gross_ret = np.zeros(bars)
    turnover = np.zeros(bars)
    cost_frac = np.zeros(bars)
    net_ret = np.zeros(bars)
    equity_path = np.zeros(bars)
    vol_scale_path = np.ones(bars)
    leverage_blocked = np.zeros(bars, dtype=int)
    weights_path = np.zeros((bars, width))
    orders_path = np.zeros((bars, width))

    for index in range(bars):
        if index:
            # mark to market: weights drift with the previous bar's returns
            previous = realised[index - 1]
            portfolio_return = float(np.dot(weights, previous))
            if 1.0 + portfolio_return <= 0:
                raise ValueError("portfolio is insolvent: weights and returns wiped out "
                                 "the equity")
            weights = weights * (1.0 + previous) / (1.0 + portfolio_return)

        equity_before = equity
        vol_scale = 1.0
        blocked = 0
        orders = np.zeros(width)

        if is_rebalance[index]:
            target = targets[index].copy()
            if with_overlay:
                vol_scale = _vol_target_scale(book_rets, params, ppy)
                if vol_scale > 1.0:   # leverage fee gate: never borrow to amplify a
                    # signal that cannot pay its own holding fee over the horizon
                    if float(np.dot(target, decisions[index])) < fee_horizon:
                        vol_scale = 1.0
                        blocked = 1
                target *= vol_scale
            orders = (target - weights) * equity_before
            if np.abs(weights).sum() > 0.0:   # a build from cash is exempt from the gate
                decision = decisions[index]
                if np.isnan(decision).any():   # a missing signal freezes the whole book
                    orders = np.zeros(width)
                else:
                    net_gain = float(np.dot(orders, decision))
                    if not net_gain > params.cost_gate_multiple * _commission_usd(
                            orders, params):
                        orders = np.zeros(width)
            orders = np.nan_to_num(orders, nan=0.0)
            costs = _commission_usd(orders, params)
            weights = weights + orders / equity_before
        else:
            costs = 0.0

        bar_return = float(np.dot(weights, realised[index]))
        cost = costs / equity_before
        gross_exposure = float(np.abs(weights).sum())
        cash_fraction = max(0.0, 1.0 - gross_exposure)
        leverage_fraction = max(0.0, gross_exposure - 1.0)
        net = (bar_return - cost + cash_fraction * rf_week
               - leverage_fraction * lev_week)

        equity = equity_before * (1.0 + net)
        if equity <= 0:
            raise ValueError(f"portfolio equity fell to {equity:.2f} USD on {dates[index]}")

        gross_ret[index] = bar_return
        turnover[index] = float(np.abs(orders).sum()) / equity_before
        cost_frac[index] = cost
        net_ret[index] = net
        equity_path[index] = equity
        vol_scale_path[index] = vol_scale
        leverage_blocked[index] = blocked
        weights_path[index] = weights
        orders_path[index] = orders
        book_rets.append(net)

    held = np.abs(weights_path) > 1e-12
    frame = pd.DataFrame({
        "gross_ret": gross_ret,
        "turnover": turnover,
        "cost": cost_frac,
        "net_ret": net_ret,
        "n_holdings": held.sum(axis=1).astype(int),
        "holdings": [",".join(sorted(np.asarray(tickers)[row])) for row in held],
        "n_long": ((weights_path > 0) & held).sum(axis=1).astype(int),
        "n_short": ((weights_path < 0) & held).sum(axis=1).astype(int),
        "gross_exposure": np.abs(np.where(held, weights_path, 0.0)).sum(axis=1),
        "net_exposure": np.where(held, weights_path, 0.0).sum(axis=1),
        "equity_usd": equity_path,
        "vol_scale": vol_scale_path,
        "leverage_blocked": leverage_blocked,
        "rebalanced": is_rebalance.astype(int),
    }, index=dates)
    weights_wide = pd.DataFrame(weights_path, index=dates, columns=tickers)
    orders_wide = pd.DataFrame(orders_path, index=dates, columns=tickers)
    object.__setattr__(frame, "weights_wide", weights_wide)
    object.__setattr__(frame, "orders_wide", orders_wide)
    return frame


def build_trend_portfolio(preds: pd.DataFrame, params: TrendParams,
                          initial_capital_usd: float = 100_000.0,
                          vol_history: pd.DataFrame | None = None) -> pd.DataFrame:
    """Trade the trend rule, charging broker costs per executed order.

    Same one-bar decision lag as the forecast-vol rule: on a rebalance bar ``d`` the
    target weights are built from the rule signal dated ``d-1`` and from realized
    volatility measured on returns dated strictly before ``d``; the book earns each
    bar's ``fwd_ret``. Unlike the forecast-vol rule there is no cost gate: the 4-week
    cadence keeps turnover low, so every scheduled rebalance executes and pays its costs.

    Args:
        preds: Panel with ``pred`` (the signed trend signal: -1 short / 0 flat / +1
            long), ``fwd_ret`` and ``return_1bar`` (same-bar returns used to measure
            realized volatility), indexed by ``(date, ticker)``.
        params: Volatility window, rebalance cadence, broker and cost settings.
        initial_capital_usd: Starting equity; the broker minimums are absolute dollars.
        shorting: Whether negative signals may be shorted (long/short variant).
        vol_history: Optional earlier panel whose ``return_1bar`` values seed realized
            volatility. Its returns are never traded or included in this book.

    Returns:
        Frame indexed by date with ``gross_ret``, ``turnover``, ``cost``, ``net_ret``,
        ``n_holdings``, ``holdings`` plus ``n_long``, ``n_short``, ``gross_exposure``,
        ``net_exposure``, ``equity_usd`` and ``rebalanced``. The frame also carries the
        ``weights_wide`` / ``orders_wide`` per-date, per-ticker attachments (same
        convention as :func:`build_forecast_vol_portfolio`).

    Raises:
        ValueError: if the panel lacks ``pred``, ``fwd_ret`` or ``return_1bar``, the
            starting equity is not positive, or the portfolio is wiped out.
    """
    if initial_capital_usd <= 0:
        raise ValueError("initial_capital_usd must be positive")
    missing = sorted({"pred", "fwd_ret", "return_1bar"} - set(preds.columns))
    if missing:
        raise ValueError(f"trend predictions panel is missing columns: {missing}")

    signals = preds["pred"].unstack("ticker").sort_index()
    returns = preds["fwd_ret"].unstack("ticker").reindex(signals.index)
    same_bar = preds["return_1bar"].unstack("ticker").reindex(signals.index)
    decisions = signals.shift(1)   # bar d trades on the signal dated bar d-1
    if vol_history is not None and not vol_history.empty:
        if "return_1bar" not in vol_history:
            raise ValueError("vol_history must contain a 'return_1bar' column")
        historical = vol_history["return_1bar"].unstack("ticker").sort_index()
        if historical.index.intersection(signals.index).size:
            raise ValueError("vol_history dates must precede portfolio dates")
        vol_inputs = pd.concat([historical, same_bar]).sort_index()
    else:
        vol_inputs = same_bar
    realized_vol = historical_realized_vol(vol_inputs, params).reindex(signals.index)
    rebalances = set(rebalance_dates(signals.index, params.rebalance))

    equity = initial_capital_usd
    weights = pd.Series(0.0, index=signals.columns)
    previous_date = None
    rows = []
    weight_rows: list[pd.Series] = []
    order_rows: list[pd.Series] = []
    for date in signals.index:
        if previous_date is not None:
            weights = mark_to_market_weights(weights, returns.loc[previous_date])
        equity_before = equity

        if date in rebalances:
            target = trend_target_weights(decisions.loc[date], realized_vol.loc[date])
            orders = orders_for_target(weights, target, equity_before).fillna(0.0)
            costs = float(sum(order_cost_usd(value, params) for value in orders if value != 0))
            weights = (weights.add(orders / equity_before, fill_value=0.0)
                       .reindex(signals.columns, fill_value=0.0))
        else:
            orders = pd.Series(0.0, index=signals.columns)
            costs = 0.0

        gross = float((weights * returns.loc[date].fillna(0.0)).sum())
        cost = costs / equity_before
        net = gross - cost

        equity = equity_before * (1.0 + net)
        if equity <= 0:
            raise ValueError(f"portfolio equity fell to {equity:.2f} USD on {date}")

        held = weights[weights.abs() > 1e-12]
        rows.append(dict(date=date, gross_ret=gross,
                         turnover=float(orders.abs().sum()) / equity_before, cost=cost,
                         net_ret=net, n_holdings=len(held),
                         holdings=",".join(sorted(held.index)),
                         n_long=int((held > 0).sum()), n_short=int((held < 0).sum()),
                         gross_exposure=float(held.abs().sum()),
                         net_exposure=float(held.sum()), equity_usd=equity,
                         rebalanced=int(date in rebalances)))
        previous_date = date
        weight_rows.append(weights.copy())
        order_rows.append(orders.copy())

    frame = pd.DataFrame(rows).set_index("date")
    weights_wide = pd.DataFrame(weight_rows, index=frame.index,
                                columns=signals.columns).fillna(0.0)
    orders_wide = pd.DataFrame(order_rows, index=frame.index,
                               columns=signals.columns).fillna(0.0)
    object.__setattr__(frame, "weights_wide", weights_wide)
    object.__setattr__(frame, "orders_wide", orders_wide)
    return frame


def _trend_long_flat(preds: pd.DataFrame, config, *, apply_vol_target: bool = True,
                     vol_history: pd.DataFrame | None = None,
                     forecast_vol: pd.DataFrame | None = None) -> pd.DataFrame:
    """Long the on-signal assets, weighted by inverse realized volatility, else cash."""
    return build_trend_portfolio(preds, config.trend, config.initial_capital_usd,
                                 vol_history=vol_history)


def _forecast_vol_scaled(preds: pd.DataFrame, config, *, apply_vol_target: bool = True,
                         vol_history: pd.DataFrame | None = None,
                         forecast_vol: pd.DataFrame | None = None) -> pd.DataFrame:
    """Scale predictions by their own forecast volatility and trade them long and short.

    ``forecast_vol`` lets a study reuse the rolling volatility across book rows that
    share the same volatility window (it depends only on the predictions).
    """
    from utils.config import periods_per_year  # deferred: utils.config is a leaf module
    ppy = periods_per_year(config.freq)
    return build_forecast_vol_portfolio(preds, config.forecast_vol, config.initial_capital_usd,
                                        ppy=ppy, apply_vol_target=apply_vol_target,
                                        vol_history=vol_history,
                                        forecast_vol=forecast_vol)


STRATEGIES: dict[str, Callable[..., pd.DataFrame]] = {
    "forecast_vol_scaled": _forecast_vol_scaled,
    "trend_long_flat": _trend_long_flat,
}

STRATEGY_CHOICES = tuple(STRATEGIES)


def configured_strategies(strategy: str) -> tuple[str, ...]:
    """Rules to trade for a configured ``portfolio.strategy`` value.

    Raises:
        ValueError: if ``strategy`` is not one of :data:`STRATEGY_CHOICES`.
    """
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown portfolio strategy {strategy!r}; use one of {STRATEGY_CHOICES}")
    return (strategy,)


def build_positions(preds: pd.DataFrame, config, strategy: str | None = None,
                    *, apply_vol_target: bool = True,
                    vol_history: pd.DataFrame | None = None,
                    forecast_vol: pd.DataFrame | None = None) -> pd.DataFrame:
    """Turn predictions into portfolio weights using one of the configured strategies.

    Args:
        preds: Panel with ``pred`` and ``fwd_ret``.
        config: Any object exposing ``strategy`` and that strategy's parameters.
        strategy: Rule to run; defaults to the configured one.
        apply_vol_target: Whether the vol-target overlay is applied to the forecast-vol
            book (ignored by the trend strategies). Model selection on the validation
            window passes ``False``.
        vol_history: Optional earlier out-of-sample history used only to seed the
            strategy's rolling volatility estimate.
        forecast_vol: Optional precomputed rolling volatility (forecast-vol strategies
            only), so callers can share it across book rows with the same window.

    Returns:
        Portfolio frame from the selected strategy.

    Raises:
        ValueError: if ``config.strategy`` is not registered.
    """
    rule = strategy or configured_strategies(config.strategy)[0]
    return STRATEGIES[rule](preds, config, apply_vol_target=apply_vol_target,
                            vol_history=vol_history, forecast_vol=forecast_vol)


def build_all_positions(preds: pd.DataFrame, config, *, apply_vol_target: bool = True,
                        vol_history: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Run configured strategies, optionally seeding volatility from prior history."""
    return {name: STRATEGIES[name](preds, config, apply_vol_target=apply_vol_target,
                                   vol_history=vol_history)
            for name in configured_strategies(config.strategy)}
