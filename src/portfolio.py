"""Portfolio construction: predicted returns in, per-date holdings and returns out.

Two rules are implemented and picked by ``portfolio.strategy`` in ``config.yaml``:

* ``top_k`` holds the ``k`` best predicted returns at equal weight and swaps a holding only
  when the challenger beats it by more than a cost-covering gate.
* ``forecast_vol_scaled`` divides every predicted return by its own forecast standard
  deviation, normalises the signals to a gross exposure of 1.0 and trades them long and
  short (Equation 15 of Chinco, Clark-Joseph & Ye, 2019).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import pandas as pd


def build_portfolio(
    preds: pd.DataFrame,
    top_k: int = 5,
    gate: float = 0.0,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Trade the top-k predicted assets with a cost-aware swap gate.

    On each date: fill empty slots with the highest predicted returns, then swap while the
    best outsider beats the weakest holding by more than ``gate`` (in return units), weight
    every holding at ``1/k``, and charge ``turnover * cost_bps / 1e4``.

    Args:
        preds: Panel with ``pred`` and ``fwd_ret``, as returned by
            :func:`models.walk_forward_predict`.
        top_k: Number of assets held.
        gate: Minimum predicted-return gap needed to swap.
        cost_bps: One-way transaction cost per unit of turnover, in basis points.

    Returns:
        Frame indexed by date with ``gross_ret``, ``turnover``, ``cost``, ``net_ret``,
        ``n_holdings`` and ``holdings``.
    """
    predictions = preds["pred"].unstack("ticker")
    returns = preds["fwd_ret"].unstack("ticker")

    holdings: set[str] = set()
    previous_weights = pd.Series(0.0, index=predictions.columns)
    rows = []
    for date, scores in predictions.iterrows():
        scores = scores.dropna()
        holdings &= set(scores.index)

        for ticker in scores.drop(list(holdings)).sort_values(ascending=False).index:
            if len(holdings) >= top_k:
                break
            holdings.add(ticker)

        while True:
            outsiders = scores.drop(list(holdings))
            if outsiders.empty:
                break
            best_outside, weakest_inside = outsiders.idxmax(), scores[list(holdings)].idxmin()
            if scores[best_outside] - scores[weakest_inside] > gate:
                holdings.remove(weakest_inside)
                holdings.add(best_outside)
            else:
                break

        weights = pd.Series(0.0, index=predictions.columns)
        weights[list(holdings)] = 1.0 / len(holdings)
        turnover = (weights - previous_weights).abs().sum()
        gross = float((weights * returns.loc[date].fillna(0.0)).sum())
        cost = turnover * cost_bps / 1e4
        rows.append(dict(date=date, gross_ret=gross, turnover=turnover, cost=cost,
                         net_ret=gross - cost, n_holdings=len(holdings),
                         holdings=",".join(sorted(holdings))))
        previous_weights = weights

    return pd.DataFrame(rows).set_index("date")


def gate_sweep(preds: pd.DataFrame,
               gates: tuple[float, ...] = (0.0, 0.0005, 0.001, 0.002, 0.005),
               top_k: int = 5,
               cost_bps: float = 5.0) -> dict[float, pd.DataFrame]:
    """Rebuild the portfolio for several gate values to expose the turnover trade-off.

    Args:
        preds: Panel with ``pred`` and ``fwd_ret``; use validation predictions to choose a gate.
        gates: Predicted-return gaps to try, in return units.
        top_k: Number of assets held.
        cost_bps: One-way transaction cost in basis points.

    Returns:
        Gate value -> :func:`build_portfolio` output.
    """
    return {gate: build_portfolio(preds, top_k, gate, cost_bps) for gate in gates}


@dataclass(frozen=True)
class ForecastVolParams:
    """Settings of the ``forecast_vol_scaled`` rule, read from its config block.

    Attributes:
        vol_window: Bars of past predictions in the forecast-volatility window.
        vol_min_periods: Past predictions needed before any volatility is usable.
        rebalance: How often the holdings are traded back to the target weights, which also
            corrects the weight drift caused by the positions moving with the market:
            ``"daily"`` (every bar), ``"weekly"`` (first bar of each week) or ``"monthly"``
            (first bar of each month).
        broker: ``"moomoo"`` or ``"ibkr"``; selects the commission model.
        cost_bps: Extra proportional cost charged on every executed order, in basis points.
    """

    vol_window: int = 60
    vol_min_periods: int = 20
    rebalance: str = "monthly"
    broker: str = "moomoo"
    cost_bps: float = 5.0

    def __post_init__(self) -> None:
        if self.rebalance not in ("daily", "weekly", "monthly"):
            raise ValueError(
                f"rebalance must be 'daily', 'weekly' or 'monthly', got {self.rebalance!r}"
            )
        if self.broker not in ("moomoo", "ibkr"):
            raise ValueError(f"broker must be 'moomoo' or 'ibkr', got {self.broker!r}")
        if self.vol_window < 2 or not 2 <= self.vol_min_periods <= self.vol_window:
            raise ValueError(
                "volatility needs 2 <= vol_min_periods <= vol_window, got "
                f"vol_min_periods={self.vol_min_periods}, vol_window={self.vol_window}"
            )
        if self.cost_bps < 0:
            raise ValueError("cost_bps cannot be negative")

    @classmethod
    def from_mapping(cls, block: Mapping[str, Any] | None) -> ForecastVolParams:
        """Build the settings from the YAML block, ignoring keys set to ``None``.

        Raises:
            ValueError: if the block contains a key this dataclass does not define, which
                turns a typo in ``config.yaml`` into an error instead of a silent default.
        """
        block = dict(block or {})
        unknown = set(block) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"unknown forecast_vol_scaled setting(s): {sorted(unknown)}")
        return cls(**{key: value for key, value in block.items() if value is not None})


def moomoo_commission(trade_value_usd: float) -> float:
    """Commission for one Moomoo order: 0.03% of the trade with a USD 0.77 floor, plus 9% GST.

    These are the fee assumptions of this project, not Moomoo's complete pricing schedule.
    """
    value = abs(float(trade_value_usd))
    if value <= 0:
        return 0.0
    return max(0.0003 * value, 0.77) * 1.09


def ibkr_commission(trade_value_usd: float) -> float:
    """Commission for one IBKR order: the smaller of a flat USD 1 or 1% of the trade.

    These are the fee assumptions of this project, not IBKR's complete pricing schedule.
    """
    value = abs(float(trade_value_usd))
    if value <= 0:
        return 0.0
    return min(1.0, 0.01 * value)


BROKER_COMMISSIONS = {"moomoo": moomoo_commission, "ibkr": ibkr_commission}


def order_cost_usd(trade_value_usd: float, params: ForecastVolParams) -> float:
    """Cost of one executed order: broker commission plus ``cost_bps`` on the trade value.

    The proportional cost is added here and nowhere else, so a 5 bps setting is charged once.
    An order of zero is not executed and costs nothing, so no commission floor applies.
    """
    value = abs(float(trade_value_usd))
    if value <= 0:
        return 0.0
    return BROKER_COMMISSIONS[params.broker](value) + params.cost_bps / 1e4 * value


def historical_forecast_vol(predictions: pd.DataFrame,
                            params: ForecastVolParams) -> pd.DataFrame:
    """Rolling standard deviation of each asset's past predictions.

    Args:
        predictions: Wide table, index = decision timestamps, columns = tickers, values =
            predicted simple returns, each generated strictly before its own timestamp.
        params: Volatility window and minimum observations.

    Returns:
        Wide table aligned with ``predictions``. The value dated ``t`` is computed from
        predictions dated strictly before ``t`` (a one-bar shift), so the forecast used to
        trade on ``t`` never enters its own volatility estimate, and it stays NaN until
        ``vol_min_periods`` past predictions exist for that asset.

    Raises:
        ValueError: if the timestamps are not unique.
    """
    wide = predictions.sort_index().astype(float)
    if not wide.index.is_unique:
        raise ValueError("prediction timestamps must be unique")
    return wide.shift(1).rolling(params.vol_window, min_periods=params.vol_min_periods).std(ddof=1)


def target_weights(predictions: pd.Series, forecast_vol: pd.Series) -> pd.Series:
    """Forecast-scaled target weights for a single decision date.

    Each eligible prediction is divided by its forecast volatility, then the ratios are
    divided by their absolute sum so the absolute weights add up to 1.0 (100% gross, no
    leverage). Weights are signed: positive is long, negative is short, and the two legs are
    not forced to match. Assets without a finite prediction or without a positive forecast
    volatility get weight zero.

    Args:
        predictions: Predicted returns for the date, one value per asset.
        forecast_vol: Forecast standard deviations for the same date, same index.

    Returns:
        Weights indexed like ``predictions``.
    """
    vol_floor, min_signal = 0.00001, 0.0  # a zero volatility cannot be divided by
    preds = pd.to_numeric(predictions, errors="coerce").astype(float)
    vols = pd.to_numeric(forecast_vol.reindex(preds.index), errors="coerce").astype(float)

    eligible = preds.notna() & np.isfinite(preds) & vols.notna() & np.isfinite(vols) & (vols > 0)
    eligible &= preds.abs() > min_signal

    weights = pd.Series(0.0, index=preds.index)
    weights[eligible] = preds[eligible] / vols[eligible].clip(lower=vol_floor)
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
    opening leg. Orders below ``min_trade_usd`` are skipped and therefore never charged.
    """
    min_trade_usd = 0.0
    tickers = current_weights.index.union(desired_weights.index)
    orders = (desired_weights.reindex(tickers, fill_value=0.0)
              - current_weights.reindex(tickers, fill_value=0.0)) * equity_usd
    if min_trade_usd > 0:
        orders = orders.where(orders.abs() >= min_trade_usd, 0.0)
    return orders


def rebalance_dates(dates: pd.DatetimeIndex, every: str = "monthly") -> pd.DatetimeIndex:
    """Dates on which the targets are refreshed and the weight drift is traded away.

    Args:
        dates: Bars of the prediction panel.
        every: ``"daily"`` (every bar), ``"weekly"`` (first bar of each week) or ``"monthly"``
            (first bar of each calendar month).

    Returns:
        The subset of ``dates`` on which the book is traded back to its target weights.
    """
    dates = pd.DatetimeIndex(dates)
    if every == "daily":
        return dates
    period = {"weekly": "W", "monthly": "M"}[every]
    first_of_period = pd.Series(dates, index=dates).groupby(dates.to_period(period)).first()
    return pd.DatetimeIndex(first_of_period.to_numpy())


def build_forecast_vol_portfolio(preds: pd.DataFrame, params: ForecastVolParams,
                                 initial_capital_usd: float = 100_000.0) -> pd.DataFrame:
    """Trade forecast-volatility-scaled weights, charging broker costs per order.

    The weights are built from the prediction dated each bar (available at its close) and earn
    that bar's ``fwd_ret``, i.e. the next bar's close-to-close return, so nothing from the
    future is used. Between rebalances every position is left to move with its own return, so
    the weights drift; on each rebalance date the book is marked to market and traded back to
    the target, which is what keeps gross exposure at 100%.

    Args:
        preds: Panel with ``pred`` and ``fwd_ret``, as returned by
            :func:`models.walk_forward_predict`, indexed by ``(date, ticker)``.
        params: Volatility, rebalance cadence, broker and cost settings.
        initial_capital_usd: Starting equity; the broker minimums are absolute dollars, so
            the simulated account size decides how much they bite.

    Returns:
        Frame indexed by date with ``gross_ret``, ``turnover``, ``cost``, ``net_ret``,
        ``n_holdings`` and ``holdings`` (the columns the pipeline consumes) plus
        ``n_long``, ``n_short``, ``gross_exposure``, ``net_exposure``, ``equity_usd`` and
        ``rebalanced``.

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
    forecast_vol = historical_forecast_vol(predictions, params)
    rebalances = set(rebalance_dates(predictions.index, params.rebalance))

    equity = initial_capital_usd
    weights = pd.Series(0.0, index=predictions.columns)
    previous_date = None
    rows = []
    for date in predictions.index:
        if previous_date is not None:
            weights = mark_to_market_weights(weights, returns.loc[previous_date])
        equity_before = equity

        if date in rebalances:
            target = target_weights(predictions.loc[date], forecast_vol.loc[date])
            orders = orders_for_target(weights, target, equity_before).fillna(0.0)
            costs = float(sum(order_cost_usd(value, params) for value in orders if value != 0))
            weights = (weights.add(orders / equity_before, fill_value=0.0)
                       .reindex(predictions.columns, fill_value=0.0))
        else:
            orders = pd.Series(0.0, index=predictions.columns)
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

    return pd.DataFrame(rows).set_index("date")


def _top_k(preds: pd.DataFrame, config) -> pd.DataFrame:
    """Equal-weight the ``top_k`` best predicted returns, charging turnover costs."""
    return build_portfolio(preds, config.top_k, config.gate, config.cost_bps)


def _forecast_vol_scaled(preds: pd.DataFrame, config) -> pd.DataFrame:
    """Scale predictions by their own forecast volatility and trade them long and short."""
    return build_forecast_vol_portfolio(preds, config.forecast_vol, config.initial_capital_usd)


STRATEGIES: dict[str, Callable[[pd.DataFrame, object], pd.DataFrame]] = {
    "top_k": _top_k,
    "forecast_vol_scaled": _forecast_vol_scaled,
}

BOTH = "both"
STRATEGY_CHOICES = (*STRATEGIES, BOTH)


def configured_strategies(strategy: str) -> tuple[str, ...]:
    """Rules to trade for a configured ``portfolio.strategy`` value.

    Args:
        strategy: ``"top_k"``, ``"forecast_vol_scaled"`` or ``"both"``.

    Returns:
        ``"both"`` expands to every registered rule, otherwise the single rule.

    Raises:
        ValueError: if ``strategy`` is not one of :data:`STRATEGY_CHOICES`.
    """
    if strategy == BOTH:
        return tuple(STRATEGIES)
    if strategy not in STRATEGIES:
        raise ValueError(f"unknown portfolio strategy {strategy!r}; use one of {STRATEGY_CHOICES}")
    return (strategy,)


def build_positions(preds: pd.DataFrame, config, strategy: str | None = None) -> pd.DataFrame:
    """Turn predictions into portfolio weights using one of the configured strategies.

    New strategies only need a function ``f(preds, config) -> portfolio frame`` registered
    in :data:`STRATEGIES` and a matching block in ``config.yaml``.

    Args:
        preds: Panel with ``pred`` and ``fwd_ret``.
        config: Any object exposing ``strategy`` and that strategy's parameters.
        strategy: Rule to run; defaults to the first configured one, so ``"both"`` resolves to
            ``"top_k"`` and model selection has a single objective to rank on.

    Returns:
        Portfolio frame from the selected strategy.

    Raises:
        ValueError: if ``config.strategy`` is not registered.
    """
    rule = strategy or configured_strategies(config.strategy)[0]
    return STRATEGIES[rule](preds, config)


def build_all_positions(preds: pd.DataFrame, config) -> dict[str, pd.DataFrame]:
    """Run every configured strategy: rule name -> portfolio frame."""
    return {name: STRATEGIES[name](preds, config)
            for name in configured_strategies(config.strategy)}
