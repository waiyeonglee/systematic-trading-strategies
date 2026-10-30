"""Turn predicted returns into an equal-weight top-k portfolio."""

from __future__ import annotations

from collections.abc import Callable

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


def _top_k(preds: pd.DataFrame, config) -> pd.DataFrame:
    """Equal-weight the ``top_k`` best predicted returns, charging turnover costs."""
    return build_portfolio(preds, config.top_k, config.gate, config.cost_bps)


STRATEGIES: dict[str, Callable[[pd.DataFrame, object], pd.DataFrame]] = {"top_k": _top_k}


def build_positions(preds: pd.DataFrame, config) -> pd.DataFrame:
    """Turn predictions into portfolio weights using the configured strategy.

    New strategies only need a function ``f(preds, config) -> portfolio frame`` registered
    in :data:`STRATEGIES` and a matching block in ``config.yaml``.

    Args:
        preds: Panel with ``pred`` and ``fwd_ret``.
        config: Any object exposing ``strategy`` and that strategy's parameters.

    Returns:
        Portfolio frame from the selected strategy.

    Raises:
        ValueError: if ``config.strategy`` is not registered.
    """
    strategy = STRATEGIES.get(config.strategy)
    if strategy is None:
        raise ValueError(
            f"unknown portfolio strategy {config.strategy!r}; available: {sorted(STRATEGIES)}"
        )
    return strategy(preds, config)
