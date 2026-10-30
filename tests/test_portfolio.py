"""Checks for portfolio construction from predicted returns."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from portfolio import build_portfolio, build_positions, gate_sweep


def make_predictions(dates: int = 5, tickers: int = 6) -> pd.DataFrame:
    """Build a deterministic prediction panel with a stable cross-sectional ranking."""
    index = pd.MultiIndex.from_product(
        [pd.bdate_range("2023-01-02", periods=dates), [f"T{i}" for i in range(tickers)]],
        names=["date", "ticker"],
    )
    scores = np.tile(np.linspace(0.02, -0.02, tickers), dates)
    return pd.DataFrame({"pred": scores, "fwd_ret": scores / 2}, index=index)


def test_build_portfolio_holds_top_k_equal_weight_and_nets_out_costs():
    port = build_portfolio(make_predictions(), top_k=3, gate=0.0, cost_bps=5.0)

    assert (port["n_holdings"] == 3).all()
    assert np.allclose(port["net_ret"], port["gross_ret"] - port["cost"])
    assert port["turnover"].iloc[0] == 1.0          # first bar is a full entry
    assert np.allclose(port["turnover"].iloc[1:], 0.0)  # ranking is stable, so no swaps


def test_gate_blocks_a_swap_the_ranking_would_otherwise_make():
    preds = make_predictions()
    dates = preds.index.get_level_values("date").unique()
    swapped = preds.copy()
    later = swapped.index.get_level_values("date") == dates[1]
    swapped.loc[later, "pred"] = swapped.loc[later, "pred"].to_numpy()[::-1]

    no_gate = build_portfolio(swapped, top_k=3, gate=0.0, cost_bps=0.0)
    high_gate = build_portfolio(swapped, top_k=3, gate=1.0, cost_bps=0.0)

    assert no_gate["turnover"].iloc[1] > 0
    assert high_gate["turnover"].iloc[1] == 0


def test_gate_sweep_returns_one_portfolio_per_gate():
    sweep = gate_sweep(make_predictions(), gates=(0.0, 0.001), top_k=3)

    assert list(sweep) == [0.0, 0.001]
    assert all({"gross_ret", "net_ret"} <= set(frame.columns) for frame in sweep.values())


def test_build_positions_dispatches_to_the_configured_strategy():
    config = SimpleNamespace(strategy="top_k", top_k=3, gate=0.0, cost_bps=5.0)

    port = build_positions(make_predictions(), config)

    assert (port["n_holdings"] == 3).all()


def test_build_positions_rejects_an_unknown_strategy():
    config = SimpleNamespace(strategy="momentum", top_k=3, gate=0.0, cost_bps=5.0)

    with pytest.raises(ValueError):
        build_positions(make_predictions(), config)
