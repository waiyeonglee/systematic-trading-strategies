"""Worker for the joint ranking (importable so joblib can pickle by reference).

The panel, feature columns, benchmark and cache root are sent **once per worker**
through :func:`init`; tasks then only carry a spec and its combo. That removes the
per-task panel pickle that dominated dispatch time in the first implementation.

Each task scores a small batch of candidates through every book cell, so the fixed
cost of a round trip is amortised over ~5 candidates (225 book cells). Cells that
differ only in an inert knob are simulated once — see :func:`effective_key`.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT / "src", _ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pandas as pd

from metrics import perf_stats, sharpe
from models import ModelSpec
from portfolio import build_positions, forecast_vol_frame
from prediction_cache import (DegenerateCandidate, cached_predictions,
                              cached_warmup)
from run_backtest import default_config

VALIDATION = ("2017-01-01", "2022-12-31")
VOL_WARMUP_BARS = 48            # covers the longest vol_window (48w)

# Book space: the full product of the three strategy knobs (no shared base row).
VOL_WINDOWS = (4, 8, 12, 24, 48)                              # forecast-vol window, w
VOL_TARGETS = ((0.20, "vt20"), (0.10, "vt10"), (None, "vtoff"))   # annualized target
VOL_RF_RATES = (0.03,)                                       # cash-sleeve rate, ann.
LEV_CAPS = ((1.0, 0.0), (2.0, 0.068), (3.0, 0.068))           # cap, borrow fee, ann.

# Regime-stratified selection rule (pre-registered before the run, see
# agents/validation_and_ablation.md §3). A candidate must earn its Sharpe in every
# kind of year rather than in the drawdown-rich ones only: profitable in at least
# REGIME_MIN_POS_YEARS of the validation calendar years AND keeping at least
# REGIME_MIN_UP_CAPTURE of the benchmark's upside in its up weeks. Down-capture is
# reported but not gated — requiring extra protection is exactly the bias that
# produced the failed selection (it favours defensive cells that only pay off in
# bear years).
REGIME_MIN_POS_YEARS = 5          # of REGIME_WINDOW_YEARS validation years
REGIME_WINDOW_YEARS = 6           # 2017..2022 — the rule is the 5/6 rate, so it
                                  # generalises to any window length
REGIME_MIN_UP_CAPTURE = 0.40


def regime_metrics(net_ret: pd.Series, benchmark: pd.Series, ppy: int) -> dict:
    """Per-year consistency and up/down capture of a book versus the benchmark.

    Args:
        net_ret: The book's per-bar net returns (the validation window).
        benchmark: The equal-weight benchmark's per-bar returns, same index.
        ppy: Periods per year (52 for weekly bars).

    Returns:
        ``pos_years`` / ``n_years`` / ``yr_sharpe_min`` / ``up_capture`` /
        ``down_capture``. Capture ratios are means in benchmark-up (down) bars, so
        ``up_capture`` near 1 means the book rode the upside and ``down_capture``
        near 0 means it side-stepped the downside.
    """
    yearly = [sharpe(group, ppy) for _, group in net_ret.groupby(net_ret.index.year)]
    finite = [value for value in yearly if value == value]
    up = benchmark > 0.0
    up_mean = float(benchmark[up].mean()) if up.any() else math.nan
    down_mean = float(benchmark[~up].mean()) if (~up).any() else math.nan
    return {
        "pos_years": int(sum(1 for value in finite if value > 0)),
        "n_years": len(finite),
        "yr_sharpe_min": float(min(finite)) if finite else math.nan,
        "up_capture": (float(net_ret[up].mean()) / up_mean
                       if up_mean and up_mean == up_mean else math.nan),
        "down_capture": (float(net_ret[~up].mean()) / down_mean
                         if down_mean else math.nan),
    }


def regime_pass(metrics: dict) -> bool:
    """Whether a scored row clears the regime-stratified rule in §3.

    The consistency test is the pre-registered **5-of-6 rate**, not a raw count, so
    the same rule applies to any window length.
    """
    years = metrics["n_years"]
    if not years or years != years:            # empty window / degenerate row
        return False
    return bool(metrics["pos_years"] * REGIME_WINDOW_YEARS
                >= REGIME_MIN_POS_YEARS * years
                and metrics["up_capture"] >= REGIME_MIN_UP_CAPTURE)


def _book_note(window: int, target: float | None, rf: float, cap: float,
               fee: float) -> str:
    target_txt = "off" if target is None else f"{target:.0%}"
    fee_txt = f" + {fee:.1%} borrow fee" if fee else ""
    return (f"vol window {window}w, vol target {target_txt}, rf {rf:.0%}, "
            f"cap {cap:.1f}{fee_txt}")


BOOKS = [
    (f"w{window:02d}_{tag}_rf{rf * 100:.0f}_cap{cap:.1f}",
     dict(vol_window=window, vol_target=target, vol_rf=rf, vol_cap=cap, lev_fee=fee),
     _book_note(window, target, rf, cap, fee))
    for target, tag in VOL_TARGETS
    for window in VOL_WINDOWS
    for rf in VOL_RF_RATES
    for cap, fee in LEV_CAPS
]

_STATE: dict = {}


def book_config(horizon: int, *, rebalance: str, refit: str, model: str | None,
                book: dict):
    """Config for one book cell: the candidate plus that cell's book knobs."""
    book = dict(book)
    vol_target = book.pop("vol_target", True)
    return default_config(horizon, vol_target=vol_target, model=model,
                          rebalance=rebalance, refit_every=refit, **book)


def effective_key(knobs: dict) -> tuple:
    """Signature of the book a cell actually produces.

    Without a volatility target :func:`portfolio._vol_target_scale` returns 1.0 on
    every rebalance, so the leverage cap can never bind and those cells differ only
    in the (unused) cap. They are simulated once and share their metrics.
    """
    target = knobs["vol_target"]
    return (knobs["vol_window"], target, knobs["vol_rf"],
            knobs["vol_cap"] if target else 1.0, knobs["lev_fee"])


def init(panel: pd.DataFrame, feature_cols: list[str], benchmark_close: pd.Series,
         cache_root: Path) -> None:
    """Called once per worker process."""
    _STATE.update(panel=panel, feature_cols=feature_cols,
                  benchmark_close=benchmark_close, cache_root=Path(cache_root))


def combo_slug(rebalance: str, horizon: int, refit: str) -> str:
    return f"la{horizon}_{rebalance}_refit{refit}"


def score_candidate(spec: ModelSpec, rebalance: str, horizon: int, refit: str,
                    refresh: bool = False) -> list[dict]:
    """Score one (spec, combo) through every book cell."""
    panel = _STATE["panel"]
    feature_cols = _STATE["feature_cols"]
    cache_root = _STATE["cache_root"]

    pred_config = book_config(horizon, rebalance=rebalance, refit=refit,
                              model=spec.name, book=dict(vol_target=False, vol_rf=0.0))
    try:
        preds = cached_predictions(panel, feature_cols, pred_config, spec, *VALIDATION,
                                   cache_root=cache_root, refresh=refresh, verbose=False)
    except DegenerateCandidate as exc:
        # constant predictions: no cross-sectional information, so every book cell of
        # this candidate is degenerate and no simulation is worth running
        return [_degenerate_row(spec, rebalance, horizon, refit, name, knobs, exc.record)
                for name, knobs, note in BOOKS]

    warm = cached_warmup(panel, feature_cols, pred_config, spec, VALIDATION[0],
                         bars=VOL_WARMUP_BARS, cache_root=cache_root,
                         refresh=refresh, verbose=False)

    base = _candidate_row(spec, rebalance, horizon, refit)
    benchmark = preds["fwd_ret"].groupby(level="date").mean()
    rows = []
    vol_cache: dict[tuple[int, int], pd.DataFrame] = {}
    simulated: dict[tuple, dict] = {}
    for name, knobs, note in BOOKS:
        metrics = simulated.get(effective_key(knobs))
        if metrics is None:
            config = book_config(horizon, rebalance=rebalance, refit=refit,
                                 model=spec.name, book=knobs)
            params = config.forecast_vol
            # the rolling forecast volatility depends only on the predictions and the
            # vol window, so five distinct windows cover all book cells
            vol_key = (params.vol_window, params.vol_min_periods)
            if vol_key not in vol_cache:
                vol_cache[vol_key] = forecast_vol_frame(preds, params, warm)
            book = build_positions(preds, config, apply_vol_target=True,
                                   vol_history=warm,
                                   forecast_vol=vol_cache[vol_key])
            stats = perf_stats(book["net_ret"], config.ppy)
            metrics = {"net_sharpe": stats["sharpe"],
                       "gross_sharpe": sharpe(book["gross_ret"], config.ppy),
                       "max_drawdown": stats["max_drawdown"],
                       "total_return": stats["total_return"],
                       "ann_return": stats["ann_return"],
                       "ann_vol": stats["ann_vol"],
                       "hit_rate": stats["hit_rate"],
                       "avg_turnover": float(book["turnover"].mean())}
            metrics.update(regime_metrics(book["net_ret"],
                                          benchmark.reindex(book.index), config.ppy))
            metrics["regime_pass"] = regime_pass(metrics)
            simulated[effective_key(knobs)] = metrics
        row = dict(base, book=name, degenerate=False, **knobs)
        row.update(metrics)
        rows.append(row)
    return rows


def _candidate_row(spec: ModelSpec, rebalance: str, horizon: int, refit: str) -> dict:
    """Model + combo columns shared by every book cell of a candidate."""
    return {"model": spec.name, "family": spec.name.split("_")[0],
            "alpha": spec.alpha, "l1_ratio": spec.l1_ratio,
            "rff_dim": spec.rff_dim, "gamma": spec.gamma,
            "target": f"absolute{horizon}", "look_ahead": horizon,
            "rebalance": rebalance, "refit": refit,
            "combo": combo_slug(rebalance, horizon, refit)}


def _degenerate_row(spec: ModelSpec, rebalance: str, horizon: int, refit: str,
                    name: str, knobs: dict, record: dict) -> dict:
    """Ranking row for a degenerate candidate (no books are simulated)."""
    row = _candidate_row(spec, rebalance, horizon, refit)
    row.update(book=name, degenerate=True, degenerate_reason=record.get("reason"),
               **knobs)
    row.update(net_sharpe=math.nan, gross_sharpe=math.nan, max_drawdown=math.nan,
               total_return=math.nan, ann_return=math.nan, ann_vol=math.nan,
               hit_rate=math.nan, avg_turnover=math.nan, pos_years=math.nan,
               n_years=math.nan, yr_sharpe_min=math.nan, up_capture=math.nan,
               down_capture=math.nan, regime_pass=False)
    return row


def score_batch(tasks: list[tuple[ModelSpec, str, int, str]],
                refresh: bool = False) -> list[dict]:
    """Score a batch of (spec, rebalance, horizon, refit) candidates."""
    out: list[dict] = []
    for spec, rebalance, horizon, refit in tasks:
        out.extend(score_candidate(spec, rebalance, horizon, refit, refresh))
    return out
