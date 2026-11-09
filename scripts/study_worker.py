"""Worker for the joint ranking (importable so joblib can pickle by reference).

The panel, feature columns, benchmark and cache root are sent **once per worker**
through :func:`init`; tasks then only carry a spec and its combo. That removes the
per-task panel pickle that dominated dispatch time in the first implementation.

Each task scores a small batch of candidates through every book cell, so the fixed
cost of a round trip is amortised over ~5 candidates (225 book cells). Cells that
differ only in an inert knob are simulated once — see :func:`effective_key`.

:func:`cache_candidate` is the same task shape without the book stage: it only fits
the walk-forward prediction sets, which is what the studies' ``cache`` command runs
before a ranking over a space whose cache is incomplete.
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for _path in (_ROOT / "src", _ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pandas as pd

from metrics import perf_stats, sharpe
from models import ModelSpec
from portfolio import build_positions, forecast_vol_frame
from prediction_cache import (DegenerateCandidate, cache_path, cached_predictions,
                              cached_warmup)
from run_backtest import default_config

VALIDATION = ("2019-01-01", "2022-12-31")
VOL_WARMUP_BARS = 48            # covers the longest vol_window (48w)

# Book space: the full product of the three strategy knobs (no shared base row).
VOL_WINDOWS = (4, 8, 12, 24, 48)                              # forecast-vol window, w
VOL_TARGETS = ((0.20, "vt20"), (0.10, "vt10"), (None, "vtoff"))   # annualized target
VOL_RF_RATES = (0.03,)                                       # cash-sleeve rate, ann.
LEV_CAPS = ((1.0, 0.0), (2.0, 0.068), (3.0, 0.068))           # cap, borrow fee, ann.


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
                book: dict, window_train: str | None = "8y",
                grid_entries: list[dict] | None = None):
    """Config for one book cell: the candidate plus that cell's book knobs."""
    book = dict(book)
    vol_target = book.pop("vol_target", True)
    return default_config(horizon, vol_target=vol_target, model=model,
                          rebalance=rebalance, refit_every=refit,
                          window_train=window_train, grid_entries=grid_entries,
                          **book)


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
         cache_root: Path, validation: tuple[str, str] = VALIDATION,
         window_train: str | None = "8y",
         grid_entries: list[dict] | None = None) -> None:
    """Called once per worker process."""
    _STATE.update(panel=panel, feature_cols=feature_cols,
                  benchmark_close=benchmark_close, cache_root=Path(cache_root),
                  validation=tuple(validation), window_train=window_train,
                  grid_entries=grid_entries)


def combo_slug(rebalance: str, horizon: int, refit: str) -> str:
    return f"la{horizon}_{rebalance}_refit{refit}"


def forecast_groups(pairs: list[tuple[str, int, str]]) -> list[tuple[str, int, str]]:
    """One representative combo per ``(horizon, refit)`` group.

    The walk-forward forecasts depend on the model spec, the target and the refit
    cadence only, so combos that share those (e.g. ``weekly`` and ``4w`` at the same
    look-ahead and refit) share one cached prediction set.
    """
    seen: set[tuple[int, str]] = set()
    groups: list[tuple[str, int, str]] = []
    for rebalance, horizon, refit in pairs:
        if (horizon, refit) in seen:
            continue
        seen.add((horizon, refit))
        groups.append((rebalance, horizon, refit))
    return groups


def score_candidate(spec: ModelSpec, rebalance: str, horizon: int, refit: str,
                    refresh: bool = False) -> list[dict]:
    """Score one (spec, combo) through every book cell."""
    panel = _STATE["panel"]
    feature_cols = _STATE["feature_cols"]
    cache_root = _STATE["cache_root"]
    validation = _STATE["validation"]
    window_train = _STATE["window_train"]
    grid_entries = _STATE["grid_entries"]

    pred_config = book_config(horizon, rebalance=rebalance, refit=refit,
                              model=spec.name, book=dict(vol_target=False, vol_rf=0.0),
                              window_train=window_train, grid_entries=grid_entries)
    try:
        preds = cached_predictions(panel, feature_cols, pred_config, spec, *validation,
                                   cache_root=cache_root, refresh=refresh, verbose=False)
    except DegenerateCandidate as exc:
        # constant predictions: no cross-sectional information, so every book cell of
        # this candidate is degenerate and no simulation is worth running
        return [_degenerate_row(spec, rebalance, horizon, refit, name, knobs, exc.record)
                for name, knobs, note in BOOKS]

    warm = cached_warmup(panel, feature_cols, pred_config, spec, validation[0],
                         bars=VOL_WARMUP_BARS, cache_root=cache_root,
                         refresh=refresh, verbose=False)

    base = _candidate_row(spec, rebalance, horizon, refit)
    rows = []
    vol_cache: dict[tuple[int, int], pd.DataFrame] = {}
    simulated: dict[tuple, dict] = {}
    for name, knobs, note in BOOKS:
        metrics = simulated.get(effective_key(knobs))
        if metrics is None:
            config = book_config(horizon, rebalance=rebalance, refit=refit,
                                 model=spec.name, book=knobs,
                                 window_train=window_train, grid_entries=grid_entries)
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
            simulated[effective_key(knobs)] = metrics
        row = dict(base, book=name, degenerate=False, **knobs)
        row.update(metrics)
        rows.append(row)
    return rows


def _candidate_row(spec: ModelSpec, rebalance: str, horizon: int, refit: str) -> dict:
    """Model + combo columns shared by every book cell of a candidate."""
    return {"model": spec.name, "family": spec.name.split("_")[0],
            "alpha": spec.alpha, "l1_ratio": spec.l1_ratio,
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
               hit_rate=math.nan, avg_turnover=math.nan)
    return row


def cache_candidate(spec: ModelSpec, rebalance: str, horizon: int, refit: str,
                    refresh: bool = False) -> dict:
    """Fit the prediction sets of one candidate, without simulating any book cell.

    The forecasts depend on the model spec, the target and the refit cadence only — not
    on the rebalance cadence — so ``rebalance`` is carried for labelling and provenance
    completeness while the caller should dedupe on ``(horizon, refit)``.

    Returns:
        ``{"candidate", "spec", "status", "seconds"}`` where status is ``"cached"``
        (payload already valid), ``"stored"`` (written now) or ``"degenerate"``
        (constant predictions; only a marker is written).
    """
    panel = _STATE["panel"]
    feature_cols = _STATE["feature_cols"]
    cache_root = _STATE["cache_root"]
    validation = _STATE["validation"]
    window_train = _STATE["window_train"]
    grid_entries = _STATE["grid_entries"]

    pred_config = book_config(horizon, rebalance=rebalance, refit=refit,
                              model=spec.name, book=dict(vol_target=False, vol_rf=0.0),
                              window_train=window_train, grid_entries=grid_entries)
    path = cache_path(pred_config, spec.name, *validation, cache_root=cache_root)
    before = path.stat().st_mtime if path.exists() else None
    started = time.perf_counter()

    try:
        cached_predictions(panel, feature_cols, pred_config, spec, *validation,
                           cache_root=cache_root, refresh=refresh, verbose=False)
    except DegenerateCandidate:
        status = "degenerate"
    else:
        cached_warmup(panel, feature_cols, pred_config, spec, validation[0],
                      bars=VOL_WARMUP_BARS, cache_root=cache_root, refresh=refresh,
                      verbose=False)
        # the write happens after the fit, so an unchanged mtime means a cache hit
        after = path.stat().st_mtime
        status = "cached" if before == after and not refresh else "stored"

    return {"candidate": combo_slug(rebalance, horizon, refit), "spec": spec.name,
            "status": status, "seconds": round(time.perf_counter() - started, 2)}


def cache_batch(tasks: list[tuple[ModelSpec, str, int, str]],
                refresh: bool = False) -> list[dict]:
    """Fill the prediction cache for a batch of (spec, rebalance, horizon, refit)."""
    return [cache_candidate(spec, rebalance, horizon, refit, refresh)
            for spec, rebalance, horizon, refit in tasks]


def score_batch(tasks: list[tuple[ModelSpec, str, int, str]],
                refresh: bool = False) -> list[dict]:
    """Score a batch of (spec, rebalance, horizon, refit) candidates."""
    out: list[dict] = []
    for spec, rebalance, horizon, refit in tasks:
        out.extend(score_candidate(spec, rebalance, horizon, refit, refresh))
    return out
