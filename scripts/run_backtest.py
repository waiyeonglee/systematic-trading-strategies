#!/usr/bin/env python3
"""Run the walk-forward model-selection backtest for trend-following-strategy-v2.

Reads the raw weekly CSVs (16 tradables + SPY) and the Fear & Greed index, builds the
feature panel with the 1/4/12-bar forward-return labels, selects the grid point
with the best validation net Sharpe and measures it once on the test window against
SPY buy-and-hold and an equal-weight-16 benchmark (design §3 gate).

2026-11-09 grid revision (user request): Elastic Net only, with endpoints included
so ``l1_ratio=0`` covers the Ridge-like case and ``l1_ratio=1`` covers the
Lasso-like case. The grid is {1e-4 .. 0.1} x l1_ratio {0, .25, .5, .75, 1}
-> 20 specs;
look-ahead {1, 4, 12} weeks x rebalance cadence {weekly=1w, 4w, 12w} under the rule **look-ahead >=
rebalance frequency**, x model refit cadence {ME, QE} under the rule
**refit_interval >= prediction_interval** (both enforced in ``default_config``).
V1 no longer exists as a separate stage: the study ranks model x look-ahead x
frequency x refit **x book** jointly on the combined validation window
(20 specs x 11 valid combos x 45 book cells = 9,900 cells, see
``scripts/run_study.py``). This CLI measures **one pinned cell** (validation + the
single test draw); it does not re-rank the space.

Defaults encode the v2 fixed window (data ends 2026-01-01): weekly bars, 8y rolling
train / 4y combined validation (2019-2022) / quarterly refits, the 20-spec grid,
absolute12 target, embargo 12, forecast-vol-scaled book (24w vol, 4-week rebalance,
moomoo, 0 bps, cost gate 1.0) with the 10% vol target / cap 1.0 / 3% risk-free
sleeve. Selection uses the same combined validation window as the study
(``scripts/run_study.py``), seeded with ``utils.config.DEFAULT_SEED``.

Usage:
    python scripts/run_backtest.py
    python scripts/run_backtest.py --horizon 4 --no-vol-target
    python scripts/run_backtest.py --model enet_a0.01_r1 --horizon 1 --rebalance weekly \
        --test-start 2020-01-01 --test-end 2022-12-31   # single-cell smoke / ablation
    python scripts/run_backtest.py --test-start 2020-01-01 --test-end 2022-12-31
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))  # reuse the feature-build loaders

import pandas as pd

from build_features import load_fear_greed_index, load_ohlcv
from features import build_feature_frame, feature_columns
from labels import MODEL_EMBARGO, MODEL_TARGETS, make_labels
from models import specs_from_config
from pipeline import PipelineConfig, evaluate_all, run_pipeline, run_tag
from portfolio import ForecastVolParams
from utils.config import (BENCHMARKS, DEFAULT_SEED, START_DATE, TRADABLE_ETFS,
                          seed_everything)
from utils.paths import FEAR_GREED_PATH, RESULTS_DIR, raw_data_dir

# Design grid (agents/model_design.py §2), revised 2026-11-09 (user request) to an
# Elastic-Net-only grid with endpoints. The L1 side sits in the lower alpha range
# because this data's feature signal (~0.001-0.008 in alpha units) is below the
# entry threshold of a {1e-3..100} grid (intercept-only collapse, 2026-11-02 smoke
# finding).
ENET_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1)
ENET_L1_RATIOS = (0.0, 0.25, 0.5, 0.75, 1.0)
GRID_ENTRIES = [
    {"family": "enet", "alpha": list(ENET_ALPHAS), "l1_ratio": list(ENET_L1_RATIOS)},
]

# Rebalance cadence -> bars (design rule: look-ahead >= frequency).
FREQ_TO_BARS = {"daily": 1, "weekly": 1, "4w": 4, "12w": 12}
# Valid (look-ahead, frequency) pairs under that rule — the cadence x look-ahead
# dimension of the ranking grid.
FREQ_LA_PAIRS = [("weekly", 1), ("weekly", 4), ("weekly", 12),
                 ("4w", 4), ("4w", 12),
                 ("12w", 12)]

# Model refit cadence -> approximate bars, and a dimension of the ranking grid
# (2026-11-09 user request): refit monthly / quarterly only.
# Design rule: refit is **no more frequent than the acting cadence** —
# refit_interval >= prediction_interval (ME=4, QE=12 bars vs cadence
# 1/4/12), analogous in spirit (an economy rule, NOT statistical validity) to
# look-ahead >= frequency: refits finer than the cadence fit models whose
# intermediate predictions the book never uses; refit == cadence captures the
# freshness. Valid combos: ME x 1w/4w cadences (5), QE x all pairs (6) -> 11.
REFIT_TO_BARS = {"ME": 4, "QE": 12}
REFIT_OPTIONS = ("ME", "QE")      # monthly / quarterly


def build_panel() -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Build the runnable panel from the raw weekly CSVs (design conventions).

    Returns:
        ``(panel, benchmark_close, feature_cols)`` where the panel is indexed by
        ``(date, ticker)`` with the 21 features, ``fwd_ret`` (1-bar, earned by the
        book) and the 1/4/12-bar training labels; rows without a realised label
        are dropped, so the panel ends at 2025-09-26 (13-bar 12-week label tail, shared by
        all look-aheads).
    """
    fear_greed = load_fear_greed_index(FEAR_GREED_PATH)
    raw_dir = raw_data_dir("weekly")
    prices = load_ohlcv(raw_dir, TRADABLE_ETFS)
    benchmark = load_ohlcv(raw_dir, BENCHMARKS)[BENCHMARKS[0]]

    features = build_feature_frame(prices, frequency="weekly", sectors=TRADABLE_ETFS,
                                   fear_greed_index=fear_greed,
                                   benchmark_prices=benchmark)
    close_wide = pd.DataFrame({ticker: prices[ticker]["Close"] for ticker in TRADABLE_ETFS}
                              ).sort_index()
    labels = make_labels(close_wide)
    panel = (features[feature_columns(use_fear_greed=True)]
             .join(labels, how="inner")
             .dropna()          # drop the warmup rows with NaN features and the unlabelled tail
             .sort_index())

    benchmark_close = benchmark["Close"].astype(float).sort_index()
    benchmark_close.index = pd.to_datetime(benchmark_close.index)
    return panel, benchmark_close, feature_columns(use_fear_greed=True)


def default_config(horizon: int, vol_target: bool | float | None,
                   model: str | None = None,
                   vol_window: int = 24, rebalance: str = "4w",
                   refit_every: str = "QE",
                   vol_cap: float = 1.0, vol_rf: float = 0.03,
                   lev_fee: float = 0.0,
                   window_train: str | None = "8y",
                   grid_entries: list[dict] | None = None) -> PipelineConfig:
    """Pipeline settings with the fixed v2 window.

    ``model`` pins a single spec (``mode: single``, a single ranked cell); ``None``
    keeps the 20-spec grid. ``refit_every`` is a ranking-grid dimension (monthly /
    quarterly), default QE. ``vol_target``
    accepts a float (annualized target, e.g. ``0.20`` for 20%), ``True`` (default
    10%), or ``False`` / ``None`` (overlay off). ``vol_min_periods`` is clamped to
    the window so short vol windows (4w/8w/12w) stay valid.

    ``window_train`` is the rolling training window (``"8y"`` = the 416-bar v2
    default shared with the macro study). ``grid_entries`` replaces
    :data:`GRID_ENTRIES` when the pinned ``model`` comes from a different grid (e.g.
    the elastic-net-only macro grid, which adds ``l1_ratio`` 0 and 1).

    Raises:
        ValueError: if ``horizon`` is not a valid target, if the rebalance
            frequency exceeds the look-ahead (design rule: look-ahead >= frequency),
            or if the refit cadence is finer than the rebalance cadence (design
            rule: refit_interval >= prediction_interval).
    """
    if f"absolute{horizon}" not in MODEL_TARGETS:
        raise ValueError(f"invalid look-ahead horizon {horizon}; use one of "
                         f"{sorted(k[8:] for k in MODEL_TARGETS)}")
    if refit_every not in REFIT_TO_BARS:
        raise ValueError(f"invalid refit cadence {refit_every!r}; use one of {REFIT_OPTIONS}")
    if FREQ_TO_BARS[rebalance] > horizon:
        raise ValueError(
            f"look-ahead must be >= rebalance frequency: horizon {horizon} "
            f"< freq {FREQ_TO_BARS[rebalance]} ({rebalance!r})")
    if REFIT_TO_BARS[refit_every] < FREQ_TO_BARS[rebalance]:
        raise ValueError(
            f"refit must be no more frequent than the rebalance cadence: "
            f"refit {refit_every} ({REFIT_TO_BARS[refit_every]}w) < "
            f"rebalance {rebalance} ({FREQ_TO_BARS[rebalance]}w)")
    target_col = MODEL_TARGETS[f"absolute{horizon}"]
    specs = tuple(specs_from_config(GRID_ENTRIES if grid_entries is None else grid_entries))
    grid = (next(s for s in specs if s.name == model),) if model else specs
    target_vol = 0.10 if vol_target is True else (vol_target if vol_target else None)
    forecast_vol = ForecastVolParams(
        vol_window=vol_window, vol_min_periods=min(20, vol_window),
        rebalance=rebalance, broker="moomoo",
        cost_bps=0.0, cost_gate_multiple=1.0,
        vol_target=target_vol,
        vol_target_cap=vol_cap, vol_target_rf=vol_rf, vol_target_lev_fee=lev_fee,
    )
    return PipelineConfig(
        freq="weekly", strategy="forecast_vol_scaled",
        mode="single" if model else "selection", initial_capital_usd=100_000.0,
        refit_every=refit_every, window_train=window_train,
        embargo=MODEL_EMBARGO[f"absolute{horizon}"],
        grid=grid, seed=DEFAULT_SEED, target_col=target_col, forecast_vol=forecast_vol,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-start", default="2023-01-01",
                        help="first test date (default: 2023-01-01)")
    parser.add_argument("--test-end", default="2025-12-31",
                        help="last test date (default: 2025-12-31)")
    parser.add_argument("--window-val", default="4y",
                        help="validation span before test_start (default: 4y = the "
                             "combined 2019-2022 study window)")
    parser.add_argument("--window-train", default="8y",
                        help="rolling training window (default: 8y = 416 weekly bars)")
    parser.add_argument("--refit-every", default="QE", choices=("ME", "QE"),
                        help="refit cadence (default: QE; ranking dimension: "
                             "ME|QE, refit must be no more frequent than "
                             "--rebalance)")
    parser.add_argument("--horizon", type=int, default=12, choices=(1, 4, 12),
                        help="label look-ahead in bars, absolute1/4/12 (default: 12)")
    parser.add_argument("--no-vol-target", action="store_true",
                        help="disable the 10%% vol-target overlay (plain ADR-07 book)")
    parser.add_argument("--model", default=None,
                        choices=[s.name for s in specs_from_config(GRID_ENTRIES)],
                        help="pin one spec (mode: single, e.g. enet_a0.01_r1); "
                             "default: 20-spec selection grid")
    parser.add_argument("--vol-window", type=int, default=24,
                        help="realized-vol estimation window in bars (default: 24; "
                             "ranking winner: 12)")
    parser.add_argument("--rebalance", default="4w",
                        choices=("weekly", "4w", "12w"),
                        help="rebalance cadence (weekly=1w, 4w, 12w), "
                             "predictions locked to it; look-ahead must be >= it "
                             "(default: 4w)")
    parser.add_argument("--vol-cap", type=float, default=1.0,
                        help="vol-target leverage cap (default: 1.0)")
    parser.add_argument("--vol-rf", type=float, default=0.03,
                        help="risk-free sleeve rate for de-risked cash (default: 0.03)")
    parser.add_argument("--lev-fee", type=float, default=0.0,
                        help="moomoo holding fee on borrowed amount (default: 0.0)")
    parser.add_argument("--out-dir", default=None, help="results folder (default: results/)")
    args = parser.parse_args()

    seed_everything()
    panel, benchmark_close, feature_cols = build_panel()
    config = default_config(args.horizon, vol_target=not args.no_vol_target,
                            model=args.model, vol_window=args.vol_window,
                            rebalance=args.rebalance, refit_every=args.refit_every,
                            vol_cap=args.vol_cap, vol_rf=args.vol_rf,
                            lev_fee=args.lev_fee)

    dates = panel.index.get_level_values("date")
    print(f"panel: weekly bars, {panel.index.get_level_values('ticker').nunique()} tickers, "
          f"{len(feature_cols)} features, {dates.min().date()} -> {dates.max().date()}")
    print(f"  window_train {config.window_train} = {config.window_train_bars} bars, "
          f"window_val {args.window_val}, refit {config.refit_every}, "
          f"embargo {config.embargo} bars, target {config.target_col}, "
          f"mode {config.mode}")
    if config.forecast_vol.vol_target is not None:
        print(f"  vol-target overlay: target {config.forecast_vol.vol_target:g} ann, "
              f"cap {config.forecast_vol.vol_target_cap:g}, "
              f"risk-free sleeve {config.forecast_vol.vol_target_rf:g} ann")
    print(f"  cadence: {config.forecast_vol.rebalance} rebalance "
          f"({FREQ_TO_BARS[config.forecast_vol.rebalance]}w), "
          f"predictions on rebalance bars")

    tag = run_tag(config, args.test_start, args.test_end, args.window_val)
    out_dir = Path(args.out_dir) if args.out_dir else (RESULTS_DIR / tag)
    if config.mode == "single":
        print(f"  model [single]: {config.grid[0].name}")
    else:
        print(f"  model [selection]: {config.grid[0].name} + {len(config.grid) - 1} more")
    print(f"  results -> {out_dir}")

    results = run_pipeline(panel, feature_cols, benchmark_close, config,
                           test_start=args.test_start, test_end=args.test_end,
                           window_val=args.window_val, out_dir=out_dir)

    ranking = results["ranking"]
    pd.set_option("display.float_format", "{:.4f}".format)
    pd.set_option("display.width", 200)
    print(f"\n== Validation ranking (top 5) ==\n"
          f"{ranking.head(5).to_string(index=False)}")
    print(f"\n== Test comparison (vs SPY + EW-16) ==\n"
          f"{pd.read_csv(out_dir / 'comparison.csv', index_col=0)}")
    print(f"\n== Forecast ==\n{pd.read_csv(out_dir / 'forecast.csv', index_col=0)['value']}")
    print(f"\n== OOS R2 by year ==\n{pd.read_csv(out_dir / 'oos_r2_by_year.csv', index_col=0)['value']}")
    print(f"\n== OOS R2 vs SPY / EW-16 ==\n{pd.read_csv(out_dir / 'oos_r2_benchmarks.csv', index_col=0)}")
    print(f"\nresults saved to {out_dir}")


if __name__ == "__main__":
    main()
