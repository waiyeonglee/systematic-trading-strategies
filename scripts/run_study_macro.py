#!/usr/bin/env python3
"""Macro-interaction study — joint ranking of elastic-net x strategy parameters.

Same protocol shape as ``run_study.py`` (one walk-forward prediction cache, one joint
ranking over model x combo x book), but on a different design:

    features   19 firm features x (5 macro + const) = 114 columns  (src/macro.py)
    specs      elastic net only, alpha {1e-4..1e-1} x l1_ratio {0, .25, .5, .75, 1} = 20
    combos     14  (rebalance, look-ahead, refit) under look-ahead >= cadence and
                   refit_interval >= cadence
    books      45  (vol window x vol target x leverage cap, rf sleeve 3%)
    ------------
    candidates 20 x 14 x 45 = 12,600

Split (calendar-anchored, fixed before the run)::

    train        2011-01-01 .. 2018-12-31   (8y = 416-week rolling window per refit)
    validation   2019-01-01 .. 2022-12-31   (the ranking window)
    test         2023-01-01 .. 2025-12-31   (single draw of the winner, measured once)

The study keeps its own outputs (``results/_study_macro``) and its own prediction
cache, so it can never read — or overwrite — the v2 study's 21-feature predictions or
the pre-registered figures in ``results/plots/``.

Usage:
    python scripts/run_study_macro.py rank
    python scripts/run_study_macro.py rank --refits QE      # quarterly arm only (~8 min)
    python scripts/run_study_macro.py rank --jobs 4
    python scripts/run_study_macro.py test
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import pandas as pd
from joblib import Parallel, delayed

import study_worker
from macro import MACRO_COLUMNS
from macro_panel import build_panel
from models import ModelSpec, specs_from_config
from pipeline import run_pipeline, run_tag
from run_backtest import FREQ_LA_PAIRS, FREQ_TO_BARS, REFIT_OPTIONS, REFIT_TO_BARS
from study_worker import BOOKS, book_config, combo_slug
from utils.artifacts import save_table
from utils.config import DEFAULT_SEED, seed_everything
from utils.paths import RESULTS_DIR

SEED = DEFAULT_SEED
TRAIN = ("2011-01-01", "2018-12-31")
VALIDATION = ("2019-01-01", "2022-12-31")
TEST = ("2023-01-01", "2025-12-31")
WINDOW_TRAIN = "8y"                 # 416 weekly bars, the stated train span
VAL_WINDOW_SLUG = "4y"              # window_val that reproduces 2019-2022 before 2023

STUDY_DIR = RESULTS_DIR / "_study_macro"
CACHE_ROOT = STUDY_DIR / "predictions"
PLOTS_DIR = RESULTS_DIR / "plots_macro"
BATCH = 5

# Elastic net only: alpha x l1_ratio, with 0 (ridge) and 1 (lasso) as the family ends.
ENET_ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1)
ENET_L1_RATIOS = (0.0, 0.25, 0.5, 0.75, 1.0)
GRID_ENTRIES = [{"family": "enet", "alpha": list(ENET_ALPHAS),
                 "l1_ratio": list(ENET_L1_RATIOS)}]

COMBOS = [(rebalance, horizon, refit)
          for refit in REFIT_OPTIONS
          for rebalance, horizon in FREQ_LA_PAIRS
          if REFIT_TO_BARS[refit] >= FREQ_TO_BARS[rebalance]]

PANEL: pd.DataFrame | None = None
FEATURE_COLS: list[str] = []
BENCHMARK_CLOSE: pd.Series | None = None


def _sanitize(value):
    """Convert numpy/pandas scalars to plain JSON types."""
    import numpy as np
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, pd.Timestamp):
        return str(value.date())
    if isinstance(value, dict):
        return {str(k): _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    return value


def _save(obj, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_sanitize(obj), indent=2, default=str))
    return path


def _rel(path: Path) -> str:
    """Project-relative posix path, for portable records."""
    return Path(path).relative_to(PROJECT_ROOT).as_posix()


def _optional(value):
    """NaN -> None so records round-trip cleanly through JSON."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    return value


def _specs(text: str | None) -> list[ModelSpec]:
    specs = specs_from_config(GRID_ENTRIES)
    if not text:
        return specs
    wanted = {part.strip() for part in text.split(",") if part.strip()}
    unknown = wanted - {spec.name for spec in specs}
    if unknown:
        raise SystemExit(f"unknown model spec(s) {sorted(unknown)}")
    return [spec for spec in specs if spec.name in wanted]


def _parse_combos(text: str | None, combos: list[tuple[str, int, str]]) -> list[tuple[str, int, str]]:
    if not text:
        return combos
    known = {combo_slug(rebalance, horizon, refit): (rebalance, horizon, refit)
             for rebalance, horizon, refit in combos}
    wanted = [part.strip() for part in text.split(",") if part.strip()]
    unknown = [name for name in wanted if name not in known]
    if unknown:
        raise SystemExit(f"unknown combo(s) {unknown}; choose from {sorted(known)}")
    return [known[name] for name in wanted]


def _combos(refits: str | None) -> list[tuple[str, int, str]]:
    """Combos restricted to the requested refit cadences (default: all)."""
    if not refits:
        return COMBOS
    wanted = {part.strip() for part in refits.split(",") if part.strip()}
    unknown = wanted - set(REFIT_OPTIONS)
    if unknown:
        raise SystemExit(f"unknown refit cadence(s) {sorted(unknown)}; "
                         f"choose from {sorted(REFIT_OPTIONS)}")
    return [combo for combo in COMBOS if combo[2] in wanted]


def _config(horizon: int, rebalance: str, refit: str, model: str, book: dict):
    """Config of one cell under the macro study's window and grid."""
    return book_config(horizon, rebalance=rebalance, refit=refit, model=model,
                       book=book, window_train=WINDOW_TRAIN,
                       grid_entries=GRID_ENTRIES)


# ---------------------------------------------------------------------------
# rank — the single validation stage
# ---------------------------------------------------------------------------

def run_rank(models: str | None, combos: str | None, refits: str | None, refresh: bool,
             jobs: int) -> Path:
    """Replay the cached predictions through the whole model x book space and rank it.

    The winner is the highest net Sharpe on the validation window.

    ``refits`` restricts the refit cadences (``"QE"`` = the quarterly arm only, which
    is ~85% cheaper than including the 208 weekly refits per prediction set).
    """
    specs = _specs(models)
    pairs = _parse_combos(combos, _combos(refits))
    total = len(specs) * len(pairs) * len(BOOKS)
    print(f"macro joint ranking: {len(specs)} specs x {len(pairs)} combos x "
          f"{len(BOOKS)} books = {total} candidates on {VALIDATION[0]}..{VALIDATION[1]} "
          f"(train {TRAIN[0]}..{TRAIN[1]}, window {WINDOW_TRAIN}, seed {SEED}, "
          f"{len(FEATURE_COLS)} features, jobs {jobs})", flush=True)

    tasks = [(spec, rebalance, horizon, refit)
             for rebalance, horizon, refit in pairs for spec in specs]
    batches = [tasks[i:i + BATCH] for i in range(0, len(tasks), BATCH)]
    progress_every = max(1, len(batches) // 20)

    # n_jobs=1 runs in-process and never calls the worker initializer, so seed the
    # state here as well (harmless when the pool does call it).
    study_worker.init(PANEL, FEATURE_COLS, BENCHMARK_CLOSE, CACHE_ROOT,
                      VALIDATION, WINDOW_TRAIN, GRID_ENTRIES)

    rows: list[dict] = []
    for number, produced in enumerate(
            Parallel(n_jobs=jobs, return_as="generator_unordered",
                     initializer=study_worker.init,
                     initargs=(PANEL, FEATURE_COLS, BENCHMARK_CLOSE, CACHE_ROOT,
                               VALIDATION, WINDOW_TRAIN, GRID_ENTRIES))(
                delayed(study_worker.score_batch)(batch, refresh) for batch in batches),
            start=1):
        rows.extend(produced)
        if number % progress_every == 0 or number == len(batches):
            done = min(number * BATCH, len(tasks))
            print(f"  {done}/{len(tasks)} candidates scored ({len(rows)} rows)",
                  flush=True)

    ranking = (pd.DataFrame(rows)
               .sort_values("net_sharpe", ascending=False, na_position="last")
               .reset_index(drop=True))
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    save_table(ranking, STUDY_DIR, "validation_ranking", index=False)
    _save({"scheme": "macro-interaction joint model x strategy ranking",
           "split": {"train": list(TRAIN), "validation": list(VALIDATION),
                     "test": list(TEST)},
           "window_train": WINDOW_TRAIN,
           "features": {"firm": 19, "macro": 5, "const": 1,
                        "interaction_columns": len(FEATURE_COLS)},
           "window": list(VALIDATION), "seed": SEED,
           "specs": [spec.name for spec in specs],
           "combos": [combo_slug(*pair) for pair in pairs],
           "refits": sorted({pair[2] for pair in pairs}),
           "books": [{"name": name, "note": note, "knobs": knobs}
                     for name, knobs, note in BOOKS],
           "candidates": total,
           "cache": _rel(CACHE_ROOT)},
          STUDY_DIR / "run_parameters.json")

    live = ranking[~ranking["degenerate"] & ranking["net_sharpe"].notna()]
    if live.empty:
        raise SystemExit("no non-degenerate candidate; nothing to rank")
    best = live.iloc[0]

    winner_config = _config(int(best["look_ahead"]), best["rebalance"], best["refit"],
                            best["model"],
                            {"vol_window": int(best["vol_window"]),
                             "vol_target": _optional(best["vol_target"]),
                             "vol_rf": float(best["vol_rf"]),
                             "vol_cap": float(best["vol_cap"]),
                             "lev_fee": float(best["lev_fee"])})
    payload = {
        "phase": "macro joint-ranking",
        "split": {"train": list(TRAIN), "validation": list(VALIDATION),
                  "test": list(TEST)},
        "window": list(VALIDATION), "seed": SEED,
        "window_train": WINDOW_TRAIN,
        "features": {"firm_features": 19, "macro": list(MACRO_COLUMNS),
                     "interaction_columns": len(FEATURE_COLS)},
        "candidates": {"specs": len(specs), "combos": len(pairs),
                       "books": len(BOOKS), "scored": int(len(ranking))},
        "model": {"name": best["model"], "family": best["family"],
                  "alpha": float(best["alpha"]),
                  "l1_ratio": _optional(best["l1_ratio"]),
                  "rff_dim": _optional(best["rff_dim"]),
                  "gamma": _optional(best["gamma"])},
        "combo": {"target": best["target"], "look_ahead": int(best["look_ahead"]),
                  "rebalance": best["rebalance"], "refit": best["refit"]},
        "book": {"name": best["book"], "vol_window": int(best["vol_window"]),
                 "vol_target": _optional(best["vol_target"]),
                 "vol_rf": float(best["vol_rf"]), "vol_cap": float(best["vol_cap"]),
                 "lev_fee": float(best["lev_fee"])},
        "metrics": {"net_sharpe": float(best["net_sharpe"]),
                    "gross_sharpe": float(best["gross_sharpe"]),
                    "max_drawdown": float(best["max_drawdown"]),
                    "total_return": float(best["total_return"]),
                    "ann_return": float(best["ann_return"]),
                    "ann_vol": float(best["ann_vol"]),
                    "hit_rate": float(best["hit_rate"]),
                    "avg_turnover": float(best["avg_turnover"])},
        "cache": _rel(CACHE_ROOT / f"{VALIDATION[0]}_{VALIDATION[1]}"
                      / f"weekly_la{int(best['look_ahead'])}_refit{best['refit']}"
                      / f"{best['model']}.npz"),
        "window_train_bars": winner_config.window_train_bars,
        "features_used": len(FEATURE_COLS),
        "ranking": _rel(STUDY_DIR / "validation_ranking.csv"),
        "note": "elastic-net-only macro-interaction study; promotion still requires the "
                "pre-registered ADR and the single test draw",
    }
    winner_path = _save(payload, STUDY_DIR / "validation_winner.json")
    print(f"winner: {best['model']} / {best['target']} / "
          f"{best['rebalance']} / {best['refit']} / {best['book']} -> net Sharpe "
          f"{best['net_sharpe']:.4f}")
    print(f"written: {_rel(winner_path)}")
    return winner_path


# ---------------------------------------------------------------------------
# test — the single draw of the ranking winner
# ---------------------------------------------------------------------------

def run_test() -> None:
    """Measure the ranking winner once on 2023-2025 and publish its figures.

    The figures go to ``results/plots_macro/{test,val}/`` — a separate root from the
    v2 study's ``results/plots/``, so both figure sets coexist.
    """
    winner_path = STUDY_DIR / "validation_winner.json"
    if not winner_path.exists():
        raise SystemExit(f"missing {winner_path}; run `python scripts/run_study_macro.py "
                         f"rank` first")
    winner = json.loads(winner_path.read_text())
    if (int(winner.get("seed", -1)) != SEED
            or list(winner.get("window", [])) != list(VALIDATION)):
        raise SystemExit("winner file was written under another seed/window; rerun rank")
    model, combo = winner["model"], winner["combo"]
    book = {key: value for key, value in winner["book"].items() if key != "name"}
    config = _config(int(combo["look_ahead"]), combo["rebalance"], combo["refit"],
                     model["name"], book)
    out_dir = RESULTS_DIR / (run_tag(config, *TEST, window_val=VAL_WINDOW_SLUG) + "_macro")
    print(f"macro TEST draw: {model['name']} / {combo['target']} / "
          f"{combo['rebalance']} / {combo['refit']} with the {winner['book']['name']} "
          f"book on {TEST[0]}..{TEST[1]} (train 2011-2018, val 2019-2022)")
    run_pipeline(PANEL, FEATURE_COLS, BENCHMARK_CLOSE, config,
                 test_start=TEST[0], test_end=TEST[1], window_val=VAL_WINDOW_SLUG,
                 out_dir=out_dir, publish_figures=True, plots_root=PLOTS_DIR)
    print(f"results: {_rel(out_dir)}")
    print(f"figures: {_rel(PLOTS_DIR)}/{{test,val}}/  (v2 results/plots/ untouched)")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global PANEL, FEATURE_COLS, BENCHMARK_CLOSE, STUDY_DIR, CACHE_ROOT
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="rank", choices=("rank", "test"))
    parser.add_argument("--models", default=None,
                        help="comma-separated spec names (subset runs)")
    parser.add_argument("--combos", default=None,
                        help="comma-separated la<h>_<cadence>_refit<X> (subset runs)")
    parser.add_argument("--refits", default=None,
                        help="comma-separated refit cadences to keep, e.g. QE "
                             "(default: all of W-FRI,ME,QE; the weekly arm costs "
                             "~85%% of the prediction compute)")
    parser.add_argument("--study-dir", default=None,
                        help="study output root (default: results/_study_macro)")
    parser.add_argument("--jobs", type=int, default=0,
                        help="worker processes (0 = min(8, cpu count))")
    parser.add_argument("--refresh-cache", action="store_true",
                        help="recompute cached predictions instead of reusing them")
    args = parser.parse_args()

    if args.study_dir:
        root = Path(args.study_dir)
        STUDY_DIR = root if root.is_absolute() else PROJECT_ROOT / root
    CACHE_ROOT = STUDY_DIR / "predictions"

    seed_everything(SEED)
    PANEL, BENCHMARK_CLOSE, FEATURE_COLS = build_panel()
    dates = PANEL.index.get_level_values("date")
    print(f"macro panel: {len(PANEL)} rows, {dates.min():%Y-%m-%d}..{dates.max():%Y-%m-%d}, "
          f"{len(FEATURE_COLS)} interaction features, seed {SEED}")
    print(f"split: train {TRAIN[0]}..{TRAIN[1]} | validation {VALIDATION[0]}..{VALIDATION[1]} "
          f"| test {TEST[0]}..{TEST[1]} | rolling window {WINDOW_TRAIN} "
          f"({len(GRID_ENTRIES)}-family grid, no rff)")
    print(f"study: {_rel(STUDY_DIR)}")

    if args.command == "rank":
        jobs = args.jobs if args.jobs > 0 else min(8, os.cpu_count() or 1)
        run_rank(args.models, args.combos, args.refits, args.refresh_cache, jobs)
    else:
        run_test()


if __name__ == "__main__":
    main()
