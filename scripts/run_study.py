#!/usr/bin/env python3
"""Validation study — ONE joint ranking over model x strategy parameters.

There is no separate model-selection phase and no separate book-ablation phase. The
walk-forward predictions for every (spec, look-ahead, refit cadence) are already
stored as CSVs under ``results/_study/predictions``; this script replays them through
every book configuration and ranks the **whole combined space** by net Sharpe on the
combined validation window (2017-01-01..2022-12-31):

    model spec   93   (5 ridge + 4 lasso + 12 enet + 72 rff)
    combo        14   (rebalance, look-ahead, refit) under look-ahead >= cadence and
                      refit_interval >= cadence
    book         45   (vol window 4/8/12/24/48w x vol target 20%/10%/off x leverage
                      cap 1.0/2.0/3.0, rf sleeve fixed at 3%; caps 2.0/3.0 pay a
                      6.8% ann. borrow fee)
    ---------------
    candidates   58,590

Only the book simulation runs — no refitting — so a full ranking is minutes, and the
predictions are reused byte-identically across every book cell of a candidate. Cells
that differ only in an inert knob (the leverage cap when the vol target is off) are
simulated once and share their metrics.

Determinism: ``SEED`` (== ``utils.config.DEFAULT_SEED``) is recorded in each cache
entry and in the ranking/winner files; a cache entry written under another seed is
recomputed. All book cells of a candidate share one 48-bar volatility warmup.

Outputs (``results/_study``):
    validation_ranking.csv   one row per candidate (58,590), sorted by net Sharpe
    validation_winner.json   the top row plus its provenance
    run_parameters.json      the space definition actually searched

Optional: ``test`` runs the single pre-registered draw of the winner on 2023-2025.

Usage:
    python scripts/run_study.py rank                 # joint ranking (default)
    python scripts/run_study.py rank --jobs 4
    python scripts/run_study.py cache                # fit only missing predictions
    python scripts/run_study.py test                 # single draw of the winner
    python scripts/run_study.py rank --models ridge_a10 --combos la4_4w_refitQE
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import pandas as pd
from joblib import Parallel, delayed

import study_worker
from models import ModelSpec, specs_from_config
from pipeline import run_pipeline, run_tag
from prediction_cache import cache_path
from run_backtest import (FREQ_LA_PAIRS, FREQ_TO_BARS, GRID_ENTRIES,
                          REFIT_OPTIONS, REFIT_TO_BARS, build_panel)
from study_worker import (BOOKS, VALIDATION, book_config, combo_slug)
from utils.artifacts import save_table
from utils.config import DEFAULT_SEED, seed_everything
from utils.paths import RESULTS_DIR, STUDY_DIR

SEED = DEFAULT_SEED                          # written into every cache entry
TEST = ("2023-01-01", "2025-12-31")          # single pre-registered draw
VAL_WINDOW_SLUG = "6y"
BATCH = 5                                    # candidates per worker task

# (rebalance, look-ahead, refit) combos under the two design rules.
COMBOS = [(rebalance, horizon, refit)
          for refit in REFIT_OPTIONS
          for rebalance, horizon in FREQ_LA_PAIRS
          if REFIT_TO_BARS[refit] >= FREQ_TO_BARS[rebalance]]

# Set in ``main`` (the panel is built once and handed to the workers).
PANEL: pd.DataFrame | None = None
FEATURE_COLS: list[str] = []
BENCHMARK_CLOSE: pd.Series | None = None
CACHE_ROOT: Path = STUDY_DIR / "predictions"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

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


def _parse_combos(text: str | None) -> list[tuple[str, int, str]]:
    if not text:
        return COMBOS
    known = {combo_slug(rebalance, horizon, refit): (rebalance, horizon, refit)
             for rebalance, horizon, refit in COMBOS}
    wanted = [part.strip() for part in text.split(",") if part.strip()]
    unknown = [name for name in wanted if name not in known]
    if unknown:
        raise SystemExit(f"unknown combo(s) {unknown}; choose from {sorted(known)}")
    return [known[name] for name in wanted]


def _specs(text: str | None) -> list[ModelSpec]:
    specs = specs_from_config(GRID_ENTRIES)
    if not text:
        return specs
    wanted = {part.strip() for part in text.split(",") if part.strip()}
    unknown = wanted - {spec.name for spec in specs}
    if unknown:
        raise SystemExit(f"unknown model spec(s) {sorted(unknown)}")
    return [spec for spec in specs if spec.name in wanted]


# ---------------------------------------------------------------------------
# cache — fill the walk-forward prediction cache, no book simulation
# ---------------------------------------------------------------------------

def run_cache(models: str | None, combos: str | None, refresh: bool, jobs: int) -> None:
    """Fit every missing walk-forward prediction set of the requested space.

    Only the model fits run — no book is simulated and no ranking is written — so this
    is the cheap way to complete a partial cache before ``rank``. Sets already stored
    under matching provenance are reused.
    """
    specs = _specs(models)
    groups = study_worker.forecast_groups(_parse_combos(combos))
    total = len(specs) * len(groups)
    print(f"prediction cache: {len(specs)} specs x {len(groups)} forecast groups = "
          f"{total} prediction sets on {VALIDATION[0]}..{VALIDATION[1]} "
          f"(seed {SEED}, jobs {jobs})", flush=True)

    tasks = [(spec, rebalance, horizon, refit)
             for rebalance, horizon, refit in groups for spec in specs]
    batches = [tasks[i:i + BATCH] for i in range(0, len(tasks), BATCH)]
    progress_every = max(1, len(batches) // 10)

    study_worker.init(PANEL, FEATURE_COLS, BENCHMARK_CLOSE, CACHE_ROOT)

    counts = {"cached": 0, "stored": 0, "degenerate": 0}
    started = time.perf_counter()
    for number, produced in enumerate(
            Parallel(n_jobs=jobs, return_as="generator_unordered",
                     initializer=study_worker.init,
                     initargs=(PANEL, FEATURE_COLS, BENCHMARK_CLOSE, CACHE_ROOT))(
                delayed(study_worker.cache_batch)(batch, refresh) for batch in batches),
            start=1):
        for record in produced:
            counts[record["status"]] += 1
            if record["status"] != "cached":
                print(f"  {record['candidate']}/{record['spec']}: "
                      f"{record['status']} ({record['seconds']:.1f}s)", flush=True)
        if number % progress_every == 0 or number == len(batches):
            done = min(number * BATCH, len(tasks))
            print(f"  {done}/{total} prediction sets checked", flush=True)

    print(f"cache: {counts['stored']} stored, {counts['degenerate']} degenerate, "
          f"{counts['cached']} already cached "
          f"({time.perf_counter() - started:.0f}s) -> {_rel(CACHE_ROOT)}")


# ---------------------------------------------------------------------------
# rank — the single validation stage
# ---------------------------------------------------------------------------

def run_rank(models: str | None, combos: str | None, refresh: bool, jobs: int) -> Path:
    """Replay the cached predictions through the whole model x book space and rank it.

    The winner is the highest net Sharpe on the validation window.
    """
    specs = _specs(models)
    pairs = _parse_combos(combos)
    total = len(specs) * len(pairs) * len(BOOKS)
    print(f"joint ranking: {len(specs)} specs x {len(pairs)} combos x "
          f"{len(BOOKS)} books = {total} candidates on {VALIDATION[0]}..{VALIDATION[1]} "
          f"(seed {SEED}, jobs {jobs}, batch {BATCH})", flush=True)

    tasks = [(spec, rebalance, horizon, refit)
             for rebalance, horizon, refit in pairs for spec in specs]
    batches = [tasks[i:i + BATCH] for i in range(0, len(tasks), BATCH)]
    progress_every = max(1, len(batches) // 20)

    # n_jobs=1 runs in-process and never calls the worker initializer, so seed the
    # state here as well (harmless when the pool does call it).
    study_worker.init(PANEL, FEATURE_COLS, BENCHMARK_CLOSE, CACHE_ROOT)

    rows: list[dict] = []
    for number, produced in enumerate(
            Parallel(n_jobs=jobs, return_as="generator_unordered",
                     initializer=study_worker.init,
                     initargs=(PANEL, FEATURE_COLS, BENCHMARK_CLOSE, CACHE_ROOT))(
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
    _save({"scheme": "joint model x strategy ranking",
           "window": list(VALIDATION), "seed": SEED,
           "specs": [spec.name for spec in specs],
           "combos": [combo_slug(*pair) for pair in pairs],
           "books": [{"name": name, "note": note, "knobs": knobs}
                     for name, knobs, note in BOOKS],
           "candidates": total,
           "cache": _rel(CACHE_ROOT)},
          STUDY_DIR / "run_parameters.json")

    live = ranking[~ranking["degenerate"] & ranking["net_sharpe"].notna()]
    if live.empty:
        raise SystemExit("no non-degenerate candidate; nothing to rank")
    best = live.iloc[0]

    spec = ModelSpec.from_record({"family": best["family"], "alpha": best["alpha"],
                                  "l1_ratio": _optional(best["l1_ratio"]),
                                  "rff_dim": _optional(best["rff_dim"]),
                                  "gamma": _optional(best["gamma"])})
    winner_config = book_config(int(best["look_ahead"]), rebalance=best["rebalance"],
                                refit=best["refit"], model=best["model"],
                                book={"vol_window": int(best["vol_window"]),
                                      "vol_target": _optional(best["vol_target"]),
                                      "vol_rf": float(best["vol_rf"]),
                                      "vol_cap": float(best["vol_cap"]),
                                      "lev_fee": float(best["lev_fee"])})
    payload = {
        "phase": "joint-ranking",
        "window": list(VALIDATION), "seed": SEED,
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
        "cache": _rel(cache_path(winner_config, spec.name, *VALIDATION,
                                 cache_root=CACHE_ROOT)),
        "ranking": _rel(STUDY_DIR / "validation_ranking.csv"),
        "note": "best net Sharpe over the joint model x strategy space; promotion "
                "still requires the pre-registered ADR and the single test draw",
    }
    _save(payload, STUDY_DIR / "validation_winner.json")
    print("\n======== JOINT-RANKING WINNER ========")
    print(json.dumps(_sanitize(payload), indent=2))
    print(f"\nranking: {_rel(STUDY_DIR / 'validation_ranking.csv')} "
          f"({len(ranking)} rows)")
    return STUDY_DIR / "validation_ranking.csv"


# ---------------------------------------------------------------------------
# test — single pre-registered draw of the joint winner
# ---------------------------------------------------------------------------

def run_test() -> None:
    """Measure the ranking winner once on 2023-2025."""
    winner_path = STUDY_DIR / "validation_winner.json"
    if not winner_path.exists():
        raise SystemExit(f"missing {winner_path}; run `python scripts/run_study.py "
                         f"rank` first")
    winner = json.loads(winner_path.read_text())
    if int(winner.get("seed", -1)) != SEED or list(winner.get("window", [])) != list(VALIDATION):
        raise SystemExit("winner file was written under another seed/window; rerun rank")
    model, combo = winner["model"], winner["combo"]
    book = {key: value for key, value in winner["book"].items() if key != "name"}
    config = book_config(int(combo["look_ahead"]), rebalance=combo["rebalance"],
                         refit=combo["refit"], model=model["name"], book=book)
    out_dir = RESULTS_DIR / run_tag(config, *TEST, window_val=VAL_WINDOW_SLUG)
    print(f"TEST draw: {model['name']} / {combo['target']} / "
          f"{combo['rebalance']} / {combo['refit']} with the {winner['book']['name']} "
          f"book on {TEST[0]}..{TEST[1]}")
    run_pipeline(PANEL, FEATURE_COLS, BENCHMARK_CLOSE, config,
                 test_start=TEST[0], test_end=TEST[1], window_val=VAL_WINDOW_SLUG,
                 out_dir=out_dir)
    print(f"results: {_rel(out_dir)}")


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------

def main() -> None:
    global PANEL, FEATURE_COLS, BENCHMARK_CLOSE, STUDY_DIR, CACHE_ROOT
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="rank",
                        choices=("cache", "rank", "test"))
    parser.add_argument("--models", default=None,
                        help="comma-separated spec names (subset runs)")
    parser.add_argument("--combos", default=None,
                        help="comma-separated la<h>_<cadence>_refit<X> (subset runs)")
    parser.add_argument("--study-dir", default=None,
                        help="study output root (default: results/_study)")
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
    print(f"panel: {len(PANEL)} rows, {dates.min():%Y-%m-%d}..{dates.max():%Y-%m-%d}, "
          f"{len(FEATURE_COLS)} features, seed {SEED}")
    print(f"study: {_rel(STUDY_DIR) if str(STUDY_DIR).startswith(str(PROJECT_ROOT)) else STUDY_DIR}")

    if args.command == "cache":
        jobs = args.jobs if args.jobs > 0 else min(8, os.cpu_count() or 1)
        run_cache(args.models, args.combos, args.refresh_cache, jobs)
    elif args.command == "rank":
        jobs = args.jobs if args.jobs > 0 else min(8, os.cpu_count() or 1)
        run_rank(args.models, args.combos, args.refresh_cache, jobs)
    else:
        run_test()


if __name__ == "__main__":
    main()
