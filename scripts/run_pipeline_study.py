#!/usr/bin/env python3
"""End-to-end study run for ONE feature block: panel -> cache -> rank -> draw -> plots.

The single knob is the feature block. Everything else (split, grid, books, refit arms,
seed) is fixed by the study protocol in ``run_study.py``, so a run is reproducible and
the two blocks are directly comparable:

    micro   the 19 firm features, no macro interaction          ( 19 model columns)
    macro   the 19 firm features x (const + 5 macro variables)  (114 model columns)

Stages, in order:

    1. cache   fit every missing walk-forward prediction set on the validation window
               (no book simulation); a fully filled cache makes this a no-op
    2. rank    replay the cache through model x combo x book (9,900 candidates) and
               write validation_ranking.csv / validation_winner.json
    3. draw    measure the best candidate of **each refit arm** (ME and QE) once on
               2023-2025 and publish its figures under
               ``results/_study_<block>/plots/<arm>/{test,val}/`` (10 per window),
               plus ``draws/arm_comparison.txt``

Both refit arms are therefore drawn and plotted, not just the winner: the ME and QE
arm folders both end up populated, so the two cadences can be compared directly.

Arm-vs-arm comparison (ME-best, QE-best, net best) with its own overlay figures and
metric tables is a separate step: ``python scripts/compare_arms.py <block>``.

Outputs all live inside ``results/_study_<block>`` so the two blocks never mix.

Usage:
    python scripts/run_pipeline_study.py micro          # one block, end to end
    python scripts/run_pipeline_study.py macro
    python scripts/run_pipeline_study.py both           # micro then macro
    python scripts/run_pipeline_study.py macro --jobs 4
    python scripts/run_pipeline_study.py micro --skip-cache
    python scripts/run_pipeline_study.py micro --refresh-cache   # refit everything

Exits non-zero if any stage fails, so it is safe to chain in CI or a shell script.
"""

from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for _path in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import run_study
from macro_panel import FIRM_COLUMNS, FEATURE_SETS, build_panel, feature_columns
from models import specs_from_config
from run_backtest import GRID_ENTRIES
from study_worker import VALIDATION


def _heading(text: str) -> None:
    print(f"\n{'=' * 74}\n{text}\n{'=' * 74}", flush=True)


def run_block(feature_set: str, jobs: int, skip_cache: bool, refresh: bool) -> Path:
    """Run cache -> rank -> draw for one feature block; return the winner file.

    The draw stage measures the best candidate of *every* refit arm (ME and QE), so
    both arms' figures are published under ``plots/<arm>/{test,val}/``.

    Raises:
        SystemExit: if a stage fails (the caller records it and keeps going).
    """
    settings = run_study.study_settings(feature_set)
    columns = settings["feature_cols"]

    _heading(f"{feature_set}: {len(columns)} model columns -> "
             f"{run_study._rel(settings['study_dir'])}")
    started = time.perf_counter()

    if skip_cache:
        print("stage 1/3 cache: skipped (--skip-cache)", flush=True)
    else:
        print("stage 1/3 cache", flush=True)
        run_study.run_cache(
            models=None, combos=None, refits=None,
            refresh=refresh, jobs=jobs, settings=settings)

    print("\nstage 2/3 rank", flush=True)
    winner = run_study.run_rank(
        models=None, combos=None, refits=None,
        refresh=False, jobs=jobs, settings=settings)

    print("\nstage 3/3 per-arm draws + figures (ME and QE)", flush=True)
    run_study.run_draw(settings=settings)

    if not winner.exists():
        raise SystemExit(f"{feature_set}: rank stage wrote no {winner}")

    print(f"\n{feature_set}: done in {time.perf_counter() - started:.0f}s", flush=True)
    return winner


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("features", nargs="?", default="both",
                        choices=(*FEATURE_SETS, "both"),
                        help="feature block to run: micro, macro, or both (default)")
    parser.add_argument("--jobs", type=int, default=0,
                        help="worker processes per stage (0 = min(8, cpu count))")
    parser.add_argument("--skip-cache", action="store_true",
                        help="skip stage 1 entirely (assume the cache is complete)")
    parser.add_argument("--refresh-cache", action="store_true",
                        help="refit every prediction set instead of reusing the cache")
    args = parser.parse_args()

    blocks = list(FEATURE_SETS) if args.features == "both" else [args.features]
    import os
    jobs = args.jobs if args.jobs > 0 else min(8, os.cpu_count() or 1)

    _heading(f"end-to-end study: {' + '.join(blocks)}")
    panel, benchmark, _ = build_panel()
    dates = panel.index.get_level_values("date")
    print(f"panel   : {len(panel)} rows, {dates.min():%Y-%m-%d}..{dates.max():%Y-%m-%d}, "
          f"{len(FIRM_COLUMNS)} firm features")
    print(f"split   : validation {VALIDATION[0]}..{VALIDATION[1]} | "
          f"test {run_study.TEST[0]}..{run_study.TEST[1]}")
    print(f"grid    : {len(specs_from_config(GRID_ENTRIES))} specs x "
          f"{len(run_study.COMBOS)} combos x {len(run_study.BOOKS)} books per block")
    print(f"seed    : {run_study.SEED} | jobs {jobs}", flush=True)

    run_study.init_run(panel, benchmark)
    failures: list[tuple[str, str]] = []
    winners: dict[str, Path] = {}
    started = time.perf_counter()

    for block in blocks:
        try:
            winners[block] = run_block(block, jobs, args.skip_cache,
                                       args.refresh_cache)
        except SystemExit as exc:
            failures.append((block, str(exc)))
            print(f"\n!! {block} failed: {exc}", flush=True)
        except Exception:
            failures.append((block, traceback.format_exc(limit=3)))
            print(f"\n!! {block} failed:\n{traceback.format_exc()}", flush=True)

    _heading(f"summary ({time.perf_counter() - started:.0f}s)")
    import json
    for block in blocks:
        if block in failures:
            detail = next(msg for name, msg in failures if name == block)
            print(f"{block:5s} FAILED  {detail.splitlines()[0]}")
            continue
        payload = json.loads(winners[block].read_text())
        model, combo, book = payload["model"], payload["combo"], payload["book"]
        study_dir = winners[block].parent
        print(f"{block:5s} ok      winner {model['name']} / la{combo['look_ahead']} / "
              f"{combo['rebalance']} / refit {combo['refit']} / {book['name']} "
              f"-> val net Sharpe {payload['metrics']['net_sharpe']:.4f}")
        arms = sorted(p.name for p in (study_dir / "plots").iterdir() if p.is_dir())
        for arm in arms:
            n = len(list((study_dir / "plots" / arm).rglob("*.png")))
            print(f"       {run_study._rel(study_dir / 'plots' / arm)}/"
                  f"{{test,val}}/  ({n} figures)")
        if (study_dir / "draws" / "arm_comparison.txt").exists():
            print(f"       {run_study._rel(study_dir / 'draws' / 'arm_comparison.txt')}")

    if failures:
        print(f"\n{len(failures)} block(s) failed", flush=True)
        return 1
    print("\nall blocks complete", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
