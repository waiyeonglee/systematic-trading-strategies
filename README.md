# systematic-trading-strategies

Standalone implementation of a cross-asset weekly trend-following strategy
(DSA5205 Project 2). Universe: 16 tradable ETFs (9 US sectors + 7 diversifiers)
with SPY as benchmark, plus macro state interactions including the CNN Fear & Greed
index.

## Quick start (end to end)

```bash
# 0. setup: create .venv with pinned dependencies (Python 3.13)
uv sync
source .venv/bin/activate

# 1. data (optional): raw weekly OHLCV, Data2025.xlsx and fear-greed are committed;
#    only re-fetch from Yahoo Finance if you want to refresh them
# python scripts/fetch_data.py

# 2. features: features.csv (micro) + features_macro.csv (114-column macro panel)
python scripts/build_features.py

# 3. study, per feature block: cache predictions -> rank on 2019-2022 val -> draw 2023-2025 test -> plots
#    also writes the selected micro/macro validation comparison
#    -> results/_study_{micro,macro}/ + results/validation_comparison/
python scripts/run_pipeline_study.py both

# 4. ME vs QE arm comparison (+ AR(1) baseline) -> results/_study_<block>/comparison/
python scripts/compare_arms.py both

# 5. standalone baseline on validation/test windows
python scripts/run_forecast_baseline.py --horizon 4  # AR(1) forecasts + traded book -> results/forecast_baseline/

```

## Dataset (fixed window)

- End date fixed at **2026-01-01** (fetch end exclusive **2026-01-02**).
- Start 2004-01-01 (pre-2011 history is feature warmup only); study window 2011+.
- See `data/README.md` for the full conventions.

### Macro inputs and the derived feature file

The macro feature block adds five monthly state variables on top of the
price-based features:

| Variable | Meaning | Source |
| --- | --- | --- |
| `fear_greed` | CNN Fear & Greed index (0-100), monthly average | `data/additional/fear-greed.csv` (committed) |
| `tbl` | 3-month T-bill rate | `data/raw/Data2025.xlsx`, sheet `Monthly` |
| `tms` | term spread, `lty - tbl` | same workbook |
| `dfy` | default yield spread, `BAA - AAA` | same workbook |
| `dfr` | default return spread, `corpr - ltr` | same workbook |

The workbook is the Welch-Goyal predictor dataset (monthly, 1871-01..2025-12),
downloaded from [Amit Goyal's predictor-data page](https://sites.google.com/view/agoyal145)
— see Goyal & Welch (2008), *A Comprehensive Look at the Empirical Performance of
Equity Premium Prediction*, RFS 21(4). `dfr` has no FRED equivalent, which is why the
workbook is the source. A bar dated in month `M` uses
the month `M - 1` value, so nothing is used before it is published. Crossing the 19
firm features with `(const + 5 macro)` gives **114 columns**; see
`agents/macro_study.md` for the protocol.

`data/features/weekly/features_macro.csv` is the derived 114-column panel written for
inspection. It is **~30 MB and not committed** (listed in `.gitignore`) — the study
rebuilds the panel in memory, so a fresh clone trains without it. `scripts/build_features.py`
builds it alongside `features.csv`; regenerate both with:

```bash
python scripts/build_features.py            # features.csv + features_macro.csv
python scripts/build_features.py --no-macro # features.csv only
python scripts/build_features.py --macro-out /tmp/macro.csv  # custom macro destination
```

## Layout

```
scripts/fetch_data.py        # optional re-fetch from Yahoo Finance (end exclusive 2026-01-02)
scripts/build_features.py    # builds data/features/weekly/ from the raw CSVs:
                             # features.csv (micro) + features_macro.csv (114-column panel)
scripts/run_study.py         # joint feature-block x model x strategy ranking on the combined
                             # window + test draw; --features micro|macro (default: both),
                             # micro -> results/_study_micro, macro -> results/_study_macro
scripts/run_pipeline_study.py  # END TO END: cache -> rank -> draw -> plots
                             #   running both also writes results/validation_comparison/
                             #   python scripts/run_pipeline_study.py micro|macro|both
scripts/compare_arms.py      # ME vs QE side-by-side figures + metric txt (validation & test)
                             #   python scripts/compare_arms.py micro|macro|both
scripts/run_backtest.py      # single walk-forward backtest / test draw (grid or pinned spec)
scripts/run_forecast_baseline.py  # AR(1) baseline: forecasts + traded book, per window
                             # -> results/forecast_baseline/ar1_h<H>/
src/                         # labels.py, features.py, macro.py, models.py, splits.py,
                             # prediction_cache.py, portfolio.py, metrics.py,
                             # baselines.py, pipeline.py, plots.py, utils/
agents/                      # current_state.md (history log), feature_engineering.md,
                             # model_design.py, validation_and_ablation.md, macro_study.md, plots.md
data/                        # raw weekly OHLCV + fear-greed + derived features
results/                     # run tables per tag + _study_<block>/ summaries and the prediction
                             # cache (committed, ~22 MB npz); study draws (tables, series)
                             # and figures live under results/_study_<block>/draws|plots/<arm>/;
                             # results/forecast_baseline/ holds the AR(1) forecast metrics,
                             # its traded book, and the per-year breakdown
```

## Setup

```bash
uv sync          # create .venv and install pinned dependencies (Python 3.13)
```

## Run

```bash
python scripts/build_features.py                # weekly, H=12
                                                # -> features.csv + features_macro.csv
python scripts/build_features.py --horizon 4
python scripts/build_features.py --no-macro     # skip the 114-column macro panel

python scripts/run_backtest.py                  # single walk-forward run: select on the combined
                                                # 2019-2022 window (8y rolling train, 4y val),
                                                # 20-spec Elastic-Net grid, refit QE, 4w cadence,
                                                # vol target 10% / rf 3%; measures the test span
                                                # (default 2023-2025). Writes tables/series only;
                                                # report plots are study outputs below.
python scripts/run_backtest.py --model enet_a0.01_r1 --horizon 1 --rebalance weekly # single cell
python scripts/run_backtest.py --horizon 4 --no-vol-target
python scripts/run_backtest.py --test-start 2020-01-01 --test-end 2022-12-31 --window-val 3y
# ad-hoc spans write tables/series only: report figures are published solely for the
# pre-registered 2023-01-01..2025-12-31 window (pipeline.REPORT_TEST_WINDOW)
# design rules (enforced in default_config):
#   look-ahead must be >= rebalance frequency: weekly(1w) x {1,4,12} · 4w x {4,12} · 12w x {12}
#   refit must be no more frequent than the cadence (refit_interval >= prediction_interval):
#   ME x {1w,4w cadences} (5) · QE x all pairs (6) -> 11 combos

python scripts/run_study.py rank                # ONE joint ranking over the combined
                                                # space of ONE feature block: 20 specs x
                                                # 11 (rebalance, look-ahead, refit)
                                                # combos x 45 book cells = 9,900
                                                # candidates on 2019-2022, best net
                                                # Sharpe wins. Runs both blocks in turn
                                                # (micro, then macro) unless narrowed.
                                                # -> results/_study_<block>/
                                                #    validation_ranking.csv
                                                # (~3-4 min on 8 workers; the model fits
                                                # come from the prediction cache)
python scripts/run_study.py rank --features macro  # one block only
python scripts/run_study.py cache               # fit only the missing predictions of
                                                # that space (no book simulation), then
                                                # re-run rank
python scripts/run_study.py draw                # final holdout only, after selection lock:
                                                # one 2023-2025 draw per refit arm
                                                # -> results/_study_<block>/draws/<arm>/
                                                # + draws/arm_comparison.txt
python scripts/run_study.py test                # final holdout only, after selection lock:
                                                # single pre-registered draw (2023-2025)
                                                # of that block's ranking winner
python scripts/run_study.py rank --jobs 4       # worker processes (0 = min(8, cpu))

# ONE command for the whole chain of a block: cache -> rank -> draw -> figures.
# The draw stage measures the best candidate of BOTH refit arms, so each block ends
# up with plots/me/ and plots/qe/. This is the usual entry point; the run_study.py
# stages above are for partial / debugging runs. When run for both blocks, it also
# writes results/validation_comparison/cumulative_val_micro_macro_ar1.png.
python scripts/run_pipeline_study.py both       # micro then macro, end to end
python scripts/run_pipeline_study.py micro      # one block only
python scripts/run_pipeline_study.py macro --jobs 4

# Arm comparison: the ME and QE arms side by side (ME left, QE right), with a red ★
# on the arm selected on the validation window, plus the validation and test metrics.
# Every panel and table also carries the AR(1) baseline: a one-lag regression of the
# H-bar forward return on the trailing H-bar return (no features), fitted on the same
# split and traded with the selected arm's own book, costs and refit cadence - so the
# difference between it and the arms is purely the feature set.
# Needs a ranking to exist (run `rank`, or the end-to-end command above, first).
#   -> results/_study_<block>/comparison/
#        figures/cumulative_{test,val}.png   ME | QE, one panel each, + AR(1) baseline
#        figures/drawdown_test.png           underwater curves, same layout
#        metrics_aggregate.txt               book metrics per arm x window
#        metrics_forecast.txt                OOS R² vs zero / SPY / EW-16
#        metrics_by_year.txt                 the same split by calendar year
#      (* marks the validation-selected arm in every table; the AR(1) baseline is
#       the last row of every table and a grey dashed line in every figure)
python scripts/compare_arms.py both             # both blocks
python scripts/compare_arms.py macro            # one block only

# AR(1) baseline: forecasts (OOS R² vs zero and vs the historical mean) plus a traded
# book on the study's own cadence, so it is comparable on Sharpe and drawdown.
# Writes results/forecast_baseline/ar1_h<H>/{metrics,by_year,book_metrics,book_by_year}.csv
# plus per-window predictions.csv and book.csv; H=4 matches the micro winner's horizon.
python scripts/run_forecast_baseline.py --horizon 4
python scripts/run_forecast_baseline.py --horizon 1

python scripts/build_features.py --no-macro      # micro only; default also writes the
                                                # 19 firm x (5 macro + const) = 114-column
                                                # panel -> data/features/weekly/features_macro.csv
                                                # (git-ignored, inspection only)

# the macro block of the same ranking: --features macro reads the 114 interaction
# columns instead of the 19 firm features, into its own results/_study_macro folder
python scripts/run_study.py rank --features macro   # 20 enet specs x
                                                # 11 ME/QE combos x 45 books = 9,900
                                                # candidates on 2019-2022 (same 8y
                                                # rolling train)
                                                # -> results/_study_macro/validation_ranking.csv
python scripts/run_study.py cache --features macro   # fit only the missing prediction sets
                                                # (6 (look-ahead, refit) forecast groups)
python scripts/run_study.py rank --features macro --refits QE  # quarterly arm only
python scripts/run_study.py draw --features macro   # final holdout only, after selection lock:
                                                # one 2023-2025 draw per refit arm
                                                # -> results/_study_macro/draws/<arm>/
python scripts/run_study.py test --features macro   # final holdout only, after selection lock:
                                                # single draw of the overall winner
python scripts/plot_validation_comparison.py    # optional: regenerate only the selected
                                                # micro/macro validation comparison figure
```

Study outputs: the walk-forward predictions are cached under
`results/_study_<block>/predictions/<window>/<freq>_<la>_refit<slug>/` as
`<spec>.npz` (float32 matrix + axes) with a `<spec>.json` provenance sidecar
(seed `utils.config.DEFAULT_SEED`, feature block); degenerate prediction sets — constant
across tickers — are not stored,
they get a marker under `<window>/_degenerate/` and are skipped by the ranking. The
joint ranking lands in `results/_study_<block>/validation_ranking.csv` (one row per
candidate, sorted by net Sharpe) with the winner plus its provenance in
`results/_study_<block>/validation_winner.json` and the searched space in
`results/_study_<block>/run_parameters.json`. Only the book simulation runs — the model
fits come from the cache — and a cache entry written under another seed is recomputed.

When the final holdout is run, test draws and their figures live inside the study
folder:

```
results/_study_{micro,macro}/
  draws/<arm>/<run_tag>/      # arm = me | qe; tables, series/ (no per-run plots/)
  plots/<arm>/{test,val}/     # that draw's 10 figures per window (with AR(1) baseline)
  arm_comparison.txt          # only after `draw`: one row per arm + which arm wins
                              # on the test window (descriptive, not a selection rule)
  comparison/                 # written by scripts/compare_arms.py
    figures/cumulative_{test,val}.png   # ME | QE panels, red ★ on the selected arm
    figures/drawdown_test.png           # underwater curves, same layout
    metrics_aggregate.txt               # book metrics per arm x window (+ AR(1))
    metrics_forecast.txt                # OOS R² vs zero / SPY / EW-16 (+ AR(1))
    metrics_by_year.txt                 # the same split by calendar year (+ AR(1))
```

The `comparison/` block is a *read* of the existing ranking: it re-runs each arm's
best candidate through the book (predictions come from the cache) and needs no
re-ranking, so `python scripts/compare_arms.py both` takes ~20 s once the rankings
exist. In every figure and table the arms appear in a fixed order — **ME first, then
QE** — and a red ★ (figures) or a trailing `*` (tables) marks the arm selected on the
validation window. The test window is reported for both arms but selects nothing.

Every table in `comparison/`, and every `plots/<arm>/{test,val}/` figure that shows a
benchmark (01 cumulative, 03 drawdown, 09 annual), also carries the **AR(1) baseline**
(`baselines.ar1_baseline`): a per-ticker one-lag regression of the H-bar forward
return on the trailing H-bar return, with no features at all. It is reported last, in
a grey dashed line, and is fitted on exactly the split the selected arm used (same
refit cadence, rolling window and label) and traded with that arm's own book and costs
— so any difference between it and the arms is attributable to the features, not to
the book. It is deliberately kept out of `benchmarks`, which feed the success gate and
`comparison.csv`. On the test window it is a demanding bar: in the current draws it
beats the micro arms outright and comes close to the macro QE arm, which is worth
weighing before reading the feature blocks as an improvement over a trailing-return
rule.

The direct backtest saves tables under `results/<run_tag>/` (validation ranking,
comparison vs SPY + EW-16, forecast metrics, coefficients, run config) and raw time
series under `series/`; it runs with `publish_figures=False`, so it does not create
report plots.

Report plots are study outputs with an explicit plot root. The final macro test draw,
for example, is reproduced by:

```bash
python scripts/run_study.py test --features macro
```

Internally that call passes:

```python
plots_root=Path("results/_study_macro/plots/me")
```

and publishes the 10 holdout figures to `results/_study_macro/plots/me/test/` and the
10 validation figures to `results/_study_macro/plots/me/val/`. The end-to-end `both`
pipeline also writes the combined report comparison:

```bash
python scripts/run_pipeline_study.py both
```

To regenerate only that figure after both validation winners exist, run:

```bash
python scripts/plot_validation_comparison.py
```

Both paths write `results/validation_comparison/cumulative_val_micro_macro_ar1.png`. See
`agents/plots.md` for the figure list and rationale. Both blocks have been ranked and
drawn once on the 2023-2025 test window after their validation winners were locked;
the test window is reported, never used for selection.
See `agents/current_state.md` for the current results state, `agents/feature_engineering.md`
for the features, `agents/model_design.py` for the design (grid, portfolio
defaults, split & gates), `agents/validation_and_ablation.md` for the validation
protocol and holdout discipline.
