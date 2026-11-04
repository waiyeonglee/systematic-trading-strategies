# trend-following-strategy-v2

Standalone re-implementation of a cross-asset weekly trend-following strategy
(DSA5205 Project 2). Universe: 16 tradable ETFs (9 US sectors + 7 diversifiers)
with SPY as benchmark, plus the CNN Fear & Greed index as an alternative feature.

This project is built step by step and deliberately carries **no stale imports,
derived data, or old results** from the reference project
(`../trend-following-strategy`).

## Dataset (fixed window)

- End date fixed at **2026-01-01** (fetch end exclusive **2026-01-02**).
- Start 2004-01-01 (pre-2011 history is feature warmup only); study window 2011+.
- See `data/README.md` for the full conventions.

### Macro inputs and the derived feature file

The macro-interaction study adds five monthly state variables on top of the
price-based features:

| Variable | Meaning | Source |
| --- | --- | --- |
| `fear_greed` | CNN Fear & Greed index (0-100), month-end | `data/additional/fear-greed.csv` (committed) |
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
rebuilds the panel in memory, so a fresh clone trains without it. Regenerate it with:

```bash
python scripts/build_features_macro.py                       # -> data/features/weekly/features_macro.csv
python scripts/build_features_macro.py --out /tmp/macro.csv  # custom destination
```

## Layout

```
scripts/fetch_data.py        # optional re-fetch from Yahoo Finance (end exclusive 2026-01-02)
scripts/build_features.py    # builds data/features/weekly/ from the raw CSVs
scripts/run_study.py         # joint model x strategy ranking on the combined window + test draw
scripts/build_features_macro.py  # writes the 114-column macro-interaction panel (git-ignored)
scripts/run_study_macro.py   # macro-interaction study: enet-only ranking, train 2011-2018,
                             # val 2019-2022, test 2023-2025 (results/_study_macro)
scripts/run_backtest.py      # single walk-forward backtest / test draw (grid or pinned spec)
src/                         # labels.py, features.py, macro.py, models.py, splits.py,
                             # prediction_cache.py, portfolio.py, metrics.py, pipeline.py,
                             # plots.py, utils/
agents/                      # current_state.md (history log), feature_engineering.md,
                             # model_design.py, validation_and_ablation.md, macro_study.md, plots.md
data/                        # raw weekly OHLCV + fear-greed + derived features
results/                     # run tables per tag + _study/ summaries and the prediction
                             # cache (committed, ~22 MB npz); figures land in results/plots/{test,val}/
```

## Setup

```bash
uv sync          # create .venv and install pinned dependencies (Python 3.13)
```

## Run

```bash
python scripts/build_features.py                # weekly, fear-greed on, H=12
python scripts/build_features.py --no-fear-greed
python scripts/build_features.py --horizon 4

python scripts/run_backtest.py                  # single walk-forward run: select on the combined
                                                # 2017-2022 window (5y rolling train, 6y val),
                                                # 93-spec grid, refit QE, 4w cadence,
                                                # vol target 10% / rf 3%; measures the test span
                                                # (default 2023-2025)
python scripts/run_backtest.py --model lasso_a0.01 --horizon 1 --rebalance weekly   # single cell
python scripts/run_backtest.py --horizon 4 --no-vol-target
python scripts/run_backtest.py --test-start 2020-01-01 --test-end 2022-12-31 --window-val 3y
# design rules (enforced in default_config):
#   look-ahead must be >= rebalance frequency: weekly(1w) x {1,4,12} · 4w x {4,12} · 12w x {12}
#   refit must be no more frequent than the cadence (refit_interval >= prediction_interval):
#   W-FRI x weekly pairs (3) · ME x {1w,4w cadences} (5) · QE x all pairs (6) -> 14 combos

python scripts/run_study.py rank                # ONE joint ranking over the combined
                                                # space: 93 specs x 14 (rebalance,
                                                # look-ahead, refit) combos x 45 book
                                                # cells = 58,590 candidates on
                                                # 2017-2022, best net Sharpe wins
                                                # -> results/_study/validation_ranking.csv
                                                # (~3-4 min on 8 workers; the model fits
                                                # come from the prediction cache)
python scripts/run_study.py test                # single pre-registered draw (2023-2025)
                                                # of the ranking winner
python scripts/run_study.py rank --criterion regime   # same space, regime-stratified
                                                # rule: profitable in >=5/6 years and
                                                # up-capture >=0.40, then the best
                                                # worst-year Sharpe
                                                # -> validation_ranking_regime.csv
python scripts/run_study.py test --criterion regime   # draw of that winner; figures
                                                # in results/plots/regime_{test,val}/
python scripts/run_study.py rank --jobs 4       # worker processes (0 = min(8, cpu))

python scripts/build_features_macro.py          # 19 firm x (5 macro + const) = 114 columns
                                                # -> data/features/weekly/features_macro.csv
                                                # (git-ignored, inspection only)
python scripts/run_study_macro.py rank          # macro-interaction study: 20 enet specs x
                                                # 6 QE combos x 45 books = 5,400 candidates
                                                # on 2019-2022 (train 2011-2018, 8y rolling)
                                                # -> results/_study_macro/validation_ranking.csv
python scripts/run_study_macro.py rank --refits W-FRI,ME,QE   # add the costlier refit arms
python scripts/run_study_macro.py test          # single draw of that winner on 2023-2025;
                                                # figures -> results/plots_macro/{test,val}/
```

Study outputs: the walk-forward predictions are cached under
`results/_study/predictions/<window>/<freq>_<la>_refit<slug>/` as
`<spec>.npz` (float32 matrix + axes) with a `<spec>.json` provenance sidecar
(seed `utils.config.DEFAULT_SEED`); degenerate prediction sets — constant across
tickers — are not stored,
they get a marker under `<window>/_degenerate/` and are skipped by the ranking. The
joint ranking lands in `results/_study/validation_ranking.csv` (one row per
candidate, sorted by net Sharpe) with the winner plus its provenance in
`results/_study/validation_winner.json` and the searched space in
`results/_study/run_parameters.json`. Only the book simulation runs — the model fits
come from the cache — and a cache entry written under another seed is recomputed.

The backtest saves tables under `results/<run_tag>/` (validation ranking,
comparison vs SPY + EW-16, forecast metrics, coefficients, run config) and their
raw time series under `series/`. Report figures are generated by the run itself
(no separate plot script, no flag) and land in `results/plots/test/` **and**
`results/plots/val/` — 10 figures each: the single pre-registered test draw and the
validation window the winner was selected on. See `agents/plots.md` for the figure
list and rationale. The 2023–2025 test window has now been measured twice (the retired
two-window protocol and the 45-cell joint ranking), so any further run there is a
remeasurement, not an untouched holdout. See `agents/validation_and_ablation.md`
§5/§6.
See `agents/current_state.md` for the history log, `agents/feature_engineering.md`
for the features, `agents/model_design.py` for the design (grid, portfolio
defaults, split & gates), `agents/validation_and_ablation.md` for the validation
protocol and the measured record.
