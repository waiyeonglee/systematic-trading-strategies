# trend-following-strategy

Standalone implementation of a cross-asset weekly trend-following strategy
(DSA5205 Project 2). Universe: 16 tradable ETFs (9 US sectors + 7 diversifiers)
with SPY as benchmark, plus the CNN Fear & Greed index as an alternative feature.

## Dataset (fixed window)

- End date fixed at **2026-01-01** (fetch end exclusive **2026-01-02**).
- Start 2004-01-01 (pre-2011 history is feature warmup only); study window 2011+.
- See `data/README.md` for the full conventions.

### Macro inputs and the derived feature file

The macro feature block adds five monthly state variables on top of the
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
scripts/run_backtest.py      # single walk-forward backtest / test draw (grid or pinned spec)
scripts/run_trend_baseline.py  # rule-based baseline: SMA40 / MOM52 long/flat on val + test
                             # -> results/trend_baseline/
scripts/run_forecast_baseline.py  # AR(1) forecast baseline (prediction space only)
                             # -> results/forecast_baseline/
src/                         # labels.py, features.py, macro.py, models.py, splits.py,
                             # prediction_cache.py, portfolio.py, metrics.py,
                             # baselines.py, pipeline.py, plots.py, utils/
agents/                      # current_state.md (history log), feature_engineering.md,
                             # model_design.py, validation_and_ablation.md, macro_study.md, plots.md
data/                        # raw weekly OHLCV + fear-greed + derived features
results/                     # run tables per tag + _study_<block>/ summaries and the prediction
                             # cache (committed, ~22 MB npz); study draws (tables, series)
                             # and figures live under results/_study_<block>/draws|plots/<arm>/;
                             # results/trend_baseline/ holds the rule-path measurement;
                             # results/forecast_baseline/ holds the AR(1) OOS-R² tables
```

## Setup

```bash
uv sync          # create .venv and install pinned dependencies (Python 3.13)
```

## Run

```bash
python scripts/build_features.py                # weekly, fear-greed on, H=12
                                                # -> features.csv + features_macro.csv
python scripts/build_features.py --no-fear-greed
python scripts/build_features.py --horizon 4
python scripts/build_features.py --no-macro     # skip the 114-column macro panel

python scripts/run_backtest.py                  # single walk-forward run: select on the combined
                                                # 2019-2022 window (8y rolling train, 4y val),
                                                # 20-spec Elastic-Net grid, refit QE, 4w cadence,
                                                # vol target 10% / rf 3%; measures the test span
                                                # (default 2023-2025)
python scripts/run_backtest.py --model enet_a0.01_r1 --horizon 1 --rebalance weekly # single cell
python scripts/run_backtest.py --horizon 4 --no-vol-target
python scripts/run_backtest.py --test-start 2020-01-01 --test-end 2022-12-31 --window-val 3y
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

# final-holdout baseline only after selection lock: SMA40 / MOM52 long/flat, same book
# settings and windows as the study; writes results/trend_baseline/<rule>/{val,test}/
python scripts/run_trend_baseline.py
python scripts/run_trend_baseline.py --rule sma40

# AR(1) forecast baseline (no positions, no costs): OOS R² against zero and against
# the historical mean, refit QE on a rolling 8y window; H=4 matches the winner's horizon
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
  draws/<arm>/<run_tag>/      # arm = me | qe; tables, series/, plots/
  plots/<arm>/{test,val}/     # that draw's 10 figures per window
  arm_comparison.txt          # one row per arm + which arm wins on the test window
```

The backtest saves tables under `results/<run_tag>/` (validation ranking,
comparison vs SPY + EW-16, forecast metrics, coefficients, run config) and their
raw time series under `series/`. Report figures are generated by the run itself
(no separate plot script, no flag); a study draw publishes them to
`plots/<arm>/test/` **and** `plots/<arm>/val/` — 10 figures each: the
2023-2025 holdout draw and the validation window the winner was selected on. See
`agents/plots.md` for the figure list and rationale. As of the current reset, old
generated results have been removed and the 2023-2025 test window is reserved as an
unseen final holdout for the fresh experiment cycle; it should only be run after the
validation-selected configuration is locked.
See `agents/current_state.md` for the history log, `agents/feature_engineering.md`
for the features, `agents/model_design.py` for the design (grid, portfolio
defaults, split & gates), `agents/validation_and_ablation.md` for the validation
protocol and holdout discipline.
