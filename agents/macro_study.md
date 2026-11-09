# Macro-interaction study (elastic net only)

> **Purpose.** A second, separate study that asks whether the model's failure to hold
> the right *level* is a regime-awareness problem. The firm features are interacted
> with five macro state variables, so a feature can change sign and magnitude with the
> cycle. It is a separate study on purpose: different split, different feature set,
> its own prediction cache and its own output folder. Nothing here touches the v2
> study's ranking, cache or published figures.

## 1. Inputs

`tbl` / `tms` / `dfy` / `dfr` come from the Welch-Goyal predictor workbook
`Data2025.xlsx`, downloaded from Amit Goyal's predictor-data page
<https://sites.google.com/view/agoyal145> (source data for Goyal & Welch 2008,
*A Comprehensive Look at the Empirical Performance of Equity Premium Prediction*,
RFS 21(4), 1455–1508). `fear_greed` comes from the project's local CNN index CSV.

| Input | Source | Coverage |
| --- | --- | --- |
| `tbl` | `data/raw/Data2025.xlsx`, sheet `Monthly`, column `tbl` | 1920-01 .. 2025-12 |
| `tms` | same workbook, column `tms` (**= `lty - tbl`**, long govt yield - T-bill) | 1871-01 .. 2025-12 |
| `dfy` | same workbook, column `dfy` (**= `BAA - AAA`**) | 1871-01 .. 2025-12 |
| `dfr` | same workbook, column `dfr` (**= `corpr - ltr`**) | 1926-01 .. 2025-12 |
| `fear_greed` | `data/additional/fear-greed.csv`, month-end (last daily value) | 2011-01 .. 2025-12 |

Verified before the first run: the `Monthly` sheet reaches **2025-12**, every macro
column is non-null from 2011 on, and the workbook `ReadMe` defines `tms` / `dfy` /
`dfr` exactly as above (`dfr` has no FRED equivalent — that is why the workbook is the
source). Because the Fear & Greed series starts 2011-01, the interaction panel starts
**2011-02-04** (its first available lagged month), not 2011-01-07.

## 2. Timing rule (no lookahead)

Month `M`'s value is only published after `M` closes. A bar dated inside month `M`
therefore uses the month `M - 1` value:

```
published(month M) = month_end(M) + 1 day      # then shifted by (lag_months - 1) months
bar date d         = latest published stamp <= d
```

Verified on the built panel (`macro_state(..., lag_months=1)`):

| bar date | uses month | example `tbl` |
| --- | --- | --- |
| 2019-01-25 | 2018-12 | 0.0237 |
| 2019-02-01 .. 2019-02-22 | 2019-01 | 0.0237 |
| 2019-03-01 .. 2019-03-08 | 2019-02 | 0.0239 |

`tests/test_macro.py` locks this down (including the month-end boundary case and
`lag_months=0`).

## 3. Features — 19 x (5 macro + const) = 114

- **Firm block (19):** the 21-column v2 set minus the two market-wide columns
  `fear_greed_index` and `spy_return_1bar` — i.e. 17 price/volume features + the two
  trend inputs (`trend_sma40_ratio`, `trend_mom52`).
- **Macro block (6):** `const`, `fear_greed`, `tbl`, `tms`, `dfy`, `dfr`.
- **Interaction:** Kronecker product, column names `<firm>__x__<macro>`; the `const`
  block is the plain firm feature. Built by `macro.kron_features`.
- **Scaling:** `ModelSpec.build` puts a `StandardScaler` inside the pipeline, so the 114
  columns are z-scored per refit on training rows only (the % units of `tbl` / `dfy`
  do not dominate the penalty, and no scaler sees validation data).

## 4. Split and search space

| Layer | Window | Notes |
| --- | --- | --- |
| Train | **2011-01-01 .. 2018-12-31** | rolling **8y = 416 weekly bars** per refit (not 260) |
| Validation | **2019-01-01 .. 2022-12-31** | the single ranking window; 4 calendar years |
| Test | **2023-01-01 .. 2025-12-31** | reserved final draw of the winner, run once after selection is locked |

| Dimension | Values | Count |
| --- | --- | --- |
| model | elastic net, alpha {1e-4, 1e-3, 1e-2, 1e-1} x `l1_ratio` {0, 0.25, 0.5, 0.75, 1} | **20** |
| combo | (rebalance, look-ahead) pairs under look-ahead >= cadence, refit **ME or QE** | **11** |
| book | vol window {4,8,12,24,48}w x target {20%,10%,off} x cap {1.0,2.0,3.0}, rf 3% | **45** |
| | **candidates** | **9,900** |

The refit cadence is restricted with `--refits` (default: all of `ME,QE`). Weekly
refits are removed from the active study space; monthly and quarterly refits remain.

`python scripts/run_study_macro.py cache [--refits ...]` fits the **missing** prediction
sets of that space and nothing else (no book simulation, no ranking). Combos that share
a `(look-ahead, refit)` group collapse to one task, so the 11 combos are 6 forecast
groups: `weekly_la{1,4,12}_refit{ME,QE}`. Run it after changing `--refits`, then
re-rank; `--models` restricts the specs and `--refresh-cache` refits everything.

After the validation-selected configuration is locked,
`python scripts/run_study_macro.py draw [--refits ...]` can draw the best validation
candidate of every refit arm on 2023-2025 (one folder per arm, its own figures) and
write `draws/arm_comparison.txt`; `test` draws only the overall winner.

No RFF family: the point of this study is fast iteration over a linear model with
explicit interactions. `l1_ratio=0` is ridge and `l1_ratio=1` is lasso, so the family
ends are covered.

Selection is the same single criterion as `run_study.py`: the highest `net_sharpe` on
the validation window wins. (A regime-stratified second criterion ran on 2026-11-04
and was removed on 2026-11-06; see `agents/current_state.md` Step 19.)

## 5. Outputs and isolation

```
results/_study_macro/
  predictions/<start>_<end>/<freq>_la<h>_refit<slug>/<spec>.npz   # own cache, own panel signature
  validation_ranking.csv
  validation_winner.json               # includes split, window_train, feature block
  run_parameters.json
  draws/<arm>/<run_tag>/               # arm = me | qe: tables + series/
  plots/<arm>/{test,val}/              # 10 figures per window for that arm
  arm_comparison.txt                   # per-arm test draws + the winning arm
```

- The cache key records seed, target, refit, rolling window, embargo, evaluation
  window and panel signature — but **not the feature set**, so the study must never
  share `results/_study/predictions`; `run_study_macro.py` always writes to
  `results/_study_macro/predictions`.
- After selection is locked, `run_study_macro.py test` draws the overall winner and publishes to
  `plots/<its arm>/{test,val}/` via
  `run_pipeline(..., plots_root=arm_plots_root(refit))`; `draw` does the same for every
  arm. The whole tree lives inside this study's folder, so the v2 study's draws and
  figures are never touched.

## 6. Discipline notes before drawing the test window

- As of the fresh reset, old generated results have been removed and the 2023-2025
  window is treated as the final unseen holdout for the current macro experiment.
- Do not run or inspect 2023-2025 macro test outputs until the validation-selected
  macro configuration is locked.
- The enet alpha grid was calibrated on 21 features. With 114 collinear interaction
  columns the useful alpha range may sit below `1e-4`; check the number of non-zero
  coefficients per refit (degenerate candidates are flagged in the ranking) before
  interpreting a null result.

## 7. Current macro state

No macro ranking or 2023-2025 macro test draw has been recorded for the current reset
experiment. Any earlier macro result tables, test draws, plots, or arm comparisons are
stale unless regenerated after the reset.
