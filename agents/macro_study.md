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
| Test | **2023-01-01 .. 2025-12-31** | one draw of the winner, measured once |

| Dimension | Values | Count |
| --- | --- | --- |
| model | elastic net, alpha {1e-4, 1e-3, 1e-2, 1e-1} x `l1_ratio` {0, 0.25, 0.5, 0.75, 1} | **20** |
| combo | (rebalance, look-ahead) pairs under look-ahead >= cadence, refit **QE only** | **6** |
| book | vol window {4,8,12,24,48}w x target {20%,10%,off} x cap {1.0,2.0,3.0}, rf 3% | **45** |
| | **candidates** | **5,400** |

The refit cadence is restricted with `--refits` (default: all of `W-FRI,ME,QE`). The
weekly-refit arm is **208 fits per prediction set** versus 48 (`ME`) and 16 (`QE`), so
it costs ~85% of the prediction compute for 3 of the 14 combos; it is dropped here
because QE is the cadence that won every v2 draw and this study is meant to iterate
fast. Adding it back later is a `--refits W-FRI,ME,QE` re-rank — the completed
prediction sets stay cached.

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
results/weekly_roll8y_val4y_qrefit_<spec>_..._2023-2025_macro/     # test draw tables + series
results/plots_macro/test/01_cumulative_returns.png … 10_oos_r2_by_year.png
results/plots_macro/val/01_cumulative_returns.png  … 10_oos_r2_by_year.png
```

- The cache key records seed, target, refit, rolling window, embargo, evaluation
  window and panel signature — but **not the feature set**, so the study must never
  share `results/_study/predictions`; `run_study_macro.py` always writes to
  `results/_study_macro/predictions`.
- The test draw publishes to `results/plots_macro/{test,val}/` via
  `run_pipeline(..., plots_root=RESULTS_DIR / "plots_macro")`, a separate root from
  the v2 study's `results/plots/{test,val}/`, so both figure sets coexist and the
  pre-registered v2 figures are never overwritten.

## 6. Discipline notes before drawing the test window

- 2023-2025 has already produced three measurements (v2 §5.1 and the two criteria of
  §5.2). It is a **remeasurement window**, not a fresh holdout; a new hypothesis should
  be written down before the draw, per `validation_and_ablation.md` §1.2.
- The enet alpha grid was calibrated on 21 features. With 114 collinear interaction
  columns the useful alpha range may sit below `1e-4`; check the number of non-zero
  coefficients per refit (degenerate candidates are flagged in the ranking) before
  interpreting a null result.

## 7. First cycle — measured (2026-11-04, seed 5205)

**Ranking:** 5,400 candidates (20 enet specs x 6 QE combos x 45 books) on 2019-2022,
1,350 of them degenerate (25%; the `alpha=0.1` L1/enet arms collapse to the intercept,
plus part of the `1e-2` arm).

| row | model | target | cadence | book | net Sharpe | worst year |
| --- | --- | --- | --- | --- | --- | --- |
| winner | `enet_a0.01_r1` (lasso) | absolute12 | weekly / QE | `w48_vt20_rf3_cap1.0` | **1.1763** | +0.26 |

Winner detail: gross 1.1792, maxDD −14.04%, ann return 16.86%, ann vol 14.11%,
hit rate 0.598, turnover 0.025, 4/4 positive years, up/down capture 0.99 / 0.82.
Realized vol (14.1%) sits below the 20% target, so the vol overlay never binds — the
top four rows are that same book with `vt20` and `vtoff` at every cap.

**Forecast quality of the winner** (validation window, target `fwd_ret_12bar`):
OOS R² **+0.0507**, rank-IC **0.1514** (t 5.51), direction accuracy 0.632; by year
+0.047 / +0.083 / +0.164 / −0.112 (2019-2022).

**Control A — same search on the 21-feature v2 panel** (identical window, 8y rolling
window, 20 specs x 6 QE combos x 45 books): best net Sharpe **1.2026**
(`enet_a0.001_r0.5` / absolute1 / weekly / `w04_vtoff_rf3_cap1.0`), top-20 median
1.126, 95th pct 0.985, median 0.658. So the 114-column interaction block does **not**
beat the 21-column panel on this window (1.176 vs 1.203 — inside selection noise).

**Control B — matched cell** (`enet_a0.01_r1`, absolute12, weekly, QE, `w48_vt20`):

| features | net Sharpe | OOS R² | rank-IC | direction |
| --- | --- | --- | --- | --- |
| 21 firm features | 0.8406 | 0.0468 | 0.1478 | 0.631 |
| 114 macro interactions | 1.1763 | 0.0507 | 0.1514 | 0.632 |

The forecast statistics are unchanged; the book difference comes only from the
prediction level, not from better ordering. **Null result**: on 2019-2022 the five
macro state variables do not add measurable forecast information beyond the price-based
features, which is consistent with the earlier decomposition showing the v2 failure
lives in the level/common component rather than in the cross-section.

