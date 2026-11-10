# Macro-interaction feature block (elastic net only)

> **Purpose.** One of the two feature blocks of the joint study, and the answer to
> "is the model's failure to hold the right *level* a regime-awareness problem?" The
> firm features are interacted with five macro state variables, so a feature can
> change sign and magnitude with the cycle. It is a separate block on purpose: its own
> columns, its own prediction cache and its own output folder. Nothing here touches the
> micro block's ranking, cache or published figures.
>
> Both blocks are run by the same command, `scripts/run_study.py`, selected with
> `--features micro|macro` (default: both in turn).

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

- **Firm block (19):** the price/volume features (the feature panel with the two
  market-wide columns `fear_greed_index` and `spy_return_1bar` removed, since those
  become macro state instead) plus the two trend inputs (`trend_sma40_ratio`,
  `trend_mom52`) — 17 price/volume + 2 trend, asserted in `macro_panel.FIRM_COLUMNS`.
- **Macro block (6):** `const`, `fear_greed`, `tbl`, `tms`, `dfy`, `dfr`.
- **Interaction:** Kronecker product, column names `<firm>__x__<macro>`; the `const`
  block is the plain firm feature. Built by `macro.kron_features`.
- **Scaling:** `ModelSpec.build` puts a `StandardScaler` inside the pipeline, so the 114
  columns are z-scored per refit on training rows only (the % units of `tbl` / `dfy`
  do not dominate the penalty, and no scaler sees validation data).

## 4. Split and search space

| Layer | Window | Notes |
| --- | --- | --- |
| Train | **2011-01-01 .. 2018-12-31** | rolling **8y = 416 weekly bars** per refit |
| Validation | **2019-01-01 .. 2022-12-31** | the single ranking window; 4 calendar years |
| Test | **2023-01-01 .. 2025-12-31** | reserved final draw of the winner, run once after selection is locked |

| Dimension | Values | Count |
| --- | --- | --- |
| model | elastic net, alpha {1e-4, 1e-3, 1e-2, 1e-1} x `l1_ratio` {0, 0.25, 0.5, 0.75, 1} | **20** |
| combo | (rebalance, look-ahead) pairs under look-ahead >= cadence, refit **ME or QE** | **11** |
| book | vol window {4,8,12,24,48}w x target {20%,10%,off} x cap {1.0,2.0,3.0}, rf 3% | **45** |
| | **candidates** | **9,900** |

The refit cadence is restricted with `--refits` (default: all of `ME,QE`). The study
space holds monthly and quarterly refits only; weekly refits are not part of it.

`python scripts/run_study.py cache --features macro [--refits ...]` fits the **missing**
prediction sets of that space and nothing else (no book simulation, no ranking). Combos
that share a `(look-ahead, refit)` group collapse to one task, so the 11 combos are 6
forecast groups: `weekly_la{1,4,12}_refit{ME,QE}`. Run it after changing `--refits`, then
re-rank; `--models` restricts the specs and `--refresh-cache` refits everything.

After the validation-selected configuration is locked,
`python scripts/run_study.py draw --features macro [--refits ...]` can draw the best
validation candidate of every refit arm on 2023-2025 (one folder per arm, its own
figures) and write `draws/arm_comparison.txt`; `test --features macro` draws only the
overall winner.

No RFF family: the point of this study is fast iteration over a linear model with
explicit interactions. `l1_ratio=0` is ridge and `l1_ratio=1` is lasso, so the family
ends are covered.

Selection is the same single criterion as the micro block: the highest `net_sharpe` on
the validation window wins.

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
  comparison/                          # scripts/compare_arms.py: ME|QE + AR(1) tables/figures
```

Every published figure in `plots/<arm>/{test,val}/` draws the **AR(1) baseline** as a
grey dashed line next to the strategy and the two benchmarks: a per-ticker one-lag
regression of the forward return on the trailing return, fitted on the same split and
traded with the same book, costs and cadence. It is a no-feature reference, so a panel
shows at a glance whether the fitted features add anything over the trailing return.

- The cache provenance records seed, target, refit, rolling window, embargo,
  evaluation window, feature block and panel signature. Each block also gets its own
  cache root — `results/_study_micro/predictions` and
  `results/_study_macro/predictions` — so the two blocks can never read or overwrite
  each other's predictions.
- After selection is locked, `run_study.py test --features macro` draws the overall
  winner and publishes to `plots/<its arm>/{test,val}/` via
  `run_pipeline(..., plots_root=arm_plots_root(refit))`; `draw --features macro` does
  the same for every arm. The whole tree lives inside this block's folder, so the
  micro block's draws and figures are never touched.

## 6. Discipline notes on the test window

- The 2023-2025 window has been drawn for both arms (see §7): treat it as a
  descriptive out-of-sample check. Any further modelling decision must come from the
  validation window only.
- The enet alpha grid was calibrated on 21 features. With 114 collinear interaction
  columns the useful alpha range may sit below `1e-4`; check the number of non-zero
  coefficients per refit (degenerate candidates are flagged in the ranking) before
  interpreting a null result.

## 7. Current macro state

The macro block has been ranked and drawn (9,900 candidates):

- **Validation winner:** `enet_a0.01_r1`, absolute12 / weekly / QE, book
  `w48_vt20_rf3_cap1.0` → net Sharpe **1.1763** on 2019-2022.
- **Ties matter here.** Four books score exactly 1.1763 (the leverage cap cannot bind
  while the volatility overlay is off, so `cap1`/`2`/`3` are the same book). The
  ranking breaks ties deterministically (candidate order, then a stable sort), with the
  alphabetical book name deciding between otherwise equal candidates.
- **Test window (2023-2025), descriptive only:** the QE arm drew net Sharpe 1.3856
  (maxDD -7.97%) and the ME arm 0.8586; the AR(1) no-feature baseline drew 1.2640. The
  9,900-row ranking is in `validation_ranking.csv` and the per-arm figures under
  `plots/{me,qe}/{test,val}/`.
