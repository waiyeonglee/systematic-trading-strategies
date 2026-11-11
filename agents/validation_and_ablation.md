# Validation Study — joint model x strategy ranking

> **Purpose.** ONE validation stage. The walk-forward predictions for every
> (model spec, look-ahead, refit cadence) are computed once and cached as npz
> matrices; the study then replays them through every book cell and ranks the **whole
> combined space of model x strategy parameters** by net Sharpe on a single combined
> validation window (2019-01-01..2022-12-31). There is no separate model-selection
> phase and no separate book-ablation phase — everything competes in one ranking.
> The final test window (2023-2025) stays untouched by the ranking: it will be
> measured once, afterwards, for the winner only after selection is locked.
>
> Design constants live in `agents/model_design.py`; this file holds the protocol and
> holdout discipline. The seed is fixed everywhere through the constant
> `utils.config.DEFAULT_SEED` (`run_study.SEED`) and recorded in each cache entry and
> in the ranking/winner files — never spelled out by hand.

---

## 1. Data split & test-window discipline

### 1.1 The split (fixed, mechanical, pre-decided)

Data is used **only through the fixed dataset window — no 2026 data** (data end inclusive
2026-01-01; fetch bound exclusive 2026-01-02). The layers are calendar-anchored and
were fixed **before** looking at market regimes.

| Layer | Window | Role |
| --- | --- | --- |
| Warmup | 2004-01-02..2010-12-31 | feature / indicator warmup before training starts |
| Train | **rolling history before each refit** | rolling **8y (416 weeks)** minus the **embargo (= look-ahead: 1/4/12 bars)** at each refit; the window advances at every refit |
| **Validation (combined)** | **2019-01-01..2022-12-31** | the single ranking window for the whole model x strategy space |
| **Test (final)** | **2023-01-01..2025-12-31** | drawn once per block after selection was locked; measured to 2025-12-19 (the last bar of the rebuilt panel) |

### 1.2 Rules that keep the numbers honest

1. **One ranking, one window.** Every (model, cadence, look-ahead, refit, book) cell
   is scored on the same dates with the same metric (net Sharpe). No cell is selected
   on the test window.
2. **The test window selects nothing.** The 2023-2025 window was drawn once per block
   after the validation-selected configuration was locked, and is reported for the
   record only.
3. **Gate bars are recomputed on the test window.** SPY and EW-16 buy-and-hold stats
   on 2023-01-01..2025-12-31 are pre-registered before the draw
   (`pipeline.trend_success_verdict`: net Sharpe >= the weaker benchmark AND
   maxDD <= the better benchmark).
4. **Promotion rule.** The ranking winner is a *candidate*, not a headline. It is
   promoted only by (a) writing the hypothesis + expected mechanism before measuring,
   (b) measuring **once** on the final test window after selection is locked, and
   (c) recording the outcome in `agents/current_state.md`. A null result is a
   completed experiment.
5. **Determinism.** Every estimator is seeded with `utils.config.DEFAULT_SEED`
   (`utils.config.DEFAULT_SEED`, echoed as `run_study.SEED`). The cache key records the
   seed, so an entry written under another seed is recomputed, never silently reused.

## 2. Prediction cache

`src/prediction_cache.py` stores one prediction set per (spec, feature block,
look-ahead, refit) under
`results/_study_<block>/predictions/<window>/<freq>_<la>_refit<slug>/`:

| file | content |
| --- | --- |
| `<spec>.npz` | the prediction matrix as `float32` (dates, tickers) + the two axes (≈ 21 KB for six years of weekly bars) |
| `<spec>.json` | provenance: spec, feature block, target, refit, training window, embargo, **seed**, evaluation window, panel signature, format |
| `<window>/_degenerate/<group>__<spec>.json` | marker for a **degenerate** prediction set (see below) |

- Predictions do not depend on the rebalance cadence or on any book knob, so the
  weekly / 4w / 12w candidates of one (look-ahead, refit) share a single file.
- The 20 specs x 6 (look-ahead, refit) pairs give **120 prediction sets**; the full
  study reuses them, so the ranking costs only book simulation.
- **Degenerate sets are not stored.** If the predictions are constant across tickers
  on the window (mean cross-sectional std < 1e-12 — e.g. L1 models that collapse to
  the intercept), the cache writes a marker and removes the payload; later calls raise
  `DegenerateCandidate`, the study records the candidate as degenerate and skips all
  of its book cells without simulating anything. Warmup sets are exempt: a flat warmup
  is harmless (it only seeds the volatility estimate) and is always stored.
- **Precision:** `float32` storage is used for cached predictions to keep the cache
  compact while preserving ranking-scale numerical precision.
- Every book row of a candidate shares one **48-bar** volatility warmup, so no row
  starts from a different volatility history.

## 3. The ranked space

| dimension | values | count |
| --- | --- | --- |
| model spec | enet {1e-4 ... 1e-1} x l1_ratio {0, 0.25, 0.5, 0.75, 1}; endpoints cover Ridge-like (`0`) and Lasso-like (`1`) cases | **20** |
| cadence, look-ahead, refit | (rebalance, look-ahead) pairs with look-ahead >= cadence: weekly x {1,4,12}, 4w x {4,12}, 12w x {12}; refit {ME, QE} with refit_interval >= cadence | **11** |
| book | **full product** of vol window {4, 8, 12, 24, 48}w x vol target {20%, 10%, off} x leverage cap {1.0, 2.0, 3.0} (caps 2.0/3.0 pay a 6.8% ann. borrow fee), rf sleeve fixed at 3% | **45** |
| | **candidates** | **9,900** |

- Metric: **net Sharpe** on the validation window (zero risk-free rate,
  annualised x sqrt(52)).
- Book cells are the full cross-product — there is no shared "base" row — so every
  combination is ranked on equal footing. The 0% cash-sleeve arm is not tested: the
  sleeve earns a flat 3% in every cell because leaving cash at 0% is a strictly
  worse book for the same signal.
- The leverage cap can only bind while the vol-target overlay is live
  (`scale = min(cap, target / realised vol)`), so the three `off x cap 2.0/3.0` cells
  reproduce the cap-1.0 book apart from the borrow fee charged on drift leverage.
  Those cells are **simulated once** (40 distinct books per candidate) but all 45 are
  reported.
- Degenerate cells (constant / near-zero predictions on the window) are kept in the
  ranking for the record but can never win.
- **Cost:** the book simulation is a numpy state machine
  (`portfolio.build_forecast_vol_portfolio`), so one book takes 17–19 ms; the rolling
  forecast volatility is shared across the book cells that use the same vol window.
- The search is pre-declared: the grid, the combos and the book cells are fixed
  before the ranking is read.

### 3.1 Selection criterion

One criterion runs over the scored space: the row with the highest `net_sharpe` on
the validation window wins, and its files are `validation_ranking.csv` and
`validation_winner.json`. Each block's winner was then measured once on the test
window.

A regime-stratified second criterion (gate on profitable calendar years plus
up-capture, then maximin over the worst calendar year) was tried and dropped; only the
`net_sharpe` criterion is implemented.

## 4. Output format

`results/_study_<block>/validation_ranking.csv` — one row per candidate, sorted by
`net_sharpe` descending, with `rank`, the model parameters, `target` /
`look_ahead` / `rebalance` / `refit`, the book knobs (`book`, `vol_window`,
`vol_target`, `vol_rf`, `vol_cap`, `lev_fee`), the metrics (`net_sharpe`,
`gross_sharpe`, `max_drawdown`, `total_return`, `ann_return`, `ann_vol`, `hit_rate`,
`avg_turnover`), `degenerate` / `degenerate_reason` and the `features` block
(`micro` | `macro`). `book` is the cell slug (`w12_vt20_rf3_cap1.0`); its
human-readable note is kept once per name in `run_parameters.json` rather than repeated
on all 9,900 rows.

`results/_study_<block>/validation_winner.json` — the top row plus its provenance (spec
record, feature block, combo, book, metrics, cache path, seed, window, candidate
counts).

`results/_study_<block>/run_parameters.json` — the space definition actually searched.

When the final holdout is run, test draws are kept inside the study folder, one
directory per refit arm:

```
results/_study_{micro,macro}/
  draws/<arm>/<run_tag>/           # arm = me | qe: the usual tables + series/
  plots/<arm>/{test,val}/          # 10 PNGs per window for that arm
  arm_comparison.txt               # one row per arm + the arm with the highest
                                   # test-window net Sharpe
```

`arm_comparison.txt` is written by `draw` (and only by `draw`): the main table has
`arm`, `model`, `combo`, `book`, `val_sharpe` (from the ranking), `test_sharpe`,
`max_drawdown`, `total_return`, `ann_return`, `ann_vol`, `gate`; a second table gives
the **cumulative return** of the drawn book against both benchmarks on both windows
(`arm`, `val_strategy`, `val_ew`, `val_spy`, `test_strategy`, `test_ew`, `test_spy`,
read from the run's own `val_*`/`test_*` series so it matches `comparison.csv`). It
states in the file itself that picking the arm by test Sharpe would be snooping — the
arm is a validation-window choice.

Command: `python scripts/run_study.py rank` (8 worker processes by default; only the
book simulation runs — the model fits come from the cache). The active space is 9,900
candidates. After the validation-selected configuration is locked, `test` draws the
overall winner once on the 2023-2025 holdout; `draw` does the same for **each refit
arm's** best candidate and writes `draws/arm_comparison.txt`.

`python scripts/run_study.py cache` fits any **missing** prediction sets without
simulating a book — the way to complete a partial cache (e.g. after adding a model spec
or a combo) before re-ranking. It walks the same space as `rank`, collapses combos that
share a `(look-ahead, refit)` group to one task (the forecasts do not depend on the
rebalance cadence), reuses every set whose provenance matches, and prints one line per
newly stored or degenerate set plus a `stored / degenerate / cached` summary.
`--models`, `--combos` and `--refresh-cache` behave as in `rank`.

## 5. Current Experiment State

Both feature blocks are ranked and drawn. The active state is:

- **Selection was validation-only:** the ranking that chose each winner saw only
  2019-2022.
- **Test window measured once:** the 2023-2025 window was drawn after selection was
  locked and is reported for the record; it selected nothing.
- **Re-measurement:** `python scripts/run_study.py test` re-draws a block's locked
  winner from the cache — add `--features micro` or `--features macro`; the bare
  command runs both. Output is deterministic, so a re-run reproduces the same figures.

## 6. Status

- **Scheme in force:** one joint ranking over model x strategy parameters on the
  combined 2019-2022 validation window, run once per feature block (`micro` = 19 firm
  features, `macro` = the 114 interaction columns); predictions cached under
  `results/_study_<block>/predictions`; `utils.config.DEFAULT_SEED`; one selection
  criterion (§3.1) over the scored space.
- **Promotion state:** each block has one measured 2023-2025 draw of its locked
  winner. Any further modelling decision must come from the validation window.
- **Holdout discipline:** do not re-select on the test window. Treat the existing
  draws as descriptive out-of-sample checks.
