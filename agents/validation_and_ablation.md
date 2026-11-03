# Validation Study — joint model x strategy ranking (trend-following-strategy-v2)

> **Purpose.** ONE validation stage. The walk-forward predictions for every
> (model spec, look-ahead, refit cadence) are computed once and cached as npz
> matrices; the study then replays them through every book cell and ranks the **whole
> combined space of model x strategy parameters** by net Sharpe on a single combined
> validation window (2017-01-01..2022-12-31). There is no separate model-selection
> phase and no separate book-ablation phase — everything competes in one ranking.
> The final test window (2023-2025) stays untouched by the ranking: it is measured
> once, afterwards, for the winner only.
>
> Design constants live in `agents/model_design.py`; this file holds the protocol and
> the measured record. The seed is fixed everywhere through the constant
> `utils.config.DEFAULT_SEED` (`run_study.SEED`) and recorded in each cache entry and
> in the ranking/winner files — never spelled out by hand.

---

## 1. Data split & test-window discipline

### 1.1 The split (fixed, mechanical, pre-decided)

Data is used **only through the v2 fixed window — no 2026 data** (data end inclusive
2026-01-01; fetch bound exclusive 2026-01-02). The layers are calendar-anchored and
were fixed **before** looking at market regimes.

| Layer | Window | Role |
| --- | --- | --- |
| Warmup | 2004-01-02..2010-12-31 | feature / indicator warmup before training starts |
| Train | **2011-2016** | rolling **5y (260 weeks)** minus the **embargo (= look-ahead: 1/4/12 bars)** at each refit; the window advances at every refit |
| **Validation (combined)** | **2017-01-01..2022-12-31** | the single ranking window for the whole model x strategy space |
| **Test (final)** | **2023-01-01..2025-12-31** | one draw of the ranking winner; effective end 2025-09-26 under a 12-week hold |

### 1.2 Rules that keep the numbers honest

1. **One ranking, one window.** Every (model, cadence, look-ahead, refit, book) cell
   is scored on the same dates with the same metric (net Sharpe). No cell is selected
   on the test window.
2. **The test window selects nothing.** It measures the winner once. Any rerun is a
   remeasurement, not a fresh holdout.
3. **Gate bars are recomputed on the test window.** SPY and EW-16 buy-and-hold stats
   on 2023-01-01..2025-12-31 are pre-registered before the draw
   (`pipeline.trend_success_verdict`: net Sharpe >= the weaker benchmark AND
   maxDD <= the better benchmark).
4. **Promotion rule.** The ranking winner is a *candidate*, not a headline. It is
   promoted only by (a) writing the hypothesis + expected mechanism before measuring,
   (b) measuring **once** on the final test window, and (c) recording the outcome in
   `agents/current_state.md`. A null result is a completed experiment.
5. **Determinism.** Every estimator is seeded with `utils.config.DEFAULT_SEED`
   (`utils.config.DEFAULT_SEED`, echoed as `run_study.SEED`). The cache key records the
   seed, so an entry written under another seed is recomputed, never silently reused.

## 2. Prediction cache

`src/prediction_cache.py` stores one prediction set per (spec, look-ahead, refit) under
`results/_study/predictions/<window>/<freq>_<la>_refit<slug>/`:

| file | content |
| --- | --- |
| `<spec>.npz` | the prediction matrix as `float32` `(dates, tickers)` + the two axes (≈ 21 KB for six years of weekly bars) |
| `<spec>.json` | provenance: spec, target, refit, training window, embargo, **seed**, evaluation window, panel signature, format |
| `<window>/_degenerate/<group>__<spec>.json` | marker for a **degenerate** prediction set (see below) |

- Predictions do not depend on the rebalance cadence or on any book knob, so the
  weekly / 4w / 12w candidates of one (look-ahead, refit) share a single file.
- The 93 specs x 9 (look-ahead, refit) pairs give **837 prediction sets**; the full
  study reuses them, so the ranking costs only book simulation.
- **Degenerate sets are not stored.** If the predictions are constant across tickers
  on the window (mean cross-sectional std < 1e-12 — e.g. L1 models that collapse to
  the intercept), the cache writes a marker and removes the payload; later calls raise
  `DegenerateCandidate`, the study records the candidate as degenerate and skips all
  of its book cells without simulating anything. Warmup sets are exempt: a flat warmup
  is harmless (it only seeds the volatility estimate) and is always stored.
- **Precision:** `float32` storage moves the book Sharpe by < 1e-8 versus the
  `float64` simulations (measured 9.4e-9 on the winner); the ranking and the test draw
  are unchanged at 4 decimal places.
- **Size (2026-11-04):** 175.1 MB of long CSVs → **21.8 MB** of npz + JSON (8×
  smaller, ≈1,592 sets). Stored as compressed binary, so a git blob is ≈ the same
  size, against ≈58 MB if the old CSVs were committed. The cache is **committed on
  purpose** (it is not git-ignored) so a fresh clone never has to retrain; the
  pre-migration zip backup was deleted once the migration was verified against it.
- Every book row of a candidate shares one **48-bar** volatility warmup, so no row
  starts from a different volatility history.

## 3. The ranked space

| dimension | values | count |
| --- | --- | --- |
| model spec | ridge {1e-3, 1e-2, 1e-1, 1, 10} (5); lasso {1e-4, 1e-3, 1e-2, 1e-1} (4); enet {1e-4 … 1e-1} x l1_ratio {0.5, 0.25, 0.75} (12); rff {1e-4 … 10} x D {100, 200, 500} x gamma {0.01, 0.1, 1, 10} (72) | **93** |
| cadence, look-ahead, refit | (rebalance, look-ahead) pairs with look-ahead >= cadence: weekly x {1,4,12}, 4w x {4,12}, 12w x {12}; refit {W-FRI, ME, QE} with refit_interval >= cadence | **14** |
| book | **full product** of vol window {4, 8, 12, 24, 48}w x vol target {20%, 10%, off} x leverage cap {1.0, 2.0, 3.0} (caps 2.0/3.0 pay a 6.8% ann. borrow fee), rf sleeve fixed at 3% | **45** |
| | **candidates** | **58,590** |

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
  (`portfolio.build_forecast_vol_portfolio`), so one book takes 17–19 ms and the
  58,590-candidate ranking runs in **~5 min** on 8 processes; the rolling forecast
  volatility is shared across the book cells that use the same vol window.
- The search is pre-declared: the grid, the combos and the book cells are fixed
  before the ranking is read.

### 3.1 Selection criteria

Two criteria run over the same scored space; each writes its own files, so both
records coexist.

| criterion | rule (`run_study.py --criterion`) | files |
| --- | --- | --- |
| `raw` (default) | highest net Sharpe | `validation_ranking.csv`, `validation_winner.json` |
| `regime` | **gate** profitable in at least **5 of the 6** validation calendar years (2017-2022, i.e. a 5/6 rate so the rule generalises) *and* **up-capture >= 0.40** (keeps at least 40% of the equal-weight benchmark's up-week return); **then** rank the survivors by **worst calendar-year Sharpe** (maximin), tie-break net Sharpe | `validation_ranking_regime.csv`, `validation_winner_regime.json` |

- Why the gate: a single aggregate Sharpe cannot tell an edge that exists every year
  from one that was manufactured by the drawdown-rich years (2018/2020/2022) inside
  the window. Consistency across years plus upside participation is the cheapest
  observable test of that.
- Why **not** a down-capture gate: requiring extra downside protection rewards the
  defensive cells that produced the failed selection in the first place. Down-capture
  is recorded (`down_capture`) but not gated.
- Every row carries the regime metrics (`pos_years`, `n_years`, `yr_sharpe_min`,
  `up_capture`, `down_capture`, `regime_pass`), so both criteria are auditable from
  one CSV.
- The gate is a filter, not a fix: in the 2026-11-04 run the raw winner *passed* it
  (6/6 years, worst year +0.27, up-capture 0.60), so the gate alone is non-binding —
  the maximin objective is what moves the pick. See §5.2.

## 4. Output format

`results/_study/validation_ranking.csv` — one row per candidate, sorted by
`net_sharpe` descending, with `rank`, the model parameters, `target` /
`look_ahead` / `rebalance` / `refit`, the book knobs (`book`, `vol_window`,
`vol_target`, `vol_rf`, `vol_cap`, `lev_fee`), the metrics (`net_sharpe`,
`gross_sharpe`, `max_drawdown`, `total_return`, `ann_return`, `ann_vol`, `hit_rate`,
`avg_turnover`), the regime metrics (`pos_years`, `n_years`, `yr_sharpe_min`,
`up_capture`, `down_capture`, `regime_pass`) and `degenerate` /
`degenerate_reason`. `book` is the cell slug (`w12_vt20_rf3_cap1.0`); its
human-readable note is kept once per name in `run_parameters.json` rather than
repeated on 58,590 rows.

`results/_study/validation_ranking_regime.csv`, `…/validation_winner_regime.json`
and `…/run_parameters_regime.json` — the same space scored once more under the
`regime` criterion. Identical numbers per row; only the winner/selection differ, so
the `raw` files stay untouched for comparison.

`results/_study/validation_winner.json` — the top row plus its provenance (spec
record, combo, book, metrics, regime block, cache path, seed, window, candidate
counts, and the rule when one was applied).

`results/_study/run_parameters.json` — the space definition actually searched.

Command: `python scripts/run_study.py rank` (8 worker processes by default; only the
book simulation runs — the model fits come from the cache). Measured: ≈3 min for the
full 58,590-candidate space on 2026-11-04 (≈4.5 min with the regime metrics). Add
`--criterion regime` for the §3.1 rule, and `test --criterion regime` to draw that
winner; its figures land in `results/plots/regime_{test,val}/` so the `raw` figure
set survives.

## 5. Measured record

Filled after each executed ranking. The current record is §5.1.

### 5.1 Executed — joint ranking, 45-cell book space (2026-11-04)

Command `python scripts/run_study.py rank` → **≈3 min** for the full space (8 worker
processes; 17–19 ms per book simulation, 40 distinct simulations per candidate).

- **Rows:** 58,590 candidates, **2,655 degenerate** (59 degenerate spec x combo pairs
  x 45 book cells, constant predictions on the window).
- **Winner (rank 1):** `rff_a0.1_d100_g0.01` · absolute4 · 4w rebalance · QE refit ·
  book `w12_vt20_rf3_cap1.0` (12w window, 20% target, cap 1.0) — net Sharpe **1.3361**,
  maxDD −10.15%, total +121.8%, ann return 14.15%, ann vol 10.31%, hit rate 61.7%,
  avg turnover 0.246.
- **Runners-up:** each of the next four rows is the *same* model and combo with a
  different book cell, i.e. the top of the ranking is one candidate x its book space:

| rank | model | combo | book | net Sharpe | maxDD |
| --- | --- | --- | --- | --- | --- |
| 2 | rff_a0.1_d100_g0.01 | abs4 · 4w · QE | w12_vt10_rf3_cap1.0 | 1.3239 | −9.82% |
| 3 | rff_a0.1_d100_g0.01 | abs4 · 4w · QE | w12_vtoff_rf3_cap1.0 | 1.3117 | −10.15% |
| 4 | rff_a0.1_d100_g0.01 | abs4 · 4w · QE | w12_vtoff_rf3_cap2.0 | 1.3090 | −10.16% |
| 5 | rff_a0.1_d100_g0.01 | abs4 · 4w · QE | w12_vtoff_rf3_cap3.0 | 1.3090 | −10.16% |
| 6 | rff_a0.01_d100_g0.01 | abs4 · 4w · QE | w04_vtoff_rf3_cap1.0 | 1.2906 | −11.11% |
| 7 | rff_a0.01_d100_g0.01 | abs4 · 4w · QE | w04_vt20_rf3_cap1.0 | 1.2893 | −11.03% |
| 8 | rff_a0.01_d100_g0.01 | abs4 · 4w · QE | w04_vtoff_rf3_cap2.0 | 1.2881 | −11.11% |

- **What each knob is worth (best cell per level):** vol target 20% 1.3361 > 10%
  1.3239 > off 1.3117; vol window 12w 1.3361 > 4w 1.2906 > 8w 1.2242 > 24w 1.2115 >
  48w 1.2022; leverage cap 1.0 1.3361 > 2.0 = 3.0 1.3090. Borrowing never wins, and the
  two levered caps are identical whenever the overlay is off (the cap cannot bind).
- The winner is again a single cell inside a tight rff / D = 100 / gamma = 0.01
  cluster, not a broad plateau: only **five rows** sit within 0.03 Sharpe of the top
  and **nine** within 0.05, and all of them are `4w · QE` with a 12w or 4w window and
  cap 1.0 (or the inert-cap duplicate). Nothing with a weekly / 12w cadence or a
  levered cap comes near.

**Test draw — 2023-01-01..2025-12-31, one measurement of rank 1.**

| metric (2023–2025) | book | EW-16 | SPY B&H |
| --- | --- | --- | --- |
| ann return (net) | 0.97% | 12.55% | 23.54% |
| ann vol | 8.16% | 9.33% | 14.61% |
| total return (3y) | +2.69% | +38.42% | +78.86% |
| max drawdown (net) | −10.42% | **−7.79%** | −16.88% |
| Sharpe (net) | 0.1595 | **1.3145** | 1.5230 |

**Gate verdict: FAIL — SUCCESS=False** on both conditions (Sharpe 0.1595 < 1.3145 bar;
maxDD −10.42% < −7.79% bar). Forecast OOS R² −0.0387 (rank-IC 0.0973); pooled OOS R²
vs SPY −0.150, vs EW-16 −0.618. Null result: the validation edge does not transfer,
nothing is promoted, `HEADLINE_*` stays `None`. Folder
`results/weekly_fg_roll5y_val6y_qrefit_rff_a0.1_d100_g0.01_vt0.2c1rf3_2023-2025`;
figures in `results/plots/test/` (2023-2025) and `results/plots/val/` (2017-2022,
the selection window, reported with the winning book).

**Superseded record (kept for continuity).** The 11-row one-knob-at-a-time book space
(2026-11-03) picked the same model and combo with `volwin_12w` (10% target) at 1.3239
and drew 0.2906 net Sharpe on the test window — the same null conclusion. Its folder
and CSV were replaced by the 45-cell run above; the numbers survive in
`agents/current_state.md` Steps 13-15.

### 5.2 Executed — regime-stratified selection (§3.1), same space (2026-11-04)

Command `python scripts/run_study.py rank --criterion regime` → **≈4.6 min**, then
`test --criterion regime` (≈6 s). The `raw` files from §5.1 are untouched.

- **Gate pass rate:** **7,181 of 55,935** live rows (12.8%). Sensitivity ladder:
  283 rows are profitable in 6/6 years, 8,409 in ≥5/6, 41,733 in ≥4/6.
- **The gate does not bind on the raw winner:** `rff_a0.1_d100_g0.01` ·
  `w12_vt20_rf3_cap1.0` is profitable in **6/6** years (worst year +0.27) with
  up-capture 0.60 → it passes. Filtering alone therefore changes nothing; the
  **maximin** objective is what moves the pick, to a lower-Sharpe but flatter
  candidate.
- **Regime winner:** `rff_a0.01_d100_g0.01` · absolute4 · 4w · QE · book
  `w04_vtoff_rf3_cap1.0` (4w window, overlay off) — validation net Sharpe **1.2906**
  (vs 1.3361), **worst calendar year +0.41** (vs +0.27), up-capture 0.606,
  down-capture 0.296, maxDD −11.11%, turnover 0.282.

| | val Sharpe | val worst yr | val up/down capture | test Sharpe | test maxDD | test total |
| --- | --- | --- | --- | --- | --- | --- |
| raw (max Sharpe) | 1.337 | +0.27 | 0.61 / 0.30 | **0.160** | −10.42% | +2.69% |
| regime (maximin) | 1.291 | **+0.41** | 0.61 / 0.30 | **−0.075** | −12.77% | −2.66% |

- **Test draw of the regime winner (2023-2025):** net Sharpe **−0.0750**, maxDD
  −12.77%, total −2.66%, ann −0.98%, OOS R² −0.0509, rank-IC 0.0919 (t 3.42);
  by year 2023 −0.153 / 2024 +0.009 / 2025 +0.005. Gate: **FAIL** again
  (Sharpe −0.075 < 1.3145 bar, maxDD −12.77% < −7.79% bar).
- **Why it does not help:** the two picks are behaviourally the same book — their test
  returns correlate **0.96** (same sign in 89% of weeks). The regime rule changed the
  ridge strength (alpha 0.1 → 0.01) and the vol cell (12w/20% → 4w/off) but neither
  the signal nor the outcome. Both books had 1 profitable test year out of 3.
- **Conclusion:** the §5.1 failure is **signal decay**, not year-concentration in the
  selection. Regime-stratifying the criterion is still the right default (it costs one
  column of metrics and rejects the 88% of cells whose Sharpe rests on 4 years or
  less), but it cannot rescue a factor whose out-of-sample IC is ~0.09 with negative
  R². Folder
  `results/weekly_fg_roll5y_val6y_qrefit_rff_a0.01_d100_g0.01_2023-2025`; figures in
  `results/plots/regime_test/` and `results/plots/regime_val/`.
- **Holdout status:** 2023-2025 has now produced **three** measurements (the retired
  two-window protocol, §5.1, and this). It is a remeasurement window, not a holdout.

## 6. Status

- **Scheme in force:** one joint ranking over model x strategy parameters on the
  combined 2017-2022 window; predictions cached under `results/_study/predictions`;
  `utils.config.DEFAULT_SEED`; two selection criteria (§3.1) over the same scored
  space; only the winner of a criterion is measured once on 2023-2025.
- **Promotion state:** closed on 2026-11-04 — both criteria fail the gate: max-Sharpe
  0.1595 and regime-stratified −0.0750 net Sharpe against the EW-16 bar 1.3145, with
  maxDD −10.42% / −12.77% against the −7.79% bar → **null result**, nothing promoted,
  `HEADLINE_*` stays `None`.
- **Open question:** the top of the ranking sits inside a tight rff / D = 100 /
  gamma = 0.01 cluster whose validation edge does not survive out of sample (test OOS
  R² −0.039 / −0.051), and the 45 book cells rank almost randomly inside the window
  (Sharpe correlation between its two halves: 0.27 Pearson / 0.14 Spearman; against
  the test window: 0.05). Any follow-up is a new pre-registered cycle on new data, not
  a tweak of this one.
- **History:** earlier two-window protocols and their outputs were removed on
  2026-11-03 at the user's request; the summary numbers survive only in
  `agents/current_state.md` (Steps 1-17).
