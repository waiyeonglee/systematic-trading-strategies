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
- The 20 specs x 6 (look-ahead, refit) pairs give **120 prediction sets**; the full
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
  (`portfolio.build_forecast_vol_portfolio`), so one book takes 17–19 ms and the
  9,900-candidate ranking runs faster than the old mixed-family/RFF/weekly-refit spaces; the rolling forecast
  volatility is shared across the book cells that use the same vol window.
- The search is pre-declared: the grid, the combos and the book cells are fixed
  before the ranking is read.

### 3.1 Selection criterion

One criterion runs over the scored space: the row with the highest `net_sharpe` on
the validation window wins, and its files are `validation_ranking.csv` and
`validation_winner.json`. The winner is measured once on the test window (§5.1).

A regime-stratified second criterion (gate on profitable calendar years plus
up-capture, then maximin over the worst calendar year) ran beside it on 2026-11-04
and was removed on 2026-11-06 — see the history note in §6.

## 4. Output format

`results/_study/validation_ranking.csv` — one row per candidate, sorted by
`net_sharpe` descending, with `rank`, the model parameters, `target` /
`look_ahead` / `rebalance` / `refit`, the book knobs (`book`, `vol_window`,
`vol_target`, `vol_rf`, `vol_cap`, `lev_fee`), the metrics (`net_sharpe`,
`gross_sharpe`, `max_drawdown`, `total_return`, `ann_return`, `ann_vol`, `hit_rate`,
`avg_turnover`) and `degenerate` / `degenerate_reason`. `book` is the cell slug
(`w12_vt20_rf3_cap1.0`); its human-readable note is kept once per name in
`run_parameters.json` rather than repeated on 9,900 rows.

`results/_study/validation_winner.json` — the top row plus its provenance (spec
record, combo, book, metrics, cache path, seed, window, candidate counts).

`results/_study/run_parameters.json` — the space definition actually searched.

Test draws are kept inside the study folder, one directory per refit arm:

```
results/_study/
  draws/<arm>/<run_tag>/           # arm = w-fri | me | qe: the usual tables + series/
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
book simulation runs — the model fits come from the cache). The active space is now
9,900 candidates after removing separate Ridge/Lasso families, RFF, and weekly refits. `test` draws the overall winner once on the
test window; `draw` does the same for **each refit arm's** best candidate and writes
`draws/arm_comparison.txt`.

`python scripts/run_study.py cache` fits any **missing** prediction sets without
simulating a book — the way to complete a partial cache (e.g. after adding a model spec
or a combo) before re-ranking. It walks the same space as `rank`, collapses combos that
share a `(look-ahead, refit)` group to one task (the forecasts do not depend on the
rebalance cadence), reuses every set whose provenance matches, and prints one line per
newly stored or degenerate set plus a `stored / degenerate / cached` summary.
`--models`, `--combos` and `--refresh-cache` behave as in `rank`.

## 5. Measured record

Filled after each executed ranking. The current record is §5.1; the fixed rule-based
baseline is §5.1c; the forecast-space AR(1) baseline is §5.1d.

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
`results/_study/draws/qe/weekly_fg_roll5y_val6y_qrefit_rff_a0.1_d100_g0.01_vt0.2c1rf3_2023-2025`;
figures in `results/_study/plots/qe/test/` (2023-2025) and
`results/_study/plots/qe/val/` (2017-2022, the selection window, reported with the
winning book).

### 5.1b Refit-arm comparison, one draw per arm (2026-11-06)

`python scripts/run_study.py draw` draws the best validation candidate of each refit
arm on the same 2023-2025 window (folders under `results/_study/draws/<arm>/`, figures
under `results/_study/plots/<arm>/{test,val}/`, summary in
`results/_study/arm_comparison.txt`):

| arm | model / combo | val Sharpe | test Sharpe | maxDD | total | gate |
| --- | --- | --- | --- | --- | --- | --- |
| ME | `rff_a0.1_d100_g0.01` · abs4 · 4w | 1.1847 | **0.2040** | −9.87% | +3.70% | fail |
| QE | `rff_a0.1_d100_g0.01` · abs4 · 4w | 1.3361 | 0.1595 | −10.42% | +2.69% | fail |
| W-FRI | `rff_a0.1_d500_g0.01` · abs4 · weekly | 1.0450 | −0.3425 | −15.49% | −8.48% | fail |

All three arms fail the gate and none transfers a validation edge to 2023-2025; the
weekly refit arm is the worst. Selecting the arm by test Sharpe would be snooping —
the arm is a validation-window choice, so this table is descriptive only.

**Superseded record (kept for continuity).** The 11-row one-knob-at-a-time book space
(2026-11-03) picked the same model and combo with `volwin_12w` (10% target) at 1.3239
and drew 0.2906 net Sharpe on the test window — the same null conclusion. Its folder
and CSV were replaced by the 45-cell run above; the numbers survive in
`agents/current_state.md` Steps 13-15.

### 5.1c Rule-based trend baseline — SMA40 / MOM52 (2026-11-09)

Not a ranking candidate: a fixed, pre-declared reference book (`pred = +1` when the
trend feature is > 0, else flat; inverse-vol weights, 4w cadence, moomoo, no cost gate,
no vol-target overlay). It exists to answer "does the naive rule do as well as the
selected model?" out of sample. Command: `python scripts/run_trend_baseline.py`;
tables under `results/trend_baseline/<rule>/{val,test}/`.

| book | val Sharpe | test Sharpe | test ann return | test maxDD | test total |
| --- | --- | --- | --- | --- | --- |
| `sma40` long/flat | 0.667 | 0.922 | 6.97% | −6.54% | +20.35% |
| `mom52` long/flat | 0.425 | 1.266 | 10.18% | −7.52% | +30.56% |
| ML joint winner (QE, w12/vt20) | 1.336 | 0.160 | 0.97% | −10.42% | +2.69% |
| EW-16 B&H | 0.699 | 1.314 | 12.55% | −7.79% | +38.42% |
| SPY B&H | 0.657 | 1.523 | 23.54% | −16.88% | +78.86% |

Both rules fail the §1.2 gate on Sharpe (bar 1.3145) and pass on drawdown (bar
−7.79%); `mom52` is 0.048 Sharpe short. Neither is promoted. The reading: the winner's
1.336 is the max of 58,590 candidates (median 0.406, p90 0.682, p99 0.904, 0.5% above
1.0), so validation ranking is an order statistic; the rule is fixed, and its test
score is *higher* than its validation score while the winner's collapses. This is a
diagnostic of selection noise, not a claim that trend beats ML in general — one
window, and both books lose to EW-16.

### 5.1d AR(1) forecast baseline (2026-11-09)

§4.3's optional naive forecast baseline, in prediction space only: no positions and
no costs. Per ticker, an OLS one-lag model of the H-bar forward return on the
**trailing** H-bar return, `y_t = a + b * return_Hbar`, refit quarterly on a rolling
5y window with embargo = H, so every training label is realised before the first
prediction date. `pred_mean` is the AR(0) comparison forecast (each ticker's
training-window mean). Command:
`python scripts/run_forecast_baseline.py --horizon 4`; `--horizon 1` is the
non-overlapping weekly case. Tables under `results/forecast_baseline/ar1_h{1,4}/`.

| forecast | window | OOS R² vs 0 | OOS R² vs hist. mean | rank IC (t) | direction acc |
| --- | --- | --- | --- | --- | --- |
| AR(1), H=4 | val 2017-2022 | 0.0222 | 0.0132 | 0.112 (5.49) | 57.6% |
| AR(1), H=4 | test 2023-2025 | 0.0487 | −0.0012 | 0.069 (2.45) | 56.7% |
| historical mean, H=4 | val 2017-2022 | 0.0091 | 0 (by construction) | — | — |
| historical mean, H=4 | test 2023-2025 | 0.0498 | 0 (by construction) | — | — |
| AR(1), H=1 | val 2017-2022 | −0.0069 | −0.0087 | 0.034 (1.68) | 53.8% |
| AR(1), H=1 | test 2023-2025 | 0.0076 | −0.0017 | 0.053 (1.97) | 55.0% |
| ML winner, H=4 | val 2017-2022 | −0.1243 | — | 0.045 (2.29) | 54.7% |
| ML winner, H=4 | test 2023-2025 | −0.0387 | — | 0.097 (3.75) | 52.1% |

- The H=4 slope is negative at 95% of refits (mean −0.080, median −0.074): a mild
  four-week reversal, not momentum. Its apparent R² against zero is the
  unconditional mean — the historical-mean forecast alone scores 0.0091 / 0.0498 on
  the two windows, and the AR(1) is *below* that in test (R² vs mean −0.0012). No
  conditional information survives out of sample.
- H=1 confirms it: slope ≈ 0 (mean 0.0006), R² ≈ 0 and negative against the mean. At
  weekly frequency these returns are effectively white noise, so the H=4 "edge" is
  drift, not autocorrelation.
- The selected ML model's level forecasts are *worse than zero* on both windows
  (−0.1243 / −0.0387), while its test rank IC (0.097) is the highest in the table. The
  book monetised cross-sectional ranking and volatility sizing, not calibrated levels;
  an R²-only comparison would rank a one-parameter mean-reversion model above the
  whole pipeline, which is why both forecast metrics are reported.
- Caveats: 16 ETFs, weekly bars, one 2017-2025 span. The AR(1) validation IC
  (0.112) is largely persistent per-ticker drift differences carried by `pred_mean` —
  the mean forecast alone ranks at IC 0.091 (val) / 0.072 (test) — so it is a static
  cross-sectional bet, not time-varying timing skill, and it decays to 0.069 in test
  (below what `pred_mean` alone achieves there).

### 5.2 Removed — regime-stratified selection (2026-11-04)

A second criterion (gate on profitable calendar years plus up-capture, then maximin
over the worst calendar year) was executed here on 2026-11-04 and removed with its
code and artifacts on 2026-11-06. The numbers survive only in
`agents/current_state.md` Step 19.

## 6. Status

- **Scheme in force:** one joint ranking over model x strategy parameters on the
  combined 2017-2022 window; predictions cached under `results/_study/predictions`;
  `utils.config.DEFAULT_SEED`; one selection criterion (§3.1) over the scored space;
  only the winner is measured once on 2023-2025.
- **Promotion state:** closed on 2026-11-04 — the ranking winner fails the gate:
  0.1595 net Sharpe against the EW-16 bar 1.3145, maxDD −10.42% against the −7.79%
  bar → **null result**, nothing promoted, `HEADLINE_*` stays `None`.
- **Open question:** the top of the ranking sits inside a tight rff / D = 100 /
  gamma = 0.01 cluster whose validation edge does not survive out of sample (test OOS
  R² −0.039 / −0.051), and the 45 book cells rank almost randomly inside the window
  (Sharpe correlation between its two halves: 0.27 Pearson / 0.14 Spearman; against
  the test window: 0.05). Any follow-up is a new pre-registered cycle on new data, not
  a tweak of this one.
- **History:** earlier two-window protocols and their outputs were removed on
  2026-11-03 at the user's request; the summary numbers survive only in
  `agents/current_state.md` (Steps 1-17). The regime-stratified selection criterion
  (§5.2), its code, its `*_regime` ranking/winner files and its
  `results/plots{,_macro}/regime_{test,val}/` figure sets were removed on 2026-11-06
  at the user's request; the numbers survive in `agents/current_state.md` Step 19.
