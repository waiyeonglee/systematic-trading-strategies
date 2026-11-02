# Validation & Ablation Study — trend-following-strategy-v2

> **Purpose.** Two validation layers around the main design (`agents/model_design.py`)
> **without contaminating the final test window (2023–2025)**:
> - **Validation 1 — Hyperparameter tuning / model selection** (2017–2019): choose the
>   **(look-ahead, model spec)** pair with a pre-declared rule. Locked once.
> - **Validation 2 — Ablation** (2020–2022): portfolio-layer permutations around the
>   locked model. Anything that looks better must be **promoted through the
>   pre-registration ADR process** before it can enter the headline record — never by
>   picking it from this table.
>
> This file is the v2-owned copy of the reference project's `agents/ablation_study.md`
> (copied after a staleness check, renamed to make the dual role explicit:
> protocol + measured record). Design constants live in `agents/model_design.py`; this
> file holds the run rules and the results log. The reference project's protocol text
> and this project's first executed run were compared: the v2 engine reproduced every
> number byte-identically on the frozen v2 dataset (see §5.1).

> **Label correction (2026-11-02):** `fwd_ret_4bar` and `fwd_ret_12bar` now end at
> `t+5` and `t+13`, respectively, to represent 4-/12-bar holds after entry at `t+1`.
> Labels are generated only by `src/labels.py`. The results below were measured with
> the prior shorter targets and are retained as history, not valid current model
> selection. Rerun V1 and V2 before further conclusions. The 2023–2025 test was already
> measured, so any corrected run on that window is a remeasurement, not a fresh holdout.

---

## 1. Data split & test-window discipline (read this first)

### 1.1 The split (fixed, mechanical, pre-decided)

Data is used **only through the v2 fixed window — no 2026 data** (data end inclusive
2026-01-01; fetch bound exclusive 2026-01-02). The four layers are a mechanical,
calendar-anchored rule (the last 9 years split into three contiguous 3-year blocks),
fixed **before** looking at market regimes. Choosing years because of what happened in
them would be an outcome-based leak, so whatever regimes land in each block are
accepted as-is.

| Layer | Window | Role |
| --- | --- | --- |
| Warmup | 2004-01-02..2010-12-31 | feature / indicator warmup before training starts |
| **Train** | **2011–2016** | rolling **5y (260 weeks)** minus the **embargo (matched to the look-ahead target: 4w or 12w)** at each **quarterly refit**; the first refit trains on ~2011–2016, and the window advances every quarter. Retraining happens inside **every** layer. |
| **Validation 1 (hyperparameter tuning / model selection)** | **2017-01-01..2019-12-31** | choose the **(look-ahead, model spec)** pair — pre-declared rule: best **validation net Sharpe** on the plain (overlay-free) base book, degenerate excluded. Locked once, then used unchanged. |
| **Validation 2 (ablation)** | **2020-01-01..2022-12-31** | portfolio-layer permutations, run with the **model locked on 2017–2019** (`mode: single`). May be re-used freely; this is where the rows in §4 are measured. |
| **Test (final)** | **2023-01-01..2025-12-31** | previously measured once; effective end with corrected labels is **2025-09-26** under a 12-week hold (≈ 2025-11-21 under 4 weeks; see `agents/feature_engineering.md` §4). Benchmark gate bars need no labels and run to 2025-12-31. |

### 1.2 Rules that keep the headline numbers honest

1. **The record is re-run from scratch on this split.** No comparability with earlier
   (old-window) measurements is claimed; the new record starts fresh with the windows
   above.
2. **The test window is final and untouched.** It selects nothing: not the model, not
   the portfolio config, not any row below. A test set's job is **confirmation, not
   selection** — running many unplanned permutations on it and keeping whichever looks
   best would silently turn the test into a training set.
3. **Gate bars are recomputed on the test window.** Before any headline run, SPY and
   EW-16 buy-and-hold stats are recomputed on **2023-01-01..2025-12-31** and
   pre-registered (new ADR).
4. **Promotion rule.** If a permutation looks better on the ablation window, it is
   promoted by (a) writing the hypothesis + expected mechanism **before** measuring
   (a new ADR), (b) measuring **once** on the final test window, and (c) recording the
   outcome in `agents/current_state.md`. No pre-registration → no promotion.
   A null result is logged here and forgotten — that is a completed experiment, not a
   failure of discipline.
5. **One variable at a time.** Each row changes exactly one knob relative to its
   stated base. Combined permutations are allowed only after each leg is understood.

## 2. Data-window map

| Window | Years | Role | May be touched by |
| --- | --- | --- | --- |
| Warmup | 2004–2010 | indicator history before training | (nothing) |
| Train | rolling **5y** (refit quarterly, embargo = target horizon); first refit ≈2011–2016 | model fitting | every run |
| Validation 1 (model selection) | **2017–2019** | **(model, look-ahead, frequency) joint selection** (best val net Sharpe on the plain overlay-free book, degenerate excluded) | selection runs only |
| **Validation 2 (ablation)** | **2020–2022** | portfolio-layer permutation exploration | ablation study only |
| **Test (final)** | **2023–2025** | previously measured; corrected effective end 2025-09-26 under a 12-week target | corrected reruns are remeasurements, not fresh holdouts |

## 3. Validation 1 — Hyperparameter tuning / Model selection (2017–2019)

- **Window:** validation 2017-01-01..2019-12-31 only.
- **Selection rule (pre-declared):** best **validation net Sharpe** on the plain
  (overlay-free) base book, degenerate excluded (e.g., constant / near-zero
  predictions). Locked once, then used unchanged in every ablation row. Runs under
  `mode: selection`.
- **Default portfolio (fixed across all candidates):** the plain base book — with
  predictions computed on rebalance bars (prediction frequency is locked to cadence),
  vol-scaled sizing with the fixed **24-week vol window**, cost-gated, moomoo costs,
  **overlay-free** — no vol target, no rf sleeve, no leverage (those are Validation 2
  decisions and must not influence which (model, look-ahead, frequency) wins).
- **Walk-forward mechanics (fixed, not tuned):** rolling 5y window, quarterly refit,
  **embargo = target horizon** (1 bar for a 1-week target, 4 for 4-week, 12 for
  12-week).

| # | Hyperparameter | Levels | Mechanism / design rationale | Config / notes |
| --- | --- | --- | --- | --- |
| 1 | Rebalance frequency | 1w / 4w / 12w (`weekly` / `4w` / `12w`) | How often the portfolio acts. Faster acting = more responsive, higher turnover; slower = fewer decisions, staler exposure. **n-week cadence:** 1w = first bar of each week; 4w / 12w = every 4th / 12th bar (horizon-aligned, each position held exactly n bars). | `rebalance: weekly|4w|12w`; predictions on rebalance bars |
| 2 | Prediction look-ahead | 1w / 4w / 12w | What the model predicts (the target label). Longer horizon → more stable, learnable signal; shorter → more responsive but noisier. **Rule: look-ahead >= frequency** — a position must be held at least as long as the acting interval. | `target: absolute1/4/12`; **embargo = horizon** (1 / 4 / 12 bars) |
| 3 | Model family | Ridge / Lasso / Elastic Net / **RFF ridge** | Ridge shrinks all 21 features uniformly (dense); lasso zeroes weak features (sparse); elastic net mixes both. **RFF ridge** (added 2026-11-02, `agents/current_state.md` Step 6) = ridge on seeded random Fourier features — a Monte-Carlo approximation of Gaussian-kernel ridge, which is the design's nonlinear candidate for patterns a linear fit misses. Relevant because only a subset of features may carry trend signal. | 114-spec pre-set grid |
| 4 | Regularization strength | **per-family α**: ridge & rff {1e-3, 1e-2, 1e-1, 1, 10, 100}; lasso & enet {1e-4, 1e-3, 1e-2, 1e-1, 1, 10}; enet × l1_ratio {0.5, 0.25} | Strength of shrinkage vs in-sample overfit. Higher α = simpler model, lower variance; too high = underfit. Prior: mid-strength ridge tends to beat both extremes. RFF α follows the ridge grid (same L2 range; α = 1e-4 with D = 500 features would sit near the interpolating regime). **2026-11-02:** L1 families moved to smaller alphas — on this data max feature signal is ~0.001–0.008 (α units), so a shared {1e-3..100} grid put lasso/enet above the entry threshold (intercept-only, degenerate); the per-family grids restore them to the active range (smoke-verified, see `agents/current_state.md`). | grid: 6 ridge + 6 lasso + 12 enet + 120 rff = 144 specs |
| 5 | Random features (D) | **rff only**: 50 / 100 / 500 / 1000 | Kernel-approximation capacity of RFF: more random features → closer to the exact Gaussian-kernel ridge. D must exceed p = 21 — a D = 10 projection (the first proposal) is a bottleneck with ~1/√D ≈ 32% approximation error and selection noise from the W draw, so the low anchor is 50. D = 1000 added 2026-11-02 (Step 6 follow-up 5) for completeness. | `rff_dim: 50\|100\|500\|1000` |
| 6 | Kernel bandwidth (γ) | **rff only**: 1e-3 / 1e-2 / 5e-2 / 1e-1 / 5e-1 | γ is the RBF **inverse bandwidth** in k(x,y) = exp(−γ‖x−y‖²) — NOT the lengthscale (σ = 1/√(2γ)). Features are standardized per fit (p = 21), so typical pair distance² ≈ 2p ≈ 42 and the median-heuristic γ ≈ 0.01; the grid spans smooth → local. γ ≥ 1 collapses the kernel to a point mass and aliases the RFF frequencies to noise — the proposal's {1, 10, 100} tail was dropped. | `gamma: 0.001|0.01|0.05|0.1|0.5` |
| 7 | Model refit cadence | **weekly / monthly / quarterly** (`W-FRI` / `ME` / `QE`) | How often the model is retrained — controls model staleness at prediction bars vs compute. **Rule (economy, NOT statistical validity like look-ahead ≥ frequency): refit is no more frequent than the acting cadence — refit_interval ≥ prediction_interval** (W-FRI=1, ME=4, QE=12 bars vs cadence 1/4/12). Refits finer than the cadence fit models whose intermediate predictions the book never uses; refit == cadence captures the freshness. QE stays valid for every cadence (the current QE+weekly config remains admissible, just re-measured). | `refit_every: W-FRI\|ME\|QE`; 14 valid (freq, la, refit) combos |
- **Selection rule (declared, not tuned):** best validation net Sharpe, degenerate excluded. Pre-registered **before** measuring. | recorded in this file before the run |

> **Design revision (2026-11-02):** the grid was expanded to 4 families ×
> (frequency × look-ahead) × refit cadence — 144 specs (24 linear + 120 RFF ridge)
> × 14 valid combos = **2016 candidates**, selected **jointly** on 2017–2019.
> The RFF family (added later the same day, `agents/current_state.md` Step 6)
> re-anchors its grid to the data: α = ridge's {1e-3..100}, D = {50, 100, 500,
> 1000} (Step 6 follow-up 5 extended D by 1000), γ = {1e-3..0.5} (rows 4–6).
> The refit cadence dimension (Step 6 follow-up 2, row 7) adds
> weekly/monthly/quarterly refits under the rule
> **refit_interval ≥ prediction_interval**. The old Phase 2b deferred
> look-ahead tie-break (the old `validation_and_ablation_extension.md`, removed
> 2026-11-02 — verdict archived in §5.2) is superseded by this joint selection.

**Grid → decision:** **144 specs × 14 (frequency, look-ahead, refit) combos = 2016
candidates**, one joint selection on 2017–2019 (not sequential). Valid combos
(look-ahead ≥ frequency AND refit_interval ≥ prediction_interval): W-FRI ×
{weekly pairs} (3), ME × {1w, 4w cadences} (5), QE × {all 6} (14). The winner —
(model, look-ahead, frequency, refit) — is locked and feeds **every** ablation row
on 2020–2022. This keeps the ablation window purely portfolio-layer.

## 4. Validation 2 — Ablation: portfolio-layer permutations (2020–2022)

Base for every row unless noted: the **base book** — predictions computed on
rebalance bars at the V1-locked frequency (cadence is locked in V1, not ablated),
vol-target 10% / cap 1.0, 3% rf sleeve, moomoo costs, no leverage, 24-week vol
window, **model + look-ahead + frequency locked from Validation 1**. Status: `to run`
= to be measured on 2020–2022 (with the lock selected on 2017–2019, `mode: single`);
results are logged in §5.
In the matrix below, the **Base** column is the default for each row and the
**Variant** column is the single change tested.

| # | Layer | Dimension | Base | Variant | Config delta | Condition / mechanism |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Sizing | Vol estimation window | 24w | 4w / 8w / 12w / 24w / 48w | `vol_window: 4 / 8 / 12 / 24 / 48` | — how reactive risk sizing is; the mechanism that can explain vol overshoots / drawdown amplification. **No hard validity rule** (unlike look-ahead >= frequency): the window is a backward estimator, so any window >= 2 bars is structurally valid; soft minimum is window >= one full holding period (>= rebalance cadence in bars), and with weekly bars a 4w window is only ~3 dof — very noisy, which is exactly what the sweep measures. 24w = base (in-sweep; the volwin_24w cell duplicates the base row as a consistency check) |
| 2 | Sizing | Vol target | 10% | None / 20% | `vol_target: null / 0.20` | whether the portfolio reduces overall exposure during a volatile period; 20% tests the opposite side (more exposure in calm periods, within cap 1.0) |
| 3 | Sizing | Risk-free sleeve | 3% | 0% (cash idle) | `vol_target_rf: 0.0` | **only when vol target ON** — rewards holding cash during de-risked (volatile) periods |
| 4 | Sizing | Leverage | cap 1.0 (no leverage) | cap 2.0 / cap 3.0 + fee gate | `vol_target_cap: 2.0 / 3.0`, `vol_target_lev_fee: 0.068` | **only when vol target ON** — borrow in low-volatility periods; prior: higher total return, lower Sharpe. Leverage sweep {1.0 (base), 2.0, 3.0} — cap 3.0 added 2026-11-02 (Step 6 follow-up 8), same 6.8% borrow fee |

**Run order:** 1 → 2 → 3 → 4. Row 4 after row 1: leverage's timing depends on the
risk estimate, so the vol-window finding comes first. One variable at a time; combined
permutations only after each leg is understood. Cadence is **not** a row: frequency is
a V1 selection dimension (2026-11-02 design revision) — the locked (look-ahead,
frequency) pair is used as-is in every row, and off-diagonal pairs (look-ahead <
frequency) are excluded by design. Vol-knob rows were expanded on 2026-11-02
(`agents/current_state.md` Step 6 follow-up 6) from {30w, 90w} x {None} to
{4w, 8w, 12w, 24w, 48w} x {None, 10% (base), 20%}; the base book was then re-based
to the new defaults (vol_window **24w** — was 60w — and vol_target 10%, follow-up 7),
so Validation 1 re-runs on the 24w base book (the vol window enters the V1-scored
book via the forecast-vol scaling, `historical_forecast_vol`).

## 5. Log format for new results

For every validation run (V1 selection or V2 ablation), append a row with:

```
| date | stage (#V1row / #V2row) | window (2017–2019 / 2020–2022) | net Sharpe | maxDD | total | vol | TO | verdict vs base | promoted? (ADR #) |
```

Keep the full run tag and result folder path next to the row. If promoted, the ADR
record in `agents/current_state.md` becomes the authoritative record; the row here is
only a breadcrumb.

### 5.1 Executed results — V1 joint selection (2026-11-02, expanded grid)

**Current record — 2016-candidate joint selection (144-spec grid, 24w base book).**
Grid: 144 specs (24 linear + 120 rff, rff_dim incl. 1000) × 14 valid (frequency,
look-ahead, refit) combos on 2017–2019 (overlay-free base book), degenerate
excluded by the pre-registered guard (see the NaN/merge note below). The base
book's vol window was re-based 60w → 24w (Step 6 follow-up 7) before this run.

- **Degenerates by family:** lasso 55/84, enet 95/168, ridge 0/84, **rff 0/1680**.
  The L1 collapses concentrate on the 1-week target (entry-threshold finding);
  every NaN-Sharpe row is also flagged degenerate (150 rows → None in the merged
  ranking), so no degenerate survives selection.
- **Per-combo winners (validation net Sharpe, non-degenerate):**

| prediction | look-ahead | refit | winner | val net Sharpe |
| --- | --- | --- | --- | --- |
| 1w (weekly) | 1 | ME | enet_a0.001_r0.5 | **1.3370** |
| 1w | 1 | W-FRI | enet_a0.001_r0.5 | 1.3300 |
| 1w | 1 | QE | enet_a0.001_r0.5 | 1.2056 |
| 1w | 4 | ME | rff_a1_d1000_g0.01 | 1.2487 |
| 1w | 4 | W-FRI | rff_a100_d50_g0.05 | 1.1117 |
| 1w | 4 | QE | rff_a1_d1000_g0.01 | 1.2403 |
| 1w | 12 | ME | ridge_a100 | 1.0828 |
| 1w | 12 | W-FRI | rff_a10_d50_g0.1 | 1.0606 |
| 1w | 12 | QE | enet_a0.001_r0.25 | 1.0802 |
| 4w | 4 | ME | rff_a10_d100_g0.05 | 1.2313 |
| 4w | 4 | QE | rff_a10_d100_g0.05 | 1.2654 |
| 4w | 12 | ME | rff_a0.01_d500_g0.01 | 1.1220 |
| 4w | 12 | QE | rff_a0.01_d500_g0.01 | 1.0517 |
| 12w | 12 | QE | rff_a100_d100_g0.5 | 0.9329 |

- **Locked winner (joint selection):** **(enet_a0.001_r0.5, absolute1, weekly, ME)** —
  val net Sharpe **1.3370** (gross 1.3660, avg TO 0.1053), non-degenerate.
  `results/_study/validation_part1_winner.json`; 14 per-combo summaries
  `validation_part1_{laX_freqYw}_{refit-slug}.json`; merged 2016-row ranking
  `validation_part1_rankings.json`; run folders `results/<tag>__v1_<slug>/`
  (tag `..._grid144_...`).
- **What the D=1000 extension changed:** 420 d1000 rows (30 specs × 14 combos); the
  best d1000 spec ranks **#4 overall** and wins two 4w-look-ahead weekly cells
  (rff_a1_d1000_g0.01, ME 1.2487 / QE 1.2403 — beating the d500/d100 winners of the
  114-grid). D=1000 adds ~0.02–0.03 Sharpe in the rff 4w cells but never threatens
  the enet 1w lock. Top-20 has 6 d1000 rows (~30% vs 21% of the grid).
- **Refit effect (the new dimension pays):** in the winning cell (weekly × 1w,
  enet_a0.001_r0.5), monthly refits edge weekly (1.3370 > 1.3300) and beat
  quarterly (1.2056). Refit cadence remains a first-order knob.
- **Finding — refit cadence, monthly over weekly (real-life practicality, user
  note):** V1 shows monthly and weekly refits deliver nearly identical net Sharpe
  (1.337 vs 1.330 on 2017–2019), with essentially the same gross Sharpe — the gap
  is purely cost: monthly refits turn the book over more slowly (avg turnover
  0.105 vs 0.122), so they shed less to execution costs. In practice, when two
  cadences perform equivalently, the slower refit is preferable: it reduces
  modeling cost (half the retraining) and slows fund turnover (fewer rebalances,
  lower transaction costs). The locked model is therefore not just the best
  performer, but also the most practical to run live. (V2 keeps the locked ME
  refit; no W-FRI robustness row.)
- **RFF note:** RFF won 10/14 per-combo bests but never the global — its best
  (rff_a10_d100_g0.05, 4w × 4w, QE, 1.2654) trails the enet lock (1.3370). RFF
  owns every 4w-look-ahead cell; ridge wins one 12w cell (1w cadence); lasso never
  wins a cell. The 12w × 12w cell is the weakest (0.9329), as before.
- **Merge/NaN note (process):** NaN-Sharpe rows (lasso/enet degenerate
  manifestations) are stored as null and sorted last; the 14/14 cell winners
  cross-check against the merged ranking; ranking is monotonic non-increasing.
- **Breadcrumb — 114-grid run (superseded same day, 60w base):** same lock
  (enet_a0.001_r0.5, absolute1, weekly, ME) @ **1.3674**; per-combo winners table
  differs only in the rff cells (d500/d100 winners; e.g., 1w×4 ME rff_a1_d500_g0.01
  1.2216, 4w×4 QE rff_a10_d100_g0.05 1.2218, 12w×12 QE 0.9909). The lock's
  stability across the 60w → 24w rebase and the D=1000 extension is itself a
  robustness signal.
- **Next (handoff):** ✅ **executed 2026-11-02** — V2 (§5.3, 11 rows → volwin_12w
  winner), promotion ADR (current_state.md Step 6 follow-up 10), then the single
  pre-registered 2023–2025 test draw (§5.4) → **gate FAIL, null result — nothing
  promoted, experiment cycle closed.**

**Breadcrumb — pre-refit 684-candidate run (same day, superseded):** grid without
the refit dimension (6 (freq, la) pairs, all QE refits) locked
(enet_a0.001_r0.5, absolute1, weekly) @ 1.1950 — the same model, now re-measured
in the expanded grid (QE cell of the winning combo = 1.1950, matching exactly).
Kept as history; not the current lock.

**Historical note:** the pre-revision execution (2 look-aheads × 6 specs → locked
(absolute4, ridge_a10); V2 rows 1–5 → nothing promoted) is preserved as history in
`agents/current_state.md` (Step 5). It is **not** the current grid and must not be
read as the current experiment.

### 5.2 Phase 2b — deferred look-ahead tie-break (LEGACY, file removed)

**Superseded 2026-11-02** by the joint V1 selection — the look-ahead is now selected
directly (coupled to frequency), so there is no separate deferred tie-break phase.
`validation_and_ablation_extension.md` was **removed 2026-11-02** (pure history,
no active configs); its executed v2b record is archived here in compact form:

- v2b verdict (old design, ridge_a10 lock, 2020–2022, margin rule M = 0.20 over
  incumbent (a4, base) 0.5564, threshold 0.7564): best cell a4 row2 weekly 0.6635 →
  **margin not cleared → NOT promoted**; keep (absolute4, base). The absolute12
  look-ahead was systematically worse (6 of 7 books); the weekly edge died under
  absolute12 (0.6635 → 0.4481, Δ −0.215). Superseded since by joint V1 selection.
  Full 14-cell matrix was in the removed file (recoverable via backups).

### 5.3 Executed results — Validation 2 (ablation, 2020–2022) — current record

**Current record (2026-11-02, 144-grid lock).** Runs on 2020-01-01..2022-12-31 with
the V1-locked winner **(enet_a0.001_r0.5, absolute1, weekly, ME refit)**; only the
portfolio knob varies per row (11 rows, one at a time from the base book). Net
Sharpe / total / maxDD on the ablation window (for ablation comparison only, NOT
headline):

| row | config change | net Sharpe | total | maxDD | notes |
| --- | --- | --- | --- | --- | --- |
| base | 24w · vt 10% · rf 3% · cap 1.0 | 0.5180 | +18.8% | −20.9% | default book |
| **volwin_12w** | vol window **12w** | **0.6574** | **+28.0%** | −20.7% | **winner** (V1 echo 1.5604 > base 1.3957) |
| volwin_48w | vol window 48w | 0.6021 | +21.3% | −18.5% | runner-up |
| volwin_24w | 24w (== base) | 0.5180 | — | — | consistency check: bit-identical to base ✓ |
| row4_lev2 / lev3 | cap 2.0 / 3.0 + 6.8% fee | 0.5177 | — | — | == base: leverage caps never bind in 2020–22 |
| vt_none | vol target off | 0.4855 | +17.9% | −19.8% | 10% target confirmed |
| row3_rf0 | rf 0% | 0.4836 | +17.3% | — | rf 3% confirmed (identical trades, lost carry) |
| vt_20 | target 20% | 0.4558 | +16.6% | — | 10% target confirmed |
| volwin_4w / 8w | 4w / 8w | 0.4058 / 0.2437 | — | — | short windows noisy (non-monotonic) |

- **Verdict: volwin_12w wins** (both on 2020–22 and on the 2017–19 echo); promoted
  per the ADR (current_state.md Step 6 follow-up 10). Summaries:
  `results/_study/validation_part2_{row}.json`, folders
  `results/<tag>__v2_<row>/` (tag `..._enet_a0.001_r0.5_vt0.1c1rf3_rweekly_2020-2022`).
- **Breadcrumb — 114-grid V2 (superseded same day):** same lock, 60w base **0.6274**
  (rows: 30w 0.5751 · 90w 0.6077 · notarget 0.5096 · rf0 0.5748 · lev2 0.6273),
  verdict then "base stands". Superseded by the 144-grid re-run + 24w rebase.

### 5.4 Executed results — TEST window: single pre-registered draw (2023–2025)

The one draw, run after the promotion ADR (Step 6 follow-up 10) with the promoted
stack: enet_a0.001_r0.5 · absolute1 (1w) · weekly · ME refit · **vol_window 12w** ·
vt 10% · rf 3% · cap 1.0 · moomoo. Folder
`results/test_2023-2025_promoted` (tag `..._mrefit_enet_a0.001_r0.5_vt0.1c1rf3_rweekly_2023-2025`).
**The test window had never been touched before this draw.**

| metric (2023–2025) | promoted book | EW-16 | SPY B&H |
| --- | --- | --- | --- |
| ann return (net) | 11.95% | 11.90% | 22.28% |
| ann vol | 10.08% | 9.35% | 14.66% |
| total return (3y) | +36.70% | +36.53% | +74.53% |
| hit rate | 63.2% | 59.0% | 59.0% |
| max drawdown (net) | −11.95% | **−7.79%** | −16.88% |
| Sharpe (net) | 1.1716 | **1.2502** | ~1.52 |

Forecast quality: OOS R² 1.65% (2023 1.96% / 2024 0.81% / 2025 2.06%), RMSE 0.022,
direction acc 55.5%, rank-IC 0.0665 (t = 2.47).

- **Gate verdict (§3.2): FAIL — SUCCESS=False.** sharpe_ok=False (book 1.1716 <
  bar 1.2502 = weaker benchmark Sharpe, EW-16); max_drawdown_ok=False (book
  −11.95% < bar −7.79% = EW-16's less-negative drawdown, the harder target; the
  book is deeper than EW but shallower than SPY's −16.88%). The book beats nothing
  on the final window: it misses the EW Sharpe bar and carries a deeper drawdown
  than the better benchmark; its positive total is essentially market exposure
  (SPY returned +74.53% over the same window).
- **Protocol outcome (rule 1.2.4):** a null result — logged here and treated as a
  completed experiment, not a promotion. The promoted stack is **not** confirmed
  out-of-sample. No further draws on this window; any future iteration starts as a
  NEW pre-registered cycle (new hypothesis + new ADR before any test measurement).

## 6. Current status (label correction, 2026-11-02)

- **Prior-cycle outcome (superseded for current selection):** the recorded experiment
  cycle closed with a null test result.
  V1 lock: **(enet_a0.001_r0.5, absolute1, weekly, ME refit)** @ 1.3370 (2017–19,
  144-spec/2016-candidate joint selection). V2 winner: **vol_window 12w** book @
  0.6574 (2020–22). Test draw (2023–25, once): **gate FAIL** — Sharpe 1.1716 <
  1.2502 bar, maxDD −11.95% > −7.79% bar → **nothing promoted, no
  headline stack exists.** Those metrics used the previous label formulas.
- **Next:** rebuild the feature panel and rerun V1 and V2 with corrected labels.
  Keep historical run files, but do not use their rankings/results as current
  evidence. `HEADLINE_*` remains `None`.
- The 2023–2025 period is already consumed. A corrected run may be recorded for
  methodological completeness, but it is a remeasurement and cannot be presented
  as an independent untouched test.
