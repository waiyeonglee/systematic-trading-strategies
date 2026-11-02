# current_state.md — trend-following-strategy-v2

History log of decisions and actions for this project. Updated at each step.
Reference project: `../trend-following-strategy` (v1). Both live under
`/Users/bytedance/Projects/personal/DSQF/Project2`.

---

## Step 1 — Dataset (2026-11-01)

### Goal
Stand up the standalone v2 dataset with the same data-folder format as the
reference project, with a **fixed end date of 2026-01-01**, and make sure no stale
imports / old results / derived files leak over from v1.

### Decisions (recorded so future steps stay consistent)

1. **Fixed data window.** End date fixed at `2026-01-01`. Last weekly bar =
   `2025-12-26` (final complete Friday week ≤ end date); Fear & Greed daily series
   ends `2025-12-31`. Re-fetches must respect this fixed end; data is treated as
   frozen after 2025-12-31.
2. **Start / warmup.** Download start `2004-01-01` (or instrument inception, e.g.
   GLD 2004-11-19, DBC 2006-02-10, UUP 2007-03-02). The study window is 2011+;
   pre-2011 history is warmup only (25w vol, long trend MAs, etc.).
3. **Universe.** 16 tradable + 1 benchmark, identical to v1:
   sectors XLB XLE XLF XLI XLK XLP XLU XLV XLY; cross-asset TLT IEF GLD DBC UUP
   EFA EEM; benchmark SPY.
4. **Frequency / format.** Weekly Friday-ending bars resampled from daily
   (Open=first, High=max, Low=min, Close=last, Volume=sum); yfinance
   `auto_adjust=True` adjusted OHLCV; columns `Date,Open,High,Low,Close,Volume`.
   Same layout as v1: `data/raw/weekly/<TICKER>.csv`, `data/additional/fear-greed.csv`.
5. **Fear & Greed.** CNN index (0–100) from the public
   [fear-greed-data](https://github.com/whit3rabbit/fear-greed-data) repo;
   2011-01-03 → 2025-12-31, 3,750 rows, no nulls/duplicates. Pipeline uses only the
   numeric `Fear Greed` column.

### Actions taken

- Verified v1 source data: all 17 tickers present, consistent end `2025-12-26`,
  pre-2011 warmup intact; fear-greed validated (no nulls/dupes, range 0–97).
- Created v2 skeleton: `data/raw/weekly/`, `data/additional/`, `agents/`.
- Copied **only source inputs** into v2:
  - `data/raw/weekly/*.csv` (17 tickers) + `.gitkeep`
  - `data/additional/fear-greed.csv`
- Wrote `data/README.md` (window convention, universe table, per-ticker coverage,
  fear-greed provenance) and project `.gitignore` (raw CSVs ignored, features/
  results/ ignored, config.yaml ignored).

### Addendum — fetch script + trim bound (same day)

- Created `scripts/fetch_data.py`: standalone Yahoo Finance regeneration script
  with the fixed settings baked in — `START_DATE = "2004-01-01"` (inclusive),
  `END_DATE_INCLUSIVE = "2026-01-01"`, `END_DATE_EXCLUSIVE = "2026-01-02"`.
  The pipeline imports the CSVs; this script exists only to rebuild them.
  - Weekly Friday bars (Open=first, High=max, Low=min, Close=last, Volume=sum),
    `auto_adjust=True`, trailing partial week dropped, rows beyond the fixed end
    trimmed, duplicate dates dropped.
  - `--dry-run` prints settings without downloading (verified: end=2026-01-02
    exclusive, 17 tickers); `py_compile` passes.
- Ran the manual trim across **all** CSVs (`data/**/*.csv`, 18 files): keep only
  `Date < 2026-01-02` (exclusive). Result: **0 rows dropped** — the dataset
  already conformed (weekly max 2025-12-26, fear-greed max 2025-12-31). The bound
  is now an enforced invariant, not just a convention.

### Explicitly NOT carried over (standalone hygiene)

- **No stale imports / code**: no `src/`, `main.py`, `config.yaml`, `pyproject.toml`
  from v1. v2 code will be written fresh.
- **No derived / stale data**: `data/features/` (feature tables + correlation
  artifacts) excluded — rebuilt by v2's own pipeline.
- **No old results**: v1 `results/` (backtests, model artifacts, plots) excluded.
- **No v1 ADR/design docs** copied into v2's `agents/`.

### Files now in v2

```
trend-following-strategy-v2/
  .gitignore
  scripts/
    fetch_data.py            # re-fetch tool; end exclusive 2026-01-02
  data/
    README.md
    raw/weekly/<17 ticker CSVs> + .gitkeep
    additional/fear-greed.csv
  agents/
    current_state.md          # this file
```

### Verified (addendum)

- `scripts/fetch_data.py`: compiles; `--dry-run` prints the contract settings
  (start 2004-01-01 inclusive, end 2026-01-02 exclusive, 17 tickers).
- Trim pass on all 18 CSVs: 0 rows on/after 2026-01-02 (no-op, invariant holds).
- Data validation from Step 1 still passes (row counts, ranges, OHLC sanity).

### Notes / open items

- Actual re-fetch via Yahoo Finance is **untested** (HTTP 429 rate-limit at build
  time). The script is ready and bound-correct; when network allows, a re-fetch
  must use `END_DATE_EXCLUSIVE = "2026-01-02"` (already the script default) and
  should reproduce the same CSVs.
- Fear-greed source zip (`fear_and_greed_index.zip`) lives in the Project2 root if
  a re-extraction is ever needed. The trim bound applies to it as well.

---

## Next steps

- [x] Project scaffolding: venv + pyproject/uv.lock + src layout + README (Steps 2–3).
      Config conventions for runs (config.yaml loader, if wanted) not yet added —
      settings currently live as constants in `src/utils/config.py`.
- [x] Feature engineering: warmup-correct features (e.g. 25w realized vol, trend MAs) on weekly bars.
- [ ] Labels panel assembly / model / backtest — to be planned in later steps.

---

## Step 2 — Feature engineering (2026-11-01)

### Goal
Own the feature engineering in v2: copy the reference implementation doc
(`../trend-following-strategy/agents/feature_engineering.md`), check it for
staleness, and re-implement in this project's own `src/`, then build and validate
v2's feature panel.

### Staleness check on the reference doc (before copying)

Validated the doc against the reference implementation and outputs:
- Feature list (19 base + 2 trend = 21 with fear-greed) matches `src/features.py`.
- Label timing `price[t+H]/price[t+1]-1` and the three label columns match
  `src/labels.py` (verified formula-by-formula).
- Panel claims verified against reference `data/features/weekly/features.csv`:
  18,047 rows × 24 cols (date, ticker, 21 features, label_forward_return_12bar),
  dates 2004-01-02..2025-12-26, 16 tickers, 192 trailing NaN-label rows.
- Reproduction commands exist (`main.py fetch/features`).
- Only refresh needed: the end-date convention (reference said "data ends
  2025-12-31"; v2 fixes end inclusive **2026-01-01**, fetch exclusive **2026-01-02**)
  and the file paths (v2 has no `main.py` yet — the build entry is
  `scripts/build_features.py`).

Conclusion: **doc is current, not stale**; copied as v2-owned
`agents/feature_engineering.md` with the v2 convention updates.

### Implementation (written fresh, no stale imports)

- `src/utils/__init__.py`, `src/utils/paths.py` (data locations, FEAR_GREED_PATH),
  `src/utils/config.py` (universe constants + fixed window:
  START 2004-01-01 / END inclusive 2026-01-01 / fetch exclusive 2026-01-02),
  `src/utils/checks.py` (`require_columns`).
- `src/labels.py` — `make_labels` (fwd_ret / fwd_ret_4bar / fwd_ret_12bar),
  MODEL_TARGETS, MODEL_EMBARGO, label_column/target_horizon.
- `src/features.py` — `build_feature_frame` (21 features + label column),
  `feature_columns`, `feature_label_correlation`, `save_correlation_heatmap`.
- `scripts/build_features.py` — CLI entry (defaults: weekly, fear-greed on, H=12;
  flags `--frequency`, `--no-fear-greed`, `--horizon`).

### Build + validation (run with the reference venv's interpreter, which has talib)

- Output: `data/features/weekly/features.csv` (18,047 × 24),
  `feature_label_correlation.csv`, `feature_label_correlation.png`
  (heatmap visually verified: title "…through 2025-10-03", 22×22 annotated).
- **Exact-match validation:** compared v2 panel vs reference panel on all 22
  numeric columns → max abs diff **0.0** (byte-identical), same shape/cols/dates/
  tickers/NaN-label rows. The fresh re-implementation reproduces the validated
  reference exactly on the frozen dataset.
- Effective last labelled date: **2025-10-03** (12-week label lookahead).
- Note: talib is not installed in the default interpreter; build ran with the
  reference venv. v2's own venv (uv sync) is planned at the scaffolding step.  ← resolved in Step 3

### Files now in v2

```
trend-following-strategy-v2/
  .gitignore
  pyproject.toml, uv.lock, .python-version (3.13), README.md
  .venv/                        # uv-managed, git-ignored
  scripts/
    fetch_data.py            # re-fetch tool; end exclusive 2026-01-02
    build_features.py        # feature panel + correlation report
  src/
    features.py, labels.py
    utils/{__init__,config,paths,checks}.py
  agents/
    current_state.md         # this file
    feature_engineering.md   # v2-owned copy (refreshed)
  data/
    README.md
    raw/weekly/<17 ticker CSVs> + .gitkeep
    additional/fear-greed.csv
    features/weekly/         # features.csv + correlation report (git-ignored)
```

---

## Step 3 — Project venv (uv sync) (2026-11-01)

### Goal
Give v2 its own pinned environment so nothing depends on the reference project's
interpreter.

### Actions taken

- `pyproject.toml` (v2-owned): name `trend-following-strategy-v2`, requires-python
  `>=3.13`, dependency pins identical to the reference project's proven set
  (pandas 2.3.3, numpy 2.5.3, ta-lib 0.8.2, yfinance 1.7.0, scikit-learn 1.9.1,
  scipy, matplotlib, seaborn, pyyaml, joblib, tqdm, ipython; dev group:
  ipykernel, pytest). `[tool.pytest.ini_options]` testpaths=tests,
  pythonpath=src.
- `.python-version` = `3.13` (matches the reference venv).
- `README.md` — minimal top-level project doc (setup + run).
- `uv sync` → created `.venv` (Python 3.13.17) + `uv.lock`.

### Verified

- `.venv/bin/python --version` → Python 3.13.17; all core imports OK
  (talib 0.8.2, pandas 2.3.3, sklearn 1.9.1, yfinance, seaborn, etc.).
- Re-ran `scripts/build_features.py` with v2's own interpreter →
  **18,047 rows × 21 features, labels through 2025-10-03** — identical to the
  earlier build. The feature step is fully reproducible inside v2's environment.
- Resolved the Step-2 open item: v2 no longer needs the reference venv.

### Files now in v2 (additions)

- `pyproject.toml`, `uv.lock`, `.python-version`, `README.md`, `.venv/`.

---

## Step 4 — Model design & backtest layer (2026-11-02)

### Goal
Own the model design in v2: copy the reference design doc
(`../trend-following-strategy/agents/model_design.py`), check it for staleness, and
implement the model / walk-forward / portfolio / evaluation layer fresh in v2, then
validate by reproducing the reference project's measured ADR 09 stack exactly.

### Staleness check on the reference doc (before copying)

Validated the design doc against the reference implementation and the v2 frozen data:
- Design constants (6-spec grid ridge a1/a10/a100 + enet a1e-4-r0.5 / a1e-3-r0.5 /
  a1e-4-r0.25; targets absolute4/absolute12 with embargo = horizon; portfolio
  defaults vol_window 60 / vol_min_periods 20 / monthly / moomoo / cost 0 / gate 1.0 /
  vol target 10% / cap 1.0 / rf 3%; split train/val/test; success gate) match the
  reference `src/models.py`, `src/portfolio.py`, `src/pipeline.py` and `config.yaml`
  one-for-one.
- Design claims about the panel match v2's frozen output exactly: last labelled bar
  2025-10-03 under absolute12; 21 features; 16 tradables + SPY; benchmark #2 = EW-16.
- Walk-forward mechanics (rolling 5y ≈ 260 bars, quarterly refit, embargo = horizon,
  AR(1) naive baseline, selection = best overlay-free validation net Sharpe with
  degenerate exclusion) match the reference implementation.

Conclusion: **doc is current, not stale**; copied as v2-owned
`agents/model_design.py` refreshed only for v2 conventions — fixed window wording
(end inclusive 2026-01-01 / fetch exclusive 2026-01-02; equivalent to the doc's
"no 2026 data" cut), own file paths/lineage, and a status note recording v2's first
measured run.

### Implementation (written fresh, no stale imports)

- `src/models.py` — `ModelSpec` (ridge/lasso/enet builders, seed 5205),
  `specs_from_config` (expands alpha / l1_ratio ranges into the 6-spec grid),
  `walk_forward_predict` (grouped by `refit_every`, past dates minus embargo,
  optional rolling window, fresh fit per block), `ar1_walk_forward` (per-ticker
  AR(1) on the lagged target).
- `src/splits.py` — `build_splits` (train/val/test by `window_val` before
  `test_start`), `split_panel`.
- `src/portfolio.py` — `ForecastVolParams` / `TrendParams`, moomoo & IBKR
  commissions, `order_cost_usd`, forecast/realized vol windows (one-bar shift),
  `target_weights` (pred/vol → gross 1.0), `trend_target_weights`,
  `mark_to_market_weights`, `orders_for_target`, `cost_gated_orders`,
  `rebalance_dates`, `_vol_target_scale`, `build_forecast_vol_portfolio`
  (ADR 08 vol-target overlay + ADR 09 risk-free sleeve + ADR 11 leverage gate),
  `build_trend_portfolio` (long/flat + long/short), `STRATEGIES` registry,
  `build_positions` / `build_all_positions`.
- `src/metrics.py` — `perf_stats`, `sharpe`, `max_drawdown`, `strategy_report`,
  `comparison_table`, `spy_buy_and_hold`, `equal_weight_universe`, `capm_alpha`,
  `period_stats`, `oos_r2`, `oos_r2_vs_baseline`, `forecast_metrics`,
  `oos_r2_by_period`.
- `src/pipeline.py` — `PipelineConfig`, `run_tag` (folders like
  `weekly_fg_roll5y_val3y_qrefit_grid6_vt0.1c1rf3_2023-2025`), `select_model`
  (validation net Sharpe ranking, degenerate exclusion), `run_pipeline` (select →
  walk-forward test → AR(1) baseline → book + benchmarks → verdict),
  `trend_success_verdict` (design §3.2 gate), `evaluate_all` (tables only, no
  figures yet), `run_parameters` (run_config.yaml).
- `src/utils/` — `config.py` extended (`DEFAULT_SEED`, duration parse/convert
  helpers), `checks.py` (+`require_panel`), `paths.py` (+`RESULTS_DIR`),
  `artifacts.py` (new: `save_table` / `save_nested` / `save_parameters`).
- `scripts/run_backtest.py` — CLI entry. Loads raw weekly CSVs + Fear & Greed,
  builds the runnable panel (21 features + fwd_ret/fwd_ret_4bar/fwd_ret_12bar,
  NaN rows dropped → starts 2011-01-07, ends 2025-10-03), defaults = ADR 09 stack
  (test 2023-2025, window_train 5y, window_val 3y, QE refits, embargo 12,
  6-spec grid, forecast_vol_scaled book, vol target 10% / cap 1.0 / rf 3%);
  flags: `--test-start/--test-end/--window-val/--window-train/--refit-every/
  --horizon/--no-vol-target/--out-dir`.

### Build + validation (v2's own `.venv`, Python 3.13)

- Ran `scripts/run_backtest.py` → panel 2011-01-07..2025-10-03 (16 tickers × 21
  features), selected **enet_a0.001_r0.5** (validation net Sharpe 0.5771).
- **Exact-match validation:** diffed every table against the reference run
  `results/weekly_fg_roll5y_val3y_qrefit_grid6_vt0.1c1rf3_2023-2025`
  (same config, same frozen panel) → **all 11 CSVs byte-identical**:
  validation_ranking / comparison / comparison_no_spy / strategy / forecast /
  naive_forecast / oos_r2_by_year / capm / coefficients / performance_by_period
  (+_no_spy). The only `run_config.yaml` difference is the intended v2 window
  wording (data end 2026-01-01 vs 2025-12-31).
- Test-window headline (matches reference): net Sharpe **0.5755**, maxDD
  **−11.37%**, total **+14.54%**, hit rate 61.8%; gate vs EW-16/SPY SUCCESS=False
  (book trails both benchmarks on Sharpe and drawdown — same verdict as reference).

### Explicitly NOT carried over (standalone hygiene)

- **No v1 results / derived artifacts**: v2's `results/` was produced by v2's own
  run; no CSV/PNG copied from the reference `results/`.
- **No v1 code imports**: every module written fresh against the design; the only
  shared inputs are the frozen raw CSVs and the validated design numbers.
- **Not ported yet** (documented scope): figure outputs (`src/plots.py` equivalents
  — tables only for now), the rule-path CLI (`--rule`), the config.yaml loader,
  and the V1/V2 ablation protocol run (the engine supports it via split flags).

### Files now in v2 (additions)

```
trend-following-strategy-v2/
  scripts/
    run_backtest.py          # walk-forward selection backtest CLI
  src/
    models.py, splits.py, portfolio.py, metrics.py, pipeline.py
    utils/artifacts.py       # save_table / save_nested / save_parameters
  agents/
    model_design.py          # v2-owned copy (refreshed, staleness-checked)
  results/                   # run tables per tag (git-ignored)
```

### Notes / open items

- Reference figures (`plots.py` in v1) are not required for the design's core; a
  reporting step can add cumulative-return / drawdown / Sharpe-by-period figures.
- The design's fuller protocol (V1 selection 2017-2019 → V2 ablation 2020-2022 →
  TEST 2023-2025) is runnable with `--test-start/--window-val` flags; the first v2
  run intentionally used the reference's current stack for byte-exact validation.
- `--horizon 4` / `--no-vol-target` / custom splits are supported and can be
  verified against the reference's other runs when needed.

---

## Next steps

- [x] Project scaffolding: venv + pyproject/uv.lock + src layout + README (Steps 2–3).
      Config conventions for runs (config.yaml loader, if wanted) not yet added —
      settings currently live as constants in `src/utils/config.py`.
- [x] Feature engineering: warmup-correct features (e.g. 25w realized vol, trend MAs) on weekly bars.
- [x] Model design: ML path implemented and validated byte-identical to the reference
      ADR 09 stack (Step 4).
- [x] Validation & ablation study: V1 selection + V2 rows executed, byte-identical to
      the reference (Step 5).
- [ ] Rule path CLI, figures/reporting, promoted-ADR re-measurement — follow-up options.

---

## Step 5 — Validation & ablation study (2026-11-02)

### Goal
Run the two validation layers from `agents/model_design.py`: Validation 1 (model
selection on 2017–2019) and Validation 2 (portfolio ablation on 2020–2022), following
the reference project's `agents/ablation_study.md` as the implementation reference.

### Doc-structure decision (user question: overlap with model_design.py?)

- The reference intentionally keeps the **protocol summary** in `model_design.py`
  (§5–§7) and the **full rules + measured record** in `ablation_study.md` — the
  measured numbers live only in the study doc. Verdict: keep that separation, and
  **rename** the study doc to make its dual role explicit.
- v2 now owns **`agents/validation_and_ablation.md`** (protocol + results log,
  refreshed for v2 conventions), and `agents/model_design.py` was **slimmed**: it
  keeps the design constants (split windows, gate criterion, candidate count) and
  points to the study doc instead of restating the protocol — the duplication is
  resolved by a single source of truth for the run rules.

### Staleness check on the reference doc (before copying)

- Split/window definitions, selection rule, V2 row matrix, gate rules and the
  measured §5.1 numbers were cross-checked against the reference run folders under
  `results/` (tags, run_config.yaml, validation_ranking.csv) and against the v2
  engine's capabilities — all consistent. Only refresh needed: v2 window wording and
  the driver path (`scripts/run_study.py`).

### Implementation (extended v2 engine, no stale imports)

- `src/pipeline.py` — `PipelineConfig.mode` (`"selection"` | `"single"`); `run_tag`
  uses the pinned spec name (e.g. `..._qrefit_ridge_a10_...`) in single mode and adds
  the `_rweekly` segment for weekly cadence; `run_parameters` writes the true mode.
- `scripts/run_backtest.py` — new flags: `--model` (pin one spec → mode single),
  `--vol-window`, `--rebalance` (monthly|weekly), `--vol-cap`, `--vol-rf`,
  `--lev-fee` (the five V2 knobs).
- `scripts/run_study.py` (new, v2-owned driver) — `v1`: runs the 2 look-aheads on
  `test 2020-01-01..2020-01-31` (Jan-2020 byproduct window) with the overlay-free
  book, merges the two 6-spec rankings into 12 candidates, locks the winner
  (`results/_study/v1_*.json`); `v2`: runs the locked model (mode single) on
  2020–2022 for the base book and rows 1–5, one knob each
  (`results/_study/v2_*.json`).

### Executed results (all byte-identical to the reference)

- **V1 — 12-candidate selection on 2017–2019 (overlay-free).** All non-degenerate.
  Locked **(absolute4, ridge_a10)**, val net Sharpe 0.9698 (absolute12 ridge_a10
  0.9677 near tie). `validation_ranking.csv` per look-ahead byte-identical.
- **V2 — ablation rows on 2020–2022 (locked ridge_a10, absolute4):**
  base 0.5564 / −9.61% / +17.46% · row1 30w 0.5439 · row1 90w 0.5351 ·
  row2 weekly **0.6635** (+22.07%, TO 0.450 — candidate, ADR required, not promoted) ·
  row3 no vol-target 0.5468 · row4 rf 0% 0.5045 · row5 cap 2.0+6.8% fee 0.3280
  (null result: rejects the leverage prior). Same verdicts as the reference.
- **Validation sweep:** all **99 CSVs** across the 9 study folders (2× V1 + 7× V2)
  byte-identical to the reference `results/` folders; `run_config.yaml` differs only
  in the intended v2 fixed-window wording (end_date 2026-01-01 vs 2025-12-31).

### Explicitly NOT carried over (standalone hygiene)

- No reference `results/_study/*.json` copied — v2 regenerated its own extraction
  summaries; validation is by CSV byte-diff, not by copying numbers.
- No figures ported (tables only, as in Step 4); no `run_study.py` copied — v2 wrote
  its own driver against its own CLI.

### Files now in v2 (additions)

```
trend-following-strategy-v2/
  agents/
    validation_and_ablation.md   # v2-owned study doc (renamed from ablation_study)
  scripts/
    run_study.py                 # v1/v2 study driver
  results/
    weekly_fg_roll5y_val3y_qrefit_grid6_2020-2020__v1_absolute{4,12}
    weekly_fg_roll5y_val3y_qrefit_ridge_a10_{...}_2020-2022__v2_{base,row1_30w,row1_90w,row2_weekly,row3_notarget,row4_rf0,row5_lev2}
    _study/                      # v2 extraction summaries
                                 #   validation_part1_{absolute4,absolute12,rankings,winner}.json
                                 #   validation_part2_{base,row1_30w,row1_90w,row2_weekly,row3_notarget,row4_rf0,row5_lev2}.json
```

### Step 5 follow-up — headline re-point + summary naming (2026-11-02)

- **Headline re-pointed to the V1-locked model** (user request): the
  (absolute4, ridge_a10) pair is promoted as the headline candidate, replacing the
  Step-4 stack. The ADR test step (single measurement on the untouched 2023–2025
  window) was **deferred — not executed**; it is scheduled after the expanded
  experiment under the design revision below.
- **`results/_study/` summaries renamed** to the verbose scheme
  `validation_part1_*` / `validation_part2_*` (user request; `scripts/run_study.py`
  now writes these names; the run **folders** keep the reference-matching
  `__v1_*/__v2_*` names, untouched).
- `agents/model_design.py` gained the `HEADLINE_*` fields and an updated status;
  `agents/validation_and_ablation.md` §6 records the headline as pending (per the
  design revision below, the fields are now `None` until the expanded V1 run).

### Step 5 follow-up 2 — V2b: deferred look-ahead tie-break (2026-11-02)

- **User wrote the pre-registered protocol** (renamed 2026-11-02 to
  `agents/validation_and_ablation_extension.md`, the extension to
  `validation_and_ablation.md`): since V1's look-ahead was a 0.002 coin flip
  (ridge_a10 won under both), lock the spec at V1 and defer the look-ahead to the
  ablation window — 2 look-aheads × 7 books on 2020–2022 (14 cells), spec never
  re-searched.
- **Implemented** `phase_v2b()` in `scripts/run_study.py` (`python scripts/run_study.py
  v2b`): runs the absolute12 column (7 new runs, folders `<tag>__v2b_a12_<book>`),
  reuses the byte-identical absolute4 column from the v2 phase, applies the
  pre-registered selection rule (§5.2: incumbent (a4, base) 0.5564; promote iff best
  cell ≥ +0.20 AND not a lone-cell fluke), writes
  `results/_study/validation_part2b_*.json` + `validation_part2b_verdict.json`.
- **Results (absolute12 column, all v2-only — the reference has no a12 2020–2022
  runs):** base 0.4727 / row1 30w 0.5500 / row1 90w 0.4551 / row2 weekly 0.4481 /
  row3 0.4036 / row4 0.4223 / row5 0.2624. absolute12 is worse in **6 of 7 books**;
  the weekly edge **does not survive** under absolute12 (row2 0.6635 → 0.4481,
  Δ −0.215 — the largest cell delta).
- **Verdict:** best cell = a4 row2 weekly 0.6635 < threshold 0.7564 →
  **margin not cleared, nothing promoted; (absolute4, ridge_a10) headline stands,
  now confirmed on the window where the tie is decidable.** Executed 14-cell matrix
  + verdict logged at the end of `agents/validation_and_ablation_extension.md` §8
  (moved from `validation_and_ablation.md` §5.2, which now points forward to the
  extension).
- **Note on the protocol doc:** `validation_and_ablation_extension.md` §5.1 prose
  says "2 of 14 cells already measured … 12 new runs", but its own table (and the
  actual state) show all 7 absolute4 cells measured in v2 — so 7 new runs (the
  absolute12 column) were executed, following the table.

### Step 5 close-out — final decision & carry-forward (2026-11-02)

- **Carry-forward stack (locked, confirmed):**
  `(absolute4, ridge_a10)` + **base book** — monthly rebalance · 60w vol window ·
  vol-target 10% (cap 1.0) · rf sleeve 3% · moomoo 0 bps · 16 tradables.
- **Why base, not weekly:** row 2 (weekly, 0.6635) was the best ablation cell but
  was **not promoted** — it missed the pre-registered margin (threshold 0.7564 =
  incumbent 0.5564 + 0.20) *and* failed the mechanism check (weekly edge did not
  survive under absolute12: 0.4481). Both gates recorded in
  `validation_and_ablation_extension.md` §8.
- **Unseen test (2023–2025): NOT executed / withdrawn.** The single test measurement
  was deferred, and the 2026-11-02 design revision reset the test window — it is
  **untouched and pending** (see the design-revision entry below; the expanded
  experiment runs validation first, then the final test draw).
- **Document structure final state:** `validation_and_ablation.md` (base: split,
  V1/V2 protocols, §5 results, §6 headline — rewritten 2026-11-02 for the expanded
  grid) points **forward** to `validation_and_ablation_extension.md` (now marked
  SUPERSEDED, historical record of the old deferred tie-break); the 2b file was
  renamed to the extension name on 2026-11-02 and all `validation_phase_2b`
  references were cleared.
- **Project status:** Steps 1–5 delivered — dataset (frozen < 2026-01-02) → feature
  engineering → model design → backtest vs reference (byte-identical) → validation/
  ablation (old grid: V1 lock + V2 + V2b). Superseded by the 2026-11-02 design
  revision (below), which re-opens validation on the expanded grid before the final
  test.

### Design revision (2026-11-02) — expanded grid, unseen-test reset, handoff

**User request** (independent of the test result): (1) add **Lasso** → 3 families;
(2) prediction/rebalance frequency {**1w, 4w, 12w**}; (3) look-ahead {**1w, 4w,
12w**} with the rule **look-ahead ≥ frequency** (6 valid pairs: f1×{l1,l4,l12},
f4×{l4,l12}, f12×{l12}); (4) **remove the unseen-test results** and hand the
expanded experiment to a fresh conversation that runs the new validation (V1 joint
selection + V2 ablation) before a final unseen test.

**Design decisions (user-confirmed or adopted):**
- **V1 joint selection:** 24 specs × 6 (frequency, look-ahead) pairs = **144
  candidates**, one selection on 2017–2019 (overlay-free); the winner locks
  **(model, look-ahead, frequency)** together. Frequency is a V1 dimension, not a V2
  knob — this is the only design where a 1-week look-ahead can ever be chosen.
- **Grid:** per-family alpha sets — ridge {1e-3, 1e-2, 1e-1, 1, 10, 100}, lasso and
  enet {1e-4, 1e-3, 1e-2, 1e-1, 1, 10}; enet × l1_ratio {0.5, 0.25} → 6 ridge + 6
  lasso + 12 enet = **24 specs** (user decision 2026-11-02, per-family: the shared
  {1e-3..100} grid put the L1 families above the entry threshold — see the
  smoke-test finding below; the reference's old 1e-4 enet alphas return for L1).
- **1-week label definition (timing note):** the strict H=1 form
  `price[t+1]/price[t+1]−1` is identically zero, so `fwd_ret_1bar` is defined as
  `price[t+2]/price[t+1]−1` — the return realised over the bar *after* the trade,
  exactly what a weekly book earns (see `src/labels.py`).
- **V2 rows (now 4):** vol window 30w/90w, vol-target off, rf 0%, leverage cap 2.0
  + 6.8% fee. The cadence row is removed (frequency is locked in V1).
- **V2b extension superseded:** the deferred look-ahead tie-break
  (`validation_and_ablation_extension.md`) no longer applies — retained as history,
  re-specify before reuse.
- **Cadence naming standardized to n-weeks (2026-11-02 follow-up):** the 4-week
  cadence is now native **`"4w"`** — first bar, then every 4th bar
  (`rebalance_dates` `dates[::4]`, horizon-aligned like `12w`'s `dates[::12]`) —
  replacing the earlier `"monthly"` alias (first bar of each calendar month).
  `FREQ_TO_BARS` = {daily: 1, weekly: 1, 4w: 4, 12w: 12}; CLI `--rebalance`
  choices (weekly, 4w, 12w), default `4w`; portfolio defaults `"4w"`. Historical
  Step-4/5 and V2/V2b runs used the monthly cadence and keep their original naming
  in `results/` and the history log (untouched).

**Engine changes (all in v2, `.venv/bin/python` only):**
- `src/labels.py`: `MODEL_TARGETS`/`MODEL_EMBARGO` now {absolute1, absolute4,
  absolute12}; `make_labels` emits `fwd_ret_1bar` (H=1 note above).
- `src/pipeline.py`: `run_tag` appends `r12w` for the 12w cadence (was weekly-only).
- `scripts/run_backtest.py`: `GRID_ENTRIES` → 24 specs; `FREQ_TO_BARS` +
  `FREQ_LA_PAIRS` (6 valid pairs); `default_config` **raises if look-ahead <
  frequency**; `--horizon` choices (1, 4, 12); `--rebalance` choices
  (weekly, 4w, 12w); docstring updated.
- `scripts/run_study.py`: `phase_v1` = 6-pair joint selection →
  `validation_part1_{laX_freqYw}.json` (6 per-pair files) + `_rankings` (144) +
  `_winner` (locks model + target + rebalance); `phase_v2` reads the locked
  frequency and runs the 4 rows → `validation_part2_{book}.json`; `phase_v2b`
  marked LEGACY (kept, re-specify first).
- `agents/model_design.py`: 3 families, per-family alpha grids (ridge
  {1e-3..100}, lasso/enet {1e-4..10}), targets 1/4/12,
  `FREQ_LA_PAIRS`, `V1_CANDIDATES = 144`, V2 rows 1–4, `HEADLINE_* = None`
  (pending); status rewritten.
- `agents/validation_and_ablation.md`: §3 V1 joint protocol, §4 V2 rows 1–4,
  §5.1 pending record, §5.2 extension legacy, §6 headline pending.
- `agents/validation_and_ablation_extension.md`: SUPERSEDED status block; test
  numbers removed from §7/§8.
- `agents/current_state.md`: this entry; earlier headline/test mentions scrubbed.

**Unseen-test reset (executed):** the v2-only headline run folder under
`results/` (the ridge_a10 2023–2025 run) was **deleted**; all v2 test-window
measurement mentions were removed from the docs (scrubbed across 5 files). The
Step-4 ADR-09 replication folder (`..._grid6_..._2023-2025`) is **kept** as the
reference-comparison record (byte-identical to the reference project; the reference
itself documents that measurement — it is not v2's draw). The test window is now
**untouched**: no 2023–2025 measurement exists in v2's record.

**Handoff to the fresh conversation (what to run next):**
1. `.venv/bin/python scripts/run_study.py v1` → 144-candidate joint selection →
   `validation_part1_winner.json` locks (model, look-ahead, frequency).
2. `.venv/bin/python scripts/run_study.py v2` → 4-row ablation →
   `validation_part2_*.json`; decide promotions per the pre-registered rules
   (nothing is promoted from the table without an ADR).
3. Log results in `validation_and_ablation.md` §5.1, then the promoted stack is
   measured **once** on 2023–2025 as the final unseen test (update
   `model_design.py` `HEADLINE_*`).

**Notes / open items**

- The **headline** is **pending** — it will be the V1-locked (model, look-ahead,
  frequency) triple with the base book after the expanded validation; the test
  window is untouched.
- The `fwd_ret_1bar` label tail: the runnable panel is shared across look-aheads
  and still ends 2025-10-03 (the 12-bar label is the longest), so the a1/a4 test
  windows share that effective end; benchmark gate bars still run to 2025-12-31.
- `scripts/run_study.py` uses the frozen panel (rebuilt from raw CSVs each run), so
  re-running reproduces the numbers deterministically.
- Project state as of this entry: **validation re-open** — the expanded experiment
  runs V1 (144) then V2 (4 rows) before any 2023–2025 activity.

**Smoke-test finding (2026-11-02) — L1 alphas sit above the entry threshold.** On
the 2011–2016 training window the max per-feature signal in lasso-alpha units
(|corr| · std(target)) is **0.00125 (1w) / 0.00315 (4w) / 0.00789 (12w)**. With the
shared grid {1e-3 … 100}, only α=0.001 is below the L1 entry threshold, so lasso
and enet at α ∈ {0.01, 0.1, 1, 10, 100} fit intercept-only models → flagged
degenerate → excluded by the pre-registered rule (selection proceeds on the rest;
the 1w pair stays viable because ridge is dense). Verified end to end:
- `walk_forward_predict` (hist ≤2016-12-31, val 2017–2019, target `fwd_ret_1bar`):
  lasso_a0.01 / lasso_a1 / enet_a0.01_r0.5 → degenerate=True (per-date cross-ticker
  pred std = 0, preds = intercept only); ridge_a1 → degenerate=False.
- Direct fits on the same window: lasso_a0.01 → 0/21 nonzero coefficients,
  lasso_a0.001 → 1/21, ridge_a1 → 21/21 (consistent with the KKT entry rule).
- Label alignment is correct: corr(`fwd_ret_1bar`[t], `fwd_ret`[t+1]) = 1.0 per
  ticker; features do carry signal (18/21 features have |corr| > 0.01 vs the 1w
  label). This is a genuine data property (weekly momentum does not predict the
  next week), not an editing bug — the smoke runs that hit the degenerate guard
  raised the pre-registered ValueError as designed.
- **Decision (user, 2026-11-02):** per-family grids — ridge {1e-3 … 100}, lasso &
  enet {1e-4 … 10} — keeping 24 specs / 144 candidates while restoring the L1
  families to their active range (implemented in `scripts/run_backtest.py`,
  `agents/model_design.py`, and §3 of `agents/validation_and_ablation.md`; the old
  1e-4 enet alphas are back for L1 only, exactly the reference's active range).
  Verified after the change: lasso_a0.0001 is non-degenerate on the 1-week target
  (coefficients enter below the 0.00125 threshold); spec enumeration still yields
  the same 24 names with the new alpha values.

---

## Step 6 — RFF ridge family (2026-11-02)

### Goal

Add a fourth model family — **RFF ridge** (Random Fourier Features → Ridge, a
Monte-Carlo approximation of Gaussian-kernel ridge) — as the design's nonlinear
candidate, with the hyperparameter grid sanity-checked against the existing
per-family configs and the standardized 21-feature panel.

### Hyperparameter assessment (user proposal vs edited grid)

User proposal: α = {1e-4, 1e-3, 1e-1, 1, 10, 100} · D = {10, 100, 500} ·
γ = {1e-4, 1e-3, 1e-1, 1, 10, 100}. Edited to:

| Dim | User proposal | Edited (implemented) | Why |
| --- | --- | --- | --- |
| α (regularization) | {1e-4, 1e-3, 1e-1, 1, 10, 100} | **{1e-3, 1e-2, 1e-1, 1, 10, 100}** = ridge grid | RFF ridge is ridge on transformed features — same L2 range as the existing ridge family. The proposal skipped 0.01 (likely mid-range) and added 1e-4 (near-interpolating with D=500). |
| D (random features) | {10, 100, 500} | **{50, 100, 500}** | D must exceed p = 21. D=10 is a bottleneck projection: ~1/√D ≈ 32% kernel-approximation error, and (α, γ) selection would be confounded by the W draw. 50 is the low anchor > p. |
| γ (kernel bandwidth) | {1e-4, 1e-3, 1e-1, 1, 10, 100} | **{1e-3, 1e-2, 5e-2, 1e-1, 5e-1}** | γ is the RBF **inverse bandwidth** (k = exp(−γ‖x−y‖²); σ = 1/√(2γ)) — not the lengthscale. Features are standardized (p = 21), typical pair distance² ≈ 2p ≈ 42 → median-heuristic γ ≈ 0.01. γ ≥ 1 collapses the kernel to a point mass and aliases the RFF frequencies to noise; γ = 1e-4 is a near-constant kernel (≈ linear). |

- **Grid size:** 6 α × 3 D × 5 γ = **90 RFF specs** → total grid **114 specs**
  (24 linear + 90 RFF) → V1 candidates = 114 × 6 (freq, la) pairs = **684**.
- **Determinism:** `RBFSampler(random_state=DEFAULT_SEED=5205)` (sklearn ≥ 1.6
  name for the RBF Random Fourier Features sampler) inside `ModelSpec.build` —
  every quarterly refit samples the same projection W, so predictions are
  reproducible and the kernel approximation is time-consistent.

### Implementation (all in v2, `.venv/bin/python` only)

- `src/models.py`: `ModelSpec` gains `rff_dim` / `gamma` fields and an `"rff"`
  family (`FAMILY_PENALTIES["rff"]="rff"`); `for_family` names specs
  `rff_a<alpha>_d<dim>_g<gamma>`; `build()` returns
  `scaler → RandomFourierFeatures(D, γ, seed) → Ridge(α)`; `specs_from_config`
  expands α × D × γ for `rff` entries (requires `rff_dim` + `gamma`).
- `src/pipeline.py`: `evaluate_all` writes a summary `coefficients` table for RFF
  (coefs live in the D-dim feature space, not the 21 inputs; family/α/D/γ +
  coef stats); `run_parameters` records `rff_dim` / `gamma` per spec.
- `scripts/run_backtest.py`: `RFF_ALPHAS / RFF_DIMS / RFF_GAMMAS` + the rff entry
  in `GRID_ENTRIES` (24 + 90 = 114 specs); `--model` choices include rff names.
- `scripts/run_study.py`: `spec_dict` parses `rff_a<α>_d<D>_g<γ>` names.
- Docs: `agents/model_design.py` (4 families, RFF grid + rationale,
  `V1_CANDIDATES = 684`), `agents/validation_and_ablation.md` (§3 grid rows 3–6,
  684 candidates), `README.md` counts.

### Verified (smoke tests)

- Spec enumeration: **114 specs** (24 linear + 90 rff); names unique, rff names
  parse back via `spec_dict`.
- Walk-forward determinism: two runs of
  `walk_forward_predict` (train ≤2016-12-31, val 2017–2019, target `fwd_ret_4bar`)
  for `rff_a0.01_d100_g0.01` → **byte-identical preds** (same seeded projection
  across refits).
- Non-degeneracy across the grid corners on the same window: γ = 1e-3 (smooth)
  and γ = 0.5 (local) at α = 0.01, D = 50/500 → all non-degenerate
  (per-date cross-ticker pred std > 1e-12); α = 100 still yields varied preds.
  No RFF spec collapses to the intercept-only degenerate case the L1 families hit.
- End-to-end pinned run
  `.venv/bin/python scripts/run_backtest.py --model rff_a0.01_d100_g0.01
  --horizon 4 --rebalance 4w --test-start 2020-01-01 --test-end 2022-12-31`
  (out to /tmp, not results/): selection → val ranking (1 spec) → walk-forward
  test → book vs SPY/EW-16 → all 11 tables written, incl. the RFF summary
  `coefficients.csv`; `run_config.yaml` records the rff spec fields.

### Notes / open items

- The full V1 (684 candidates) / V2 re-run is **NOT executed** — the study is
  re-run from scratch by the expanded experiment before any 2023–2025 activity
  (unchanged handoff order: `.venv/bin/python scripts/run_study.py v1` then `v2`).
- RFF runs about 4.75× the linear-grid walk-forward cost (3 D × 5 γ dims); the
  per-fit cost of D=500 is small on the 5y pooled panel, so the study's runtime
  increase is modest relative to the 144-candidate baseline.
- The test window 2023–2025 remains **untouched**; `HEADLINE_*` still `None`.

### Step 6 follow-up — V1 executed (2026-11-02)

- Ran `.venv/bin/python scripts/run_study.py v1` → **684-candidate joint selection**
  on 2017–2019 (6 pairs × 114 specs), overlay-free base book, degenerate excluded.
  All 6 per-pair summaries + merged 684-row ranking + winner saved under
  `results/_study/` (`validation_part1_{laX_freqYw}.json`,
  `validation_part1_rankings.json`, `validation_part1_winner.json`); run folders
  `results/<tag>__v1_<slug>/`. Runtime ≈ 7 min.
- **Degenerates by family:** lasso 24/36, enet 40/72, ridge 0/36, rff 0/540 —
  L1 collapses concentrate on the 1-week target (the known entry-threshold
  finding); RFF never collapses on the anchored grid.
- **Per-pair winners:** weekly×1w enet_a0.001_r0.5 **1.1950** · weekly×4w
  rff_a1_d500_g0.01 1.1476 · weekly×12w rff_a0.1_d50_g0.001 1.0524 · 4w×4w
  rff_a10_d100_g0.05 **1.2218** · 4w×12w rff_a100_d100_g0.001 0.9955 · 12w×12w
  rff_a100_d100_g0.001 0.9909.
- **Locked winner (joint):** **(enet_a0.001_r0.5, absolute1, weekly)**, val net
  Sharpe 1.1950 (gross 1.2331, TO 0.1475). RFF won 5/6 pairs but its best
  (rff_a10_d50_g0.001, 1.0221, weekly×1w) trails the enet lock on the winning
  pair; RFF edges sit at D=50 + low γ (smooth kernels), consistent with the
  anchored grid dropping the high-γ tail.
- Recorded in `agents/validation_and_ablation.md` §5.1; `agents/model_design.py`
  §6 measured-record line updated. **V2 ablation (rows 1–4, 2020–2022, locked
  model) is the next handoff step**; the 2023–2025 test window stays untouched
  until V2 + ADR.

### Step 6 follow-up 2 — refit cadence becomes a V1 hyperparameter (2026-11-02)

- User request: make the model refit cadence (was fixed at quarterly QE) a
  hyperparameter/config — weekly / monthly / quarterly — and asked whether the
  validity rule "refit ≥ prediction" holds (analogous to look-ahead ≥ frequency).
- **Adopted rule (implemented in `default_config`, `run_backtest.py`):** refit is
  **no more frequent than the acting cadence** — `REFIT_TO_BARS[refit] >=
  FREQ_TO_BARS[rebalance]` (W-FRI=1, ME=4, QE=12 bars vs cadence 1/4/12). Framing:
  this is an **economy rule, NOT statistical validity** like look-ahead ≥ frequency
  (the label must span the holding period). Refits finer than the cadence fit
  models whose intermediate predictions the book never uses; refit == cadence
  captures the freshness. QE stays valid for every cadence, so the previously
  executed V1 config remains admissible — it just gets re-measured.
- **New V1 grid:** 3 refits × valid (freq, la, refit) combos = **14 combos**
  (W-FRI × weekly pairs (3); ME × {1w, 4w cadences} (5); QE × all 6) × 114 specs =
  **1596 candidates**. `REFIT_OPTIONS`/`REFIT_TO_BARS` added to run_backtest.py +
  model_design.py; `run_study.py` phase_v1 loops `V1_COMBOS`, slugs are
  `laX_freqYw_<refit-slug>` (`REFIT_SLUGS`: wkrefit/mrefit/qrefit); the merged
  ranking and `validation_part1_winner.json` gain a `refit` field; phase_v2 feeds
  `refit_every=locked_refit` into every ablation row.
- **Consequence:** the executed 684-candidate V1 (all QE, lock
  enet_a0.001_r0.5/absolute1/weekly @ 1.1950) is **superseded** as a *selection*
  (kept as the §5.1 breadcrumb) — V1 must be **re-run on 1596 candidates** before
  V2. The 2023–2025 test window remains untouched. Est. runtime ≈ 3× the 684 run
  (W-FRI/ME cells refit more often; QE cells identical to before).

### Step 6 follow-up 3 — V1 re-run executed (1596 candidates, 2026-11-02)

- Ran `.venv/bin/python scripts/run_study.py v1` → **1596-candidate joint
  selection** (114 specs × 14 (freq, la, refit) combos, 2017–2019 overlay-free).
  14 per-combo summaries (`validation_part1_laX_freqYw_<refit>.json`) + merged
  1596-row ranking + winner under `results/_study/`. Runtime ≈ 30 min
  (log `/tmp/v1_study_refit.log`).
- **Locked winner: (enet_a0.001_r0.5, absolute1, weekly, ME)** — val net Sharpe
  **1.3674** (gross 1.3951, TO 0.1072). Refit effect: monthly > weekly > quarterly
  on the winning cell (1.3674 > 1.2636 > 1.1950) — the new dimension pays.
- **Degenerates:** lasso 55/84, enet 95/168, ridge 0/84, rff 0/1260; every
  NaN-Sharpe row (150) is also degenerate-flagged.
- **Per-combo winners:** RFF won 8/14 cells (best rff_a10_d100_g0.05, 4w×4w, QE,
  1.2218) but the enet lock (1.3674) is global. RFF needs larger D at weekly
  cadence (d500/d100) vs quarterly.
- **Process bug caught + fixed:** the first merge mis-ranked rows because
  NaN-Sharpe rows are not caught by `net is not None` (NaN comparisons break the
  sort). Fixed `scripts/run_study.py` (NaN → null, sort-safe, live excludes null)
  and rebuilt `validation_part1_rankings.json` / `validation_part1_winner.json`
  from the saved per-combo files — deterministic, no re-run needed. Verified:
  14/14 cell winners match the merged ranking; ranking monotone descending.
- Recorded in `agents/validation_and_ablation.md` §5.1 + `agents/model_design.py`
  §6. **Next handoff: Validation 2** (4 ablation rows, 2020–2022, locked model +
  ME refit). Test window 2023–2025 untouched.

### Step 6 follow-up 4 — Validation 2 executed (2026-11-02)

- Ran `.venv/bin/python scripts/run_study.py v2` → 6 single-cell runs on
  2020-01-01..2022-12-31 with the locked **(enet_a0.001_r0.5, absolute1, weekly,
  ME refit)** (confirmed by run tags `..._mrefit_enet_a0.001_r0.5_..._rweekly_2020-2022`).
  Log `/tmp/v2_study.log`; summaries `results/_study/validation_part2_{row}.json`
  (overwrote the pre-revision grid's stale part2 files).
- **Net Sharpe on the ablation window (for ablation comparison only):** base 60w/10%/3%/cap1.0
  **0.6274** · row1_30w 0.5751 · row1_90w 0.6077 · row2_notarget 0.5096 ·
  row3_rf0 0.5748 · row4_lev2 (cap 2.0 + 6.8% fee) 0.6273. All rows pass the
  pipeline gate (bar 0.4459 / −24.83%).
- **Verdict: base book stands** — no row beats it (row4_lev2 is identical: the
  leverage cap minus the 6.8% fee nets zero; vol-target off is the clear worst).
  Promoted stack = V1 lock + base book (60w, vt 10%, rf 3%, cap 1.0, moomoo, 0bps).
- Recorded in `agents/validation_and_ablation.md` §5.3 + `model_design.py` §6.
  **Next: pre-register the promotion ADR, then ONE 2023–2025 test draw. The test
  window remains untouched (no draw recorded).**

### Step 6 follow-up 5 — rff_dim = 1000 added (2026-11-02)

- User request: extend the RFF grid's random-feature count by D = 1000 for
  completeness, then re-run the hyperparameter permutations.
- `RFF_DIMS = (50, 100, 500, 1000)` in `scripts/run_backtest.py` +
  `agents/model_design.py`; GRID_ENTRIES expands automatically →
  **144 specs** (24 linear + 120 rff: 6 α × 4 D × 5 γ) and **V1 = 144 × 14 =
  2016 candidates**. Counts updated in run_backtest.py / run_study.py docstrings,
  model_design.py, README, validation_and_ablation.md (row 5 + §3 + §6).
- **Consequence:** the just-executed 114-grid results (V1 lock
  enet_a0.001_r0.5/absolute1/weekly/ME @ 1.3674; V2 base 0.6274) are **superseded
  as selections** (kept as §5.1/§5.3 breadcrumbs) — **V1 re-runs on 2016
  candidates**, then V2 re-runs on whatever locks. Test window 2023–2025 untouched.

### Step 6 follow-up 6 — V2 vol-knob rows expanded (2026-11-02, while 144-grid V1 runs)

- User request: expand V2's vol-target values to {None, 10%, 20%} and overwrite the
  vol-window row values to {4, 8, 12, 24, 48} weeks. Also asked whether a
  "valid_volatility" rule exists (vol window vs prediction frequency).
- **Answer (design): no hard rule.** look-ahead >= frequency and refit >= cadence are
  hard ValueError rules because they constrain *forward-looking* structure (what the
  model predicts vs how often you trade/refit). The vol window is a backward-looking
  estimator — any window >= 2 bars is structurally valid, so no ValueError is added.
  Documented soft minimum in validation_and_ablation.md §4 row 1: window should span
  at least one full holding period (>= rebalance cadence); with weekly bars a 4w
  window is ~3 dof (very noisy) — the sweep measures exactly that.
- **Implemented:** `run_backtest.py` `default_config` now takes `vol_target: bool |
  float | None` (`True` = 10%, float passes through, falsy = off) and clamps
  `vol_min_periods = min(20, vol_window)` so 4w/8w/12w windows satisfy the
  `2 <= min_periods <= window` validation. `run_study.py` STUDY_ROWS = base
  (volwin 24w/10% after follow-up 7, vt 10%) + volwin_4w/8w/12w/24w/48w +
  vt_none + vt_20 + row3_rf0 + row4_lev2 + lev3 (11 rows; old row1_30w/row1_90w/
  row2_notarget names retired — their part2 JSONs remain as stale breadcrumbs until
  the new run).
- **Status:** V2 rows ready; V2 runs after the 144-grid V1 lock lands. Test window
  untouched.

### Step 6 follow-up 7 — base book re-based to new vol defaults (2026-11-02)

- User decision: overwrite the base defaults with the new value sets and re-run
  Validation 1 on them ("rebase to use the new defaults for validation phase 1").
- **New defaults:** `default_config` vol_window **60 -> 24w** (src/portfolio.py
  dataclass defaults too); vol_target stays **10%** (`True`). Docstrings + §4 +
  model_design.py + STUDY_ROWS base note updated; volwin_24w row now duplicates
  base (kept as a consistency check).
- **Why it changes V1:** verified in code — the V1-scored book uses
  `historical_forecast_vol` (rolling std of past predictions over `vol_window`,
  portfolio.py:169-184) inside the base strategy, i.e. even with
  `apply_vol_target=False` (select_model, pipeline.py:146). So the window default
  affects every spec's ranking; `vol_target` does NOT (overlay off in selection).
- **Status:** user re-runs V1 on the 24w default; V2 follows on the new lock;
  test window untouched.

### Step 6 follow-up 8 — leverage sweep widened (2026-11-02)

- User request: leverage values {1, 2, 3} — add a **cap 3.0** row.
- Added `lev3` (vol_cap=3.0, lev_fee=0.068 — same borrow fee as lev2) to STUDY_ROWS
  (11 rows now); docstrings + validation_and_ablation.md §4 row 4 + model_design.py
  §7 updated. Leverage sweep: {1.0 = base, 2.0, 3.0}.
- Note: cap/rf are overlay-only, so V1 selection is unaffected — no V1 rerun cost.
  Test window untouched.

### Step 6 follow-up 9 — 144-grid V1 executed (2026-11-02, 24w base)

- User ran `run_study.py v1` on the 144-spec grid / 2016 candidates / 24w base book
  (files stamped 15:59–16:30; tag `..._grid144_...`).
- **Verification (clean):** 2016 rows, monotonic non-increasing, 150 degenerate
  rows stored as null, 14/14 per-combo cell winners match the merged ranking,
  degenerate pattern unchanged (lasso 55/84, enet 95/168, ridge 0/84, rff 0/1680).
- **Lock: (enet_a0.001_r0.5, absolute1, weekly, ME refit)** @ **1.3370** (gross
  1.3660, TO 0.1053) — same stack as the 114-grid lock (1.3674); the 60w→24w rebase
  re-scores the book (slightly lower). Lock stability across the D=1000 extension
  AND the window rebase is a robustness signal.
- **D=1000:** 420 rows; best rank #4 (rff_a1_d1000_g0.01, absolute4/weekly, ME
  1.2487) and wins two 4w-la weekly cells (ME/QE), beating the 114-grid d500/d100
  winners there; never threatens the enet 1w lock.
- Recorded in validation_and_ablation.md §5.1 (current record; 114-grid kept as
  breadcrumb). **Next: V2 — 11 rows on 2020–2022 with the new lock.** Test window
  untouched.

### Step 6 follow-up 10 — Validation 2 executed + promotion ADR (2026-11-02)

**V2 (11 rows, 2020–2022, new 144-grid lock enet_a0.001_r0.5/absolute1/weekly/ME):**
`run_study.py v2` at 17:01; summaries `validation_part2_{row}.json` (new rows;
old row1_30w/row1_90w/row2_notarget part2 files remain as stale breadcrumbs).

| row | 2020–22 net Sharpe | total | maxDD | notes |
| --- | --- | --- | --- | --- |
| base (24w/10%/3%/cap1) | 0.5180 | +18.8% | −20.9% | default book |
| **volwin_12w** | **0.6574** | **+28.0%** | −20.7% | **winner** (V1 echo 1.5604 > base 1.3957) |
| volwin_48w | 0.6021 | +21.3% | −18.5% | runner-up |
| volwin_24w | 0.5180 | — | — | == base (consistency check PASSED, bit-identical) |
| row4_lev2 / lev3 | 0.5177 | — | — | == base: leverage caps **never bind** in 2020–22 (realized vol > 10% target throughout) |
| vt_none | 0.4855 | +17.9% | −19.8% | vol target off loses |
| row3_rf0 | 0.4836 | +17.3% | — | rf 0% loses (identical trades, lost carry) |
| vt_20 | 0.4558 | +16.6% | — | 20% target loses |
| volwin_4w / 8w | 0.4058 / 0.2437 | — | — | short windows noisy (non-monotonic) |

**ADR — promotion (recorded; the authoritative selection record):**
- **Promoted stack:** model enet_a0.001_r0.5 · target absolute1 (1w look-ahead) ·
  weekly rebalance · **ME (monthly) refit** · **vol_window 12w** (V2 winner; default
  stays 24w) · vol_target 10% · rf 3% · cap 1.0 (no leverage) · moomoo costs.
- Evidence: V1 joint selection (2016 candidates, 2017–19) — enet lock 1.3370,
  stable across the D=1000 extension and 60w→24w rebase; refit ME > W-FRI (1.3370
  vs 1.3300, gross identical — pure cost) > QE (1.2056). V2 (2020–22) — 12w window
  wins on both the ablation window (0.6574 vs 0.5180 base) and the V1-window echo
  (1.5604 vs 1.3957).
- Caveats recorded: vol-window response non-monotonic (8w worst, 0.24) — 12w edge
  may be partly window-specific, but it is supported by BOTH windows; leverage
  caps 2.0/3.0 non-binding in 2020–22 (high-vol regime) — untested when vol <
  target for sustained periods.
- **Test draw (the single pre-registered 2023–2025 measurement):** run in this
  turn via `run_backtest.py` single mode with the promoted config (folder
  `results/test_2023-2025_promoted`); results appended in follow-up 11. The test
  window had never been touched before this draw.

### Step 6 follow-up 11 — TEST draw executed: gate FAIL (2026-11-02)

- **The single pre-registered 2023–2025 draw** ran via `run_backtest.py` single mode
  with the promoted stack (enet/absolute1/weekly/ME, vol_window 12w, vt 10%, rf 3%,
  cap 1.0, moomoo); folder `results/test_2023-2025_promoted`, tag
  `weekly_fg_roll5y_val3y_mrefit_enet_a0.001_r0.5_vt0.1c1rf3_rweekly_2023-2025`.
- **Test metrics (net):** Sharpe **1.1716** · maxDD **−11.95%** · total **+36.70%** ·
  ann ret 11.95% / ann vol 10.08% · hit 63.2%. Benchmarks: EW-16 +36.53% (Sharpe
  1.2502), SPY B&H +74.53% (maxDD −16.88%). Forecast: OOS R² 1.65%, rank-IC 0.0665
  (t 2.47). CAPM: **alpha ≈ 0.0001 (t ≈ 0.00), beta 0.556**.
- **Verdict (§3.2 gate): FAIL — SUCCESS=False** (sharpe_ok=False: 1.1716 < 1.2502
  EW bar; max_drawdown_ok=False: −11.95% < −7.79% bar = EW-16's less-negative
  drawdown (book deeper than EW, shallower than SPY's −16.88%)). The book beats no
  benchmark on the final window; returns are essentially market beta.
- **Protocol outcome (rule 1.2.4): null result — completed experiment, not a
  promotion.** Promoted stack NOT confirmed out-of-sample. Test window now
  consumed (one draw, pre-registered); no further draws; any future iteration =
  new pre-registered cycle. Recorded in validation_and_ablation.md §5.4.

### Step 6 follow-up 12 — agents/ cleanup: stale docs fixed, extension file removed (2026-11-02)

- **Audit result:** validation_and_ablation.md §5.3 was still the 114-grid V2 record
  ("base stands", 60w) and §6 still said "pending / test untouched" — both rewritten
  to the current record: §5.3 = 11-row V2 (volwin_12w winner 0.6574, base 0.5180,
  114-grid as breadcrumb), §5.4 = test draw (gate FAIL, null), §6 = closed cycle
  (nothing promoted, HEADLINE_* stays None). §5.1 "Next" line updated. model_design.py
  header + §6 record + HEADLINE comments updated; README status paragraph updated
  (the `--vol-window` CLI default 60→24 was already fixed in the earlier V2 work).
- **Removed `agents/validation_and_ablation_extension.md`** (pure history of the
  superseded v2b design; no active configs; not in git, so its essential v2b
  verdict was archived into validation_and_ablation.md §5.2 before deletion; full
  14-cell matrix is recoverable from the earlier trace if ever needed).
- **Not stale (kept as history by design):** current_state.md Step 1–5 entries and
  follow-ups 1–8 (chronological log; each later follow-up supersedes); the
  114-grid/684/1596 records in validation_and_ablation.md §5.1/§5.3 are explicitly
  labeled breadcrumbs; feature_engineering.md has no stale references.

### Step 6 follow-up 13 — results cleanup + GitHub prep (2026-11-02)

- **Removed superseded result artifacts** (kept only current records): stale
  `results/_study/` JSONs (684-era part1 without refit suffix, 114-grid part2
  rows, LEGACY `validation_part2b_*` + verdict) and superseded run folders
  (`*grid114*`, `*ridge_a10*` v2/v2b, `*grid6*` incl. the withdrawn old 2023–2025
  measurement, plus non-current `__v2_row1_30w/row1_90w/row2_notarget`). Kept:
  `validation_part1_winner.json` (current lock), `rankings.json`, 14 grid144 V1
  folders, 11 current V2 folders, `test_2023-2025_promoted/`.
- **`.gitignore` extended** for GitHub: added `.DS_Store` (removed the stray
  `agents/.DS_Store`), `*.pyc`/`*.pyo`, `.env*`, tool caches, `*.log`; existing
  `.venv/`, `data/raw|features/`, `config.yaml`, `results/` already covered.
- **README verified** GitHub-ready: title, dataset conventions, layout, setup
  (`uv sync`), run commands; status paragraph reflects the closed validation
  cycle; `data/README.md` reference resolves.

---

## Step 7 — Report figure suite, generated inside the experiment scripts (2026-11-02)

### Goal (user request)

Include plots for each best-selected run's metrics **throughout time** (V1 / V2
validation windows and the test window), and generate them **inside the
experiment scripts** so no second "plot image" script has to be run afterwards.

### What was added

- **`src/plots.py`** — the 10-figure report suite, drawn from objects the
  pipeline already computes (deterministic, Agg backend, fixed style): cumulative
  net return vs Equal-weight/SPY, gross vs net (drag shaded), drawdown
  (strategy + benchmarks), cumulative transaction cost + turnover with net-buy /
  net-sell markers and long/short shading, gross/net exposure + vol-target scale
  + leverage-blocked markers, per-ticker weights held (stacked area) + weight
  heatmap, holdings counts, calendar-year returns, OOS R² by year. Every PNG is
  self-contained: header (model · window dates) + subtitle (Sharpe / maxDD /
  total / ann ret / ann vol + run tag).
- **`src/portfolio.py`** — `build_forecast_vol_portfolio` / `build_trend_portfolio`
  now attach `weights_wide` / `orders_wide` (per-date × per-ticker weights held
  and signed order USD) to the book via `object.__setattr__`.
- **`src/pipeline.py`** — `select_model` returns the winner's validation-window
  preds + books; `run_pipeline` carries `val_portfolio` / `val_preds` /
  `val_benchmarks`; `evaluate_all` writes `series/` (book / weights / orders /
  benchmarks) + renders `plots/` on every run, then **auto-publishes** the
  phase's 10 figures to `results/plots/<phase>/` (determined from the run's test
  window by `report_phase_for` / `REPORT_PHASES`) and **removes the transient
  per-run `plots/`** — so no flag, no CLI change, and no stale figures in
  individual config folders.
- **`agents/plots.md`** — figure list, why each belongs in the report, the
  two-window convention, and how to regenerate.

### Verification & backfill

- Smoke run to `/tmp`: all 20 PNGs + 8 series CSVs produced; test metrics still
  reproduce (alpha ≈ 0.0001, beta 0.556).
- Backfilled (deterministic re-run into the same folders; existing tables and
  `_study` JSONs verified **byte-identical** — md5 diff): `test_2023-2025_promoted`
  (test 2023–2025, volwin 12w), the V2 winner row `__v2_volwin_12w` (test
  2020–2022, net Sharpe 0.657), and the V1 winner combo folder
  (`...__v1_la1_freq1w_mrefit`), whose `validation_*` figures cover the 2017–2019
  selection window (net Sharpe 1.337, matching the lock).
- **Scope per owner (final, 2026-11-02):** the canonical figures live **only**
  under `results/plots/validation_phase_1/`, `validation_phase_2/`, `test/` (10
  PNGs each, window prefix stripped, auto-published); per-run `plots/` are
  transient and removed; the suite is kept for the three best-selected runs only,
  and the other 10 V2 rows keep tables only (their stale `plots/`/`series/` were
  deleted earlier). Series stay per run folder under `series/`. Re-running the
  three designated commands (no flag) regenerates identical figures + tables.

### Notes

- `validation_*` figures always show the overlay-free book (selection basis);
  `test_*` figures show the vol-targeted book actually traded.
- `results/` stays git-ignored; the added code is `src/plots.py` + edits to
  `src/portfolio.py`, `src/pipeline.py`, `README.md`, `agents/plots.md`,
  `agents/current_state.md` (this entry).
