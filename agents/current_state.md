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
  selection = best overlay-free validation net Sharpe with
  degenerate exclusion) match the reference implementation.

Conclusion: **doc is current, not stale**; copied as v2-owned
`agents/model_design.py` refreshed only for v2 conventions — fixed window wording
(end inclusive 2026-01-01 / fetch exclusive 2026-01-02; equivalent to the doc's
"no 2026 data" cut), own file paths/lineage, and a status note recording v2's first
measured run.

### Implementation (written fresh, no stale imports)

- `src/models.py` — `ModelSpec` (ridge/lasso/enet builders, `DEFAULT_SEED`),
  `specs_from_config` (expands alpha / l1_ratio ranges into the 6-spec grid),
  `walk_forward_predict` (grouped by `refit_every`, past dates minus embargo,
  optional rolling window, fresh fit per block).
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
  `comparison_table`, `spy_buy_and_hold`, `equal_weight_universe`,
  `period_stats`, `oos_r2`, `forecast_metrics`, `oos_r2_by_period`.
- `src/pipeline.py` — `PipelineConfig`, `run_tag` (folders like
  `weekly_fg_roll5y_val3y_qrefit_grid6_vt0.1c1rf3_2023-2025`), `select_model`
  (validation net Sharpe ranking, degenerate exclusion), `run_pipeline` (select →
  walk-forward test → book + benchmarks → verdict),
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
  oos_r2_by_year / coefficients / performance_by_period
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
- [x] Rule path CLI + baseline measurement — `scripts/run_trend_baseline.py`,
      results in `results/trend_baseline/` (Step 23). Rule-path figures and a
      promoted-ADR re-measurement remain follow-ups.

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
- **Determinism:** `RBFSampler(random_state=DEFAULT_SEED)` (sklearn ≥ 1.6
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
  (t 2.47).
- **Verdict (§3.2 gate): FAIL — SUCCESS=False** (sharpe_ok=False: 1.1716 < 1.2502
  EW bar; max_drawdown_ok=False: −11.95% < −7.79% bar = EW-16's less-negative
  drawdown (book deeper than EW, shallower than SPY's −16.88%)). The book beats no
  benchmark on the final window; returns are essentially market exposure.
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

## Step 7 — Label source and horizon correction (2026-11-02)

- `build_feature_frame` now emits only features. `scripts/build_features.py` selects
  the requested `fwd_ret_{H}bar` from `make_labels()` and joins it as
  `label_forward_return_{H}bar`; the backtest also gets its labels from
  `make_labels()`.
- Corrected the 4-/12-bar targets to cover the full holding periods after entry:
  `price[t+5]/price[t+1]-1` and `price[t+13]/price[t+1]-1`. `make_labels()` retains
  rows with unrealized targets as NaN; consumers choose when to drop them.
- Added tests for target endpoints, trailing NaNs, and the feature-only builder.
- **Impact on results:** existing validation and test artifacts were generated under
  the shorter 4-/12-bar targets, so they are archived history, not current model
  selection evidence. Rebuild features and rerun V1/V2. The 2023–2025 test was
  already measured, so any corrected run on it is a remeasurement, not a new
  untouched holdout. `HEADLINE_*` remains `None`.

---

## Step 8 — Report figure suite, generated inside the experiment scripts (2026-11-02)

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
  reproduce.
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

---

## Step 9 — CAPM and AR(1) naive baseline removed (2026-11-03)

### Goal (user request)

Remove the CAPM regression output (`capm.csv`) and the AR(1) naive forecast
baseline as if they were never implemented — including their source, generated
artifacts, and doc references — without touching the actual benchmarks
(EW-16, SPY B&H), the success gate, or the model's own OOS R² / rank-IC metrics.

### What was removed (source)

- `src/metrics.py` — `capm_alpha` (OLS of book net returns on SPY B&H) and
  `oos_r2_vs_baseline` (the AR(1)-relative R²) deleted.
- `src/models.py` — `ar1_walk_forward` deleted (per-ticker AR(1) on the lagged
  target); the now-unused `numpy` import and `target_horizon` import removed.
- `src/pipeline.py` — `naive = ar1_walk_forward(...)` block, `naive` key in the
  results dict, `oos_r2_vs_ar1` in `forecast.csv`, `naive_forecast.csv`, and the
  `capm.csv` table all removed from `evaluate_all`; imports and docstrings updated.
- `scripts/run_backtest.py` — the `== CAPM ==` console print removed.

### What was removed (generated artifacts, results/)

- Deleted all 26 `capm.csv` and 26 `naive_forecast.csv` files (one per run folder).
- Stripped the `oos_r2_vs_ar1` row from all 26 `forecast.csv` files, so the run
  tables match what the current code produces.

### Docs scrubbed

- `agents/model_design.py` — AR(1) mention removed from §6 selection rule.
- `agents/validation_and_ablation.md` — AR(1) baseline sentence (§3 mechanics),
  the test-draw CAPM numbers, and the "(alpha ≈ 0)" verdict phrasing replaced
  with benchmark-grounded wording.
- `agents/current_state.md` — earlier entries' implementation lists (Step 4) and
  the test-record CAPM numbers scrubbed; "market beta" verdict phrasing
  reworded to "market exposure".

### Verified

- `py_compile` passes on all four edited source files.
- `pytest` passes; `grep` over `src/ scripts/ tests/` shows no remaining
  capm/ar1/naive references.
- Smoke run of `scripts/run_backtest.py` (single spec, out to `/tmp`) completes
  and writes no `capm.csv` / `naive_forecast.csv`; `forecast.csv` contains only
  the model's own forecast metrics.
- The pipeline's remaining tables (`comparison`, `strategy`, `oos_r2_by_year`,
  `coefficients`, `performance_by_period`) are unchanged by this cleanup — the
  removed code only wrote the deleted outputs.

---

## Step 10 — OOS R² benchmarked vs SPY / EW-16, merged into the test run (2026-11-03)

### Goal (user request)

The test-phase OOS R² of the promoted model/config should be benchmarked against
SPY and the Equal-weight-16 ETF portfolio — same statistic, same panel — and the
benchmark computation should live in the script where the model executes on the
test set (one fresh run, no backward compatibility).

### What was added

- `src/metrics.py` — `oos_r2_benchmarks(preds, benchmark_close, target_col)`:
  the **model's OOS R² relative to each benchmark forecast** —
  `R² = 1 - SSE_model / SSE_bench` on the identical (date, ticker) test panel,
  where `SSE_model = Σ(y - pred)²` uses the pipeline's own predictions and
  `SSE_bench` uses the benchmark's forecast over the same label window:
  `spy` = SPY's realised return over the label window
  (`price[t+H+1]/price[t+1]-1`, applied to every ticker) ·
  `ew16` = the cross-sectional mean of the realised target per date (the EW-16
  portfolio's return over the same window). Positive ⇒ model beats the benchmark
  forecast; negative ⇒ benchmark forecast better. Returns `pooled` +
  per-calendar-year rows, one column per benchmark.
- `src/pipeline.py` — `evaluate_all` now writes `oos_r2_benchmarks.csv` next to
  `forecast.csv` / `oos_r2_by_year.csv` (benchmark_close carried in the results dict).
- `scripts/run_backtest.py` — console prints the SPY/EW-16 table after the forecast
  metrics; no separate analysis script needed.

### Fresh test measurement (corrected labels — a remeasurement, not a new holdout)

Ran the promoted stack (enet_a0.001_r0.5 · absolute1 · weekly · ME refit · volwin 12w)
fresh on 2023–2025 under the current feature/label code; results in
`results/test_2023-2025_fresh/` (the pre-registered draw in
`test_2023-2025_promoted/` used the pre-Standardization labels and stays archived).

| period | OOS R² model wrt SPY | OOS R² model wrt EW-16 |
| --- | --- | --- |
| pooled | -0.1070 | -0.5060 |
| 2023 | -0.2391 | -0.5914 |
| 2024 | -0.0896 | -0.4426 |
| 2025 | -0.0100 | -0.4817 |

Cross-check against the absolute forecast metrics (model OOS R² 0.0160 vs zero;
SPY-as-forecast 0.1112, EW-16-as-forecast 0.3466, both vs zero): the relative
values reconcile exactly (`1 - 0.9840/0.8888 = -0.107` and
`1 - 0.9840/0.6534 = -0.506`). Interpretation: the model's 1-week forecasts have
~11% higher squared error than using SPY's return as the forecast, and ~51%
higher than using the EW-16 mean return — consistent with the gate FAIL
(book trails both benchmarks on 2023–2025).

---
## Step 11 — Combined 6-year validation window + prediction cache (2026-11-03)
### Goal (user request)
Merge the two validation periods (2017–2019 selection, 2020–2022 ablation) into a
single **6-year validation window (2017-01-01..2022-12-31)** used by both phases, so
the model and the trading strategy are chosen/tuned on the **same** period. Cache the
walk-forward predictions under `results/` so phase 2 rebuilds books from the exact
forecasts phase 1 scored. Pin **`DEFAULT_SEED`** everywhere. No backwards
compatibility:
the old two-window flow, the simulation-mode v2b phase and the test-span-keyed report
phases were removed.
### What changed
- **`scripts/run_study.py` (rewritten).** Phases `v1` (selection), `v2` (ablation)
  and `test` (single pre-registered draw). One window, `VALIDATION`, drives both
  study phases; `V1_COMBOS` = 14 valid (rebalance, look-ahead, refit) combos and the
  93-spec grid (Step 12) give **1302 candidates**. Phase 1 scores the plain overlay-free
  book; phase 2 rebuilds the 11 book rows with `apply_vol_target=True` from the
  locked candidate's cached predictions. Study runs write tables/series/summaries
  only — **no figures**. `--study-dir` keeps smoke runs out of the record;
  `--models` / `--combos` filter for smoke runs; `--refresh-cache` recomputes.
- **`src/prediction_cache.py` (new).** `cached_predictions` / `cached_warmup` write
  `date,ticker,pred` CSVs plus a JSON provenance record (spec, target, refit,
  training window, embargo, **seed**, window, panel signature) under
  `results/_study/predictions/<window>/<freq>_<la>_refit<slug>/`. The key omits the
  rebalance cadence and book knobs (predictions do not depend on them), so the
  weekly / 4w / 12w candidates of one (look-ahead, refit) share a single file, and a
  cached entry is reused only when the provenance matches — a different seed or panel
  recomputes. Predictions are written with 17 significant digits, so reloading is
  bit-identical to recomputing.
- **`src/splits.py`** — `fixed_splits(panel, validation)` for an explicit validation
  span (the study no longer derives it from `test_start - window_val`).
- **`src/pipeline.py`** — `_vol_warmup_predictions` delegates to
  `prediction_cache.warmup_frame` (one warmup implementation); `REPORT_PHASES` /
  `report_phase_for` replaced by `REPORT_TEST_WINDOW`: only the 2023–2025 test draw
  publishes `results/plots/test/`.
- **`src/models.py`** — `ModelSpec.from_record(...)` rebuilds a spec from a stored
  lock record.
- **`src/utils/config.py`** — `seed_everything(seed=DEFAULT_SEED)` (seeds `random` +
  `numpy`);
  **`src/utils/paths.py`** — `STUDY_DIR`, `PREDICTIONS_DIR`.
- **`scripts/run_backtest.py`** — `--window-val` default `6y` (validation = the
  combined window), `seed_everything()` at start.
- **Volatility warmup** — one shared **48-bar** warmup per locked candidate covers
  every vol-window row (4–48), so all ablation rows start from the same volatility
  history instead of a per-row warmup.
- **Design rule fix surfaced by the cache check:** the `vol_target_none` row now also
  sets `rf 0` — the book's net return always credits the cash sleeve, so leaving
  `rf 3%` made "vol target off" ≠ the phase-1 selection book. With `rf 0` the row is
  exactly the selection book.
### Consistency checks (asserted every `v2` run, exit non-zero on failure)
- `volwin_24w` net Sharpe == `base` net Sharpe (bit-identical book).
- `vol_target_none` net Sharpe == the phase-1 lock's net Sharpe — proof that phase 1
  and phase 2 trade the same cached forecasts on the same dates.
### Verified
- `pytest`: 9 passed (new `tests/test_study_scheme.py`: explicit split, target slug
  mapping, cache round-trip bit-identity, cadence-agnostic cache key, seed/panel
  invalidation).
- Smoke (`--study-dir results/_smoke`, 1 candidate, `ridge_a10` / la1 / weekly / QE):
  phase 1 locked at 0.1977; a second phase-1 run hit the cache and reproduced the same
  Sharpe; a 4w-cadence run of the same (look-ahead, refit) **added no cache file**
  (~3 s, panel build + book only). Phase 2 ran all 11 rows from the two cached files
  and both consistency checks passed. Smoke artifacts deleted.
- `py_compile` clean on every edited module; `agents/model_design.py` prints the new
  split/seed/cache summary.
### Housekeeping

- Superseded study JSONs moved to `results/_study/archive_two_window/` and the old
  phase figure folders to `results/plots/_archive_two_window/`, so
  `results/_study/` holds only the current scheme's outputs and
  `results/plots/` only `test/`.

### Not done / open

- Phase 1 was smoke-tested only (Step 11); the full run is the next action (Step 12).
- The 2023–2025 window was already consumed by the previous cycle; the `test` phase is
  available but must be treated as a remeasurement.

---

## Step 12 — Reduced model grid (93 specs / 1302 candidates) + clean rerun (2026-11-03)

### Goal (user request)

Shrink the candidate grid (RFF dominated the runtime and the count), then delete the
old results and rerun the study end to end.

### Grid now in force (user-specified)

| family | alpha | extra dims | specs |
| --- | --- | --- | --- |
| lasso | 1e-4, 1e-3, 1e-2, 1e-1 | — | 4 |
| ridge | 1e-3, 1e-2, 1e-1, 1, 10 | — | 5 |
| enet | 1e-4, 1e-3, 1e-2, 1e-1 | l1_ratio 0.5, 0.25, 0.75 | 12 |
| rff | 1e-4, 1e-3, 1e-2, 1e-1, 1, 10 | D 100, 200, 500; gamma 0.01, 0.1, 1, 10 | 72 |
| | | **total** | **93** |

93 specs x 14 valid (rebalance, look-ahead, refit) combos = **1302 candidates**
(was 144 x 14 = 2016). RFF compute is ~2.6x cheaper by the sum-of-D metric
(19,200 vs 49,500) since D ∈ {100, 200, 500} replaces {50, 100, 500, 1000}.

**Recorded caveat:** gamma is the RBF *inverse* bandwidth (k = exp(-gamma||x-y||^2));
on the standardized 21-feature panel the mean pair distance^2 is ~42, so gamma ≥ 1
collapses the kernel towards a point mass. The user chose to keep gamma 1 and 10 in
the grid, so those 36 cells are a deliberate measured check on that claim, not
expected signal. The design doc says so explicitly.

### Changed

- `scripts/run_backtest.py` — `RIDGE_ALPHAS`, `LASSO_ALPHAS`, `ENET_ALPHAS`,
  `ENET_L1_RATIOS`, `RFF_ALPHAS`, `RFF_DIMS`, `RFF_GAMMAS` + docstrings/counts.
- `agents/model_design.py` — mirrored constants, `V1_CANDIDATES = 1302`, grid comment
  and `main()` summary.
- `scripts/run_study.py` — module docstring counts.
- `agents/validation_and_ablation.md` — §3 hyperparameter table (levels, per-family
  specs, gamma caveat), grid → decision line, §6 status.
- `README.md` — grid/candidate counts and the `--model` example (`lasso_a0.01`).
- `agents/current_state.md` — this entry; Step 11's candidate count refreshed.

### Deleted (old scheme outputs)

- `results/_study/predictions/` (the whole walk-forward cache: 267 files from the
  aborted 144-spec run) and any partial `validation_part1/2_*` outputs from it.
- Run folders from the aborted grid (`results/*_val6y_*`), so the new record starts
  clean. `results/_study/archive_two_window/` and
  `results/plots/_archive_two_window/` are kept — they are the documented
  two-window archive referenced by §5.

### Then

- `python scripts/run_study.py all` (phase 1 in an 8-process pool → phase 2 → test).

### Result (executed same day, 2026-11-03)

- **Phase 1 lock:** `rff_a0.1_d100_g0.01` · absolute4 · 4w · QE, val net Sharpe
  **1.2068** (gross 1.2491, TO 0.2549) out of 1302 candidates; 59 degenerate
  (enet 42/168, lasso 17/56, ridge 0/70, rff 0/1008). Best linear cells ~0.81.
- **Gamma evidence:** best net Sharpe by gamma — 0.01 → 1.207, 0.1 → 0.859,
  1 → 0.697, 10 → 0.777. The near-point-mass kernels did underperform, as the
  a-priori argument predicted, but none were degenerate — RBF noise features still
  fit, so the "dead zone" shows up as *worse score*, not as a flagged collapse.
- **Phase 2:** volwin_12w wins at **1.3239** vs base 1.2026; both consistency
  checks passed (volwin_24w == base, vol_target_none == the phase-1 lock), proving
  phase 1 and phase 2 traded byte-identical cached forecasts.
- **Test draw (2023–2025, once):** net Sharpe **0.2906**, maxDD −10.42%, total
  +5.79% vs EW-16 1.3145 / −7.79% / +38.42% and SPY 1.5230 / −16.88% / +78.86%
  → **gate FAIL on both conditions. Null result, nothing promoted, `HEADLINE_*`
  stays None.** Forecast OOS R² **−0.0387** (worse than a zero forecast), rank-IC
  0.0973 (t 3.75); OOS R² vs SPY −0.150, vs EW-16 −0.618.
- **Read:** the winner sits in a tight cluster (rff, D ≤ 200, gamma 0.01, absolute4,
  4w) and its validation edge does not survive out of sample — the phase-1 ranking
  looks like selection noise, not a broad plateau. Full numbers in
  `agents/validation_and_ablation.md` §5.0.
- **Artifacts:** cache 1,634 files (≈817 prediction sets, CSV + JSON provenance) in
  `results/_study/predictions`; run tables for the 11 phase-2 rows in
  `results/<tag>__v2_<row>/`; test draw in
  `results/weekly_fg_roll5y_val6y_qrefit_rff_a0.1_d100_g0.01_vt0.1c1rf3_2023-2025/`
  with figures in `results/plots/test/`.


---
## Step 13 — Single joint ranking (no phases), old results deleted (2026-11-03)
### Goal (user request)
Drop the phase-1 / phase-2 split. The prediction stage is just the walk-forward
prediction cache (already stored as CSVs); validation is then **one ranking over the
combined space of model x strategy parameters**, written to a CSV under
`results/_study`. Also remove every archived result.
### What changed
- **`scripts/run_study.py` (rewritten again).** Two commands only:
  - `rank` (default) — replays the cached predictions through the whole space and
    writes `results/_study/validation_ranking.csv` (14,322 rows, sorted by net
    Sharpe), `validation_winner.json` and `run_parameters.json`. No phases, no book
    "ablation rows" stage, no per-row run folders, no figures.
  - `test` — the single pre-registered draw of the ranking winner on 2023-2025.
  - `--models` / `--combos` subset the space, `--jobs` sets workers, `--study-dir`
    redirects the output root, `--refresh-cache` recomputes predictions.
- **Space definition:** 93 specs x 14 valid (rebalance, look-ahead, refit) combos x
  11 book rows = **14,322 candidates**, all scored with the same metric (net Sharpe)
  on the same window (2017-01-01..2022-12-31). Each candidate's 11 books are built
  from that candidate's cached predictions and one shared 48-bar volatility warmup.
- **Ranking CSV columns:** `rank`, model parameters (`model`, `family`, `alpha`,
  `l1_ratio`, `rff_dim`, `gamma`), `target`, `look_ahead`, `rebalance`, `refit`,
  `combo`, book knobs (`book`, `book_note`, `vol_window`, `vol_target`, `vol_rf`,
  `vol_cap`, `lev_fee`), and metrics (`net_sharpe`, `gross_sharpe`, `max_drawdown`,
  `total_return`, `ann_return`, `ann_vol`, `hit_rate`, `avg_turnover`, `degenerate`).
- **`src/models.py`:** `ModelSpec.from_record` now coerces `rff_dim` to int and
  treats NaN as None, so a lock/winner record rebuilds the exact spec name
  (`..._d100_...`, not `..._d100.0_...`) and therefore the right cache key.
- **Docs:** `agents/validation_and_ablation.md` rewritten to the single-stage
  protocol (§1 split, §2 cache, §3 ranked space, §4 output format, §5 record, §6
  status); `agents/model_design.py` header/§6/§7 and `RANK_CANDIDATES = 14_322`;
  `README.md` commands and outputs; `agents/plots.md` (only the test draw plots).
### Deleted (user request: no old archived results)
- `results/_study/archive_two_window/` (27 JSONs, moved there in Step 11).
- `results/plots/_archive_two_window/` and `results/plots/test/` (old figures).
- All 14 `*_grid144*` V1 run folders, all 11 `*_2020-2022__v2_*` folders, all 11
  `*_2017-2022__v2_*` folders, `test_2023-2025_fresh/`, `test_2023-2025_promoted/`
  and the 2026-11-03 test run folder.
- Superseded `results/_study/validation_part1_*` / `validation_part2_*` files and
  the aborted-run log.
Result: `results/` now holds only `_study/predictions/` (the cache) plus whatever
the current ranking/test writes. The summary numbers of the removed cycles remain in
Steps 1-12 above.

### Result + book-builder rewrite (executed, same day)
- **Ranking:** `python scripts/run_study.py rank` → **14,322 rows in 76 s** (8 workers,
  17–19 ms per book). Winner = `rff_a0.1_d100_g0.01` · absolute4 · 4w · QE ·
  `volwin_12w`, net Sharpe **1.3239** (maxDD −9.82%, total +96.7%); 649 degenerate
  rows. Top-2/3: same candidate with `volwin_8w` 1.2087 and `vol_target_none` 1.2068;
  base 1.2026. The top 100 is 56 × `4w/abs4/QE`, 35 × `4w/abs4/ME`, 9 weekly.
- **Test draw (once):** net Sharpe **0.2906**, maxDD −10.42%, total +5.79% vs EW-16
  1.3145 / −7.79% / +38.42% and SPY 1.5230 / −16.88% / +78.86% → **gate FAIL on both
  conditions. Null result, nothing promoted, `HEADLINE_*` = None.** Same numbers as
  the pre-rewrite run, which doubles as the end-to-end regression check.
- **Why the first attempt took 45+ min and had to be stopped:** the model side was
  cached (0.01 s per candidate) but the book simulation was a pandas loop over 313
  bars (~0.3–0.7 s per book), i.e. 14,322 × ~0.5 s ≈ 70 min of CPU, and joblib was
  pickling the 12k-row panel per task (measured ~1.8 of 8 cores, then ~6 after
  batching).
- **Fixes applied (no change to the book space — all 11 rows kept):**
  1. `portfolio.build_forecast_vol_portfolio` rewritten as a numpy state machine:
     target weights for every bar, the rebalance calendar and the commission function
     are precomputed; the loop keeps only the inherent weight-drift recurrence.
     Added `_commission_usd` (vectorised `order_cost_usd`) and `forecast_vol_frame`.
  2. `build_positions` / `_forecast_vol_scaled` accept an optional precomputed
     `forecast_vol`, and `study_worker` shares the rolling volatility across the six
     distinct vol windows instead of recomputing it for all eleven rows.
  3. `scripts/study_worker.py` (new): the worker state (panel, features, benchmark,
     cache root) is installed once per process via a joblib `initializer`, and tasks
     carry a batch of 5 candidates — no per-task panel pickle.
- **Equivalence check:** 11 books × 3 candidates (rff/ridge/enet, 4w and weekly
  cadence) reproduce the previous implementation's net/gross Sharpe, maxDD, total
  return and turnover to **max |Δ| = 2.2e-15**; the test draw reproduces 0.2906 /
  −10.42% / +5.79% exactly.
- Artifacts: `results/_study/validation_ranking.csv` (14,322 rows),
  `validation_winner.json`, `run_parameters.json`, the run folder and
  `results/plots/test/` (10 figures).

---
## Step 14 — Compact cache (npz), degenerate markers, pre-migration backup (2026-11-04)
### Goal (user request)
Keep the prediction cache so nothing is retrained, but store it in a representation
that is much smaller on disk **and** in git; and stop storing predictions for
degenerate models — just mark them and skip.
### Backup (first action)
`results/_study/predictions_backup_2026-11-04.zip` — 58.3 MB zip of the pre-migration
cache (3,352 files, 175.1 MB). Restorable, and nothing below required refitting.
### New storage format
| before | after |
| --- | --- |
| `<spec>.csv` long `date,ticker,pred` (%.17g) | `<spec>.npz`: float32 `(dates, tickers)` matrix + both axes |
| — | `<spec>.json` provenance (adds `format: npz/float32`) |
| degenerate sets stored like any other | `<window>/_degenerate/<group>__<spec>.json` marker; payload deleted |
- 175.1 MB → **21.8 MB** for 1,592 sets (8x). The npz is already compressed, so a git
  blob is ~the same size, against ~58 MB for the committed CSVs.
- **Precision:** float32 moves the winner's book Sharpe by 9.4e-9; the full ranking
  reproduces the same winner at the same 1.3239 and the test draw the same
  0.2906 / −10.42% / +5.79%.
### Degenerate policy
`prediction_cache.is_degenerate` flags a set whose mean cross-sectional prediction
std is < 1e-12 (the L1 models collapsing to their intercept). On first sight the set
is marked and never written; later calls raise `DegenerateCandidate`, and
`study_worker.score_candidate` turns that into eleven `degenerate=True` ranking rows
with NaN metrics — **no book simulation runs for them**. Warmups are exempt (a flat
warmup only seeds the volatility estimate, so it is always stored).
Current cache: **42 degenerate prediction sets** (59 degenerate ranking candidates),
84 markers in total (42 sets + their warmups).
### Code
- `src/prediction_cache.py` — npz read/write (`write_predictions`), `is_degenerate`,
  `mark_degenerate`, `degenerate_markers`, `DegenerateCandidate` (picklable: `record`
  defaults to `{}`), `check_degenerate` flag on the shared hit/compute path so warmups
  bypass the marker check.
- `scripts/study_worker.py` — catches `DegenerateCandidate`, emits degenerate rows,
  skips the vol/books path.
- `scripts/compact_prediction_cache.py` (new) — in-place re-encoder
  (`--dry-run` supported); never refits. **Deleted in Step 18**: the committed cache
  is already npz, so the migration path is dead for a fresh clone.
- Tests: round-trip is now float32-exact (`rtol=1e-7`), plus a degenerate test
  (marker written, no payload, second call raises without refitting) and an
  `is_degenerate` unit test. 11 passed.
### Verified
- `run_study.py rank` against the migrated cache: 14,322 rows, 649 degenerate,
  winner `rff_a0.1_d100_g0.01` / abs4 / 4w / QE / `volwin_12w` at 1.3239 (81 s).
- `run_study.py test`: identical to Step 13 (0.2906, maxDD −10.42%, total +5.79%,
  gate FAIL). Nothing retrained at any point.
### Final cache policy (2026-11-04, supersedes the temporary ignore rule)
The cache was briefly git-ignored while the migration was unproven. Both that rule
and the `predictions_backup_*.zip` ignore were **removed**, and the backup zip was
**deleted** — nothing needed it once the npz cache reproduced the ranking and the
test draw exactly.
- **The cache is meant to be committed.** `results/_study/predictions/` is 21.9 MB
  across 3,268 files (1,592 `.npz` payloads + 1,676 `.json` provenance/markers), so a
  fresh clone can reproduce `validation_ranking.csv` and the test draw without
  refitting anything. Only the noise paths stay ignored (`.venv/`, `__pycache__/`,
  tool caches, `*.log`).
- Restore path, if it is ever lost: `python scripts/run_study.py rank` retrains from
  scratch and rewrites the cache (93 specs × 14 combos, seeded `DEFAULT_SEED` ⇒
  reproducible).

---

## Step 15 — Book space is the full cross-product, base row dropped (2026-11-04)
### Goal (user request)
The book dimension must be the **whole product** of the strategy knobs rather than one
knob at a time around a shared base row: vol window {4, 8, 12, 24, 48}w x vol target
{20%, 10%, off} x rf {0%, 3%} x leverage cap {1.0, 2.0, 3.0} (+6.8% fee). Then the rf
0% arm was dropped, leaving the sleeve fixed at 3% and **45 book cells**.
### What changed
- **`scripts/study_worker.py`.** `BOOKS` is now generated from `VOL_WINDOWS`,
  `VOL_TARGETS`, `VOL_RF_RATES = (0.03,)` and `LEV_CAPS = ((1.0, 0.0), (2.0, 0.068),
  (3.0, 0.068))`; the explicit base/one-knob rows are gone. Cell names are derived, e.g.
  `w12_vt20_rf3_cap1.0`, `w48_vtoff_rf3_cap3.0`. `vol_target` is `None` for "off".
- **`effective_key`.** A cell's real book is `(window, target, rf, cap_eff, lev_fee)`
  with `cap_eff = cap if target else 1.0`: without a target `_vol_target_scale`
  returns 1.0 on every rebalance, so the cap can never bind. Those cells are simulated
  **once** and share metrics — 45 cells, 40 distinct simulations per candidate. All 45
  are still written to the ranking, so the declared space is exactly what is reported.
- **`scripts/run_study.py`** — docstring/space description (93 x 14 x 45 = **58,590**
  candidates, ≈3 min); the driver already reads `len(BOOKS)`.
- **`agents/model_design.py`** — §6/§7 rewritten for the product grid,
  `RANK_CANDIDATES = 58_590`, `BOOK_SIMULATIONS = 40`.
- **`agents/validation_and_ablation.md`** — §3 space table, §4/§5.1 measured record,
  §6 status. **`README.md`** — run comment and cache line.
### Measured (2026-11-04)
- `run_study.py rank`: **58,590 rows in ≈3 min** (8 workers), 2,655 degenerate rows
  (59 degenerate spec x combo pairs x 45 cells).
- **Winner:** `rff_a0.1_d100_g0.01` · absolute4 · 4w · QE · `w12_vt20_rf3_cap1.0` —
  net Sharpe **1.3361**, maxDD −10.15%, total +121.8%, ann 14.15% / vol 10.31%.
  Runners-up are the same candidate's other book cells (1.3239 vt10, 1.3117 vol-off,
  1.3090 levered); the next model (`rff_a0.01`) reaches 1.2906.
- **Knob values at their best:** vol target 20% > 10% > off; window 12w > 4w > 8w >
  24w > 48w; cap 1.0 > 2.0 = 3.0 (borrowing never wins). Five rows sit within 0.03
  Sharpe of the top, nine within 0.05, all `4w · QE`.
- **Test draw (2023-2025, single measurement):** net Sharpe **0.1595**, maxDD −10.42%,
  total +2.69%, ann 0.97% vs EW-16 1.3145 / −7.79% / +38.42% and SPY 1.5230 /
  −16.88% / +78.86% → **gate FAIL on both conditions**, OOS R² −0.0387. Nothing
  promoted; folder `results/weekly_fg_roll5y_val6y_qrefit_rff_a0.1_d100_g0.01_vt0.2c1rf3_2023-2025`
  (the superseded `vt0.1c1rf3` folder was deleted).
- The prediction cache was untouched — only book simulations reran, so the whole
  exercise cost ≈3 min of compute.

---

## Step 16 — Validation-window figures published next to the test ones (2026-11-04)
### Goal (user request)
"Have the plots been updated, also generate plots for the best val" — a figure set
for the **validation window** of the ranking winner, not just the out-of-sample draw.
### What changed
- **`src/pipeline.py`.** `report_phase_for` (one phase) → `report_phases_for`
  returning both phases, `REPORT_PHASES = (("test", "test"), ("val", "validation"))`;
  `evaluate_all` publishes each. The test draw now keeps **both** halves of the figure
  set it already rendered (the validation half used to be deleted with the transient
  `plots/` folder) → `results/plots/test/` + `results/plots/val/`, 10 PNGs each.
- **The validation book is now the overlaid one.** `select_model` also returns
  `overlay_books` (the winner with `apply_vol_target=True`) and `run_pipeline` uses it
  as `val_portfolio`, so the validation figures show the **book the joint ranking
  scored** (1.337 Sharpe, maxDD −10.15%, total +121.96%) instead of the overlay-free
  selection book (1.313 / −10.15% / +119.4%). The overlay-free book remains the
  internal selection basis for the in-pipeline ranking.
- **`src/plots.py`.** `_stats_line` no longer hard-codes "overlay-free (selection
  basis)" for the validation window; the new `_overlay_label` reads the config and
  reports `vol target 20% cap 1 (scaled on N/M bars)`, which also exposes that the
  overlay is inert here: it scales **4 of 313** validation bars and **0 of 143** test
  bars (the inverse-vol book already runs ≈10% vol, so `target/vol > cap` and the
  overlay clamps at 1.0). The reason `vol target 20%` ≈ `off` in Step 15 is now
  visible in the figures themselves.
- Docs: `agents/plots.md` (both phases, why two windows, subtitle format),
  `README.md`, `agents/validation_and_ablation.md` §5.1.
### Verified
- `python scripts/run_study.py test` (≈6 s, deterministic): test net Sharpe 0.1595,
  maxDD −10.42%, total +2.69%, gate FAIL — unchanged. `results/plots/test/` and
  `results/plots/val/` each hold 10 PNGs; the val panels are headed
  "validation window 2017-01-06 → 2022-12-30" with the winner's 1.337 / −10.15% /
  +121.96% subtitle. `pytest` 12 passed.

---

## Step 17 — Pre-push consistency sweep (2026-11-04)
### Goal (user request)
"Check for any stale docs, data, or files which do not match the latest design" before
pushing the combined-window / joint-ranking branch.
### Stale docs fixed
- **`agents/model_design.py` §4/§5** still described the retired V1/V2 ablation rows
  ("V2 row 1 base", "OFF in V1", "rows 1 -> 2 -> 3 -> 4 -> 5"). Rewritten for the 45-cell
  book space; `PREDICTIONS_CACHE` comment said "git-ignored … one CSV"
  (now committed npz); the lineage block called the protocol an "ablation protocol".
- **`scripts/run_backtest.py`** docstring claimed "V1 selects … (93 specs x 14 valid
  combos = 1302 candidates)" and its flags/help referenced V1/V2 rows. Now describes the
  joint ranking (93 x 14 x 45 = 58,590) and that the CLI measures one pinned cell.
- **`src/utils/paths.py`** said `results/` and the study cache are "git-ignored".
- **`agents/validation_and_ablation.md`** said the cache stores "CSVs"; §4 now records
  that `book` is a slug and its note lives in `run_parameters.json`; history pointer
  Steps 1-13 → 1-16.
- **`README.md`** ranking runtime 5 min → 3-4 min; test-window wording now says the
  holdout has been measured **twice** (retired two-window protocol + this ranking).
### Stale data / files removed
- **4 orphan cache entries** named `rff_a0.1_d100.0_g0.01*` (a float-formatted
  `rff_dim` from an older `ModelSpec`; the grid writes `d100`). Deleted from both the
  validation and the warmup window, leaving 0 files not in the grid and 0 orphan
  npz/JSON pairs across all 9 (look-ahead, refit) groups (86-90 payloads + markers
  = 93 per group).
- **`__pycache__` (repo root, agents/, scripts/, src/, tests/) and `.pytest_cache`**
  deleted (git-ignored, but they were the only untracked noise).
### Artifact hygiene
- `save_table` gained `index: bool = True`; the ranking is written with `index=False`,
  dropping a meaningless `Unnamed: 0` column (20.5 MB → 15.9 MB).
- `book_note` was repeated on all 58,590 rows (3.4 MB of prose): the slug plus the
  note once per name in `run_parameters.json` replaces it.
- `run_parameters.json` was **YAML with a .json extension**; it is now real JSON via
  the existing `_save` helper.
- Re-ran `rank` (218 s) + `test` after the changes: same winner
  (`w12_vt20_rf3_cap1.0`, 1.3361), same test draw (0.1595 / −10.42% / +2.69%, gate
  FAIL). `pytest` 12 passed (new `test_book_space_is_the_declared_product` guards the
  45 cells / 40 distinct books / 3% rf / 6.8% fee).
### Commit payload
- 360 deletions (every old `results/` artifact), 17 modified files, 5 new files
  (`scripts/study_worker.py`, `scripts/compact_prediction_cache.py`,
  `src/prediction_cache.py`, `tests/test_study_scheme.py`, `results/` tree).
- `results/` totals 3,306 files / 40.1 MB: prediction cache 22.9 MB (npz, committed on
  purpose), ranking CSV 15.9 MB, 20 report figures ≈1.3 MB, one test-run folder.

---

## Step 18 — Seed literals out of the docs, legacy cache tool deleted (2026-11-04)
### Goal (user requests)
1. "Do not hard-code any SEED values in docstrings, only load from the constant."
2. "Is `compact_prediction_cache` needed if it is fresh and existing run?" — no.
### Seed literals
`utils.config.DEFAULT_SEED` is now the **only** place the value appears (plus the
mirror `SEED` in `agents/model_design.py`, which is labelled as a mirror). Every
docstring/doc that used to spell the number now names the constant:
`scripts/run_study.py` (module docstring + `SEED = DEFAULT_SEED`), 
`scripts/run_backtest.py`, `agents/model_design.py` (§ intro, §6, the mirror note),
`agents/validation_and_ablation.md` (§ intro, §1.5, §6), `agents/plots.md`,
`README.md`, and the historical entries in this log. A future seed change is now a
one-line edit with no doc drift.
### `scripts/compact_prediction_cache.py` deleted
It existed only to re-encode the pre-2026-11-04 **long-CSV** cache into npz. The
committed cache is npz already and the encoder lives in
`prediction_cache.write_predictions`, so a fresh clone (or a user with no prior
cache) never needs the migration path: `run_study.py rank` simply writes npz from
scratch. Removed the script and its README/§2 references; the Step 14 record keeps
the migration history.
### Verified
- `pytest` 12 passed; no seed digit remains in any docstring or doc — only
  `utils.config.DEFAULT_SEED` (source of truth) and the course code `DSA5205`.

---

## Step 19 — Regime-stratified selection criterion, both records kept (2026-11-04) — removed in Step 20
### Goal (user request)
After reviewing the figures the user judged the model/book selection sound but the
validation regime too distant from the test: "implement the regime-stratified and pick
the best val and corresponding effect on the test from there", keeping the old results
for the report.
### Diagnosis first (read-only)
- **Payoff shape flipped.** Up-capture against EW-16 fell 0.60 (val) -> 0.21 (test)
  while down-capture stayed ~0.27: the book was convex inside validation (it paid for
  protection) and concave on the test window (all three years up, two of them for the
  benchmarks). Validation Sharpe 1.34 came from 2018/2020/2022.
- **Book selection is mostly noise.** Across the 45 cell books of one model, Sharpe
  correlates 0.27 Pearson / 0.14 Spearman between the two validation halves and 0.05
  with the test window; the winner ranks 8/45 in 2017-19 and 22/45 on the test.
- **Signal is weak everywhere.** Winner's validation rank-IC by year: +0.154, -0.018,
  -0.062, +0.120, +0.086, -0.012 (pooled 0.045, t 2.3). Re-running the top six models
  with the winning book on 2023-2025 gives 0.158 / 0.065 / 0.047 / 0.045 / 0.285 /
  -0.008 — no model choice transfers.
### What changed
- **`scripts/study_worker.py`** — `regime_metrics()` (per-year Sharpe, `pos_years`,
  `yr_sharpe_min`, up/down capture vs the equal-weight benchmark) and `regime_pass()`
  with `REGIME_MIN_POS_YEARS = 5`, `REGIME_WINDOW_YEARS = 6`, `REGIME_MIN_UP_CAPTURE =
  0.40`. Every ranking row now carries the five regime metrics plus `regime_pass`.
  The consistency test is the **5/6 rate**, not a raw count, so it generalises to
  other window lengths.
- **`scripts/run_study.py`** — `--criterion {raw,regime}` for both `rank` and `test`.
  `raw` is unchanged byte-for-byte (same filenames); `regime` writes
  `validation_ranking_regime.csv` / `validation_winner_regime.json` /
  `run_parameters_regime.json`, prints a sensitivity ladder (rows profitable in 6/6,
  5/6, 4/6 years), and ranks the survivors by worst-year Sharpe (maximin).
- **`src/pipeline.py`** — `report_prefix` threaded through `run_pipeline` /
  `evaluate_all`, so the regime draw publishes `results/plots/regime_{test,val}/`
  beside the untouched `results/plots/{test,val}/`.
- Tests: `test_regime_metrics_and_gate` + `test_regime_gate_thresholds_...` (14 passed).
- Docs: `agents/validation_and_ablation.md` §3.1 (the two criteria and why
  down-capture is **not** gated), §4 (files), §5.2 (record), §6; `README.md`;
  `agents/model_design.py` §6.
### Measured
- `rank --criterion regime` (≈4.6 min): gate passes **7,181 of 55,935** live rows.
- **The gate alone changes nothing** — the raw winner is profitable in 6/6 years
  (worst year +0.27) with up-capture 0.60, so it passes. The maximin objective is what
  moves the pick.
- **Regime winner:** `rff_a0.01_d100_g0.01` · abs4 · 4w · QE · `w04_vtoff_rf3_cap1.0`
  — val Sharpe **1.2906** (vs 1.3361), worst year **+0.41** (vs +0.27), up/down
  capture 0.61 / 0.30, maxDD −11.11%.
- **Test draw:** net Sharpe **−0.0750**, maxDD −12.77%, total −2.66%, OOS R² −0.0509,
  rank-IC 0.0919 (t 3.42) -> gate **FAIL**, worse than the raw pick (0.1595 /
  −10.42% / +2.69%).
- **Why:** the two picks behave like the same book (test return correlation **0.96**,
  same sign in 89% of weeks). Regime-stratification changed the regularisation and the
  vol cell, not the signal. Conclusion: the §5.1 failure is **signal decay**, not
  year-concentration in the selection.
- 2023-2025 now carries **three** measurements; documented as a remeasurement window.

- No stale references to the deleted script outside this history log.

---

## Step 20 — Regime-stratified criterion removed (2026-11-06)

### Goal (user request)
"Remove all artifacts and code of the regime criterion." The Step 19 criterion and
its outputs are gone; the single max-Sharpe ranking is the only selection rule again.

### What changed
- **`scripts/study_worker.py`** — deleted `REGIME_MIN_POS_YEARS`,
  `REGIME_WINDOW_YEARS`, `REGIME_MIN_UP_CAPTURE`, `regime_metrics()` and
  `regime_pass()`; the per-row columns `pos_years` / `n_years` / `yr_sharpe_min` /
  `up_capture` / `down_capture` / `regime_pass` and the benchmark series they needed
  are gone, so a ranking row is the model/combo/book metrics only.
- **`scripts/run_study.py`**, **`scripts/run_study_macro.py`** — `--criterion` and the
  `_{criterion}` filename suffix removed; `rank`/`test` always write the plain
  `validation_ranking.csv` / `validation_winner.json` / `run_parameters.json`.
- **`src/pipeline.py`** — `report_prefix` removed from `run_pipeline` /
  `evaluate_all` / figure publishing (it existed only to namespace the regime figure
  sets).
- **Tests** — `test_regime_metrics_and_gate` and
  `test_regime_gate_thresholds_are_the_pre_registered_ones` deleted.
- **Artifacts deleted:** `results/_study*/validation_ranking_regime.csv`,
  `validation_winner_regime.json`, `run_parameters_regime.json`;
  `results/plots{,_macro}/regime_{test,val}/`; the regime test draws
  `results/weekly_fg_roll5y_val6y_qrefit_rff_a0.01_d100_g0.01_2023-2025` and
  `results/weekly_fg_roll8y_val4y_qrefit_enet_a0.01_r0.75_2023-2025_macro`.
- **Artifacts rewritten:** the regime columns stripped from
  `results/_study_macro/validation_ranking.csv`; the `criterion` / `rule` / `regime` /
  `candidates.regime_passed` keys stripped from
  `results/_study_macro/{validation_winner,run_parameters}.json`.
- **Docs** — `agents/validation_and_ablation.md` §3.1 (one criterion), §4 (schema and
  files), §5.2 (marked removed), §6 (status + history); `agents/macro_study.md`
  (selection + outputs); `agents/model_design.py` §6; `README.md` (usage).

### Notes
- The deleted numbers are preserved in Step 19 above; nothing else referenced them.




## Step 21 — `cache` command, full refit space re-ranked, new test draw (2026-11-06)

The macro study had only ever been ranked on the QE arm (5,400 candidates). The other
two arms' prediction cache was incomplete (W-FRI: 39 of 60 sets missing, plus warmups),
so the study CLIs gained a **`cache` command** that fills a partial cache without
running any book simulation:

- **`scripts/study_worker.py`** — `cache_candidate()` / `cache_batch()` fit one
  candidate's predictions and 48-bar warmup through the same `init()` state as
  `score_candidate()`, and return `{"candidate","spec","status","seconds"}` with status
  `cached` / `stored` / `degenerate`. `forecast_groups()` collapses combos onto the
  `(look-ahead, refit)` forecast groups they share, so 14 combos cost 9 groups of fits.
- **`scripts/run_study.py`**, **`scripts/run_study_macro.py`** — third command `cache`
  (same `--models` / `--combos` / `--refits` / `--refresh-cache` / `--jobs` filters),
  one line per newly stored or degenerate set plus a `stored / degenerate / cached`
  summary; nothing is ranked or written.
- **`src/prediction_cache.py`** — the degenerate-marker helper is public
  (`marker_path(path)`); the unused `_marker_path()` wrapper is gone.
- **Docs** — `README.md` usage, `agents/validation_and_ablation.md` §4,
  `agents/macro_study.md` §4 (command) and §7.1 (measured record).

**Cache:** `run_study_macro.py cache` → 180 prediction sets (20 specs x 9 groups):
131 cached, 49 degenerate, 0 stored on the re-check; the W-FRI arm's 39 sets and their
warmups were fitted before that. Provenance is uniform within every group (only
`rebalance`, which is not a cache key, varies).

**Ranking:** `run_study_macro.py rank` (14 combos) → **12,600 candidates, 3,195
degenerate, 9,405 live**. The QE and ME arms reproduce their earlier numbers exactly
(1.1763 and 1.1555), so widening the space changed the selection only.

| arm | best | net Sharpe | maxDD |
| --- | --- | --- | --- |
| W-FRI | `enet_a0.001_r1` · abs4 · weekly · `w04_vt10_rf3_cap1.0` | **1.1945** | −17.77% |
| QE | `enet_a0.01_r1` · abs12 · weekly · `w48_vt20_rf3_cap1.0` | 1.1763 | −14.04% |
| ME | `enet_a0.01_r0.5` · abs4 · weekly · `w04_vt10_rf3_cap1.0` | 1.1555 | −13.84% |

Median live Sharpe is 0.355 (W-FRI) vs 0.423 (QE) / 0.439 (ME): the weekly arm wins the
top row and loses the middle, and its 0.018 Sharpe edge costs 208 refits per prediction
set against 16.

**Test draw (2023-2025), new winner:** net Sharpe **0.5286**, maxDD −8.85%, total
+12.70%, ann vol 14.5% (by year: −0.02 / 0.30 / 1.40). Gate **fails** (bars: Sharpe
1.3145, maxDD −7.79%); in every year the book also trails SPY (1.56 / 2.10 / 1.03) and
equal weight. Draw folder (moved into the study tree in Step 22)
`results/_study_macro/draws/w-fri/weekly_fg_roll8y_val4y_wkrefit_enet_a0.001_r1_vt0.1c1rf3_rweekly_2023-2025`,
figures in what Step 22 made `plots/w-fri/{test,val}/`. The previous QE winner's
draw (`..._qrefit_...`, now `draws/qe/`) is kept beside it.


## Step 22 — Output reorg into `_study*/draws/`, one test draw per refit arm (2026-11-06)

Study outputs were flat: run folders sat in `results/` and figures in
`results/plots{,_macro}/`, so a second draw could only overwrite the first and the
folder names carried a `_macro` suffix to stay apart. Everything a study produces now
lives inside its own folder:

```
results/_study{,_macro}/
  predictions/  validation_ranking.csv  validation_winner.json  run_parameters.json
  draws/<arm>/<run_tag>/      # arm = w-fri | me | qe
  plots/<arm>/{test,val}/     # 10 PNGs per window, no sharing across arms
  arm_comparison.txt          # per-arm test results + the winning arm
```

- **`scripts/run_study.py`**, **`scripts/run_study_macro.py`** — new `draw` command:
  the best live ranking row of each refit arm (optionally filtered with
  `--refits`) is drawn once on 2023-2025 into `draws/<arm>/<run_tag>/`, its figures
  published to `plots/<arm>/{test,val}/`, and `arm_comparison.txt` written at the
  study root. `test` keeps its meaning (single draw of the overall winner) but now routes
  to the winner's arm folder; the macro `_macro` run-tag suffix is gone because the
  folder supplies the isolation.
- **`src/utils/artifacts.py`** — `save_arm_summary()` renders the fixed-width
  comparison table (one row per arm, best `test_sharpe` first, plus the caveat that
  ranking arms by test Sharpe is descriptive only).
- **Moved with `git mv`:** `results/weekly_fg_roll5y_val6y_qrefit_rff_a0.1_d100_g0.01_vt0.2c1rf3_2023-2025`
  → `results/_study/draws/qe/…`; the two macro draws (`…_qrefit_…` → `draws/qe/`,
  `…_wkrefit_…` → `draws/w-fri/`, `_macro` suffix dropped); `results/plots/{test,val}`
  → `results/_study/plots/qe/{test,val}`; `results/plots_macro/{test,val}` →
  `results/_study_macro/plots/w-fri/{test,val}`. `results/` now holds only
  `_study/` and `_study_macro/`.
- **Docs** — `README.md` (usage + outputs tree), `agents/plots.md` (where figures live,
  which run publishes), `agents/validation_and_ablation.md` (§4 tree and
  `arm_comparison.txt` schema, new §5.1b), `agents/macro_study.md` (§4 `draw`, §5
  tree, new §7.2).

**Measured (2023-2025, one draw per arm):**

| study | W-FRI | ME | QE |
| --- | --- | --- | --- |
| v2 (`rff` specs) | −0.3425 | **0.2040** | 0.1595 |
| macro (`enet`) | 0.5286 | 0.8586 | **1.3856** |

All six draws fail the success gate. In the macro study the QE arm — third on
validation — beats the W-FRI winner by 31.6 pp total return, and its 1.386 Sharpe clears
the bar while its −7.97% drawdown misses by 0.18 pp; in the v2 study the weekly arm has
the best validation score and the worst draw. With one window and three draws this is a
description of the arms, not a way to pick one.

### Notes
- `run_backtest.py` still writes an ad-hoc run folder under `results/<tag>/` and, for a
  full 2023-2025 span, figures to the pipeline default `results/plots/`; only the two
  study drivers were re-homed.

### Follow-up (same day)
`plots/` and `arm_comparison.txt` were lifted out of `draws/` to the study root, so the
shape is `results/_study{,_macro}/{predictions,draws,plots}/ + ranking/winner/params +
arm_comparison.txt`. Moved with `git mv` (the summary file was untracked, so it was
moved with the filesystem); `arm_plots_root()` now returns `plots/<arm>` and
`save_arm_summary()` writes to `STUDY_DIR`. Both `draw` runs were repeated so every
figure exists at its final path.

`arm_comparison.txt` also gained a second table: cumulative return of the drawn book
versus EW and SPY B&H on **both** measured windows (`metrics.cumulative_return()` +
`pipeline.cumulative_by_window()`, which read the run's own `val_*`/`test_*` series so
the numbers agree with each draw's `comparison.csv`).

---

## Step 23 — Rule path wired and measured as the naive baseline (2026-11-09)

### Trigger
Draft feedback: the rule-based trend path was half-referenced — `src/portfolio.py`
implements `trend_long_flat`, but no numbers existed under `results/`. The assignment's
§4.3 invites a naive baseline, so the path was measured once rather than demoted to an
unmeasured stub.

### Audit — what existed, what did not
- **Present (book layer):** `build_trend_portfolio` — one-bar decision lag, inverse-vol
  weights on the on-signal assets, cash otherwise, fixed 4w cadence, moomoo
  commissions, no cost gate; `trend_target_weights`; the `trend_long_flat` /
  `trend_long_short` entries in `STRATEGIES`.
- **Missing (decision + runner):** `TREND_SIGNAL_COLUMNS` (`sma40` →
  `trend_sma40_ratio`, `mom52` → `trend_mom52`) was declared in `src/features.py` and
  `agents/model_design.py` but consumed nowhere. Nothing turned a trend feature into
  the `pred` column the book requires, and no script could run it —
  `scripts/run_backtest.py` hard-codes `strategy="forecast_vol_scaled"` and
  `run_pipeline` always walks a fitted-model grid. No tests. So the path was
  **implemented at the book layer, not implemented end to end**.

### Actions
- `src/features.py` → `trend_rule_signals(panel, rule)`: `pred = +1` where the
  configured trend feature is `> 0`, else `0` (`shorting=True` maps the off-signal to
  `-1`, so the registered `trend_long_short` variant is drivable too); no fitted model,
  no estimation window.
- `scripts/run_trend_baseline.py`: runs both rules through `build_trend_portfolio` on
  the same 2017-2022 validation and 2023-2025 test windows, with the reference book
  (24w vol, 4w rebalance, moomoo, 0 bps) and volatility seeded only from bars before
  each window. Writes `results/trend_baseline/<rule>/{val,test}/`
  (`comparison.csv`, `strategy.csv`, `periods.csv`, `series.csv`) plus `summary.csv` and
  `run_info.json`.
- `tests/test_trend_signals.py`: mapping, per-rule column, shorting variant,
  unknown-rule and missing-column errors (5 tests).
- Command: `python scripts/run_trend_baseline.py`.

### Measured (net of moomoo commissions, no vol-target overlay)
| rule | window | Sharpe | ann return | ann vol | maxDD | total |
| --- | --- | --- | --- | --- | --- | --- |
| sma40 | val 2017-2022 | 0.667 | 6.51% | 10.27% | −19.61% | +46.18% |
| sma40 | test 2023-2025 | 0.922 | 6.97% | 7.63% | −6.54% | +20.35% |
| mom52 | val 2017-2022 | 0.425 | 3.91% | 10.29% | −19.61% | +25.98% |
| mom52 | test 2023-2025 | 1.266 | 10.18% | 7.91% | −7.52% | +30.56% |

Next to the other measured books on the same windows:

| book | val Sharpe | test Sharpe | test total | test maxDD |
| --- | --- | --- | --- | --- |
| ML joint winner (`rff` · abs4 · 4w · QE, w12/vt20) | 1.336 | 0.160 | +2.69% | −10.42% |
| best refit arm (ME, w08/vt10) | 1.185 | 0.204 | +3.70% | −9.87% |
| **sma40 long/flat** | **0.667** | **0.922** | **+20.35%** | **−6.54%** |
| **mom52 long/flat** | **0.425** | **1.266** | **+30.56%** | **−7.52%** |
| EW-16 B&H | 0.699 | 1.314 | +38.42% | −7.79% |
| SPY B&H | 0.657 | 1.523 | +78.86% | −16.88% |

**Gate verdicts on 2023-2025** (bar: Sharpe ≥ 1.3145 = weaker benchmark, maxDD ≥
−7.79% = better benchmark): both rules **fail on Sharpe** (0.922 / 1.266, mom52 short
by 0.048) and **pass on drawdown**. Neither is promoted — the rule is a baseline, not a
candidate, and the ML promotion path is unchanged.

### Reading
- The rule is a fixed, pre-declared specification: no fitted parameters and no
  selection, so its test numbers are a clean out-of-sample benchmark.
- The ML winner's 1.336 is the **maximum of 58,590 candidates** (median 0.406, p90
  0.682, p99 0.904; only 0.5% score above 1.0), i.e. a tail order statistic, while its
  0.160 is what the same book delivered out of sample. The rule's validation score sits
  near the candidate median and its test score is *higher* than its validation score —
  the opposite of the winner's decay. On this window the fitted model added nothing
  over the naive rule.
- Do not over-read it: one test window, both books still lose to EW-16, and the trend
  book is a long/flat, ~8%-vol book rather than a like-for-like replacement for the
  long/short ML book. Two small level differences are deliberate: the rule holds cash
  at 0% (design: no financing modelled) whereas the ML book credits 3% on the de-risked
  sleeve, and both books are flat for the first four weeks of each window because the
  one-bar decision lag has no prior bar inside the slice.
- Because the rule *is* a benchmark, this is a third measurement of 2023-2025 and must
  be counted in the remeasurement tally (README already records two).

### Follow-up
- The report can cite the rule path as **measured** in the §4.3 baseline slot with the
  numbers above; no §11 "implemented-but-unmeasured" caveat is needed.
- If the team still prefers to demote it, the demotion is now an explicit choice rather
  than an omission; nothing in the pipeline depends on the rule path.
- Figures for the baseline were not produced (tables only); add them from `series.csv`
  if the report needs a cumulative-return panel.

## Step 24 — AR(1) forecast baseline implemented and measured (2026-11-09)

### Trigger
Discussion of strategy vs forecast baselines: SMA40/MOM52 are **strategy** baselines
(a rule turned into a book) and cannot fill §4.3's "naive forecast → OOS R²" slot. The
user asked for the AR(1) baseline to be implemented, executed and interpreted.

### Design
- **Regressor must be trailing.** `y_t = a + b * return_Hbar`, where the regressor is
  the return over `t-H..t` (known at the close of `t`) and `y_t = fwd_ret_Hbar` is
  realised over `t+1..t+H+1`. Using the lagged *target* as the regressor would look
  ahead H bars, so it was rejected. For H=1 this is the textbook AR(1) on weekly
  returns; H=4 is the one-lag model on the 4-week return series, matching the ranking
  winner's horizon.
- Per-ticker OLS with intercept; refit blocks mirror `models.walk_forward_predict`:
  QE blocks, training dates strictly before the block, last `embargo = H` dates dropped
  (the last kept label is realised exactly at the first prediction date), rolling
  window 5y = 260 bars. Below `min_obs = 40` pairs the slope is fixed at 0 and the
  ticker's training mean is used.
- `pred_mean` is the AR(0) historical-mean forecast, so the report can quote OOS R²
  against the mean (Campbell-Thompson) as well as against zero (project convention).
- Files: `src/baselines.py`, `scripts/run_forecast_baseline.py`,
  `tests/test_baselines.py` (exact slope recovery, no look-ahead under a future-target
  shock, mean fallback below `min_obs`, R² identity). Suite: **26 passed**.
- Command: `python scripts/run_forecast_baseline.py --horizon 4` (and `--horizon 1`);
  output under `results/forecast_baseline/ar1_h{4,1}/`.

### Measured
| forecast | window | OOS R² vs 0 | OOS R² vs mean | rank IC (t) | direction acc |
| --- | --- | --- | --- | --- | --- |
| AR(1) H=4 | val 2017-2022 | 0.0222 | 0.0132 | 0.112 (5.49) | 57.6% |
| AR(1) H=4 | test 2023-2025 | 0.0487 | −0.0012 | 0.069 (2.45) | 56.7% |
| historical mean H=4 | val / test | 0.0091 / 0.0498 | 0 | — | — |
| AR(1) H=1 | val / test | −0.0069 / 0.0076 | −0.0087 / −0.0017 | 0.034 (1.68) / 0.053 (1.97) | 53.8% / 55.0% |
| ML winner H=4 | val / test | −0.1243 / −0.0387 | — | 0.045 (2.29) / 0.097 (3.75) | 54.7% / 52.1% |

H=4 slopes: mean −0.080, median −0.074, 95% of refits negative. H=1 slopes: mean
0.0006, 34% negative.

### Reading
- The AR(1)'s R² against zero is essentially the **unconditional mean**: the
  historical-mean forecast alone scores 0.0091 / 0.0498, and in test the AR(1) is
  slightly *below* it (−0.0012). The negative slope is a real but tiny four-week
  reversal; it contributes nothing out of sample.
- At H=1 the slope is ~0 and both R² columns are ≈0/negative: weekly returns are
  white noise here, so the H=4 result is drift, not autocorrelation. This is the
  cleanest statement §4.3 can make — a naive forecast has no conditional edge.
- The ML winner is worse than zero on **levels** (−0.1243 val, −0.0387 test) but has
  the best test rank IC (0.097). The book consumed cross-sectional ranking and
  volatility, not calibrated return levels; quoting only R² would make a
  one-parameter mean-reversion model look superior to the whole pipeline. Report both.
- Do not over-read: 16 ETFs, weekly, one span; the AR(1)'s validation IC (0.112) is
  mostly persistent per-ticker drift carried by `pred_mean` (mean-only IC 0.091 val /
  0.072 test), not timing skill, and the full model decays to 0.069 in test.

### Follow-up
- §4.3 now has a genuine forecast baseline to cite; no "implemented-but-unmeasured"
  caveat is needed for it either.
- When the report quotes the ML model's OOS R², quote it against the historical mean
  if that number is produced, not only against zero — otherwise the baseline comparison
  flatters the AR(1) and understates how badly calibrated the ML levels were.
- `--horizon 12` is supported by the same script if the report needs the 12-bar case.

---

## Results Reset — Fresh Experiment Baseline (2026-11-09)

### Action

All generated artifacts under `results/` were cleared so the next experiment can be
treated as a fresh run. Removed top-level outputs included prior base-study results,
macro-study results, prediction caches, plots, trend baselines, and forecast
baselines.

### State

- `results/` is now the clean output location for the next run.
- Any previously recorded result files, rankings, plots, cached predictions, draws,
  or baseline outputs should be considered **stale** unless regenerated after this
  reset.
- Historical notes in this document may still describe earlier experiments for
  provenance, but they should not be treated as current generated artifacts.
- New report numbers should come only from results regenerated after this reset.

---

## RFF Removal — Active Model Space Reset (2026-11-09)

### Action

Removed the RFF model family from the active implementation and study design. The
fresh-run model space is now Ridge, Lasso, and Elastic Net only.

### State

- Active model grid: 5 Ridge specs + 4 Lasso specs + 12 Elastic Net specs = **21**
  specs.
- Active joint ranking size: 21 specs x 14 valid `(rebalance, look-ahead, refit)`
  combos x 45 book cells = **13,230** candidates.
- RFF-specific fields (`rff_dim`, `gamma`) were removed from model specs, cache
  provenance, study rows, winner metadata, run parameters, and coefficient reporting.
- Earlier RFF-related notes remain historical provenance only and are stale for any
  experiment regenerated after this reset.
