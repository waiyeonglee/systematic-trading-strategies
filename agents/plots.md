# plots.md — report figure suite

Report figures are generated **inside the experiment scripts** (`src/plots.py`,
invoked by `pipeline.evaluate_all` right after the tables are written), so there
is no separate "plot image" script and **no flag to pass** — the pipeline decides
the report phase automatically from the run's test window.

## Where the figures live

```
results/plots/
  validation_phase_1/           # V1 winner combo (10 PNGs, no window prefix)
  validation_phase_2/           # V2 winner row    (10 PNGs, no window prefix)
  test/                         # promoted test draw (10 PNGs, no window prefix)
    01_cumulative_returns.png   # ... 10_oos_r2_by_year.png
```

- The canonical report figures live **only** under `results/plots/<phase>/`,
  named `01_<stem>.png` … `10_<stem>.png` (the folder name already identifies the
  phase, so the `test_`/`validation_` window prefix is stripped).
- The per-run `plots/` folder is **transient**: `evaluate_all` renders it, copies
  the phase's window into `results/plots/<phase>/`, then removes it — individual
  run folders never keep stale plots.
- Per-run **series** (the data behind the figures) still stay in each run folder
  under `series/` (8 CSVs: book / weights / orders / benchmarks per window), so a
  report can re-plot or extend any figure without re-running the simulation.

## Which run feeds which phase (automatic)

| phase | published when the run's test window is | figures held |
| --- | --- | --- |
| `validation_phase_1` | `2020-01-01` … `2020-01-31` (V1 winner combo) | the combo's **validation** window = 2017–2019 selection (net Sharpe 1.337); its `test_*` is only a Jan-2020 byproduct |
| `validation_phase_2` | `2020-01-01` … `2022-12-31` (V2 winner row) | the row's **test** window = 2020–2022, the span the phase-2 ablation was ranked on (net Sharpe 0.657) |
| `test` | `2023-01-01` … `2025-12-31` (promoted draw) | the draw's **test** window = 2023–2025 (net Sharpe 1.172, gate FAIL) |

Runs whose test window matches none of these produce no report figures at all
(tables + series only). The mapping lives in `pipeline.report_phase_for`
(`REPORT_PHASES` in `src/pipeline.py`).

## Why two windows per run

`run_pipeline` measures the book on the **test** window and selects the model on
the **validation** window, so both windows are plotted:

| window | book | used for |
| --- | --- | --- |
| `test_*` | vol-targeted book (what would have been traded) | the run's headline measurement |
| `validation_*` | overlay-free book (`apply_vol_target=False`, the selection basis) | the window that chose the model |

For the executed study: the V1 winner folder → `validation_*` = 2017–2019
selection window (+ a 4-bar Jan-2020 byproduct as `test_*`); the V2 winner and
promoted test draws → `validation_*` = 2017–2019 / 2020–2022 and `test_*` =
2020–2022 / 2023–2025. Only the phase's window (see the table above) is kept in
`results/plots/<phase>/`; the other window's figures are discarded.

## The ten figures (per window)

| # | file | shows | why it belongs in the report |
| --- | --- | --- | --- |
| 01 | `cumulative_returns` | net wealth of the strategy vs **Equal-weight-16** and **SPY B&H** (growth of 1) | the core result: does the book beat its benchmarks over time, and where |
| 02 | `gross_vs_net` | cumulative gross vs net wealth, drag shaded | separates **alpha (gross)** from **cost drag**; the shaded gap is the total cost + rf/lev sleeve |
| 03 | `drawdown` | underwater curves, strategy + both benchmarks | risk view; supports the §3.2 gate argument (maxDD vs bars) |
| 04 | `costs_and_trades` | cumulative transaction cost (step) + turnover bars with **net-buy ▲ / net-sell ▼** markers on rebalance bars and long/short regime shading | the "pointer where the action was": when the book bought, sold, and whether it was net long or short |
| 05 | `exposure` | gross / net exposure %, vol-target scale (twin axis), leverage-blocked markers | shows the 10% vol-target de-risking and that leverage caps never bind here |
| 06 | `weights_held` | per-ticker weights held over time (longs stack up, shorts down) | what the portfolio actually held — sector/cross-asset tilts and rotation |
| 07 | `weights_heatmap` | date x ticker weight heatmap | compact read of which assets carried the risk at which times |
| 08 | `holdings_counts` | n holdings / n long / n short | breadth of the book over time (long-short balance) |
| 09 | `annual_returns` | calendar-year net returns, strategy vs benchmarks | year-by-year consistency (vs the OOS R² by year) |
| 10 | `oos_r2_by_year` | forecast OOS R² per calendar year | the model-quality view behind the book — shows whether signal decayed |

Every PNG is self-contained for a report: header = model + window dates,
subtitle = net Sharpe / maxDD / total / ann ret / ann vol (+ "overlay-free
(selection basis)" on validation figures) and the run tag.

## Data recorded for plotting (portfolio.py)

`build_forecast_vol_portfolio` / `build_trend_portfolio` now attach two
per-date, per-ticker frames to the book (`object.__setattr__`, so they never
collide with column access):

- `weights_wide` — the weights **held into the next bar** (after each rebalance);
- `orders_wide` — executed order value in USD, **signed** (positive = buy,
  negative = sell, zero on non-rebalance bars; a flip long→short is a single
  order whose sign is the net leg).

These feed figures 04/06/07 and are also dumped to `series/`.

## Regeneration

Plots are produced automatically on every run — **no flag, no second script**:

```bash
python scripts/run_backtest.py --test-start 2023-01-01 --test-end 2025-12-31 ...   # -> results/plots/test/
python scripts/run_backtest.py --test-start 2020-01-01 --test-end 2022-12-31 ...   # -> results/plots/validation_phase_2/
python scripts/run_backtest.py --test-start 2020-01-01 --test-end 2020-01-31 ...   # -> results/plots/validation_phase_1/
```

Re-running a designated run overwrites that phase folder with identical figures
(the run is deterministic, and existing tables are rewritten byte-identical —
verified 2026-11-02 on all three designated runs). Non-phase runs simply write no
figures; per-run `plots/` is always removed by the pipeline.
