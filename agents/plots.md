# plots.md — report figure suite

Report figures are generated **inside the experiment scripts** (`src/plots.py`,
invoked by `pipeline.evaluate_all` right after the tables are written), so there
is no separate "plot image" script to run afterwards:

```
results/<run-folder>/
  plots/                       # 20 PNGs (10 figures x test + validation windows)
    test_01_cumulative_returns.png
    test_02_gross_vs_net.png
    ...
    validation_01_cumulative_returns.png
    ...
  series/                      # raw per-bar / per-ticker series behind the figures
    test_book.csv              # gross_ret, turnover, cost, net_ret, exposure, ...
    test_weights.csv           # per-date x per-ticker weights held (long +, short -)
    test_orders.csv            # per-date x per-ticker order USD (signed: buy/sell)
    test_benchmarks.csv        # Equal-weight / SPY B&H returns
    val_*.csv                  # same five files for the validation window
```

**Scope (per project owner, 2026-11-02):** the suite is retained for the **three
best-selected runs only** — one per validation phase plus the test draw. All
other folders are tables-only; their `plots/` + `series/` were removed and are
regenerated on demand by re-running the row's command.

| phase | run folder | figures cover |
| --- | --- | --- |
| V1 validation (2017–2019) | `results/weekly_fg_roll5y_val3y_mrefit_grid144_rweekly_2020-2020__v1_la1_freq1w_mrefit/` | `validation_*` = 2017–2019 selection window (net Sharpe 1.337); `test_*` = Jan-2020 byproduct |
| V2 validation (2020–2022) | `results/weekly_fg_roll5y_val3y_mrefit_enet_a0.001_r0.5_vt0.1c1rf3_rweekly_2020-2022__v2_volwin_12w/` | `test_*` = 2020–2022 (net Sharpe 0.657); `validation_*` = 2017–2019 |
| Test (2023–2025) | `results/test_2023-2025_promoted/` | `test_*` = 2023–2025 (net Sharpe 1.172, gate FAIL); `validation_*` = 2020–2022 |

The `series/` files let a report re-plot or extend any figure without re-running
the simulation.

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
2020–2022 / 2023–2025.

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

Plots are produced automatically on every run:

```bash
python scripts/run_backtest.py ...        # any run -> plots/ + series/
python scripts/run_study.py v1            # all 14 V1 combo folders get plots
python scripts/run_study.py v2            # all 11 V2 row folders get plots
```

If a folder should **not** carry the suite (non-best-selected runs), remove its
`plots/` + `series/` after the run; the tables are unaffected. Old folders (from
before this step) can be refreshed by re-running their exact command into the
same `--out-dir` / phase — the run is deterministic, so the existing tables are
rewritten byte-identical and only `plots/` + `series/` are (re)created (verified
2026-11-02 on the V1 winner combo, the V2 winner volwin_12w row, and
`test_2023-2025_promoted`).
