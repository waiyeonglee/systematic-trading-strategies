# plots.md — report figure suite

Report figures are generated **inside the experiment scripts** (`src/plots.py`,
invoked by `pipeline.evaluate_all`), so there is no separate "plot image" script and
**no flag to pass**. Under the joint-ranking scheme only the **single pre-registered
test draw** produces report figures; the ranking step
(`scripts/run_study.py rank`) writes one CSV and nothing else.

## Where the figures live

```
results/plots/
  test/                         # promoted test draw (10 PNGs, no window prefix)
    01_cumulative_returns.png   # ... 10_oos_r2_by_year.png
  val/                          # the same 10 figures for the validation window the
    01_cumulative_returns.png   # winner was chosen on (2017-2022)
```

- The canonical report figures live **only** under `results/plots/<phase>/`, named
  `01_<stem>.png` … `10_<stem>.png` (the folder name already identifies the phase,
  so the `test_` / `validation_` window prefix is stripped).
- The per-run `plots/` folder is **transient**: `evaluate_all` renders it, copies
  each measured window into `results/plots/<phase>/`, then removes it — individual
  run folders never keep stale plots.
- Per-run **series** (the data behind the figures) still stay in each run folder
  under `series/` (book / weights / orders / benchmarks per window), so a report
  can re-plot or extend any figure without re-running the simulation.

## Which run publishes figures (automatic)

| run | test window | publishes |
| --- | --- | --- |
| the pre-registered test draw | `2023-01-01` … `2025-12-31` | `results/plots/test/` (the run's `test_*` figures) **and** `results/plots/val/` (its `validation_*` figures) |
| anything else (study rows, ad-hoc windows) | — | nothing; tables + series only |

The mapping lives in `pipeline.report_phases_for` (`REPORT_TEST_WINDOW` /
`REPORT_PHASES` in `src/pipeline.py`).

## Why two windows per run

`run_pipeline` measures the book on the **test** window and selects the model on the
**validation** window, so both windows are plotted inside the run folder:

| window | book | used for |
| --- | --- | --- |
| `test_*` | the configured book, overlay applied | the run's headline measurement |
| `validation_*` | the same book on 2017-2022, overlay applied (`selection["overlay_books"]`) | the window that chose the model, reported with the book the joint ranking scored |

Both figure sets are kept (`results/plots/test/`, `results/plots/val/`) so the
validation result — the number that selected the model — is inspectable next to the
headline. The overlay-free book (`apply_vol_target=False`) is still what the
in-pipeline ranking scores; it is simply not plotted, because the joint ranking picks
the winner on the overlaid book. The ranking step (`run_study.py rank`) plots nothing
at all — it emits `results/_study/validation_ranking.csv`.

## The ten figures (per window)

| # | file | shows | why it belongs in the report |
| --- | --- | --- | --- |
| 01 | `cumulative_returns` | net wealth of the strategy vs **Equal-weight-16** and **SPY B&H** (growth of 1) | the core result: does the book beat its benchmarks over time, and where |
| 02 | `gross_vs_net` | cumulative gross vs net wealth, drag shaded | separates **alpha (gross)** from **cost drag**; the shaded gap is the total cost + rf/lev sleeve |
| 03 | `drawdown` | underwater curves, strategy + both benchmarks | risk view; supports the §3.2 gate argument (maxDD vs bars) |
| 04 | `costs_and_trades` | cumulative transaction cost (step) + turnover bars with **net-buy ▲ / net-sell ▼** markers on rebalance bars and long/short regime shading | the "pointer where the action was": when the book bought, sold, and whether it was net long or short |
| 05 | `exposure` | gross / net exposure %, vol-target scale (twin axis), leverage-blocked markers | shows when the vol-target overlay actually de-risked (and that the leverage cap never binds here — the scale is clamped at 1.0) |
| 06 | `weights_held` | per-ticker weights held over time (longs stack up, shorts down) | what the portfolio actually held — sector/cross-asset tilts and rotation |
| 07 | `weights_heatmap` | date x ticker weight heatmap | compact read of which assets carried the risk at which times |
| 08 | `holdings_counts` | n holdings / n long / n short | breadth of the book over time (long-short balance) |
| 09 | `annual_returns` | calendar-year net returns, strategy vs benchmarks | year-by-year consistency (vs the OOS R² by year) |
| 10 | `oos_r2_by_year` | forecast OOS R² per calendar year | the model-quality view behind the book — shows whether signal decayed |

Every PNG is self-contained for a report: header = model + window dates,
subtitle = net Sharpe / maxDD / total / ann ret / ann vol, the volatility overlay as
configured and how many bars it actually scaled (e.g. `vol target 20% cap 1 (scaled
on 4/313 bars)`), plus the run tag.

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
python scripts/run_study.py test     # the pre-registered draw -> results/plots/test/ + val/
# or the equivalent single draw through the backtest CLI:
python scripts/run_backtest.py --test-start 2023-01-01 --test-end 2025-12-31 --window-val 6y ...
```

Re-running the test draw overwrites `results/plots/test/` and `results/plots/val/`
with identical figures (the run is deterministic: `utils.config.DEFAULT_SEED`; it
takes ≈6 s because
the models are cheap to refit). Any other span writes no figures, and the per-run
`plots/` folder is always removed by the pipeline.
