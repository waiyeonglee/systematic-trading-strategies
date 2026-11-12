# current_state.md - project history and handover notes

Current source-of-truth state for the experiment cycle. Both feature blocks have been
ranked on the 2019-2022 validation window and drawn once on the 2023-2025 test window;
their tables and figures are the current results.

---

## Dataset

- Raw market data are downloaded from Yahoo Finance at daily frequency using adjusted
  prices and then resampled to Friday-ending weekly bars.
- Weekly bars use first adjusted open, maximum adjusted high, minimum adjusted low,
  final adjusted close, and summed volume across each Friday-ending week.
- The fixed data window starts in 2004 and ends at the beginning of 2026. Pre-2011 data
  is warmup history for long-lookback features.
- The main modelling period starts in 2011, aligned with the availability of the Fear
  & Greed Index.
- The tradable universe contains 16 ETFs, with `SPY` used as the market benchmark and
  a market-level feature source.

## Features And Labels

- Base/non-macro runs use ETF technical features and the `SPY` return helper.
- Macro runs use 19 firm-level technical features interacted with six macro states:
  a constant, monthly Fear & Greed, `tbl`, `tms`, `dfy`, and `dfr`.
- Macro state variables are lagged by one month before being joined to weekly rows to
  avoid look-ahead bias.
- Labels are forward adjusted close-to-close returns for 1-, 4-, and 12-week horizons.
- Signals are formed after observing the complete weekly bar at week `t`; trades are
  assumed to execute at the close of week `t+1`.

## Active Model Space

- Active model family: Elastic Net only.
- Active alpha grid: `1e-4`, `1e-3`, `1e-2`, `1e-1`.
- Active `l1_ratio` grid: `0.0`, `0.25`, `0.5`, `0.75`, `1.0`.
- `l1_ratio=0.0` represents the Ridge-like endpoint; `l1_ratio=1.0` represents the
  Lasso-like endpoint.
- Removed from the active search: RFF, separate Ridge, separate Lasso, and weekly
  refits. Only the Elastic Net family is instantiated.

## Active Refit Space

- Active refit frequencies: monthly (`ME`) and quarterly (`QE`) only.
- Weekly refit (`W-FRI`) is removed from the active study design.
- Valid `(rebalance, look-ahead, refit)` combinations: 11.
- Book cells: 45.
- Active ranking size: `20 x 11 x 45 = 9,900` candidates.
- Forecast cache groups: 6.

## Split And Holdout

- Warmup period: pre-2011 history.
- Base validation window: 2019-01-01 to 2022-12-31.
- Macro validation window: 2019-01-01 to 2022-12-31.
- Final holdout window: 2023-01-01 to 2025-12-31.
- Model and strategy selection uses validation results only.
- The 2023-2025 test window has been drawn once per block, after the
  validation-selected configuration was locked; it selects nothing.

## Results State

- Both blocks are ranked (9,900 candidates each) and drawn on the test window.
- Validation winners: micro `enet_a0.001_r1`, absolute4 / 4w / ME, book
  `w04_vt10_rf3_cap1.0`; macro `enet_a0.0001_r1`, absolute12 / 4w / ME, book
  `w04_vt10_rf3_cap1.0`.
- Artefacts live under `results/_study_{micro,macro}/` (ranking, winner, draws,
  figures, `comparison/`); the standalone AR(1) baseline under
  `results/forecast_baseline/`; the micro/macro validation comparison under
  `results/validation_comparison/`.
- Cite report numbers only from these artefacts, and only after the command that wrote
  them (see `README.md`) — the prediction cache makes a re-run deterministic.

## Verification State

- The test suite passes.
- Active sanity checks confirmed:
  - 20 Elastic Net specs.
  - refits restricted to `ME` and `QE`.
  - 11 valid strategy/refit combinations.
  - 9,900 ranking candidates.
  - `W-FRI` rejected as an invalid active refit cadence.
