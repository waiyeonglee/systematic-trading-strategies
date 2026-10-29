# systematic-trading-strategies
DSA5205 Project 2 - Designing, backtesting, and evaluating systematic trading strategies using machine learning and quantitative financial analysis.

## Environment setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Install uv first, then use the commands below from the project root.

### Install dependencies

```bash
# Install runtime and dev dependencies (creates .venv if missing)
uv sync

# Install runtime dependencies only, without the dev group
uv sync --no-dev
```

### Add / remove dependencies

Runtime dependencies go into `[project].dependencies`; dev-only dependencies go into `[dependency-groups].dev`.

```bash
# Add a runtime dependency
uv add <package>

# Add a pinned version
uv add "<package>==<version>"

# Add a dev-only dependency
uv add --dev <package>

# Remove a runtime dependency
uv remove <package>

# Remove a dev-only dependency
uv remove --dev <package>
```

> If you remove a dev dependency without `--dev`, uv looks in the runtime dependencies and will not find it.

### Run code inside the environment

```bash
uv run python script.py
uv run jupyter lab
```

### Fetch market data

Use `src/fetch_data.py` to download sector ETF and SPY OHLCV data. The shared date range is configured in `config.yaml` and is also used by the EDA notebook's Fear & Greed Index section.

```bash
# Daily data
uv run python src/fetch_data.py --frequency daily

# Weekly Friday-ending data
uv run python src/fetch_data.py --frequency weekly
```

Outputs are saved under `data/stocks_daily/raw/` and `data/stocks_weekly/raw/`.

## Daily and weekly technical features

The feature analysis reads daily OHLCV files for daily features and Friday-ending
weekly OHLCV files for weekly features. `fetch_data.py --frequency weekly` creates
the weekly bars by resampling the daily source data (Open=first, High=max, Low=min,
Close=last, Volume=sum). Every indicator lookback is measured in bars, so a 14-bar
RSI means 14 daily observations in the daily dataset and 14 weekly observations in
the weekly dataset. Volatility is annualized with 252 or 52 periods per year,
respectively. TA-Lib computes RSI, MACD, Bollinger Bands, moving averages, rolling
standard deviation, and rolling highs. Returns/momentum and volume changes are
calculated from bar-to-bar changes. Return features are consistently named for their
lookback: 1/2/3/4/12/26/52 bars. Other features include moving-average ratios,
12/26-bar volatility, 12-bar drawdown, RSI-14, MACD, Bollinger position, four-bar
volume change, the Fear & Greed Index (0–100), and SPY's one-bar return. Daily
features use the SPY daily return for that date; weekly features use the SPY return
for that weekly bar. SPY values are aligned by date without using a later bar.
Sentiment is aligned by carrying its latest dated observation forward to each price
bar; no future sentiment values are backfilled. The index CSV is read from
`data/fear_and_greed_index/fear-greed.csv` by default.

The feature script only builds a feature DataFrame and plots a Pearson correlation
matrix. It does not fit models or select a subset of features. Initial rows with
insufficient lookback history are retained with missing feature values.

Run from the project root:

```bash
uv run python src/fetch_data.py --frequency daily
uv run python src/fetch_data.py --frequency weekly
uv run python src/feature_selection.py
```

Outputs are written separately to `reports/feature_selection/daily/` and
`reports/feature_selection/weekly/`:

- `features.csv` contains one row per date and sector ETF, with technical features
  plus `label_forward_return_1bar`, the next bar's close-to-close return. The label
  is not included in `FEATURES`, so it remains separate from the predictor set. Its
  final value per ticker is missing because there is no following bar.
- `feature_label_correlation.csv` and `.png` contain the feature-plus-label
  correlation matrix and its heatmap, calculated across all dates with available
  labels for exploratory analysis. Use the separate chronological split function
  for model training, feature selection, and out-of-sample evaluation.

## Train and backtest on real data

`main.py` runs the whole model pipeline on the real sector-ETF data: it builds the
19 features above, joins the cross-sectional next-bar label, and backtests a single
penalised logistic regression (`lasso_C1.0`, fixed in `GRID`) that is refitted every
week over a rolling window (default two years: 504 daily bars or 104 weekly bars)
with a one-bar embargo. The months before the test period form the validation
window, which only drives the gate sweep. All randomness is seeded with `SEED`
(5205), so repeated runs give identical output.

### Setup

```bash
uv sync                        # create the environment from pyproject.toml
uv run pytest                  # optional: confirm the environment works (29 tests)
```

### Options

| Flag | Default | Meaning |
| --- | --- | --- |
| `--frequency {daily,weekly}` | `weekly` | Bar frequency; must match a fetched `data/stocks_<freq>/raw/` folder |
| `--test-start` | `2025-01-01` | First date of the reported (out-of-sample) period |
| `--test-end` | `2025-12-31` | Last date of the reported period |
| `--window-years` | `2.0` | Rolling training window length, in years of bars |
| `--val-months` | `24` | Validation window before the test period, in months |
| `--compare-refit` | off | Also compare every-bar refitting against weekly refitting |

### Sample runs

Fetch the data once first — it needs network access and writes the CSVs everyone
else will read:

```bash
uv run python src/fetch_data.py --frequency weekly
uv run python src/fetch_data.py --frequency daily
```

Then pick a config. Each command below is self-contained:

```bash
# 1. Quick start: weekly bars, one test year (~1-2 min)
uv run python main.py --frequency weekly

# 2. Daily bars instead of weekly (same splits, ~5 min)
uv run python main.py --frequency daily

# 3. Six-year test period, weekly bars (~3 min)
uv run python main.py --frequency weekly --test-start 2020-01-01

# 4. Four-year test period, weekly bars (~2 min)
uv run python main.py --frequency weekly --test-start 2022-01-01

# 5. Custom test window
uv run python main.py --frequency weekly --test-start 2016-01-01 --test-end 2019-12-31

# 6. Shorter training window and validation window (~3 min)
uv run python main.py --frequency weekly --test-start 2020-01-01 \
    --window-years 1 --val-months 6

# 7. Longer training window (~3 min)
uv run python main.py --frequency weekly --test-start 2020-01-01 --window-years 3

# 8. Refit cadence study: every-bar vs weekly refitting (~5-10 min, daily only)
uv run python main.py --frequency daily --compare-refit
```

Do not shorten `--val-months` below ~24 if you plan to tune the gate: a 6-month
validation window has a Sharpe standard error above 1.0, so the sweep just reports
noise.

The panel starts in **2011** (the Fear & Greed index's first date), so keep
`--test-start` at **2016 or later** — with 24 months of validation, earlier starts
leave fewer training bars than the rolling window needs (2015 gives exactly 104, so
the first refits lose one bar to the embargo; 2014 gives only 52 and triggers the
`WARNING` that `main.py` prints).

### Outputs

Each run prints, in order: the split sizes, the validation ranking, the validation
gate sweep, the test comparison against equal-weight, the per-gate test comparison,
the SPY reference table, gross-vs-net strategy stats, forecast metrics (AUC, rank
IC) and CAPM alpha/beta.

What to expect: pooled AUC lands around **0.49–0.51** and rank IC near zero, so the
selected portfolio roughly tracks equal-weight before costs and trails it after
costs. That is the current finding, not a broken setup — treat a large AUC (>0.55)
as a leak to investigate.

Figures go to their own `figures/<run_tag>/` folder so configurations never
overwrite each other:

```
<frequency>_roll<window>y_val<months>m_<refit>_<model>_g<gate>_<test years>

figures/weekly_roll2y_val24m_wkrefit_lassoC1.0_g0.00_2020-2025/
    cumulative_returns.png            # gates vs equal-weight
    cumulative_returns_vs_spy.png     # gates vs equal-weight vs SPY
    drawdown.png
    gross_vs_net.png
    turnover_vs_gate.png
    sharpe_by_model.png
    sharpe_by_period.png
    coefficients.png
    refit_cadence.png                 # only with --compare-refit
```

Runs that already exist, for reference while you compare results:

```
figures/weekly_roll2y_val24m_wkrefit_lassoC1.0_g0.00_2020-2025
figures/weekly_roll2y_val24m_wkrefit_lassoC1.0_g0.00_2022-2025
figures/weekly_roll2y_val24m_wkrefit_lassoC1.0_g0.00_2025-2025
figures/weekly_roll1y_val6m_wkrefit_lassoC1.0_g0.00_2020-2025
figures/daily_roll2y_val24m_wkrefit_lassoC1.0_g0.00_2025-2025
```

Re-running the same command overwrites that folder; a different config creates a
new one.

### Run the tests

```bash
uv run pytest
```

Tests import from `src/` directly, so no install step is needed; `pyproject.toml` sets `pythonpath` and `testpaths`.
