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

### Run the tests

```bash
uv run pytest
```

Tests import from `src/` directly, so no install step is needed; `pyproject.toml` sets `pythonpath` and `testpaths`.
