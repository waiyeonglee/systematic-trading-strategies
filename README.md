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

### Run the tests

```bash
uv run pytest
```

Tests import from `src/` directly, so no install step is needed; `pyproject.toml` sets `pythonpath` and `testpaths`.
