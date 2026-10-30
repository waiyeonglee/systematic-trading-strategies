# systematic-trading-strategies

DSA5205 Project 2 - Designing, backtesting, and evaluating systematic trading strategies
using machine learning and quantitative financial analysis.

Nine US sector ETFs (XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY) are ranked every bar by a
predicted **excess return** (the asset's next-bar return minus the cross-sectional median).
Two portfolio construction rules read those predictions:

- **`top_k`** holds the top `k` at equal weight and only swaps a holding when the challenger
  is expected to beat it by more than a gate that covers the trading cost.
- **`forecast_vol_scaled`** divides each prediction by its forecast volatility and trades the
  scaled signals long and short at 100% gross exposure.

`portfolio.strategy` picks one rule or `both` (the default), which trades the same
predictions with both rules and compares them in the same tables and figures.

Models are penalised linear regressions refitted walk-forward, scored on a validation window,
and evaluated once on the test period.

## How it works

```mermaid
flowchart LR
    cfg["config.yaml"] --> fetch["main.py fetch"]
    fetch --> raw["data/raw/frequency/TICKER.csv"]
    raw --> feats["main.py features"]
    feats --> fdir["data/features/frequency/"]
    fdir --> bt["main.py backtest"]
    bt --> res["results/run_tag/"]
```

Every bar a model is fitted on the past only, and the split is strictly chronological:

```mermaid
flowchart LR
    train["train: everything before the validation window"] --> val["validation: window_val before test_start"]
    val --> test["test: test_start to test_end"]
```

- **Validation** ranks model specifications (net Sharpe) and sweeps the gate.
- **Test** is scored once with the selected model and gate; nothing is tuned on it.
- Inside both windows the model is *refitted* every `refit_every` period:

```mermaid
flowchart LR
    past["bars dated before the block"] --> embargo["drop the last embargo bar"]
    embargo --> window["keep the most recent window_train bars"]
    window --> fit["fit StandardScaler + regression"]
    fit --> predict["predict every row in the block"]
```

The scaler lives inside the model pipeline, so it is refitted on each training window
rather than on the whole sample - no future mean or standard deviation leaks in.

## The `top_k` trading strategy

`k` names are held at equal weight, with a cost-aware rule for changing them. The whole
decision for one bar lives in `portfolio.build_portfolio`:

```mermaid
flowchart TD
    A["bar t: predicted excess return for every ETF"] --> B["keep holdings still present in the universe"]
    B --> C{"fewer than k holdings?"}
    C -- yes --> D["add the highest-ranked names not held"]
    C -- no --> E["start the swap loop"]
    D --> E
    E --> F{"best outsider minus weakest holding greater than gate?"}
    F -- yes --> G["swap weakest holding for the outsider"]
    G --> F
    F -- no --> H["weight each holding 1/k, everything else 0"]
    H --> I["turnover = sum of absolute weight changes versus the previous bar"]
    I --> J["cost = turnover x cost_bps / 10000"]
    J --> K["gross = sum of weight x next-bar return"]
    K --> L["net = gross - cost"]
```

Why the gate exists:

| Gate | Behaviour | Turnover |
| --- | --- | --- |
| `0.0` | swap whenever the ranking changes | highest, pays the most cost |
| `2 * cost_bps / 10000` | swap only if the expected gain covers one round trip | moderate |
| large (e.g. `0.01`) | ignore small ranking changes | lowest, may lag the signal |

Because the gate is in *return* units, a value of `0.001` means "only swap if the newcomer is
expected to earn 0.1% more". The sweep in `portfolio.top_k.gate_sweep` replays the
validation window at each gate so the trade-off between turnover and Sharpe is visible
before the test period is touched.

Other details that matter:

- **Equal weight**, so no position sizing model is involved: the strategy is purely a
  selection and timing rule.
- **First bar is a full entry** (turnover = 1.0), so the cost model charges for building the
  book.
- **Weight drift inside a period is ignored**; turnover is measured between rebalances only.
- **Missing forward returns** for a held name are treated as zero, so a stale ticker cannot
  inject a return that was never tradable.
- The forward return used to score the portfolio is the next bar's close-to-close return, the
  same quantity the model predicts.

Adding another strategy only needs a function `f(preds, config) -> portfolio frame`
registered in `portfolio.STRATEGIES` plus a matching block in `config.yaml`; the pipeline
dispatches to it by name.

## The `forecast_vol_scaled` trading strategy

`portfolio.strategy: forecast_vol_scaled` implements Equation (15) of Chinco, Clark-Joseph &
Ye (2019): each predicted return is divided by its own forecast standard deviation, and the
signals are then scaled so their absolute values add up to 100% gross exposure. The book is
long *and* short, and the long and short legs are not forced to balance.

```mermaid
flowchart TD
    A["bar t: predicted excess return for every ETF"] --> B["drift the book: yesterday's returns<br/>move every weight"]
    B --> C{"rebalance date?<br/>daily, weekly or monthly"}
    C -- no --> D["keep the drifted weights, no trades, no cost"]
    C -- yes --> E["past predictions only: shift one bar"]
    E --> F["rolling 60-bar std per ETF<br/>needs 20 observations, floored at 1e-5"]
    F --> G["signal = prediction / forecast std"]
    G --> H["divide by the sum of absolute signals<br/>so gross exposure is 100%"]
    H --> I["order = (target - drifted) x equity<br/>commission + cost_bps per order"]
    D --> J["gross = sum of weight x next-bar return"]
    I --> J
    J --> K["net = gross - cost / equity"]
```

Why the forecast volatility comes from predictions and not from returns:

| Choice | Reason |
| --- | --- |
| rolling std of **past out-of-sample predictions** | the paper scales the signal by the uncertainty of *that signal*; realised return volatility would measure the asset, not the forecast |
| shifted by one bar, so the prediction of `t` is excluded | the forecast traded on `t` cannot sit inside its own denominator |
| `vol_min_periods` of 20 before an ETF is tradable | a 60-bar window on three observations is noise, so early bars stay in cash |
| history is never reset at a monthly refit | a refit changes the model, not the fact that older predictions are out of sample |
| same window carried across refits | the rolling std simply keeps consuming predictions, so no artificial 60-bar warm-up every month |

Transaction costs are charged per executed order, never on the portfolio value:

- **Moomoo**: `max(0.03% x notional, USD 0.77)` plus 9% GST on that commission.
- **IBKR**: `min(USD 1.00, 1% x notional)`.
- **Every order**: an extra `cost_bps` (default 5 bps) of the traded notional, added exactly
  once in `portfolio.order_cost_usd`.

Only `vol_window`, `vol_min_periods`, `rebalance`, `broker` and `cost_bps` are configurable.
Everything else is a literal inside the function that uses it: the broker rates, fee floors
and GST rate in `portfolio.moomoo_commission` and `portfolio.ibkr_commission`, the 1e-5
volatility floor in `portfolio.target_weights`, the (zero) minimum order size in
`portfolio.orders_for_target` and the 100% gross exposure. The simulated account size is
`portfolio.initial_capital_usd`, because the commission floors are absolute dollars. An order
that closes one position and opens the opposite one is a single order covering both legs.

Other details that matter:

- **Weight drift is corrected by rebalancing, not continuously.** Between rebalances every
  position is left alone, so a winner grows past its target weight and a loser shrinks below
  it; that is what the `gross_exposure`/`net_exposure` columns show drifting. On each
  rebalance date the book is marked to market and traded back to the freshly computed target,
  which restores gross exposure to 100%. Because the positions move, this correction costs
  money even when the signal has not changed, which is why the cadence matters:
  `rebalance` takes `"daily"` (every bar), `"weekly"` (first bar of each week) or
  `"monthly"` (first bar of each month, the default). Monthly trades ~12 times a year and
  weekly ~52, whereas `"daily"` re-fixes the drift on every bar and paid about 22% a year in
  costs on the daily test run.
- **Exposure is normalised, not capped**: every eligible asset is scaled by the same factor,
  so the absolute weights add up to 100% at each rebalance and the whole book is invested.
- **Turnover compares targets with the drifted book**, so at a rebalance the trade is the
  drift plus the signal change, and it is zero on every bar in between.
- **Predictions are excess returns** (the cross-sectional median is subtracted), which is a
  shift in the same units as a simple return, so dividing by a dispersion of predictions is
  meaningful and no return conversion is needed.
- **Costs are per order**: nine ETFs rebalanced monthly pay nine commissions twelve times a
  year, so check `ann_cost_drag` in `strategy_<name>.csv` before reading the Sharpe ratio.
- **No financing, borrow or margin interest** is modelled: uninvested cash earns zero and
  gross exposure stays at 100%, so no leverage is assumed.

## Layout

```
main.py            command-line entry point (fetch | features | backtest)
config.yaml        local run settings (git-ignored; copy config.yaml.example)
data/              raw bars, feature tables and the alternative-data index (see data/README.md)
src/
  data.py          download and load prices, assemble the feature + label panel
  features.py      19 technical features and the feature/label correlation report
  labels.py        forward return and the cross-sectional excess-return target
  splits.py        date-window and sliding-period train/validation/test splits
  models.py        model specifications and walk-forward prediction
  portfolio.py     predicted returns -> portfolio, both strategies plus broker costs
  metrics.py       performance, benchmark, CAPM and forecast metrics
  plots.py         every figure written by the pipeline
  pipeline.py      model selection, test run and evaluation in one call
  utils/           paths, settings, validation, plotting, artifacts and CLI helpers
```

## Environment setup

This project uses [uv](https://docs.astral.sh/uv/) for dependency management. Install uv
first, then use the commands below from the project root.

```bash
# Install runtime and dev dependencies (creates .venv if missing)
uv sync

# Install runtime dependencies only, without the dev group
uv sync --no-dev
```

Runtime dependencies go into `[project].dependencies`; dev-only dependencies go into
`[dependency-groups].dev`.

```bash
uv add <package>                 # add a runtime dependency
uv add "<package>==<version>"    # add a pinned version
uv add --dev <package>           # add a dev-only dependency
uv remove <package>              # remove a runtime dependency
uv remove --dev <package>        # remove a dev-only dependency
```

> If you remove a dev dependency without `--dev`, uv looks in the runtime dependencies and
> will not find it.

Run anything inside that environment with `uv run python <script>.py`.

## Configuration

There are no command-line flags: every setting lives in `config.yaml`.

```bash
cp config.yaml.example config.yaml   # once; config.yaml is git-ignored
```

| Section | Controls |
| --- | --- |
| `seed` | random state for the global numpy seed and every model refit (default `5205`) |
| `data` | download date range |
| `features` | bar frequency (`daily`/`weekly`) and whether the Fear & Greed alternative data is included |
| `validation` | test window, rolling training window, validation window, refit cadence |
| `model` | `mode` is `single` or `selection`, with both a single spec and a selection grid declared |
| `portfolio` | `strategy` is `both` (default), `top_k` or `forecast_vol_scaled`, `initial_capital_usd` sizes the simulated account, plus one block per strategy holding its own parameters |

Edit `config.yaml`, then re-run `uv run python main.py backtest`. Keep `validation.test_start`
at **2016 or later**: the panel starts in 2011 (the Fear & Greed index's first date) and the
default 3-year validation window needs history before it.

## Commands

All three commands take no flags; they read `config.yaml`, and all of them use
`features.frequency`.

```bash
# 1. Download the raw data at features.frequency (needs network access)
uv run python main.py fetch

# 2. Write the feature table and correlation report for that frequency
uv run python main.py features

# 3. Backtest on those features
uv run python main.py backtest
```

`fetch` downloads adjusted OHLCV bars for the sector ETFs and SPY for the configured bar
size, over `data.start_date`..`data.end_date` (currently **2004-01-01 through 2026-09-30**).
Daily bars land in `data/raw/daily/`; the weekly setting resamples the daily download into
Friday-ending bars (Open=first, High=max, Low=min, Close=last, Volume=sum) under
`data/raw/weekly/`. A trailing week whose Friday falls after the end date is dropped, so the
weekly series stops at the last complete week inside the range.

`features` builds one row per date and sector ETF with the 19 technical features in
`src/features.py`, plus `label_forward_return_1bar` (the next bar's close-to-close return).
Indicator lookbacks count bars, so a 14-bar RSI means 14 daily or 14 weekly observations, and
volatility is annualised with 252 or 52 periods per year. The Fear & Greed Index (0-100) and
SPY's one-bar return are aligned by carrying the latest observation dated on or before each
bar forward, so no future value leaks in. Outputs go to `data/features/<frequency>/`:

| File | Contents |
| --- | --- |
| `features.csv` | one row per date and ticker with the active features plus the label (git-ignored) |
| `feature_label_correlation.csv` | Pearson correlation of every feature with the label |
| `feature_label_correlation.png` | heatmap of that matrix |

The label is not part of `FEATURES`, so it stays separate from the predictor set. Initial
rows with insufficient lookback history are kept with missing values.

## Outputs

Each backtest prints the split sizes, the validation score for the configured model, the test
comparison twice (with and without SPY) alongside equal-weight, gross-versus-net strategy
stats, forecast metrics (out-of-sample R², RMSE, direction accuracy, rank IC) and CAPM
alpha/beta, then reports where the artifacts went.

Every run writes into its own `results/<run_tag>/` folder. The tag is deliberately limited to
the settings that change the *predictions*; the portfolio settings are not part of it because
they are recorded in `run_config.yaml` inside the folder:

```
<frequency>_<fg|nofg>_roll<window_train>_val<window_val>_<refit>_<model>_<test years>

results/weekly_nofg_roll5y_val3y_mrefit_lasso_a0.0001_2024-2025/
results/daily_nofg_roll5y_val3y_mrefit_lasso_a1e-05_2024-2025/
```

So re-running the same features and model with a different strategy, gate, cadence or broker
overwrites that folder - use `run_config.yaml` to see which portfolio settings produced it.

`run_config.yaml` is not a copy of `config.yaml`: it records the resolved settings for that
run - the feature columns actually used, the expanded model specs (a grid is written out in
full), the training window in bars, and the parameters of every strategy that traded.

With `strategy: both`, the comparison tables and the cumulative-return figures carry one row
or line per strategy (`<model> top_k` and `<model> forecast_vol_scaled`), and the
per-strategy outputs are suffixed with the strategy name. The gate sweep only exists for
`top_k`, because `forecast_vol_scaled` has no swap threshold: with a strategy list that omits
`top_k`, the two gate figures and the two gate tables below are simply not written.

### Figures

| File | Shows |
| --- | --- |
| `cumulative_returns.png` | growth of $1: each strategy (net) vs SPY vs equal-weight |
| `cumulative_returns_no_spy.png` | the same chart without SPY |
| `cumulative_returns_with_trades.png` | growth of $1 per strategy with a marker on every bar that traded (the `turnover > 0` bars), plus a trade-count rug underneath |
| `cumulative_returns_by_gate.png` | every gate in `gate_sweep` on the validation window (`top_k` only) |
| `cumulative_returns_by_gate_test.png` | every gate in `gate_sweep` on the test period (`top_k` only) |
| `drawdown.png` / `drawdown_no_spy.png` | drawdown from the running peak, with / without SPY |
| `gross_vs_net.png` | gross vs net growth of $1, and cumulative costs in bps (`_<strategy>` when both run) |
| `sharpe_by_model.png` | validation net Sharpe for each configured model |
| `sharpe_by_period.png` / `sharpe_by_period_no_spy.png` | Sharpe ratio per calendar year, with / without SPY |
| `turnover_vs_gate.png` | annual turnover and net Sharpe against the gate (`top_k` only) |
| `coefficients.png` | standardised coefficients of the final model |
| `oos_r2.png` | out-of-sample R² by year and cumulatively |

### Tables

| File | Contents |
| --- | --- |
| `run_config.yaml` | resolved settings actually used by the run |
| `validation_ranking.csv` | one row per model spec, best validation net Sharpe first |
| `gate_sweep.csv` | annual turnover and net Sharpe per gate on validation (`top_k` only) |
| `gate_comparison_test.csv` | return, vol, Sharpe, drawdown and hit rate per gate on test (`top_k` only) |
| `comparison.csv` | the same stats for every strategy, SPY and equal-weight |
| `comparison_no_spy.csv` | the same table without SPY |
| `strategy.csv` | gross and net stats, turnover and cost drag (`strategy_<name>.csv` when both run) |
| `forecast.csv` | out-of-sample R², RMSE, direction accuracy, correlation and rank IC |
| `oos_r2_by_year.csv` | out-of-sample R² per calendar year |
| `capm.csv` | alpha, beta and the alpha t-statistic against SPY, one row per strategy |
| `performance_by_period.csv` | calendar-year stats per strategy |
| `performance_by_period_no_spy.csv` | the same, without SPY |
| `coefficients.csv` | final model coefficients per feature |

Re-running the same features and model overwrites that folder whatever the portfolio settings
are, and a different model or window creates a new one. The folder is tracked in git, so it
holds the artifacts of the last run at each tag.

## Tests

```bash
uv run pytest
```

Tests import from `src/`, so no install step is needed; `pyproject.toml` sets `pythonpath`
and `testpaths`.