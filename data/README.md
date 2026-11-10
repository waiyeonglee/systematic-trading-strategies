# Data

Standalone dataset for this project. Everything the pipeline reads or writes is grouped by role.

```
data/
  raw/weekly/XLB.csv ...           one adjusted OHLCV CSV per tradable ETF + benchmark (17 tickers)
  raw/Data2025.xlsx                Welch-Goyal monthly macro predictors (tbl/tms/dfy/dfr)
  additional/
    fear-greed.csv                 CNN Fear & Greed Index, one dated row per observation
  features/weekly/                 rebuilt by `scripts/build_features.py`
    features.csv                   (date, ticker, 21 features, selected label) — git-ignored
    features_macro.csv             (date, ticker, 114 macro-interaction columns) — git-ignored    feature_label_correlation.csv
    feature_label_correlation.png
  README.md
```

## Fixed dataset window

- **End date is fixed at 2026-01-01.** The last weekly bar is the final complete
  week ending on or before 2025-12-26 (2025-12-31 for the Fear & Greed daily series).
  No data beyond 2025-12-31 is included; future re-fetches must respect this fixed end.
- **Start date 2004-01-01** (or each ETF's actual inception, whichever is later):
  the study window is 2011 onward, and everything before 2011 exists solely for
  feature warmup (e.g. 25-week volatility, long look-back trend MAs).
- **Enforced bound:** every CSV in this folder holds only rows with
  `Date < 2026-01-02` (exclusive). The regeneration script below applies the same
  bound, so re-runs cannot introduce out-of-window rows.

## Universe (16 tradable + 1 benchmark)

| Group | Tickers |
|---|---|
| US sector ETFs (9) | XLB, XLE, XLF, XLI, XLK, XLP, XLU, XLV, XLY |
| Cross-asset ETFs (7) | TLT, IEF, GLD, DBC, UUP, EFA, EEM |
| Benchmark | SPY |

## raw/weekly/

One CSV per ticker with columns `Date,Open,High,Low,Close,Volume`. Prices are
split- and dividend-adjusted (yfinance `auto_adjust=True`), so `Close` is the
adjusted close. Bars are Friday week-ending bars resampled from the daily download
(Open=first, High=max, Low=min, Close=last, Volume=sum), stopping at the last
complete week inside the fixed end date.

Coverage (fetched 2026-11-01 from Yahoo Finance):

- XLB–XLY, TLT, IEF, EFA, EEM, SPY: 2004-01-02 → 2025-12-26 (1,148 weekly bars)
- GLD: 2004-11-19 → 2025-12-26 (1,102 bars, inception)
- DBC: 2006-02-10 → 2025-12-26 (1,038 bars, inception)
- UUP: 2007-03-02 → 2025-12-26 (983 bars, inception)

The CSVs are git-ignored (reproducible source data); `.gitkeep` keeps the folder.

### Regeneration (`scripts/fetch_data.py`)

A standalone re-fetch script exists so the dataset can be rebuilt from Yahoo
Finance when needed (the pipeline itself imports these CSVs — no live pull on
every run). Fixed settings inside the script:

```
START_DATE        = "2004-01-01"   # inclusive
END_DATE_INCLUSIVE = "2026-01-01"  # fixed end of the dataset
END_DATE_EXCLUSIVE = "2026-01-02"  # yfinance exclusive end  <-- the setting
```

- 17 tickers (16 tradable + SPY), `auto_adjust=True`, Friday weekly bars
  (Open=first, High=max, Low=min, Close=last, Volume=sum), trailing partial week
  dropped, rows beyond the fixed end trimmed.
- `python scripts/fetch_data.py` — full re-fetch; `--dry-run` prints the settings
  without downloading.

## raw/Data2025.xlsx (macro predictors)

Welch-Goyal "PredictorData" workbook (the standard monthly predictor dataset),
1871-01 → 2025-12. Downloaded from **Amit Goyal's predictor-data page,
<https://sites.google.com/view/agoyal145>** (the `PredictorData2025` workbook);
the accompanying paper is Goyal & Welch (2008), *A Comprehensive Look at the
Empirical Performance of Equity Premium Prediction*, Review of Financial Studies
21(4), 1455–1508. Only the `Monthly` sheet is read, and only four columns:

| Column | Definition | Used as |
|---|---|---|
| `tbl` | 3-month T-bill rate (decimal) | macro state `tbl` |
| `tms` | `lty - tbl` (long govt yield − T-bill) | macro state `tms` |
| `dfy` | `BAA - AAA` | macro state `dfy` |
| `dfr` | `corpr - ltr` (corporate − long govt return) | macro state `dfr` |

`dfr` is why the workbook is needed — it has no FRED equivalent. The `ReadMe` sheet
in the same file documents every column. Fifth macro variable, `fear_greed`, comes
from `additional/fear-greed.csv` (month-end value). Bars dated in month `M` use the
month `M - 1` values (1-month publication lag, no lookahead).

## additional/fear-greed.csv

CNN Fear & Greed Index from the open-source
[fear-greed-data](https://github.com/whit3rabbit/fear-greed-data/tree/main) repository.

- Columns: `Date,Fear Greed,Rating`; index value 0–100, ratings
  Extreme Fear | Fear | Neutral | Greed | Extreme Greed.
- Coverage: 2011-01-03 → 2025-12-31 (3,750 rows, no nulls, no duplicate dates).
- The pipeline uses only the numeric `Fear Greed` column, carried forward to each price bar.

## features/weekly/

Rebuilt by `scripts/build_features.py` from `raw/weekly/` + `additional/fear-greed.csv`
+ `raw/Data2025.xlsx` (this one run writes both `features.csv` and `features_macro.csv`):

- `features.csv` — one row per `(date, ticker)` for the 16 tradables: 21 features
  (7 momentum returns, 2 SMA ratios, 2 volatilities, drawdown, RSI, MACD pair,
  Bollinger position, volume change, fear-greed, SPY return, 2 trend inputs) +
  the selected `label_forward_return_{H}bar` computed by `src/labels.py`.
  For H=12, the last 13 weekly bars per ticker carry NaN labels (208 rows), so the
  last fully labelled date is 2025-09-26. Rebuild generated files after label changes.
- `feature_label_correlation.csv` / `.png` — Pearson correlation of each feature
  vs the label across all labelled dates.
- `features_macro.csv` — the macro-interaction panel used by the macro feature block:
  one row per `(date, ticker)` with the **19 firm features × (5 macro + const) = 114
  interaction columns** (named `<firm>__x__<macro>`), the five raw macro levels
  (`macro_*`), `fwd_ret` and the 1/4/12-bar labels. The `micro` block reads the 19 firm
  features from the same frame. It is **~30 MB and git-ignored**,
  so it is not part of a clone; the study rebuilds it in memory. Written by the same
  `python scripts/build_features.py` run; pass `--no-macro` to skip it or
  `--macro-out <path>` for a custom destination, and see
  `../agents/macro_study.md` for the construction and timing rules.
- See `../agents/feature_engineering.md` for the full definition.

## Notes

- `data/features/` from the reference project is **not** carried over;
  derived feature tables are rebuilt by this project's own pipeline.
- Old backtest/model results from the reference project are **not** carried over.
