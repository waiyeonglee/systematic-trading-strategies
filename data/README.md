# Data

Everything the pipeline reads or writes, grouped by role rather than by download step.

```
data/
  raw/                          downloaded source bars (git-ignored, rebuilt by `main.py fetch`)
    daily/XLB.csv ...           one adjusted OHLCV CSV per sector ETF, plus SPY
    weekly/XLB.csv ...          Friday-ending bars resampled from the daily download
  features/                     derived tables (rebuilt by `main.py features`)
    daily/
      features.csv              19 features + next-bar label per date and ticker (git-ignored)
      feature_label_correlation.csv
      feature_label_correlation.png
    weekly/
      ... same three files
  alternative/
    fear-greed.csv              CNN Fear & Greed Index, one dated row per observation
```

## raw/

`main.py fetch` writes one CSV per ticker with `Date`, `Open`,
`High`, `Low`, `Close` and `Volume`, for the frequency set by `features.frequency`
(`data/raw/daily/` or `data/raw/weekly/`). Prices are split- and dividend-adjusted (yfinance
`auto_adjust=True`), so `Close` is already an adjusted close and no separate adjusted-close
file is produced. Weekly bars are built by resampling the daily download
(Open=first, High=max, Low=min, Close=last, Volume=sum) and stop at the last complete week
inside `data.end_date`.

The CSVs are git-ignored because they are reproducible; `.gitkeep` files keep the folders.

## features/

`main.py features` reads `raw/`, then writes the feature table and the feature/label
correlation matrix and heatmap for the configured frequency. `features.csv` is git-ignored
(it is tens of thousands of rows); the correlation outputs are small enough to keep.

## alternative/

Fear & Greed Index dataset from the open-source
[fear-greed-data](https://github.com/whit3rabbit/fear-greed-data/tree/main) repository,
itself assembled from CNN, macro-micro, hacking-the-markets, open-stock-alert and
archive.org.

- Index value range: 0 to 100
- Ratings: Extreme Fear | Fear | Neutral | Greed | Extreme Greed
- Date range: 2011-01-03 to 2026-10-22

The pipeline only uses the numeric `Fear Greed` column, carried forward to each price bar.
