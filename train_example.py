"""
run_example.py — end-to-end example: data -> split -> train/select -> test -> evaluate.

Purpose
-------
Shows the call order for train.py and evaluate.py. It runs on SYNTHETIC random-walk prices,
so results are meaningless; with random data, AUC should be ~0.5 (a no-leakage sanity check).
Replace load_data() with your real data and features.

Run
---
    python run_example.py
Prints validation ranking, test comparison table, gross vs net stats, forecast metrics and CAPM;
saves figures to ./figures/.
"""
import numpy as np
import pandas as pd

from train import PipelineConfig, gate_sweep, make_labels, run_pipeline
from evaluate import evaluate_all, plot_turnover_vs_gate

TICKERS = ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY"]


def load_data():
    """Build the full panel of features + labels. TODO: replace with your real data.

    Real version should: load data/stocks_daily, compute features using only data up to
    each date t, then join make_labels(prices).

    Input  : none
    Output : tuple (panel, spy_close, feature_cols)
        panel        : pd.DataFrame, MultiIndex (date, ticker), columns = feature_cols +
                       ["fwd_ret", "target"], no NaNs (see train.py module docstring)
        spy_close    : pd.Series, DatetimeIndex, SPY adjusted close
        feature_cols : list[str], e.g. ["mom_20", "mom_60", "vol_20", "ret_lag1", "ma_ratio"]

    Test
    ----
        >>> panel, spy, feats = load_data()
        >>> panel[feats + ["fwd_ret", "target"]].isna().sum().sum()     # 0
        >>> panel.index.names                                          # ["date", "ticker"]
    """
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2014-01-02", "2025-12-31")
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(3e-4, 0.012, (len(dates), len(TICKERS))), 0)),
                          index=dates, columns=TICKERS)
    spy = prices.mean(axis=1)

    feats = {}
    ret = prices.pct_change()
    feats["mom_20"] = prices.pct_change(20)
    feats["mom_60"] = prices.pct_change(60)
    feats["vol_20"] = ret.rolling(20).std()
    feats["ret_lag1"] = ret
    feats["ma_ratio"] = prices / prices.rolling(50).mean() - 1
    X = pd.concat({k: v.stack() for k, v in feats.items()}, axis=1)
    X.index.names = ["date", "ticker"]
    panel = X.join(make_labels(prices), how="inner").dropna()
    return panel, spy, list(feats)


def split(panel):
    """Split the panel by date into train (2015-2020), validation (2021-2022), test (2023-2025).

    Input  : panel — pd.DataFrame, MultiIndex (date, ticker)
    Output : tuple (train, val, test) — three panels with the same columns, non-overlapping dates

    Test
    ----
        >>> tr, va, te = split(panel)
        >>> tr.index.get_level_values("date").max() < va.index.get_level_values("date").min()  # True
    """
    d = panel.index.get_level_values("date")
    return (panel[(d >= "2015-01-01") & (d <= "2020-12-31")],
            panel[(d >= "2021-01-01") & (d <= "2022-12-31")],
            panel[(d >= "2023-01-01") & (d <= "2025-12-31")])


if __name__ == "__main__":
    panel, spy_close, tech_features = load_data()
    train, val, test = split(panel)

    cfg = PipelineConfig(freq="daily", top_k=5, gate=0.0, cost_bps=5.0, refit_every="QE")
    results = run_pipeline(train, val, test, tech_features, cfg)

    # Gate sweep on VALIDATION predictions of the chosen model
    sweep = gate_sweep(results["val_preds"][results["best_spec"].name], top_k=5, cost_bps=5.0)
    plot_turnover_vs_gate(sweep)

    # Experiment 4: same chosen spec, add Google Trends feature(s)
    # trend_cfg = PipelineConfig(**{**cfg.__dict__, "grid": [results["best_spec"]]})
    # trends_results = run_pipeline(train, val, test, tech_features + ["gt_uncertainty"], trend_cfg)
    trends_results = None

    report = evaluate_all(results, spy_close, ppy=252, trends_results=trends_results)
    pd.set_option("display.float_format", "{:.4f}".format)
    print("\n== Test comparison ==\n", report["comparison"])
    print("\n== Strategy (gross vs net) ==\n", report["strategy"])
    print("\n== Forecast ==\n", pd.Series(report["forecast"]))
    print("\n== CAPM ==\n", pd.Series(report["capm"]))
    print("\nFigures saved to ./figures/")
