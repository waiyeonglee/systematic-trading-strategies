"""Command-line definition for ``main.py``.

Every option lives in ``config.yaml``, so the commands take no flags.
"""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    """Build the parser with the ``fetch``, ``features`` and ``backtest`` commands."""
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="Fetch market data, build features, and backtest a systematic strategy.",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("fetch", help="download daily and weekly OHLCV data for ETFs and SPY")
    commands.add_parser("features", help="write feature tables and correlation reports")
    commands.add_parser("backtest", help="run walk-forward training, trading and evaluation")
    return parser