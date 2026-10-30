"""Figure output helpers shared by every plotting function."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt


def save_figure(fig: plt.Figure, out_dir: Path, name: str) -> plt.Figure:
    """Tighten the layout, write ``<out_dir>/<name>.png`` and return the figure.

    Args:
        fig: Matplotlib figure to save.
        out_dir: Destination folder, created if missing.
        name: File stem, e.g. ``"drawdown"``.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_dir / f"{name}.png", dpi=150)
    return fig


def close_all() -> None:
    """Close every open figure so long runs do not leak memory."""
    plt.close("all")
