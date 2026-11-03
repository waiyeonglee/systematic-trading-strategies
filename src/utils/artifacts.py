"""Save result tables into the run folder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml


def save_table(data: pd.DataFrame | pd.Series | dict, out_dir: Path, name: str,
               index: bool = True) -> Path:
    """Write ``data`` to ``<out_dir>/<name>.csv``.

    Args:
        data: Frame, Series or plain mapping to persist.
        out_dir: Destination folder, created if missing.
        name: File stem, e.g. ``"comparison"``.
        index: Whether to write the frame index (keep it for labelled tables such as
            ``comparison.csv``; drop it for row-per-candidate tables).

    Returns:
        The path that was written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.csv"

    if isinstance(data, dict):
        data = pd.Series(data)
    if isinstance(data, pd.Series):
        data = data.to_frame(name=data.name or "value")
    data.to_csv(path, index=index)
    return path


def save_nested(tables: dict[str, pd.DataFrame], out_dir: Path, name: str) -> Path:
    """Stack several frames into one CSV, with their dict keys as a leading column."""
    frame = pd.concat(tables, names=["name"]).reset_index(level=0)
    return save_table(frame, out_dir, name)


def save_parameters(parameters: dict, out_dir: Path, name: str = "run_config.yaml") -> Path:
    """Write the effective run parameters as YAML next to the results.

    Args:
        parameters: Plain mapping of the settings actually used by the run.
        out_dir: Destination folder, created if missing.
        name: File name.

    Returns:
        The path that was written.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text(yaml.safe_dump(parameters, sort_keys=False))
    return path
