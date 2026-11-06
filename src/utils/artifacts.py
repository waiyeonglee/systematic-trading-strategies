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


def save_arm_summary(arms: list[dict], out_dir: Path, *, header: list[str],
                     name: str = "arm_comparison.txt", best_key: str,
                     title: str, note: str | None = None,
                     tables: tuple[tuple[str, list[str], list[dict]], ...] = ()
                     ) -> Path:
    """Write a small fixed-width text report comparing per-arm draws.

    Args:
        arms: One row per refit arm; every row must carry the ``header`` keys plus
            ``best_key``. Floats are printed to four decimals.
        out_dir: Destination folder, created if missing.
        header: Column names of the main table, in display order.
        name: File name.
        best_key: Row key the main table is ordered and decided on (descending).
        title: First line of the file.
        note: Optional paragraph appended under the verdict.
        tables: Extra ``(title, header, rows)`` blocks printed after the verdict, in
            the order given (each row carries its block's header keys).

    Returns:
        The path that was written.
    """
    rows = sorted(arms, key=lambda row: row[best_key], reverse=True)
    lines = _table_lines(title, header, rows)

    best = rows[0]
    lines += ["", f"best arm: {best[header[0]]} "
                  f"(highest {best_key} = {_cell(best[best_key])})"]
    for table_title, table_header, table_rows in tables:
        lines += [""] + _table_lines(table_title, table_header, table_rows)
    if note:
        lines += ["", note]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    path.write_text("\n".join(lines) + "\n")
    return path


def _table_lines(title: str, header: list[str],
                 rows: list[dict]) -> list[str]:
    """One fixed-width table: title line, header, rule, then the rows."""
    cells = [[_cell(row[name]) for name in header] for row in rows]
    widths = [max(len(header[i]), *(len(cell[i]) for cell in cells))
              for i in range(len(header))]
    lines = [title, "  ".join(value.ljust(width)
                              for value, width in zip(header, widths)),
             "  ".join("-" * width for width in widths)]
    lines += ["  ".join(value.ljust(width) for value, width in zip(cell, widths))
              for cell in cells]
    return lines


def _cell(value) -> str:
    """Text form of one summary cell (floats to four decimals)."""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)
