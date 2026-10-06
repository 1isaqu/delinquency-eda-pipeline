"""A tiny Markdown builder. Every report-writing stage uses it, so tables look alike."""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import pandas as pd


def fmt_int(value: Any) -> str:
    return f"{int(value):,}"


def fmt_money(value: float | None) -> str:
    """Amounts in the currency of the data, two decimals, thousands separators."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{value:,.2f}"


def fmt_pct(part: float, whole: float, digits: int = 1) -> str:
    """``part`` as a percentage of ``whole``; ``-`` when ``whole`` is zero."""
    return f"{100.0 * part / whole:.{digits}f}%" if whole else "-"


def fmt_rate(value: float | None, digits: int = 1) -> str:
    """A 0-1 rate as a percentage."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "-"
    return f"{100.0 * value:.{digits}f}%"


def _cell(value: Any, float_format: str) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return "" if math.isnan(value) else float_format.format(value)
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item") and not isinstance(value, str):
        return _cell(value.item(), float_format)
    return str(value).replace("|", "\\|").replace("\n", " ")


def md_table(
    data: pd.DataFrame | Sequence[Sequence[Any]],
    headers: Sequence[str] | None = None,
    *,
    float_format: str = "{:,.2f}",
    align_right: Iterable[str] = (),
) -> str:
    """A GitHub-flavoured Markdown table from a DataFrame or from rows plus ``headers``."""
    if isinstance(data, pd.DataFrame):
        columns = [str(c) for c in data.columns]
        rows = data.to_numpy(dtype=object).tolist()
    else:
        if headers is None:
            raise ValueError("headers are required when rows are given")
        columns = [str(h) for h in headers]
        rows = [list(r) for r in data]
    right = set(align_right)
    rule = "|" + "|".join("---:" if c in right else "---" for c in columns) + "|"
    lines = ["| " + " | ".join(columns) + " |", rule]
    lines += ["| " + " | ".join(_cell(v, float_format) for v in row) + " |" for row in rows]
    return "\n".join(lines)


class Doc:
    """Collects Markdown lines. ``render()`` returns the text, ``save()`` writes it."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def h1(self, text: str) -> Doc:
        return self._block(f"# {text}")

    def h2(self, text: str) -> Doc:
        return self._block(f"## {text}")

    def h3(self, text: str) -> Doc:
        return self._block(f"### {text}")

    def p(self, text: str) -> Doc:
        return self._block(text)

    def bullets(self, items: Iterable[str]) -> Doc:
        return self._block("\n".join(f"- {item}" for item in items))

    def numbered(self, items: Iterable[str]) -> Doc:
        return self._block("\n".join(f"{i}. {item}" for i, item in enumerate(items, 1)))

    def table(
        self,
        data: pd.DataFrame | Sequence[Sequence[Any]],
        headers: Sequence[str] | None = None,
        **kwargs: Any,
    ) -> Doc:
        return self._block(md_table(data, headers, **kwargs))

    def quote(self, text: str) -> Doc:
        return self._block("\n".join(f"> {line}" for line in text.splitlines()))

    def rule(self) -> Doc:
        return self._block("---")

    def _block(self, text: str) -> Doc:
        self.lines.append(text)
        self.lines.append("")
        return self

    def render(self) -> str:
        return "\n".join(self.lines).rstrip() + "\n"

    def save(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(self.render(), encoding="utf-8", newline="\n")
        return target
