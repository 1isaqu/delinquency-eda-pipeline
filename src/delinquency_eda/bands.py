"""Right-closed bands over a numeric variable, defined by their upper edges.

``Bands((0, 2, 5))`` has four bands: ``<=0``, ``1-2``, ``3-5``, ``>5``. A value goes into the
first band whose upper edge is greater than or equal to it. Edges are configuration: they are
calibrated on a business's own data, never on a general rule.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Iterable
from dataclasses import dataclass
from itertools import pairwise
from typing import Literal

Kind = Literal["int", "money", "ratio"]


@dataclass(frozen=True)
class Bands:
    edges: tuple[float, ...]
    kind: Kind = "int"

    def __post_init__(self) -> None:
        if not self.edges:
            raise ValueError("a band set needs at least one edge")
        if any(b <= a for a, b in pairwise(self.edges)):
            raise ValueError(f"band edges must be strictly increasing, got {self.edges}")

    @property
    def labels(self) -> list[str]:
        """Band labels in ascending order."""
        e = self.edges
        out = [f"<={self._fmt(e[0])}"]
        for lo, hi in pairwise(e):
            out.append(self._between(lo, hi))
        out.append(f">{self._fmt(e[-1])}")
        return out

    def label(self, value: float | None) -> str | None:
        """The label of the band that holds ``value``; ``None`` for a missing value."""
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return None
        return self.labels[bisect_left(self.edges, value)]

    def assign(self, values: Iterable[float | None]) -> list[str | None]:
        labels = self.labels
        out: list[str | None] = []
        for v in values:
            if v is None or (isinstance(v, float) and math.isnan(v)):
                out.append(None)
            else:
                out.append(labels[bisect_left(self.edges, v)])
        return out

    def _between(self, lo: float, hi: float) -> str:
        if self.kind == "int":
            first = int(lo) + 1
            return str(first) if first == int(hi) else f"{first}-{int(hi)}"
        if self.kind == "ratio":
            return f"{lo * 100:g}-{hi * 100:g}%"
        return f"{_money(lo)}-{_money(hi)}"

    def _fmt(self, edge: float) -> str:
        if self.kind == "int":
            return str(int(edge))
        if self.kind == "ratio":
            return f"{edge * 100:g}%"
        return _money(edge)


def _money(value: float) -> str:
    return f"{int(value):,}" if float(value).is_integer() else f"{value:,g}"
