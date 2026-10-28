"""Intake: what a build sees of every row before the row becomes part of the dataset.

The extraction hands each row to ``accept`` as the driver returned it. Three entry rules are judged
there, on the raw values, because after normalization they cannot be told apart from good data:

* ``row.required``: a column the contract marks not null is present,
* ``row.readable``: dates, amounts, counts and flags read as what the contract says they are. A
  flag that arrives as ``'N'`` would otherwise become ``True``, and an unreadable date would stop
  the build on the first row without saying how many more there are,
* ``row.payment``: ``paid_amount`` is NULL or positive, and it comes with a payment date.

``accept`` counts every violation and returns only the rows that are clean, so the build always
reaches its end and the admission report can say how many rows broke which rule. A build with any
violation is rejected, so the rows left out here never reach a dataset file.

``run`` is the one place the build runs a query. It keeps a fingerprint of each statement (its SQL
text with the length of every ``IN`` list ignored), how many times it ran and how many rows it
returned. The fingerprints go into the dataset, so a later build can be compared with this one.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from decimal import Decimal
from typing import Any

from delinquency_eda.contract import EXTRACTED, ExtractedTable
from delinquency_eda.dates import to_date, to_float
from delinquency_eda.db.base import Database
from delinquency_eda.errors import DatasetError
from delinquency_eda.queries import Query

REQUIRED, READABLE, PAYMENT = "row.required", "row.readable", "row.payment"
INTAKE_RULES = (REQUIRED, READABLE, PAYMENT)

_ISO_DAY = re.compile(r"\d{4}-\d{2}-\d{2}")
_IN_LIST = re.compile(r"\?(?:, \?)+")
_TABLES = {t.name: t for t in EXTRACTED}


@dataclass
class Tally:
    """Rows examined by one rule, rows that broke it, and how they broke it."""

    checked: int = 0
    violations: int = 0
    detail: Counter[str] = field(default_factory=Counter)


def is_missing(value: Any) -> bool:
    """NULL, NaN or blank text."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    return isinstance(value, str) and not value.strip()


def reads_as_date(value: Any) -> bool:
    """A date or datetime, or text that starts with an ISO ``YYYY-MM-DD`` day."""
    if isinstance(value, (date, datetime)):
        return True
    if not isinstance(value, str) or not _ISO_DAY.match(value.strip()):
        return False
    try:
        to_date(value)
    except DatasetError:
        return False
    return True


def reads_as_number(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        number = to_float(value)
    except DatasetError:
        return False
    return number is None or math.isfinite(number)


def reads_as_count(value: Any) -> bool:
    if not reads_as_number(value):
        return False
    number = to_float(value)
    return number is None or number.is_integer()


def reads_as_flag(value: Any) -> bool:
    """0, 1, ``False`` or ``True``. Text such as ``'Y'`` is not a flag: ``bool('N')`` is true."""
    if isinstance(value, bool):
        return True
    return isinstance(value, (int, float, Decimal)) and value in (0, 1)


def fingerprint(sql: str) -> str:
    """SHA-256 of the statement with every ``IN`` list collapsed, so chunk sizes do not matter."""
    return hashlib.sha256(_IN_LIST.sub("?...", sql).encode("utf-8")).hexdigest()


class Intake:
    """Counts what the rules find in the rows of one build and logs the queries it runs."""

    def __init__(self) -> None:
        self.tallies: dict[str, Tally] = {rule: Tally() for rule in INTAKE_RULES}
        self._runs: dict[tuple[str, str, str], list[int]] = {}

    # ---- queries ----------------------------------------------------------------------------
    def run(self, db: Database, line: str, name: str, query: Query) -> list[dict[str, Any]]:
        rows = db.query(query.sql, query.params)
        entry = self._runs.setdefault((line, name, fingerprint(query.sql)), [0, 0])
        entry[0] += 1
        entry[1] += len(rows)
        return rows

    def queries(self) -> list[dict[str, Any]]:
        return [
            {"product_line": line, "name": name, "sha256": digest, "runs": runs, "rows": rows}
            for (line, name, digest), (runs, rows) in sorted(self._runs.items())
        ]

    # ---- rows -------------------------------------------------------------------------------
    def accept(self, table: str, rows: Iterable[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
        """The rows of ``table`` that break no rule. The others are counted and left out."""
        spec = _TABLES[table]
        clean: list[Mapping[str, Any]] = []
        for row in rows:
            missing = [c for c in spec.required if is_missing(row.get(c))]
            unreadable = _unreadable(spec, row)
            payment = _payment_flaws(row) if table == "slips" and not unreadable else []
            self._count(REQUIRED, table, [f"{table}.{c}" for c in missing])
            self._count(READABLE, table, [f"{table}.{c}" for c in unreadable])
            if table == "slips":
                self._count(PAYMENT, table, payment)
            if not (missing or unreadable or payment):
                clean.append(row)
        return clean

    def _count(self, rule: str, table: str, flaws: list[str]) -> None:
        tally = self.tallies[rule]
        tally.checked += 1
        if flaws:
            tally.violations += 1
            tally.detail.update(flaws)


def _unreadable(spec: ExtractedTable, row: Mapping[str, Any]) -> list[str]:
    bad: list[str] = []
    for columns, reads in (
        (spec.dates, reads_as_date),
        (spec.numbers, reads_as_number),
        (spec.counts, reads_as_count),
        (spec.flags, reads_as_flag),
    ):
        bad += [c for c in columns if not is_missing(row.get(c)) and not reads(row.get(c))]
    return bad


def _payment_flaws(row: Mapping[str, Any]) -> list[str]:
    amount = None if is_missing(row.get("paid_amount")) else to_float(row.get("paid_amount"))
    paid_on = None if is_missing(row.get("paid_date")) else row.get("paid_date")
    flaws: list[str] = []
    if amount is not None and amount <= 0:
        flaws.append("slips.paid_amount is not positive")
    if amount is not None and paid_on is None:
        flaws.append("slips.paid_amount without paid_date")
    if amount is None and paid_on is not None:
        flaws.append("slips.paid_date without paid_amount")
    return flaws
