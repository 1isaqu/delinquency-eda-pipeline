"""Date and number normalization for values that come out of different database drivers.

SQL Server drivers return ``datetime`` objects, SQLite returns text. Everything that enters a
dataset is normalized here so the rest of the pipeline only ever sees ISO ``YYYY-MM-DD`` strings
and plain floats.
"""

from __future__ import annotations

import argparse
import math
from datetime import date, datetime, timedelta
from decimal import Decimal
from typing import Any

from delinquency_eda.errors import ConfigError, DatasetError


def to_date(value: Any) -> date | None:
    """Return the calendar date of ``value`` or ``None`` for NULL / NaN / empty text."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text or text.lower() in {"nan", "nat", "none", "null"}:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        raise DatasetError(f"cannot read {value!r} as an ISO date") from None


def iso(value: Any) -> str | None:
    """``YYYY-MM-DD`` text for a date-like value, ``None`` if it is NULL."""
    parsed = to_date(value)
    return parsed.isoformat() if parsed else None


def to_float(value: Any) -> float | None:
    """A float for numbers, ``Decimal`` and numeric text; ``None`` for NULL / NaN."""
    if value is None:
        return None
    if isinstance(value, Decimal):
        return float(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise DatasetError(f"cannot read {value!r} as a number") from None
    return None if math.isnan(number) else number


def to_id(value: Any) -> Any:
    """Identifiers keep their type, except ``Decimal`` which becomes ``int`` when it is whole."""
    if isinstance(value, Decimal):
        return int(value) if value == value.to_integral_value() else float(value)
    if isinstance(value, str):
        return value.strip()
    return value


def parse_date_arg(text: str) -> date:
    """``YYYY-MM-DD`` text to a date. Raises ``ConfigError`` on anything else."""
    try:
        return date.fromisoformat(text.strip())
    except ValueError:
        raise ConfigError(f"expected a date as YYYY-MM-DD, got {text!r}") from None


def date_argument(text: str) -> date:
    """``type=`` callback for argparse: a bad date becomes a usage error, not a traceback."""
    try:
        return parse_date_arg(text)
    except ConfigError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None


def next_day(value: date) -> date:
    return value + timedelta(days=1)


def days_between(start: date, end: date) -> int:
    """Whole days from ``start`` to ``end`` (negative when ``end`` is earlier)."""
    return (end - start).days
