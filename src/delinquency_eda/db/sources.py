"""Product lines and how to reach their databases.

A *product line* is one database with the contract schema (for example ``auto`` and ``consumer``).
The same agreement id can exist in two product lines, so every key in the dataset is prefixed with
the product line.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass

from delinquency_eda.db.base import Database
from delinquency_eda.db.sqlite import SQLiteDatabase
from delinquency_eda.db.sqlserver import SQLServerDatabase, SQLServerSettings
from delinquency_eda.errors import ConfigError

BACKENDS = ("sqlserver", "sqlite")
_NAME = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


@dataclass(frozen=True)
class Source:
    product_line: str
    target: str


def parse_sources(specs: Iterable[str]) -> list[Source]:
    """``["auto=db_one", "consumer=db_two"]`` -> sources. Names must be unique identifiers."""
    sources: list[Source] = []
    for spec in specs:
        name, sep, target = spec.partition("=")
        name, target = name.strip(), target.strip()
        if not sep or not name or not target:
            raise ConfigError(f"a source must look like <product_line>=<target>, got {spec!r}")
        if not _NAME.match(name):
            raise ConfigError(f"product line {name!r} must be letters, digits and underscores")
        if any(s.product_line == name for s in sources):
            raise ConfigError(f"product line {name!r} is listed twice")
        sources.append(Source(name, target))
    return sources


def sources_from_env(environ: Mapping[str, str]) -> list[Source]:
    raw = environ.get("DEA_SOURCES", "").strip()
    return parse_sources(part for part in raw.split(",") if part.strip()) if raw else []


def open_source(
    source: Source,
    backend: str,
    environ: Mapping[str, str],
    *,
    connector: Callable[..., Database] | None = None,
) -> Database:
    """Open one product line. ``connector`` is an injection point for tests."""
    if connector is not None:
        return connector(source, backend)
    if backend == "sqlite":
        return SQLiteDatabase.open(source.target)
    if backend == "sqlserver":
        return SQLServerDatabase.connect(source.target, SQLServerSettings.from_env(environ))
    raise ConfigError(f"unknown backend {backend!r}; choose one of {', '.join(BACKENDS)}")
