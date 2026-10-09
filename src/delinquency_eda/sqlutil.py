"""Small SQL text helpers: a read-only guard and parameter-list utilities.

The guard catches mistakes in the pipeline's own SQL. Access control still has to come from a
database login that only holds read permission; use one.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from typing import TypeVar

from delinquency_eda.errors import ReadOnlyViolation

T = TypeVar("T")

# One pass over the text so that "--" inside a literal and a quote inside a comment are both safe.
_TOKEN = re.compile(
    r"(?P<lit>'(?:[^']|'')*')"
    r"|(?P<lc>--[^\n]*)"
    r"|(?P<bc>/\*.*?\*/)"
    r"|(?P<qid>\"[^\"]*\"|\[[^\]]*\])",
    re.DOTALL,
)

_FORBIDDEN = re.compile(
    r"(?<![\w$#@])(?:INSERT|UPDATE|DELETE|MERGE|DROP|CREATE|ALTER|TRUNCATE|GRANT|REVOKE|"
    r"EXEC|EXECUTE|INTO)(?!\w)"
    r"|(?<![\w$#@])(?:SP|XP)_\w*",
    re.IGNORECASE,
)
_FIRST_WORD = re.compile(r"^[\s(]*([A-Za-z_]+)")
ALLOWED_STARTS = ("SELECT", "WITH")


def strip_noise(sql: str) -> str:
    """Remove comments, blank out string literals and neutralize quoted identifiers."""

    def repl(match: re.Match[str]) -> str:
        kind = match.lastgroup
        if kind == "lit":
            return "''"
        if kind == "qid":
            return "_q_"
        return " "

    return _TOKEN.sub(repl, sql)


def ensure_read_only(sql: str) -> None:
    """Refuse anything that is not one plain ``SELECT`` / ``WITH ... SELECT`` statement."""
    statements = [s.strip() for s in strip_noise(sql).split(";") if s.strip()]
    if not statements:
        raise ReadOnlyViolation("empty SQL")
    if len(statements) > 1:
        raise ReadOnlyViolation("only a single statement is allowed")
    statement = statements[0]
    first = _FIRST_WORD.match(statement)
    if first is None or first.group(1).upper() not in ALLOWED_STARTS:
        shown = first.group(1) if first else statement[:20]
        raise ReadOnlyViolation(
            f"read-only SQL only: a statement must start with {list(ALLOWED_STARTS)}, got {shown!r}"
        )
    hit = _FORBIDDEN.search(statement)
    if hit:
        raise ReadOnlyViolation(f"read-only SQL only, found {hit.group(0).upper()!r}")


def placeholders(count: int) -> str:
    """``?, ?, ?`` for ``count`` bound parameters."""
    if count < 1:
        raise ValueError("an IN list needs at least one value")
    return ", ".join("?" * count)


def chunked(items: Sequence[T] | Iterable[T], size: int) -> Iterator[list[T]]:
    """Consecutive lists of at most ``size`` items, in order."""
    if size < 1:
        raise ValueError("chunk size must be at least 1")
    batch: list[T] = []
    for item in items:
        batch.append(item)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch
