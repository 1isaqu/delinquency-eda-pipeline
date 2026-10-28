"""The few places where SQL Server and SQLite need different text.

Everything else in ``queries.py`` is portable SQL (CTEs, ``ROW_NUMBER``, ``IN`` lists), which is
what lets the same extraction run against a SQLite demo database in the tests and against SQL
Server in real use. Only these differ:

* table hints: SQL Server reads carry ``WITH (NOLOCK)``, SQLite has no such hint,
* top-one lookups: ``OUTER APPLY (SELECT TOP 1 ...)`` versus a scalar subquery with ``LIMIT 1``,
* date parameters: real ``date`` objects for the ODBC driver, ISO text for SQLite,
* calendar-day arithmetic, which only the admission queries use (they re-derive the outcome in SQL).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import date
from typing import Any


class Dialect(ABC):
    name = "generic"

    @abstractmethod
    def table(self, name: str, alias: str, schema: str | None = None) -> str:
        """A table reference with alias, such as ``agreements AS r WITH (NOLOCK)``."""

    @abstractmethod
    def bind_date(self, value: date) -> Any:
        """The Python value to bind for a date parameter."""

    @abstractmethod
    def first_portfolio_label(self, schema: str | None, outer_alias: str) -> tuple[str, str]:
        """``(join_sql, select_expression)`` resolving one portfolio label per agreement.

        The label hangs off the debt, not the agreement, and an agreement can link several debts.
        Exactly one label is taken (lowest ``debt_id`` first) so that no join can multiply rows.
        """

    @abstractmethod
    def day(self, expression: str) -> str:
        """The calendar day of a date or datetime expression; only the day is ever compared."""

    @abstractmethod
    def plus_days(self, day_expression: str, days: int) -> str:
        """``day_expression`` moved forward by a whole number of days."""

    @abstractmethod
    def unreadable_day(self, expression: str) -> str:
        """A condition that holds when the expression is NULL or does not read as a date."""


class SQLServerDialect(Dialect):
    name = "sqlserver"

    def table(self, name: str, alias: str, schema: str | None = None) -> str:
        qualified = f"{schema}.{name}" if schema else name
        return f"{qualified} AS {alias} WITH (NOLOCK)"

    def bind_date(self, value: date) -> Any:
        return value

    def first_portfolio_label(self, schema: str | None, outer_alias: str) -> tuple[str, str]:
        link = self.table("agreement_debts", "ad", schema)
        attrs = self.table("debt_attributes", "da", schema)
        join = (
            "OUTER APPLY (\n"
            "    SELECT TOP 1 da.portfolio_label\n"
            f"    FROM {link}\n"
            f"    INNER JOIN {attrs} ON da.debt_id = ad.debt_id\n"
            f"    WHERE ad.agreement_id = {outer_alias}.agreement_id\n"
            f"      AND ad.portfolio_id = {outer_alias}.portfolio_id\n"
            "      AND da.portfolio_label IS NOT NULL\n"
            "    ORDER BY ad.debt_id\n"
            ") AS pl"
        )
        return join, "pl.portfolio_label"

    def day(self, expression: str) -> str:
        return f"CAST({expression} AS date)"

    def plus_days(self, day_expression: str, days: int) -> str:
        return f"DATEADD(day, {int(days)}, {day_expression})"

    def unreadable_day(self, expression: str) -> str:
        return f"TRY_CAST({expression} AS date) IS NULL"


class SQLiteDialect(Dialect):
    name = "sqlite"

    def table(self, name: str, alias: str, schema: str | None = None) -> str:
        qualified = f"{schema}.{name}" if schema else name
        return f"{qualified} AS {alias}"

    def bind_date(self, value: date) -> Any:
        return value.isoformat()

    def first_portfolio_label(self, schema: str | None, outer_alias: str) -> tuple[str, str]:
        link = self.table("agreement_debts", "ad", schema)
        attrs = self.table("debt_attributes", "da", schema)
        expr = (
            "(SELECT da.portfolio_label\n"
            f"     FROM {link}\n"
            f"     INNER JOIN {attrs} ON da.debt_id = ad.debt_id\n"
            f"     WHERE ad.agreement_id = {outer_alias}.agreement_id\n"
            f"       AND ad.portfolio_id = {outer_alias}.portfolio_id\n"
            "       AND da.portfolio_label IS NOT NULL\n"
            "     ORDER BY ad.debt_id\n"
            "     LIMIT 1)"
        )
        return "", expr

    def day(self, expression: str) -> str:
        return f"date({expression})"

    def plus_days(self, day_expression: str, days: int) -> str:
        return f"date({day_expression}, '+{int(days)} days')"

    def unreadable_day(self, expression: str) -> str:
        return f"date({expression}) IS NULL"
