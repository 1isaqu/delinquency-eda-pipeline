"""What the pipeline needs from a database: run one read-only query, describe a table, close.

``DBAPIDatabase`` covers any PEP 249 driver whose catalog is exposed through
``INFORMATION_SCHEMA``. Subclasses only choose the dialect and how the connection is opened.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from delinquency_eda.db.dialect import Dialect
from delinquency_eda.errors import AdapterError
from delinquency_eda.sqlutil import ensure_read_only


class Cursor(Protocol):
    """The subset of a PEP 249 cursor this package uses."""

    description: Any

    def execute(self, operation: str, parameters: Sequence[Any] = ..., /) -> Any: ...

    def fetchall(self) -> list[Any]: ...

    def close(self) -> None: ...


class Connection(Protocol):
    """The subset of a PEP 249 connection this package uses."""

    def cursor(self) -> Cursor: ...

    def close(self) -> None: ...


@dataclass(frozen=True)
class ColumnInfo:
    name: str
    type: str
    nullable: bool


class Database(ABC):
    """A read-only handle on one database (one product line)."""

    dialect: Dialect
    schema: str | None = None

    @abstractmethod
    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        """Run one read-only statement; return rows as ``{column: value}`` dictionaries."""

    @abstractmethod
    def table_columns(self, table: str) -> list[ColumnInfo]:
        """Columns of one table in ordinal order. Empty if the table does not exist."""

    @abstractmethod
    def close(self) -> None: ...

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class DBAPIDatabase(Database):
    """A ``Database`` over any PEP 249 connection with an ``INFORMATION_SCHEMA`` catalog."""

    def __init__(self, connection: Connection, dialect: Dialect, *, schema: str | None = None):
        self._connection = connection
        self.dialect = dialect
        self.schema = schema

    def query(self, sql: str, params: Sequence[Any] = ()) -> list[dict[str, Any]]:
        ensure_read_only(sql)
        cursor = self._connection.cursor()
        try:
            if params:
                cursor.execute(sql, tuple(params))
            else:
                cursor.execute(sql)
            if cursor.description is None:
                raise AdapterError("statement returned no result set")
            columns = [str(d[0]) for d in cursor.description]
            return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]
        except AdapterError:
            raise
        except Exception as exc:  # driver exception types differ per driver
            raise AdapterError(f"query failed: {exc}") from exc
        finally:
            cursor.close()

    def table_columns(self, table: str) -> list[ColumnInfo]:
        sql = (
            "SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, NUMERIC_PRECISION, "
            "NUMERIC_SCALE, IS_NULLABLE FROM INFORMATION_SCHEMA.COLUMNS "
            "WHERE UPPER(TABLE_NAME) = UPPER(?)"
        )
        params: list[Any] = [table]
        if self.schema:
            sql += " AND UPPER(TABLE_SCHEMA) = UPPER(?)"
            params.append(self.schema)
        rows = self.query(sql + " ORDER BY ORDINAL_POSITION", params)
        return [
            ColumnInfo(
                str(r["COLUMN_NAME"]),
                format_type(
                    str(r["DATA_TYPE"]),
                    r["CHARACTER_MAXIMUM_LENGTH"],
                    r["NUMERIC_PRECISION"],
                    r["NUMERIC_SCALE"],
                ),
                str(r["IS_NULLABLE"]).upper() == "YES",
            )
            for r in rows
        ]

    def close(self) -> None:
        self._connection.close()


def format_type(
    data_type: str, char_length: int | None, precision: int | None, scale: int | None
) -> str:
    """Render an INFORMATION_SCHEMA type: ``varchar(40)``, ``varchar(max)``, ``decimal(12,2)``."""
    if char_length == -1:
        return f"{data_type}(max)"
    if char_length is not None:
        return f"{data_type}({char_length})"
    if data_type.lower() in ("decimal", "numeric") and precision is not None:
        return f"{data_type}({precision},{scale})"
    return data_type
