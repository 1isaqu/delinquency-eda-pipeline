"""SQLite adapter, used by the demo database and the tests. Opens the file read-only."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from delinquency_eda.db.base import ColumnInfo, DBAPIDatabase
from delinquency_eda.db.dialect import SQLiteDialect
from delinquency_eda.errors import AdapterError


class SQLiteDatabase(DBAPIDatabase):
    @classmethod
    def open(cls, path: str | Path) -> SQLiteDatabase:
        target = Path(path)
        if not target.is_file():
            raise AdapterError(f"SQLite database not found: {target}")
        try:
            connection = sqlite3.connect(f"{target.resolve().as_uri()}?mode=ro", uri=True)
        except sqlite3.Error as exc:
            raise AdapterError(f"cannot open {target}: {exc}") from None
        return cls(connection, SQLiteDialect())  # type: ignore[arg-type]

    def table_columns(self, table: str) -> list[ColumnInfo]:
        rows = self.query(
            'SELECT name, type, "notnull" AS not_null FROM pragma_table_info(?) ORDER BY cid',
            [table],
        )
        return [
            ColumnInfo(str(r["name"]), str(r["type"] or "").lower(), not r["not_null"])
            for r in rows
        ]
