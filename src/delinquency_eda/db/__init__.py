"""Database adapters: SQLite (demo and tests) and SQL Server (real use)."""

from delinquency_eda.db.base import ColumnInfo, Database, DBAPIDatabase
from delinquency_eda.db.dialect import Dialect, SQLiteDialect, SQLServerDialect
from delinquency_eda.db.sqlite import SQLiteDatabase
from delinquency_eda.db.sqlserver import SQLServerDatabase, SQLServerSettings

__all__ = [
    "ColumnInfo",
    "DBAPIDatabase",
    "Database",
    "Dialect",
    "SQLServerDatabase",
    "SQLServerDialect",
    "SQLServerSettings",
    "SQLiteDatabase",
    "SQLiteDialect",
]
