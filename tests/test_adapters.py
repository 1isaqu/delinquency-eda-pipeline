"""Adapters and product-line sources. SQL Server is exercised through a fake DB-API connection."""

from __future__ import annotations

import sqlite3

import pytest

from delinquency_eda.db import (
    SQLiteDatabase,
    SQLServerDatabase,
    SQLServerDialect,
    SQLServerSettings,
)
from delinquency_eda.db.base import format_type
from delinquency_eda.db.sources import Source, open_source, parse_sources, sources_from_env
from delinquency_eda.env import load_env_file
from delinquency_eda.errors import AdapterError, AdapterUnavailable, ConfigError, ReadOnlyViolation

from world import make_world

# ---- a fake PEP 249 connection -------------------------------------------------------------


class FakeCursor:
    def __init__(self, log, result):
        self.log, self.result, self.description = log, result, None

    def execute(self, sql, params=()):
        self.log.append((sql, tuple(params)))
        columns, rows = self.result(sql)
        self._rows = rows
        self.description = [(c,) for c in columns] if columns else None

    def fetchall(self):
        return self._rows

    def close(self):
        pass


class FakeConnection:
    def __init__(self, result):
        self.log, self.result, self.closed = [], result, False

    def cursor(self):
        return FakeCursor(self.log, self.result)

    def close(self):
        self.closed = True


CATALOG = (
    ["COLUMN_NAME", "DATA_TYPE", "CHARACTER_MAXIMUM_LENGTH", "NUMERIC_PRECISION",
     "NUMERIC_SCALE", "IS_NULLABLE"],
    [
        ("row_id", "int", None, 10, 0, "NO"),
        ("login", "varchar", 40, None, None, "YES"),
        ("notes", "varchar", -1, None, None, "YES"),
        ("amount", "decimal", None, 12, 2, "NO"),
    ],
)  # fmt: skip


# ---- SQL Server ----------------------------------------------------------------------------


def test_env_settings():
    s = SQLServerSettings.from_env({})
    assert (s.host, s.port, s.user, s.trust_server_cert) == ("localhost", None, "", False)
    s = SQLServerSettings.from_env(
        {
            "DEA_DB_HOST": "db.example.invalid",
            "DEA_DB_PORT": "1433",
            "DEA_DB_USER": "reader",
            "DEA_DB_PASSWORD": "change-me",
            "DEA_DB_TRUST_SERVER_CERT": "Yes",
            "DEA_DB_SCHEMA": "dbo",
            "DEA_DB_CHAR_ENCODING": "cp1252",
        }
    )
    assert (s.host, s.port) == ("db.example.invalid", 1433)
    assert (s.schema, s.char_encoding) == ("dbo", "cp1252")
    assert s.trust_server_cert is True


def test_settings_errors():
    with pytest.raises(AdapterError, match="DEA_DB_PORT"):
        SQLServerSettings.from_env({"DEA_DB_PORT": "abc"})
    assert "change-me" not in repr(SQLServerSettings(password="change-me"))


def test_connection_string():
    with_login = SQLServerSettings(host="h", port=1433, user="u", password="p;w", driver="D")
    text = with_login.connection_string("collections")
    assert text.startswith("DRIVER={D};SERVER=h,1433;DATABASE=collections;UID=u;PWD={p;w}")
    assert text.endswith("TrustServerCertificate=no")
    assert "Trusted_Connection=yes" in SQLServerSettings().connection_string("x")


def test_connect():
    seen = {}

    def connector(connection_string, settings):
        seen["cs"] = connection_string
        return FakeConnection(lambda sql: (["a"], [(1,)]))

    db = SQLServerDatabase.connect(
        "db_one", SQLServerSettings(host="h", schema="dbo"), connector=connector
    )
    assert "DATABASE=db_one" in seen["cs"]
    assert isinstance(db.dialect, SQLServerDialect) and db.schema == "dbo"
    assert db.query("SELECT 1 AS a") == [{"a": 1}]


def test_password_not_echoed():
    def connector(connection_string, settings):
        raise RuntimeError(f"login failed for {settings.password}")

    with pytest.raises(AdapterError) as info:
        settings = SQLServerSettings(user="u", password="change-me")
        SQLServerDatabase.connect("x", settings, connector=connector)
    assert "change-me" not in str(info.value) and "***" in str(info.value)


def test_missing_pyodbc(monkeypatch):
    import builtins

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "pyodbc":
            raise ImportError("no pyodbc")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    with pytest.raises(AdapterUnavailable, match="pyodbc"):
        SQLServerDatabase.connect("x", SQLServerSettings())


def test_query_binding():
    conn = FakeConnection(lambda sql: (["n"], [(7,)]))
    db = SQLServerDatabase(conn, SQLServerDialect())
    assert db.query("SELECT ? AS n", [7]) == [{"n": 7}]
    assert conn.log == [("SELECT ? AS n", (7,))]
    with pytest.raises(ReadOnlyViolation):
        db.query("UPDATE agents SET login = 'x'")
    assert len(conn.log) == 1  # the write never reached the driver


def test_driver_errors():
    def boom(sql):
        raise RuntimeError("deadlock")

    with pytest.raises(AdapterError, match="query failed: deadlock"):
        SQLServerDatabase(FakeConnection(boom), SQLServerDialect()).query("SELECT 1")
    with pytest.raises(AdapterError, match="no result set"):
        empty = FakeConnection(lambda s: (None, []))
        SQLServerDatabase(empty, SQLServerDialect()).query("SELECT 1")


def test_table_columns():
    conn = FakeConnection(lambda sql: CATALOG)
    db = SQLServerDatabase(conn, SQLServerDialect(), schema="dbo")
    cols = db.table_columns("agents")
    assert [(c.name, c.type, c.nullable) for c in cols] == [
        ("row_id", "int", False),
        ("login", "varchar(40)", True),
        ("notes", "varchar(max)", True),
        ("amount", "decimal(12,2)", False),
    ]
    sql, params = conn.log[0]
    assert "INFORMATION_SCHEMA.COLUMNS" in sql and "TABLE_SCHEMA" in sql
    assert params == ("agents", "dbo")


def test_close_closes_the_connection():
    conn = FakeConnection(lambda sql: (["a"], []))
    with SQLServerDatabase(conn, SQLServerDialect()):
        pass
    assert conn.closed


def test_format_type():
    assert format_type("int", None, 10, 0) == "int"
    assert format_type("numeric", None, 9, 3) == "numeric(9,3)"


# ---- SQLite --------------------------------------------------------------------------------


def test_sqlite_columns(tmp_path):
    db = SQLiteDatabase.open(make_world(tmp_path / "w.db"))
    names = [c.name for c in db.table_columns("agents")]
    assert names == ["agent_id", "display_name", "login"]
    assert db.table_columns("no_such_table") == []


def test_sqlite_read_only(tmp_path):
    db = SQLiteDatabase.open(make_world(tmp_path / "w.db"))
    with pytest.raises(sqlite3.OperationalError, match="readonly"):
        db._connection.execute("DELETE FROM agents")  # bypasses the SQL guard on purpose


def test_sqlite_open_errors(tmp_path):
    with pytest.raises(AdapterError, match="not found"):
        SQLiteDatabase.open(tmp_path / "nope.db")


# ---- sources -------------------------------------------------------------------------------


def test_parse_sources():
    assert parse_sources(["auto=db_one", " consumer = data/x.db "]) == [
        Source("auto", "db_one"),
        Source("consumer", "data/x.db"),
    ]


@pytest.mark.parametrize("bad", ["auto", "=x", "auto=", "1bad=x", "has space=x"])
def test_malformed_sources(bad):
    with pytest.raises(ConfigError):
        parse_sources([bad])


def test_duplicate_lines():
    with pytest.raises(ConfigError, match="twice"):
        parse_sources(["a=1", "a=2"])


def test_sources_from_env():
    assert sources_from_env({"DEA_SOURCES": "auto=a, consumer=b"}) == [
        Source("auto", "a"),
        Source("consumer", "b"),
    ]
    assert sources_from_env({}) == []


def test_open_source_dispatch(tmp_path):
    path = make_world(tmp_path / "w.db")
    assert isinstance(open_source(Source("auto", str(path)), "sqlite", {}), SQLiteDatabase)
    with pytest.raises(ConfigError, match="unknown backend"):
        open_source(Source("auto", "x"), "oracle", {})
    sentinel = object()
    assert open_source(Source("a", "b"), "sqlite", {}, connector=lambda s, b: sentinel) is sentinel


def test_env_file_names_only(tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("# c\nA=1\nB = 'two'\nbadline\nA=ignored\n", encoding="utf-8")
    environ = {"B": "kept"}
    added = load_env_file(env_file, environ)
    assert added == ["A"]
    assert environ == {"B": "kept", "A": "1"}
    with pytest.raises(ConfigError):
        load_env_file(tmp_path / "missing", {})
