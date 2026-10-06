"""SQL Server adapter (optional extra: ``pip install "delinquency-eda-pipeline[sqlserver]"``).

Connection settings come from ``DEA_DB_*`` environment variables only. The adapter is unit-tested
through an injected fake DB-API connection and by inspecting the SQL it is given; it has not been
run against a live server.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

from delinquency_eda.db.base import Connection, DBAPIDatabase
from delinquency_eda.db.dialect import SQLServerDialect
from delinquency_eda.errors import AdapterError, AdapterUnavailable

_TRUE = {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class SQLServerSettings:
    driver: str = "ODBC Driver 18 for SQL Server"
    host: str = "localhost"
    port: int | None = None
    user: str = ""
    password: str = field(default="", repr=False)
    trust_server_cert: bool = False
    char_encoding: str = "utf-8"
    schema: str | None = None
    login_timeout: int = 10
    query_timeout: int = 120

    @classmethod
    def from_env(cls, env: Mapping[str, str]) -> SQLServerSettings:
        def text(name: str, default: str) -> str:
            return env.get(f"DEA_DB_{name}", "").strip() or default

        port = text("PORT", "")
        return cls(
            driver=text("DRIVER", cls.driver),
            host=text("HOST", cls.host),
            port=_to_int("DEA_DB_PORT", port) if port else None,
            user=text("USER", ""),
            password=env.get("DEA_DB_PASSWORD", ""),
            trust_server_cert=text("TRUST_SERVER_CERT", "no").lower() in _TRUE,
            char_encoding=text("CHAR_ENCODING", cls.char_encoding),
            schema=text("SCHEMA", "") or None,
            login_timeout=_to_int("DEA_DB_LOGIN_TIMEOUT", text("LOGIN_TIMEOUT", "10")),
            query_timeout=_to_int("DEA_DB_QUERY_TIMEOUT", text("QUERY_TIMEOUT", "120")),
        )

    def connection_string(self, database: str) -> str:
        server = f"{self.host},{self.port}" if self.port else self.host
        pairs = [
            ("DRIVER", _odbc_value(self.driver, braces=True)),
            ("SERVER", _odbc_value(server)),
            ("DATABASE", _odbc_value(database)),
        ]
        if self.user:
            pairs += [("UID", _odbc_value(self.user)), ("PWD", _odbc_value(self.password))]
        else:
            pairs.append(("Trusted_Connection", "yes"))
        pairs.append(("TrustServerCertificate", "yes" if self.trust_server_cert else "no"))
        return ";".join(f"{key}={value}" for key, value in pairs)


def _to_int(name: str, value: str) -> int:
    try:
        return int(value)
    except ValueError:
        raise AdapterError(f"{name} must be an integer, got {value!r}") from None


def _odbc_value(value: str, *, braces: bool = False) -> str:
    """Quote an ODBC connection-string value when it contains characters that would end it."""
    if braces or any(ch in value for ch in ";{}") or value != value.strip():
        return "{" + value.replace("}", "}}") + "}"
    return value


Connector = Callable[[str, SQLServerSettings], Connection]


def pyodbc_connector(connection_string: str, settings: SQLServerSettings) -> Connection:
    """Open an autocommit pyodbc connection. Imported lazily so the extra stays optional."""
    try:
        import pyodbc
    except ImportError:
        raise AdapterUnavailable(
            "the SQL Server adapter needs pyodbc: pip install 'delinquency-eda-pipeline[sqlserver]'"
        ) from None
    conn = pyodbc.connect(connection_string, timeout=settings.login_timeout, autocommit=True)
    conn.timeout = settings.query_timeout
    conn.setdecoding(pyodbc.SQL_CHAR, encoding=settings.char_encoding)
    conn.setdecoding(pyodbc.SQL_WCHAR, encoding="utf-16le")
    return conn  # type: ignore[no-any-return]


class SQLServerDatabase(DBAPIDatabase):
    """SQL Server access through any DB-API connection (pyodbc by default).

    Every table reference in the generated SQL carries ``WITH (NOLOCK)``, so analytical reads
    never block the transactional system they run next to.
    """

    @classmethod
    def connect(
        cls,
        database: str,
        settings: SQLServerSettings | None = None,
        *,
        connector: Connector = pyodbc_connector,
    ) -> SQLServerDatabase:
        settings = settings or SQLServerSettings()
        try:
            connection = connector(settings.connection_string(database), settings)
        except AdapterError:
            raise
        except Exception as exc:  # driver errors are not a stable type
            raise AdapterError(
                f"cannot connect to database {database!r}: {_scrub(exc, settings)}"
            ) from None
        return cls(connection, SQLServerDialect(), schema=settings.schema)


def _scrub(exc: Exception, settings: SQLServerSettings) -> str:
    """Error text with the password removed, in case a driver echoes the connection string."""
    text = str(exc)
    return text.replace(settings.password, "***") if settings.password else text
