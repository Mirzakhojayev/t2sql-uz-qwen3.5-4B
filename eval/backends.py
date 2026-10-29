"""Run one SQL string against one database and return rows or a classified failure.

Query failures (syntax error, timeout, ...) come back as ExecResult(ok=False).
Setup problems (missing database file, cannot connect) raise, so they are never scored as wrong answers.
"""
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

import psycopg
from psycopg import errors as pg_errors

from .config import PgConfig

MAX_ROWS = 100_000


@dataclass
class ExecResult:
    ok: bool
    rows: list[tuple] | None = None
    error: str | None = None
    timed_out: bool = False


def _first_line(exc: Exception) -> str:
    return (str(exc).splitlines() or [type(exc).__name__])[0]


class SQLiteBackend:
    dialect = "sqlite"

    def __init__(self, db_dir: Path, timeout_s: float = 10.0):
        self.db_dir = Path(db_dir).resolve()  # as_uri() needs an absolute path
        self.timeout_s = timeout_s
        self._conns: dict[str, sqlite3.Connection] = {}

    def _path(self, db_id: str) -> Path:
        for candidate in (self.db_dir / f"{db_id}.sqlite", self.db_dir / db_id / f"{db_id}.sqlite"):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"SQLite database for db_id={db_id!r} not found under {self.db_dir}")

    def _conn(self, db_id: str) -> sqlite3.Connection:
        conn = self._conns.get(db_id)
        if conn is None:
            # mode=ro: the file cannot be modified, whatever the SQL says
            conn = sqlite3.connect(f"{self._path(db_id).as_uri()}?mode=ro", uri=True)
            conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
            self._conns[db_id] = conn
        return conn

    def execute(self, db_id: str, sql: str) -> ExecResult:
        conn = self._conn(db_id)
        deadline = time.monotonic() + self.timeout_s
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
        try:
            cur = conn.execute(sql)
            rows = cur.fetchmany(MAX_ROWS + 1)
        except sqlite3.Error as exc:
            timed_out = time.monotonic() > deadline
            return ExecResult(ok=False, error="timeout" if timed_out else _first_line(exc), timed_out=timed_out)
        finally:
            conn.set_progress_handler(None, 0)
        if len(rows) > MAX_ROWS:
            return ExecResult(ok=False, error=f"result has more than {MAX_ROWS} rows")
        return ExecResult(ok=True, rows=[tuple(r) for r in rows])

    def close(self) -> None:
        for conn in self._conns.values():
            conn.close()
        self._conns.clear()


class PostgresBackend:
    dialect = "postgres"

    def __init__(self, cfg: PgConfig, timeout_s: float = 10.0):
        self.cfg = cfg
        self.timeout_s = timeout_s
        self._conns: dict[str, psycopg.Connection] = {}

    def _conn(self, db_id: str) -> psycopg.Connection:
        conn = self._conns.get(db_id)
        if conn is None or conn.closed:
            conn = psycopg.connect(
                host=self.cfg.host,
                port=self.cfg.port,
                dbname=db_id,
                user=self.cfg.ro_user,
                password=self.cfg.ro_password,
                options=f"-c statement_timeout={int(self.timeout_s * 1000)} -c default_transaction_read_only=on",
            )
            conn.read_only = True
            self._conns[db_id] = conn
        return conn

    def execute(self, db_id: str, sql: str) -> ExecResult:
        conn = self._conn(db_id)
        try:
            with conn.cursor() as cur:
                cur.execute(sql)
                if cur.description is None:
                    return ExecResult(ok=False, error="statement returned no result set")
                rows = cur.fetchmany(MAX_ROWS + 1)
                # psycopg runs "SELECT 1; SELECT 2" without complaint; SQLite refuses it, so refuse here too
                if cur.nextset():
                    return ExecResult(ok=False, error="multiple statements are not allowed")
        except pg_errors.QueryCanceled:
            return ExecResult(ok=False, error="timeout", timed_out=True)
        except psycopg.Error as exc:
            return ExecResult(ok=False, error=_first_line(exc))
        finally:
            if not conn.closed:
                conn.rollback()
        if len(rows) > MAX_ROWS:
            return ExecResult(ok=False, error=f"result has more than {MAX_ROWS} rows")
        return ExecResult(ok=True, rows=[tuple(r) for r in rows])

    def close(self) -> None:
        for conn in self._conns.values():
            conn.close()
        self._conns.clear()
