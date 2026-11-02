"""Run one SQL string against one database and return rows or a classified failure.

Query failures (syntax error, timeout, ...) come back as ExecResult(ok=False).
Setup problems (missing database file, cannot connect) raise, so they are never scored as wrong answers.
"""
import sqlite3
import threading
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
    elapsed: float = 0.0  # seconds, filled in by the harness


class _ThreadLocalConns:
    """One connection per (thread, database): driver connections are not shared between threads."""

    def __init__(self):
        self._local = threading.local()
        self._all: list = []
        self._lock = threading.Lock()

    def get(self, db_id, connect):
        conns = getattr(self._local, "conns", None)
        if conns is None:
            conns = self._local.conns = {}
        conn = conns.get(db_id)
        if conn is None or getattr(conn, "closed", False):
            conn = conns[db_id] = connect()
            with self._lock:
                self._all.append(conn)
        return conn

    def close_all(self):
        with self._lock:
            for conn in self._all:
                try:
                    conn.close()
                except Exception:
                    pass
            self._all.clear()
        self._local = threading.local()


def _first_line(exc: Exception) -> str:
    return (str(exc).splitlines() or [type(exc).__name__])[0]


class SQLiteBackend:
    dialect = "sqlite"

    def __init__(self, db_dir: Path, timeout_s: float = 10.0):
        self.db_dir = Path(db_dir).resolve()  # as_uri() needs an absolute path
        self.timeout_s = timeout_s
        self._conns = _ThreadLocalConns()

    def _path(self, db_id: str) -> Path:
        for candidate in (self.db_dir / f"{db_id}.sqlite", self.db_dir / db_id / f"{db_id}.sqlite"):
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"SQLite database for db_id={db_id!r} not found under {self.db_dir}")

    def _connect(self, db_id: str) -> sqlite3.Connection:
        # mode=ro: the file cannot be modified, whatever the SQL says.
        # check_same_thread=False only so close() may run on another thread; each connection is used by one thread.
        conn = sqlite3.connect(f"{self._path(db_id).as_uri()}?mode=ro", uri=True, check_same_thread=False)
        conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
        return conn

    def _conn(self, db_id: str) -> sqlite3.Connection:
        return self._conns.get(db_id, lambda: self._connect(db_id))

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
        self._conns.close_all()


class PostgresBackend:
    dialect = "postgres"

    def __init__(self, cfg: PgConfig, timeout_s: float = 10.0):
        self.cfg = cfg
        self.timeout_s = timeout_s
        self._conns = _ThreadLocalConns()

    def _connect(self, db_id: str) -> psycopg.Connection:
        conn = psycopg.connect(
            host=self.cfg.host,
            port=self.cfg.port,
            dbname=db_id,
            user=self.cfg.ro_user,
            password=self.cfg.ro_password,
            connect_timeout=10,  # fail fast when the server is down instead of hanging for minutes
            options=f"-c statement_timeout={int(self.timeout_s * 1000)} -c default_transaction_read_only=on",
        )
        conn.read_only = True
        return conn

    def _conn(self, db_id: str) -> psycopg.Connection:
        return self._conns.get(db_id, lambda: self._connect(db_id))

    def execute(self, db_id: str, sql: str) -> ExecResult:
        # "database/schema" addresses one schema of a shared database (how thousands of small databases are stored)
        dbname, _, schema = db_id.partition("/")
        conn = self._conn(dbname)
        try:
            with conn.cursor() as cur:
                if schema:
                    cur.execute(f'SET LOCAL search_path TO "{schema}"')
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
        self._conns.close_all()
