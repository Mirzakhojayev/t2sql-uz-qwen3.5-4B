"""Loader and converter, on a tiny SQLite database with SQLite's typical looseness."""
import sqlite3

import psycopg
import pytest

from eval.backends import PostgresBackend
from eval.config import pg_config
from t2sql import convert, synsql
from t2sql.convert import check_question, load_database, pg_type, schema_name, to_postgres

DB = "conv_test"


@pytest.mark.parametrize("declared, values, expected", [
    ("INTEGER", [1, 2, None], "BIGINT"),
    ("INTEGER", [1, "n/a"], "TEXT"),               # SQLite allows this; PostgreSQL would refuse the insert
    ("INTEGER", [1, 2.5], "DOUBLE PRECISION"),
    ("BOOLEAN", [0, 1], "BIGINT"),                 # gold SQL writes flag = 1
    ("REAL", [1.5, 2], "DOUBLE PRECISION"),
    ("REAL", ["high"], "TEXT"),
    ("NUMERIC(10,2)", [1.5], "NUMERIC"),
    ("TEXT", ["a", None], "TEXT"),
    ("DATE", ["2024-01-31"], "DATE"),
    ("DATE", ["31/01/2024"], "TEXT"),              # not ISO: keep the text
    ("DATETIME", ["2024-01-31 10:00:00"], "TIMESTAMP"),
    ("", [1, 2], "BIGINT"),
    ("", ["x"], "TEXT"),
    ("VARCHAR(20)", [], "TEXT"),
])
def test_pg_type(declared, values, expected):
    assert pg_type(declared, values) == expected


def test_schema_name_is_short_and_stable():
    long = "x" * 200
    assert schema_name(long) == schema_name(long) and len(schema_name(long)) <= 63


def test_to_postgres():
    assert to_postgres("SELECT `a b` FROM t;") == 'SELECT "a b" FROM t'
    assert to_postgres("SELEC nonsense (((") is None


@pytest.fixture(scope="module")
def pg_db(tmp_path_factory):
    d = tmp_path_factory.mktemp("synsql_dbs")
    (d / DB).mkdir()
    conn = sqlite3.connect(d / DB / f"{DB}.sqlite")
    conn.executescript("""
        CREATE TABLE dept (id INTEGER PRIMARY KEY, name TEXT);
        INSERT INTO dept VALUES (1, 'Sales'), (2, 'HR');
        CREATE TABLE emp (id INTEGER PRIMARY KEY, name TEXT, dept_id INTEGER REFERENCES dept, pay REAL,
                          code INTEGER, boss INTEGER, flag BOOLEAN, FOREIGN KEY (boss) REFERENCES missing(id));
        INSERT INTO emp VALUES (1, 'Aziz', 1, 100.0, 7, NULL, 1), (2, 'Bobur', 2, 50.0, 'n/a', 1, 0), (3, 'madina', 2, 50.5, 9, 1, 1);
        CREATE TABLE odd (a INTEGER PRIMARY KEY, txt TEXT);
        INSERT INTO odd VALUES (NULL, 'nul' || char(0) || 'byte');
    """)
    conn.close()
    cfg = pg_config()
    admin = dict(host=cfg.host, port=cfg.port, user=cfg.admin_user, password=cfg.admin_password, connect_timeout=10)
    try:
        with psycopg.connect(dbname="postgres", autocommit=True, **admin) as c:
            if not c.execute("SELECT 1 FROM pg_database WHERE datname = %s", (convert.PG_DATABASE,)).fetchone():
                c.execute(f"CREATE DATABASE {convert.PG_DATABASE}")
    except psycopg.OperationalError as exc:
        pytest.skip(f"PostgreSQL not reachable: {exc}")
    schema = schema_name(DB)
    with psycopg.connect(dbname=convert.PG_DATABASE, autocommit=True, **admin) as c:
        c.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        report = load_database(c, d / DB / f"{DB}.sqlite", schema)
    backend = PostgresBackend(cfg, timeout_s=5.0)
    original = synsql.DB_DIR, convert.DB_DIR
    synsql.DB_DIR, convert.DB_DIR = d, d
    yield report, backend
    backend.close()
    synsql.DB_DIR, convert.DB_DIR = original
    with psycopg.connect(dbname=convert.PG_DATABASE, autocommit=True, **admin) as c:
        c.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')


def rec(sql):
    return {"id": "q", "db_id": DB, "sql": sql}


def test_loader_report(pg_db):
    report, _ = pg_db
    assert report["tables"] == 3 and report["rows"] == 6
    assert report["foreign_keys"] == 1 and report["foreign_keys_skipped"] == 1  # table `missing` does not exist


def test_readonly_user_reads_the_schema(pg_db):
    _, backend = pg_db
    res = backend.execute(f"{convert.PG_DATABASE}/{schema_name(DB)}", "SELECT COUNT(*) FROM emp")
    assert res.ok and res.rows == [(3,)]
    assert not backend.execute(f"{convert.PG_DATABASE}/{schema_name(DB)}", "DELETE FROM emp").ok


def test_same_result_is_kept(pg_db):
    _, backend = pg_db
    r = check_question(rec("SELECT d.name, COUNT(*) FROM emp e JOIN dept d ON e.dept_id = d.id GROUP BY d.name;"), backend)
    assert r["reason"] is None and "sql_postgres" in r


def test_nul_byte_removed_and_loose_types_survive(pg_db):
    _, backend = pg_db
    db = f"{convert.PG_DATABASE}/{schema_name(DB)}"
    assert backend.execute(db, "SELECT txt FROM odd").rows == [("nulbyte",)]
    assert backend.execute(db, "SELECT code FROM emp ORDER BY id").rows == [("7",), ("n/a",), ("9",)]  # column became TEXT


def test_like_case_difference_is_rejected(pg_db):
    # SQLite LIKE ignores case, PostgreSQL LIKE does not: the two answers differ, so this must not be kept
    _, backend = pg_db
    assert check_question(rec("SELECT name FROM emp WHERE name LIKE 'M%'"), backend)["reason"] == "result_differs"


def test_null_ordering_difference_is_detected(pg_db):
    # ascending order: SQLite puts NULL first, PostgreSQL last
    _, backend = pg_db
    r = check_question(rec("SELECT boss FROM emp ORDER BY boss"), backend)
    assert r["reason"] in (None, "order_differs")
    if r["reason"] is None:  # sqlglot may add NULLS FIRST to keep SQLite's meaning
        assert "NULLS FIRST" in r["sql_postgres"].upper()


def test_pg_error_is_reported(pg_db):
    _, backend = pg_db
    r = check_question(rec("SELECT code + 1 FROM emp"), backend)  # code became TEXT in PostgreSQL
    assert r["reason"] == "pg_error" and r["error"]
