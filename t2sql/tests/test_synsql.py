import sqlite3

import pytest

from t2sql import synsql
from t2sql.synsql import gold_status, sql_template


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(synsql, "DB_DIR", tmp_path)
    (tmp_path / "d").mkdir()
    conn = sqlite3.connect(tmp_path / "d" / "d.sqlite")
    conn.executescript("CREATE TABLE t (a INTEGER, b TEXT); INSERT INTO t VALUES (1, 'x'), (2, NULL);")
    conn.close()
    return "d"


def test_gold_status_kinds(db):
    assert gold_status(db, "SELECT a, b FROM t") == ("ok", 2, 2)
    assert gold_status(db, "SELECT a FROM t WHERE a > 99")[0] == "empty"
    assert gold_status(db, "SELECT b FROM t WHERE a = 2") == ("all_null", 1, 1)
    assert gold_status(db, "SELECT nope FROM t")[0] == "error"


def test_gold_status_timeout(db):
    q = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"
    assert gold_status(db, q, timeout_s=0.2)[0] == "timeout"


def test_template_ignores_values_and_names():
    a = sql_template("SELECT name FROM users WHERE age > 30 AND city = 'Tashkent'")
    b = sql_template("SELECT title FROM books WHERE pages > 500 AND lang = 'uz'")
    assert a == b


def test_template_separates_different_shapes():
    a = sql_template("SELECT name FROM users WHERE age > 30")
    assert a != sql_template("SELECT name FROM users WHERE age > 30 ORDER BY name")
    assert a != sql_template("SELECT COUNT(*) FROM users WHERE age > 30")


def test_template_survives_unparsable_sql():
    assert sql_template("SELEC FROM WHERE ((( 1") == sql_template("SELEC FROM WHERE ((( 2")
