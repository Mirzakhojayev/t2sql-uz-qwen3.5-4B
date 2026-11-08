import random
from collections import Counter

import sqlite3

from t2sql.bird_train import lowercase_names, lowercase_quoted, round_robin, same_result


def test_simple_names_go_lower_case_and_quoted_names_keep_theirs(tmp_path):
    path = tmp_path / "db.sqlite"
    conn = sqlite3.connect(path)
    conn.execute('CREATE TABLE RA (groupName TEXT, "Customer ID" INTEGER, year INTEGER)')
    conn.close()
    assert lowercase_names(path) == {"ra": "ra", "groupname": "groupname", "year": "year"}


def test_quoted_simple_names_match_the_lower_case_copy():
    sql = 'SELECT "Year", "Customer ID" FROM t WHERE name = \'Year\''
    assert lowercase_quoted(sql) == 'SELECT "year", "Customer ID" FROM t WHERE name = \'Year\''


def test_round_robin_spreads_across_databases():
    rows = [{"id": f"a{i}", "db_id": "a"} for i in range(50)] + [{"id": f"b{i}", "db_id": "b"} for i in range(5)]
    picked = round_robin(rows, 10, random.Random(0))
    assert Counter(r["db_id"] for r in picked) == {"a": 5, "b": 5}
    assert len({r["id"] for r in picked}) == 10


def test_round_robin_fills_from_the_rest_when_one_runs_out():
    rows = [{"id": f"a{i}", "db_id": "a"} for i in range(50)] + [{"id": "b0", "db_id": "b"}]
    picked = round_robin(rows, 10, random.Random(0))
    assert Counter(r["db_id"] for r in picked) == {"a": 9, "b": 1}
    assert len(round_robin(rows, 500, random.Random(0))) == 51


def test_same_result_follows_the_gold_order():
    assert same_result([(1,), (2,)], [(2,), (1,)], "SELECT a FROM t", "sqlite")
    assert not same_result([(1,), (2,)], [(2,), (1,)], "SELECT a FROM t ORDER BY a", "sqlite")
    assert same_result([(3,)], [(3.0,)], "SELECT COUNT(*) FROM t", "sqlite")
