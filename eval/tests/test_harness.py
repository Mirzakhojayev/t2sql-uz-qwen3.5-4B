"""Harness tests against the tiny database. Every test runs on SQLite and on PostgreSQL.

Expected answers were worked out by hand from eval/testdb.py (7 employees, salaries sum to 33000).
"""
import json
import time

import pytest

from eval import testdb
from eval.harness import main, majority_pick, summarize

GOOD = "SELECT name FROM employees WHERE salary > 5000"   # Aziz, Dilnoza, Nodira
WRONG = "SELECT name FROM employees WHERE salary > 3500"  # adds Bobur, Madina
SLEEPS = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c"


def rec(dialect, gold, preds, id="r1", language="en", slice="main"):
    r = {"id": id, "language": language, "dialect": dialect, "slice": slice, "db_id": testdb.DB_ID, "gold_sql": gold}
    r["pred_sqls" if isinstance(preds, list) else "pred_sql"] = preds
    return r


def test_identical_query_is_correct(evaluator, dialect):
    s = evaluator.score(rec(dialect, GOOD, GOOD))
    assert s.gold_ok and s.acc and s.maj and s.oracle and s.pred_status == ["ok"]


def test_different_sql_same_result_is_correct(evaluator, dialect):
    s = evaluator.score(rec(dialect, GOOD, "SELECT name FROM employees WHERE salary >= 5500"))
    assert s.acc


def test_wrong_result_is_wrong(evaluator, dialect):
    s = evaluator.score(rec(dialect, GOOD, WRONG))
    assert not s.acc and s.pred_status == ["ok"]


def test_row_order_ignored_without_order_by(evaluator, dialect):
    gold = "SELECT name FROM employees WHERE dept_id = 2"
    assert evaluator.score(rec(dialect, gold, gold + " ORDER BY name DESC")).acc


def test_row_order_matters_with_order_by(evaluator, dialect):
    gold = "SELECT name FROM employees ORDER BY salary DESC, id"
    assert evaluator.score(rec(dialect, gold, gold)).acc
    assert not evaluator.score(rec(dialect, gold, "SELECT name FROM employees ORDER BY salary, id")).acc


def test_duplicate_rows_count(evaluator, dialect):
    gold = "SELECT salary FROM employees"  # 4000 appears twice
    assert not evaluator.score(rec(dialect, gold, "SELECT DISTINCT salary FROM employees")).acc


def test_column_order_matters(evaluator, dialect):
    gold = "SELECT name, salary FROM employees"
    assert not evaluator.score(rec(dialect, gold, "SELECT salary, name FROM employees")).acc


def test_numeric_types_across_engines(evaluator, dialect):
    # AVG gives REAL in SQLite and NUMERIC in PostgreSQL; both must equal 33000 / 7
    gold = "SELECT AVG(salary) FROM employees"
    assert evaluator.score(rec(dialect, gold, "SELECT SUM(salary) / COUNT(*) FROM employees")).acc


def test_dates_compare_as_text(evaluator, dialect):
    gold = "SELECT hired FROM employees WHERE id = 2"
    assert evaluator.score(rec(dialect, gold, "SELECT CAST(hired AS TEXT) FROM employees WHERE id = 2")).acc


def test_null_semantics(evaluator, dialect):
    gold = "SELECT name FROM employees WHERE dept_id <> 1"  # NULL dept (Sardor) is not included
    pred = "SELECT name FROM employees WHERE dept_id IS NULL OR dept_id <> 1"
    assert evaluator.score(rec(dialect, gold, gold)).acc
    assert not evaluator.score(rec(dialect, gold, pred)).acc


def test_empty_gold_is_flagged(evaluator, dialect):
    q = "SELECT name FROM employees WHERE salary > 99999"
    s = evaluator.score(rec(dialect, q, q))
    assert s.acc and s.gold_empty


def test_syntax_error_is_pred_error(evaluator, dialect):
    s = evaluator.score(rec(dialect, GOOD, "SELEC name FROM employees"))
    assert s.gold_ok and not s.acc and s.pred_status == ["error"]


def test_unknown_column_is_pred_error(evaluator, dialect):
    s = evaluator.score(rec(dialect, GOOD, "SELECT nme FROM employees"))
    assert s.pred_status == ["error"]


def test_timeout_is_classified_and_returns_promptly(evaluator, dialect):
    start = time.monotonic()
    s = evaluator.score(rec(dialect, GOOD, SLEEPS))
    assert s.pred_status == ["timeout"] and not s.acc
    assert time.monotonic() - start < 5


def test_connection_usable_after_timeout(evaluator, dialect):
    evaluator.score(rec(dialect, GOOD, SLEEPS))
    assert evaluator.score(rec(dialect, GOOD, GOOD)).acc


@pytest.mark.parametrize("sql", [
    "DELETE FROM employees",
    "DROP TABLE employees",
    "UPDATE employees SET salary = 0",
    "INSERT INTO departments VALUES (99, 'X')",
])
def test_writes_are_blocked_and_data_survives(evaluator, dialect, sql):
    s = evaluator.score(rec(dialect, GOOD, sql))
    assert s.pred_status == ["error"]
    count = evaluator.run(dialect, testdb.DB_ID, "SELECT COUNT(*) FROM employees")
    assert count.ok and count.rows == [(7,)]


def test_multiple_statements_are_rejected(evaluator, dialect):
    s = evaluator.score(rec(dialect, GOOD, f"{GOOD}; SELECT 1"))
    assert s.pred_status == ["error"] and not s.acc


def test_percent_in_like_is_not_a_placeholder(evaluator, dialect):
    q = "SELECT name FROM employees WHERE name LIKE '%bur' ORDER BY id"  # only Bobur (Jasur ends in 'sur')
    assert evaluator.run(dialect, testdb.DB_ID, q).rows == [("Bobur",)]


def test_failing_gold_is_excluded(evaluator, dialect):
    s = evaluator.score(rec(dialect, "SELECT * FROM no_such_table", GOOD))
    assert not s.gold_ok and s.gold_error and not s.acc
    assert summarize([s])["overall"]["all"]["n"] == 0


def test_missing_database_raises_instead_of_scoring_wrong(evaluator, dialect):
    r = rec(dialect, GOOD, GOOD)
    r["db_id"] = "does_not_exist"
    with pytest.raises(Exception):
        evaluator.score(r)


# --- majority voting -------------------------------------------------------------------------

def test_maj_recovers_when_first_is_wrong(evaluator, dialect):
    s = evaluator.score(rec(dialect, GOOD, [WRONG, GOOD, GOOD]))
    assert not s.acc and s.maj and s.oracle


def test_maj_can_be_wrong_when_wrong_answers_agree(evaluator, dialect):
    s = evaluator.score(rec(dialect, GOOD, [GOOD, WRONG, WRONG + " "]))
    assert s.acc and not s.maj and s.oracle


def test_maj_tie_goes_to_first_prediction(evaluator, dialect):
    assert evaluator.score(rec(dialect, GOOD, [GOOD, WRONG])).maj
    assert not evaluator.score(rec(dialect, GOOD, [WRONG, GOOD])).maj


def test_failed_executions_do_not_vote(evaluator, dialect):
    s = evaluator.score(rec(dialect, GOOD, ["SELEC x", GOOD, "SELEC y", "SELEC z"]))
    assert s.maj and not s.acc


def test_all_failed_is_wrong(evaluator, dialect):
    s = evaluator.score(rec(dialect, GOOD, ["SELEC x", "SELEC y"]))
    assert not (s.acc or s.maj or s.oracle)


def test_majority_pick_unit():
    assert majority_pick([None, None]) is None
    assert majority_pick(["a", "b", "b"]) == "b"
    assert majority_pick(["a", "b"]) == "a"


# --- reporting --------------------------------------------------------------------------------

def test_report_groups_and_denominators(evaluator, dialect):
    scored = [
        evaluator.score(rec(dialect, GOOD, GOOD, id="1", language="en", slice="traps")),
        evaluator.score(rec(dialect, GOOD, WRONG, id="2", language="uz", slice="traps")),
        evaluator.score(rec(dialect, GOOD, GOOD, id="3", language="uz", slice="main")),
        evaluator.score(rec(dialect, "SELECT * FROM nope", GOOD, id="4", language="uz", slice="main")),
    ]
    summary = summarize(scored)
    assert summary["overall"]["all"]["n"] == 3 and summary["overall"]["all"]["gold_errors"] == 1
    assert summary["language"]["en"]["acc"] == 100.0
    assert summary["language"]["uz"]["n"] == 2 and summary["language"]["uz"]["acc"] == 50.0
    assert summary["slice"]["traps"]["acc"] == 50.0
    assert summary["language/dialect"][f"uz/{dialect}"]["n"] == 2


def test_parallel_prefetch_matches_sequential(dialect, backend):
    from eval.harness import Evaluator
    records = [rec(dialect, GOOD, [GOOD, WRONG, "SELEC x"], id=str(i)) for i in range(20)]
    records += [rec(dialect, "SELECT * FROM nope", GOOD, id="g"), rec(dialect, GOOD, SLEEPS, id="t")]
    sequential = [Evaluator({dialect: backend}).score(r) for r in records]
    parallel_ev = Evaluator({dialect: backend})
    parallel_ev.prefetch(records, workers=6)
    assert [parallel_ev.score(r) for r in records] == sequential


def test_sqlite_dir_may_be_relative(sqlite_dir, monkeypatch):
    from eval.backends import SQLiteBackend
    monkeypatch.chdir(sqlite_dir.parent)
    b = SQLiteBackend(sqlite_dir.relative_to(sqlite_dir.parent), 1.0)
    assert b.execute(testdb.DB_ID, "SELECT COUNT(*) FROM employees").rows == [(7,)]
    b.close()


def test_cli_end_to_end(dialect, sqlite_dir, tmp_path, capsys):
    preds = tmp_path / "preds.jsonl"
    preds.write_text(
        "\n".join(json.dumps(r) for r in [rec(dialect, GOOD, GOOD, id="a"), rec(dialect, GOOD, WRONG, id="b")]),
        encoding="utf-8",
    )
    out = tmp_path / "report.json"
    assert main([str(preds), "--sqlite-dir", str(sqlite_dir), "--timeout", "2", "--out", str(out)]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["summary"]["overall"]["all"]["acc"] == 50.0
    assert len(report["records"]) == 2
    assert "acc" in capsys.readouterr().out
