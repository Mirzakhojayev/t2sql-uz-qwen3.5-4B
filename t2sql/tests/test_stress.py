import random

from t2sql.stress import is_read_only, new_value, planted_mistakes, verdict


def test_only_plain_reads_are_allowed():
    assert is_read_only("SELECT 1")
    assert is_read_only("  with t as (select 1) select * from t")
    assert not is_read_only("DROP TABLE users")
    assert not is_read_only("WITH d AS (DELETE FROM users RETURNING id) SELECT * FROM d")


def test_planted_mistakes_change_one_thing_each():
    sql = "SELECT a FROM t LEFT JOIN u ON t.id = u.id WHERE a > 1 ORDER BY a DESC"
    assert planted_mistakes(sql) == [
        "SELECT a FROM t LEFT JOIN u ON t.id = u.id WHERE a >= 1 ORDER BY a DESC",
        "SELECT a FROM t JOIN u ON t.id = u.id WHERE a > 1 ORDER BY a DESC",
        "SELECT a FROM t LEFT JOIN u ON t.id = u.id WHERE a > 1 ORDER BY a ASC",
    ]
    assert planted_mistakes("SELECT 1") == []


def test_names_stay_distinct_but_categories_and_dates_repeat():
    rng = random.Random(0)
    assert new_value(["Alice", "Bob"], 3, rng) in ("Alice 3", "Bob 3")
    assert new_value(["active", "active", "closed"], 3, rng) in ("active", "closed")
    assert new_value(["2023-05-01", "2023-05-02"], 3, rng) in ("2023-05-01", "2023-05-02")
    assert new_value([], 3, rng) is None


def test_any_difference_makes_the_question_different():
    seeds = lambda *statuses: {"seeds": [{"status": s} for s in statuses]}
    assert verdict(seeds("same", "different", "same")) == "different"
    assert verdict(seeds("same", "order_only")) == "order_only"
    assert verdict(seeds("error", "error")) == "error"
    assert verdict(seeds("same", "error")) == "same"
