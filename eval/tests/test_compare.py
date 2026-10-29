import datetime
from decimal import Decimal

from eval.compare import is_ordered, normalize_rows, signature


def sig(rows, ordered=False):
    return signature(normalize_rows(rows), ordered)


def test_row_order_ignored_when_unordered():
    assert sig([(1,), (2,), (3,)]) == sig([(3,), (1,), (2,)])


def test_row_order_matters_when_ordered():
    assert sig([(1,), (2,)], ordered=True) != sig([(2,), (1,)], ordered=True)


def test_duplicates_count():
    assert sig([(1,), (1,), (2,)]) != sig([(1,), (2,)])


def test_column_order_matters():
    assert sig([(1, "a")]) != sig([("a", 1)])


def test_numeric_types_equal():
    assert sig([(3,)]) == sig([(3.0,)]) == sig([(Decimal("3.00"),)])
    assert sig([(True,)]) == sig([(1,)])


def test_float_rounding():
    assert sig([(4.3333333333,)]) == sig([(Decimal("4.3333333333333333"),)])
    assert sig([(4.33,)]) != sig([(4.34,)])


def test_null_is_not_zero_or_empty_string():
    assert sig([(None,)]) != sig([(0,)])
    assert sig([(None,)]) != sig([("",)])


def test_date_matches_iso_string():
    assert sig([(datetime.date(2020, 7, 15),)]) == sig([("2020-07-15",)])


def test_arrays_and_dicts_hashable():
    assert sig([([1, 2],)]) == sig([([1, 2],)])
    assert sig([({"b": 1, "a": 2},)]) == sig([({"a": 2, "b": 1},)])


def test_is_ordered_detects_top_level_order_by():
    assert is_ordered("SELECT name FROM employees ORDER BY salary DESC", "sqlite")
    assert is_ordered("WITH x AS (SELECT 1 AS a) SELECT a FROM x ORDER BY a", "postgres")
    assert is_ordered("SELECT a FROM t UNION SELECT a FROM u ORDER BY a", "postgres")


def test_is_ordered_ignores_nested_order_by():
    assert not is_ordered("SELECT name FROM employees", "sqlite")
    assert not is_ordered("SELECT * FROM (SELECT name FROM employees ORDER BY salary LIMIT 3) t", "postgres")
