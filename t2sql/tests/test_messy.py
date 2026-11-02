import sqlite3

import psycopg
import pytest

from eval.backends import PostgresBackend
from eval.config import pg_config
from t2sql import convert
from t2sql.convert import load_database
from t2sql.messy import (
    abbreviate, build_map, messy_schema_name, parse_forms, rename_hint, rename_sql, sanitize, style_of, verify,
)
from t2sql.pg_prompts import ddl_is_valid, schema_ddl

FORMS = {
    "employees": {"uz": "xodimlar", "ru": "сотрудники", "translit": "sotrudniki"},
    "emp_name": {"uz": "xodim_ismi", "ru": "имя_сотрудника", "translit": "imya_sotrudnika"},
    "pay": {"uz": "maosh", "ru": "зарплата", "translit": "zarplata"},
}


def test_sanitize_removes_apostrophes_and_spaces():
    assert sanitize("O'quvchi Ismi") == "oquvchi_ismi"
    assert sanitize("bo‘lim") == "bolim"
    assert sanitize("Имя Сотрудника") == "имя_сотрудника"


def test_abbreviate():
    assert abbreviate("customer_name") == "cstmr_nm"
    assert abbreviate("user_id") == "usr_id"
    assert abbreviate("id") == "id"


def test_build_map_styles():
    ids = ["employees", "emp_name", "pay"]
    assert build_map(ids, FORMS, "uz_latin", "s") == {"employees": "xodimlar", "emp_name": "xodim_ismi", "pay": "maosh"}
    assert build_map(ids, FORMS, "cyrillic", "s")["employees"] == "сотрудники"
    assert build_map(ids, FORMS, "translit", "s")["pay"] == "zarplata"
    assert build_map(ids, {}, "abbrev", "s")["employees"] == "emply"


def test_build_map_never_merges_two_columns():
    forms = {"a": {"uz": "nom"}, "b": {"uz": "Nom"}}
    assert build_map(["a", "b"], forms, "uz_latin", "s") == {"a": "nom", "b": "nom_2"}


def test_build_map_missing_translation_raises():
    with pytest.raises(ValueError):
        build_map(["unknown"], FORMS, "uz_latin", "s")


def test_style_is_stable():
    assert style_of("some_db") == style_of("some_db")


def test_parse_forms_accepts_fenced_json():
    assert parse_forms('```json\n{"a": {"uz": "b"}}\n```') == {"a": {"uz": "b"}}


MAP = {"employees": "xodimlar", "emp_name": "xodim ismi", "pay": "maosh"}


def test_rename_sql_changes_names_not_values_or_functions():
    out = rename_sql("SELECT e.emp_name, AVG(e.pay) FROM employees AS e WHERE e.emp_name = 'pay' GROUP BY e.emp_name", MAP)
    assert '"xodimlar"' in out and '"xodim ismi"' in out and '"maosh"' in out
    assert "'pay'" in out  # the value stays
    assert "AVG(" in out.upper()
    assert "employees" not in out


def test_rename_hint_only_underscored_names():
    assert rename_hint("'emp_name' holds the pay person", MAP) == "'xodim ismi' holds the pay person"
    assert rename_hint(None, MAP) is None


@pytest.fixture(scope="module")
def loaded(tmp_path_factory):
    d = tmp_path_factory.mktemp("messy")
    path = d / "m.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE dept (dept_id INTEGER PRIMARY KEY, title TEXT);
        INSERT INTO dept VALUES (1, 'Sales'), (2, 'HR');
        CREATE TABLE employees (id INTEGER PRIMARY KEY, emp_name TEXT, pay REAL, dept_id INTEGER REFERENCES dept(dept_id));
        INSERT INTO employees VALUES (1, 'Aziz', 100.0, 1), (2, 'Bobur', 50.0, 2), (3, 'Madina', 70.0, 2);
    """)
    conn.close()
    mapping = {"dept": "bolimlar", "dept_id": "bolim_id", "title": "nomi", "employees": "Ходимлар", "id": "ид",
               "emp_name": "исм шарифи", "pay": "maosh"}
    cfg = pg_config()
    admin = dict(host=cfg.host, port=cfg.port, user=cfg.admin_user, password=cfg.admin_password, connect_timeout=10)
    plain, messy = "t_messy_plain", "t_messy_renamed"
    try:
        with psycopg.connect(dbname=convert.PG_DATABASE, autocommit=True, **admin) as c:
            for s in (plain, messy):
                c.execute(f'DROP SCHEMA IF EXISTS "{s}" CASCADE')
            load_database(c, path, plain)
            load_database(c, path, messy, rename=mapping)
    except psycopg.OperationalError as exc:
        pytest.skip(f"PostgreSQL or its synsql database is not available: {exc}")
    backend = PostgresBackend(cfg, timeout_s=5.0)
    yield backend, mapping, f"{convert.PG_DATABASE}/{plain}", f"{convert.PG_DATABASE}/{messy}", messy
    backend.close()
    with psycopg.connect(dbname=convert.PG_DATABASE, autocommit=True, **admin) as c:
        for s in (plain, messy):
            c.execute(f'DROP SCHEMA IF EXISTS "{s}" CASCADE')


def test_renamed_query_returns_same_result(loaded):
    backend, mapping, plain, messy, _ = loaded
    gold = "SELECT d.title, COUNT(*), AVG(e.pay) FROM employees AS e JOIN dept AS d ON e.dept_id = d.dept_id GROUP BY d.title"
    row = {"pg_db": plain, "gold_sql": gold}
    assert verify(row, rename_sql(gold, mapping), messy, backend) is None


def test_verify_catches_a_wrong_rewrite(loaded):
    backend, mapping, plain, messy, _ = loaded
    row = {"pg_db": plain, "gold_sql": "SELECT COUNT(*) FROM employees"}
    assert verify(row, 'SELECT COUNT(*) FROM "bolimlar"', messy, backend) == "result_differs"
    assert verify(row, "SELECT COUNT(*) FROM employees", messy, backend) == "renamed_failed"


def test_ddl_of_renamed_schema_is_valid_with_foreign_keys(loaded):
    _, _, _, _, schema = loaded
    cfg = pg_config()
    with psycopg.connect(host=cfg.host, port=cfg.port, dbname=convert.PG_DATABASE, user=cfg.ro_user,
                         password=cfg.ro_password, connect_timeout=10) as conn:
        ddl = schema_ddl(conn, "ignored", schema)
    assert '"Ходимлар"' in ddl and '"исм шарифи"' in ddl
    assert "REFERENCES bolimlar(bolim_id)" in ddl
    assert ddl_is_valid(ddl)


def test_messy_schema_name():
    assert messy_schema_name("x").startswith("m_")
