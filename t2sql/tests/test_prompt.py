"""Spec for t2sql/prompt.py. When every test here passes, the prompt builder is done."""
import datetime
import sqlite3
from decimal import Decimal

import pytest

from eval import testdb
from t2sql.prompt import (
    MAX_VALUE_CHARS, build_prompt, extract_sql, format_value, quote_ident, read_sqlite_schema, schema_to_ddl,
)


# --- quote_ident --------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name, expected", [
    ("employees", "employees"),
    ("_id2", "_id2"),
    ("School Name", '"School Name"'),
    ("CDSCode", '"CDSCode"'),                 # mixed case: PostgreSQL would fold it to cdscode
    ("2col", '"2col"'),
    ("order", '"order"'),                     # reserved word
    ("Ходим", '"Ходим"'),                     # Cyrillic
    ("To'liq ismi", '"To\'liq ismi"'),        # apostrophe is fine inside double quotes
    ('say "hi"', '"say ""hi"""'),             # double quote inside is doubled
    ("add", '"add"'),                         # SQLite keyword, found as a column name in SynSQL
    ("action", '"action"'),                   # SQLite keyword
    ("analyse", '"analyse"'),                 # PostgreSQL reserved word
])
def test_quote_ident(name, expected):
    assert quote_ident(name) == expected


# --- format_value -------------------------------------------------------------------------------------------

def test_format_value_basic():
    assert format_value(None) == "NULL"
    assert format_value(True) == "TRUE"
    assert format_value(3) == "3"
    assert format_value(4000.0) == "4000.0"
    assert format_value("HR") == "'HR'"
    assert format_value("O'Brien") == "'O''Brien'"


def test_format_value_line_breaks_stay_on_one_line():
    # found in SynSQL: song lyrics with newlines broke the "-- e.g." comment and made the DDL invalid
    assert format_value("Verse 1:\nI was born\r\ninto\tthis") == "'Verse 1: I was born into this'"


def test_format_value_postgres_types():
    # PostgreSQL hands back Decimal for NUMERIC and date objects for DATE
    assert format_value(Decimal("4000.00")) == "4000.00"
    assert format_value(datetime.date(2019, 3, 1)) == "'2019-03-01'"


def test_format_value_long_text_is_cut_before_escaping():
    long = "O'g'il bolalar maktabi — juda uzun nom bilan yozilgan maktab nomi"
    assert len(long) > MAX_VALUE_CHARS
    assert format_value(long) == "'" + long[:MAX_VALUE_CHARS].replace("'", "''") + "...'"


# --- extract_sql --------------------------------------------------------------------------------------------

def test_extract_sql_simple():
    assert extract_sql("Count rows.\n\n```sql\nSELECT COUNT(*) FROM t;\n```") == "SELECT COUNT(*) FROM t"


def test_extract_sql_takes_last_block():
    text = "First try:\n```sql\nSELECT 1\n```\nBetter:\n```sql\nSELECT 2\n```"
    assert extract_sql(text) == "SELECT 2"


def test_extract_sql_ignores_thinking():
    text = "<think>\nmaybe\n```sql\nSELECT wrong\n```\n</think>\n\nAnswer:\n```sql\nSELECT right\n```"
    assert extract_sql(text) == "SELECT right"


def test_extract_sql_only_inside_thinking_is_none():
    assert extract_sql("<think>\n```sql\nSELECT 1\n```\n</think>\nI am not sure.") is None


def test_extract_sql_unfinished_thinking_is_none():
    # the model hit the token limit while thinking: a draft inside the unclosed block is not an answer
    assert extract_sql("<think>\nlet me try\n```sql\nSELECT draft\n```\nwait, I should check") is None
    # but a real answer that came before is kept
    assert extract_sql("```sql\nSELECT 1\n```\n<think>\nmore thoughts, never closed") == "SELECT 1"


def test_extract_sql_case_and_plain_fence():
    assert extract_sql("```SQL\nSELECT 3;;\n```") == "SELECT 3"
    assert extract_sql("```\nSELECT 4\n```") == "SELECT 4"
    assert extract_sql("```sql\nSELECT 5\n```\n```\nnot sql\n```") == "SELECT 5"  # tagged block wins


def test_extract_sql_none_or_cut_off():
    assert extract_sql("I cannot answer that.") is None
    assert extract_sql("```sql\nSELECT name FROM employees WHERE") is None  # never closed


# --- build_prompt -------------------------------------------------------------------------------------------

DDL = "CREATE TABLE t (\n  a INTEGER\n);"


def test_build_prompt_with_hint():
    assert build_prompt("sqlite", DDL, "How many rows?", hint="a is an id") == (
        "Dialect: SQLite\n\nSchema:\nCREATE TABLE t (\n  a INTEGER\n);\n\nHint: a is an id\n\nQuestion: How many rows?"
    )


def test_build_prompt_without_hint():
    for hint in (None, ""):
        assert build_prompt("postgres", DDL, "How many rows?", hint=hint) == (
            "Dialect: PostgreSQL\n\nSchema:\nCREATE TABLE t (\n  a INTEGER\n);\n\nQuestion: How many rows?"
        )
    assert build_prompt("postgresql", DDL, "Q").startswith("Dialect: PostgreSQL\n")
    with pytest.raises(ValueError):
        build_prompt("mysql", DDL, "Q")


# --- read_sqlite_schema + schema_to_ddl ---------------------------------------------------------------------

TINY_DDL = """\
CREATE TABLE departments (
  id INTEGER PRIMARY KEY, -- e.g. 1, 2, 3
  name TEXT -- e.g. 'Engineering', 'HR', 'Sales'
);

CREATE TABLE employees (
  id INTEGER PRIMARY KEY, -- e.g. 1, 2, 3
  name TEXT, -- e.g. 'Aziz', 'Bobur', 'Dilnoza'
  dept_id INTEGER, -- e.g. 2, 1, 3
  salary REAL, -- e.g. 4000.0, 3000.0, 3500.0
  hired TEXT, -- e.g. '2019-03-01', '2020-07-15', '2021-01-10'
  manager_id INTEGER, -- e.g. 1, 2, 3
  FOREIGN KEY (dept_id) REFERENCES departments(id),
  FOREIGN KEY (manager_id) REFERENCES employees(id)
);"""


@pytest.fixture
def tiny_sqlite(tmp_path):
    path = tmp_path / "tiny.sqlite"
    testdb.build_sqlite(path)
    return path


def test_tiny_schema_objects(tiny_sqlite):
    tables = read_sqlite_schema(tiny_sqlite)
    by_name = {t.name: t for t in tables}
    assert set(by_name) == {"departments", "employees"}
    emp = by_name["employees"]
    assert [c.name for c in emp.columns] == ["id", "name", "dept_id", "salary", "hired", "manager_id"]
    assert [c.primary_key for c in emp.columns] == [True, False, False, False, False, False]
    assert emp.primary_key == []  # single-column key is marked on the column, not here
    assert [(f.column, f.ref_table, f.ref_column) for f in emp.foreign_keys] == [
        ("dept_id", "departments", "id"), ("manager_id", "employees", "id")]
    # most frequent first: 4000 appears twice; ties (3000, 3500, ...) go ascending
    assert next(c for c in emp.columns if c.name == "salary").examples == [4000.0, 3000.0, 3500.0]


def test_tiny_ddl(tiny_sqlite):
    assert schema_to_ddl(read_sqlite_schema(tiny_sqlite)) == TINY_DDL


def test_n_examples(tiny_sqlite):
    tables = read_sqlite_schema(tiny_sqlite, n_examples=1)
    assert all(len(c.examples) <= 1 for t in tables for c in t.columns)


MESSY_DDL = """\
CREATE TABLE "Maktablar" (
  "Maktab ID" INTEGER PRIMARY KEY, -- e.g. 1, 2, 3
  "Nomi" TEXT, -- e.g. '1-maktab', '3-maktab', 'O''g''il bolalar'
  "Ходимлар сони" INTEGER, -- e.g. 40, 55
  "order" TEXT, -- e.g. 'A', 'B'
  izoh
);

CREATE TABLE baholar (
  maktab_id INTEGER, -- e.g. 1, 2
  yil INTEGER, -- e.g. 2024, 2025
  ball REAL, -- e.g. 4.5, 4.7
  PRIMARY KEY (maktab_id, yil),
  FOREIGN KEY (maktab_id) REFERENCES "Maktablar"("Maktab ID")
);

CREATE TABLE xodimlar (
  id INTEGER PRIMARY KEY, -- e.g. 1, 2, 3
  maktab INTEGER, -- e.g. 1, 2
  FOREIGN KEY (maktab) REFERENCES "Maktablar"("Maktab ID")
);"""


@pytest.fixture
def messy_sqlite(tmp_path):
    """Local-style messy schema: Uzbek names, a space, Cyrillic, a reserved word, an untyped all-NULL column."""
    path = tmp_path / "messy.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript("""
        CREATE TABLE "Maktablar" ("Maktab ID" INTEGER PRIMARY KEY, "Nomi" TEXT, "Ходимлар сони" INTEGER,
                                  "order" TEXT, izoh);
        INSERT INTO "Maktablar" VALUES (1, '1-maktab', 40, 'A', NULL), (2, 'O''g''il bolalar', 55, 'B', NULL),
                                       (3, '3-maktab', 40, 'A', NULL);
        CREATE TABLE baholar (maktab_id INTEGER, yil INTEGER, ball REAL, PRIMARY KEY (maktab_id, yil),
                              FOREIGN KEY (maktab_id) REFERENCES "Maktablar"("Maktab ID"));
        INSERT INTO baholar VALUES (1, 2024, 4.5), (1, 2025, 4.7), (2, 2024, 4.5);
        -- BIRD-style quirks: a foreign key naming only the table, declared twice
        CREATE TABLE xodimlar (id INTEGER PRIMARY KEY, maktab INTEGER REFERENCES "Maktablar",
                               FOREIGN KEY (maktab) REFERENCES "Maktablar");
        INSERT INTO xodimlar VALUES (1, 1), (2, 1), (3, 2);
    """)
    conn.commit()
    conn.close()
    return path


def test_messy_ddl(messy_sqlite):
    assert schema_to_ddl(read_sqlite_schema(messy_sqlite)) == MESSY_DDL


def test_ddl_runs_in_sqlite(messy_sqlite, tmp_path):
    """The DDL we show the model must be valid SQL: rebuild an empty copy from it."""
    ddl = schema_to_ddl(read_sqlite_schema(messy_sqlite))
    conn = sqlite3.connect(tmp_path / "copy.sqlite")
    conn.executescript(ddl)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    conn.close()
    assert names == {"Maktablar", "baholar", "xodimlar"}


def test_keyword_column_names(tmp_path):
    """Columns named like SQL keywords must be quoted, both in our sampling queries and in the DDL."""
    path = tmp_path / "kw.sqlite"
    conn = sqlite3.connect(path)
    conn.executescript('CREATE TABLE t ("add" INTEGER, "action" TEXT, "group" TEXT); INSERT INTO t VALUES (1, \'x\', \'g\');')
    conn.close()
    ddl = schema_to_ddl(read_sqlite_schema(path))
    assert '"add" INTEGER' in ddl and '"action" TEXT' in ddl and '"group" TEXT' in ddl
    copy = sqlite3.connect(tmp_path / "copy.sqlite")
    copy.executescript(ddl)  # must be valid SQL
    copy.close()


def test_read_only(tiny_sqlite):
    before = tiny_sqlite.read_bytes()
    read_sqlite_schema(tiny_sqlite)
    assert tiny_sqlite.read_bytes() == before
