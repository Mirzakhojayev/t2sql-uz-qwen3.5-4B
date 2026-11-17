import sqlite3

from t2sql.describe import describe, load_descriptions, matching_values, value_index
from t2sql.prompt import Column, Table, build_prompt, schema_to_ddl

CSV = (
    "﻿original_column_name,column_name,column_description,data_format,value_description\n"
    "district_id,location of branch,location of branch,integer ,\n"
    "A2,district_name,district_name,text,\n"
    "A9,,,integer,not useful\n"
    "A11,average salary,average salary,integer,\n"
    'gender,,,text,"F: female\nM: male"\n'
    "name,name,name,text,\n"
)


def test_descriptions_keep_only_what_the_name_does_not_say(tmp_path):
    (tmp_path / "District.csv").write_text(CSV, encoding="utf-8")
    d = load_descriptions(tmp_path)["district"]
    assert d["a11"] == "average salary"
    assert d["a2"] == "district_name"
    assert d["district_id"] == "location of branch"
    assert d["gender"] == "F: female M: male"
    assert "a9" not in d and "name" not in d
    assert load_descriptions(tmp_path / "missing") == {}


def test_description_comes_before_the_examples_in_the_schema(tmp_path):
    (tmp_path / "district.csv").write_text(CSV, encoding="utf-8")
    tables = [Table("district", [Column("a11", "BIGINT", examples=[8369, 8110]), Column("a9", "BIGINT", examples=[6])])]
    ddl = schema_to_ddl(describe(tables, load_descriptions(tmp_path)))
    assert "  a11 BIGINT, -- average salary; e.g. 8369, 8110\n" in ddl
    assert "  a9 BIGINT -- e.g. 6\n" in ddl


def test_matching_values_finds_stored_spellings(tmp_path):
    path = tmp_path / "db.sqlite"
    conn = sqlite3.connect(path)
    conn.execute('CREATE TABLE Schools (County TEXT, "School Type" TEXT, note TEXT, n INTEGER)')
    conn.executemany("INSERT INTO Schools VALUES (?, ?, ?, ?)", [
        ("Los Angeles", "High Schools (Public)", "the", 1), ("Alameda", "Elementary", "12", 2)])
    conn.commit()
    conn.close()
    index = value_index(path)
    assert "the" in index and "12" not in index
    block = matching_values("Qaysi maktablar los angeles okrugida, Alameda emas? the", index, rename=str.lower)
    assert block.splitlines() == ["schools.county: 'Los Angeles'", "schools.county: 'Alameda'"]
    assert matching_values("How many schools are there?", index) == ""


def test_values_block_sits_between_schema_and_hint():
    prompt = build_prompt("postgres", "CREATE TABLE t (a INT);", "Q?", hint="H", values="t.a: 'x'")
    assert prompt == ("Dialect: PostgreSQL\n\nSchema:\nCREATE TABLE t (a INT);\n\n"
                      "Values in the database that match the question:\nt.a: 'x'\n\nHint: H\n\nQuestion: Q?")
    assert "Values in" not in build_prompt("postgres", "CREATE TABLE t (a INT);", "Q?")
