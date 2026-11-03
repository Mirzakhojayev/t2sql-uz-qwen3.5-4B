import json

from t2sql.leakage import EvalSet, find_leaks, load_eval_sets, norm_question, norm_sql


def example(**kw):
    return {"question": "Сколько певцов?", "question_en": "How many singers?", "gold_sql": "SELECT 1", "target_sql": "SELECT 2", **kw}


def test_norm_question_ignores_case_spacing_and_end_marks():
    assert norm_question("  How  many Singers? ") == "how many singers"


def test_norm_sql_ignores_case_spacing_quotes_and_semicolon():
    assert norm_sql('SELECT  "A"\nFROM `t`;') == "select a from t"


def test_finds_english_and_translated_questions():
    sets = {"s": EvalSet()}
    sets["s"].add("how many singers")
    assert find_leaks(example(), sets) == ["s:question"]
    sets = {"s": EvalSet()}
    sets["s"].add("СКОЛЬКО ПЕВЦОВ")
    assert find_leaks(example(), sets) == ["s:question"]


def test_finds_gold_and_target_sql():
    sets = {"s": EvalSet()}
    sets["s"].add(None, "select 2;")
    assert find_leaks(example(), sets) == ["s:sql"]


def test_clean_example_has_no_leaks():
    sets = {"s": EvalSet()}
    sets["s"].add("something else", "SELECT 3")
    assert find_leaks(example(), sets) == []


def test_loader_reads_every_format(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq

    mini = tmp_path / "mini"
    mini.mkdir()
    (mini / "mini_dev_sqlite.jsonl").write_text(json.dumps({"question": "M", "SQL": "SELECT m", "db_id": "d1"}) + "\n", encoding="utf-8")
    (tmp_path / "bird_dev/data").mkdir(parents=True)
    (tmp_path / "bird_dev/data/dev_20251106-00000-of-00001.json").write_text(
        json.dumps([{"question": "B", "SQL": "SELECT b", "db_id": "d2"}]), encoding="utf-8")
    (tmp_path / "spider/spider").mkdir(parents=True)
    pq.write_table(pa.Table.from_pylist([{"question": "S", "query": "SELECT s", "db_id": "d3"}]),
                   tmp_path / "spider/spider/validation-00000-of-00001.parquet")
    (tmp_path / "pauq/splits").mkdir(parents=True)
    (tmp_path / "pauq/splits/ru_os_test.json").write_text(
        json.dumps([{"question": "П", "query": "SELECT p", "db_id": "d4"}]), encoding="utf-8")
    (tmp_path / "redsql/responses").mkdir(parents=True)
    (tmp_path / "redsql/responses/text2SQL_x_responses.json").write_text(
        json.dumps({"x": [{"question": "Р", "sql_query": "SELECT r"}]}), encoding="utf-8")
    (tmp_path / "bird_critic_open/data").mkdir(parents=True)
    (tmp_path / "bird_critic_open/data/open-00000-of-000001.jsonl").write_text(
        json.dumps({"query": "C", "issue_sql": "['SELECT c']", "db_id": "d5"}) + "\n", encoding="utf-8")

    sets = load_eval_sets(tmp_path, mini)
    assert sets["bird_minidev"].sqls == {"select m"} and sets["bird_minidev"].databases == {"d1"}
    assert sets["bird_dev"].questions == {"b"}
    assert sets["spider_dev"].sqls == {"select s"}
    assert sets["pauq_test"].questions == {"п"}
    assert sets["redsql"].sqls == {"select r"}
    assert sets["bird_critic"].questions == {"c"} and sets["bird_critic"].sqls == {"select c"}
