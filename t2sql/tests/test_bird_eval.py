import json

from t2sql.bird_eval import build_rows, seed_with_reviewed

Q = "How many patients in 2023?"


def english(dialect="postgres", qid=7):
    return {"id": f"{dialect}-{qid}", "db_id": "d", "dialect": dialect, "slice": "simple", "gold_sql": "SELECT 1",
            "source": "bird_minidev", "license": "CC BY-SA 4.0", "split": "eval", "language": "en",
            "messages": [{"role": "system", "content": "sys"},
                         {"role": "user", "content": f"Dialect: X\n\nSchema:\nCREATE TABLE t (a INT);\n\nHint: h\n\nQuestion: {Q}"}]}


def translated(qid=7, ru=f"Сколько пациентов в 2023?", uz="2023-yilda nechta bemor bo'lgan?"):
    return {(qid, "ru"): {"language": "ru", "english": Q, "content": ru},
            (qid, "uz"): {"language": "uz", "english": Q, "content": uz}}


def test_every_language_keeps_gold_and_hint_and_swaps_the_question():
    files = build_rows([english()], translated(), {7})
    assert {k: len(v) for k, v in files.items()} == {"ru": 1, "uz": 1, "uz_cyrillic": 1}
    uz = files["uz"][0]
    assert uz["id"] == "postgres-7-uz_latin" and uz["language"] == "uz_latin" and uz["gold_sql"] == "SELECT 1"
    assert uz["messages"][1]["content"].endswith("Hint: h\n\nQuestion: 2023-yilda nechta bemor bo'lgan?")
    assert uz["question_en"] == Q and uz["verified"] is False
    assert "2023-йилда" in files["uz_cyrillic"][0]["question"]


def test_only_the_chosen_questions_get_cyrillic():
    files = build_rows([english()], translated(), set())
    assert files["uz_cyrillic"] == []


def test_a_flagged_translation_is_kept_with_its_flag():
    files = build_rows([english()], translated(ru="Сколько пациентов?"), set())
    assert len(files["ru"]) == 1
    assert files["ru"][0]["flags"] == ["numbers missing: ['2023']"]


def test_seed_copies_only_wanted_rows_once_and_at_no_cost(tmp_path):
    reviewed = tmp_path / "reviewed.jsonl"
    reviewed.write_text("\n".join(json.dumps({"id": i, "cost_usd": 0.5}) for i in ("ru-1", "ru-2")) + "\n", encoding="utf-8")
    out = tmp_path / "out.jsonl"
    assert seed_with_reviewed(out, {"ru-1"}, reviewed) == 1
    assert seed_with_reviewed(out, {"ru-1"}, reviewed) == 0
    rows = [json.loads(line) for line in open(out, encoding="utf-8")]
    assert rows == [{"id": "ru-1", "cost_usd": 0.0}]
