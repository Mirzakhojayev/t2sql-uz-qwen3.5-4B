import random
from collections import Counter

import pytest

from t2sql.pilot import (
    CALLS_PER_LEVEL, LANGUAGE_SHARES, MESSY_SHARE_OF_PG, assign_languages, clean_examples, clean_trace, drop_repeats,
    make_examples, pick_balanced, pick_jobs, split_trace, swap_question,
)
from t2sql.prompt import extract_sql


def example(qid, reasoning="Join users to orders.", style="plain", dialect="postgres"):
    return {"question_id": qid, "reasoning": reasoning, "schema_style": style, "dialect": dialect}


def test_clean_examples_drops_meta_talk_messy_styles_and_empty_reasoning():
    rows = [
        example("keep"), example("short", style="abbrev"),
        example("gold", "We need guess gold."), example("bench", "Which is more likely in benchmark?"),
        example("memory", "Let's search memory: similar questions."), example("setup", "The training example should be simple."),
        example("uz", style="uz_latin"), example("mix", style="mixed"), example("cyr", style="cyrillic"),
        example("tr", style="translit"), example("empty", ""),
    ]
    assert {r["question_id"] for r in clean_examples(rows)} == {"keep", "short"}


def test_clean_examples_keeps_every_language_of_a_chosen_question():
    rows = [example(f"pg{i}") for i in range(20) for _ in range(2)] + [
        example(f"lite{i}", dialect="sqlite") for i in range(20) for _ in range(2)]
    kept = clean_examples(rows, questions=20)
    assert len({r["question_id"] for r in kept}) == 20
    assert len(kept) == 40
    assert len({r["question_id"] for r in kept if r["dialect"] == "postgres"}) == 13  # 65% of 20
    assert kept == clean_examples(rows, questions=20)


def rows(prefix, per_level):
    return [{"id": f"{prefix}-{lvl}-{i}", "level": lvl} for lvl in CALLS_PER_LEVEL for i in range(per_level)]


def count(picked, level, prefix=""):
    return sum(r["level"] == level and r["id"].startswith(prefix) for r in picked)


def test_calls_per_level_and_messy_share():
    picked = pick_jobs(rows("plain", 3000), rows("messy", 3000), set(), random.Random(0))
    for lvl, calls in CALLS_PER_LEVEL.items():
        if calls is None:
            assert count(picked, lvl) == 6000  # all of them
        else:
            assert count(picked, lvl) == calls
            assert count(picked, lvl, "messy") == round(calls * MESSY_SHARE_OF_PG)


def test_short_supply_is_filled_by_the_other_kind():
    picked = pick_jobs(rows("plain", 3000), rows("messy", 100), set(), random.Random(0))
    assert count(picked, "Simple") == CALLS_PER_LEVEL["Simple"]
    assert count(picked, "Simple", "messy") == 100
    picked = pick_jobs(rows("plain", 100), rows("messy", 3000), set(), random.Random(0))
    assert count(picked, "Simple") == CALLS_PER_LEVEL["Simple"]
    assert count(picked, "Simple", "plain") == 100


def test_already_answered_questions_are_kept():
    answered = {"plain-Simple-2999", "messy-Moderate-2999"}
    picked = pick_jobs(rows("plain", 3000), rows("messy", 3000), answered, random.Random(0))
    assert answered <= {r["id"] for r in picked}


def test_no_question_is_picked_twice():
    picked = pick_jobs(rows("plain", 3000), rows("messy", 3000), set(), random.Random(0))
    ids = [r["id"] for r in picked]
    assert len(ids) == len(set(ids))


TRACE = "Steps first.\n\n```sql\nSELECT 1;\n```\n\nThen a better one:\n\n```sql\nSELECT 2;\n```\n\n### Explanation\nIt selects 2."


def test_clean_trace_ends_at_the_final_sql_block():
    cleaned = clean_trace(TRACE)
    assert cleaned.endswith("SELECT 2;\n```")
    assert "Explanation" not in cleaned
    assert extract_sql(cleaned) == extract_sql(TRACE)


def test_clean_trace_drops_reference_mentions_and_traces_without_sql():
    assert clean_trace("The Reference Solution joins too much.\n\n```sql\nSELECT 1;\n```") is None
    assert clean_trace("the reference  query is wrong\n\n```sql\nSELECT 1;\n```") is None
    assert clean_trace("no query here") is None
    assert clean_trace("```sql\nSELECT 1;") is None  # never closed


def test_pick_balanced_is_equal_when_supply_allows():
    picked = pick_balanced(rows("x", 100), 200, random.Random(0))
    assert all(count(picked, lvl) == 50 for lvl in CALLS_PER_LEVEL)


def test_pick_balanced_gives_a_short_levels_share_to_the_others():
    short = [r for r in rows("x", 100) if r["level"] != "Highly Complex"]
    short += [{"id": f"hc-{i}", "level": "Highly Complex"} for i in range(10)]
    picked = pick_balanced(short, 200, random.Random(0))
    assert len(picked) == 200
    assert count(picked, "Highly Complex") == 10
    assert all(count(picked, lvl) == 64 or count(picked, lvl) == 63 for lvl in ("Simple", "Moderate", "Complex"))


def test_pick_balanced_never_exceeds_supply_or_repeats():
    picked = pick_balanced(rows("x", 2), 100, random.Random(0))
    assert len(picked) == 8 == len({r["id"] for r in picked})


def test_assign_languages_two_different_and_close_to_the_shares():
    group = rows("x", 500)
    assign_languages(group, random.Random(0))
    assert all(len(set(r["languages"])) == 2 for r in group)
    counts = Counter(l for r in group for l in r["languages"])
    for lang, share in LANGUAGE_SHARES.items():
        assert abs(counts[lang] / (2 * len(group)) - share) < 0.01


def test_assign_languages_works_on_tiny_groups():
    for n in (0, 1, 2, 3):
        group = rows("x", 1)[:n]
        assign_languages(group, random.Random(0))
        assert all(len(set(r["languages"])) == 2 for r in group)


def test_drop_repeats_ignores_case_and_spacing_across_calls():
    seen = set()
    a = drop_repeats([{"question": "How many  users?"}, {"question": "how many users?"}], seen)
    b = drop_repeats([{"question": "HOW MANY USERS?"}, {"question": "Other?"}], seen)
    assert len(a) == 1 and [r["question"] for r in b] == ["Other?"]


def test_split_trace_puts_only_the_final_block_in_the_answer():
    reasoning, answer = split_trace(clean_trace(TRACE))
    assert "SELECT 1;" in reasoning and "SELECT 2;" not in reasoning
    assert answer == "```sql\nSELECT 2;\n```"


def test_swap_question_keeps_schema_and_hint():
    prompt = "Dialect: PostgreSQL\n\nSchema:\nCREATE TABLE q (a INT);\n\nHint: Question: is not here\n\nQuestion: How many?"
    assert swap_question(prompt, "How many?", "Нечта?") == \
        "Dialect: PostgreSQL\n\nSchema:\nCREATE TABLE q (a INT);\n\nHint: Question: is not here\n\nQuestion: Нечта?"
    with pytest.raises(ValueError):
        swap_question(prompt, "Another?", "Нечта?")


def plan_row(languages, trace_source="deepseek-flash", qid="q1"):
    answer = "Count them.\n\n```sql\nSELECT 1\n```" if trace_source == "synsql" else "Short.\n\n```sql\nSELECT 1\n```"
    return {"id": qid, "level": "Simple", "style": "Formal", "question": "How many orders in 2023?", "hint": None,
            "source": "synsql_2.5m", "license": "Apache-2.0", "schema_style": "plain", "dialect": "postgres",
            "db_id": "synsql/s_x", "source_db": "x", "gold_sql": "SELECT 1", "languages": languages,
            "prompt": "Dialect: PostgreSQL\n\nSchema:\nCREATE TABLE o (a INT);\n\nQuestion: How many orders in 2023?",
            "reasoning": None if trace_source == "synsql" else "Think.", "answer": answer, "target_sql": "SELECT 1",
            "trace_source": trace_source}


def tr(lang, text, finish="stop", qid="q1"):
    return {(qid, lang): {"question_id": qid, "language": lang, "english": "How many orders in 2023?", "content": text,
                          "finish_reason": finish}}


def test_examples_use_the_right_text_per_language():
    translated = {**tr("ru", "Сколько заказов в 2023?"), **tr("uz", "2023-yilda nechta buyurtma bo'lgan?")}
    made, dropped = make_examples(plan_row(["en", "ru"]), translated)
    assert not dropped
    assert [e["language"] for e in made] == ["en", "ru"]
    assert made[0]["question"] == "How many orders in 2023?"
    assert made[1]["messages"][1]["content"].endswith("Question: Сколько заказов в 2023?")
    assert made[1]["question_en"] == "How many orders in 2023?" and made[1]["reasoning"] == "Think."
    made, _ = make_examples(plan_row(["uz_cyrillic", "uz_latin"]), translated)
    assert made[1]["question"] == "2023-yilda nechta buyurtma bo'lgan?"
    assert "2023-йилда" in made[0]["question"] and made[0]["id"] == "q1-uz_cyrillic"


def test_examples_split_a_synsql_trace():
    made, _ = make_examples(plan_row(["en", "en"][:1], trace_source="synsql"), {})
    assert made[0]["reasoning"] == "Count them." and made[0]["answer"] == "```sql\nSELECT 1\n```"


def test_examples_drop_missing_flagged_and_cut_off_translations():
    row = plan_row(["en", "ru"])
    assert make_examples(row, {})[1] == {"no translation": 1}
    assert make_examples(row, tr("ru", "Сколько заказов?"))[1] == {"translation flagged": 1}  # the 2023 was lost
    assert make_examples(row, tr("ru", "Сколько заказов в 2023?", finish="length"))[1] == {"no translation": 1}
    made, _ = make_examples(row, {})
    assert [e["language"] for e in made] == ["en"]
