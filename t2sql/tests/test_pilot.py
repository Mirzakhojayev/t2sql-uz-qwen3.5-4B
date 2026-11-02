import random

from t2sql.pilot import CALLS_PER_LEVEL, MESSY_SHARE_OF_PG, pick_jobs


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
