import importlib.util
import random
from pathlib import Path

spec = importlib.util.spec_from_file_location("train_script", Path(__file__).parents[2] / "modal" / "train.py")
train_script = importlib.util.module_from_spec(spec)
spec.loader.exec_module(train_script)

lr_at, is_holdout, target_text, token_batches = (train_script.lr_at, train_script.is_holdout, train_script.target_text,
                                                 train_script.token_batches)


def test_target_closes_thinking_then_answer_and_stop_token():
    assert target_text({"reasoning": " Count rows.\n", "answer": "```sql\nSELECT 1\n```\n"}) == \
        "Count rows.\n</think>\n\n```sql\nSELECT 1\n```<|im_end|>"


def test_target_without_thinking_is_the_answer_alone():
    assert target_text({"reasoning": "", "answer": "```sql\nSELECT 1\n```\n"}, thinking=False) == "```sql\nSELECT 1\n```<|im_end|>"


def test_holdout_is_stable_and_small():
    ids = [f"synsql-{i}" for i in range(5000)]
    held = [i for i in ids if is_holdout(i)]
    assert held == [i for i in ids if is_holdout(i)]
    assert 0.01 < len(held) / len(ids) < 0.03


def lengths_for_test():
    rng = random.Random(1)
    return [rng.randint(500, 16000) for _ in range(2000)]


def test_batches_cover_everything_once():
    lengths = lengths_for_test()
    batches = token_batches(lengths, 32768, 8, random.Random(0))
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))


def test_batches_stay_within_limits():
    lengths = lengths_for_test()
    for b in token_batches(lengths, 32768, 8, random.Random(0)):
        assert len(b) <= 8
        assert max(lengths[i] for i in b) * len(b) <= 32768


def test_one_example_over_the_budget_still_gets_a_batch():
    assert token_batches([40000], 32768, 8, random.Random(0)) == [[0]]


def test_batches_group_similar_lengths():
    lengths = lengths_for_test()
    batches = token_batches(lengths, 32768, 8, random.Random(0))
    padded = sum(max(lengths[i] for i in b) * len(b) for b in batches)
    assert padded < 1.2 * sum(lengths)


def test_longest_first_puts_the_largest_batch_first():
    lengths = lengths_for_test()
    batches = token_batches(lengths, 32768, 8, random.Random(0), longest_first=True)
    sizes = [max(lengths[i] for i in b) * len(b) for b in batches]
    assert sizes[0] == max(sizes)
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))


def test_different_seeds_give_different_order():
    lengths = lengths_for_test()
    assert token_batches(lengths, 32768, 8, random.Random(0)) != token_batches(lengths, 32768, 8, random.Random(1))


def test_lr_warms_up_then_falls_to_zero():
    total = 1000
    assert lr_at(0, total, 1.0) < lr_at(49, total, 1.0)
    assert lr_at(49, total, 1.0) == 1.0
    assert lr_at(500, total, 1.0) < 1.0
    assert lr_at(1000, total, 1.0) < 1e-6
