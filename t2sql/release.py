"""Build the files published as the Hugging Face dataset `mirzakhojayev/uzbird`.

  python -m t2sql.release          # data/release/uzbird/

  standard.jsonl   eval: 500 BIRD dev questions x EN/RU/UZ Latin (+100 UZ Cyrillic) = 1,600 rows
  hard.jsonl       eval: the 500 BIRD Mini-Dev questions x RU/UZ Latin (+100 UZ Cyrillic) = 1,100 rows
  train.jsonl      train: 8,090 BIRD train / SynSQL questions in up to 4 languages = 30,088 rows
  prompts/*.jsonl  the exact model input (system + user message) of every eval row, for reproduction

Gold SQL is PostgreSQL for the eval sets, and PostgreSQL or SQLite (column `dialect`) for train. The databases
themselves are not included: they come from BIRD (bird-bench.github.io) and SynSQL-2.5M.
"""
import json
from collections import Counter
from pathlib import Path

from .final import HARD_V2, STANDARD, TRAIN_FILE, old_hint

OUT = Path("data/release/uzbird")


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def eval_row(r: dict, source: str) -> dict:
    return {
        "id": r["id"], "question_id": r["question_id"], "source": source, "language": r["language"],
        "script_typing": r["typing"], "difficulty": r["slice"], "db_id": r["db_id"], "question": r["question"],
        "question_en": r["question_en"], "evidence": old_hint(r["messages"][1]["content"]),
        "gold_sql_postgres": r["gold_sql"], "auto_flags": r.get("flags", []),
    }


def build() -> dict:
    standard = load_jsonl(STANDARD)
    hard = load_jsonl(HARD_V2)
    write_jsonl(OUT / "standard.jsonl", [eval_row(r, "bird_dev") for r in standard])
    write_jsonl(OUT / "hard.jsonl", [eval_row(r, "bird_minidev") for r in hard])
    write_jsonl(OUT / "prompts" / "standard_prompts.jsonl", [{"id": r["id"], "messages": r["messages"]} for r in standard])
    write_jsonl(OUT / "prompts" / "hard_prompts.jsonl", [{"id": r["id"], "messages": r["messages"]} for r in hard])
    train = [{
        "id": e["id"], "question_id": e["question_id"], "source": e["source"], "license": e["license"],
        "language": e["language"], "script_typing": e["typing"], "dialect": e["dialect"], "db_id": e["source_db"],
        "question": e["question"], "question_en": e["question_en"], "evidence": e["hint"], "gold_sql": e["gold_sql"],
    } for e in load_jsonl(TRAIN_FILE)]
    write_jsonl(OUT / "train.jsonl", train)
    return {"out": str(OUT),
            "standard": dict(Counter(r["language"] for r in standard)), "hard": dict(Counter(r["language"] for r in hard)),
            "train": len(train), "train_by_source": dict(Counter(r["source"] for r in train)),
            "train_by_language": dict(Counter(r["language"] for r in train)),
            "train_by_dialect": dict(Counter(r["dialect"] for r in train)),
            "standard_with_evidence": sum(bool(old_hint(r["messages"][1]["content"])) for r in standard),
            "hard_with_evidence": sum(bool(old_hint(r["messages"][1]["content"])) for r in hard)}


if __name__ == "__main__":
    print(json.dumps(build(), indent=1, ensure_ascii=False))
