"""Leakage check: nothing from an eval set may reach training.

Compares questions and SQL, after normalising case, spacing and quotes, against every set in data/eval_sources and
BIRD Mini-Dev. Also reports training databases that share a name with an eval database.
"""
import ast
import json
import re
from dataclasses import dataclass, field
from pathlib import Path

SOURCES = Path("data/eval_sources")
MINI_DEV = Path("data/bird_minidev/data")


@dataclass
class EvalSet:
    questions: set[str] = field(default_factory=set)
    sqls: set[str] = field(default_factory=set)
    databases: set[str] = field(default_factory=set)

    def add(self, question: str | None, sql: str | None = None, db: str | None = None) -> None:
        if question:
            self.questions.add(norm_question(question))
        if sql:
            self.sqls.add(norm_sql(sql))
        if db:
            self.databases.add(db.lower())


def norm_question(text: str) -> str:
    return " ".join(text.lower().split()).rstrip(" ?.!")


def norm_sql(sql: str) -> str:
    return " ".join(re.sub(r"[\"`\[\]]", "", sql).lower().split()).rstrip(" ;")


def _read_json(path: Path) -> list[dict]:
    return json.load(open(path, encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def load_eval_sets(root: Path = SOURCES, mini_dev: Path = MINI_DEV) -> dict[str, EvalSet]:
    sets = {name: EvalSet() for name in ("bird_minidev", "bird_dev", "spider_dev", "pauq_test", "redsql", "bird_critic")}

    for path in mini_dev.glob("mini_dev_*.jsonl"):
        for r in _read_jsonl(path):
            sets["bird_minidev"].add(r["question"], r["SQL"], r["db_id"])

    for r in _read_json(root / "bird_dev/data/dev_20251106-00000-of-00001.json"):
        sets["bird_dev"].add(r["question"], r["SQL"], r["db_id"])

    import pyarrow.parquet as pq  # only this reader needs it
    for r in pq.read_table(root / "spider/spider/validation-00000-of-00001.parquet").to_pylist():
        sets["spider_dev"].add(r["question"], r["query"], r["db_id"])

    for path in (root / "pauq/splits").glob("*_test.json"):
        for r in _read_json(path):
            sets["pauq_test"].add(r["question"], r["query"], r["db_id"])

    for path in (root / "redsql/responses").glob("*.json"):
        for rows in json.load(open(path, encoding="utf-8")).values():
            for r in rows:
                sets["redsql"].add(r["question"], r.get("sql_query"))

    # BIRD-CRITIC's "query" is the user's description of the problem; its SQL is a list of broken attempts
    for r in _read_jsonl(root / "bird_critic_open/data/open-00000-of-000001.jsonl"):
        sets["bird_critic"].add(r["query"], db=r["db_id"])
        attempts = r["issue_sql"] if isinstance(r["issue_sql"], list) else ast.literal_eval(r["issue_sql"])
        for sql in attempts:
            sets["bird_critic"].add(None, sql)
    return sets


def find_leaks(example: dict, sets: dict[str, EvalSet]) -> list[str]:
    """What this training example shares with an eval set, as 'set:kind' labels. Empty = clean."""
    questions = {norm_question(example["question"]), norm_question(example["question_en"])}
    sqls = {norm_sql(s) for s in (example["gold_sql"], example["target_sql"]) if s}
    leaks = []
    for name, s in sets.items():
        if questions & s.questions:
            leaks.append(f"{name}:question")
        if sqls & s.sqls:
            leaks.append(f"{name}:sql")
    return leaks
