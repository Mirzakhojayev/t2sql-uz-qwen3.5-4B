"""The final run (decided 2026-11-16): prompt v2 everywhere, a standard-difficulty eval set, and a training file of
gold SQL for direct (no-thinking) fine-tuning.

  python -m t2sql.final hard                      # data/eval/hard_prompts_v2.jsonl: the 1,100 Mini-Dev prompts, v2
  python -m t2sql.final standard --max-usd 0.5    # data/eval/standard_prompts.jsonl: 500 BIRD dev questions, 4 languages
  python -m t2sql.final translate --max-usd 2.5   # Russian and Uzbek for the BIRD train questions not translated yet
  python -m t2sql.final train                     # data/pipeline/final_train.jsonl

Prompt v2 = schema with column descriptions and sample values, plus stored values that match the question
(t2sql/describe.py). Training and eval use the same builder.

Standard set: BIRD dev is mostly simple and moderate questions, while Mini-Dev (the hard set) is 92% challenging.
The 500 questions are a seeded random sample taken before any model saw them, from the BIRD dev questions that are
not in Mini-Dev and whose gold gives the same result in PostgreSQL as in SQLite. EVAL DATA: never train on it.

Training targets are the human gold SQL of BIRD train (run in the row's dialect, non-empty), plus SynSQL gold that an
independent answer matched on enlarged databases (t2sql/rft.py). No reasoning is trained: the answer is the query.
"""
import argparse
import json
import random
import zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psycopg

from eval.backends import PostgresBackend, SQLiteBackend
from eval.config import pg_config

from . import teacher, translate
from .bird import ZERO_SHOT_SYSTEM
from .bird_train import PG_DATABASE, SIMPLE_NAME, db_dir, lowercase_quoted, same_result
from .convert import to_postgres
from .describe import describe, load_descriptions, matching_values, value_index
from .leakage import find_leaks, load_eval_sets
from .pilot import RUSSIAN_KEYBOARD_PCT, question_in
from .prompt import build_prompt, read_sqlite_schema, schema_to_ddl
from .schema_pg import read_postgres_schema
from .uz_cyrillic import russian_keyboard, to_uz_cyrillic

EVAL = Path("data/eval")
OUT = Path("data/pipeline/final")
INDEX_DIR = Path("data/pipeline/value_index")
DEV_SQLITE = Path("data/sqlite")
DEV_DESCRIPTIONS = Path("data/bird_minidev/MINIDEV/dev_databases")
BIRD_DEV = Path("data/eval_sources/bird_dev/data/dev_20251106-00000-of-00001.json")
MINI_DEV = Path("data/bird_minidev/data/mini_dev_sqlite.jsonl")
HARD_V1 = EVAL / "baseline_prompts.jsonl"
HARD_V2 = EVAL / "hard_prompts_v2.jsonl"
STANDARD = EVAL / "standard_prompts.jsonl"
STANDARD_TRANSLATIONS = EVAL / "standard_translations.jsonl"
TRAIN_QUESTIONS = OUT / "bird_questions.jsonl"       # python -m t2sql.bird_train prepare --questions 7731 --factor 1 --jobs <this>
TRAIN_TRANSLATIONS = Path("data/pipeline/rft/translations.jsonl")
TRAIN_FILE = Path("data/pipeline/final_train.jsonl")
LANGUAGES = ("en", "ru", "uz_latin", "uz_cyrillic")
STANDARD_QUESTIONS = 500
STANDARD_CYRILLIC = 100


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def pg_name(name: str) -> str:
    """The name a SQLite table or column has in the PostgreSQL copies: simple names are lower case there."""
    return name.lower() if SIMPLE_NAME.fullmatch(name) else name


class Schemas:
    """Schema text and value index per (kind, dialect, database), built once. kind: "dev" or "train"."""

    def __init__(self):
        self._ddl: dict[tuple, str] = {}
        self._index: dict[tuple, dict] = {}

    def _sqlite(self, kind: str, db: str) -> Path:
        return (DEV_SQLITE if kind == "dev" else db_dir()) / db / f"{db}.sqlite"

    def _descriptions(self, kind: str, db: str) -> dict:
        return load_descriptions((DEV_DESCRIPTIONS if kind == "dev" else db_dir()) / db / "database_description")

    def ddl(self, kind: str, dialect: str, db: str) -> str:
        key = (kind, dialect, db)
        if key not in self._ddl:
            if dialect == "sqlite":
                tables = read_sqlite_schema(self._sqlite(kind, db))
            else:
                cfg = pg_config()
                with psycopg.connect(host=cfg.host, port=cfg.port, dbname=db if kind == "dev" else PG_DATABASE,
                                     user=cfg.ro_user, password=cfg.ro_password, connect_timeout=10) as conn:
                    conn.execute("SET statement_timeout = '300s'")  # sample-value queries scan whole columns
                    if kind == "train":
                        conn.execute(f'SET search_path TO "{db}"')
                    tables = read_postgres_schema(conn)
            self._ddl[key] = schema_to_ddl(describe(tables, self._descriptions(kind, db)))
        return self._ddl[key]

    def index(self, kind: str, db: str) -> dict:
        key = (kind, db)
        if key not in self._index:
            cache = INDEX_DIR / f"{kind}_{db}.json"
            if not cache.exists():
                cache.parent.mkdir(parents=True, exist_ok=True)
                cache.write_text(json.dumps(value_index(self._sqlite(kind, db)), ensure_ascii=False), encoding="utf-8")
            self._index[key] = {k: [tuple(h) for h in v] for k, v in json.loads(cache.read_text(encoding="utf-8")).items()}
        return self._index[key]

    def prompt(self, kind: str, dialect: str, db: str, question: str, hint: str | None) -> str:
        values = matching_values(f"{question} {hint or ''}", self.index(kind, db),
                                 rename=pg_name if dialect == "postgres" else (lambda n: n))
        return build_prompt(dialect, self.ddl(kind, dialect, db), question, hint, values or None)


def messages(user: str) -> list[dict]:
    return [{"role": "system", "content": ZERO_SHOT_SYSTEM}, {"role": "user", "content": user}]


def old_hint(user: str) -> str | None:
    """The hint of a v1 prompt (it sits between the schema and the question)."""
    head = user.rsplit("\n\nQuestion: ", 1)[0]
    return head.rsplit("\n\nHint: ", 1)[1] if "\n\nHint: " in head else None


def hard() -> dict:
    """The same 1,100 questions as baseline_prompts.jsonl, with v2 prompts. Ids and gold are unchanged."""
    schemas, rows = Schemas(), []
    for r in load_jsonl(HARD_V1):
        user = schemas.prompt("dev", r["dialect"], r["db_id"], r["question"], old_hint(r["messages"][1]["content"]))
        rows.append({**r, "messages": messages(user), "prompt_version": 2})
    write_jsonl(HARD_V2, rows)
    return {"file": str(HARD_V2), "prompts": len(rows),
            "with_matching_values": sum("Values in the database" in r["messages"][1]["content"] for r in rows),
            "prompt_chars_median": sorted(len(r["messages"][1]["content"]) for r in rows)[len(rows) // 2]}


def standard_candidates() -> tuple[list[dict], Counter]:
    """BIRD dev questions outside Mini-Dev whose gold runs, is non-empty and agrees in both engines."""
    mini = {str(r["question_id"]) for r in load_jsonl(MINI_DEV)}
    sqlite, pg = SQLiteBackend(DEV_SQLITE, 30), PostgresBackend(pg_config(), 60)
    dropped = Counter()

    def check(r: dict) -> dict | None:
        if str(r["question_id"]) in mini:
            dropped["in the hard set"] += 1
            return None
        gold = sqlite.execute(r["db_id"], r["SQL"])
        if not gold.ok or not gold.rows or all(v is None for row in gold.rows for v in row):
            dropped["gold fails or is empty"] += 1
            return None
        converted = to_postgres(r["SQL"])
        converted = lowercase_quoted(converted) if converted else None
        res = pg.execute(r["db_id"], converted) if converted else None
        if res is None or not res.ok or not same_result(gold.rows, res.rows, r["SQL"], "sqlite"):
            dropped["gold differs in PostgreSQL"] += 1
            return None
        return {"question_id": str(r["question_id"]), "db_id": r["db_id"], "question": r["question"].strip(),
                "hint": (r.get("evidence") or "").strip() or None, "gold_sql": converted, "difficulty": r["difficulty"]}

    with ThreadPoolExecutor(max_workers=6) as ex:
        kept = [r for r in ex.map(check, json.load(open(BIRD_DEV, encoding="utf-8"))) if r]
    sqlite.close()
    pg.close()
    return kept, dropped


def standard(max_usd: float, seed: int = 0) -> dict:
    kept, dropped = standard_candidates()
    rng = random.Random(seed)
    picked = sorted(rng.sample(kept, min(STANDARD_QUESTIONS, len(kept))), key=lambda r: int(r["question_id"]))
    jobs = translate.jobs_for([{"question_id": f"dev{r['question_id']}", "question": r["question"], "db_id": r["db_id"],
                                "difficulty": r["difficulty"], "source": "bird_dev", "license": "CC BY-SA 4.0",
                                "split": "eval", "translate_to": ["ru", "uz"]} for r in picked])
    teacher.run(jobs, STANDARD_TRANSLATIONS, effort="high", max_usd=max_usd)
    translated = {(t["question_id"], t["language"]): t for t in load_jsonl(STANDARD_TRANSLATIONS)}
    cyrillic = set(rng.sample([r["question_id"] for r in picked], min(STANDARD_CYRILLIC, len(picked))))

    schemas, rows, missing = Schemas(), [], 0
    for r in picked:
        texts = {"en": (r["question"], "typed", [])}
        for lang, key in (("ru", "ru"), ("uz_latin", "uz")):
            t = translated.get((f"dev{r['question_id']}", key))
            if t is None or not (t["content"] or "").strip():
                missing += 1
                continue
            texts[lang] = (t["content"].strip(), "typed", translate.checks(t))  # flagged ones are kept, as in the hard set
        if "uz_latin" in texts and r["question_id"] in cyrillic:
            cyr = to_uz_cyrillic(texts["uz_latin"][0], r["question"])
            keyboard = zlib.crc32(r["question_id"].encode()) % 100 < RUSSIAN_KEYBOARD_PCT
            texts["uz_cyrillic"] = (russian_keyboard(cyr) if keyboard else cyr,
                                    "russian_keyboard" if keyboard else "typed", texts["uz_latin"][2])
        for lang, (text, typing, flags) in texts.items():
            rows.append({
                "id": f"postgres-dev{r['question_id']}-{lang}", "question_id": f"dev{r['question_id']}", "language": lang,
                "typing": typing, "dialect": "postgres", "slice": r["difficulty"], "db_id": r["db_id"],
                "gold_sql": r["gold_sql"], "source": "bird_dev", "license": "CC BY-SA 4.0", "split": "eval",
                "question": text, "question_en": r["question"], "verified": False, "flags": flags, "prompt_version": 2,
                "messages": messages(schemas.prompt("dev", "postgres", r["db_id"], text, r["hint"])),
            })
    write_jsonl(STANDARD, rows)
    return {"file": str(STANDARD), "bird_dev_usable": len(kept), "dropped": dict(dropped), "questions": len(picked),
            "prompts": len(rows), "by_language": dict(Counter(r["language"] for r in rows)),
            "by_difficulty": dict(Counter(r["slice"] for r in rows if r["language"] == "en")), "translations_missing": missing}


def translate_train(max_usd: float) -> dict:
    questions = [{"question_id": q["id"], "question": q["question"], "db_id": q["source_db"], "difficulty": "BIRD",
                  "source": q["source"], "license": q["license"], "split": "train", "translate_to": ["ru", "uz"]}
                 for q in load_jsonl(TRAIN_QUESTIONS)]
    return teacher.run(translate.jobs_for(questions), TRAIN_TRANSLATIONS, effort="high", max_usd=max_usd)


def answer(sql: str) -> str:
    return f"```sql\n{sql.strip().rstrip(';').strip()}\n```"


def train(seed: int = 0) -> dict:
    schemas, sets = Schemas(), load_eval_sets()
    translated = {(t["question_id"], t["language"]): t for t in load_jsonl(TRAIN_TRANSLATIONS)}
    examples, dropped = [], Counter()

    for q in load_jsonl(TRAIN_QUESTIONS):
        for lang in LANGUAGES:
            text, typing = question_in(q, lang, translated)
            if text is None:
                dropped[typing] += 1
                continue
            examples.append({
                "id": f"{q['id']}-{lang}", "question_id": q["id"], "language": lang, "typing": typing, "split": "train",
                "source": q["source"], "license": q["license"], "dialect": q["dialect"], "schema_style": "plain",
                "db_id": q["db_id"], "source_db": q["source_db"], "question": text, "question_en": q["question"],
                "hint": q["hint"], "gold_sql": q["gold_sql"], "target_sql": q["gold_sql"], "trace_source": "gold",
                "prompt_version": 2, "reasoning": "", "answer": answer(q["gold_sql"]),
                "messages": messages(schemas.prompt("train", q["dialect"], q["source_db"], text, q["hint"])),
            })

    # SynSQL: gold that an independent answer matched on the original and the enlarged databases (t2sql/rft.py)
    rft = Path("data/pipeline/rft")
    verified = {e["question_id"] for e in load_jsonl(Path("data/pipeline/rft_train.jsonl")) if e["source"] != "bird_train"}
    synsql = {q["id"]: q for q in load_jsonl(rft / "synsql_questions.jsonl")}
    for p in load_jsonl(rft / "prompts_part3.jsonl"):
        q = synsql.get(p["question_id"])
        if q is None or q["id"] not in verified:
            continue
        examples.append({
            "id": p["id"], "question_id": q["id"], "language": p["language"], "typing": p["typing"], "split": "train",
            "source": q["source"], "license": q["license"], "dialect": q["dialect"], "schema_style": q["schema_style"],
            "db_id": q["db_id"], "source_db": q["source_db"],
            "question": p["messages"][1]["content"].rsplit("Question: ", 1)[1], "question_en": q["question"],
            "hint": q["hint"], "gold_sql": q["gold_sql"], "target_sql": q["gold_sql"], "trace_source": "gold",
            "prompt_version": 1, "reasoning": "", "answer": answer(q["gold_sql"]), "messages": p["messages"],
        })

    clean = []
    for ex in examples:
        if find_leaks(ex, sets):
            dropped["leak"] += 1
        else:
            clean.append(ex)
    random.Random(seed).shuffle(clean)
    write_jsonl(TRAIN_FILE, clean)
    return {"file": str(TRAIN_FILE), "examples": len(clean), "questions": len({e["question_id"] for e in clean}),
            "dropped": dict(dropped), **{f"by_{k}": dict(Counter(e[k] for e in clean)) for k in ("source", "language", "dialect")},
            "prompt_chars_median": sorted(len(e["messages"][1]["content"]) for e in clean)[len(clean) // 2]}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("hard")
    sub.add_parser("train")
    for name in ("standard", "translate"):
        sub.add_parser(name).add_argument("--max-usd", type=float, required=True)
    args = ap.parse_args(argv)
    result = {"hard": hard, "train": train}[args.cmd]() if args.cmd in ("hard", "train") else (
        standard(args.max_usd) if args.cmd == "standard" else translate_train(args.max_usd))
    print(json.dumps(result, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
