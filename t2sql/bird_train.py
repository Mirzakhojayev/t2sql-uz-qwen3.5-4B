"""BIRD train as training data: real databases with real amounts of data, real hints, English names.

  python -m t2sql.bird_train unzip               # data/bird_train/train.zip -> train.json + train_databases/
  python -m t2sql.bird_train prepare --questions 500
  python -m t2sql.bird_train call --max-usd 1.6  # teacher answers (resumes, never pays twice)
  python -m t2sql.bird_train select              # keep answers that match BIRD's gold, pick the set, languages
  python -m t2sql.bird_train translate --max-usd 1
  python -m t2sql.bird_train build               # data/pipeline/bird_train_500.jsonl

Rules (docs/data-rules.md): every target must match BIRD's gold on the full database in its own dialect (rule 2);
no meta talk (rule 7); same 65/35 PostgreSQL/SQLite mix, two languages per question and teacher prompt as the clean
pilot, so the two can be compared at the same size (rule 9). BIRD train shares no database with BIRD dev or Mini-Dev;
the leakage check still runs on every question.

PostgreSQL copies live in the database `bird_train`, one schema per BIRD database (db_id "bird_train/<db>").
Databases larger than MAX_DB_MB are skipped: the loader holds a whole table in memory.
"""
import argparse
import json
import random
import re
import sqlite3
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psycopg

from eval.backends import PostgresBackend, SQLiteBackend
from eval.compare import is_ordered, normalize_rows, signature
from eval.config import pg_config
from eval.harness import Evaluator

from . import teacher, translate
from .convert import load_database, to_postgres
from .leakage import find_leaks, load_eval_sets
from .pilot import META_TALK, PG_SHARE, assign_languages, drop_repeats, make_examples
from .prompt import build_prompt, extract_sql, read_sqlite_schema, schema_to_ddl
from .schema_pg import read_postgres_schema
from .teacher_test import TEACHER_SYSTEM

ROOT = Path("data/bird_train")
OUT = Path("data/pipeline/bird_train")
JOBS, ANSWERS, PLAN = OUT / "teacher_jobs.jsonl", OUT / "teacher_answers.jsonl", OUT / "plan.jsonl"
TRANSLATIONS = OUT / "translations.jsonl"
PG_DATABASE = "bird_train"
MAX_DB_MB = 400
# The teacher matches BIRD's gold on about 70% of BIRD dev; ask for enough to fill the set at that rate.
CANDIDATES_PER_KEPT = 1.6
LICENSE = "CC BY-SA 4.0"
SIMPLE_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
QUOTED_SIMPLE = re.compile(r'"([A-Za-z_][A-Za-z0-9_]*)"')


def lowercase_names(sqlite_path: Path) -> dict[str, str]:
    """Rename map for the PostgreSQL copy, as BIRD's own PostgreSQL release does it: simple names (groupName, RA)
    become lower case, so the unquoted names in BIRD's SQL still find them; names that need quotes anyway
    ("Customer ID") keep their case. Kept in mixed case, 64% of converted gold queries failed on unknown columns."""
    conn = sqlite3.connect(f"{sqlite_path.resolve().as_uri()}?mode=ro", uri=True)
    try:
        names = [t for (t,) in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")]
        names += [c[1] for t in list(names) for c in conn.execute(f'PRAGMA table_info("{t}")')]
    finally:
        conn.close()
    return {n.lower(): n.lower() for n in names if SIMPLE_NAME.fullmatch(n)}


def lowercase_quoted(sql: str) -> str:
    """BIRD quotes some simple names (`Year`), which become "Year" in PostgreSQL: match the lower-case copy."""
    return QUOTED_SIMPLE.sub(lambda m: f'"{m.group(1).lower()}"', sql)


def db_dir() -> Path:
    return ROOT / "train" / "train_databases"


def round_robin(rows: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Up to n rows, spread across databases as evenly as supply allows, so no database dominates."""
    by_db: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_db[r["db_id"]].append(r)
    for pool in by_db.values():
        rng.shuffle(pool)
    order = sorted(by_db)
    rng.shuffle(order)
    picked: list[dict] = []
    while len(picked) < n and any(by_db.values()):
        for db in order:
            if by_db[db] and len(picked) < n:
                picked.append(by_db[db].pop())
    return picked


def same_result(a: list[tuple], b: list[tuple], gold_sql: str, dialect: str) -> bool:
    ordered = is_ordered(gold_sql, dialect)
    return signature(normalize_rows(a), ordered) == signature(normalize_rows(b), ordered)


def unzip() -> dict:
    """train.zip holds train.json and a second zip with the databases. Paths are checked before writing."""
    def extract(zpath: Path, dest: Path) -> int:
        with zipfile.ZipFile(zpath) as z:
            for name in z.namelist():
                if not (dest / name).resolve().is_relative_to(dest.resolve()):
                    raise ValueError(f"unsafe path in {zpath}: {name}")
            z.extractall(dest)
            return len(z.namelist())

    n = extract(ROOT / "train.zip", ROOT)
    inner = next((ROOT / "train").glob("train_databases.zip"), None)
    m = extract(inner, ROOT / "train") if inner and not db_dir().exists() else 0
    return {"files": n, "database_files": m, "databases": len(list(db_dir().glob("*/*.sqlite")))}


def load_postgres(db_ids: list[str]) -> dict:
    cfg = pg_config()
    admin = dict(host=cfg.host, port=cfg.port, user=cfg.admin_user, password=cfg.admin_password, connect_timeout=10)
    with psycopg.connect(dbname="postgres", autocommit=True, **admin) as c:
        if not c.execute("SELECT 1 FROM pg_database WHERE datname = %s", (PG_DATABASE,)).fetchone():
            c.execute(f"CREATE DATABASE {PG_DATABASE}")
    with psycopg.connect(dbname=PG_DATABASE, autocommit=True, **admin) as c:
        existing = {r[0] for r in c.execute("SELECT schema_name FROM information_schema.schemata")}
    failed: dict[str, str] = {}

    def load(db: str) -> None:
        try:
            with psycopg.connect(dbname=PG_DATABASE, autocommit=True, **admin) as conn:
                path = db_dir() / db / f"{db}.sqlite"
                load_database(conn, path, db, rename=lowercase_names(path))
        except Exception as exc:  # one odd database must not stop the rest; it is reported
            failed[db] = f"{type(exc).__name__}: {str(exc)[:200]}"

    todo = [d for d in db_ids if d not in existing]
    with ThreadPoolExecutor(max_workers=4) as ex:
        list(ex.map(load, todo))
    return {"loaded_now": len(todo) - len(failed), "already_loaded": len(db_ids) - len(todo), "failed": failed}


def postgres_ddl(db: str) -> str:
    cfg = pg_config()
    with psycopg.connect(host=cfg.host, port=cfg.port, dbname=PG_DATABASE, user=cfg.ro_user, password=cfg.ro_password,
                         connect_timeout=10) as conn:
        conn.execute("SET statement_timeout = '300s'")  # sample-value queries scan whole columns
        conn.execute(f'SET search_path TO "{db}"')
        return schema_to_ddl(read_postgres_schema(conn))


def prepare(questions: int, seed: int = 0) -> dict:
    rng = random.Random(seed)
    rows = json.load(open(ROOT / "train" / "train.json", encoding="utf-8"))
    sizes = {p.parent.name: p.stat().st_size / 1e6 for p in db_dir().glob("*/*.sqlite")}
    small = sorted(d for d, mb in sizes.items() if mb <= MAX_DB_MB)
    sets = load_eval_sets()
    candidates, leaks = [], Counter()
    for i, r in enumerate(rows):
        if r["db_id"] not in small:
            continue
        probe = {"question": r["question"], "question_en": r["question"], "gold_sql": r["SQL"], "target_sql": None}
        found = find_leaks(probe, sets)
        if found:
            leaks.update(found)
            continue
        candidates.append({"id": f"birdtrain-{i}", "db_id": r["db_id"], "question": r["question"].strip(),
                           "hint": (r.get("evidence") or "").strip() or None, "gold_sqlite": r["SQL"]})
    candidates = drop_repeats(candidates, set())

    n_pg = round(questions * PG_SHARE)
    pg_pick = round_robin(candidates, round(n_pg * CANDIDATES_PER_KEPT), rng)
    taken = {r["id"] for r in pg_pick}
    lite_pick = round_robin([r for r in candidates if r["id"] not in taken],
                            round((questions - n_pg) * CANDIDATES_PER_KEPT), rng)
    used_dbs = sorted({r["db_id"] for r in pg_pick + lite_pick})
    loaded = load_postgres(sorted({r["db_id"] for r in pg_pick}))

    # gold must run, be non-empty, and (for PostgreSQL) give the same result after conversion
    sqlite, pg = SQLiteBackend(db_dir(), 30), PostgresBackend(pg_config(), 60)
    dropped = Counter()

    def check(r: dict, dialect: str) -> dict | None:
        gold = sqlite.execute(r["db_id"], r["gold_sqlite"])
        if not gold.ok or not gold.rows or all(v is None for row in gold.rows for v in row):
            dropped["gold fails or is empty"] += 1
            return None
        if dialect == "sqlite":
            return {**r, "dialect": "sqlite", "gold_sql": r["gold_sqlite"], "run_db": r["db_id"]}
        if r["db_id"] in loaded["failed"]:
            dropped["database did not load"] += 1
            return None
        converted = to_postgres(r["gold_sqlite"])
        converted = lowercase_quoted(converted) if converted else None
        res =pg.execute(f"{PG_DATABASE}/{r['db_id']}", converted) if converted else None
        if res is None or not res.ok or not same_result(gold.rows, res.rows, r["gold_sqlite"], "sqlite"):
            dropped["gold differs in PostgreSQL"] += 1
            return None
        return {**r, "dialect": "postgres", "gold_sql": converted, "run_db": f"{PG_DATABASE}/{r['db_id']}"}

    with ThreadPoolExecutor(max_workers=6) as ex:
        checked = list(ex.map(lambda rd: check(*rd), [(r, "postgres") for r in pg_pick] + [(r, "sqlite") for r in lite_pick]))
    sqlite.close()
    pg.close()

    ddl: dict[tuple[str, str], str] = {}
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [r for r in checked if r]
    with open(JOBS, "w", encoding="utf-8") as fh:
        for r in jobs:
            key = (r["dialect"], r["db_id"])
            if key not in ddl:
                ddl[key] = (postgres_ddl(r["db_id"]) if r["dialect"] == "postgres"
                            else schema_to_ddl(read_sqlite_schema(db_dir() / r["db_id"] / f"{r['db_id']}.sqlite")))
            fh.write(json.dumps({
                "id": r["id"], "level": "BIRD", "style": "bird", "question": r["question"], "hint": r["hint"],
                "source": "bird_train", "license": LICENSE, "schema_style": "plain", "dialect": r["dialect"],
                "db_id": r["run_db"], "source_db": r["db_id"], "gold_sql": r["gold_sql"],
                "messages": [{"role": "system", "content": TEACHER_SYSTEM},
                             {"role": "user", "content": build_prompt(r["dialect"], ddl[key], r["question"], r["hint"])}],
            }, ensure_ascii=False) + "\n")
    return {"train_questions": len(rows), "databases": len(sizes), "databases_used": len(used_dbs),
            "skipped_large_databases": sorted(d for d, mb in sizes.items() if mb > MAX_DB_MB),
            "leaks": dict(leaks), "candidates": len(candidates), "jobs": len(jobs),
            "jobs_by_dialect": dict(Counter(r["dialect"] for r in jobs)), "dropped": dict(dropped),
            "postgres_load": loaded, "with_hint_pct": round(100 * sum(bool(r["hint"]) for r in jobs) / max(1, len(jobs)), 1)}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8")]


def select(questions: int, seed: int = 0) -> dict:
    rng = random.Random(seed)
    jobs = {j["id"]: j for j in load_jsonl(JOBS)}
    answers = load_jsonl(ANSWERS)
    # Answers from an earlier, smaller job list are paid for too. Each answer carries its job's fields except the
    # prompt, which is rebuilt the way prepare() builds it.
    ddl: dict[tuple[str, str], str] = {}
    for a in answers:
        if a["id"] not in jobs:
            key = (a["dialect"], a["source_db"])
            if key not in ddl:
                ddl[key] = (postgres_ddl(a["source_db"]) if a["dialect"] == "postgres" else
                            schema_to_ddl(read_sqlite_schema(db_dir() / a["source_db"] / f"{a['source_db']}.sqlite")))
            prompt = build_prompt(a["dialect"], ddl[key], a["question"], a["hint"])
            jobs[a["id"]] = {**a, "messages": [{"role": "system", "content": TEACHER_SYSTEM},
                                               {"role": "user", "content": prompt}]}
    records = [{"id": a["id"], "language": "en", "dialect": jobs[a["id"]]["dialect"], "slice": "BIRD",
                "db_id": jobs[a["id"]]["db_id"], "gold_sql": jobs[a["id"]]["gold_sql"],
                "pred_sql": extract_sql(a.get("content") or "") or ""} for a in answers]
    ev = Evaluator({"sqlite": SQLiteBackend(db_dir(), 30), "postgres": PostgresBackend(pg_config(), 60)})
    ev.prefetch(records, workers=6)
    scores = {s.id: s for s in (ev.score(r) for r in records)}
    why = Counter()
    ok = []
    for a in answers:
        s = scores[a["id"]]
        if a["finish_reason"] != "stop":
            why["cut off"] += 1
        elif not s.acc:
            why["does not match gold"] += 1
        elif META_TALK.search(a.get("reasoning") or ""):
            why["meta talk"] += 1
        else:
            job = jobs[a["id"]]
            ok.append({**{k: job[k] for k in ("id", "level", "style", "question", "hint", "source", "license",
                                              "schema_style", "dialect", "db_id", "source_db", "gold_sql")},
                       "prompt": job["messages"][1]["content"], "reasoning": a["reasoning"], "answer": a["content"],
                       "target_sql": extract_sql(a["content"]), "trace_source": "deepseek-flash"})
    n_pg = round(questions * PG_SHARE)
    plan = []
    for dialect, n in (("postgres", n_pg), ("sqlite", questions - n_pg)):
        group = round_robin([r for r in ok if r["dialect"] == dialect], n, rng)
        assign_languages(group, rng)
        plan += group
    rng.shuffle(plan)
    with open(PLAN, "w", encoding="utf-8") as fh:
        for r in plan:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    tried = Counter(jobs[a["id"]]["dialect"] for a in answers)
    kept = Counter(r["dialect"] for r in ok)
    return {"answers": len(answers), "accepted": len(ok), "rejected": dict(why),
            "accept_pct_by_dialect": {d: round(100 * kept[d] / tried[d], 1) for d in tried},
            "picked": dict(Counter(r["dialect"] for r in plan)),
            "cost_usd": round(sum(a.get("cost_usd", 0) for a in answers), 3)}


def translation_jobs() -> list[dict]:
    questions = []
    for r in load_jsonl(PLAN):
        to = [lang for lang, needs in (("ru", {"ru"}), ("uz", {"uz_latin", "uz_cyrillic"})) if needs & set(r["languages"])]
        questions.append({"question_id": r["id"], "question": r["question"], "db_id": r["source_db"],
                          "difficulty": r["level"], "source": r["source"], "license": r["license"], "split": "train",
                          "translate_to": to})
    return translate.jobs_for(questions)


def build(questions: int) -> dict:
    translated = {(r["question_id"], r["language"]): r for r in load_jsonl(TRANSLATIONS)}
    sets = load_eval_sets()
    examples, dropped = [], Counter()
    for row in load_jsonl(PLAN):
        made, why = make_examples(row, translated)
        dropped += why
        for ex in made:
            if find_leaks(ex, sets):
                dropped["leak"] += 1
            else:
                examples.append(ex)
    out = Path("data/pipeline") / f"bird_train_{questions}.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    return {"file": str(out), "examples": len(examples), "questions": len({e["question_id"] for e in examples}),
            "dropped": dict(dropped), **{f"by_{k}": dict(Counter(e[k] for e in examples)) for k in ("language", "dialect", "typing")}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("unzip")
    for name in ("prepare", "select", "build"):
        sub.add_parser(name).add_argument("--questions", type=int, default=500)
    for name in ("call", "translate"):
        sub.add_parser(name).add_argument("--max-usd", type=float, required=True)
    args = ap.parse_args(argv)
    if args.cmd == "unzip":
        result = unzip()
    elif args.cmd == "prepare":
        result = prepare(args.questions)
    elif args.cmd == "call":
        result = teacher.run(load_jsonl(JOBS), ANSWERS, effort="high", max_usd=args.max_usd)
    elif args.cmd == "select":
        result = select(args.questions)
    elif args.cmd == "translate":
        result = teacher.run(translation_jobs(), TRANSLATIONS, effort="high", max_usd=args.max_usd)
    else:
        result = build(args.questions)
    print(json.dumps(result, indent=1, ensure_ascii=False, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
