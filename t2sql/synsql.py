"""SynSQL-2.5M helpers (Apache 2.0, SQLite, English). data.json is one 9.4 GB JSON array, so it is streamed.

  python -m t2sql.synsql sample --n 500 --out data/runs/teacher_test/questions.jsonl
  python -m t2sql.synsql candidates --per-level 10000     # pipeline step 1 -> data/pipeline/candidates.jsonl

Each record: db_id, sql_complexity (Simple / Moderate / Complex / Highly Complex), question_style, question,
external_knowledge, cot (SynSQL's own reasoning trace), sql (SQLite gold).
Databases live in data/synsql/databases/<db_id>/<db_id>.sqlite.
"""
import argparse
import json
import random
import re
import sqlite3
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import ijson
import sqlglot
from sqlglot import exp

DATA = Path("data/synsql/data.json")
DB_DIR = Path("data/synsql/databases")
LEVELS = ["Simple", "Moderate", "Complex", "Highly Complex"]


def stream(path: Path = DATA):
    """Yield (index, record) for every sample without loading the whole file."""
    with open(path, "rb") as fh:
        yield from enumerate(ijson.items(fh, "item"))


def sample(n: int, seed: int = 0, pool_size: int = 3000) -> tuple[list[dict], dict]:
    """n questions spread evenly over complexity levels, each from a different database.

    One pass, reservoir sampling per level (uniform over the whole file, not biased to its start),
    then pick from the pools without repeating a database.
    """
    rng = random.Random(seed)
    pools: dict[str, list] = {lvl: [] for lvl in LEVELS}
    seen: Counter = Counter()
    styles: Counter = Counter()
    started = time.time()
    for i, rec in stream():
        lvl = rec["sql_complexity"]
        seen[lvl] += 1
        styles[rec["question_style"]] += 1
        pool = pools.setdefault(lvl, [])
        if len(pool) < pool_size:
            pool.append((i, rec))
        else:
            j = rng.randrange(seen[lvl])
            if j < pool_size:
                pool[j] = (i, rec)
        if i and i % 250_000 == 0:
            print(f"  read {i:,} records ({time.time() - started:.0f}s)", flush=True)

    per_level = n // len(LEVELS)
    used_dbs, picked = set(), []
    for lvl in LEVELS:
        candidates = pools.get(lvl, [])
        rng.shuffle(candidates)
        taken = 0
        for i, rec in candidates:
            if taken == per_level:
                break
            if rec["db_id"] in used_dbs or not (DB_DIR / rec["db_id"] / f"{rec['db_id']}.sqlite").exists():
                continue
            used_dbs.add(rec["db_id"])
            picked.append({"id": f"synsql-{i}", "synsql_index": i, **rec,
                           "source": "synsql_2.5m", "license": "Apache-2.0"})
            taken += 1
    stats = {"records": sum(seen.values()), "by_complexity": dict(seen), "by_style": dict(styles),
             "seconds": round(time.time() - started)}
    return picked, stats


def gold_status(db_id: str, sql: str, timeout_s: float = 5.0) -> tuple[str, int, int]:
    """Run gold SQL on its SQLite database. Returns (status, rows, columns); status is one of
    ok, empty, all_null, error, timeout. Rows are capped at 1000: we only need to know there are some."""
    path = (DB_DIR / db_id / f"{db_id}.sqlite").resolve()
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        cur = conn.execute(sql)
        rows = cur.fetchmany(1000)
        n_cols = len(cur.description or [])
    except sqlite3.Error:
        return ("timeout" if time.monotonic() > deadline else "error"), 0, 0
    finally:
        conn.close()
    if not rows:
        return "empty", 0, n_cols
    if all(v is None for row in rows for v in row):
        return "all_null", len(rows), n_cols
    return "ok", len(rows), n_cols


def sql_template(sql: str) -> str:
    """The shape of a query: literals and names removed, so near-copies with other values look identical."""
    try:
        tree = sqlglot.parse_one(sql, read="sqlite")
        for lit in tree.find_all(exp.Literal):
            lit.set("this", "0")
            lit.set("is_string", False)
        for ident in tree.find_all(exp.Identifier):
            ident.set("this", "x")
            ident.set("quoted", False)
        return tree.sql(dialect="sqlite")
    except Exception:
        return " ".join(re.sub(r"'[^']*'|\b\d+(\.\d+)?\b", "0", sql).split()).lower()


def candidates(per_level: int, seed: int = 0, max_per_db: int = 5, max_per_template: int = 5,
               workers: int = 8) -> tuple[list[dict], dict]:
    """Step 1 of the pipeline: filtered, spread-out questions.

    Multi-turn transcripts are skipped (v2 scope). Gold SQL must return real rows: empty and all-NULL results
    prove little on these tiny databases. At most max_per_db questions per database and max_per_template per
    SQL shape, so the data does not repeat itself. Levels are sampled evenly.
    """
    rng = random.Random(seed)
    pool_size = per_level * 3  # about half of the golds return nothing, and the caps remove more
    pools: dict[str, list] = {lvl: [] for lvl in LEVELS}
    seen: Counter = Counter()
    started = time.time()
    skipped_multi_turn = 0
    for i, rec in stream():
        if rec["question_style"] == "Multi-turn Dialogue":
            skipped_multi_turn += 1
            continue
        lvl = rec["sql_complexity"]
        seen[lvl] += 1
        pool = pools[lvl]
        if len(pool) < pool_size:
            pool.append((i, rec))
        else:
            j = rng.randrange(seen[lvl])
            if j < pool_size:
                pool[j] = (i, rec)
    print(f"read the file in {time.time() - started:.0f}s; pool {pool_size:,} per level", flush=True)

    per_db: Counter = Counter()
    per_template: Counter = Counter()
    picked, outcome = [], {lvl: Counter() for lvl in LEVELS}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for lvl in LEVELS:
            pool = pools[lvl]
            rng.shuffle(pool)
            results = ex.map(lambda item: gold_status(item[1]["db_id"], item[1]["sql"]), pool)
            taken = 0
            for (i, rec), (status, n_rows, n_cols) in zip(pool, results):
                if taken == per_level:
                    outcome[lvl]["not_needed"] += 1
                    continue
                if status != "ok":
                    outcome[lvl][status] += 1
                    continue
                if per_db[rec["db_id"]] >= max_per_db:
                    outcome[lvl]["database_cap"] += 1
                    continue
                template = sql_template(rec["sql"])
                if per_template[template] >= max_per_template:
                    outcome[lvl]["template_cap"] += 1
                    continue
                per_db[rec["db_id"]] += 1
                per_template[template] += 1
                outcome[lvl]["kept"] += 1
                taken += 1
                picked.append({"id": f"synsql-{i}", "synsql_index": i, **rec, "gold_rows": n_rows, "gold_cols": n_cols,
                               "source": "synsql_2.5m", "license": "Apache-2.0"})
            print(f"  {lvl}: {dict(outcome[lvl])} ({time.time() - started:.0f}s)", flush=True)
    stats = {"per_level_target": per_level, "seed": seed, "max_per_db": max_per_db,
             "max_per_template": max_per_template, "skipped_multi_turn": skipped_multi_turn,
             "kept": len(picked), "databases": len(per_db), "templates": len(per_template),
             "outcome_by_level": {k: dict(v) for k, v in outcome.items()}, "seconds": round(time.time() - started)}
    return picked, stats


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=500)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("candidates", help="pipeline step 1: filtered questions")
    c.add_argument("--per-level", type=int, default=10_000)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--out", type=Path, default=Path("data/pipeline/candidates.jsonl"))
    args = ap.parse_args(argv)

    if args.cmd == "candidates":
        picked, stats = candidates(args.per_level, args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            for rec in picked:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        args.out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
        print(json.dumps(stats, indent=1))
        print(f"wrote {len(picked)} candidates to {args.out}")
        return 0

    if args.cmd == "sample":
        picked, stats = sample(args.n, args.seed)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            for rec in picked:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        args.out.with_suffix(".stats.json").write_text(json.dumps(stats, indent=1), encoding="utf-8")
        print(json.dumps(stats, indent=1))
        print(f"wrote {len(picked)} questions to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
