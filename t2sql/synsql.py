"""SynSQL-2.5M helpers (Apache 2.0, SQLite, English). data.json is one 9.4 GB JSON array, so it is streamed.

  python -m t2sql.synsql sample --n 500 --out data/runs/teacher_test/questions.jsonl

Each record: db_id, sql_complexity (Simple / Moderate / Complex / Highly Complex), question_style, question,
external_knowledge, cot (SynSQL's own reasoning trace), sql (SQLite gold).
Databases live in data/synsql/databases/<db_id>/<db_id>.sqlite.
"""
import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import ijson

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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=500)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

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
