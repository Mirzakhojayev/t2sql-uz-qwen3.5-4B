"""Does a matching result really mean two queries agree? Re-run gold and target on bigger copies of the database.

  python -m t2sql.stress data/pipeline/pilot_10k.jsonl --out data/runs/stress.jsonl --limit 1000

The SynSQL databases hold about 20 rows, so a wrong query often returns the right answer by luck: in the audit of
2026-11-05 the original data caught 19% of small planted mistakes, and data with added rows caught 54%. A difference
does not say who is wrong (by hand, gold was the worse one in 10 of 24 cases and the teacher in none), so use this to
find examples worth reading or re-checking, not as a verdict.

PostgreSQL rows only. Temp tables shadow the real ones inside a transaction that is always rolled back, so nothing
changes in the database. Needs the admin role, because the read-only role cannot create tables.
"""
import argparse
import decimal
import json
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psycopg
from psycopg import sql as pgsql

from eval.compare import is_ordered, normalize_rows, signature
from eval.config import pg_config

SEEDS = (11, 22, 33)
NEW_ROWS = 25  # per table
TIMEOUT_MS = 8000
DATE_LIKE = re.compile(r"\d{4}-\d{2}|\d{1,2}[/:]\d{1,2}")
WRITES = re.compile(r"\b(insert\s+into|update\s+\S+\s+set|delete\s+from|drop\s|alter\s|truncate\s|grant\s)", re.IGNORECASE)
MISTAKES = ((r"\s>=\s", " > "), (r"\s>\s", " >= "), (r"\s<=\s", " < "), (r"\s<\s", " <= "), (r"\bLEFT JOIN\b", "JOIN"),
            (r"\bDESC\b", "ASC"), (r"\bCOUNT\(DISTINCT\s", "COUNT("))


def is_read_only(sql: str) -> bool:
    return bool(re.match(r"\s*(select|with)\b", sql, re.IGNORECASE)) and not WRITES.search(sql)


def planted_mistakes(sql: str) -> list[str]:
    """The query with one small realistic mistake each, to measure what a database can detect."""
    out = []
    for pattern, wrong in MISTAKES:
        changed = re.sub(pattern, wrong, sql, count=1, flags=re.IGNORECASE)
        if changed != sql:
            out.append(changed)
    return out


def new_value(values: list, n: int, rng: random.Random):
    """A new value for a plain column, drawn from what the column already holds."""
    if not values:
        return None
    if isinstance(values[0], str):
        # A column of all-distinct text is a name or an id: repeating its values would invent duplicates that real
        # data would not have, and split GROUP BY name from GROUP BY id for no good reason.
        if len(values) >= 2 and len(set(values)) == len(values) and not DATE_LIKE.search(values[0]):
            return f"{rng.choice(values)} {n}"
        return rng.choice(values)
    value = rng.choice(values)
    if isinstance(value, bool) or rng.random() < .5:
        return value
    if isinstance(value, int):
        return value + rng.randint(-5, 25)
    if isinstance(value, float):
        return round(value * rng.uniform(.5, 1.6), 2)
    if isinstance(value, decimal.Decimal):
        return value * decimal.Decimal(str(round(rng.uniform(.5, 1.6), 2)))
    return value


def key_columns(cur, schema: str) -> tuple[dict, dict]:
    cur.execute("""select t.relname, a.attname from pg_constraint c
                   join pg_class t on t.oid=c.conrelid join pg_namespace n on n.oid=c.connamespace
                   join pg_attribute a on a.attrelid=c.conrelid and a.attnum=any(c.conkey)
                   where c.contype='p' and n.nspname=%s""", (schema,))
    primary = defaultdict(set)
    for table, column in cur.fetchall():
        primary[table].add(column)
    cur.execute("""select t.relname, a.attname, p.relname, f.attname from pg_constraint c
                   join pg_class t on t.oid=c.conrelid join pg_class p on p.oid=c.confrelid
                   join pg_namespace n on n.oid=c.connamespace
                   join pg_attribute a on a.attrelid=c.conrelid and a.attnum=c.conkey[1]
                   join pg_attribute f on f.attrelid=c.confrelid and f.attnum=c.confkey[1]
                   where c.contype='f' and cardinality(c.conkey)=1 and n.nspname=%s""", (schema,))
    return primary, {(t, a): (p, f) for t, a, p, f in cur.fetchall()}


def enlarge(cur, schema: str, rng: random.Random) -> None:
    """Shadow every table with a temp copy that has NEW_ROWS more rows. Copies keep their keys: without them PostgreSQL
    refuses `GROUP BY id` with a selected name, which would look like a broken query."""
    cur.execute("select table_name from information_schema.tables where table_schema=%s and table_type='BASE TABLE'", (schema,))
    tables = [t for (t,) in cur.fetchall()]
    primary, foreign = key_columns(cur, schema)
    rows_by_table = {}
    for table in tables:
        source = pgsql.Identifier(schema, table)
        cur.execute(pgsql.SQL("select * from {}").format(source))
        columns = [d.name for d in cur.description]
        data = cur.fetchall()
        values = {c: [r[i] for r in data if r[i] is not None] for i, c in enumerate(columns)}
        added = []
        for n in range(NEW_ROWS):
            row = {}
            for c in columns:
                if c in primary[table] and len(primary[table]) == 1:
                    numbers = [v for v in values[c] if isinstance(v, int)]
                    row[c] = max(numbers, default=0) + 1 + n if numbers or not values[c] else f"{values[c][0]}_{n}"
                elif (table, c) in foreign or c in primary[table]:
                    row[c] = rng.choice(values[c]) if values[c] else None
                else:
                    row[c] = new_value(values[c], n, rng)
            added.append(row)
        rows_by_table[table] = (columns, added)
    for (table, column), (parent, key) in foreign.items():  # half of the children point at the parent's new rows
        if table in rows_by_table and parent in rows_by_table:
            keys = [r[key] for r in rows_by_table[parent][1] if r.get(key) is not None]
            for row in rows_by_table[table][1]:
                if keys and rng.random() < .5:
                    row[column] = rng.choice(keys)
    for table, (columns, added) in rows_by_table.items():
        target, source = pgsql.Identifier(table), pgsql.Identifier(schema, table)
        cur.execute(pgsql.SQL("create temp table {} (like {} including defaults including indexes)").format(target, source))
        cur.execute(pgsql.SQL("insert into pg_temp.{} select * from {}").format(target, source))
        insert = pgsql.SQL("insert into pg_temp.{} ({}) values ({})").format(
            target, pgsql.SQL(",").join(map(pgsql.Identifier, columns)), pgsql.SQL(",").join(pgsql.Placeholder() * len(columns)))
        for row in added:
            try:  # a row that breaks a composite key or a type is simply left out
                cur.execute("savepoint new_row")
                cur.execute(insert, [row[c] for c in columns])
                cur.execute("release savepoint new_row")
            except psycopg.Error:
                cur.execute("rollback to savepoint new_row")


def run(cur, sql: str) -> tuple[bool, object]:
    try:
        cur.execute("savepoint query")
        cur.execute(sql.replace("%", "%%"))
        rows = cur.fetchall()
        cur.execute("release savepoint query")
        return True, rows
    except psycopg.Error as exc:
        cur.execute("rollback to savepoint query")
        return False, str(exc).split("\n")[0][:160]


def sig(rows: list, ordered: bool):
    return signature(normalize_rows(rows), ordered)


def caught(cur, target: str, target_rows: list, ordered: bool) -> list[bool]:
    found = []
    for wrong in planted_mistakes(target):
        if is_read_only(wrong):
            ok, rows = run(cur, wrong)
            found.append(not ok or sig(rows, ordered) != sig(target_rows, ordered))
    return found


def compare(cur, row: dict) -> dict:
    gold_ok, gold = run(cur, row["gold_sql"])
    target_ok, target = run(cur, row["target_sql"])
    if not (gold_ok and target_ok):
        return {"status": "error", "gold_error": None if gold_ok else gold, "target_error": None if target_ok else target}
    ordered = is_ordered(row["gold_sql"], "postgres")
    status = ("same" if sig(gold, ordered) == sig(target, ordered) else
              "order_only" if sig(gold, False) == sig(target, False) else "different")
    return {"status": status, "gold_rows": len(gold), "target_rows": len(target),
            "mistakes_caught": caught(cur, row["target_sql"], target, ordered)}


def check_database(db_id: str, rows: list[dict]) -> list[dict]:
    cfg = pg_config()
    database, _, schema = db_id.partition("/")
    rows = [r for r in rows if is_read_only(r["gold_sql"]) and is_read_only(r["target_sql"])]
    results = {r["question_id"]: {"question_id": r["question_id"], "db_id": db_id, "level": r["level"], "seeds": []} for r in rows}
    with psycopg.connect(host=cfg.host, port=cfg.port, dbname=database, user=cfg.admin_user, password=cfg.admin_password,
                         connect_timeout=15) as conn:
        cur = conn.cursor()
        cur.execute(f"set statement_timeout={TIMEOUT_MS}")
        cur.execute(pgsql.SQL("set search_path to {}").format(pgsql.Identifier(schema)))
        conn.commit()
        cur.execute("set local transaction_read_only=on")
        for r in rows:
            results[r["question_id"]]["original"] = compare(cur, r)
        conn.rollback()
        for seed in SEEDS:
            enlarge(cur, schema, random.Random(f"{db_id}-{seed}"))
            cur.execute("set local transaction_read_only=on")
            for r in rows:
                results[r["question_id"]]["seeds"].append(compare(cur, r))
            conn.rollback()
    return list(results.values())


def verdict(result: dict) -> str:
    statuses = {s["status"] for s in result["seeds"]}
    return ("different" if "different" in statuses else "error" if statuses == {"error"} else
            "order_only" if "order_only" in statuses else "same")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("data", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="random sample of questions; 0 = all")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)

    questions: dict[str, dict] = {}
    for line in open(args.data, encoding="utf-8"):
        r = json.loads(line)
        if r["dialect"] == "postgres":
            questions.setdefault(r["question_id"], r)
    chosen = list(questions.values())
    if args.limit:
        chosen = random.Random(1).sample(chosen, min(args.limit, len(chosen)))
    by_db = defaultdict(list)
    for r in chosen:
        by_db[r["db_id"]].append(r)

    results = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh, ThreadPoolExecutor(args.workers) as pool:
        for found in pool.map(lambda item: check_database(*item), by_db.items()):
            results += found
            for d in found:
                fh.write(json.dumps(d, ensure_ascii=False) + "\n")
    counts = Counter(verdict(r) for r in results)
    print(f"{len(results)} questions: " + ", ".join(f"{k} {v} ({100 * v / len(results):.1f}%)" for k, v in counts.most_common()))
    original = [x for r in results for x in r["original"].get("mistakes_caught", [])]
    bigger = [x for r in results for s in r["seeds"] for x in s.get("mistakes_caught", [])]
    print(f"planted mistakes caught: original data {100 * sum(original) / max(1, len(original)):.1f}%, "
          f"bigger data {100 * sum(bigger) / max(1, len(bigger)):.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
