"""Pipeline steps 2-3: load SynSQL databases into PostgreSQL, convert gold SQL with sqlglot, keep the matches.

  python -m t2sql.convert --max-dbs 1500

Each source database becomes one schema of the shared PostgreSQL database `synsql` (a separate database for
each would cost about 7 MB of overhead per database). A question is kept only if its SQLite gold and its
converted PostgreSQL query return the same result under the eval harness's comparison rules.

Writes data/pipeline/converted.jsonl, rejected.jsonl and convert_stats.json.
"""
import argparse
import datetime
import hashlib
import json
import random
import re
import sqlite3
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psycopg
import sqlglot

from eval.backends import PostgresBackend
from eval.compare import is_ordered, normalize_rows, signature
from eval.config import PgConfig, pg_config

from .synsql import DB_DIR

PG_DATABASE = "synsql"
OUT_DIR = Path("data/pipeline")
INT8_MAX = 2 ** 63


def schema_name(db_id: str) -> str:
    """SynSQL names can exceed PostgreSQL's 63-character identifier limit, so schemas are named by hash."""
    return "s_" + hashlib.sha1(db_id.encode("utf-8")).hexdigest()[:16]


def _q(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _parses_as(values: list, parse) -> bool:
    try:
        for v in values:
            if not isinstance(v, str):
                return False
            parse(v)
        return True
    except ValueError:
        return False


def pg_type(declared: str, values: list) -> str:
    """PostgreSQL column type from the declared SQLite type AND the data. SQLite stores anything in any
    column, so a declared type is only used if every value really fits it; otherwise the column is TEXT.
    BOOLEAN stays numeric because SQLite stores 0/1 and gold SQL writes `flag = 1`."""
    d = (declared or "").upper()
    vals = [v for v in values if v is not None]
    is_int = all(isinstance(v, int) and abs(v) < INT8_MAX for v in vals)
    is_num = all(isinstance(v, (int, float)) for v in vals)
    if vals and all(isinstance(v, (bytes, bytearray)) for v in vals):
        return "BYTEA"
    if "INT" in d or "BOOL" in d:
        if is_int:
            return "BIGINT"
        return "DOUBLE PRECISION" if is_num else "TEXT"
    if any(k in d for k in ("REAL", "FLOA", "DOUB")):
        return "DOUBLE PRECISION" if is_num else "TEXT"
    if any(k in d for k in ("NUMERIC", "DECIMAL")):
        return "NUMERIC" if is_num else "TEXT"
    if d == "DATE" and vals and _parses_as(vals, datetime.date.fromisoformat):
        return "DATE"
    if d in ("DATETIME", "TIMESTAMP") and vals and _parses_as(vals, datetime.datetime.fromisoformat):
        return "TIMESTAMP"
    if not d and vals:
        if is_int:
            return "BIGINT"
        if is_num:
            return "DOUBLE PRECISION"
    return "TEXT"


def _clean(v, col_type: str):
    if v is None:
        return None
    if col_type == "TEXT":
        v = v.hex() if isinstance(v, (bytes, bytearray)) else str(v)
        return v.replace("\x00", "")  # PostgreSQL text cannot hold NUL
    return v


def load_database(conn: psycopg.Connection, sqlite_path: Path, schema: str) -> dict:
    """Copy one SQLite file into a new schema, in one transaction. conn must be autocommit (admin user)."""
    src = sqlite3.connect(f"{Path(sqlite_path).resolve().as_uri()}?mode=ro", uri=True)
    src.text_factory = lambda b: b.decode("utf-8", errors="replace")
    report = {"tables": 0, "rows": 0, "primary_keys": 0, "foreign_keys": 0, "foreign_keys_skipped": 0}
    try:
        with conn.transaction():
            conn.execute(f"CREATE SCHEMA {_q(schema)}")
            tables = [r[0] for r in src.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")]
            pk_of: dict[str, list[str]] = {}
            for table in tables:
                info = src.execute(f"PRAGMA table_info({_q(table)})").fetchall()
                rows = src.execute(f"SELECT * FROM {_q(table)}").fetchall()
                types = [pg_type(col[2], [r[i] for r in rows]) for i, col in enumerate(info)]
                pk = [c[1] for c in sorted(info, key=lambda c: c[5]) if c[5] > 0]
                pk_of[table] = pk
                # A real key matters: PostgreSQL's GROUP BY rule allows other columns when grouping by the key.
                # SQLite does not enforce keys strictly, so add it only if the data allows it.
                positions = [next(i for i, c in enumerate(info) if c[1] == name) for name in pk]
                keys = [tuple(r[i] for i in positions) for r in rows]
                key_ok = pk and len(set(keys)) == len(keys) and not any(v is None for k in keys for v in k)
                cols = [f"{_q(c[1])} {t}" for c, t in zip(info, types)]
                if key_ok:
                    cols.append(f"PRIMARY KEY ({', '.join(_q(n) for n in pk)})")
                    report["primary_keys"] += 1
                target = f"{_q(schema)}.{_q(table)}"
                conn.execute(f"CREATE TABLE {target} ({', '.join(cols)})")
                if rows:
                    with conn.cursor().copy(f"COPY {target} FROM STDIN") as copy:
                        for row in rows:
                            copy.write_row([_clean(v, t) for v, t in zip(row, types)])
                report["tables"] += 1
                report["rows"] += len(rows)
            # NOT VALID: the foreign key is declared (the prompt shows it) without checking existing rows
            for table in tables:
                for fk in src.execute(f"PRAGMA foreign_key_list({_q(table)})").fetchall():
                    ref_table, from_col, to_col = fk[2], fk[3], fk[4]
                    if to_col is None and pk_of.get(ref_table):
                        to_col = pk_of[ref_table][0]
                    try:
                        with conn.transaction():
                            conn.execute(
                                f"ALTER TABLE {_q(schema)}.{_q(table)} ADD FOREIGN KEY ({_q(from_col)}) "
                                f"REFERENCES {_q(schema)}.{_q(ref_table)} ({_q(to_col)}) NOT VALID")
                        report["foreign_keys"] += 1
                    except (psycopg.Error, TypeError):
                        report["foreign_keys_skipped"] += 1
    finally:
        src.close()
    return report


def to_postgres(sql: str) -> str | None:
    """SQLite gold -> PostgreSQL with sqlglot. None if sqlglot cannot parse it or it is not one statement."""
    try:
        statements = sqlglot.transpile(sql, read="sqlite", write="postgres")
    except Exception:
        return None
    return statements[0] if len(statements) == 1 else None


def error_kind(message: str) -> str:
    """Group error messages: names and values removed, so the same mistake counts as one kind."""
    m = re.sub(r'"[^"]*"', '"X"', message)
    m = re.sub(r"'[^']*'", "'X'", m)
    m = re.sub(r"\b\d+\b", "N", m)
    return m[:80]


def sqlite_rows(db_id: str, sql: str, timeout_s: float = 10.0) -> list[tuple] | None:
    path = (DB_DIR / db_id / f"{db_id}.sqlite").resolve()
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    deadline = time.monotonic() + timeout_s
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 10_000)
    try:
        return [tuple(r) for r in conn.execute(sql).fetchmany(100_001)]
    except sqlite3.Error:
        return None
    finally:
        conn.close()


def check_question(rec: dict, pg: PostgresBackend, gold_rows: list[tuple] | None = None) -> dict:
    """Convert one question and compare both engines. Returns the record with sql_postgres, or a reason."""
    db_id, sql = rec["db_id"], rec["sql"]
    pg_id = f"{PG_DATABASE}/{schema_name(db_id)}"
    out = {**rec, "pg_db": pg_id}
    gold_rows = sqlite_rows(db_id, sql) if gold_rows is None else gold_rows
    if gold_rows is None:
        return {**out, "reason": "sqlite_failed"}
    converted = to_postgres(sql)
    if converted is None:
        return {**out, "reason": "cannot_convert"}
    out["sql_postgres"] = converted
    res = pg.execute(pg_id, converted)
    if res.timed_out:
        return {**out, "reason": "pg_timeout"}
    if not res.ok:
        return {**out, "reason": "pg_error", "error": res.error}
    ordered = is_ordered(sql, "sqlite")
    a, b = normalize_rows(gold_rows), normalize_rows(res.rows)
    if signature(a, ordered) == signature(b, ordered):
        return {**out, "reason": None}
    if ordered and signature(a, False) == signature(b, False):
        return {**out, "reason": "order_differs"}
    return {**out, "reason": "result_differs"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", type=Path, default=OUT_DIR / "candidates.jsonl")
    ap.add_argument("--max-dbs", type=int, default=1500, help="how many source databases to load")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)

    started = time.time()
    by_db: dict[str, list[dict]] = defaultdict(list)
    for line in open(args.candidates, encoding="utf-8"):
        rec = json.loads(line)
        by_db[rec["db_id"]].append(rec)
    db_ids = sorted(by_db)
    random.Random(args.seed).shuffle(db_ids)
    db_ids = db_ids[:args.max_dbs]
    questions = sum(len(by_db[d]) for d in db_ids)
    print(f"{len(by_db):,} databases in candidates; using {len(db_ids):,} with {questions:,} questions", flush=True)

    cfg: PgConfig = pg_config()
    admin = dict(host=cfg.host, port=cfg.port, user=cfg.admin_user, password=cfg.admin_password, connect_timeout=10)
    with psycopg.connect(dbname="postgres", autocommit=True, **admin) as c:
        if not c.execute("SELECT 1 FROM pg_database WHERE datname = %s", (PG_DATABASE,)).fetchone():
            c.execute(f"CREATE DATABASE {PG_DATABASE}")
    with psycopg.connect(dbname=PG_DATABASE, autocommit=True, **admin) as c:
        existing = {r[0] for r in c.execute("SELECT schema_name FROM information_schema.schemata")}

    # --- load the databases not loaded yet ---
    local = threading.local()
    totals: Counter = Counter()
    load_failed: dict[str, str] = {}
    lock = threading.Lock()

    def load(db_id: str) -> None:
        if not hasattr(local, "conn"):
            local.conn = psycopg.connect(dbname=PG_DATABASE, autocommit=True, **admin)
        try:
            report = load_database(local.conn, DB_DIR / db_id / f"{db_id}.sqlite", schema_name(db_id))
            with lock:
                totals.update(report)
                totals["databases_loaded"] += 1
                if totals["databases_loaded"] % 250 == 0:
                    print(f"  loaded {totals['databases_loaded']:,} databases ({time.time() - started:.0f}s)", flush=True)
        except Exception as exc:  # one odd database must not stop the run; it is reported below
            with lock:
                load_failed[db_id] = f"{type(exc).__name__}: {str(exc)[:200]}"

    todo = [d for d in db_ids if schema_name(d) not in existing]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(load, todo))
    print(f"loaded {totals['databases_loaded']:,} new databases, {len(load_failed)} failed ({time.time() - started:.0f}s)", flush=True)

    # --- convert and compare ---
    pg = PostgresBackend(cfg, timeout_s=10.0)
    recs = [r for d in db_ids if d not in load_failed for r in by_db[d]]
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = list(ex.map(lambda r: check_question(r, pg), recs))
    pg.close()

    kept = [r for r in results if r["reason"] is None]
    rejected = [r for r in results if r["reason"] is not None]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "converted.jsonl", "w", encoding="utf-8") as fh:
        for r in kept:
            r.pop("reason")
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT_DIR / "rejected.jsonl", "w", encoding="utf-8") as fh:
        for r in rejected:
            fh.write(json.dumps({k: r.get(k) for k in ("id", "db_id", "sql_complexity", "question", "sql", "sql_postgres", "reason", "error")},
                                ensure_ascii=False) + "\n")

    level = Counter(r["sql_complexity"] for r in results)
    level_kept = Counter(r["sql_complexity"] for r in kept)
    stats = {
        "databases": len(db_ids), "load_totals": dict(totals), "load_failed": load_failed,
        "questions": len(results), "kept": len(kept),
        "kept_pct": round(100 * len(kept) / max(len(results), 1), 1),
        "kept_pct_by_level": {k: round(100 * level_kept[k] / n, 1) for k, n in sorted(level.items())},
        "rejected_by_reason": dict(Counter(r["reason"] for r in rejected).most_common()),
        "pg_error_kinds": dict(Counter(error_kind(r["error"]) for r in rejected if r.get("error")).most_common(20)),
        "seconds": round(time.time() - started),
    }
    (OUT_DIR / "convert_stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
