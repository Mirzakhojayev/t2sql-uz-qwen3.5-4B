"""Pipeline: build the PostgreSQL prompt for every converted question.

  python -m t2sql.pg_prompts

Reads data/pipeline/converted.jsonl, writes data/pipeline/prompts_pg.jsonl. Each row keeps its source, license,
level and style tags. `prompt` is the user message only: the teacher and training code add their own system text.
"""
import argparse
import json
import sqlite3
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psycopg

from eval.config import pg_config

from .convert import OUT_DIR, PG_DATABASE, schema_name
from .prompt import build_prompt, schema_to_ddl
from .schema_pg import read_postgres_schema


def schema_ddl(conn: psycopg.Connection, db_id: str, schema: str | None = None) -> str:
    # plain SET inside the transaction: read_postgres_schema's rollback at the end undoes it
    conn.execute(f'SET search_path TO "{schema or schema_name(db_id)}"')
    return schema_to_ddl(read_postgres_schema(conn))


def ddl_is_valid(ddl: str) -> bool:
    """Syntax check in an empty database (SQLite accepts the PostgreSQL types we use)."""
    conn = sqlite3.connect(":memory:")
    try:
        conn.executescript(ddl)
        return True
    except sqlite3.Error:
        return False
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--converted", type=Path, default=OUT_DIR / "converted.jsonl")
    ap.add_argument("--out", type=Path, default=OUT_DIR / "prompts_pg.jsonl")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args(argv)

    started = time.time()
    by_db: dict[str, list[dict]] = defaultdict(list)
    for line in open(args.converted, encoding="utf-8"):
        rec = json.loads(line)
        by_db[rec["db_id"]].append(rec)

    cfg = pg_config()
    local = threading.local()
    done = {"n": 0}
    lock = threading.Lock()

    def one(db_id: str) -> tuple[str, str]:
        if not hasattr(local, "conn"):
            local.conn = psycopg.connect(host=cfg.host, port=cfg.port, dbname=PG_DATABASE, user=cfg.ro_user,
                                         password=cfg.ro_password, connect_timeout=10)
        ddl = schema_ddl(local.conn, db_id)
        with lock:
            done["n"] += 1
            if done["n"] % 1000 == 0:
                print(f"  {done['n']:,}/{len(by_db):,} schemas ({time.time() - started:.0f}s)", flush=True)
        return db_id, ddl

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        ddls = dict(ex.map(one, list(by_db)))

    invalid = {d for d, ddl in ddls.items() if not ddl_is_valid(ddl)}
    n = 0
    with open(args.out, "w", encoding="utf-8") as out:
        for db_id, recs in by_db.items():
            if db_id in invalid:
                continue
            for rec in recs:
                out.write(json.dumps({
                    "id": rec["id"], "db_id": db_id, "pg_db": rec["pg_db"], "dialect": "postgres",
                    "language": "en", "level": rec["sql_complexity"], "style": rec["question_style"],
                    "question": rec["question"], "hint": rec.get("external_knowledge") or None,
                    "gold_sql": rec["sql_postgres"], "gold_sql_sqlite": rec["sql"], "synsql_cot": rec["cot"],
                    "source": rec["source"], "license": rec["license"],
                    "prompt": build_prompt("postgres", ddls[db_id], rec["question"], hint=rec.get("external_knowledge")),
                }, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n:,} prompts from {len(ddls) - len(invalid):,} schemas to {args.out}; "
          f"invalid DDL in {len(invalid)} ({time.time() - started:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
