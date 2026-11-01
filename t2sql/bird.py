"""Build zero-shot prompts for BIRD Mini-Dev (SQLite + PostgreSQL). EVAL DATA: never use these for training.

  python -m t2sql.bird --out-dir data/runs/bird_zs

Writes prompts.jsonl (one record per question and dialect, with the gold SQL for scoring) and the schema
text shown to the model for each database under ddl/, so you can read exactly what the model sees.
"""
import argparse
import json
from pathlib import Path

import psycopg

from eval.config import pg_config

from .prompt import build_prompt, read_sqlite_schema, schema_to_ddl
from .schema_pg import read_postgres_schema

BIRD_DIR = Path("data/bird_minidev/data")
FILES = {"sqlite": "mini_dev_sqlite.jsonl", "postgres": "mini_dev_postgresql.jsonl"}

ZERO_SHOT_SYSTEM = (
    "You are an expert data analyst. Write one SQL query that answers the question for the given database. "
    "Use only tables and columns from the schema, and the SQL dialect named in the prompt. "
    "Briefly explain your reasoning, then give the final query in a ```sql code block."
)


def database_ddl(dialect: str, db_id: str, sqlite_dir: Path) -> str:
    if dialect == "sqlite":
        return schema_to_ddl(read_sqlite_schema(sqlite_dir / db_id / f"{db_id}.sqlite"))
    cfg = pg_config()
    with psycopg.connect(host=cfg.host, port=cfg.port, dbname=db_id, user=cfg.ro_user,
                         password=cfg.ro_password, connect_timeout=10) as conn:
        conn.execute("SET statement_timeout = '300s'")  # sample-value queries scan whole columns
        return schema_to_ddl(read_postgres_schema(conn))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", type=Path, default=Path("data/runs/bird_zs"))
    ap.add_argument("--sqlite-dir", type=Path, default=Path("data/sqlite"))
    ap.add_argument("--dialects", default="sqlite,postgres", help="comma-separated subset of sqlite,postgres")
    args = ap.parse_args(argv)
    dialects = args.dialects.split(",")

    (args.out_dir / "ddl").mkdir(parents=True, exist_ok=True)
    ddl_cache: dict[tuple[str, str], str] = {}
    n = 0
    with open(args.out_dir / "prompts.jsonl", "w", encoding="utf-8") as out:
        for dialect, fname in FILES.items():
            if dialect not in dialects:
                continue
            for line in open(BIRD_DIR / fname, encoding="utf-8"):
                q = json.loads(line)
                key = (dialect, q["db_id"])
                if key not in ddl_cache:
                    print(f"reading schema {dialect}/{q['db_id']}", flush=True)
                    ddl_cache[key] = database_ddl(dialect, q["db_id"], args.sqlite_dir)
                    (args.out_dir / "ddl" / f"{dialect}__{q['db_id']}.sql").write_text(ddl_cache[key], encoding="utf-8")
                user = build_prompt(dialect, ddl_cache[key], q["question"], hint=q.get("evidence"))
                out.write(json.dumps({
                    "id": f"{dialect}-{q['question_id']}", "db_id": q["db_id"], "dialect": dialect,
                    "language": "en", "slice": q["difficulty"], "gold_sql": q["SQL"],
                    "source": "bird_minidev", "license": "CC BY-SA 4.0", "split": "eval",
                    "messages": [{"role": "system", "content": ZERO_SHOT_SYSTEM}, {"role": "user", "content": user}],
                }, ensure_ascii=False) + "\n")
                n += 1
    print(f"wrote {n} prompts to {args.out_dir / 'prompts.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
