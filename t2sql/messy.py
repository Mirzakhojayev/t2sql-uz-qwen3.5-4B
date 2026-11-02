"""Pipeline step 4: about 25% of databases get local-style "messy" table and column names.

  python -m t2sql.messy names --limit 20    # DeepSeek translates identifiers (first a small batch, check the cost)
  python -m t2sql.messy names --max-usd 3   # the rest (resumes)
  python -m t2sql.messy apply               # load renamed copies into PostgreSQL, rewrite gold SQL, verify

One consistent rename map per database: the same original name always gets the same new name, in tables, columns
and hints. Questions stay as they are, because a person does not know the schema's odd names. A question is kept
only if the renamed query returns the same result as the original. Traces must be rewritten by the teacher.

Styles: uz_latin (xodimlar), cyrillic (сотрудники), translit (sotrudniki), abbrev (emp_nm), mixed.
Writes messy_names.jsonl, messy_maps.jsonl, prompts_messy.jsonl and messy_stats.json under data/pipeline/.
"""
import argparse
import hashlib
import json
import random
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import psycopg
import sqlglot
from sqlglot import exp

from eval.backends import PostgresBackend
from eval.compare import is_ordered, normalize_rows, signature
from eval.config import pg_config

from . import teacher
from .convert import OUT_DIR, PG_DATABASE, load_database, schema_name
from .pg_prompts import ddl_is_valid, schema_ddl
from .prompt import build_prompt, read_sqlite_schema
from .synsql import DB_DIR

STYLES = ["uz_latin", "cyrillic", "translit", "abbrev", "mixed"]
SHARE = 0.25

NAMES_SYSTEM = (
    "You rename database identifiers the way developers in Uzbekistan name things. For each English identifier "
    "give three forms: uz = Uzbek in Latin letters; ru = Russian in Cyrillic; translit = the Russian word written in "
    "Latin letters. Use lower case and underscores, keep names short, and do not use apostrophes or spaces. "
    "Keep common technical suffixes meaningful (user_id -> foydalanuvchi_id). Reply with only one JSON object: "
    '{"identifier": {"uz": "...", "ru": "...", "translit": "..."}, ...} containing every identifier.'
)


def style_of(db_id: str) -> str:
    return STYLES[int(hashlib.sha1(db_id.encode("utf-8")).hexdigest(), 16) % len(STYLES)]


def messy_schema_name(db_id: str) -> str:
    return "m" + schema_name(db_id)[1:]


def sanitize(name: str) -> str:
    """Identifier-safe: letters of any script, digits and underscores only."""
    name = re.sub(r"['‘’ʻʼ`\"]", "", name.strip().lower())
    name = re.sub(r"[^\w]+", "_", name, flags=re.UNICODE).strip("_")
    return name


def abbreviate(name: str) -> str:
    """customer_name -> cstmr_nm: vowels dropped after the first letter of each word, at most 5 letters a word."""
    parts = []
    for word in name.lower().split("_"):
        if word in ("id", "no", "at", "by") or len(word) <= 3:
            parts.append(word)
        else:
            parts.append((word[0] + re.sub(r"[aeiou]", "", word[1:]))[:5])
    return "_".join(p for p in parts if p) or name.lower()


def build_map(identifiers: list[str], forms: dict[str, dict], style: str, seed: str) -> dict[str, str]:
    """{lower-case original: new name}. Raises ValueError if a needed translation is missing. Two originals never
    share a new name: a collision would silently merge two columns."""
    rng = random.Random(seed)
    result: dict[str, str] = {}
    used: set[str] = set()
    for original in identifiers:
        chosen = rng.choice(["uz_latin", "cyrillic", "translit", "abbrev"]) if style == "mixed" else style
        if chosen == "abbrev":
            candidate = abbreviate(original)
        else:
            key = {"uz_latin": "uz", "cyrillic": "ru", "translit": "translit"}[chosen]
            entry = forms.get(original) or {}
            if not entry.get(key):
                raise ValueError(f"no {key} form for {original!r}")
            candidate = sanitize(entry[key])
        if not candidate:
            raise ValueError(f"empty name for {original!r}")
        name, k = candidate, 2
        while name in used:
            name, k = f"{candidate}_{k}", k + 1
        used.add(name)
        result[original.lower()] = name
    return result


def rename_sql(sql: str, mapping: dict[str, str]) -> str:
    """Rewrite PostgreSQL SQL with the map. Every matching identifier changes, so an alias or CTE that happens to
    share a column's name changes too, which is still consistent. String values and function names are untouched."""
    tree = sqlglot.parse_one(sql, read="postgres")
    for ident in tree.find_all(exp.Identifier):
        new = mapping.get(ident.name.lower())
        if new is not None:
            ident.set("this", new)
            ident.set("quoted", True)
    return tree.sql(dialect="postgres")


def rename_hint(hint: str | None, mapping: dict[str, str]) -> str | None:
    """Hints quote column names. Only names with an underscore are replaced: plain words like `name` are English."""
    if not hint:
        return hint
    names = sorted((k for k in mapping if "_" in k), key=len, reverse=True)
    if not names:
        return hint
    return re.sub(r"\b(" + "|".join(map(re.escape, names)) + r")\b", lambda m: mapping[m.group(1).lower()], hint,
                  flags=re.IGNORECASE)


def identifiers_of(db_id: str) -> list[str]:
    seen: dict[str, None] = {}
    for table in read_sqlite_schema(DB_DIR / db_id / f"{db_id}.sqlite", n_examples=0):
        seen.setdefault(table.name)
        for col in table.columns:
            seen.setdefault(col.name)
    return list(seen)


def parse_forms(text: str) -> dict:
    return json.loads(re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip()))


def verify(row: dict, new_sql: str, messy_id: str, pg: PostgresBackend) -> str | None:
    """None if the renamed query returns the same result as the original PostgreSQL gold; else the reason."""
    original = pg.execute(row["pg_db"], row["gold_sql"])
    renamed = pg.execute(messy_id, new_sql)
    if not original.ok:
        return "original_failed"
    if not renamed.ok:
        return "renamed_failed"
    ordered = is_ordered(row["gold_sql"], "postgres")
    if signature(normalize_rows(original.rows), ordered) != signature(normalize_rows(renamed.rows), ordered):
        return "result_differs"
    return None


def chosen_databases(prompts: Path, share: float, seed: int) -> list[str]:
    db_ids = sorted({json.loads(line)["db_id"] for line in open(prompts, encoding="utf-8")})
    random.Random(seed).shuffle(db_ids)
    return db_ids[:round(len(db_ids) * share)]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    n = sub.add_parser("names")
    n.add_argument("--limit", type=int, default=0)
    n.add_argument("--max-usd", type=float, default=0.25)
    a = sub.add_parser("apply")
    a.add_argument("--workers", type=int, default=6)
    ap.add_argument("--prompts", type=Path, default=OUT_DIR / "prompts_pg.jsonl")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    db_ids = chosen_databases(args.prompts, SHARE, args.seed)
    names_file = OUT_DIR / "messy_names.jsonl"

    if args.cmd == "names":
        # abbrev needs no translation, so those databases cost nothing
        need = [d for d in db_ids if style_of(d) != "abbrev"]
        if args.limit:
            need = need[:args.limit]
        jobs = [{"id": d, "db_id": d, "style": style_of(d), "messages": [
            {"role": "system", "content": NAMES_SYSTEM},
            {"role": "user", "content": json.dumps(identifiers_of(d), ensure_ascii=False)},
        ]} for d in need]
        print(f"{len(db_ids):,} messy databases, {len(need):,} need names; styles {dict(Counter(map(style_of, db_ids)))}")
        teacher.run(jobs, names_file, thinking=False, workers=16, max_usd=args.max_usd)
        return 0

    # --- apply ---
    started = time.time()
    forms_by_db: dict[str, dict] = {}
    stats: Counter = Counter()
    if names_file.exists():
        for line in open(names_file, encoding="utf-8"):
            r = json.loads(line)
            try:
                forms_by_db[r["db_id"]] = parse_forms(r["content"])
            except ValueError:
                stats["names_bad_json"] += 1

    maps: dict[str, dict] = {}
    for d in db_ids:
        style = style_of(d)
        if style != "abbrev" and d not in forms_by_db:
            stats["names_missing"] += 1
            continue
        try:
            # "mixed" also picks abbreviations, which need no translation
            maps[d] = build_map(identifiers_of(d), forms_by_db.get(d, {}), style, d)
        except ValueError:
            stats["names_incomplete"] += 1

    cfg = pg_config()
    admin = dict(host=cfg.host, port=cfg.port, user=cfg.admin_user, password=cfg.admin_password, connect_timeout=10)
    with psycopg.connect(dbname=PG_DATABASE, autocommit=True, **admin) as c:
        existing = {r[0] for r in c.execute("SELECT schema_name FROM information_schema.schemata")}

    local = threading.local()

    def load(d: str) -> None:
        if not hasattr(local, "conn"):
            local.conn = psycopg.connect(dbname=PG_DATABASE, autocommit=True, **admin)
        load_database(local.conn, DB_DIR / d / f"{d}.sqlite", messy_schema_name(d), rename=maps[d])

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        list(ex.map(load, [d for d in maps if messy_schema_name(d) not in existing]))
    print(f"loaded {len(maps):,} renamed databases ({time.time() - started:.0f}s)", flush=True)

    rows_by_db: dict[str, list[dict]] = defaultdict(list)
    for line in open(args.prompts, encoding="utf-8"):
        r = json.loads(line)
        if r["db_id"] in maps:
            rows_by_db[r["db_id"]].append(r)

    pg = PostgresBackend(cfg, timeout_s=10.0)
    ro = threading.local()

    def process(d: str) -> list[dict]:
        if not hasattr(ro, "conn"):
            ro.conn = psycopg.connect(host=cfg.host, port=cfg.port, dbname=PG_DATABASE, user=cfg.ro_user,
                                      password=cfg.ro_password, connect_timeout=10)
        ddl = schema_ddl(ro.conn, d, messy_schema_name(d))
        if not ddl_is_valid(ddl):
            return [{"reason": "invalid_ddl"}]
        out = []
        messy_id = f"{PG_DATABASE}/{messy_schema_name(d)}"
        for row in rows_by_db[d]:
            try:
                new_sql = rename_sql(row["gold_sql"], maps[d])
            except Exception:
                out.append({"reason": "rename_failed"})
                continue
            reason = verify(row, new_sql, messy_id, pg)
            if reason:
                out.append({"reason": reason})
                continue
            hint = rename_hint(row["hint"], maps[d])
            out.append({
                **{k: row[k] for k in ("id", "db_id", "dialect", "language", "level", "style", "question", "source", "license")},
                "pg_db": messy_id, "schema_style": style_of(d), "hint": hint, "gold_sql": new_sql,
                "prompt": build_prompt("postgres", ddl, row["question"], hint=hint),
            })
        return out

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = [x for part in ex.map(process, list(maps)) for x in part]
    pg.close()

    kept = [r for r in results if "reason" not in r]
    stats.update(r["reason"] for r in results if "reason" in r)
    with open(OUT_DIR / "prompts_messy.jsonl", "w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(OUT_DIR / "messy_maps.jsonl", "w", encoding="utf-8") as fh:
        for d, m in maps.items():
            fh.write(json.dumps({"db_id": d, "style": style_of(d), "map": m}, ensure_ascii=False) + "\n")
    report = {"databases_chosen": len(db_ids), "databases_renamed": len(maps), "questions_kept": len(kept),
              "by_style": dict(Counter(r["schema_style"] for r in kept)), "problems": dict(stats),
              "seconds": round(time.time() - started)}
    (OUT_DIR / "messy_stats.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
