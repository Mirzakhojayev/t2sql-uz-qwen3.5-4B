"""Plan and run the teacher stage for the 10k pilot.

  python -m t2sql.pilot plan                  # teacher_jobs.jsonl (PostgreSQL) and sqlite_reuse.jsonl (no calls)
  python -m t2sql.pilot probe --n 100         # first N teacher calls + a few translations: real cost numbers
  python -m t2sql.pilot call --max-usd 8.5    # the teacher run (resumes, never pays twice)

Pilot shape (decided 2026-11-02): 10k examples from 5,000 questions, 2 languages per question.
65% PostgreSQL: 3,250 accepted questions, of which 1,250 messy (25% of all questions) and 2,000 plain.
35% SQLite: 1,750 questions that reuse SynSQL's own traces, so they cost no teacher calls.

Order matters: teacher first, then pick the pilot from accepted questions, then translate only those. Translating
first would pay for questions the teacher stage later discards.

Hard questions are accepted less often (100-call test: Simple 80%, Moderate 69%, Complex 50%, Highly Complex 33%),
so more of them are sent to the teacher, to keep the accepted set balanced across levels.
Questions of a messy database appear only in their messy form.
"""
import argparse
import json
import random
from collections import Counter
from pathlib import Path

from eval.backends import PostgresBackend, SQLiteBackend
from eval.config import pg_config
from eval.harness import Evaluator, summarize

from . import teacher, translate
from .convert import OUT_DIR
from .prompt import build_prompt, extract_sql, read_sqlite_schema, schema_to_ddl
from .synsql import DB_DIR
from .teacher_test import TEACHER_SYSTEM

JOBS = OUT_DIR / "teacher_jobs.jsonl"
ANSWERS = OUT_DIR / "teacher_answers.jsonl"
SQLITE_REUSE = OUT_DIR / "sqlite_reuse.jsonl"
PG_SHARE = 0.65
MESSY_SHARE_OF_PG = 1250 / 3250
# teacher calls per level, sized for about 800 accepted questions each; None = all that exist
CALLS_PER_LEVEL = {"Simple": 1100, "Moderate": 1300, "Complex": 1800, "Highly Complex": None}


def pick_jobs(plain_pg: list[dict], messy: list[dict], already_answered: set[str], rng: random.Random) -> list[dict]:
    """Choose which PostgreSQL questions go to the teacher, level by level. Questions already answered come first
    so their calls are not wasted; if one kind runs short, the other fills the gap."""
    def by_level(rows: list[dict]) -> dict[str, list[dict]]:
        out: dict[str, list[dict]] = {lvl: [] for lvl in CALLS_PER_LEVEL}
        shuffled = list(rows)
        rng.shuffle(shuffled)
        for r in sorted(shuffled, key=lambda r: r["id"] not in already_answered):  # stable: answered first
            out[r["level"]].append(r)
        return out

    plain_by, messy_by = by_level(plain_pg), by_level(messy)
    picked: list[dict] = []
    for lvl, calls in CALLS_PER_LEVEL.items():
        if calls is None:
            picked += plain_by[lvl] + messy_by[lvl]
            continue
        n_messy = min(round(calls * MESSY_SHARE_OF_PG), len(messy_by[lvl]))
        n_plain = min(calls - n_messy, len(plain_by[lvl]))
        n_messy = min(calls - n_plain, len(messy_by[lvl]))
        picked += plain_by[lvl][:n_plain] + messy_by[lvl][:n_messy]
    rng.shuffle(picked)  # so that --limit gives a fair mix
    return picked


def plan(seed: int = 0) -> dict:
    rng = random.Random(seed)
    plain = [json.loads(line) for line in open(OUT_DIR / "prompts_pg.jsonl", encoding="utf-8")]
    messy = [json.loads(line) for line in open(OUT_DIR / "prompts_messy.jsonl", encoding="utf-8")]
    messy_dbs = {json.loads(line)["db_id"] for line in open(OUT_DIR / "messy_maps.jsonl", encoding="utf-8")}
    plain = [r for r in plain if r["db_id"] not in messy_dbs]

    total = len(plain) + len(messy)
    pg_plain_needed = max(0, round(total * PG_SHARE) - len(messy))
    rng.shuffle(plain)
    plain_pg, plain_sqlite = plain[:pg_plain_needed], plain[pg_plain_needed:]

    # --- plain SQLite: SynSQL's own trace is the answer, no teacher call ---
    ddl_cache: dict[str, str] = {}
    with open(SQLITE_REUSE, "w", encoding="utf-8") as fh:
        for r in plain_sqlite:
            db = r["db_id"]
            if db not in ddl_cache:
                ddl_cache[db] = schema_to_ddl(read_sqlite_schema(DB_DIR / db / f"{db}.sqlite"))
            fh.write(json.dumps({
                **{k: r[k] for k in ("id", "level", "style", "question", "hint", "source", "license")},
                "schema_style": "plain", "dialect": "sqlite", "db_id": db, "source_db": db,
                "gold_sql": r["gold_sql_sqlite"], "trace": r["synsql_cot"], "trace_source": "synsql",
                "prompt": build_prompt("sqlite", ddl_cache[db], r["question"], hint=r["hint"]),
            }, ensure_ascii=False) + "\n")

    # --- PostgreSQL: the teacher writes the trace ---
    answered = set()
    if ANSWERS.exists():
        answered = {json.loads(line)["id"] for line in open(ANSWERS, encoding="utf-8")}
    for r in plain_pg:
        r["schema_style"] = "plain"
    picked = pick_jobs(plain_pg, messy, answered, rng)
    with open(JOBS, "w", encoding="utf-8") as fh:
        for r in picked:
            fh.write(json.dumps({
                **{k: r[k] for k in ("id", "level", "style", "question", "hint", "source", "license", "schema_style")},
                "dialect": "postgres", "db_id": r["pg_db"], "source_db": r["db_id"], "gold_sql": r["gold_sql"],
                "messages": [{"role": "system", "content": TEACHER_SYSTEM}, {"role": "user", "content": r["prompt"]}],
            }, ensure_ascii=False) + "\n")
    return {
        "teacher_jobs": len(picked), "already_answered_in_jobs": sum(r["id"] in answered for r in picked),
        "by_level": dict(Counter(r["level"] for r in picked)),
        "by_schema_style": dict(Counter(r["schema_style"] for r in picked)),
        "sqlite_reuse_available": len(plain_sqlite),
        "sqlite_reuse_by_level": dict(Counter(r["level"] for r in plain_sqlite)),
        "projected_teacher_usd_off_peak": round(len(picked) * 0.00121, 2),
    }


def load_jobs() -> list[dict]:
    return [json.loads(line) for line in open(JOBS, encoding="utf-8")]


def score_answers(jobs_by_id: dict[str, dict], answers: list[dict]) -> dict:
    """Accepted = the teacher's SQL returns the same result as gold, in the job's dialect."""
    records = [{
        "id": a["id"], "language": "en", "dialect": jobs_by_id[a["id"]]["dialect"],
        "slice": jobs_by_id[a["id"]]["level"], "db_id": jobs_by_id[a["id"]]["db_id"],
        "gold_sql": jobs_by_id[a["id"]]["gold_sql"], "pred_sql": extract_sql(a.get("content") or "") or "",
    } for a in answers]
    backends = {"sqlite": SQLiteBackend(DB_DIR, 15), "postgres": PostgresBackend(pg_config(), 15)}
    ev = Evaluator(backends)
    ev.prefetch(records, workers=8)
    scored = [ev.score(r) for r in records]
    for b in backends.values():
        b.close()
    return summarize(scored)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan")
    p = sub.add_parser("probe")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--translations", type=int, default=50)
    c = sub.add_parser("call")
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--max-usd", type=float, required=True)
    args = ap.parse_args(argv)

    if args.cmd == "plan":
        print(json.dumps(plan(), indent=1))
        return 0

    jobs = load_jobs()
    if args.cmd == "call":
        teacher.run(jobs[:args.limit] if args.limit else jobs, ANSWERS, effort="high", max_usd=args.max_usd)
        return 0

    # --- probe: goes into the same answers file, so tomorrow's run does not pay for these again ---
    teacher.run(jobs[:args.n], ANSWERS, effort="high", max_usd=1.0)
    by_id = {j["id"]: j for j in jobs}
    answers = [a for a in map(json.loads, open(ANSWERS, encoding="utf-8")) if a["id"] in by_id]
    summary = summarize_probe(by_id, answers)

    questions = [{"question_id": j["id"], "question": j["question"], "db_id": j["source_db"], "difficulty": j["level"],
                  "source": j["source"], "license": j["license"], "split": "train"}
                 for j in jobs[:args.translations]]
    tr_file = OUT_DIR / "translations_probe.jsonl"
    teacher.run(translate.jobs_for(questions), tr_file, effort="high", workers=8, max_usd=0.25)
    translations = [json.loads(line) for line in open(tr_file, encoding="utf-8")]
    summary["cost_per_translation"] = round(sum(t["cost_usd"] for t in translations) / len(translations), 6)
    summary["projected_translation_usd_7000"] = round(summary["cost_per_translation"] * 7000, 2)
    summary["translation_flags"] = sum(bool(translate.checks(t)) for t in translations)
    (OUT_DIR / "probe_report.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(json.dumps(summary, indent=1))
    return 0


def summarize_probe(by_id: dict[str, dict], answers: list[dict]) -> dict:
    report = score_answers(by_id, answers)
    cost = sum(a["cost_usd"] for a in answers)
    per_call = cost / len(answers)
    tokens = sorted(a["usage"].get("completion_tokens", 0) for a in answers)
    n_jobs = len(by_id)
    return {
        "calls": len(answers), "cost_usd": round(cost, 4), "cost_per_call": round(per_call, 6),
        "projected_teacher_usd_all_jobs": round(per_call * n_jobs, 2), "jobs": n_jobs,
        "accept_pct": report["overall"]["all"]["acc"],
        "accept_pct_by_dialect": {k: v["acc"] for k, v in report["dialect"].items()},
        "accept_pct_by_level": {k: v["acc"] for k, v in report["slice"].items()},
        "pred_errors": report["overall"]["all"]["pred_errors"],
        "pred_timeouts": report["overall"]["all"]["pred_timeouts"],
        "gold_errors": report["overall"]["all"]["gold_errors"],
        "cut_off": sum(a["finish_reason"] == "length" for a in answers),
        "output_tokens_p50_p90_max": [tokens[len(tokens) // 2], tokens[int(.9 * len(tokens))], tokens[-1]],
        "note": "costs are at the price of the moment: off-peak now, double at peak",
    }


if __name__ == "__main__":
    raise SystemExit(main())
