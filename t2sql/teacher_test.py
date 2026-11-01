"""Teacher test (Phase 0): how good and how expensive is DeepSeek as a teacher, and how do its traces compare
with SynSQL's own chain-of-thought traces?

  python -m t2sql.teacher_test prepare                       # questions.jsonl -> jobs.jsonl (prompts)
  python -m t2sql.teacher_test call --effort low --limit 20   # first 20 calls, then look at cost
  python -m t2sql.teacher_test call --effort low --max-usd 3  # the rest (resumes, never pays twice)
  python -m t2sql.teacher_test score                          # accept rate, cost per accepted, trace lengths

Accepted = the answer's SQL returns the same result as SynSQL's gold SQL (eval harness rules).
Trace lengths are counted with the Qwen3.5 tokenizer, the same unit as our 16k training cap.
"""
import argparse
import json
import random
from pathlib import Path

from eval.backends import SQLiteBackend
from eval.harness import Evaluator, format_report, summarize

from . import teacher
from .prompt import build_prompt, extract_sql, read_sqlite_schema, schema_to_ddl
from .synsql import DB_DIR, LEVELS

RUN_DIR = Path("data/runs/teacher_test")

TEACHER_SYSTEM = (
    "You are an expert SQL engineer writing training examples for a smaller text-to-SQL model. "
    "Given a database schema and a question, write a short step-by-step explanation in English "
    "(which tables and columns, which joins, filters and aggregations, and any dialect pitfalls), "
    "then exactly one final query in a ```sql code block. The query must run on the dialect named in the "
    "prompt and return exactly what the question asks for: no extra columns."
)


def prepare(questions: Path, jobs_out: Path) -> int:
    n = 0
    # questions.jsonl is sorted by complexity level; shuffle so that --limit N gives a mix of levels
    lines = open(questions, encoding="utf-8").readlines()
    random.Random(0).shuffle(lines)
    with open(jobs_out, "w", encoding="utf-8") as out:
        for line in lines:
            q = json.loads(line)
            ddl = schema_to_ddl(read_sqlite_schema(DB_DIR / q["db_id"] / f"{q['db_id']}.sqlite"))
            user = build_prompt("sqlite", ddl, q["question"], hint=q.get("external_knowledge"))
            out.write(json.dumps({
                "id": q["id"], "db_id": q["db_id"], "dialect": "sqlite", "level": q["sql_complexity"],
                "style": q["question_style"], "gold_sql": q["sql"],
                "messages": [{"role": "system", "content": TEACHER_SYSTEM}, {"role": "user", "content": user}],
            }, ensure_ascii=False) + "\n")
            n += 1
    return n


def _tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_pretrained("Qwen/Qwen3.5-4B")


def _pct(values: list[int], p: float):
    values = sorted(values)
    return values[min(len(values) - 1, int(p * len(values)))] if values else None


def score(jobs_path: Path, runs: list[Path], questions: Path) -> dict:
    jobs = {j["id"]: j for j in map(json.loads, open(jobs_path, encoding="utf-8"))}
    synsql_cot = {q["id"]: q["cot"] for q in map(json.loads, open(questions, encoding="utf-8"))}
    tok = _tokenizer()
    n_tok = lambda s: len(tok.encode(s or "").ids)

    # "synsql_cot" is SynSQL's own trace, scored exactly like a teacher run
    candidates = {"synsql_cot": [{"id": i, "content": c, "reasoning": None} for i, c in synsql_cot.items() if i in jobs]}
    for run in runs:
        candidates[run.stem] = [json.loads(line) for line in open(run, encoding="utf-8")]

    report = {}
    for name, answers in candidates.items():
        records = [{
            "id": a["id"], "language": "en", "dialect": "sqlite", "slice": jobs[a["id"]]["level"],
            "db_id": jobs[a["id"]]["db_id"], "gold_sql": jobs[a["id"]]["gold_sql"],
            "pred_sql": extract_sql(a.get("content") or "") or "",
        } for a in answers if a["id"] in jobs]
        ev = Evaluator({"sqlite": SQLiteBackend(DB_DIR, 30)})
        ev.prefetch(records, workers=8)
        scored = [ev.score(r) for r in records]
        summary = summarize(scored)
        accepted = {s.id for s in scored if s.acc}
        reasoning = [n_tok(a.get("reasoning")) for a in answers if a.get("reasoning")]
        content = [n_tok(a.get("content")) for a in answers]
        trace = [n_tok(a.get("reasoning")) + n_tok(a.get("content")) for a in answers]
        cost = sum(a.get("cost_usd", 0.0) for a in answers)
        report[name] = {
            "answers": len(answers), "gold_errors": summary["overall"]["all"]["gold_errors"],
            "accept_pct": summary["overall"]["all"]["acc"],
            "accept_pct_by_level": {lvl: summary["slice"].get(lvl, {}).get("acc") for lvl in LEVELS},
            "cost_usd": round(cost, 4),
            "cost_per_1k_accepted_usd": round(1000 * cost / len(accepted), 3) if accepted and cost else None,
            "reasoning_tokens_qwen": {"p50": _pct(reasoning, .5), "p90": _pct(reasoning, .9), "max": max(reasoning, default=None)},
            "answer_tokens_qwen": {"p50": _pct(content, .5), "p90": _pct(content, .9), "max": max(content, default=None)},
            "trace_tokens_qwen": {"p50": _pct(trace, .5), "p90": _pct(trace, .9), "max": max(trace, default=None)},
            "cut_off": sum(a.get("finish_reason") == "length" for a in answers),
        }
        print(f"\n===== {name} =====")
        print(format_report({k: summary[k] for k in ("overall", "slice")}))
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    c = sub.add_parser("call")
    c.add_argument("--effort", choices=["low", "high", "max"], required=True)
    c.add_argument("--limit", type=int, default=0, help="only the first N jobs (for a cost check)")
    c.add_argument("--max-usd", type=float, default=1.0)
    c.add_argument("--workers", type=int, default=16)
    s = sub.add_parser("score")
    s.add_argument("--runs", nargs="*", type=Path, default=None, help="default: every teacher_*.jsonl")
    args = ap.parse_args(argv)

    questions, jobs_path = RUN_DIR / "questions.jsonl", RUN_DIR / "jobs.jsonl"
    if args.cmd == "prepare":
        print(f"wrote {prepare(questions, jobs_path)} jobs to {jobs_path}")
    elif args.cmd == "call":
        jobs = [json.loads(line) for line in open(jobs_path, encoding="utf-8")]
        if args.limit:
            jobs = jobs[:args.limit]
        teacher.run(jobs, RUN_DIR / f"teacher_{args.effort}.jsonl", thinking=True, effort=args.effort,
                    workers=args.workers, max_usd=args.max_usd)
    elif args.cmd == "score":
        runs = args.runs or sorted(RUN_DIR.glob("teacher_*.jsonl"))
        report = score(jobs_path, runs, questions)
        (RUN_DIR / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
        print("\n" + json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
