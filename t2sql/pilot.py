"""Plan and run the teacher stage for the 10k pilot.

  python -m t2sql.pilot plan                  # teacher_jobs.jsonl (PostgreSQL) and sqlite_reuse.jsonl (no calls)
  python -m t2sql.pilot probe --n 100         # first N teacher calls + a few translations: real cost numbers
  python -m t2sql.pilot call --max-usd 11     # the teacher run (resumes, never pays twice)
  python -m t2sql.pilot select                # score against gold, pick the pilot, assign languages: pilot_plan.jsonl
  python -m t2sql.pilot translate --max-usd 3 # Russian and Uzbek Latin for the plan: translations.jsonl (resumes)
  python -m t2sql.pilot build                 # the training file, with Uzbek Cyrillic and a leakage check: pilot_10k.jsonl
  python -m t2sql.pilot clean --questions 500 # drop meta talk and messy styles, keep N questions: pilot_clean_500.jsonl

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
import re
import zlib
from collections import Counter
from pathlib import Path

from eval.backends import PostgresBackend, SQLiteBackend
from eval.config import pg_config
from eval.harness import Evaluator, Scored, summarize

from . import teacher, translate
from .bird import ZERO_SHOT_SYSTEM
from .convert import OUT_DIR
from .leakage import find_leaks, load_eval_sets
from .prompt import build_prompt, extract_sql, read_sqlite_schema, schema_to_ddl
from .synsql import DB_DIR
from .teacher_test import TEACHER_SYSTEM
from .uz_cyrillic import russian_keyboard, to_uz_cyrillic

JOBS = OUT_DIR / "teacher_jobs.jsonl"
ANSWERS = OUT_DIR / "teacher_answers.jsonl"
SQLITE_REUSE = OUT_DIR / "sqlite_reuse.jsonl"
PLAN = OUT_DIR / "pilot_plan.jsonl"
TRANSLATIONS = OUT_DIR / "translations.jsonl"
PILOT_FILE = OUT_DIR / "pilot_10k.jsonl"
RUSSIAN_KEYBOARD_PCT = 30  # of Uzbek Cyrillic questions, typed without ў қ ғ ҳ
COMMON_FIELDS = ("id", "level", "style", "question", "hint", "source", "license", "schema_style", "dialect", "db_id",
                 "source_db", "gold_sql")
TARGETS = {"pg_messy": 1250, "pg_plain": 2000, "sqlite_plain": 1750}
LANGUAGE_SHARES = {"en": .28, "ru": .28, "uz_latin": .35, "uz_cyrillic": .09}
REFERENCE = re.compile(r"reference\s+(solution|query|sql)", re.IGNORECASE)
SQL_BLOCK = re.compile(r"```[^\S\n]*sql[^\S\n]*\n.*?```", re.IGNORECASE | re.DOTALL)
# Reasoning about the benchmark or the teaching setup instead of the question. It can also hit a real "benchmark"
# column; losing those few examples is cheaper than training on the rest.
META_TALK = re.compile(
    r"\b(gold (sql|query|answer|solution)|guess gold|benchmark|ground truth|grader|graded|test harness|likely generated"
    r"|generated (from|by|dataset|question)|dataset (generat|author|creator)|question (was|is) generated"
    r"|intended (sql|query|answer|solution)|expected (sql|query|answer|output|result)|evaluator|evaluated against"
    r"|will be compared|search memory|from memory|i recall|recall (that|the)|training example"
    r"|smaller (text-to-sql )?model)\b", re.IGNORECASE)
KEEP_STYLES = {"plain", "abbrev"}  # decided 2026-11-05: transliterated, Uzbek, Cyrillic and mixed names move out of v1
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


def score_texts(rows: list[dict], texts: list[str]) -> list[Scored]:
    """Accepted = the SQL in the text returns the same result as the row's gold, in the row's dialect."""
    records = [{
        "id": r["id"], "language": "en", "dialect": r["dialect"], "slice": r["level"], "db_id": r["db_id"],
        "gold_sql": r["gold_sql"], "pred_sql": extract_sql(text or "") or "",
    } for r, text in zip(rows, texts)]
    backends = {"sqlite": SQLiteBackend(DB_DIR, 15), "postgres": PostgresBackend(pg_config(), 15)}
    ev = Evaluator(backends)
    ev.prefetch(records, workers=8)
    scored = [ev.score(r) for r in records]
    for b in backends.values():
        b.close()
    return scored


def score_answers(jobs_by_id: dict[str, dict], answers: list[dict]) -> dict:
    return summarize(score_texts([jobs_by_id[a["id"]] for a in answers], [a.get("content") for a in answers]))


def clean_trace(trace: str) -> str | None:
    """A SynSQL trace ready to reuse: it ends at its final SQL block. None if it has no SQL block, or if it talks
    about a reference solution, which the model will never see."""
    if REFERENCE.search(trace):
        return None
    ends = [m.end() for m in SQL_BLOCK.finditer(trace)]
    return trace[:ends[-1]].strip() if ends else None


def pick_balanced(rows: list[dict], target: int, rng: random.Random) -> list[dict]:
    """Up to target rows, as equal across levels as supply allows: a level that runs out leaves its share to the rest."""
    pools: dict[str, list[dict]] = {lvl: [] for lvl in CALLS_PER_LEVEL}
    shuffled = list(rows)
    rng.shuffle(shuffled)
    for r in shuffled:
        pools[r["level"]].append(r)
    picked: list[dict] = []
    while len(picked) < target and any(pools.values()):
        for pool in pools.values():
            if pool and len(picked) < target:
                picked.append(pool.pop())
    return picked


def assign_languages(rows: list[dict], rng: random.Random) -> None:
    """Give each row two different languages, drawing from slots so that the group ends close to LANGUAGE_SHARES."""
    slots = {lang: share * 2 * len(rows) for lang, share in LANGUAGE_SHARES.items()}
    langs = list(slots)
    for r in rows:
        first = rng.choices(langs, weights=[max(slots[l], 1e-6) for l in langs])[0]
        others = [l for l in langs if l != first]
        second = rng.choices(others, weights=[max(slots[l], 1e-6) for l in others])[0]
        r["languages"] = sorted([first, second])
        slots[first] -= 1
        slots[second] -= 1


def drop_repeats(rows: list[dict], seen: set[str]) -> list[dict]:
    out = []
    for r in rows:
        key = " ".join(r["question"].lower().split())
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


def select(seed: int = 0) -> dict:
    """Score everything against gold, then pick the pilot's questions and their languages (no API calls)."""
    rng = random.Random(seed)
    jobs = {j["id"]: j for j in load_jobs()}
    answers = [a for a in map(json.loads, open(ANSWERS, encoding="utf-8")) if a["id"] in jobs]
    rows = [jobs[a["id"]] for a in answers]
    print(f"scoring {len(rows)} teacher answers", flush=True)
    teacher_scores = score_texts(rows, [a["content"] for a in answers])

    teacher_ok: list[dict] = []
    for job, a, s in zip(rows, answers, teacher_scores):
        if s.acc and s.gold_ok and not s.gold_empty and a["finish_reason"] == "stop":
            teacher_ok.append({
                **{k: job[k] for k in COMMON_FIELDS}, "prompt": job["messages"][1]["content"],
                "reasoning": a["reasoning"], "answer": a["content"], "target_sql": extract_sql(a["content"]),
                "trace_source": "deepseek-flash", "tokens": a["usage"]["total_tokens"],
            })

    reuse = [json.loads(line) for line in open(SQLITE_REUSE, encoding="utf-8")]
    cleaned = [(r, clean_trace(r["trace"])) for r in reuse]
    usable = [(r, t) for r, t in cleaned if t]
    print(f"scoring {len(usable)} of {len(reuse)} SynSQL traces", flush=True)
    trace_scores = score_texts([r for r, _ in usable], [t for _, t in usable])
    sqlite_ok = [{
        **{k: r[k] for k in COMMON_FIELDS}, "prompt": r["prompt"], "reasoning": None, "answer": t,
        "target_sql": extract_sql(t), "trace_source": "synsql",
    } for (r, t), s in zip(usable, trace_scores) if s.acc and s.gold_ok and not s.gold_empty]

    seen: set[str] = set()
    messy_ok = drop_repeats([r for r in teacher_ok if r["schema_style"] != "plain"], seen)
    plain_ok = drop_repeats([r for r in teacher_ok if r["schema_style"] == "plain"], seen)
    sqlite_ok = drop_repeats(sqlite_ok, seen)

    groups = {"pg_messy": pick_balanced(messy_ok, TARGETS["pg_messy"], rng),
              "pg_plain": pick_balanced(plain_ok, TARGETS["pg_plain"], rng),
              "sqlite_plain": pick_balanced(sqlite_ok, TARGETS["sqlite_plain"], rng)}
    pilot: list[dict] = []
    for name, group in groups.items():
        for lvl in CALLS_PER_LEVEL:  # shares hold inside each kind and level, not only overall
            stratum = [r for r in group if r["level"] == lvl]
            assign_languages(stratum, rng)
        pilot += group
    rng.shuffle(pilot)
    with open(PLAN, "w", encoding="utf-8") as fh:
        for r in pilot:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    def accept_by_level(rows_: list[dict], scores: list[Scored]) -> dict:
        tried, ok = Counter(r["level"] for r in rows_), Counter(r["level"] for r, s in zip(rows_, scores) if s.acc)
        return {lvl: f"{ok[lvl]}/{tried[lvl]} = {100 * ok[lvl] / max(tried[lvl], 1):.1f}%" for lvl in CALLS_PER_LEVEL}

    tokens = sorted(r["tokens"] for r in teacher_ok)
    report = {
        "teacher_answers": len(answers), "teacher_accepted": len(teacher_ok),
        "teacher_accept_by_level": accept_by_level(rows, teacher_scores),
        "teacher_pred_errors": sum("error" in s.pred_status for s in teacher_scores),
        "teacher_gold_errors": sum(not s.gold_ok for s in teacher_scores),
        "teacher_tokens_p50_p90_p99_max": [tokens[len(tokens) // 2], tokens[int(.9 * len(tokens))],
                                           tokens[int(.99 * len(tokens))], tokens[-1]],
        "traces": len(reuse), "traces_after_cleaning": len(usable), "traces_accepted": len(sqlite_ok),
        "trace_accept_by_level": accept_by_level([r for r, _ in usable], trace_scores),
        "available_after_repeats": {"pg_messy": len(messy_ok), "pg_plain": len(plain_ok), "sqlite_plain": len(sqlite_ok)},
        "picked": {name: {"total": len(g), **Counter(r["level"] for r in g)} for name, g in groups.items()},
        "picked_total": len(pilot), "examples": sum(len(r["languages"]) for r in pilot),
        "examples_by_language": dict(Counter(l for r in pilot for l in r["languages"])),
        "translations_needed": {"ru": sum("ru" in r["languages"] for r in pilot),
                                "uz": sum(bool({"uz_latin", "uz_cyrillic"} & set(r["languages"])) for r in pilot)},
    }
    (OUT_DIR / "select_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def translation_jobs() -> list[dict]:
    """Only what the plan uses: English needs no call, and Uzbek Cyrillic is converted from the Uzbek Latin text."""
    questions = []
    for r in map(json.loads, open(PLAN, encoding="utf-8")):
        to = [lang for lang, needs in (("ru", {"ru"}), ("uz", {"uz_latin", "uz_cyrillic"})) if needs & set(r["languages"])]
        questions.append({"question_id": r["id"], "question": r["question"], "db_id": r["source_db"],
                          "difficulty": r["level"], "source": r["source"], "license": r["license"], "split": "train",
                          "translate_to": to})
    return translate.jobs_for(questions)


def split_trace(trace: str) -> tuple[str, str]:
    """A cleaned SynSQL trace as (reasoning, answer): everything before the final SQL block, then that block."""
    start = [m.start() for m in SQL_BLOCK.finditer(trace)][-1]
    return trace[:start].strip(), trace[start:].strip()


def swap_question(prompt: str, english: str, translated: str) -> str:
    """The question is always the prompt's last part; schema and hint stay as they are."""
    tail = f"Question: {english}"
    if not prompt.endswith(tail):
        raise ValueError("prompt does not end with its question")
    return prompt[:-len(tail)] + f"Question: {translated}"


def question_in(row: dict, lang: str, translated: dict[tuple[str, str], dict]) -> tuple[str | None, str]:
    """The question text in one language and how it was typed, or (None, reason) when it cannot be used."""
    if lang == "en":
        return row["question"], "typed"
    rec = translated.get((row["id"], "ru" if lang == "ru" else "uz"))
    if rec is None or rec["finish_reason"] != "stop" or not (rec["content"] or "").strip():
        return None, "no translation"
    if translate.checks(rec):
        return None, "translation flagged"
    text = rec["content"].strip()
    if lang != "uz_cyrillic":
        return text, "typed"
    text = to_uz_cyrillic(text, row["question"])
    if zlib.crc32(row["id"].encode()) % 100 < RUSSIAN_KEYBOARD_PCT:
        return russian_keyboard(text), "russian_keyboard"
    return text, "typed"


def make_examples(row: dict, translated: dict[tuple[str, str], dict]) -> tuple[list[dict], Counter]:
    reasoning, answer = (row["reasoning"], row["answer"]) if row["trace_source"] != "synsql" else split_trace(row["answer"])
    out, dropped = [], Counter()
    for lang in row["languages"]:
        text, how = question_in(row, lang, translated)
        if text is None:
            dropped[how] += 1
            continue
        out.append({
            "id": f"{row['id']}-{lang}", "question_id": row["id"], "language": lang, "typing": how, "split": "train",
            **{k: row[k] for k in COMMON_FIELDS if k not in ("id", "question")},
            "question": text, "question_en": row["question"], "trace_source": row["trace_source"],
            "target_sql": row["target_sql"],
            "messages": [{"role": "system", "content": ZERO_SHOT_SYSTEM},
                         {"role": "user", "content": swap_question(row["prompt"], row["question"], text)}],
            "reasoning": reasoning, "answer": answer,
        })
    return out, dropped


def build() -> dict:
    translated = {(r["question_id"], r["language"]): r for r in map(json.loads, open(TRANSLATIONS, encoding="utf-8"))}
    sets = load_eval_sets()
    eval_dbs = set().union(*(s.databases for s in sets.values()))
    examples, dropped, leaks, shared_dbs = [], Counter(), Counter(), set()
    for row in map(json.loads, open(PLAN, encoding="utf-8")):
        made, why = make_examples(row, translated)
        dropped += why
        if row["source_db"].lower() in eval_dbs:
            shared_dbs.add(row["source_db"])
        for ex in made:
            found = find_leaks(ex, sets)
            if found:
                leaks.update(found)
                dropped["leak"] += 1
            else:
                examples.append(ex)
    with open(PILOT_FILE, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")

    def counts(key: str) -> dict:
        return dict(Counter(ex[key] for ex in examples))

    report = {
        "examples": len(examples), "questions": len({ex["question_id"] for ex in examples}),
        "dropped": dict(dropped), "leaks": dict(leaks), "databases_named_like_an_eval_database": sorted(shared_dbs),
        **{f"by_{key}": counts(key) for key in ("language", "dialect", "level", "schema_style", "trace_source", "typing",
                                                "source", "license")},
    }
    (OUT_DIR / "pilot_10k_report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    return report


def clean_examples(rows: list[dict], questions: int = 0, seed: int = 0) -> list[dict]:
    """Rows with usable reasoning and a kept schema style. questions > 0 keeps that many, all languages of each, at
    the planned dialect mix: cleaning removes only PostgreSQL rows, so a plain random pick would drift towards SQLite."""
    kept = [r for r in rows if r["schema_style"] in KEEP_STYLES and r["reasoning"] and not META_TALK.search(r["reasoning"])]
    if questions:
        rng = random.Random(seed)
        chosen: set[str] = set()
        for dialect, n in (("postgres", round(questions * PG_SHARE)), ("sqlite", questions - round(questions * PG_SHARE))):
            ids = sorted({r["question_id"] for r in kept if r["dialect"] == dialect})
            chosen.update(rng.sample(ids, min(n, len(ids))))
        kept = [r for r in kept if r["question_id"] in chosen]
    return kept


def clean(questions: int, seed: int = 0) -> dict:
    rows = [json.loads(line) for line in open(PILOT_FILE, encoding="utf-8")]
    kept = clean_examples(rows, questions, seed)
    out = OUT_DIR / (f"pilot_clean_{questions}.jsonl" if questions else "pilot_clean.jsonl")
    with open(out, "w", encoding="utf-8") as fh:
        for r in kept:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    return {"file": str(out), "examples": len(kept), "questions": len({r["question_id"] for r in kept}),
            **{key: dict(Counter(r[key] for r in kept)) for key in ("language", "dialect", "level", "schema_style")}}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    cl = sub.add_parser("clean")
    cl.add_argument("--questions", type=int, default=0)
    sub.add_parser("plan")
    p = sub.add_parser("probe")
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--translations", type=int, default=50)
    c = sub.add_parser("call")
    c.add_argument("--limit", type=int, default=0)
    c.add_argument("--max-usd", type=float, required=True)
    sub.add_parser("select")
    t = sub.add_parser("translate")
    t.add_argument("--limit", type=int, default=0)
    t.add_argument("--max-usd", type=float, required=True)
    sub.add_parser("build")
    args = ap.parse_args(argv)

    if args.cmd == "clean":
        print(json.dumps(clean(args.questions), indent=1))
        return 0
    if args.cmd == "build":
        print(json.dumps(build(), indent=1))
        return 0
    if args.cmd == "plan":
        print(json.dumps(plan(), indent=1))
        return 0
    if args.cmd == "select":
        print(json.dumps(select(), indent=1))
        return 0
    if args.cmd == "translate":
        jobs = translation_jobs()
        teacher.run(jobs[:args.limit] if args.limit else jobs, TRANSLATIONS, effort="high", max_usd=args.max_usd)
        records = [json.loads(line) for line in open(TRANSLATIONS, encoding="utf-8")]
        flags = Counter(flag.split(":")[0] for r in records for flag in translate.checks(r))
        print(f"{len(records)} translations, ${sum(r['cost_usd'] for r in records):.3f}, flags: {dict(flags)}")
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
