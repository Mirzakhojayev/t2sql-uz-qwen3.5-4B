"""Self-training data: Qwen answers each question several times, and its answers that return the right result
become the training examples. No teacher reasoning is copied (docs/data-rules.md: every model trained on DeepSeek's
traces thought less and scored lower than base).

  python -m t2sql.rft prompts --part 1 --questions 300      # data/pipeline/rft/prompts_part1.jsonl, 4 languages each
  modal run --detach modal/generate.py::main --prompts data/pipeline/rft/prompts_part1.jsonl --out data/pipeline/rft/answers_part1.jsonl --thinking --n 2 --max-tokens 12000 --shards 2
  python -m t2sql.rft synsql                                # plain PostgreSQL SynSQL questions of the pilot, already translated
  python -m t2sql.rft prompts --part 3 --questions 2286 --source synsql
  python -m t2sql.rft collect                               # every answers_part*.jsonl -> data/pipeline/rft_train.jsonl

SynSQL databases hold about 20 rows, so a match there proves little (docs/data-rules.md, rule 2): a SynSQL answer is
kept only if it also matches gold on the enlarged copies of t2sql/stress.py.

A question goes to exactly one part, in all its languages. Kept: answers whose thinking closed, whose SQL returns the
same result as the gold in the row's dialect on the full BIRD database, at most 2 distinct queries per question and
language. Prompts use the same system prompt as the eval (ZERO_SHOT_SYSTEM), so what is learned is what is tested.
"""
import argparse
import json
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval.backends import PostgresBackend, SQLiteBackend
from eval.compare import is_ordered, normalize_rows, signature
from eval.config import pg_config
from eval.harness import Evaluator

from .bird import ZERO_SHOT_SYSTEM
from .bird_train import db_dir
from .leakage import find_leaks, load_eval_sets
from .pilot import KEEP_STYLES, META_TALK, PLAN, question_in, swap_question
from .prompt import extract_sql
from .stress import check_database, verdict

OUT = Path("data/pipeline/rft")
QUESTIONS = OUT / "bird_questions.jsonl"
TRANSLATIONS = OUT / "translations.jsonl"
SYNSQL_QUESTIONS = OUT / "synsql_questions.jsonl"
SOURCES = {"bird": (QUESTIONS, TRANSLATIONS),
           "synsql": (SYNSQL_QUESTIONS, Path("data/pipeline/translations.jsonl"))}  # the pilot's translations
TRAIN_FILE = Path("data/pipeline/rft_train.jsonl")
LANGUAGES = ("en", "ru", "uz_latin", "uz_cyrillic")
KEEP_PER_PAIR = 2


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def split_answer(text: str) -> tuple[str, str] | None:
    """(reasoning, answer) of a finished answer, or None if the thinking never closed or there is no final SQL."""
    if "</think>" not in text:
        return None
    reasoning, answer = text.split("</think>", 1)
    reasoning, answer = reasoning.replace("<think>", "", 1).strip(), answer.strip()
    if not reasoning or extract_sql(answer) is None:
        return None
    return reasoning, answer


def sql_key(sql: str) -> str:
    return " ".join(sql.lower().split()).rstrip(";")


def synsql() -> dict:
    """The pilot's PostgreSQL questions with a kept schema style (plain, abbrev): gold already checked, and the
    pilot paid for their Russian and Uzbek translations. The teacher's answers are not used."""
    rows = [r for r in load_jsonl(PLAN) if r["dialect"] == "postgres" and r["schema_style"] in KEEP_STYLES]
    with open(SYNSQL_QUESTIONS, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({**{k: r[k] for k in ("id", "question", "hint", "source", "license", "schema_style",
                                                      "dialect", "db_id", "source_db", "gold_sql")},
                                 "messages": [{"role": "system", "content": ZERO_SHOT_SYSTEM},
                                              {"role": "user", "content": r["prompt"]}]}, ensure_ascii=False) + "\n")
    return {"file": str(SYNSQL_QUESTIONS), "questions": len(rows), "by_style": dict(Counter(r["schema_style"] for r in rows))}


def prompts(part: int, n_questions: int, source: str = "bird", seed: int = 0) -> dict:
    """Write the next part: n_questions not used by an earlier part, every language that has a usable text."""
    questions_path, translations_path = SOURCES[source]
    questions = load_jsonl(questions_path)
    translated = {(r["question_id"], r["language"]): r for r in load_jsonl(translations_path)}
    used = {r["id"].rsplit("-", 1)[0] for p in OUT.glob("prompts_part*.jsonl") if p.name != f"prompts_part{part}.jsonl"
            for r in load_jsonl(p)}
    free = [q for q in questions if q["id"] not in used]
    chosen = random.Random(f"{seed}-{part}").sample(free, min(n_questions, len(free)))
    out, skipped = OUT / f"prompts_part{part}.jsonl", Counter()
    n = 0
    with open(out, "w", encoding="utf-8") as fh:
        for q in chosen:
            english_prompt = q["messages"][1]["content"]
            for lang in LANGUAGES:
                text, how = question_in(q, lang, translated)
                if text is None:
                    skipped[how] += 1
                    continue
                fh.write(json.dumps({
                    "id": f"{q['id']}-{lang}", "question_id": q["id"], "language": lang, "typing": how,
                    "messages": [{"role": "system", "content": ZERO_SHOT_SYSTEM},
                                 {"role": "user", "content": swap_question(english_prompt, q["question"], text)}],
                }, ensure_ascii=False) + "\n")
                n += 1
    return {"file": str(out), "questions": len(chosen), "prompts": n, "skipped": dict(skipped),
            "questions_left_for_later_parts": len(free) - len(chosen)}


def collect(seed: int = 0) -> dict:
    questions = {q["id"]: q for path, _ in SOURCES.values() if path.exists() for q in load_jsonl(path)}
    prompts_by_id = {r["id"]: r for p in OUT.glob("prompts_part*.jsonl") for r in load_jsonl(p)}
    answers = [a for p in sorted(OUT.glob("answers_part*.jsonl")) for a in load_jsonl(p)]
    ev = Evaluator({"sqlite": SQLiteBackend(db_dir(), 30), "postgres": PostgresBackend(pg_config(), 60)})

    # run every distinct query once, in parallel, through the eval harness cache
    records = []
    for a in answers:
        q = questions[prompts_by_id[a["id"]]["question_id"]]
        for o in a["outputs"]:
            records.append({"dialect": q["dialect"], "db_id": q["db_id"], "gold_sql": q["gold_sql"],
                            "pred_sql": extract_sql(o["text"]) or ""})
    ev.prefetch(records, workers=8)

    why, kept, examples = Counter(), Counter(), []
    sets = load_eval_sets()
    for a in answers:
        p = prompts_by_id[a["id"]]
        q = questions[p["question_id"]]
        gold = ev.run(q["dialect"], q["db_id"], q["gold_sql"])
        ordered = is_ordered(q["gold_sql"], q["dialect"])
        seen: set[str] = set()
        for o in a["outputs"]:
            parts = split_answer(o["text"]) if o["finish_reason"] == "stop" else None
            if parts is None:
                why["unfinished or no SQL"] += 1
                continue
            sql = extract_sql(parts[1])
            res = ev.run(q["dialect"], q["db_id"], sql)
            if not (gold.ok and res.ok and signature(normalize_rows(res.rows), ordered) == signature(normalize_rows(gold.rows), ordered)):
                why["wrong result"] += 1
                continue
            if META_TALK.search(parts[0]):
                why["meta talk"] += 1
                continue
            if sql_key(sql) in seen or len(seen) >= KEEP_PER_PAIR:
                why["duplicate"] += 1
                continue
            seen.add(sql_key(sql))
            ex = {"id": f"{a['id']}-{len(seen)}", "question_id": q["id"], "language": p["language"], "typing": p["typing"],
                  "split": "train", "source": q["source"], "license": q["license"], "dialect": q["dialect"],
                  "schema_style": q["schema_style"], "db_id": q["db_id"], "source_db": q["source_db"],
                  "question": p["messages"][1]["content"].rsplit("Question: ", 1)[1], "question_en": q["question"],
                  "hint": q["hint"], "gold_sql": q["gold_sql"], "target_sql": sql, "trace_source": "qwen3.5-4b-self",
                  "messages": p["messages"], "reasoning": parts[0], "answer": parts[1]}
            examples.append(ex)

    # SynSQL: the answer must also match gold on enlarged copies of the database
    synsql_rows = [e for e in examples if e["source"] != "bird_train"]
    if synsql_rows:
        by_db: dict[str, list[dict]] = {}
        for e in synsql_rows:
            by_db.setdefault(e["db_id"], []).append({"question_id": e["id"], "level": "", "gold_sql": e["gold_sql"],
                                                     "target_sql": e["target_sql"]})
        with ThreadPoolExecutor(max_workers=6) as pool:
            checked = [r for found in pool.map(lambda item: check_database(*item), by_db.items()) for r in found]
        passed = {r["question_id"] for r in checked if verdict(r) == "same"}
        why["SynSQL: differs on enlarged data"] += len(synsql_rows) - len(passed)
        examples = [e for e in examples if e["source"] == "bird_train" or e["id"] in passed]

    clean = []
    for ex in examples:
        if find_leaks(ex, sets):
            why["leak"] += 1
            continue
        clean.append(ex)
        kept[ex["language"]] += 1
    examples = clean
    random.Random(seed).shuffle(examples)
    with open(TRAIN_FILE, "w", encoding="utf-8") as fh:
        for ex in examples:
            fh.write(json.dumps(ex, ensure_ascii=False) + "\n")
    pairs = len(answers)
    solved = len({(e["question_id"], e["language"]) for e in examples})
    return {"file": str(TRAIN_FILE), "answers": sum(len(a["outputs"]) for a in answers), "question_language_pairs": pairs,
            "pairs_solved": solved, "solved_pct": round(100 * solved / max(1, pairs), 1), "examples": len(examples),
            "questions": len({e["question_id"] for e in examples}), "by_language": dict(kept),
            "by_dialect": dict(Counter(e["dialect"] for e in examples)),
            "by_source": dict(Counter(e["source"] for e in examples)), "rejected": dict(why),
            "examples_per_pair": round(len(examples) / max(1, pairs), 2)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prompts")
    p.add_argument("--part", type=int, required=True)
    p.add_argument("--questions", type=int, required=True)
    p.add_argument("--source", choices=sorted(SOURCES), default="bird")
    sub.add_parser("synsql")
    sub.add_parser("collect")
    args = ap.parse_args(argv)
    result = (prompts(args.part, args.questions, args.source) if args.cmd == "prompts" else
              synsql() if args.cmd == "synsql" else collect())
    print(json.dumps(result, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
