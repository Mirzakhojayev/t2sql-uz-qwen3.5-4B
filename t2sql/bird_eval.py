"""Uzbek and Russian versions of BIRD Mini-Dev. EVAL DATA: never use these for training.

  python -m t2sql.bird_eval --max-usd 1

Writes under data/eval/:
  translations.jsonl          every translation (resumes; the batch already reviewed by hand is reused)
  bird_ru.jsonl               500 questions x PostgreSQL + SQLite
  bird_uz.jsonl               same, Uzbek Latin
  bird_uz_cyrillic.jsonl      100 questions x PostgreSQL + SQLite, converted from the Uzbek Latin text
  baseline_prompts.jsonl      the PostgreSQL rows of all three: what the G3 runs generate on
  review.md                   side-by-side file for the person checking them

Each row starts as verified=false and carries its automatic flags. Flagged translations are kept: every language
must be scored on the same questions. Hints stay in English, as in training.
"""
import argparse
import json
import random
import zlib
from pathlib import Path

from . import teacher, translate
from .bird import BIRD_DIR
from .pilot import RUSSIAN_KEYBOARD_PCT, swap_question
from .uz_cyrillic import russian_keyboard, to_uz_cyrillic

OUT = Path("data/eval")
ENGLISH_PROMPTS = Path("data/runs/bird_zs/prompts.jsonl")
REVIEWED = Path("data/runs/translate_sample/translations.jsonl")
CYRILLIC_QUESTIONS = 100


def question_rows() -> list[dict]:
    rows = [json.loads(line) for line in open(BIRD_DIR / "mini_dev_postgresql.jsonl", encoding="utf-8")]
    return [{"question_id": r["question_id"], "question": r["question"], "db_id": r["db_id"],
             "difficulty": r["difficulty"], "source": "bird_minidev", "license": "CC BY-SA 4.0", "split": "eval"}
            for r in rows]


def seed_with_reviewed(out: Path, wanted: set[str], reviewed: Path = REVIEWED) -> int:
    """Copy translations that were already paid for and read by hand, so teacher.run skips them."""
    done = {json.loads(line)["id"] for line in open(out, encoding="utf-8")} if out.exists() else set()
    added = 0
    with open(out, "a", encoding="utf-8") as fh:
        for line in open(reviewed, encoding="utf-8"):
            rec = json.loads(line)
            if rec["id"] in wanted and rec["id"] not in done:
                fh.write(json.dumps({**rec, "cost_usd": 0.0}, ensure_ascii=False) + "\n")  # paid in the earlier run
                added += 1
    return added


def drop_cut_off(out: Path) -> int:
    """Remove empty or truncated answers, so that teacher.run asks for them again."""
    if not out.exists():
        return 0
    records = [json.loads(line) for line in open(out, encoding="utf-8")]
    keep = [r for r in records if not translate.looks_cut_off(r["english"], r["content"] or "")]
    write_jsonl(out, keep)
    return len(records) - len(keep)


def eval_row(english: dict, qid: int, language: str, text: str, typing: str, flags: list[str]) -> dict:
    user = english["messages"][1]["content"]
    question = user.rsplit("Question: ", 1)[1]
    return {
        **{k: english[k] for k in ("db_id", "dialect", "slice", "gold_sql", "source", "license", "split")},
        "id": f"{english['id']}-{language}", "question_id": qid, "language": language, "typing": typing,
        "question": text, "question_en": question, "verified": False, "flags": flags,
        "messages": [english["messages"][0], {"role": "user", "content": swap_question(user, question, text)}],
    }


def build_rows(english_prompts: list[dict], translated: dict[tuple[int, str], dict], cyrillic_ids: set[int]) -> dict:
    files: dict[str, list[dict]] = {"ru": [], "uz": [], "uz_cyrillic": []}
    for english in english_prompts:
        qid = int(english["id"].split("-")[1])
        for lang in ("ru", "uz"):
            rec = translated[(qid, lang)]
            text, flags = rec["content"].strip(), translate.checks(rec)
            files[lang].append(eval_row(english, qid, "uz_latin" if lang == "uz" else lang, text, "typed", flags))
            if lang == "uz" and qid in cyrillic_ids:
                cyr = to_uz_cyrillic(text, rec["english"])
                keyboard = zlib.crc32(str(qid).encode()) % 100 < RUSSIAN_KEYBOARD_PCT
                files["uz_cyrillic"].append(eval_row(english, qid, "uz_cyrillic", russian_keyboard(cyr) if keyboard else cyr,
                                                     "russian_keyboard" if keyboard else "typed", flags))
    return files


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-usd", type=float, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    OUT.mkdir(parents=True, exist_ok=True)
    out = OUT / "translations.jsonl"
    questions = question_rows()
    jobs = translate.jobs_for(questions)
    print(f"reused {seed_with_reviewed(out, {j['id'] for j in jobs})} translations already reviewed", flush=True)
    print(f"asking again for {drop_cut_off(out)} cut-off translations", flush=True)
    teacher.run(jobs, out, effort="high", max_usd=args.max_usd)

    records = [json.loads(line) for line in open(out, encoding="utf-8")]
    translated = {(r["question_id"], r["language"]): r for r in records}
    missing = [j["id"] for j in jobs if (j["question_id"], j["language"]) not in translated]
    if missing:
        raise SystemExit(f"{len(missing)} translations missing, run again: {missing[:5]}")

    english = [json.loads(line) for line in open(ENGLISH_PROMPTS, encoding="utf-8")]
    ids = sorted(q["question_id"] for q in questions)
    cyrillic_ids = set(random.Random(args.seed).sample(ids, CYRILLIC_QUESTIONS))
    files = build_rows(english, translated, cyrillic_ids)
    for lang, rows in files.items():
        write_jsonl(OUT / f"bird_{lang}.jsonl", rows)
        print(f"bird_{lang}.jsonl: {len(rows)} rows, flagged {sum(bool(r['flags']) for r in rows)}")
    baseline = [r for rows in files.values() for r in rows if r["dialect"] == "postgres"]
    write_jsonl(OUT / "baseline_prompts.jsonl", baseline)
    print(f"baseline_prompts.jsonl: {len(baseline)} PostgreSQL prompts")
    translate.write_review(records, OUT / "review.md")
    print(f"cost so far ${sum(r['cost_usd'] for r in records):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
