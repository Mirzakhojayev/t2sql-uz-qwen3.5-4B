"""Translate English text-to-SQL questions with the teacher model (DeepSeek).

  python -m t2sql.translate sample --n 50 --out-dir data/runs/translate_sample

Writes translations.jsonl (resumable, like every teacher run) and review.md, a side-by-side file for a human
to read. Names, numbers and quoted values are kept exactly: the SQL has to match them in the database.
"""
import argparse
import json
import random
import re
from pathlib import Path

from . import teacher

BIRD_FILE = Path("data/bird_minidev/data/mini_dev_sqlite.jsonl")

LANGUAGES = {
    "uz": "Uzbek, modern Latin script (use o' and g' as Uzbek writers normally type them)",
    "ru": "Russian",
}

SYSTEM = (
    "You translate database questions from English into {language}. Write the way a local analyst would "
    "naturally type the question to a colleague, not a word-for-word translation. Keep exactly as written: "
    "quoted values, codes, numbers, dates, and proper names that are stored in a database. "
    "Do not answer the question or add anything. Reply with only the translation."
)


def jobs_for(questions: list[dict]) -> list[dict]:
    return [{
        "id": f"{lang}-{q['question_id']}", "question_id": q["question_id"], "language": lang,
        "db_id": q["db_id"], "difficulty": q["difficulty"], "english": q["question"],
        "source": q["source"], "license": q["license"], "split": q["split"],
        "messages": [{"role": "system", "content": SYSTEM.format(language=desc)},
                     {"role": "user", "content": q["question"]}],
    } for q in questions for lang, desc in LANGUAGES.items()]


def checks(rec: dict) -> list[str]:
    """Cheap automatic flags for the human reviewer; they do not judge translation quality."""
    text, flags = rec["content"] or "", []
    if rec["language"] == "uz" and re.search(r"[а-яА-ЯёЁ]", text):
        flags.append("Cyrillic in Uzbek Latin")
    if rec["language"] == "ru" and not re.search(r"[а-яА-ЯёЁ]", text):
        flags.append("no Cyrillic in Russian")
    missing = [n for n in re.findall(r"\d+(?:\.\d+)?", rec["english"]) if n not in text]
    if missing:
        flags.append(f"numbers missing: {missing}")
    # an opening quote follows a non-letter, so the apostrophes in "I'm" and "club's" do not count as quotes
    quoted = [v for pair in re.findall(r"(?<!\w)'([^']+)'(?!\w)|\"([^\"]+)\"", rec["english"]) for v in pair if v]
    lost = [v for v in quoted if v not in text]
    if lost:
        flags.append(f"quoted values changed: {lost}")
    return flags


def write_review(records: list[dict], path: Path) -> int:
    by_q: dict = {}
    for r in records:
        by_q.setdefault(r["question_id"], {})[r["language"]] = r
    flagged = 0
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# Translation review\n\nRead the Uzbek and Russian. Does it sound natural? Is the meaning the same?\n")
        for qid, langs in sorted(by_q.items(), key=lambda kv: int(kv[0])):
            first = next(iter(langs.values()))
            fh.write(f"\n## {qid} ({first['db_id']}, {first['difficulty']})\n\n**EN:** {first['english']}\n\n")
            for lang in LANGUAGES:
                if lang in langs:
                    fh.write(f"**{lang.upper()}:** {(langs[lang]['content'] or '').strip()}\n\n")
                    flags = checks(langs[lang])
                    if flags:
                        flagged += 1
                        fh.write(f"> check: {'; '.join(flags)}\n\n")
    return flagged


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=50)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out-dir", type=Path, required=True)
    s.add_argument("--max-usd", type=float, default=0.25)
    args = ap.parse_args(argv)

    questions = [json.loads(line) for line in open(BIRD_FILE, encoding="utf-8")]
    picked = random.Random(args.seed).sample(questions, args.n)
    for q in picked:
        q.update(source="bird_minidev", license="CC BY-SA 4.0", split="eval")
    out = args.out_dir / "translations.jsonl"
    teacher.run(jobs_for(picked), out, thinking=True, effort="high", workers=8, max_usd=args.max_usd)
    records = [json.loads(line) for line in open(out, encoding="utf-8")]
    flagged = write_review(records, args.out_dir / "review.md")
    print(f"{len(records)} translations, {flagged} flagged, review file: {args.out_dir / 'review.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
