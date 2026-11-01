"""Run DeepSeek on BIRD Mini-Dev with the same prompts as the Qwen baseline, to compare teacher and student.

EVAL DATA: these answers are for scoring only and must never become training examples.

  python -m t2sql.bird_teacher --effort high --max-usd 3
  python -m t2sql.score data/runs/bird_zs/prompts.jsonl data/runs/bird_deepseek/gen_high.jsonl
"""
import argparse
import json
from pathlib import Path

from . import teacher

PROMPTS = Path("data/runs/bird_zs/prompts.jsonl")
OUT_DIR = Path("data/runs/bird_deepseek")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--effort", choices=["low", "high", "max"], default="high")
    ap.add_argument("--max-usd", type=float, default=3.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    jobs = [json.loads(line) for line in open(PROMPTS, encoding="utf-8")]
    if args.limit:
        jobs = jobs[:args.limit]
    raw = OUT_DIR / f"raw_{args.effort}.jsonl"
    teacher.run(jobs, raw, thinking=True, effort=args.effort, workers=args.workers, max_usd=args.max_usd)

    # same shape as modal/generate.py, so t2sql.score works unchanged; only the answer text is scored
    gen = OUT_DIR / f"gen_{args.effort}.jsonl"
    with open(gen, "w", encoding="utf-8") as out:
        for line in open(raw, encoding="utf-8"):
            r = json.loads(line)
            out.write(json.dumps({
                "id": r["id"], "prompt_tokens": r["usage"].get("prompt_tokens", 0),
                "outputs": [{"text": r["content"] or "", "tokens": r["usage"].get("completion_tokens", 0),
                             "finish_reason": r["finish_reason"]}],
            }, ensure_ascii=False) + "\n")
    print(f"wrote {gen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
