"""Score a generation run locally: extract SQL, execute it with the eval harness, report token lengths.

  python -m t2sql.score data/runs/bird_zs/prompts.jsonl data/runs/bird_zs/gen_think.jsonl --workers 8

Writes <generations>.harness.jsonl (harness input) and <generations>.report.json next to the generations.
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

from eval.harness import main as harness_main

from .prompt import extract_sql


def _pct(values: list[int], p: float):
    if not values:
        return None
    values = sorted(values)
    return values[min(len(values) - 1, int(p * len(values)))]


def length_report(prompts: dict, gens: list[dict]) -> dict:
    """Prompt and output token lengths, overall and per slice: the numbers behind the training length cap."""
    groups = defaultdict(lambda: {"prompt": [], "output": [], "cut_off": 0, "no_sql": 0, "outputs": 0})
    for g in gens:
        p = prompts[g["id"]]
        for key in ("all", f"{p['dialect']}/{p['slice']}"):
            grp = groups[key]
            grp["prompt"].append(g["prompt_tokens"])
            for o in g["outputs"]:
                grp["outputs"] += 1
                grp["output"].append(o["tokens"])
                grp["cut_off"] += o["finish_reason"] == "length"
                grp["no_sql"] += extract_sql(o["text"]) is None
    return {
        k: {
            "prompt_tokens": {"p50": _pct(v["prompt"], .5), "p99": _pct(v["prompt"], .99), "max": max(v["prompt"])},
            "output_tokens": {"p50": _pct(v["output"], .5), "p90": _pct(v["output"], .9),
                              "p99": _pct(v["output"], .99), "max": max(v["output"], default=0)},
            "cut_off_pct": round(100 * v["cut_off"] / max(v["outputs"], 1), 1),
            "no_sql_pct": round(100 * v["no_sql"] / max(v["outputs"], 1), 1),
        }
        for k, v in sorted(groups.items())
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("prompts", type=Path)
    ap.add_argument("generations", type=Path)
    ap.add_argument("--sqlite-dir", type=Path, default=Path("data/sqlite"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=60.0, help="seconds per query; one BIRD gold query needs ~20-40s")
    args = ap.parse_args(argv)

    prompts = {r["id"]: r for r in map(json.loads, open(args.prompts, encoding="utf-8"))}
    gens = [json.loads(line) for line in open(args.generations, encoding="utf-8") if line.strip()]

    harness_in = args.generations.with_suffix(".harness.jsonl")
    with open(harness_in, "w", encoding="utf-8") as fh:
        for g in gens:
            p = prompts[g["id"]]
            fh.write(json.dumps({
                "id": g["id"], "language": p["language"], "dialect": p["dialect"], "slice": p["slice"],
                "db_id": p["db_id"], "gold_sql": p["gold_sql"],
                "pred_sqls": [extract_sql(o["text"]) or "" for o in g["outputs"]] or [""],
            }, ensure_ascii=False) + "\n")

    lengths = length_report(prompts, gens)
    print("\n[token lengths]")
    for k, v in lengths.items():
        print(f"  {k:24} prompt p50/p99/max {v['prompt_tokens']['p50']}/{v['prompt_tokens']['p99']}/{v['prompt_tokens']['max']}"
              f"   output p50/p90/p99/max {v['output_tokens']['p50']}/{v['output_tokens']['p90']}/"
              f"{v['output_tokens']['p99']}/{v['output_tokens']['max']}   cut off {v['cut_off_pct']}%   no SQL {v['no_sql_pct']}%")

    report = args.generations.with_suffix(".report.json")
    harness_main([str(harness_in), "--sqlite-dir", str(args.sqlite_dir), "--workers", str(args.workers),
                  "--timeout", str(args.timeout), "--out", str(report)])
    full = json.loads(report.read_text(encoding="utf-8"))
    full["lengths"] = lengths
    report.write_text(json.dumps(full, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
