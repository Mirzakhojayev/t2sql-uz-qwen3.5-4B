"""Self-correction: run each answer's SQL, and give the model its error back for another try.

  python -m t2sql.retry make data/eval/baseline_prompts.jsonl data/runs/gold_roundtrip/roundtrip_think.jsonl --out data/runs/retry/round1_prompts.jsonl
  modal run --detach modal/generate.py::main --prompts data/runs/retry/round1_prompts.jsonl --out data/runs/retry/round1.jsonl --thinking --shards 2 --max-tokens 12000
  python -m t2sql.retry merge data/runs/gold_roundtrip/roundtrip_think.jsonl data/runs/retry/round1.jsonl --out data/runs/retry/after1.jsonl
  python -m t2sql.score data/eval/baseline_prompts.jsonl data/runs/retry/after1.jsonl

Feedback is what a deployed model would see when it can run its own query: the database error, an empty result, or
nothing to run. Gold SQL is never used; only the query's own execution. An empty result counts as a failure because
a user who asks a question expects rows; this is stated in the write-up.

Only the answer after </think> goes back into the conversation: the model's reasoning stays out of the history, as
the Qwen chat template does for earlier turns.
"""
import argparse
import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from eval.backends import PostgresBackend, SQLiteBackend
from eval.config import pg_config

from .prompt import extract_sql

NO_ANSWER = "I ran out of space before writing the final query."


def answer_part(text: str) -> str:
    """The part after the thinking; empty if the thinking never closed."""
    return text.split("</think>", 1)[1].strip() if "</think>" in text else ""


def feedback(text: str, ok: bool | None, error: str | None, rows: int | None) -> str | None:
    """The user's next message, or None when the answer ran and returned rows."""
    if extract_sql(text) is None:
        return "There is no final SQL query in your answer. Give the final query in a ```sql code block; keep the reasoning short."
    if ok is False:
        return (f"Running your query gave this error:\n{error}\n\n"
                "Fix the query. Use only tables and columns from the schema, and give the corrected query in a ```sql code block.")
    if rows == 0:
        return ("Your query ran but returned no rows. Check the filters, the values and the joins against the schema "
                "and the example values, then give the corrected query in a ```sql code block.")
    return None


def retry_messages(prompt_messages: list[dict], text: str, note: str) -> list[dict]:
    return [*prompt_messages, {"role": "assistant", "content": answer_part(text) or NO_ANSWER}, {"role": "user", "content": note}]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in open(path, encoding="utf-8") if line.strip()]


def make(prompts_path: Path, gens_path: Path, out: Path, workers: int = 8) -> dict:
    prompts = {p["id"]: p for p in load_jsonl(prompts_path)}
    gens = load_jsonl(gens_path)
    backends = {"sqlite": SQLiteBackend(Path("data/sqlite"), 30), "postgres": PostgresBackend(pg_config(), 30)}

    def check(g: dict) -> tuple[dict, str | None]:
        p = prompts[g["id"]]
        text = g["outputs"][0]["text"] if g["outputs"] else ""
        sql = extract_sql(text)
        if sql is None:
            return g, feedback(text, None, None, None)
        res = backends["sqlite" if p["dialect"] == "sqlite" else "postgres"].execute(p["db_id"], sql)
        return g, feedback(text, res.ok, res.error, len(res.rows) if res.ok else None)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        checked = list(ex.map(check, gens))
    for b in backends.values():
        b.close()
    kinds = Counter()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        for g, note in checked:
            if note is None:
                kinds["fine"] += 1
                continue
            kinds["no SQL" if note.startswith("There is no") else "error" if note.startswith("Running") else "no rows"] += 1
            p = prompts[g["id"]]
            msgs = g.get("messages") or p["messages"]
            text = g["outputs"][0]["text"] if g["outputs"] else ""
            fh.write(json.dumps({"id": g["id"], "messages": retry_messages(msgs, text, note)}, ensure_ascii=False) + "\n")
    return {"answers": len(gens), **kinds, "to_retry": sum(v for k, v in kinds.items() if k != "fine")}


def merge(previous: Path, retried: Path, out: Path, retry_prompts: Path | None = None) -> dict:
    """The previous answers, with each retried one replaced by its new answer. Tokens add up over rounds."""
    new = {g["id"]: g for g in load_jsonl(retried)}
    convo = {r["id"]: r["messages"] for r in load_jsonl(retry_prompts)} if retry_prompts else {}
    replaced = 0
    with open(out, "w", encoding="utf-8") as fh:
        for g in load_jsonl(previous):
            if g["id"] in new and new[g["id"]]["outputs"]:
                spent = sum(o["tokens"] for o in g["outputs"][:1])
                g = {**new[g["id"]], "rounds": g.get("rounds", 1) + 1,
                     "tokens_all_rounds": g.get("tokens_all_rounds", spent) + new[g["id"]]["outputs"][0]["tokens"]}
                if g["id"] in convo:
                    g["messages"] = convo[g["id"]]
                replaced += 1
            fh.write(json.dumps(g, ensure_ascii=False) + "\n")
    return {"replaced": replaced}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("make")
    m.add_argument("prompts", type=Path)
    m.add_argument("generations", type=Path)
    m.add_argument("--out", type=Path, required=True)
    g = sub.add_parser("merge")
    g.add_argument("previous", type=Path)
    g.add_argument("retried", type=Path)
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--retry-prompts", type=Path)
    args = ap.parse_args(argv)
    result = (make(args.prompts, args.generations, args.out) if args.cmd == "make"
              else merge(args.previous, args.retried, args.out, args.retry_prompts))
    print(json.dumps(result, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
