"""Execution-accuracy harness.

Input: jsonl, one record per line:
  id, language, dialect (sqlite|postgres), slice, db_id, gold_sql,
  and pred_sql (one string) or pred_sqls (list of N samples).

Metrics per group:
  acc     first prediction matches gold
  maj     majority vote over successful executions (same result = same vote) matches gold
  oracle  at least one of the N predictions matches gold
Records whose gold SQL fails to execute are counted separately and left out of every accuracy.

Usage: python -m eval.harness preds.jsonl --sqlite-dir data/sqlite --out report.json
"""
import argparse
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .backends import ExecResult, PostgresBackend, SQLiteBackend
from .compare import is_ordered, normalize_rows, signature
from .config import pg_config

DIALECT_ALIASES = {"postgresql": "postgres", "pg": "postgres"}


@dataclass
class Scored:
    id: str
    language: str
    dialect: str
    slice: str
    db_id: str
    gold_ok: bool
    gold_error: str | None = None
    gold_empty: bool = False
    n_preds: int = 0
    pred_status: list[str] = field(default_factory=list)  # ok | error | timeout, per prediction
    acc: bool = False
    maj: bool = False
    oracle: bool = False


def _preds(rec: dict) -> list[str]:
    if "pred_sqls" in rec:
        return list(rec["pred_sqls"])
    if "pred_sql" in rec:
        return [rec["pred_sql"]]
    raise ValueError(f"record {rec.get('id')!r} has neither pred_sql nor pred_sqls")


def majority_pick(sigs: list) -> object | None:
    """Most common signature among successful executions; ties go to the one seen first."""
    votes = Counter(s for s in sigs if s is not None)
    if not votes:
        return None
    first_seen = {}
    for i, s in enumerate(sigs):
        if s is not None:
            first_seen.setdefault(s, i)
    return max(votes, key=lambda s: (votes[s], -first_seen[s]))


class Evaluator:
    def __init__(self, backends: dict):
        self.backends = backends
        self._cache: dict[tuple, ExecResult] = {}

    def run(self, dialect: str, db_id: str, sql: str) -> ExecResult:
        key = (dialect, db_id, sql)
        if key not in self._cache:
            start = time.monotonic()
            res = self.backends[dialect].execute(db_id, sql)
            res.elapsed = time.monotonic() - start
            self._cache[key] = res
        return self._cache[key]

    def prefetch(self, records: list[dict], workers: int) -> None:
        """Execute every distinct (dialect, db, sql) once, in parallel; score() then reads the cache."""
        keys = {}
        for rec in records:
            dialect = DIALECT_ALIASES.get(rec["dialect"], rec["dialect"])
            for sql in [rec["gold_sql"], *_preds(rec)]:
                if sql and sql.strip():
                    keys[(dialect, rec["db_id"], sql)] = None
        todo = [k for k in keys if k not in self._cache]
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda k: self.run(*k), todo))  # list() re-raises setup errors, e.g. missing database

    def slowest(self, n: int = 5) -> list[tuple[float, str, str, str]]:
        top = sorted(self._cache.items(), key=lambda kv: kv[1].elapsed, reverse=True)[:n]
        return [(res.elapsed, d, db, " ".join(sql.split())[:90]) for (d, db, sql), res in top]

    def score(self, rec: dict) -> Scored:
        dialect = DIALECT_ALIASES.get(rec["dialect"], rec["dialect"])
        base = dict(
            id=rec["id"], language=rec["language"], dialect=dialect,
            slice=rec.get("slice", "main"), db_id=rec["db_id"],
        )
        gold = self.run(dialect, rec["db_id"], rec["gold_sql"])
        if not gold.ok:
            return Scored(**base, gold_ok=False, gold_error=gold.error)

        ordered = is_ordered(rec["gold_sql"], dialect)
        gold_sig = signature(normalize_rows(gold.rows), ordered)
        sigs, status = [], []
        for sql in _preds(rec):
            if not sql or not sql.strip():  # nothing extracted; SQLite would run "" as an empty, error-free result
                sigs.append(None)
                status.append("error")
                continue
            res = self.run(dialect, rec["db_id"], sql)
            if res.ok:
                sigs.append(signature(normalize_rows(res.rows), ordered))
                status.append("ok")
            else:
                sigs.append(None)
                status.append("timeout" if res.timed_out else "error")

        return Scored(
            **base, gold_ok=True, gold_empty=not gold.rows, n_preds=len(sigs), pred_status=status,
            acc=bool(sigs) and sigs[0] == gold_sig,
            maj=majority_pick(sigs) == gold_sig,
            oracle=gold_sig in sigs,
        )


def summarize(scored: list[Scored]) -> dict:
    def stats(items: list[Scored]) -> dict:
        valid = [s for s in items if s.gold_ok]
        n = len(valid)
        preds = [st for s in valid for st in s.pred_status]
        pct = lambda k: round(100 * sum(getattr(s, k) for s in valid) / n, 2) if n else None
        return {
            "n": n,
            "acc": pct("acc"), "maj": pct("maj"), "oracle": pct("oracle"),
            "pred_errors": preds.count("error"), "pred_timeouts": preds.count("timeout"),
            "gold_errors": len(items) - n,
            "gold_empty": sum(s.gold_empty for s in valid),
        }

    groups: dict[str, dict] = {"overall": {"all": stats(scored)}}
    for name, keyfn in [
        ("language", lambda s: s.language),
        ("dialect", lambda s: s.dialect),
        ("slice", lambda s: s.slice),
        ("language/dialect", lambda s: f"{s.language}/{s.dialect}"),
    ]:
        buckets = defaultdict(list)
        for s in scored:
            buckets[keyfn(s)].append(s)
        groups[name] = {k: stats(v) for k, v in sorted(buckets.items())}
    return groups


def format_report(summary: dict) -> str:
    header = f"{'':28}{'n':>6}{'acc':>8}{'maj':>8}{'oracle':>8}{'err':>6}{'t/o':>5}{'goldErr':>8}{'goldEmpty':>10}"
    lines = []
    for group, rows in summary.items():
        lines += ["", f"[{group}]", header]
        for name, s in rows.items():
            f = lambda v: "-" if v is None else f"{v:.1f}"
            lines.append(
                f"{name:28}{s['n']:>6}{f(s['acc']):>8}{f(s['maj']):>8}{f(s['oracle']):>8}"
                f"{s['pred_errors']:>6}{s['pred_timeouts']:>5}{s['gold_errors']:>8}{s['gold_empty']:>10}"
            )
    return "\n".join(lines)


def load_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("predictions", type=Path)
    ap.add_argument("--sqlite-dir", type=Path, default=Path("data/sqlite"))
    ap.add_argument("--timeout", type=float, default=10.0, help="seconds per query")
    ap.add_argument("--workers", type=int, default=4, help="queries executed in parallel")
    ap.add_argument("--out", type=Path, help="write the full report (summary + per-record) as json")
    args = ap.parse_args(argv)

    records = load_jsonl(args.predictions)
    backends = {"sqlite": SQLiteBackend(args.sqlite_dir, args.timeout)}
    if any(DIALECT_ALIASES.get(r["dialect"], r["dialect"]) == "postgres" for r in records):
        backends["postgres"] = PostgresBackend(pg_config(), args.timeout)

    ev = Evaluator(backends)
    started = time.monotonic()
    ev.prefetch(records, args.workers)
    scored = [ev.score(r) for r in records]
    summary = summarize(scored)
    print(format_report(summary))
    print(f"\nexecuted {len(ev._cache)} distinct queries in {time.monotonic() - started:.1f}s with {args.workers} workers")
    print("slowest:")
    for secs, dialect, db_id, sql in ev.slowest():
        print(f"  {secs:6.1f}s  {dialect:8} {db_id:24} {sql}")
    if args.out:
        args.out.write_text(
            json.dumps({"summary": summary, "records": [asdict(s) for s in scored]}, ensure_ascii=False, indent=1),
            encoding="utf-8",
        )
    for b in backends.values():
        b.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
