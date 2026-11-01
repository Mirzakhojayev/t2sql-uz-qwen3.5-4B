"""Call the teacher model (DeepSeek V4.1 Flash, direct API) with cost tracking, resume and a budget stop.

Every response is appended to a jsonl file the moment it arrives. Running the same command again skips the
ids already in that file, so an interrupted run never pays twice for the same call.

  python -m t2sql.teacher --ping          # one tiny call: checks the key, shows the response and usage fields

The API key is read from DEEPSEEK_API_KEY (environment or docker/.env) and never printed.
"""
import argparse
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from openai import APIConnectionError, APIStatusError, APITimeoutError, OpenAI, RateLimitError

from eval.config import secret

BASE_URL = "https://api.deepseek.com"
MODEL = "deepseek-flash"  # = DeepSeek-V4.1-Flash
# USD per 1M tokens at peak, from api-docs.deepseek.com/quick_start/pricing (checked 2026-10-31); off-peak is half
PRICE_PEAK = {"input_miss": 0.30, "input_hit": 0.006, "output": 1.20}


def is_off_peak(t: datetime) -> bool:
    """Peak is 01:00-04:00 and 06:00-10:00 UTC, Monday to Friday. (Chinese public holidays are also
    off-peak; not modelled, so the estimate errs on the expensive side.)"""
    t = t.astimezone(timezone.utc)
    if t.weekday() >= 5:
        return True
    return not (1 <= t.hour < 4 or 6 <= t.hour < 10)


def cost_usd(usage: dict, when: datetime) -> float:
    hit = usage.get("prompt_cache_hit_tokens") or 0
    miss = usage.get("prompt_cache_miss_tokens")
    if miss is None:
        miss = (usage.get("prompt_tokens") or 0) - hit
    out = usage.get("completion_tokens") or 0
    usd = (miss * PRICE_PEAK["input_miss"] + hit * PRICE_PEAK["input_hit"] + out * PRICE_PEAK["output"]) / 1e6
    return usd / 2 if is_off_peak(when) else usd


def client() -> OpenAI:
    return OpenAI(api_key=secret("DEEPSEEK_API_KEY"), base_url=BASE_URL, timeout=900, max_retries=0)


def ask(api: OpenAI, messages: list[dict], thinking: bool = True, effort: str = "high",
        max_tokens: int | None = None, retries: int = 6) -> dict:
    """One chat call with retries on rate limits, timeouts and server errors (exponential backoff)."""
    kwargs: dict = {"model": MODEL, "messages": messages,
                    "extra_body": {"thinking": {"type": "enabled" if thinking else "disabled"}}}
    if thinking:
        kwargs["reasoning_effort"] = effort
    if max_tokens:
        kwargs["max_tokens"] = max_tokens
    for attempt in range(retries):
        try:
            started = time.time()
            resp = api.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            extra = msg.model_extra or {}
            return {
                "reasoning": getattr(msg, "reasoning_content", None) or extra.get("reasoning_content"),
                "content": msg.content,
                "finish_reason": resp.choices[0].finish_reason,
                "usage": resp.usage.model_dump() if resp.usage else {},
                "seconds": round(time.time() - started, 1),
            }
        except (RateLimitError, APITimeoutError, APIConnectionError) as exc:
            wait = min(60, 2 ** attempt * 2)
            print(f"  retry in {wait}s after {type(exc).__name__}", flush=True)
            time.sleep(wait)
        except APIStatusError as exc:
            if exc.status_code < 500 or attempt == retries - 1:
                raise
            time.sleep(min(60, 2 ** attempt * 2))
    raise RuntimeError(f"gave up after {retries} attempts")


def run(jobs: list[dict], out_path: Path, thinking: bool = True, effort: str = "high", workers: int = 16,
        max_usd: float = 5.0, max_tokens: int | None = None) -> dict:
    """jobs: [{"id": ..., "messages": [...], <any other fields, copied to the output>}].

    Appends one json line per finished call to out_path. Stops starting new calls once the money spent in
    this file reaches max_usd (calls already in flight still finish, so it can overshoot a little).
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done, spent = set(), 0.0
    if out_path.exists():
        for line in open(out_path, encoding="utf-8"):
            rec = json.loads(line)
            done.add(rec["id"])
            spent += rec.get("cost_usd", 0.0)
    todo = [j for j in jobs if j["id"] not in done]
    print(f"{len(done)} already done (${spent:.3f} spent), {len(todo)} to go, budget ${max_usd:.2f}", flush=True)

    api = client()
    lock, stop = threading.Lock(), threading.Event()
    state = {"spent": spent, "finished": 0, "errors": 0}

    def work(job: dict) -> None:
        if stop.is_set():
            return
        try:
            res = ask(api, job["messages"], thinking=thinking, effort=effort, max_tokens=max_tokens)
        except Exception as exc:  # recorded nowhere on purpose: a rerun will try this id again
            with lock:
                state["errors"] += 1
            print(f"  {job['id']}: {type(exc).__name__}: {str(exc)[:150]}", flush=True)
            return
        when = datetime.now(timezone.utc)
        rec = {**{k: v for k, v in job.items() if k != "messages"}, "thinking": thinking,
               "effort": effort if thinking else None, **res,
               "cost_usd": round(cost_usd(res["usage"], when), 6), "at": when.isoformat()}
        with lock:
            with open(out_path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            state["spent"] += rec["cost_usd"]
            state["finished"] += 1
            if state["finished"] % 25 == 0:
                print(f"  {state['finished']}/{len(todo)} done, ${state['spent']:.3f} spent", flush=True)
            if state["spent"] >= max_usd and not stop.is_set():
                stop.set()
                print(f"  budget ${max_usd:.2f} reached: no new calls", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(work, todo))
    print(f"finished {state['finished']}, errors {state['errors']}, total spent in this file ${state['spent']:.3f}")
    return state


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ping", action="store_true", help="one tiny call to check the key and response format")
    args = ap.parse_args(argv)
    if args.ping:
        res = ask(client(), [{"role": "user", "content": "Write SQL to count rows in table t. Reply with only the SQL."}],
                  thinking=True, effort="low", max_tokens=2000)
        when = datetime.now(timezone.utc)
        print(json.dumps({**res, "cost_usd": cost_usd(res["usage"], when), "off_peak": is_off_peak(when)},
                         indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
