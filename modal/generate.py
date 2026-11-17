"""Generate answers with vLLM on Modal. The GPU side only generates; scoring happens locally (eval/harness).

Input: jsonl, one record per line with at least `id` and `messages` (chat messages, e.g. system + user).
Output: jsonl with id, prompt_tokens and outputs [{text, tokens, finish_reason}], plus <out>.meta.json.

  # ALWAYS use --detach: the GPUs keep running if your laptop disconnects (a dropped connection once killed a run)
  modal run --detach modal/generate.py::main --prompts data/runs/bird_zs/prompts.jsonl --out data/runs/bird_zs/gen_think.jsonl --thinking --shards 4
  modal run --detach modal/generate.py::main --prompts data/runs/bird_zs/prompts.jsonl --out data/runs/bird_zs/gen_nothink.jsonl --no-thinking

Each GPU writes its results to the Modal volume (runs/<name>/shard_<k>.json) and commits. The command waits and
merges them into --out. If it gets interrupted, the GPUs carry on; fetch the finished work later with:
  modal run modal/generate.py::collect --out data/runs/bird_zs/gen_think.jsonl

In Git Bash, prefix commands that pass --model-dir with MSYS_NO_PATHCONV=1: it rewrites /cache/... into a Windows path.

Sampling follows the Qwen3.5 card's "precise coding" settings (temperature 0.6 thinking / 0.7 non-thinking,
top_p 0.95 / 0.8, top_k 20), except presence_penalty is 0 in both modes: SQL must repeat table and column
names, which a presence penalty discourages. The function timeout (60 min; about $2.5 on an A100 80GB, $4 on an
H100) is the cost cap per GPU.
"""
import json
import os
import time
from pathlib import Path

import modal

app = modal.App("t2sql-generate")
vol = modal.Volume.from_name("t2sql-cache", create_if_missing=True)
CACHE = "/cache"
MODEL_DIR = f"{CACHE}/models/Qwen3.5-4B"
MAX_MODEL_LEN = 49152  # about 10k prompt + 32k thinking + margin
FUNCTION_TIMEOUT_S = 3600  # 60 min: a 45-min cap cut off the larger final-run shards
GIVE_UP_AFTER_S = FUNCTION_TIMEOUT_S + 900  # image build and queueing on top; past this no GPU can still be working

# The model card asks for vLLM from the nightly index. The CUDA backend must be named explicitly: image builds
# run on a machine without a GPU, where --torch-backend=auto silently installs CPU-only torch (found 2026-10-30).
# Modal's GPU driver reports CUDA 13.3, so the cu130 build of vLLM and torch is the matching pair.
vllm_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("uv>=0.9")  # Modal's base image ships uv 0.8, which rejects --torch-backend=cu130 (found 2026-11-16)
    .run_commands(
        "uv pip install --system vllm --torch-backend=cu130 --extra-index-url https://wheels.vllm.ai/nightly/cu130"
    )
)


# whichever GPU is free first: A100-80GB queues sometimes ran 15+ minutes. Results do not depend on the GPU.
@app.function(image=vllm_image, gpu=["H100", "A100-80GB"], volumes={CACHE: vol}, timeout=FUNCTION_TIMEOUT_S)
def generate(records: list[dict], thinking: bool, n: int, max_tokens: int, name: str, shard: int, shards: int,
             model_dir: str = MODEL_DIR) -> str:
    try:
        return _generate(records, thinking, n, max_tokens, name, shard, shards, model_dir)
    except Exception as exc:  # the waiting side only reads the volume, so the failure has to be written there
        os.makedirs(f"{CACHE}/runs/{name}", exist_ok=True)
        with open(f"{CACHE}/runs/{name}/error_{shard}.txt", "w", encoding="utf-8") as fh:
            fh.write(f"{type(exc).__name__}: {exc}"[:2000])
        vol.commit()
        raise


def _generate(records: list[dict], thinking: bool, n: int, max_tokens: int, name: str, shard: int, shards: int,
              model_dir: str) -> str:
    import torch
    import vllm
    from vllm import LLM, SamplingParams

    vol.reload()
    started = time.time()
    llm = LLM(model=model_dir, max_model_len=MAX_MODEL_LEN, limit_mm_per_prompt={"image": 0, "video": 0},
              gpu_memory_utilization=0.9, seed=0)
    tok = llm.get_tokenizer()
    load_s = time.time() - started

    texts = [tok.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=True,
                                     enable_thinking=thinking) for r in records]
    lengths = [len(tok(t, add_special_tokens=False).input_ids) for t in texts]
    # a prompt that leaves no room to answer would make vLLM reject the whole batch; skip and report it
    ok = [i for i, n_tok in enumerate(lengths) if n_tok + 256 < MAX_MODEL_LEN]

    params = SamplingParams(
        n=n, max_tokens=max_tokens, seed=0, top_k=20, presence_penalty=0.0,
        temperature=0.6 if thinking else 0.7, top_p=0.95 if thinking else 0.8,
        stop_token_ids=[tok.convert_tokens_to_ids(t) for t in ("<|im_end|>", "<|endoftext|>")],
    )
    t = time.time()
    outs = llm.generate([texts[i] for i in ok], params)
    gen_s = time.time() - t

    by_index = dict(zip(ok, outs))
    results = []
    for i, r in enumerate(records):
        res = {"id": r["id"], "prompt_tokens": lengths[i]}
        if i in by_index:
            # In thinking mode the chat template ends the PROMPT with "<think>\n", so the model's text starts
            # inside the thinking and only contains the closing tag. Put the opening tag back so every saved
            # text is self-contained: "<think>\n...</think>\n\nanswer" (or unclosed if it ran out of tokens).
            prefix = "<think>\n" if texts[i].endswith("<think>\n") else ""
            res["outputs"] = [{"text": prefix + c.text, "tokens": len(c.token_ids), "finish_reason": c.finish_reason}
                              for c in by_index[i].outputs]
        else:
            res["outputs"], res["error"] = [], "prompt too long"
        results.append(res)

    out_tokens = sum(o["tokens"] for r in results for o in r["outputs"])
    meta = {
        "model_dir": model_dir, "vllm": vllm.__version__, "gpu": torch.cuda.get_device_name(0), "thinking": thinking, "n": n,
        "max_tokens": max_tokens, "temperature": params.temperature, "top_p": params.top_p, "top_k": 20,
        "presence_penalty": 0.0, "prompts": len(records), "skipped_too_long": len(records) - len(ok),
        "output_tokens": out_tokens, "load_seconds": round(load_s), "generate_seconds": round(gen_s),
        "name": name, "shard": shard, "shards": shards,
    }
    print(json.dumps(meta, indent=1), flush=True)

    path = f"{CACHE}/runs/{name}/shard_{shard}.json"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "results": results}, fh, ensure_ascii=False)
    vol.commit()  # from here on the results are safe, whatever happens to the client
    return path


# ---- local side: start the shards, wait for their files in the volume, merge ------------------------------------

def _read_text(path: str) -> str:
    return b"".join(vol.read_file(path)).decode("utf-8")


def _read_shard(path: str) -> dict:
    return json.loads(_read_text(path))


def _collect(name: str, out: str, shards: int, wait: bool, poll_s: int = 30) -> bool:
    """Merge runs/<name>/shard_*.json from the volume into `out`. shards=0: read the count from the files."""
    found: dict[int, dict] = {}
    started = time.time()
    while True:
        if wait and time.time() - started > GIVE_UP_AFTER_S:
            raise SystemExit(f"gave up after {GIVE_UP_AFTER_S // 60} min with {len(found)} shards: see `modal app logs`")
        try:
            for e in vol.listdir(f"runs/{name}"):
                base = os.path.basename(e.path)
                if base.startswith("error_"):
                    raise SystemExit(f"{base}: {_read_text(e.path)}")
                if base.startswith("shard_") and base.endswith(".json"):
                    k = int(base[len("shard_"):-len(".json")])
                    if k not in found:
                        found[k] = _read_shard(e.path)
        except Exception as exc:  # a network blip must not lose the run; the GPUs are unaffected
            print(f"  (could not read the volume, will retry: {type(exc).__name__}: {str(exc)[:100]})")
        expected = shards or (next(iter(found.values()))["meta"]["shards"] if found else 0)
        print(f"  {len(found)}/{expected or '?'} shards finished", flush=True)
        if expected and len(found) >= expected:
            break
        if not wait:
            return False
        time.sleep(poll_s)

    n_total = sum(len(f["results"]) for f in found.values())
    ordered = [None] * n_total
    for k, f in found.items():
        for pos, r in enumerate(f["results"]):
            ordered[pos * expected + k] = r  # shard k holds records[k::expected]
    metas = [found[k]["meta"] for k in sorted(found)]
    meta = {
        **metas[0], "shards": expected, "prompts": n_total,
        "skipped_too_long": sum(m["skipped_too_long"] for m in metas),
        "output_tokens": sum(m["output_tokens"] for m in metas),
        "load_seconds": max(m["load_seconds"] for m in metas),
        "generate_seconds": max(m["generate_seconds"] for m in metas),
        "gpu_seconds_total": sum(m["load_seconds"] + m["generate_seconds"] for m in metas),
        "shard_metas": metas,
    }
    meta.pop("shard", None)
    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        for r in ordered:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    out_path.with_suffix(".meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(f"wrote {out_path} and {out_path.with_suffix('.meta.json')}")
    print(f"total GPU time {meta['gpu_seconds_total'] / 3600:.2f} h -> about ${meta['gpu_seconds_total'] / 3600 * 2.5:.2f} at $2.50/h")
    return True


@app.local_entrypoint()
def main(prompts: str, out: str, thinking: bool = False, n: int = 1, max_tokens: int = 0, limit: int = 0,
         shards: int = 1, model_dir: str = MODEL_DIR):
    """--shards K splits the prompts round-robin over K GPUs that run in parallel: same total cost (plus one
    model load, ~2.5 min, per extra GPU) but 1/K of the wall time. --model-dir picks a trained model, e.g.
    /cache/out/pilot/final."""
    records = [json.loads(line) for line in open(prompts, encoding="utf-8") if line.strip()]
    if limit:
        records = records[:limit]
    max_tokens = max_tokens or (32768 if thinking else 4096)
    name = Path(out).stem
    chunks = [(k, records[k::shards]) for k in range(shards) if records[k::shards]]
    shards = len(chunks)  # fewer prompts than GPUs: use fewer GPUs

    try:  # results of an earlier run with the same name would be mistaken for this run's
        vol.remove_file(f"runs/{name}", recursive=True)
    except Exception:
        pass

    print(f"{len(records)} prompts on {shards} GPU(s), thinking={thinking}, n={n}, max_tokens={max_tokens}, run '{name}'")
    for k, chunk in chunks:
        generate.spawn(chunk, thinking, n, max_tokens, name, k, shards, model_dir)
    print("started. If this window dies, the GPUs keep going (needs `modal run --detach`); fetch with:\n"
          f"  modal run modal/generate.py::collect --out {out}")
    _collect(name, out, shards=shards, wait=True)


@app.local_entrypoint()
def collect(out: str, wait: bool = False):
    """Merge finished shards for the run that wrote `out`. --wait keeps polling until all shards are done."""
    ok = _collect(Path(out).stem, out, shards=0, wait=wait)
    if not ok:
        print("not finished yet: run again later, or add --wait")
