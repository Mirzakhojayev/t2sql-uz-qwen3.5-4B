"""Fine-tune Qwen3.5-4B on execution-verified traces (full fine-tune, thinking format), on one A100 80GB.

  # test run: about 30 batches, the longest one first so that an out-of-memory error shows up at once
  modal run --detach modal/train.py::main --data data/pipeline/pilot_10k.jsonl --name pilot_test --max-steps 30
  # the real run; run the same command again to resume after a crash
  modal run --detach modal/train.py::main --data data/pipeline/pilot_10k.jsonl --name pilot

Each example is the chat prompt ending in "<think>\n", then "reasoning\n</think>\n\nanswer<|im_end|>". The model is
exactly what generation sees, because prompt and target are tokenized separately. Only target tokens get a loss.

Memory: weights stay float32 with bf16 only in the forward pass, because updates at this learning rate would mostly
round away in bf16. Training everything, weights, gradients and Adam's two moments took 67 GB of the 79 GB and a 24k
batch ran out of memory, so the embeddings and output head (the 248k-vocabulary matrices, which barely move at this
learning rate) are frozen and a batch is held to 20k padded tokens. All 32 layers train. Logits are computed in chunks
on target tokens only: with a 248k vocabulary, logits for all positions at once are too large.

Output on the volume: out/<name>/final (bf16 weights + tokenizer, what modal/generate.py --model-dir loads),
out/<name>/summary.json, and out/<name>/resume while the run is going. A resume restores weights and position but
not Adam's moments, which rebuild within a few steps: a third of the disk and write time, for a rare event.
--max-hours (with the function timeout behind it) is the cost cap.
"""
import json
import os
import random
import time
import zlib
from pathlib import Path

import modal

app = modal.App("t2sql-train")
vol = modal.Volume.from_name("t2sql-cache", create_if_missing=True)
CACHE = "/cache"
MODEL_DIR = f"{CACHE}/models/Qwen3.5-4B"
GPU_USD_PER_HOUR = {"A100-80GB": 2.50, "H100": 3.95}  # modal.com/pricing, 2026-11-03; both have 80 GB
MAX_LENGTH = 16384  # decided 2026-10-31
HOLDOUT_PCT = 2  # of pilot questions, for validation loss only; eval sets never come here
LOGIT_CHUNK = 2048

# the versions that passed the tooling check on 2026-10-29
# causal_conv1d has no wheel on PyPI and needs nvcc to build, hence NVIDIA's image at torch 2.15's CUDA version. Without
# it the DeltaNet layers fall back to a reference PyTorch implementation that transformers warns is much slower.
train_image = (
    modal.Image.from_registry("nvidia/cuda:13.2.1-devel-ubuntu22.04", add_python="3.11")
    .pip_install("torch==2.15.0", "transformers==5.21.1", "accelerate==1.15.0", "flash-linear-attention==0.6.0",
                 "safetensors", "sentencepiece", "setuptools", "wheel", "packaging", "ninja")
    .env({"CAUSAL_CONV1D_FORCE_BUILD": "TRUE"})
    # add_python's standalone build links with clang++, which this image lacks. The compiler's output overflows
    # Modal's log limit and hides any error, so it goes to a file.
    .env({"CC": "gcc", "CXX": "g++", "LDSHARED": "g++ -pthread -shared", "LDCXXSHARED": "g++ -pthread -shared"})
    .run_commands("pip install --no-build-isolation causal-conv1d==1.7.0 > /tmp/build.log 2>&1 || (tail -60 /tmp/build.log | cut -c1-300; exit 1)",
                  "python -c 'import causal_conv1d'")
    .env({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})  # 11.8 GB sat unusable in the first test's OOM
    .pip_install("peft==0.21.3")  # last, so that adding it does not rebuild causal_conv1d
)


# ---- pure helpers, tested in t2sql/tests/test_train.py ---------------------------------------------------------

def target_text(example: dict, thinking: bool = True) -> str:
    """What the model must write after the prompt. Without thinking the prompt already holds an empty think block,
    so the target is the answer alone."""
    if not thinking:
        return f"{example['answer'].strip()}<|im_end|>"
    return f"{example['reasoning'].strip()}\n</think>\n\n{example['answer'].strip()}<|im_end|>"


def is_holdout(question_id: str, pct: int = HOLDOUT_PCT) -> bool:
    """By question, so a question's other language cannot sit on the training side."""
    return zlib.crc32(question_id.encode()) % 100 < pct


def token_batches(lengths: list[int], max_tokens: int, max_examples: int, rng: random.Random,
                  longest_first: bool = False, pool: int = 256) -> list[list[int]]:
    """Groups of similar length, so short examples do not pay for a long one's padding. Each batch's padded size
    (longest x count) stays within max_tokens. Random pools keep the batches from being sorted across the epoch."""
    order = list(range(len(lengths)))
    rng.shuffle(order)
    batches: list[list[int]] = []
    for start in range(0, len(order), pool):
        current: list[int] = []
        for i in sorted(order[start:start + pool], key=lambda k: lengths[k]):
            if current and (len(current) >= max_examples or lengths[i] * (len(current) + 1) > max_tokens):
                batches.append(current)
                current = []
            current.append(i)
        if current:
            batches.append(current)
    rng.shuffle(batches)
    if longest_first:  # the batch with the most padded tokens goes first
        largest = max(range(len(batches)), key=lambda k: max(lengths[i] for i in batches[k]) * len(batches[k]))
        batches.insert(0, batches.pop(largest))
    return batches


def lr_at(step: int, total: int, base: float, warmup_pct: float = 0.05) -> float:
    """Linear warmup, then cosine down to zero."""
    import math

    warmup = max(1, int(total * warmup_pct))
    if step < warmup:
        return base * (step + 1) / warmup
    return base * 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))


# ---- GPU side --------------------------------------------------------------------------------------------------

@app.function(image=train_image, gpu="A100-80GB", volumes={CACHE: vol}, timeout=6 * 3600)  # at most $15
def train(data_path: str, name: str, epochs: int, lr: float, max_tokens: int, max_examples: int, accum: int,
          max_steps: int, max_hours: float, save_every_minutes: int, seed: int, gpu: str,
          thinking: bool = True, lora_rank: int = 0, max_length: int = MAX_LENGTH) -> dict:
    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    started = time.time()
    vol.reload()
    out_dir = Path(f"{CACHE}/out/{name}")
    resume_dir = out_dir / "resume"
    out_dir.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    # --- tokenize ---
    rows = [json.loads(line) for line in open(data_path, encoding="utf-8")]
    encoded, dropped = {"train": [], "holdout": []}, {"no_reasoning": 0, "too_long": 0}
    for ex in rows:
        if thinking and not (ex.get("reasoning") or "").strip():
            dropped["no_reasoning"] += 1
            continue
        prompt = tok.apply_chat_template(ex["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
        if thinking and not prompt.endswith("<think>\n"):
            raise ValueError(f"unexpected end of the chat template: {prompt[-40:]!r}")
        p = tok(prompt, add_special_tokens=False).input_ids
        t = tok(target_text(ex, thinking), add_special_tokens=False).input_ids
        if len(p) + len(t) > max_length:
            dropped["too_long"] += 1
            continue
        encoded["holdout" if is_holdout(ex["question_id"]) else "train"].append((p + t, len(p)))
    train_set, holdout = encoded["train"], encoded["holdout"]
    lengths = [len(x) for x, _ in train_set]
    print(f"{len(train_set)} train, {len(holdout)} holdout, dropped {dropped}; length p50 {sorted(lengths)[len(lengths) // 2]} "
          f"max {max(lengths)}; {time.time() - started:.0f}s", flush=True)
    sample = rows[0]
    print("prompt of the last example ends with:", repr(prompt[-60:]), flush=True)
    print("target of the first example ends with:", repr(target_text(sample, thinking)[-60:]), flush=True)

    epoch_batches = [token_batches(lengths, max_tokens, max_examples, random.Random(seed + e),
                                   longest_first=max_steps > 0 and e == 0) for e in range(epochs)]
    if max_steps:  # Adam's memory appears at the first optimizer step: run the largest batch again on top of it
        epoch_batches[0].insert(accum, list(epoch_batches[0][0]))
    total_batches = sum(len(b) for b in epoch_batches)
    total_steps = total_batches // accum + 1
    total_padded = sum(max(lengths[i] for i in b) * len(b) for bs in epoch_batches for b in bs)

    # --- model ---
    model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.float32).to("cuda")
    core = model  # the Qwen model itself; with LoRA, `model` becomes the wrapper around it
    if lora_rank:
        import peft

        # LoRA trains small add-on matrices and leaves every original weight as it is: full fine-tuning changed
        # 85% of the weights and each run lost accuracy on hard questions (docs/data-rules.md).
        model = peft.get_peft_model(model, peft.LoraConfig(
            r=lora_rank, lora_alpha=2 * lora_rank, lora_dropout=0.05, target_modules="all-linear", bias="none",
            task_type="CAUSAL_LM"))
        print(f"peft {peft.__version__}, LoRA rank {lora_rank}", flush=True)
    position = {"epoch": 0, "batch": 0, "opt_step": 0}
    if (resume_dir / "state.json").exists():
        model.load_state_dict(torch.load(resume_dir / "weights.pt", map_location="cuda"), strict=not lora_rank)
        position = json.loads((resume_dir / "state.json").read_text())
        print(f"resumed at {position}", flush=True)
    model.config.use_cache = False
    core.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    decoder, lm_head = core.model, core.lm_head
    for frozen in (core.get_input_embeddings(), lm_head):
        frozen.requires_grad_(False)
    trained = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trained, lr=lr, weight_decay=0.0, fused=True)
    print(f"{type(model).__name__}, {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B params, "
          f"{sum(p.numel() for p in trained) / 1e9:.2f}B trained, {total_batches} batches, "
          f"{total_steps} optimizer steps; loaded at {time.time() - started:.0f}s", flush=True)

    def hidden_and_targets(batch: list[tuple[list[int], int]]):
        width = max(len(x) for x, _ in batch)
        ids = torch.tensor([x + [pad] * (width - len(x)) for x, _ in batch], device="cuda")
        mask = torch.tensor([[1] * len(x) + [0] * (width - len(x)) for x, _ in batch], device="cuda")
        at = torch.zeros_like(ids, dtype=torch.bool)
        for row, (x, n_prompt) in enumerate(batch):
            at[row, n_prompt - 1:len(x) - 1] = True  # the token before each target token predicts it
        targets = torch.tensor([tid for x, n_prompt in batch for tid in x[n_prompt:]], device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            hidden = decoder(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
        return hidden[at], targets

    def chunked_loss(states, targets, scale: float, backward: bool) -> float:
        total = 0.0
        for s in range(0, len(targets), LOGIT_CHUNK):
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = lm_head(states[s:s + LOGIT_CHUNK])
            loss = F.cross_entropy(logits.float(), targets[s:s + LOGIT_CHUNK], reduction="sum") / len(targets)
            if backward:
                (loss * scale).backward()
            total += loss.item()
        return total

    def validate() -> float | None:
        if not holdout:
            return None
        model.eval()
        losses = []
        with torch.no_grad():
            order = sorted(range(len(holdout)), key=lambda k: len(holdout[k][0]))
            for s in range(0, min(len(order), 64 if max_steps else len(order)), 4):
                states, targets = hidden_and_targets([holdout[k] for k in order[s:s + 4]])
                losses.append(chunked_loss(states, targets, 1.0, backward=False))
        model.train()
        return round(sum(losses) / len(losses), 4)

    def save_resume(at: dict) -> None:
        resume_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        state = model.state_dict()
        if lora_rank:  # the base weights never change: saving them would write 16 GB for nothing
            state = {k: v for k, v in state.items() if "lora_" in k}
        torch.save(state, resume_dir / "weights.tmp")
        os.replace(resume_dir / "weights.tmp", resume_dir / "weights.pt")
        (resume_dir / "state.json").write_text(json.dumps(at))
        vol.commit()
        print(f"resume point at {at}: {time.time() - t0:.0f}s", flush=True)

    # --- loop ---
    log_rows: list[dict] = []
    torch.cuda.reset_peak_memory_stats()
    opt_step, done_batches, last_save = position["opt_step"], 0, time.time()
    train_seconds, train_padded, first_batch_seconds = 0.0, 0, None
    window_loss, window_n, stop_reason = 0.0, 0, None
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(position["epoch"], epochs):
        batches = epoch_batches[epoch]
        start_at = position["batch"] if epoch == position["epoch"] else 0
        for b, indices in enumerate(batches[start_at:], start=start_at):
            t0 = time.time()
            batch = [train_set[i] for i in indices]
            states, targets = hidden_and_targets(batch)
            leaf = states.detach().requires_grad_()  # the head runs chunk by chunk, then the gradient goes back at once
            loss = chunked_loss(leaf, targets, 1.0 / accum, backward=True)
            states.backward(leaf.grad)
            del states, leaf
            window_loss, window_n = window_loss + loss, window_n + 1
            if (b + 1) % accum == 0 or b + 1 == len(batches):
                for group in optimizer.param_groups:
                    group["lr"] = lr_at(opt_step, total_steps, lr)
                torch.nn.utils.clip_grad_norm_(trained, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1
            torch.cuda.synchronize()
            seconds = time.time() - t0
            padded = max(len(x) for x, _ in batch) * len(batch)
            done_batches += 1
            if first_batch_seconds is None:
                first_batch_seconds = seconds  # includes the one-time kernel compile: not used for speed
                print(f"first batch ({len(batch)} x {max(len(x) for x, _ in batch)} tokens): {seconds:.0f}s, "
                      f"peak {torch.cuda.max_memory_allocated() / 1e9:.1f} GB", flush=True)
            else:
                train_seconds, train_padded = train_seconds + seconds, train_padded + padded
            if done_batches % 10 == 0:
                row = {"epoch": epoch, "batch": b + 1, "of": len(batches), "opt_step": opt_step,
                       "loss": round(window_loss / window_n, 4), "lr": optimizer.param_groups[0]["lr"],
                       "tok_per_s": round(train_padded / max(train_seconds, 1e-6)),
                       "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1),
                       "usd": round((time.time() - started) / 3600 * GPU_USD_PER_HOUR[gpu], 2)}
                log_rows.append(row)
                print(json.dumps(row), flush=True)
                window_loss, window_n = 0.0, 0
            if max_steps and done_batches >= max_steps:
                stop_reason = "max_steps"
            elif (time.time() - started) / 3600 >= max_hours:
                stop_reason = "max_hours"
            elif (time.time() - last_save) / 60 >= save_every_minutes and b + 1 < len(batches):
                save_resume({"epoch": epoch, "batch": b + 1, "opt_step": opt_step})
                last_save = time.time()
            if stop_reason:
                break
        if stop_reason:
            break
        val = validate()
        print(f"epoch {epoch} finished, holdout loss {val}", flush=True)
        log_rows.append({"epoch": epoch, "holdout_loss": val})

    # --- finish ---
    speed = train_padded / max(train_seconds, 1e-6)
    summary = {
        "name": name, "gpu": gpu, "stop_reason": stop_reason or "finished", "train_examples": len(train_set),
        "holdout_examples": len(holdout), "dropped": dropped, "epochs": epochs, "lr": lr, "max_tokens": max_tokens,
        "max_examples": max_examples, "accum": accum, "thinking": thinking, "lora_rank": lora_rank, "max_length": max_length,
        "batches_run": done_batches, "opt_steps": opt_step,
        "first_batch_seconds": round(first_batch_seconds or 0), "padded_tok_per_s": round(speed),
        "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1), "holdout_loss": validate(),
        # a one-batch run has no speed: its only batch is the first, which is left out of the measurement
        "planned_padded_tokens": total_padded,
        "projected_hours_full_run": round(total_padded / speed / 3600, 2) if train_padded else None,
        "projected_usd_full_run": round(total_padded / speed / 3600 * GPU_USD_PER_HOUR[gpu], 2) if train_padded else None,
        "log": log_rows,
    }
    if stop_reason == "max_hours":
        save_resume({"epoch": epoch, "batch": b + 1, "opt_step": opt_step})
    else:
        final = out_dir / "final"
        t0 = time.time()
        core.gradient_checkpointing_disable()
        core.config.use_cache = True
        if lora_rank:
            model.save_pretrained(out_dir / "adapter")  # the small add-on by itself, for publishing
            model = model.merge_and_unload()  # one ordinary model again, which is what generation loads
        model.to(torch.bfloat16).save_pretrained(final)
        tok.save_pretrained(final)
        summary["save_seconds"] = round(time.time() - t0)
        for leftover in ("weights.pt", "state.json"):
            if (resume_dir / leftover).exists():
                (resume_dir / leftover).unlink()
    summary["hours"] = round((time.time() - started) / 3600, 3)
    summary["usd"] = round(summary["hours"] * GPU_USD_PER_HOUR[gpu], 2)
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    vol.commit()
    print(json.dumps({k: v for k, v in summary.items() if k != "log"}, indent=1), flush=True)
    return summary


# ---- local side ------------------------------------------------------------------------------------------------

@app.local_entrypoint()
def main(data: str, name: str, epochs: int = 2, lr: float = 2e-5, max_tokens: int = 20480, max_examples: int = 8,
         accum: int = 3, max_steps: int = 0, max_hours: float = 5.5, save_every_minutes: int = 45, seed: int = 0,
         gpu: str = "A100-80GB", thinking: bool = True, lora_rank: int = 0, max_length: int = MAX_LENGTH):
    """--no-thinking trains the answer alone (no reasoning). --lora-rank N trains LoRA add-ons of rank N instead of
    the weights themselves; use a higher learning rate with it (1e-4). --max-length drops longer examples (in tokens):
    a few huge schemas can take a third of the training time."""
    keep = ("id", "question_id", "language", "messages", "reasoning", "answer")
    slim = [{k: r[k] for k in keep} for r in map(json.loads, open(data, encoding="utf-8"))]
    upload = Path(data).with_suffix(".train_upload.jsonl")
    with open(upload, "w", encoding="utf-8") as fh:
        for r in slim:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    remote = f"train_data/{name}.jsonl"
    with vol.batch_upload(force=True) as batch:
        batch.put_file(str(upload), remote)
    print(f"uploaded {len(slim)} examples to {remote}. Worst case ${max_hours * GPU_USD_PER_HOUR[gpu]:.2f} on {gpu} (--max-hours {max_hours})")
    call = train.with_options(gpu=gpu).spawn(f"{CACHE}/{remote}", name, epochs, lr, max_tokens, max_examples, accum,
                                             max_steps, max_hours, save_every_minutes, seed, gpu, thinking, lora_rank, max_length)
    print(f"started: {call.object_id}. If this window dies the GPU keeps going (needs `modal run --detach`); see "
          f"out/{name}/summary.json on the volume")
    summary = call.get()
    Path("data/runs").joinpath(f"train_{name}.summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print({k: v for k, v in summary.items() if k != "log"})
