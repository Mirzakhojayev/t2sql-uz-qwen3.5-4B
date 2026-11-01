"""Tooling check for Qwen3.5-4B: does the fine-tune -> merge -> GGUF -> run path work end to end?

Throwaway: trains ~40 LoRA steps on a handful of toy examples about the tiny test database. The point is
to find breakage early (multimodal class, DeltaNet kernels, chat template, llama.cpp support), not to
get a good model. Nothing here touches eval data.

  modal run modal/tooling_check.py --stage download   # CPU, fetch weights into the Modal volume
  modal run modal/tooling_check.py --stage train      # one L4 GPU: LoRA smoke test, 4096-token memory test, merge
  modal run modal/tooling_check.py --stage gguf       # CPU: convert to GGUF, quantize, run it
  modal run modal/tooling_check.py                    # download + train (gguf only when asked for)

Every function has a timeout, which is also the cost cap for that stage.
"""
import json
import os
import subprocess
import time

import modal

app = modal.App("t2sql-tooling-check")
vol = modal.Volume.from_name("t2sql-cache", create_if_missing=True)

CACHE = "/cache"
MODEL_ID = "Qwen/Qwen3.5-4B"
MODEL_DIR = f"{CACHE}/models/Qwen3.5-4B"
MERGED_DIR = f"{CACHE}/out/tooling_check_merged"
GGUF_DIR = f"{CACHE}/out/tooling_check_gguf"
PROMPT_FILE = f"{CACHE}/out/tooling_check_prompt.txt"

hf_secret = modal.Secret.from_name("huggingface")

light_image = modal.Image.debian_slim(python_version="3.11").pip_install("huggingface_hub")

# Versions are unpinned on purpose for this first run; the stage prints what got installed so we can pin it.
train_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch", "transformers", "peft", "accelerate", "huggingface_hub", "safetensors", "sentencepiece",
    "flash-linear-attention",  # fast kernels for the Gated DeltaNet layers (24 of 32 layers)
)

gguf_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "cmake", "build-essential")
    .run_commands(
        "git clone --depth 1 https://github.com/ggml-org/llama.cpp /opt/llama.cpp",
        "cd /opt/llama.cpp && cmake -B build -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF"
        " && cmake --build build -j 8 --target llama-quantize llama-cli",
    )
    .pip_install("torch", extra_index_url="https://download.pytorch.org/whl/cpu")
    .run_commands("pip install -r /opt/llama.cpp/requirements/requirements-convert_hf_to_gguf.txt")
    .pip_install("transformers", "huggingface_hub", extra_options="-U")
)

# ---- toy training data: the planned prompt format (dialect + schema + question), three languages ----------

SCHEMA = (
    "CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT NOT NULL);\n"
    "CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT NOT NULL, dept_id INTEGER REFERENCES departments(id),\n"
    "  salary NUMERIC(10,2) NOT NULL, hired DATE NOT NULL, manager_id INTEGER REFERENCES employees(id));"
)

TOY = [  # (question, short reasoning, sql)
    ("How many employees are there?", "Count all rows in employees.", "SELECT COUNT(*) FROM employees;"),
    ("Сколько всего сотрудников?", "Count all rows in employees.", "SELECT COUNT(*) FROM employees;"),
    ("Jami nechta xodim bor?", "Count all rows in employees.", "SELECT COUNT(*) FROM employees;"),
    ("List the names of employees earning more than 5000.", "Filter employees by salary and return the name.",
     "SELECT name FROM employees WHERE salary > 5000;"),
    ("Покажи имена сотрудников с зарплатой выше 5000.", "Filter employees by salary and return the name.",
     "SELECT name FROM employees WHERE salary > 5000;"),
    ("Maoshi 5000 dan yuqori xodimlarning ismlarini ko'rsating.", "Filter employees by salary and return the name.",
     "SELECT name FROM employees WHERE salary > 5000;"),
    ("What is the average salary in each department?", "Join to departments and average the salary per department.",
     "SELECT d.name, AVG(e.salary) FROM departments d JOIN employees e ON e.dept_id = d.id GROUP BY d.name;"),
    ("Какая средняя зарплата в каждом отделе?", "Join to departments and average the salary per department.",
     "SELECT d.name, AVG(e.salary) FROM departments d JOIN employees e ON e.dept_id = d.id GROUP BY d.name;"),
    ("Har bir bo'limdagi o'rtacha maoshni ko'rsating.", "Join to departments and average the salary per department.",
     "SELECT d.name, AVG(e.salary) FROM departments d JOIN employees e ON e.dept_id = d.id GROUP BY d.name;"),
    ("Who was hired most recently?", "Sort by hire date descending and take one.",
     "SELECT name FROM employees ORDER BY hired DESC LIMIT 1;"),
    ("Кто был принят на работу последним?", "Sort by hire date descending and take one.",
     "SELECT name FROM employees ORDER BY hired DESC LIMIT 1;"),
    ("Eng oxirgi ishga olingan xodim kim?", "Sort by hire date descending and take one.",
     "SELECT name FROM employees ORDER BY hired DESC LIMIT 1;"),
]
HELD_OUT = [  # not in the training set
    "How many employees have no manager?",
    "Nechta xodimning rahbari yo'q?",
]


def user_message(question: str) -> str:
    return f"Dialect: PostgreSQL\n\nSchema:\n{SCHEMA}\n\nQuestion: {question}"


def log(*a):
    print(*a, flush=True)


# ---- stage 1: download ---------------------------------------------------------------------------------------

@app.function(image=light_image, volumes={CACHE: vol}, secrets=[hf_secret], timeout=1800, cpu=4)
def download_model() -> dict:
    from huggingface_hub import snapshot_download

    t = time.time()
    snapshot_download(MODEL_ID, local_dir=MODEL_DIR)
    vol.commit()
    files = {f: os.path.getsize(os.path.join(MODEL_DIR, f)) for f in sorted(os.listdir(MODEL_DIR)) if not f.startswith(".")}
    log(f"downloaded {MODEL_ID} in {time.time() - t:.0f}s, {sum(files.values()) / 1e9:.1f} GB")
    return {"seconds": round(time.time() - t), "files": files}


# ---- stage 2: LoRA smoke test on one GPU ---------------------------------------------------------------------

@app.function(image=train_image, gpu="L4", volumes={CACHE: vol}, timeout=2400)
def lora_smoke() -> dict:
    import importlib.metadata as md
    import platform
    import warnings

    import torch
    import transformers
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer

    vol.reload()
    report: dict = {"versions": {p: md.version(p) for p in ["torch", "transformers", "peft", "accelerate", "flash-linear-attention"]}}
    report["versions"]["python"] = platform.python_version()
    log("versions:", report["versions"])
    log("gpu:", torch.cuda.get_device_name(0), f"{torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB")

    # --- tokenizer + chat template (non-thinking) ---
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    probe = tok.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
    log("rendered non-thinking prompt:", repr(probe))
    report["template_probe"] = probe
    os.makedirs(os.path.dirname(PROMPT_FILE), exist_ok=True)
    open(PROMPT_FILE, "w", encoding="utf-8").write(
        tok.apply_chat_template([{"role": "user", "content": user_message(HELD_OUT[0])}], tokenize=False,
                                add_generation_prompt=True, enable_thinking=False))

    def encode(question, reasoning, sql):
        prompt = tok.apply_chat_template([{"role": "user", "content": user_message(question)}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        answer = f"{reasoning}\n\n```sql\n{sql}\n```<|im_end|>\n"
        p = tok(prompt, add_special_tokens=False).input_ids
        f = tok(prompt + answer, add_special_tokens=False).input_ids
        if f[: len(p)] != p:
            log("WARNING: prompt tokens are not a prefix of prompt+answer tokens (label mask may be off by a token)")
        return f, [-100] * len(p) + f[len(p):]

    data = [encode(*ex) for ex in TOY]
    log("example token lengths:", [len(x[0]) for x in data])

    # --- load model; note which class it ends up as and whether the fast DeltaNet path is on ---
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t = time.time()
        loaded_with = "AutoModelForCausalLM"
        try:
            model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
        except Exception as e:  # noqa: BLE001 - we want to see and record any failure
            log("AutoModelForCausalLM failed:", repr(e)[:300])
            loaded_with = "AutoModelForImageTextToText"
            model = AutoModelForImageTextToText.from_pretrained(MODEL_DIR, dtype=torch.bfloat16)
        model.to("cuda")
        log(f"loaded in {time.time() - t:.0f}s with {loaded_with} -> {type(model).__name__}")
    report["model_class"] = type(model).__name__
    report["loaded_with"] = loaded_with
    report["params_billion"] = round(sum(p.numel() for p in model.parameters()) / 1e9, 2)
    interesting = [str(w.message)[:300] for w in caught if any(k in str(w.message).lower() for k in ("fast path", "flash-linear", "causal-conv", "fla", "torch implementation"))]
    report["kernel_warnings"] = interesting
    log("params (B):", report["params_billion"], "| kernel-related warnings:", interesting)
    top = sorted({n.split(".")[0] + "." + n.split(".")[1] for n, _ in model.named_modules() if "." in n})[:12]
    log("top-level module groups:", top)

    # --- LoRA on the language model only ---
    from peft import LoraConfig, get_peft_model

    cfg = dict(r=16, lora_alpha=32, target_modules="all-linear", task_type="CAUSAL_LM")
    try:
        lcfg = LoraConfig(**cfg, exclude_modules=r".*(visual|vision|lm_head).*")
    except TypeError:
        log("this peft has no exclude_modules; LoRA would also cover the vision tower")
        lcfg = LoraConfig(**cfg)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, lcfg)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    report["trainable_params_million"] = round(trainable / 1e6, 1)
    log("trainable params (M):", report["trainable_params_million"])
    lora_names = [n for n, _ in model.named_modules() if n.endswith("lora_A.default")]
    report["lora_modules"] = len(lora_names)
    report["lora_on_vision"] = any(("visual" in n or "vision" in n) for n in lora_names)
    log("lora modules:", report["lora_modules"], "| on vision tower:", report["lora_on_vision"])

    # --- 40 steps, batch 4 ---
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    model.train()
    losses, step_times = [], []
    torch.cuda.reset_peak_memory_stats()
    for step in range(40):
        batch = [data[(step * 4 + i) % len(data)] for i in range(4)]
        width = max(len(x[0]) for x in batch)
        ids = torch.tensor([x[0] + [pad] * (width - len(x[0])) for x in batch], device="cuda")
        labels = torch.tensor([x[1] + [-100] * (width - len(x[1])) for x in batch], device="cuda")
        mask = torch.tensor([[1] * len(x[0]) + [0] * (width - len(x[0])) for x in batch], device="cuda")
        t = time.time()
        out = model(input_ids=ids, attention_mask=mask, labels=labels)
        out.loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        step_times.append(time.time() - t)
        losses.append(round(out.loss.item(), 4))
        if step % 5 == 0 or step == 39:
            log(f"step {step:2d} loss {losses[-1]:.4f} {step_times[-1]:.2f}s")
    report["loss_first5"] = losses[:5]
    report["loss_last5"] = losses[-5:]
    report["loss_decreased"] = sum(losses[-5:]) < sum(losses[:5])
    report["median_step_s"] = round(sorted(step_times)[len(step_times) // 2], 2)
    report["peak_gb_short_seq"] = round(torch.cuda.max_memory_allocated() / 1e9, 1)

    # --- one forward+backward at each length: memory and time (LoRA on 24 GB; full fine-tune adds optimizer state) ---
    long_all = tok("SELECT name FROM employees WHERE salary > 5000; " * 1200, add_special_tokens=False).input_ids
    report["long_seq"] = {}
    for length in (4096, 6144, 8192):
        ids = torch.tensor([long_all[:length]], device="cuda")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        t = time.time()
        try:
            out = model(input_ids=ids, labels=ids.clone())
            out.loss.backward()
            torch.cuda.synchronize()
            report["long_seq"][length] = {"seconds": round(time.time() - t, 2),
                                          "peak_gb": round(torch.cuda.max_memory_allocated() / 1e9, 1)}
        except torch.cuda.OutOfMemoryError:
            report["long_seq"][length] = {"error": "out of memory on 24 GB"}
            log(f"{length} tokens: OOM")
            opt.zero_grad(set_to_none=True)
            break
        opt.zero_grad(set_to_none=True)
        del out
        log(f"{length}-token fwd+bwd:", report["long_seq"][length])

    # --- generation with the adapter on, unseen questions ---
    model.eval()
    gens = {}
    for q in HELD_OUT:
        prompt = tok.apply_chat_template([{"role": "user", "content": user_message(q)}], tokenize=False,
                                         add_generation_prompt=True, enable_thinking=False)
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=160, do_sample=False)
        gens[q] = tok.decode(gen[0][enc.input_ids.shape[1]:], skip_special_tokens=False)
        log(f"Q: {q}\nA: {gens[q]}\n")
    report["generations"] = gens
    report["generation_has_sql_block"] = all("```sql" in g for g in gens.values())

    # --- merge the adapter, save a plain bf16 checkpoint for the GGUF stage ---
    t = time.time()
    merged = model.merge_and_unload()
    os.makedirs(MERGED_DIR, exist_ok=True)
    merged.save_pretrained(MERGED_DIR, safe_serialization=True)
    tok.save_pretrained(MERGED_DIR)
    vol.commit()
    saved = {f: os.path.getsize(os.path.join(MERGED_DIR, f)) for f in sorted(os.listdir(MERGED_DIR))}
    report["merged_gb"] = round(sum(saved.values()) / 1e9, 1)
    report["merge_save_seconds"] = round(time.time() - t)
    log("merged checkpoint:", report["merged_gb"], "GB;", list(saved))
    return report


# ---- stage 3: GGUF conversion, quantization, smoke run (CPU) ----------------------------------------------------

def _run(cmd: list[str], timeout: int, tail: int = 25) -> tuple[int, str]:
    log("$", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    out = (p.stdout + "\n" + p.stderr).strip().splitlines()
    log("\n".join(out[-tail:]))
    return p.returncode, "\n".join(out[-tail:])


@app.function(image=gguf_image, volumes={CACHE: vol}, timeout=2700, cpu=8, memory=32768)
def gguf_convert() -> dict:
    vol.reload()
    report: dict = {}
    _, rev = _run(["git", "-C", "/opt/llama.cpp", "log", "-1", "--format=%H %cd"], 60)
    report["llama_cpp_commit"] = rev.strip()
    os.makedirs(GGUF_DIR, exist_ok=True)
    f16, q4 = f"{GGUF_DIR}/model-f16.gguf", f"{GGUF_DIR}/model-Q4_K_M.gguf"

    t = time.time()
    rc, out = _run(["python", "/opt/llama.cpp/convert_hf_to_gguf.py", MERGED_DIR, "--outfile", f16, "--outtype", "f16"], 1500)
    report["convert"] = {"ok": rc == 0, "seconds": round(time.time() - t), "tail": out[-1500:]}
    if rc != 0:
        return report

    t = time.time()
    rc, out = _run(["/opt/llama.cpp/build/bin/llama-quantize", f16, q4, "Q4_K_M"], 900)
    report["quantize"] = {"ok": rc == 0, "seconds": round(time.time() - t), "tail": out[-800:]}
    if rc != 0:
        return report
    report["gguf_gb"] = {"f16": round(os.path.getsize(f16) / 1e9, 1), "q4_k_m": round(os.path.getsize(q4) / 1e9, 1)}

    prompt = open(PROMPT_FILE, encoding="utf-8").read()
    t = time.time()
    rc, out = _run(["/opt/llama.cpp/build/bin/llama-cli", "-m", q4, "-p", prompt, "-n", "160", "--temp", "0",
                    "-no-cnv", "--no-display-prompt"], 600, tail=30)
    report["run"] = {"ok": rc == 0, "seconds": round(time.time() - t), "has_sql_block": "```sql" in out, "tail": out[-1500:]}
    vol.commit()
    return report


# ---- entrypoint -----------------------------------------------------------------------------------------------

@app.local_entrypoint()
def main(stage: str = "all"):
    if stage not in ("download", "train", "gguf", "all"):
        raise SystemExit("--stage must be download, train, gguf or all")
    if stage in ("download", "all"):
        print(json.dumps(download_model.remote(), indent=1)[:1500])
    if stage in ("train", "all"):
        print(json.dumps(lora_smoke.remote(), indent=1, ensure_ascii=False))
    if stage == "gguf":
        print(json.dumps(gguf_convert.remote(), indent=1, ensure_ascii=False))
