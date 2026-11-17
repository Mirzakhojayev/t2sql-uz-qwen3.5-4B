# t2sql-uz-qwen3.5-4B

Text-to-SQL for **English, Russian and Uzbek** (Latin and Cyrillic), PostgreSQL first, with a small model that can
run locally: Qwen3.5-4B plus a LoRA adapter.

| Release | Link |
|---|---|
| Model (LoRA adapter) | [`mirzakhojayev/t2sql-uz-qwen3.5-4B`](https://huggingface.co/mirzakhojayev/t2sql-uz-qwen3.5-4B) |
| Dataset: Uzbek/Russian eval sets + training data | [`mirzakhojayev/uzbird`](https://huggingface.co/datasets/mirzakhojayev/uzbird) |

## Results

Execution accuracy on 500 BIRD dev questions in four languages (the `standard` set of `uzbird`), PostgreSQL, no
thinking, first answer:

| | English | Russian | Uzbek Latin | Uzbek Cyrillic | All |
|---|---|---|---|---|---|
| Qwen3.5-4B | 54.4 | 53.0 | 44.4 | 45.0 | 50.2 |
| **t2sql-uz-qwen3.5-4B** | **60.8** | **59.4** | **57.4** | **50.0** | **58.6** |
| Change | +6.4 | +6.4 | **+13.0** | +5.0 | **+8.4** |

- All changes except Uzbek Cyrillic (only 100 questions) are very unlikely to be chance (paired McNemar test,
  p < 0.005; overall p < 10⁻¹⁰).
- Majority vote over 5 answers: 55.4 → **60.7**. Base Qwen *with* thinking scores 55.3, so the fine-tuned model
  beats it while writing only the SQL, at a fraction of the cost per answer.
- Uzbek gains the most: the gap to English shrinks from 10.0 to 3.4 points.
- By difficulty: simple 56.8 → 67.0, moderate 41.3 → 48.6, challenging 37.9 → 37.1.

**Limitation.** On the `hard` set (BIRD Mini-Dev, 92% "challenging" questions) the fine-tuned model is worse than
the base model: 5.5 against 11.1 without thinking. The training data is normal-difficulty BIRD, so the model got
better at everyday questions and worse at the hardest ones. For those, base Qwen with thinking does better (18.3).

## What worked, and what did not

Six training runs. The first five all scored *below* the base model; the sixth is the release.

| Attempt | Training data | Outcome |
|---|---|---|
| Full fine-tune, 10k examples | DeepSeek's reasoning + SQL on SynSQL questions | Worse than base |
| Full fine-tune, 500 questions | Same, cleaned; two learning rates | Worse than base |
| Full fine-tune, 460 questions | DeepSeek's reasoning on BIRD train | Worse than base |
| Full fine-tune, 12k examples | Qwen's *own* correct answers (rejection sampling) | Worse than base |
| **LoRA, 26k examples** | **Gold SQL only, no reasoning, descriptive prompts** | **Better than base** |

Lessons, with the evidence in [`docs/data-rules.md`](docs/data-rules.md):

1. **Copying a larger model's reasoning hurt the 4B model.** It learned the teacher's short style, thought less,
   and made more mistakes. Training on its own reasoning did not help either: most of it came from questions the model
   could already solve.
2. **The prompt mattered as much as training.** BIRD ships column descriptions (for example: `a11` is "average
   salary"), which we first left out. Adding them, plus database values that match the question, raised base Qwen
   from 15.5 to 18.3 on the hard set, with no training at all.
3. **Synthetic databases verify little.** SynSQL databases hold about 20 rows, so a wrong query often returns the
   right answer by luck. Re-checking on enlarged copies (`t2sql/stress.py`) rejected about 30% of "verified"
   answers.
4. **Self-correction works without training.** Running the model's query and showing it the error raised base
   Qwen from 16.1 to 20.1 on the hard set (`t2sql/retry.py`).
5. **What worked:** gold SQL as the only target, LoRA (it changes 1.4% of the model, so it forgets less),
   thinking off, and the descriptive prompt in both training and testing.

## Repository

| Folder | Purpose |
|---|---|
| `t2sql/` | Data pipeline: prompts (`prompt.py`, `describe.py`), translation, Uzbek Cyrillic, leakage check, training-data building, self-correction, release |
| `eval/` | Eval harness: runs predicted and gold SQL and compares the results |
| `modal/` | GPU scripts: model download, training (full or LoRA), generation with vLLM |
| `docker/` | PostgreSQL 17 for scoring, with a read-only user and a statement timeout |
| `docs/` | Data rules and findings |
| `data/` | Databases, datasets and run outputs (not in git) |

Generation runs on a GPU (Modal) and only writes files. Scoring runs on the local machine.

## Setup

### 1. PostgreSQL

```powershell
cd docker
copy .env.example .env     # then set your own two passwords in .env
docker compose up -d
```

- Host `127.0.0.1`, port `5433` (not 5432, to avoid clashing with a local PostgreSQL).
- `postgres` is the admin user, used only to load databases. `sql_ro` is the read-only user that runs generated
  SQL: no write rights, no temp tables, `statement_timeout = 10s`.

### 2. Python

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-eval.txt
pytest                                   # database tests run on SQLite and PostgreSQL
```

### 3. Data

Download BIRD (dev, Mini-Dev and train databases) from [bird-bench.github.io](https://bird-bench.github.io/) into
`data/`. Load the dev / Mini-Dev databases into PostgreSQL with `docker/restore_dumps.sh`; for BIRD train, put
`train.zip` in `data/bird_train/` and run `python -m t2sql.bird_train unzip` (the `prepare` step below loads it).

## Reproducing the release

```powershell
python -m t2sql.final hard                         # hard set with the descriptive prompt
python -m t2sql.final standard --max-usd 0.5       # standard set: 500 BIRD dev questions, translated
python -m t2sql.bird_train prepare --questions 7731 --factor 1 --jobs data/pipeline/final/bird_questions.jsonl
python -m t2sql.final translate --max-usd 3.5      # Russian and Uzbek for BIRD train (DeepSeek API)
python -m t2sql.final train                        # data/pipeline/final_train.jsonl, 30,088 examples
modal run modal/download_model.py
modal run --detach modal/train.py::main --data data/pipeline/final_train.jsonl --name final --epochs 1 --lr 1e-4 --no-thinking --lora-rank 32 --max-length 6000 --gpu A100-40GB
modal run --detach modal/generate.py::main --prompts data/eval/standard_prompts.jsonl --out data/runs/final/final_standard.jsonl --no-thinking --n 5 --shards 2 --model-dir /cache/out/final/final
python -m t2sql.score data/eval/standard_prompts.jsonl data/runs/final/final_standard.jsonl
python -m t2sql.release                            # the Hugging Face dataset files
```

In Git Bash, put `MSYS_NO_PATHCONV=1` in front of commands that pass `/cache/...`: otherwise Git Bash rewrites the path.

### Scoring rules

`python -m eval.harness` takes one json object per line: `id, language, dialect (sqlite|postgres), slice, db_id,
gold_sql` and `pred_sql` or `pred_sqls` (N samples). Metrics: `acc` (first answer), `maj` (majority vote over
answers that run; same result = same vote), `oracle` (any of N). Row order is ignored unless the gold SQL has a
top-level `ORDER BY`; duplicate rows count; column order matters; numbers are rounded to 6 decimals; multi-statement
SQL is rejected. Rows whose gold SQL fails are reported as `goldErr` and left out.

### Running on another GPU provider

The scripts in `modal/` use Modal only for three things: the `@app.function` decorators (which GPU, time limit),
a storage volume mounted at `/cache`, and the container image. The work inside is plain PyTorch, Hugging Face
`transformers`/`peft` and vLLM. To run elsewhere (RunPod, a university cluster, your own GPU):

1. Install the package versions pinned in the image definitions at the top of `modal/train.py` and
   `modal/generate.py` (the exact versions matter: Qwen3.5 needs a recent vLLM and `causal-conv1d` for speed).
2. Download `Qwen/Qwen3.5-4B` from Hugging Face into a local folder.
3. Call the body of `train()` or `_generate()` directly, with `/cache/...` paths replaced by local paths.

Full fine-tuning needs an 80 GB GPU; LoRA (`--lora-rank 32`) peaked at 33 GB.

## Data and licenses

- Questions and gold SQL: [BIRD](https://bird-bench.github.io/) (CC BY-SA 4.0) and
  [SynSQL-2.5M](https://huggingface.co/datasets/seeklhy/SynSQL-2.5M) (Apache 2.0).
- Translations: DeepSeek V4.1 Flash, with automatic checks and no human review.
- Base model: [Qwen3.5-4B](https://huggingface.co/Qwen/Qwen3.5-4B) (Apache 2.0).
- Eval sets are never trained on: every training example passes a leakage check against Spider dev, BIRD dev and
  Mini-Dev, PAUQ test, RedSQL and BIRD-CRITIC (`t2sql/leakage.py`).

## Secrets

Keys and passwords never go in the repo. `docker/.env` is gitignored. The DeepSeek key goes in the environment or
`docker/.env`; the Hugging Face token goes in the Modal secret `huggingface`.
