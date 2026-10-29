# Trilingual Text-to-SQL

PostgreSQL-first text-to-SQL model for English, Russian and Uzbek (Qwen3.5-4B fine-tune).
Status: Phase 0 (setup and bake-offs).

## Layout

| Folder | Purpose |
|---|---|
| `docker/` | PostgreSQL 17 for scoring: read-only user, statement timeout |
| `eval/` | Eval harness: executes generated SQL and computes execution accuracy |
| `modal/` | GPU scripts (token counts on CPU, tooling check, bake-offs) |
| `data/` | Databases and datasets (gitignored) |
| `configs/` | Training configs |

Generation runs on a GPU (Modal) and only writes files. Scoring runs here, on the local machine.

## Phase 0 commands

### 1. PostgreSQL

```powershell
cd docker
copy .env.example .env     # then set your own two passwords in .env
docker compose up -d
```

- Host: `127.0.0.1`, port `5433` (not 5432, to avoid clashing with a local PostgreSQL).
- `postgres` is the admin user, used only to load databases.
- `sql_ro` is the read-only user the harness uses for generated SQL: no write rights,
  no temp tables, `statement_timeout = 10s`.
- `docker compose down -v` deletes the data volume (and re-runs the init script next start).

### 2. Python environment

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-eval.txt
```

### 3. Eval harness

```powershell
pytest                                   # 73 tests, each run on SQLite and PostgreSQL
python -m eval.harness preds.jsonl --sqlite-dir data/sqlite --out report.json
```

Input, one json object per line: `id, language, dialect (sqlite|postgres), slice, db_id, gold_sql`
and `pred_sql` (one answer) or `pred_sqls` (a list of N samples).
SQLite files are looked up as `<sqlite-dir>/<db_id>.sqlite`; the PostgreSQL database name is `db_id`.

Metrics: `acc` (first answer correct), `maj` (majority vote over successful executions, same result = same
vote), `oracle` (any of N correct), by language, dialect, slice and language/dialect.
Rows whose gold SQL fails are reported as `goldErr` and left out of every accuracy.

Comparison rules: row order is ignored unless the gold SQL has a top-level `ORDER BY`; duplicate rows count;
column order matters; numbers are rounded to 6 decimals; multi-statement SQL is rejected.

## Secrets

Keys and passwords never go in the repo. `docker/.env` is gitignored. The DeepSeek key goes in
an environment variable; the Hugging Face token goes in the Modal secret `huggingface`.
