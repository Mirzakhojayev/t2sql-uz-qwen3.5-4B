"""Prompt v2: tell the model what the columns mean and which stored values the question refers to.

BIRD ships a `database_description/<table>.csv` per table (column meaning, value codes). Without it the model sees
`a11 BIGINT -- e.g. 8369` and has to guess that A11 is the average salary. Published systems (OmniSQL and others) put
three things in the schema comments: column descriptions, sample values, and values that match the question. Until
2026-11-16 our prompts had only the sample values.

  describe(tables, load_descriptions(dir))        # fills Column.description
  index = value_index(sqlite_path)                 # distinct text values per column, cached by the caller
  matching_values(question + hint, index, names)   # the "Values in the database that match the question" block
"""
import csv
import io
import re
import sqlite3
from pathlib import Path

from .prompt import Table, format_value, quote_ident

MAX_DESCRIPTION_CHARS = 160
MAX_DISTINCT = 3000       # columns with more distinct values are ids or free text: not indexed
MAX_VALUE_LEN = 60
MAX_MATCHES = 8
USELESS = re.compile(r"^(not useful|unuseful|useless|n/?a|none|null|-)?$", re.IGNORECASE)
# single words that are too common to be worth showing as a "matching value"
COMMON = set("""a an and are as at be by for from has have how in is it its list many much name names number of on
or per show than that the their them these this to total was were what when where which who with year years all any
each most more less least first last top average count sum date time type status id yes no true false male female
other others new old high low active one two three""".split())


def _clean(text: str | None) -> str:
    return " ".join((text or "").replace("﻿", "").split())


def load_descriptions(folder: Path) -> dict[str, dict[str, str]]:
    """{table lower-case: {column lower-case: text}} from BIRD's CSV files; missing folder -> {}.
    Text = what the column is, then what its values mean. Empty when the file adds nothing to the name."""
    out: dict[str, dict[str, str]] = {}
    for path in sorted(Path(folder).glob("*.csv")) if Path(folder).is_dir() else []:
        raw = path.read_bytes().decode("utf-8-sig", errors="replace")
        columns: dict[str, str] = {}
        for row in csv.DictReader(io.StringIO(raw)):
            row = {(k or "").strip(): v for k, v in row.items()}
            original = _clean(row.get("original_column_name"))
            if not original:
                continue
            name, meaning, values = (_clean(row.get(k)) for k in ("column_name", "column_description", "value_description"))
            same = lambda a: re.sub(r"[\W_]+", "", a.lower()) == re.sub(r"[\W_]+", "", original.lower())
            text = meaning if meaning and not same(meaning) else name if name and not same(name) else ""
            if values and not USELESS.match(values):
                text = f"{text} ({values})" if text else values
            if text:
                columns[original.lower()] = text[:MAX_DESCRIPTION_CHARS].rstrip() + ("..." if len(text) > MAX_DESCRIPTION_CHARS else "")
        out[path.stem.lower()] = columns
    return out


def describe(tables: list[Table], descriptions: dict[str, dict[str, str]]) -> list[Table]:
    """Fill Column.description in place (names compared in lower case: PostgreSQL copies are lower-cased)."""
    for table in tables:
        columns = descriptions.get(table.name.lower(), {})
        for col in table.columns:
            col.description = columns.get(col.name.lower(), "")
    return tables


def value_index(sqlite_path: Path) -> dict[str, list[tuple[str, str, str]]]:
    """{value lower-case: [(table, column, value as stored)]} for text columns with few distinct values."""
    conn = sqlite3.connect(f"{Path(sqlite_path).resolve().as_uri()}?mode=ro", uri=True)
    conn.text_factory = lambda b: b.decode("utf-8", errors="replace")
    index: dict[str, list[tuple[str, str, str]]] = {}
    try:
        tables = [r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")]
        for table in tables:
            for _, col, col_type, *_ in conn.execute(f'PRAGMA table_info("{table}")').fetchall():
                if not re.search(r"CHAR|TEXT|CLOB|^$", (col_type or "").upper()):
                    continue
                rows = conn.execute(f'SELECT DISTINCT "{col}" FROM "{table}" WHERE typeof("{col}") = \'text\' '
                                    f"LIMIT {MAX_DISTINCT + 1}").fetchall()
                if len(rows) > MAX_DISTINCT:
                    continue
                for (value,) in rows:
                    v = value.strip()
                    if 3 <= len(v) <= MAX_VALUE_LEN and not re.fullmatch(r"[\d\W]+", v):
                        index.setdefault(v.lower(), []).append((table, col, value))
    finally:
        conn.close()
    return index


def matching_values(text: str, index: dict[str, list], rename=lambda name: name) -> str:
    """Stored values that appear in the question or hint, longest first, as lines `table.column: 'value'`.
    rename: maps a SQLite name to the name shown in the prompt (lower case in the PostgreSQL copies)."""
    words = re.findall(r"[\w'’.\-/&]+", text)
    found: dict[tuple[str, str, str], int] = {}
    for n in range(5, 0, -1):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i + n]).strip(".,'’").lower()
            if n == 1 and (phrase in COMMON or len(phrase) < 4):
                continue
            for hit in index.get(phrase, []):
                found.setdefault(hit, n)
    best = sorted(found, key=lambda h: (-found[h], -len(h[2]), h))[:MAX_MATCHES]
    return "\n".join(f"{quote_ident(rename(t))}.{quote_ident(rename(c))}: {format_value(v)}" for t, c, v in best)
