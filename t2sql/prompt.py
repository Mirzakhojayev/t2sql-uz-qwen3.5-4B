"""Turn a database into the text the model reads, and pull SQL back out of what the model writes.

Used everywhere a prompt is built: zero-shot baseline, teacher calls, training data, inference.

Prompt layout (exact format is pinned by t2sql/tests/test_prompt.py):

    Dialect: PostgreSQL

    Schema:
    CREATE TABLE departments (
      id INTEGER PRIMARY KEY, -- e.g. 1, 2, 3
      name TEXT -- e.g. 'Engineering', 'HR', 'Sales'
    );

    Hint: <optional extra knowledge, e.g. BIRD "evidence">

    Question: How many departments are there?

Run the tests:  pytest t2sql
"""
import re
import sqlite3
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

N_EXAMPLES = 3          # sample values shown per column
MAX_VALUE_CHARS = 40    # longer text values are cut and get "..."

# Words that must be quoted when used as identifiers: every SQLite keyword (147, sqlite.org/lang_keywords.html)
# plus PostgreSQL 17's reserved and type/function-name keywords (pg_get_keywords() catcode R and T).
# Quoting a harmless word costs nothing; leaving a keyword bare breaks the query (found: a column named `add`).
RESERVED = {
    "abort", "action", "add", "after", "all", "alter", "always", "analyse", "analyze", "and", "any",
    "array", "as", "asc", "asymmetric", "attach", "authorization", "autoincrement", "before", "begin",
    "between", "binary", "both", "by", "cascade", "case", "cast", "check", "collate", "collation", "column",
    "commit", "concurrently", "conflict", "constraint", "create", "cross", "current", "current_catalog",
    "current_date", "current_role", "current_schema", "current_time", "current_timestamp", "current_user",
    "database", "default", "deferrable", "deferred", "delete", "desc", "detach", "distinct", "do", "drop",
    "each", "else", "end", "escape", "except", "exclude", "exclusive", "exists", "explain", "fail", "false",
    "fetch", "filter", "first", "following", "for", "foreign", "freeze", "from", "full", "generated",
    "glob", "grant", "group", "groups", "having", "if", "ignore", "ilike", "immediate", "in", "index",
    "indexed", "initially", "inner", "insert", "instead", "intersect", "into", "is", "isnull", "join",
    "key", "last", "lateral", "leading", "left", "like", "limit", "localtime", "localtimestamp", "match",
    "materialized", "natural", "no", "not", "nothing", "notnull", "null", "nulls", "of", "offset", "on",
    "only", "or", "order", "others", "outer", "over", "overlaps", "partition", "placing", "plan", "pragma",
    "preceding", "primary", "query", "raise", "range", "recursive", "references", "regexp", "reindex",
    "release", "rename", "replace", "restrict", "returning", "right", "rollback", "row", "rows",
    "savepoint", "select", "session_user", "set", "similar", "some", "symmetric", "system_user", "table",
    "tablesample", "temp", "temporary", "then", "ties", "to", "trailing", "transaction", "trigger", "true",
    "unbounded", "union", "unique", "update", "user", "using", "vacuum", "values", "variadic", "verbose",
    "view", "virtual", "when", "where", "window", "with", "without",
}


@dataclass
class Column:
    name: str
    type: str                     # declared type, upper-case ("INTEGER", "TEXT", ...); "" if none declared
    primary_key: bool = False     # True only for a single-column primary key
    examples: list = field(default_factory=list)  # raw Python values, most frequent first
    description: str = ""         # what the column means, from the database's documentation (t2sql/describe.py)


@dataclass
class ForeignKey:
    column: str
    ref_table: str
    ref_column: str


@dataclass
class Table:
    name: str
    columns: list[Column]
    primary_key: list[str] = field(default_factory=list)  # filled only for a composite (2+ column) key
    foreign_keys: list[ForeignKey] = field(default_factory=list)


# ---------------------------------------------------------------------------------------------------------
# TODO(you): implement the six functions below until `pytest t2sql` passes.
# Suggested order: quote_ident -> format_value -> extract_sql -> build_prompt -> schema_to_ddl
#                  -> read_sqlite_schema
# ---------------------------------------------------------------------------------------------------------

def quote_ident(name: str) -> str:
    """Return the identifier as it must be written in SQL that runs on both SQLite and PostgreSQL.

    Leave it bare only if it matches ^[a-z_][a-z0-9_]*$ (ASCII lower-case) and is not in RESERVED.
    Otherwise wrap it in double quotes, doubling any double quote inside it.

    Why lower-case only: PostgreSQL folds unquoted names to lower-case, so `CDSCode` written bare
    would look for a column named `cdscode` and fail. This is one of the PostgreSQL traps in the plan.
    """
    if re.fullmatch(r"[a-z_][a-z0-9_]*", name) and name not in RESERVED:
        return name
    escaped = name.replace('"', '""')
    return f'"{escaped}"'


def format_value(v) -> str:
    """Render one sample value as a SQL literal for schema comments."""
    if v is None:
        return "NULL"

    if isinstance(v, bool):
        return "TRUE" if v else "FALSE"

    if isinstance(v, (int, float, Decimal)):
        return str(v)

    # a line break would end the "-- e.g." comment and leave the rest of the value as broken SQL
    value = re.sub(r"[\r\n\t]+", " ", str(v))
    if isinstance(v, str) and len(value) > MAX_VALUE_CHARS:
        value = value[:MAX_VALUE_CHARS] + "..."

    return "'" + value.replace("'", "''") + "'"



def extract_sql(text: str) -> str | None:
    """Pull the final SQL query out of a model answer. Return None if there is none."""
    # Remove complete <think>...</think> sections.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.IGNORECASE | re.DOTALL)

    # Ignore an unclosed <think> section and everything after it.
    text = re.sub(r"<think>.*$", "", text, flags=re.IGNORECASE | re.DOTALL)

    # Match only fully closed fenced code blocks.
    blocks = re.findall(
        r"```([^\n`]*)\n(.*?)```",
        text,
        flags=re.DOTALL
    )

    if not blocks:
        return None

    # Prefer the last SQL-tagged block, regardless of case.
    sql_blocks = [
        content for tag, content in blocks
        if tag.strip().lower() == "sql"
    ]

    # If no SQL-tagged block exists, use the last plain fenced block.
    if sql_blocks:
        result = sql_blocks[-1]
    else:
        plain_blocks = [
            content for tag, content in blocks
            if not tag.strip()
        ]
        if not plain_blocks:
            return None
        result = plain_blocks[-1]

    return result.strip().rstrip(";").strip() or None


def build_prompt(dialect: str, ddl: str, question: str, hint: str | None = None, values: str | None = None) -> str:
    """Assemble the prompt text exactly as in the module docstring.

    dialect: "sqlite" -> "SQLite"; "postgres" or "postgresql" -> "PostgreSQL"; anything else -> ValueError.
    The Hint block appears only when hint is a non-empty string. No trailing newline.
    values: optional block of database values that match the question (t2sql/describe.py), placed after the schema.
    """
    names = {"sqlite": "SQLite", "postgres": "PostgreSQL", "postgresql": "PostgreSQL"}
    if dialect not in names:
        raise ValueError(f"unknown dialect: {dialect!r}")

    parts = [f"Dialect: {names[dialect]}", f"Schema:\n{ddl}"]
    if values:
        parts.append(f"Values in the database that match the question:\n{values}")
    if hint:
        parts.append(f"Hint: {hint}")
    parts.append(f"Question: {question}")
    return "\n\n".join(parts)


def schema_to_ddl(tables: list[Table]) -> str:
    """Render tables as CREATE TABLE statements, separated by one blank line.

    Per table (see the tests for exact expected output):
    - tables sorted by name with plain Python sorted()
    - one line per column, two-space indent: quoted name, then type (omit if ""), then " PRIMARY KEY" if set
    - then "PRIMARY KEY (a, b)" if the table has a composite key
    - then one "FOREIGN KEY (col) REFERENCES table(col)" line per foreign key
    - every line except the last one inside the parentheses ends with ","
    - if the column has examples, append " -- e.g. " + the formatted values joined by ", " (after the comma)
    - close with ");"
    """
    statements = []
    for table in sorted(tables, key=lambda t: t.name):
        # (line, comment) pairs; the comma goes between them, so comments never swallow it
        items = []
        for col in table.columns:
            line = quote_ident(col.name)
            if col.type:
                line += " " + col.type
            if col.primary_key:
                line += " PRIMARY KEY"
            notes = [col.description] if col.description else []
            if col.examples:
                notes.append("e.g. " + ", ".join(format_value(v) for v in col.examples))
            comment = " -- " + "; ".join(notes) if notes else ""
            items.append((line, comment))

        if table.primary_key:
            items.append(("PRIMARY KEY (" + ", ".join(quote_ident(c) for c in table.primary_key) + ")", ""))
        for fk in table.foreign_keys:
            items.append((
                f"FOREIGN KEY ({quote_ident(fk.column)}) "
                f"REFERENCES {quote_ident(fk.ref_table)}({quote_ident(fk.ref_column)})",
                "",
            ))

        lines = [
            "  " + line + ("," if i < len(items) - 1 else "") + comment
            for i, (line, comment) in enumerate(items)
        ]
        statements.append(f"CREATE TABLE {quote_ident(table.name)} (\n" + "\n".join(lines) + "\n);")

    return "\n\n".join(statements)


def read_sqlite_schema(path: Path, n_examples: int = N_EXAMPLES) -> list[Table]:
    """Read tables, columns, keys and sample values from a SQLite file. Open it read-only.

    Hints:
    - table names:  SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
    - columns:      PRAGMA table_info("<table>")       -> (cid, name, type, notnull, default, pk)
                    pk is 0, or the position (1, 2, ...) of the column inside the primary key
    - foreign keys: PRAGMA foreign_key_list("<table>") -> (id, seq, table, from, to, ...)
                    order them by the position of `from` in the table's columns
                    real data quirks (both happen in BIRD):
                      `to` is None when the DDL says just REFERENCES other_table -> it means that table's
                      primary key column, so look it up;
                      the same foreign key can be declared twice -> keep it once
    - examples: the n most frequent non-NULL values, ties broken by the value ascending.
      One GROUP BY query per column does it. Quote names with quote_ident() inside the query.
    """
    conn = sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)
    try:
        names = [r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        )]

        def primary_key(table: str) -> list[str]:
            info = conn.execute(f"PRAGMA table_info({quote_ident(table)})").fetchall()
            return [r[1] for r in sorted(info, key=lambda r: r[5]) if r[5] > 0]

        tables = []
        for name in names:
            info = conn.execute(f"PRAGMA table_info({quote_ident(name)})").fetchall()
            pk = primary_key(name)

            columns = []
            for _, col, col_type, _, _, _ in info:
                q = quote_ident(col)
                examples = [r[0] for r in conn.execute(
                    f"SELECT {q} FROM {quote_ident(name)} WHERE {q} IS NOT NULL "
                    f"GROUP BY {q} ORDER BY COUNT(*) DESC, {q} LIMIT ?",
                    (n_examples,),
                )]
                columns.append(Column(
                    name=col,
                    type=(col_type or "").upper(),
                    primary_key=len(pk) == 1 and col == pk[0],
                    examples=examples,
                ))

            position = {c.name: i for i, c in enumerate(columns)}
            foreign_keys = []
            for _, _, ref_table, from_col, to_col, *_ in conn.execute(f"PRAGMA foreign_key_list({quote_ident(name)})"):
                if to_col is None:
                    # "REFERENCES other_table" without a column means that table's primary key
                    ref_pk = primary_key(ref_table)
                    to_col = ref_pk[0] if ref_pk else from_col
                fk = ForeignKey(column=from_col, ref_table=ref_table, ref_column=to_col)
                if fk not in foreign_keys:  # BIRD declares some foreign keys twice
                    foreign_keys.append(fk)
            foreign_keys.sort(key=lambda f: position.get(f.column, len(position)))

            tables.append(Table(
                name=name,
                columns=columns,
                primary_key=pk if len(pk) > 1 else [],
                foreign_keys=foreign_keys,
            ))
        return tables
    finally:
        conn.close()
