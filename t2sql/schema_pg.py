"""Read a PostgreSQL database into the same Table/Column objects as read_sqlite_schema, so one formatter
(schema_to_ddl) serves both dialects. Only the `public` schema is read."""
import psycopg

from .prompt import N_EXAMPLES, Column, ForeignKey, Table, quote_ident

_COLUMNS = """
    SELECT cl.relname, a.attname, format_type(a.atttypid, a.atttypmod)
    FROM pg_attribute a
    JOIN pg_class cl ON cl.oid = a.attrelid
    JOIN pg_namespace n ON n.oid = cl.relnamespace
    WHERE n.nspname = 'public' AND cl.relkind IN ('r', 'p') AND a.attnum > 0 AND NOT a.attisdropped
    ORDER BY cl.relname, a.attnum
"""

# format_type() spells some types out in full; use the short names people write in DDL
_SHORT_TYPES = [
    ("character varying", "varchar"), ("timestamp without time zone", "timestamp"),
    ("timestamp with time zone", "timestamptz"), ("time without time zone", "time"), ("character", "char"),
]


def _short_type(t: str) -> str:
    for long, short in _SHORT_TYPES:
        if t.startswith(long):
            t = short + t[len(long):]
            break
    return t.upper()

# pg_catalog instead of information_schema: the latter hides constraints of tables we don't own
_PRIMARY_KEYS = """
    SELECT cl.relname, a.attname, array_position(con.conkey, a.attnum) AS pos
    FROM pg_constraint con
    JOIN pg_class cl ON cl.oid = con.conrelid
    JOIN pg_namespace n ON n.oid = cl.relnamespace
    JOIN pg_attribute a ON a.attrelid = con.conrelid AND a.attnum = ANY (con.conkey)
    WHERE con.contype = 'p' AND n.nspname = 'public'
    ORDER BY cl.relname, pos
"""

_FOREIGN_KEYS = """
    SELECT cl.relname, a.attname, rcl.relname, ra.attname
    FROM pg_constraint con
    JOIN pg_class cl ON cl.oid = con.conrelid
    JOIN pg_namespace n ON n.oid = cl.relnamespace
    JOIN pg_class rcl ON rcl.oid = con.confrelid
    CROSS JOIN LATERAL unnest(con.conkey, con.confkey) AS k(col, refcol)
    JOIN pg_attribute a ON a.attrelid = con.conrelid AND a.attnum = k.col
    JOIN pg_attribute ra ON ra.attrelid = con.confrelid AND ra.attnum = k.refcol
    WHERE con.contype = 'f' AND n.nspname = 'public'
"""


def read_postgres_schema(conn: psycopg.Connection, n_examples: int = N_EXAMPLES) -> list[Table]:
    """conn: an open connection to the database to describe (read-only is enough)."""
    cols: dict[str, list[Column]] = {}
    for table, column, dtype in conn.execute(_COLUMNS):
        cols.setdefault(table, []).append(Column(column, _short_type(dtype)))

    pks: dict[str, list[str]] = {}
    for table, column, _ in conn.execute(_PRIMARY_KEYS):
        pks.setdefault(table, []).append(column)

    fks: dict[str, list[ForeignKey]] = {}
    for table, column, ref_table, ref_column in conn.execute(_FOREIGN_KEYS):
        fk = ForeignKey(column, ref_table, ref_column)
        if fk not in fks.setdefault(table, []):  # two identical constraints would print the same line twice
            fks[table].append(fk)

    tables = []
    for name, columns in cols.items():
        pk = pks.get(name, [])
        position = {c.name: i for i, c in enumerate(columns)}
        for c in columns:
            c.primary_key = len(pk) == 1 and c.name == pk[0]
            q = quote_ident(c.name)
            # same rule as SQLite: most frequent non-NULL values, ties by value ascending.
            # No query parameters: with parameters, psycopg treats any "%" as a placeholder, and BIRD has
            # column names like "Percent (%) Eligible Free (K-12)".
            c.examples = [r[0] for r in conn.execute(
                f"SELECT {q} FROM {quote_ident(name)} WHERE {q} IS NOT NULL "
                f"GROUP BY {q} ORDER BY count(*) DESC, {q} LIMIT {int(n_examples)}")]
        tables.append(Table(
            name=name,
            columns=columns,
            primary_key=pk if len(pk) > 1 else [],
            foreign_keys=sorted(fks.get(name, []), key=lambda f: position[f.column]),
        ))
    conn.rollback()
    return tables
