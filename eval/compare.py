"""Turn result sets into comparable signatures.

Rules: row order is ignored unless the gold SQL has a top-level ORDER BY; duplicate rows count
(multiset); column order matters; numbers are compared after rounding to 6 decimals, so 3, 3.0 and
Decimal('3.00') are equal; dates compare as ISO strings.
"""
import datetime
import json
import re
from collections import Counter
from decimal import Decimal

import sqlglot
from sqlglot import exp

_SQLGLOT_DIALECT = {"sqlite": "sqlite", "postgres": "postgres", "postgresql": "postgres"}


def normalize_value(v):
    if v is None or isinstance(v, str):
        return v
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float, Decimal)):
        f = round(float(v), 6)
        return int(f) if f.is_integer() else f
    if isinstance(v, (datetime.datetime, datetime.date, datetime.time)):
        return v.isoformat()
    if isinstance(v, (bytes, bytearray, memoryview)):
        return bytes(v).hex()
    if isinstance(v, (list, tuple)):
        return tuple(normalize_value(x) for x in v)
    if isinstance(v, dict):
        return json.dumps(v, sort_keys=True, ensure_ascii=False)
    return str(v)


def normalize_rows(rows: list[tuple]) -> list[tuple]:
    return [tuple(normalize_value(v) for v in row) for row in rows]


def signature(rows: list[tuple], ordered: bool):
    """Hashable form of a normalized result set; equal signatures mean equal results."""
    if ordered:
        return ("ordered", tuple(rows))
    return ("unordered", frozenset(Counter(rows).items()))


def is_ordered(gold_sql: str, dialect: str) -> bool:
    """True if the gold query has a top-level ORDER BY, so row order is part of the answer."""
    try:
        tree = sqlglot.parse_one(gold_sql, read=_SQLGLOT_DIALECT[dialect])
        return isinstance(tree, exp.Query) and tree.args.get("order") is not None
    except Exception:
        return re.search(r"\border\s+by\b", gold_sql, re.IGNORECASE) is not None
