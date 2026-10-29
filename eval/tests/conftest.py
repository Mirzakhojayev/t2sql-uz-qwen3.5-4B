import psycopg
import pytest

from eval import testdb
from eval.backends import PostgresBackend, SQLiteBackend
from eval.config import pg_config
from eval.harness import Evaluator

TIMEOUT_S = 1.0


@pytest.fixture(scope="session")
def sqlite_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("sqlite")
    testdb.build_sqlite(d / f"{testdb.DB_ID}.sqlite")
    return d


@pytest.fixture(scope="session")
def pg_ready():
    cfg = pg_config()
    try:
        testdb.build_postgres(cfg)
    except psycopg.OperationalError as exc:
        pytest.skip(f"PostgreSQL not reachable (run docker compose up -d): {exc}")
    return cfg


@pytest.fixture(params=["sqlite", "postgres"])
def dialect(request, sqlite_dir):
    """Every test using this fixture runs once per dialect."""
    if request.param == "postgres":
        request.getfixturevalue("pg_ready")
    return request.param


@pytest.fixture
def backend(dialect, sqlite_dir):
    b = SQLiteBackend(sqlite_dir, TIMEOUT_S) if dialect == "sqlite" else PostgresBackend(pg_config(), TIMEOUT_S)
    yield b
    b.close()


@pytest.fixture
def evaluator(dialect, backend):
    return Evaluator({dialect: backend})
