"""PostgreSQL schema reader. Uses the prompt.py formatter, so these pass once prompt.py is implemented."""
import psycopg
import pytest

from eval import testdb
from eval.config import pg_config
from t2sql.prompt import schema_to_ddl
from t2sql.schema_pg import read_postgres_schema

TINY_PG_DDL = """\
CREATE TABLE departments (
  id INTEGER PRIMARY KEY, -- e.g. 1, 2, 3
  name TEXT -- e.g. 'Engineering', 'HR', 'Sales'
);

CREATE TABLE employees (
  id INTEGER PRIMARY KEY, -- e.g. 1, 2, 3
  name TEXT, -- e.g. 'Aziz', 'Bobur', 'Dilnoza'
  dept_id INTEGER, -- e.g. 2, 1, 3
  salary NUMERIC(10,2), -- e.g. 4000.00, 3000.00, 3500.00
  hired DATE, -- e.g. '2019-03-01', '2020-07-15', '2021-01-10'
  manager_id INTEGER, -- e.g. 1, 2, 3
  FOREIGN KEY (dept_id) REFERENCES departments(id),
  FOREIGN KEY (manager_id) REFERENCES employees(id)
);"""


@pytest.fixture(scope="module")
def pg():
    cfg = pg_config()
    try:
        testdb.build_postgres(cfg)
    except psycopg.OperationalError as exc:
        pytest.skip(f"PostgreSQL not reachable (run docker compose up -d): {exc}")
    # read as the read-only user, the same way the pipeline will
    with psycopg.connect(host=cfg.host, port=cfg.port, dbname=testdb.DB_ID, connect_timeout=10,
                         user=cfg.ro_user, password=cfg.ro_password) as conn:
        yield cfg, conn


def test_tiny_pg_ddl(pg):
    _, conn = pg
    assert schema_to_ddl(read_postgres_schema(conn)) == TINY_PG_DDL


def test_composite_key_and_messy_names(pg):
    cfg, _ = pg
    admin = dict(host=cfg.host, port=cfg.port, user=cfg.admin_user, password=cfg.admin_password, connect_timeout=10)
    with psycopg.connect(dbname="postgres", autocommit=True, **admin) as c:
        c.execute('DROP DATABASE IF EXISTS "t2sql_messy" WITH (FORCE)')
        c.execute('CREATE DATABASE "t2sql_messy"')
    try:
        with psycopg.connect(dbname="t2sql_messy", **admin) as c:
            c.execute("""
                CREATE TABLE "Maktablar" ("Maktab ID" integer PRIMARY KEY, "Nomi" varchar(50), "Ходимлар сони" integer,
                                          "Ulush (%)" numeric(5,2));
                CREATE TABLE baholar (maktab_id integer REFERENCES "Maktablar"("Maktab ID"), yil integer,
                                      ball double precision, vaqt timestamp, PRIMARY KEY (maktab_id, yil));
                INSERT INTO "Maktablar" VALUES (1, '1-maktab', 40, 12.50), (2, 'O''g''il', 55, NULL);
                INSERT INTO baholar VALUES (1, 2024, 4.5, '2024-06-01 10:00'), (2, 2024, 4.5, NULL);
            """)
        with psycopg.connect(dbname="t2sql_messy", **admin) as c:
            tables = {t.name: t for t in read_postgres_schema(c)}
            ddl = schema_to_ddl(list(tables.values()))
        assert tables["baholar"].primary_key == ["maktab_id", "yil"]
        assert [c.type for c in tables["baholar"].columns] == ["INTEGER", "INTEGER", "DOUBLE PRECISION", "TIMESTAMP"]
        assert [c.type for c in tables["Maktablar"].columns] == ["INTEGER", "VARCHAR(50)", "INTEGER", "NUMERIC(5,2)"]
        assert '"Ulush (%)" NUMERIC(5,2) -- e.g. 12.50' in ddl  # a "%" in a name must not break the query
        assert 'FOREIGN KEY (maktab_id) REFERENCES "Maktablar"("Maktab ID")' in ddl
        assert "\"Nomi\" VARCHAR(50), -- e.g. '1-maktab', 'O''g''il'" in ddl

        # the DDL shown to the model must be valid PostgreSQL: rebuild it in an empty database
        with psycopg.connect(dbname="postgres", autocommit=True, **admin) as c:
            c.execute('DROP DATABASE IF EXISTS "t2sql_ddlcheck" WITH (FORCE)')
            c.execute('CREATE DATABASE "t2sql_ddlcheck"')
        with psycopg.connect(dbname="t2sql_ddlcheck", **admin) as c:
            c.execute(ddl)
            names = {r[0] for r in c.execute("SELECT tablename FROM pg_tables WHERE schemaname = 'public'")}
        assert names == {"Maktablar", "baholar"}
    finally:
        with psycopg.connect(dbname="postgres", autocommit=True, **admin) as c:
            c.execute('DROP DATABASE IF EXISTS "t2sql_messy" WITH (FORCE)')
            c.execute('DROP DATABASE IF EXISTS "t2sql_ddlcheck" WITH (FORCE)')
