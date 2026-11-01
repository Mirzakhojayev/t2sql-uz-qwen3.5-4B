"""A tiny database built identically in SQLite and PostgreSQL, used to test the harness.

Small enough that every expected answer can be checked by hand:
  departments: 3 rows (Sales, Engineering, HR)
  employees:   7 rows, salaries 3000-7000, one NULL manager, one NULL dept, two people with the same salary
"""
import sqlite3
from pathlib import Path

import psycopg

from .config import PgConfig

DB_ID = "tiny"

_DDL = {
    "sqlite": """
        CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY, name TEXT NOT NULL, dept_id INTEGER REFERENCES departments(id),
            salary REAL NOT NULL, hired TEXT NOT NULL, manager_id INTEGER REFERENCES employees(id));
    """,
    "postgres": """
        CREATE TABLE departments (id INTEGER PRIMARY KEY, name TEXT NOT NULL);
        CREATE TABLE employees (
            id INTEGER PRIMARY KEY, name TEXT NOT NULL, dept_id INTEGER REFERENCES departments(id),
            salary NUMERIC(10,2) NOT NULL, hired DATE NOT NULL, manager_id INTEGER REFERENCES employees(id));
    """,
}

_DATA = """
    INSERT INTO departments VALUES (1, 'Sales'), (2, 'Engineering'), (3, 'HR');
    INSERT INTO employees VALUES
        (1, 'Aziz',    2, 7000, '2019-03-01', NULL),
        (2, 'Dilnoza', 2, 6000, '2020-07-15', 1),
        (3, 'Bobur',   1, 4000, '2021-01-10', 1),
        (4, 'Madina',  1, 4000, '2022-05-20', 3),
        (5, 'Jasur',   3, 3000, '2022-11-02', 1),
        (6, 'Nodira',  2, 5500, '2023-02-28', 2),
        (7, 'Sardor',  NULL, 3500, '2023-09-09', NULL);
"""


def build_sqlite(path: Path) -> None:
    path.unlink(missing_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(_DDL["sqlite"] + _DATA)
    conn.commit()
    conn.close()


def build_postgres(cfg: PgConfig, db_name: str = DB_ID) -> None:
    """(Re)create the database with the admin user; the harness reads it as the read-only user."""
    admin = dict(host=cfg.host, port=cfg.port, user=cfg.admin_user, password=cfg.admin_password, connect_timeout=10)
    with psycopg.connect(dbname="postgres", autocommit=True, **admin) as conn:
        conn.execute(f'DROP DATABASE IF EXISTS "{db_name}" WITH (FORCE)')
        conn.execute(f'CREATE DATABASE "{db_name}"')
    with psycopg.connect(dbname=db_name, **admin) as conn:
        conn.execute(_DDL["postgres"] + _DATA)
