"""Read-only warehouse access with a SQL guardrail.

Defence in depth:
  1. The DuckDB connection is opened read_only.
  2. Every statement is parsed with sqlglot: exactly one SELECT, only
     allow-listed tables, no file/network table functions.
  3. A row cap is always enforced.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import duckdb
import sqlglot
from sqlglot import exp

BLOCKED_FUNCTIONS = {
    "read_csv", "read_csv_auto", "read_parquet", "read_json", "read_json_auto",
    "read_text", "read_blob", "glob", "httpfs", "sqlite_scan", "postgres_scan",
}


class SQLGuardError(PermissionError):
    """Raised when a statement violates the read-only policy."""


def check_sql(sql: str, allowed_tables: set[str]) -> str:
    """Validate a statement; return it unchanged if it is safe."""
    try:
        statements = sqlglot.parse(sql, read="duckdb")
    except sqlglot.errors.ParseError as exc:
        raise SQLGuardError(f"SQL could not be parsed: {exc}") from exc
    statements = [s for s in statements if s is not None]
    if len(statements) != 1:
        raise SQLGuardError("Exactly one statement is allowed")
    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.Union, exp.Intersect, exp.Except)):
        raise SQLGuardError(f"Only SELECT queries are allowed, got {type(tree).__name__}")
    for node in tree.walk():
        if isinstance(node, (exp.Insert, exp.Update, exp.Delete, exp.Drop, exp.Create,
                             exp.Alter, exp.Command, exp.Copy)):
            raise SQLGuardError(f"Statement type {type(node).__name__} is not allowed")
    ctes = {c.alias_or_name.lower() for c in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        name = table.name.lower()
        if not name:
            raise SQLGuardError("Table functions are not allowed in FROM")
        if name not in ctes and name not in {t.lower() for t in allowed_tables}:
            raise SQLGuardError(f"Table '{table.name}' is not in the allow-list")
    for func in tree.find_all(exp.Func):
        fname = (func.sql_name() or "").lower()
        anon = func.name.lower() if isinstance(func, exp.Anonymous) else ""
        if fname in BLOCKED_FUNCTIONS or anon in BLOCKED_FUNCTIONS:
            raise SQLGuardError(f"Function '{fname or anon}' is not allowed")
    return sql


@dataclass
class QueryResult:
    columns: list[str]
    rows: list[dict[str, Any]]
    sql: str
    params: list[Any]
    governed: bool  # True when compiled from the semantic layer

    def preview(self, n: int = 20) -> list[dict[str, Any]]:
        return self.rows[:n]


class Warehouse:
    def __init__(self, path: str, allowed_tables: set[str], max_rows: int = 5000):
        self.path = path
        self.allowed_tables = allowed_tables
        self.max_rows = max_rows

    def execute(self, sql: str, params: list[Any] | None = None, governed: bool = False) -> QueryResult:
        check_sql(sql, self.allowed_tables)
        con = duckdb.connect(self.path, read_only=True)
        try:
            cur = con.execute(sql, params or [])
            columns = [d[0] for d in cur.description]
            raw = cur.fetchmany(self.max_rows)
        finally:
            con.close()
        rows = [{c: _jsonable(v) for c, v in zip(columns, r, strict=True)} for r in raw]
        return QueryResult(columns=columns, rows=rows, sql=sql, params=params or [], governed=governed)


def _jsonable(v: Any) -> Any:
    if hasattr(v, "isoformat"):
        return v.isoformat()
    if isinstance(v, float):
        return round(v, 4)
    return v
