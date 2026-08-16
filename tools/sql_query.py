"""Generic read-only SQL query tool over the evidence-base SQLite database
(findings, facts tables — see tools/db.py for schema).

SELECT-only by design — this tool is for ad-hoc exploration, not to
mutate data, so any non-SELECT statement is rejected before it reaches
the database. Structured, injection-safe lookups scoped to a run/
hypothesis (used by the analyst and synthesizer agents) live in
tools/db.py instead of going through this free-form tool.
"""

import re
import sqlite3
from pathlib import Path

import pandas as pd
from langchain_core.tools import tool

from tools.db import DB_PATH as DEFAULT_DB_PATH

_SELECT_ONLY = re.compile(r"^\s*(WITH\b.*?\bSELECT|SELECT)\b", re.IGNORECASE | re.DOTALL)
_FORBIDDEN = re.compile(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|REPLACE|ATTACH|PRAGMA)\b", re.IGNORECASE)


@tool
def run_sql_query(query: str, db_path: str | None = None) -> dict:
    """Run a read-only SQL query against the evidence-base SQLite database.

    Returns a dict: {columns, rows, row_count, error}. `rows` is a list of
    row dicts (JSON-friendly). On failure, or if the query isn't a SELECT,
    `error` is set and the other fields are empty.
    """
    if not _SELECT_ONLY.match(query) or _FORBIDDEN.search(query):
        return {"columns": [], "rows": [], "row_count": 0, "error": "Only read-only SELECT queries are allowed."}

    path = Path(db_path) if db_path else DEFAULT_DB_PATH
    if not path.exists():
        return {
            "columns": [], "rows": [], "row_count": 0,
            "error": f"Database not found at {path}. Run scripts/seed_db.py first.",
        }

    try:
        with sqlite3.connect(path) as conn:
            df = pd.read_sql_query(query, conn)
    except (sqlite3.Error, pd.errors.DatabaseError) as exc:
        return {"columns": [], "rows": [], "row_count": 0, "error": str(exc)}

    return {
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records"),
        "row_count": len(df),
        "error": None,
    }
