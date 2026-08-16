"""Initialize the evidence-base schema (findings + facts tables).

This does NOT seed any fictional data. Running SQL over invented company
data produced analysis that looked rigorous but was meaningless — the
tables now start empty and are populated at runtime by the researcher
agent as it gathers evidence with real source URLs. Run this once (or
whenever data/business.db doesn't exist yet) so the schema is in place
before a graph run.
"""

import sqlite3

from tools.db import DB_PATH, init_db


def main() -> None:
    # One-time migration: drop the old fictional "sales" table from the
    # previous scaffold version, if a pre-existing business.db has it.
    if DB_PATH.exists():
        with sqlite3.connect(DB_PATH) as conn:
            conn.execute("DROP TABLE IF EXISTS sales")
            conn.commit()

    init_db()
    print(f"Evidence base ready at {DB_PATH} (findings, facts tables). No rows seeded.")


if __name__ == "__main__":
    main()
