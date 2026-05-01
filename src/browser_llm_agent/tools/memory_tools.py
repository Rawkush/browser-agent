import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime

DB_PATH = os.path.expanduser("~/.llm-agent/memory.db")


def _init_db(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memory (
            key     TEXT PRIMARY KEY,
            value   TEXT NOT NULL,
            saved   TEXT NOT NULL,
            updated TEXT
        )
    """)
    conn.commit()


@contextmanager
def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    _init_db(conn)
    try:
        yield conn
    finally:
        conn.close()


def memory_save(key: str, value: str) -> str:
    now = datetime.now().isoformat()
    with _conn() as conn:
        conn.execute("""
            INSERT INTO memory (key, value, saved, updated)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(key) DO UPDATE SET value=excluded.value, updated=excluded.updated
        """, (key, value, now, now))
        conn.commit()
    return f"Saved memory: {key}"


def memory_get(key: str) -> str:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM memory WHERE key = ?", (key,)).fetchone()
    if not row:
        return f"No memory found for key: {key}"
    ts = row["updated"] or row["saved"]
    return f"{row['key']}: {row['value']}\n(saved: {ts})"


def memory_list() -> str:
    with _conn() as conn:
        rows = conn.execute("SELECT key, value FROM memory ORDER BY key").fetchall()
    if not rows:
        return "No memories stored."
    return "\n".join(
        f"[{r['key']}] {r['value'][:80]}{'...' if len(r['value']) > 80 else ''}"
        for r in rows
    )


def memory_search(query: str) -> str:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT key, value FROM memory WHERE key LIKE ? OR value LIKE ? ORDER BY key",
            (f"%{query}%", f"%{query}%")
        ).fetchall()
    if not rows:
        return f"No memories matching: {query}"
    return "\n".join(f"[{r['key']}] {r['value'][:80]}" for r in rows)


def memory_delete(key: str) -> str:
    with _conn() as conn:
        cur = conn.execute("DELETE FROM memory WHERE key = ?", (key,))
        conn.commit()
    if cur.rowcount == 0:
        return f"No memory found for key: {key}"
    return f"Deleted memory: {key}"
