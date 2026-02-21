import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

_db_path: Path | None = None


def init_db(db_path: Path):
    global _db_path
    _db_path = db_path
    with _conn() as conn:
        conn.executescript("""
            PRAGMA journal_mode=WAL;
            PRAGMA foreign_keys=ON;

            CREATE TABLE IF NOT EXISTS sessions (
                id           TEXT PRIMARY KEY,
                title        TEXT DEFAULT 'New Chat',
                created_at   TEXT NOT NULL,
                message_count INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                sources    TEXT DEFAULT '[]',
                created_at TEXT NOT NULL,
                FOREIGN KEY(session_id) REFERENCES sessions(id) ON DELETE CASCADE
            );

            CREATE TABLE IF NOT EXISTS documents (
                id          TEXT PRIMARY KEY,
                filename    TEXT NOT NULL,
                file_type   TEXT NOT NULL,
                file_size   INTEGER NOT NULL,
                chunk_count INTEGER DEFAULT 0,
                status      TEXT DEFAULT 'processing',
                uploaded_at TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_msg_session ON messages(session_id);
        """)


def _get_conn() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _conn():
    conn = _get_conn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── Sessions ──────────────────────────────────────────────────────────────────

def create_session(session_id: str) -> dict:
    now = _now()
    with _conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sessions(id, created_at) VALUES(?,?)",
            (session_id, now),
        )
    return get_session(session_id)


def get_session(session_id: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
    return dict(row) if row else None


def list_sessions() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM sessions ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def update_session_title(session_id: str, title: str):
    with _conn() as conn:
        conn.execute(
            "UPDATE sessions SET title=? WHERE id=?", (title[:80], session_id)
        )


def increment_session_count(session_id: str):
    with _conn() as conn:
        conn.execute(
            "UPDATE sessions SET message_count=message_count+1 WHERE id=?",
            (session_id,),
        )


def delete_session(session_id: str):
    with _conn() as conn:
        conn.execute("DELETE FROM sessions WHERE id=?", (session_id,))


# ── Messages ──────────────────────────────────────────────────────────────────

def add_message(session_id: str, role: str, content: str, sources: list | None = None):
    with _conn() as conn:
        conn.execute(
            "INSERT INTO messages(session_id, role, content, sources, created_at) VALUES(?,?,?,?,?)",
            (session_id, role, content, json.dumps(sources or []), _now()),
        )
    increment_session_count(session_id)


def get_messages(session_id: str) -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM messages WHERE session_id=? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [{**dict(r), "sources": json.loads(r["sources"] or "[]")} for r in rows]


# ── Documents ─────────────────────────────────────────────────────────────────

def create_document(doc_id: str, filename: str, file_type: str, file_size: int) -> dict:
    with _conn() as conn:
        conn.execute(
            "INSERT INTO documents(id, filename, file_type, file_size, uploaded_at) VALUES(?,?,?,?,?)",
            (doc_id, filename, file_type, file_size, _now()),
        )
    return get_document(doc_id)


def update_doc_status(doc_id: str, status: str, chunk_count: int = 0):
    with _conn() as conn:
        conn.execute(
            "UPDATE documents SET status=?, chunk_count=? WHERE id=?",
            (status, chunk_count, doc_id),
        )


def get_document(doc_id: str) -> dict | None:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM documents WHERE id=?", (doc_id,)).fetchone()
    return dict(row) if row else None


def list_documents() -> list[dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT * FROM documents ORDER BY uploaded_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def delete_document(doc_id: str):
    with _conn() as conn:
        conn.execute("DELETE FROM documents WHERE id=?", (doc_id,))


def get_stats() -> dict:
    with _conn() as conn:
        docs = conn.execute(
            "SELECT COUNT(*) FROM documents WHERE status='indexed'"
        ).fetchone()[0]
        chunks = conn.execute(
            "SELECT COALESCE(SUM(chunk_count),0) FROM documents WHERE status='indexed'"
        ).fetchone()[0]
        size = conn.execute(
            "SELECT COALESCE(SUM(file_size),0) FROM documents WHERE status='indexed'"
        ).fetchone()[0]
        sessions = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        messages = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    return {
        "total_documents": docs,
        "total_chunks": chunks,
        "total_size_bytes": size,
        "total_sessions": sessions,
        "total_messages": messages,
    }
