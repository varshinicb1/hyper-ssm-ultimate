"""
SQLite-backed session persistence for Infinite Context Memory.

Sessions survive server restarts. Each session stores:
  - Memory state (hyperbolic state vector + scalars)
  - Conversation history
  - Metadata (created_at, last_active, turn count)
"""

import json
import os
import pickle
import sqlite3
import time
import threading
from typing import Dict, List, Optional, Any


class SessionStore:
    """Thread-safe SQLite store for ICM sessions."""

    def __init__(self, db_path: str):
        self.db_path = os.path.abspath(db_path)
        os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if not hasattr(self._local, "conn") or self._local.conn is None:
            self._local.conn = sqlite3.connect(self.db_path)
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
        return self._local.conn

    def _init_db(self):
        conn = self._get_conn()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS sessions (
                session_id TEXT PRIMARY KEY,
                memory_state BLOB NOT NULL,
                history BLOB NOT NULL,
                state_dim INTEGER NOT NULL,
                num_scales INTEGER NOT NULL,
                turn_count INTEGER DEFAULT 0,
                created_at REAL NOT NULL,
                last_active REAL NOT NULL,
                metadata TEXT DEFAULT '{}'
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_sessions_active
            ON sessions(last_active)
        """)
        conn.commit()

    def save(
        self,
        session_id: str,
        memory_state: dict,
        history: List[Dict[str, str]],
        state_dim: int,
        num_scales: int,
        turn_count: int = 0,
        metadata: Optional[dict] = None,
    ) -> None:
        now = time.time()
        mem_blob = pickle.dumps(memory_state)
        hist_blob = pickle.dumps(history)
        meta_json = json.dumps(metadata or {})

        conn = self._get_conn()
        conn.execute("""
            INSERT INTO sessions
                (session_id, memory_state, history, state_dim, num_scales,
                 turn_count, created_at, last_active, metadata)
            VALUES (?, ?, ?, ?, ?, ?, COALESCE(
                (SELECT created_at FROM sessions WHERE session_id = ?), ?
            ), ?, ?)
            ON CONFLICT(session_id) DO UPDATE SET
                memory_state = excluded.memory_state,
                history = excluded.history,
                turn_count = excluded.turn_count,
                last_active = excluded.last_active
        """, (
            session_id, mem_blob, hist_blob, state_dim, num_scales,
            turn_count, session_id, now, now, meta_json,
        ))
        conn.commit()

    def load(self, session_id: str) -> Optional[dict]:
        conn = self._get_conn()
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,)
        ).fetchone()
        if row is None:
            return None
        return {
            "session_id": row[0],
            "memory_state": pickle.loads(row[1]),
            "history": pickle.loads(row[2]),
            "state_dim": row[3],
            "num_scales": row[4],
            "turn_count": row[5],
            "created_at": row[6],
            "last_active": row[7],
            "metadata": json.loads(row[8]),
        }

    def delete(self, session_id: str) -> bool:
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM sessions WHERE session_id = ?", (session_id,)
        )
        conn.commit()
        return cursor.rowcount > 0

    def list_sessions(self) -> List[dict]:
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT session_id, turn_count, created_at, last_active, state_dim, "
            "num_scales FROM sessions ORDER BY last_active DESC"
        ).fetchall()
        return [
            {
                "session_id": r[0],
                "turn_count": r[1],
                "created_at": r[2],
                "last_active": r[3],
                "state_dim": r[4],
                "num_scales": r[5],
            }
            for r in rows
        ]

    def count(self) -> int:
        conn = self._get_conn()
        return conn.execute(
            "SELECT COUNT(*) FROM sessions"
        ).fetchone()[0]

    def cleanup_expired(self, ttl_seconds: int) -> int:
        cutoff = time.time() - ttl_seconds
        conn = self._get_conn()
        cursor = conn.execute(
            "DELETE FROM sessions WHERE last_active < ?", (cutoff,)
        )
        conn.commit()
        return cursor.rowcount

    def close(self):
        if hasattr(self._local, "conn") and self._local.conn is not None:
            self._local.conn.close()
            self._local.conn = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
