"""SQLite WAL-mode cache for API responses."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import structlog

logger: structlog.BoundLogger = structlog.get_logger(__name__)

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL,
    expires_at REAL NOT NULL
)
"""


def default_db_path() -> Path:
    """Return path to default cache DB: <project_root>/.cache/api_cache.db."""
    # This file lives at src/clients/_cache.py → go up 3 levels to project root
    return Path(__file__).parent.parent.parent / ".cache" / "api_cache.db"


def init_db(db_path: Path) -> sqlite3.Connection:
    """Open or create SQLite WAL database; create cache table if absent."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute(_CREATE_TABLE)
    conn.commit()
    return conn


def get(db_path: Path, cache_key: str) -> str | None:
    """Return cached value for key if present and not expired. None otherwise."""
    conn = init_db(db_path)
    try:
        cursor = conn.execute(
            "SELECT value FROM cache WHERE key = ? AND expires_at > ?",
            (cache_key, time.time()),
        )
        row = cursor.fetchone()
        return str(row[0]) if row else None
    finally:
        conn.close()


def put(db_path: Path, cache_key: str, value: str, ttl_seconds: int) -> None:
    """Upsert value with expiry = now + ttl_seconds."""
    conn = init_db(db_path)
    try:
        expires_at = time.time() + ttl_seconds
        conn.execute(
            "INSERT OR REPLACE INTO cache (key, value, expires_at) VALUES (?, ?, ?)",
            (cache_key, value, expires_at),
        )
        conn.commit()
    finally:
        conn.close()


def prune_expired(db_path: Path) -> int:
    """Delete expired rows. Returns count of deleted rows."""
    conn = init_db(db_path)
    try:
        cursor = conn.execute("DELETE FROM cache WHERE expires_at <= ?", (time.time(),))
        conn.commit()
        deleted: int = cursor.rowcount
        logger.info("cache_pruned", deleted_rows=deleted)
        return deleted
    finally:
        conn.close()
