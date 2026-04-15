"""Unit tests for src/clients/_cache.py — SQLite WAL cache."""

from __future__ import annotations

import time
from pathlib import Path

from src.clients._cache import default_db_path, get, init_db, prune_expired, put


class TestDefaultDbPath:
    def test_returns_path(self) -> None:
        p = default_db_path()
        assert isinstance(p, Path)
        assert p.name == "api_cache.db"
        assert p.parent.name == ".cache"

    def test_is_under_project_root(self) -> None:
        p = default_db_path()
        # Should be at project root, not inside src/clients/
        assert "src" not in str(p)


class TestInitDb:
    def test_creates_db(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_cache.db"
        conn = init_db(db_path)
        conn.close()
        assert db_path.exists()

    def test_creates_cache_table(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_cache.db"
        conn = init_db(db_path)
        cursor = conn.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='cache'")
        assert cursor.fetchone() is not None
        conn.close()

    def test_wal_mode(self, tmp_path: Path) -> None:
        db_path = tmp_path / "test_cache.db"
        conn = init_db(db_path)
        cursor = conn.execute("PRAGMA journal_mode")
        mode = cursor.fetchone()[0]
        assert mode == "wal"
        conn.close()


class TestGetPut:
    def test_put_and_get(self, tmp_path: Path) -> None:
        db_path = tmp_path / "cache.db"
        put(db_path, "key1", '{"data": "value"}', ttl_seconds=3600)
        result = get(db_path, "key1")
        assert result == '{"data": "value"}'

    def test_get_missing_key_returns_none(self, tmp_path: Path) -> None:
        db_path = tmp_path / "cache.db"
        result = get(db_path, "nonexistent")
        assert result is None

    def test_get_expired_key_returns_none(self, tmp_path: Path) -> None:
        db_path = tmp_path / "cache.db"
        # TTL of 0 seconds — already expired
        put(db_path, "expired_key", "some_value", ttl_seconds=0)
        # Sleep a tiny bit to ensure expiry
        time.sleep(0.01)
        result = get(db_path, "expired_key")
        assert result is None

    def test_put_upserts(self, tmp_path: Path) -> None:
        db_path = tmp_path / "cache.db"
        put(db_path, "key1", "value1", ttl_seconds=3600)
        put(db_path, "key1", "value2", ttl_seconds=3600)
        result = get(db_path, "key1")
        assert result == "value2"

    def test_multiple_keys(self, tmp_path: Path) -> None:
        db_path = tmp_path / "cache.db"
        put(db_path, "a", "v1", ttl_seconds=3600)
        put(db_path, "b", "v2", ttl_seconds=3600)
        assert get(db_path, "a") == "v1"
        assert get(db_path, "b") == "v2"

    def test_creates_directory_if_needed(self, tmp_path: Path) -> None:
        db_path = tmp_path / "subdir" / "cache.db"
        # Should not raise even if directory doesn't exist yet
        put(db_path, "k", "v", ttl_seconds=3600)
        assert get(db_path, "k") == "v"


class TestPruneExpired:
    def test_prune_removes_expired(self, tmp_path: Path) -> None:
        db_path = tmp_path / "cache.db"
        put(db_path, "expired", "val", ttl_seconds=0)
        time.sleep(0.01)
        deleted = prune_expired(db_path)
        assert deleted == 1
        assert get(db_path, "expired") is None

    def test_prune_keeps_live_entries(self, tmp_path: Path) -> None:
        db_path = tmp_path / "cache.db"
        put(db_path, "live", "val", ttl_seconds=3600)
        deleted = prune_expired(db_path)
        assert deleted == 0
        assert get(db_path, "live") == "val"

    def test_prune_empty_db_returns_zero(self, tmp_path: Path) -> None:
        db_path = tmp_path / "cache.db"
        deleted = prune_expired(db_path)
        assert deleted == 0
