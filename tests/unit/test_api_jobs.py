"""Unit tests for src/api/jobs.py — focused on the JobStore contract."""

from __future__ import annotations

import threading
import time

from src.api.jobs import JobStore


class TestJobStoreCreate:
    def test_create_returns_pending_job(self) -> None:
        store = JobStore()
        job = store.create()
        assert job.status == "pending"
        assert job.job_id  # non-empty UUID

    def test_each_job_has_unique_id(self) -> None:
        store = JobStore()
        ids = {store.create().job_id for _ in range(50)}
        assert len(ids) == 50

    def test_created_at_is_set_to_now(self) -> None:
        store = JobStore()
        before = time.time()
        job = store.create()
        after = time.time()
        assert before <= job.created_at <= after


class TestJobStoreGet:
    def test_get_returns_existing(self) -> None:
        store = JobStore()
        job = store.create()
        assert store.get(job.job_id) is job

    def test_get_returns_none_for_unknown(self) -> None:
        store = JobStore()
        assert store.get("does-not-exist") is None


class TestJobStoreUpdate:
    def test_update_status(self) -> None:
        store = JobStore()
        j = store.create()
        store.update(j.job_id, status="running")
        assert store.get(j.job_id).status == "running"  # type: ignore[union-attr]

    def test_update_attaches_result(self) -> None:
        store = JobStore()
        j = store.create()
        store.update(j.job_id, status="completed", result={"n": 5})
        got = store.get(j.job_id)
        assert got is not None
        assert got.status == "completed"
        assert got.result == {"n": 5}

    def test_update_attaches_error(self) -> None:
        store = JobStore()
        j = store.create()
        store.update(j.job_id, status="failed", error="boom")
        got = store.get(j.job_id)
        assert got is not None
        assert got.status == "failed"
        assert got.error == "boom"

    def test_update_unknown_returns_none(self) -> None:
        store = JobStore()
        assert store.update("does-not-exist", status="completed") is None

    def test_update_advances_updated_at(self) -> None:
        store = JobStore()
        j = store.create()
        first = j.updated_at
        time.sleep(0.01)
        store.update(j.job_id, status="running")
        got = store.get(j.job_id)
        assert got is not None
        assert got.updated_at > first


class TestJobStoreListActive:
    def test_lists_only_pending_and_running(self) -> None:
        store = JobStore()
        a = store.create()
        b = store.create()
        c = store.create()
        store.update(a.job_id, status="running")
        store.update(b.job_id, status="completed")
        # c stays pending
        active = store.list_active()
        ids = {j.job_id for j in active}
        assert a.job_id in ids
        assert c.job_id in ids
        assert b.job_id not in ids


class TestJobStoreEviction:
    def test_evicts_oldest_completed_when_overflow(self) -> None:
        store = JobStore(max_size=3)
        a = store.create()
        store.update(a.job_id, status="completed")
        b = store.create()
        store.update(b.job_id, status="completed")
        c = store.create()
        d = store.create()  # 4th -> a should be evicted
        assert store.get(a.job_id) is None
        for j in (b, c, d):
            assert store.get(j.job_id) is not None

    def test_active_jobs_never_evicted(self) -> None:
        store = JobStore(max_size=2)
        # Both active — must not evict either even though we exceed max_size.
        a = store.create()
        store.update(a.job_id, status="running")
        b = store.create()
        store.update(b.job_id, status="running")
        c = store.create()  # third active, store now over capacity
        # All three remain because none can be evicted.
        for j in (a, b, c):
            assert store.get(j.job_id) is not None


class TestJobStoreThreadSafety:
    def test_concurrent_creates_yield_unique_ids(self) -> None:
        store = JobStore()
        ids: list[str] = []
        lock = threading.Lock()

        def worker() -> None:
            for _ in range(20):
                j = store.create()
                with lock:
                    ids.append(j.job_id)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(ids) == 8 * 20
        assert len(set(ids)) == 8 * 20  # all unique
