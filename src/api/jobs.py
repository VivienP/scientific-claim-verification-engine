"""In-memory job store for the Phase C lite API.

A single-process biotech on-prem deployment does not need Redis or a
database. This store keeps job state in a thread-safe dict guarded by a
``threading.Lock`` so background tasks running in the FastAPI thread pool
can update without races.

Lifecycle:
    create()  -> Job(status='pending')
    update()  -> mutates status, sets updated_at, attaches result/error
    get()     -> returns current snapshot
    list_active() -> for /health and load-shedding hooks

Eviction:
    A bounded LRU keeps the last ``max_size`` jobs. On overflow the oldest
    *completed* or *failed* job is evicted; running jobs are never evicted.
    Default 1000 — enough for ~1 month of pharma usage.

For Phase D this becomes a Postgres-backed store; the Protocol contract
in ``app.py`` lets us swap implementations without touching routes.
"""

from __future__ import annotations

import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from src.api.models import JobStatus


@dataclass
class Job:
    """A single verification job's mutable state.

    Mutable on purpose — workers update status / attach results / set errors.
    Reads happen under the store's lock so callers see a consistent snapshot.
    """

    job_id: str
    status: JobStatus
    created_at: float
    updated_at: float
    run_id: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = field(default=None)


class JobStore:
    """Thread-safe in-memory LRU store of Job objects."""

    def __init__(self, max_size: int = 1000) -> None:
        self._jobs: OrderedDict[str, Job] = OrderedDict()
        self._lock = threading.Lock()
        self._max_size = max_size

    def create(self) -> Job:
        """Create a new pending job with a fresh UUID."""
        now = time.time()
        job = Job(
            job_id=str(uuid.uuid4()),
            status="pending",
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._jobs[job.job_id] = job
            self._evict_if_needed()
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def update(
        self,
        job_id: str,
        *,
        status: JobStatus | None = None,
        run_id: str | None = None,
        error: str | None = None,
        result: dict[str, Any] | None = None,
    ) -> Job | None:
        """Update a subset of fields. Returns the updated Job or None if missing."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job is None:
                return None
            if status is not None:
                job.status = status
            if run_id is not None:
                job.run_id = run_id
            if error is not None:
                job.error = error
            if result is not None:
                job.result = result
            job.updated_at = time.time()
            return job

    def list_active(self) -> list[Job]:
        """Return jobs whose status is 'pending' or 'running'."""
        with self._lock:
            return [j for j in self._jobs.values() if j.status in {"pending", "running"}]

    def _evict_if_needed(self) -> None:
        """Drop the oldest non-active job until we are within max_size.

        Caller must hold the lock. Active jobs are never evicted, even if
        the store grows past ``max_size`` — a stuck worker should not lose
        its job. In practice that requires monitoring active count, which
        ``/health`` exposes.
        """
        while len(self._jobs) > self._max_size:
            for jid, j in self._jobs.items():
                if j.status in {"completed", "failed"}:
                    del self._jobs[jid]
                    break
            else:
                # All remaining jobs are active. Stop evicting.
                return
