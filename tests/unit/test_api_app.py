"""Unit tests for src/api/ — fully offline, real worker swapped for a fake."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api.app import create_app
from src.api.jobs import JobStore
from src.api.models import VerifyRequest

_API_KEY = "test-key-do-not-use-in-prod"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _set_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test runs with a fixed API key in env."""
    monkeypatch.setenv("VERIFIER_API_KEY", _API_KEY)


@pytest.fixture
def fake_worker_factory() -> Any:
    """Return a factory that creates worker fakes parametrised by behaviour."""

    def make(
        *,
        succeed: bool = True,
        result: dict[str, Any] | None = None,
        error: str = "boom",
        sleep_s: float = 0.0,
    ) -> Any:
        calls: list[tuple[str, VerifyRequest]] = []

        def fake_worker(
            job_id: str,
            req: VerifyRequest,
            store: JobStore,
            *,
            runs_root: Path = Path("reports/runs"),
        ) -> None:
            calls.append((job_id, req))
            if sleep_s:
                time.sleep(sleep_s)
            store.update(job_id, status="running")
            if succeed:
                store.update(
                    job_id,
                    status="completed",
                    run_id=f"api-{job_id[:8]}",
                    result=result or {"n_claims": 3, "verdict_counts": {"supported": 3}},
                )
            else:
                store.update(job_id, status="failed", error=error)

        # Attach the call list so tests can inspect.
        fake_worker.calls = calls  # type: ignore[attr-defined]
        return fake_worker

    return make


@pytest.fixture
def client(fake_worker_factory: Any, tmp_path: Path) -> TestClient:
    """A TestClient with an isolated JobStore + a successful fake worker."""
    app = create_app(
        job_store=JobStore(),
        worker=fake_worker_factory(),
        runs_root=tmp_path,
    )
    return TestClient(app)


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_returns_200(self, client: TestClient) -> None:
        r = client.get("/health")
        assert r.status_code == 200

    def test_schema(self, client: TestClient) -> None:
        body = client.get("/health").json()
        assert body["status"] == "ok"
        assert "version" in body
        assert body["n_jobs_active"] == 0

    def test_health_does_not_require_api_key(self, client: TestClient) -> None:
        # Even without X-API-Key, /health must respond — load balancers
        # poll without secrets.
        r = client.get("/health")
        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Auth gates on protected routes
# ---------------------------------------------------------------------------


class TestAuth:
    def test_verify_without_api_key_returns_401(self, client: TestClient) -> None:
        r = client.post("/verify", json={"text": "x"})
        assert r.status_code == 401

    def test_verify_with_wrong_api_key_returns_403(self, client: TestClient) -> None:
        r = client.post(
            "/verify",
            json={"text": "x"},
            headers={"X-API-Key": "wrong"},
        )
        assert r.status_code == 403

    def test_get_job_without_api_key_returns_401(self, client: TestClient) -> None:
        r = client.get("/jobs/anything")
        assert r.status_code == 401

    def test_unset_env_returns_503(
        self,
        monkeypatch: pytest.MonkeyPatch,
        fake_worker_factory: Any,
        tmp_path: Path,
    ) -> None:
        monkeypatch.delenv("VERIFIER_API_KEY", raising=False)
        app = create_app(job_store=JobStore(), worker=fake_worker_factory(), runs_root=tmp_path)
        client = TestClient(app)
        r = client.post("/verify", json={"text": "x"}, headers={"X-API-Key": "anything"})
        assert r.status_code == 503


# ---------------------------------------------------------------------------
# POST /verify — happy path
# ---------------------------------------------------------------------------


class TestVerifySubmit:
    def test_returns_202_with_job_id(self, client: TestClient) -> None:
        r = client.post(
            "/verify",
            json={"text": "Some scientific claim. [1]"},
            headers={"X-API-Key": _API_KEY},
        )
        assert r.status_code == 202
        body = r.json()
        assert "job_id" in body
        assert body["status"] in {"pending", "running", "completed"}
        assert body["poll_url"].endswith(f"/jobs/{body['job_id']}")

    def test_invokes_worker_with_request_payload(
        self, fake_worker_factory: Any, tmp_path: Path
    ) -> None:
        worker = fake_worker_factory()
        app = create_app(job_store=JobStore(), worker=worker, runs_root=tmp_path)
        client = TestClient(app)
        client.post(
            "/verify",
            json={"text": "x", "mode": "v1"},
            headers={"X-API-Key": _API_KEY},
        )
        assert len(worker.calls) == 1
        _, req = worker.calls[0]
        assert req.text == "x"
        assert req.mode == "v1"

    def test_default_mode_is_copilot(self, fake_worker_factory: Any, tmp_path: Path) -> None:
        worker = fake_worker_factory()
        app = create_app(job_store=JobStore(), worker=worker, runs_root=tmp_path)
        client = TestClient(app)
        client.post("/verify", json={"text": "x"}, headers={"X-API-Key": _API_KEY})
        _, req = worker.calls[0]
        assert req.mode == "copilot"
        assert req.copilot_mode == "pharma"

    def test_validation_rejects_missing_text(self, client: TestClient) -> None:
        r = client.post("/verify", json={}, headers={"X-API-Key": _API_KEY})
        assert r.status_code == 422

    def test_validation_rejects_empty_text(self, client: TestClient) -> None:
        r = client.post("/verify", json={"text": ""}, headers={"X-API-Key": _API_KEY})
        assert r.status_code == 422

    def test_validation_rejects_unknown_mode(self, client: TestClient) -> None:
        r = client.post(
            "/verify",
            json={"text": "x", "mode": "unknown"},
            headers={"X-API-Key": _API_KEY},
        )
        assert r.status_code == 422

    def test_validation_rejects_extra_fields(self, client: TestClient) -> None:
        # extra='forbid' on VerifyRequest — typos must surface as 422.
        r = client.post(
            "/verify",
            json={"text": "x", "modee": "v1"},
            headers={"X-API-Key": _API_KEY},
        )
        assert r.status_code == 422


# ---------------------------------------------------------------------------
# GET /jobs/{job_id} — full lifecycle
# ---------------------------------------------------------------------------


class TestJobLifecycle:
    def test_polling_returns_completed_after_worker_runs(self, client: TestClient) -> None:
        submit = client.post(
            "/verify",
            json={"text": "x"},
            headers={"X-API-Key": _API_KEY},
        )
        job_id = submit.json()["job_id"]
        # TestClient runs background tasks synchronously — by the time
        # the POST returns, the fake worker has already completed.
        r = client.get(f"/jobs/{job_id}", headers={"X-API-Key": _API_KEY})
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "completed"
        assert body["job_id"] == job_id
        assert body["run_id"] is not None
        assert body["result"] is not None
        assert body["result"]["n_claims"] == 3

    def test_polling_unknown_job_returns_404(self, client: TestClient) -> None:
        r = client.get(
            "/jobs/00000000-0000-0000-0000-000000000000",
            headers={"X-API-Key": _API_KEY},
        )
        assert r.status_code == 404

    def test_failed_job_surfaces_error(self, fake_worker_factory: Any, tmp_path: Path) -> None:
        app = create_app(
            job_store=JobStore(),
            worker=fake_worker_factory(succeed=False, error="boom: fake failure"),
            runs_root=tmp_path,
        )
        client = TestClient(app)
        submit = client.post("/verify", json={"text": "x"}, headers={"X-API-Key": _API_KEY})
        job_id = submit.json()["job_id"]
        r = client.get(f"/jobs/{job_id}", headers={"X-API-Key": _API_KEY})
        body = r.json()
        assert body["status"] == "failed"
        assert "boom" in body["error"]


# ---------------------------------------------------------------------------
# GET /runs/{run_id}/copilot_report.html
# ---------------------------------------------------------------------------


class TestReportFile:
    def test_serves_html_when_file_exists(self, fake_worker_factory: Any, tmp_path: Path) -> None:
        # Pre-write an HTML file as if the worker had finished.
        run_dir = tmp_path / "api-abc12345"
        run_dir.mkdir(parents=True)
        (run_dir / "copilot_report.html").write_text("<html>report</html>", encoding="utf-8")
        app = create_app(job_store=JobStore(), worker=fake_worker_factory(), runs_root=tmp_path)
        client = TestClient(app)
        r = client.get(
            "/runs/api-abc12345/copilot_report.html",
            headers={"X-API-Key": _API_KEY},
        )
        assert r.status_code == 200
        assert "report" in r.text
        assert r.headers["content-type"].startswith("text/html")

    def test_404_when_run_dir_missing(self, fake_worker_factory: Any, tmp_path: Path) -> None:
        app = create_app(job_store=JobStore(), worker=fake_worker_factory(), runs_root=tmp_path)
        client = TestClient(app)
        r = client.get(
            "/runs/missing-run/copilot_report.html",
            headers={"X-API-Key": _API_KEY},
        )
        assert r.status_code == 404

    def test_path_traversal_rejected(self, fake_worker_factory: Any, tmp_path: Path) -> None:
        # Place a sensitive file outside runs_root.
        outside = tmp_path.parent / "secret.html"
        outside.write_text("SECRET", encoding="utf-8")
        try:
            app = create_app(
                job_store=JobStore(),
                worker=fake_worker_factory(),
                runs_root=tmp_path,
            )
            client = TestClient(app)
            # Path-traversal style attempt; FastAPI normalises but we
            # still defend explicitly via realpath confinement.
            r = client.get(
                "/runs/..%2F..%2Fsecret/copilot_report.html",
                headers={"X-API-Key": _API_KEY},
            )
            # Either 400 (we caught the traversal) or 404 (file does not
            # exist under runs_root). Both are acceptable; what's NOT OK
            # is 200 with the secret contents.
            assert r.status_code in {400, 404}
        finally:
            outside.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Health reflects active jobs
# ---------------------------------------------------------------------------


class TestHealthActiveJobs:
    def test_health_n_jobs_active_increases_then_settles(
        self, fake_worker_factory: Any, tmp_path: Path
    ) -> None:
        store = JobStore()
        # Manually inject a "running" job that the fake worker won't touch
        # so we can observe a non-zero active count.
        running = store.create()
        store.update(running.job_id, status="running")
        app = create_app(job_store=store, worker=fake_worker_factory(), runs_root=tmp_path)
        client = TestClient(app)
        r = client.get("/health")
        assert r.json()["n_jobs_active"] == 1
