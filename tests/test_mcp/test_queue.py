"""
Tests for the SQLite extraction queue.

These are integration tests — they actually write to a temp SQLite DB.
We don't mock SQLite because the thing being tested IS the SQLite behavior.
"""

from __future__ import annotations

import json
import sqlite3
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def temp_queue_db(tmp_path, monkeypatch):
    """
    Redirects queue DB to a temp directory for test isolation.
    Each test gets a fresh empty queue — no state leakage between tests.
    """
    queue_dir = tmp_path / "queue"
    queue_dir.mkdir()
    
    # Patch get_queue_db_path to return our temp path
    def mock_get_path():
        return queue_dir / "extraction_queue.db"
    
    monkeypatch.setattr(
        "lace.mcp.queue.get_queue_db_path",
        mock_get_path,
    )
    
    # Initialize the schema
    from lace.mcp.queue import init_queue_db
    init_queue_db()
    
    return queue_dir / "extraction_queue.db"


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

class TestInitQueueDb:
    def test_creates_table(self, temp_queue_db):
        """Table exists after init."""
        conn = sqlite3.connect(str(temp_queue_db))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='extraction_queue'"
        )
        assert cursor.fetchone() is not None
        conn.close()
    
    def test_creates_index(self, temp_queue_db):
        """Index exists after init for fast pending queries."""
        conn = sqlite3.connect(str(temp_queue_db))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_status_created'"
        )
        assert cursor.fetchone() is not None
        conn.close()
    
    def test_idempotent(self, temp_queue_db):
        """Calling init multiple times doesn't raise."""
        from lace.mcp.queue import init_queue_db
        # Should not raise even though table already exists
        init_queue_db()
        init_queue_db()


class TestGetConnectionDdl:
    """Verifies _get_connection() no longer runs schema DDL (Finding 10 fix)."""

    def test_connection_does_not_create_table(self, tmp_path, monkeypatch):
        """_get_connection() alone must NOT create the queue table."""
        fresh_db = tmp_path / "fresh" / "queue.db"
        fresh_db.parent.mkdir(parents=True)

        # Point get_queue_db_path to our fresh path so init_queue_db() is NOT called
        monkeypatch.setattr(
            "lace.mcp.queue.get_queue_db_path",
            lambda: fresh_db,
        )

        from lace.mcp.queue import _get_connection
        conn = _get_connection(fresh_db)
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='extraction_queue'"
            )
            assert cursor.fetchone() is None, (
                "_get_connection() created the table — it should only be created by init_queue_db()"
            )
        finally:
            conn.close()

    def test_init_queue_db_creates_table(self, tmp_path, monkeypatch):
        """After init_queue_db(), the table must exist."""
        fresh_db = tmp_path / "fresh2" / "queue.db"
        fresh_db.parent.mkdir(parents=True)

        monkeypatch.setattr(
            "lace.mcp.queue.get_queue_db_path",
            lambda: fresh_db,
        )

        from lace.mcp.queue import _get_connection, init_queue_db
        # Confirm table does NOT exist before init
        conn = _get_connection(fresh_db)
        try:
            cursor = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='extraction_queue'"
            )
            assert cursor.fetchone() is None, "Table should not exist before init_queue_db()"
        finally:
            conn.close()

        # Call init and verify table exists
        init_queue_db(fresh_db)
        conn2 = _get_connection(fresh_db)
        try:
            cursor = conn2.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='extraction_queue'"
            )
            assert cursor.fetchone() is not None, "Table should exist after init_queue_db()"
        finally:
            conn2.close()

    def test_get_connection_sets_row_factory(self, tmp_path, monkeypatch):
        """_get_connection() must still set row_factory for dict-style access."""
        fresh_db = tmp_path / "fresh3" / "queue.db"
        fresh_db.parent.mkdir(parents=True)

        monkeypatch.setattr(
            "lace.mcp.queue.get_queue_db_path",
            lambda: fresh_db,
        )

        from lace.mcp.queue import _get_connection, init_queue_db
        init_queue_db(fresh_db)
        conn = _get_connection(fresh_db)
        try:
            assert conn.row_factory is sqlite3.Row, (
                "row_factory must be sqlite3.Row for dict-style access"
            )
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# Enqueue tests
# ---------------------------------------------------------------------------

class TestEnqueue:
    def test_returns_job_id(self, temp_queue_db):
        """enqueue() returns a non-empty string ID."""
        from lace.mcp.queue import enqueue
        job_id = enqueue("test query", "test response", "global", [])
        assert isinstance(job_id, str)
        assert len(job_id) > 0
    
    def test_job_id_is_unique(self, temp_queue_db):
        """Each call to enqueue() returns a different ID."""
        from lace.mcp.queue import enqueue
        words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa"]
        ids = {enqueue(f"query {words[i]}", "response", "global", []) for i in range(10)}
        assert len(ids) == 10  # All unique
    
    def test_job_status_is_pending(self, temp_queue_db):
        """Newly enqueued jobs have status=pending."""
        from lace.mcp.queue import enqueue, get_job_status
        job_id = enqueue("query", "response", "global", [])
        job = get_job_status(job_id)
        assert job is not None
        assert job["status"] == "pending"
    
    def test_job_stores_all_fields(self, temp_queue_db):
        """Enqueued job contains all passed data."""
        from lace.mcp.queue import enqueue, get_job_status
        
        history = [{"query": "prev q", "response": "prev r"}]
        job_id = enqueue("my query", "my response", "project:lace", history)
        job = get_job_status(job_id)
        
        assert job["query"] == "my query"
        assert job["response"] == "my response"
        assert job["scope"] == "project:lace"
        assert json.loads(job["history_json"]) == history
        assert job["retry_count"] == 0
        assert job["error_msg"] is None
    
    def test_enqueue_speed(self, temp_queue_db):
        """enqueue() completes in under 100ms (hot path requirement)."""
        from lace.mcp.queue import enqueue
        
        start = time.perf_counter()
        for _ in range(10):
            enqueue("query", "response", "global", [])
        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # 10 enqueues should average well under 10ms each, but allow headroom on Windows/CI
        assert elapsed_ms < 500, f"10 enqueues took {elapsed_ms:.1f}ms — too slow"



# ---------------------------------------------------------------------------
# Status transition tests
# ---------------------------------------------------------------------------

class TestStatusTransitions:
    def test_mark_processing(self, temp_queue_db):
        from lace.mcp.queue import enqueue, mark_processing, get_job_status
        job_id = enqueue("q", "r", "global", [])
        mark_processing(job_id)
        assert get_job_status(job_id)["status"] == "processing"
    
    def test_mark_done(self, temp_queue_db):
        from lace.mcp.queue import enqueue, mark_done, get_job_status
        job_id = enqueue("q", "r", "global", [])
        mark_done(job_id)
        job = get_job_status(job_id)
        assert job["status"] == "done"
        assert job["processed_at"] is not None
    
    def test_mark_failed(self, temp_queue_db):
        from lace.mcp.queue import enqueue, mark_failed, get_job_status
        job_id = enqueue("q", "r", "global", [])
        mark_failed(job_id, "Test error message")
        job = get_job_status(job_id)
        assert job["status"] == "failed"
        assert "Test error message" in job["error_msg"]
    
    def test_increment_retry_resets_to_pending(self, temp_queue_db):
        """
        increment_retry() resets status to pending so the worker
        picks the job up again on the next cycle.
        """
        from lace.mcp.queue import enqueue, mark_processing, increment_retry, get_job_status
        job_id = enqueue("q", "r", "global", [])
        mark_processing(job_id)
        increment_retry(job_id)
        
        job = get_job_status(job_id)
        assert job["status"] == "pending"  # Reset for retry
        assert job["retry_count"] == 1
    
    def test_full_retry_cycle(self, temp_queue_db):
        """Simulate a job failing and retrying multiple times."""
        from lace.mcp.queue import enqueue, mark_processing, increment_retry, get_job_status
        
        job_id = enqueue("q", "r", "global", [])
        
        for expected_count in range(1, 4):
            mark_processing(job_id)
            increment_retry(job_id)
            job = get_job_status(job_id)
            assert job["retry_count"] == expected_count
            assert job["status"] == "pending"


# ---------------------------------------------------------------------------
# get_pending_jobs tests
# ---------------------------------------------------------------------------

class TestGetPendingJobs:
    def test_returns_pending_only(self, temp_queue_db):
        """Only pending jobs are returned, not done/failed/processing."""
        from lace.mcp.queue import enqueue, mark_done, mark_failed, get_pending_jobs
        
        pending_id = enqueue("pending", "r", "global", [])
        done_id = enqueue("done", "r", "global", [])
        failed_id = enqueue("failed", "r", "global", [])
        
        mark_done(done_id)
        mark_failed(failed_id, "test")
        
        jobs = get_pending_jobs()
        job_ids = [j["id"] for j in jobs]
        
        assert pending_id in job_ids
        assert done_id not in job_ids
        assert failed_id not in job_ids
    
    def test_returns_oldest_first(self, temp_queue_db):
        """Jobs are returned in FIFO order — oldest created_at first."""
        from lace.mcp.queue import enqueue, get_pending_jobs
        
        # Enqueue with small delay to ensure different timestamps
        ids = []
        words = ["alpha", "beta", "gamma"]
        for i in range(3):
            ids.append(enqueue(f"query {words[i]}", "r", "global", []))
            time.sleep(0.01)  # 10ms gap
        
        jobs = get_pending_jobs(limit=3)
        returned_ids = [j["id"] for j in jobs]
        
        # Should be in creation order
        assert returned_ids == ids
    
    def test_respects_limit(self, temp_queue_db):
        """get_pending_jobs respects the limit parameter."""
        from lace.mcp.queue import enqueue, get_pending_jobs
        
        words = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta", "theta", "iota", "kappa"]
        for i in range(10):
            enqueue(f"query {words[i]}", "r", "global", [])
        
        assert len(get_pending_jobs(limit=3)) == 3
        assert len(get_pending_jobs(limit=7)) == 7
    
    def test_empty_queue_returns_empty_list(self, temp_queue_db):
        from lace.mcp.queue import get_pending_jobs
        assert get_pending_jobs() == []


# ---------------------------------------------------------------------------
# Worker thread tests
# ---------------------------------------------------------------------------

class TestWorkerThread:
    def test_starts_as_daemon(self, temp_queue_db):
        """Worker thread is a daemon so it dies with the process."""
        from lace.mcp.queue import start_worker_thread
        thread = start_worker_thread()
        assert thread.daemon is True
        assert thread.is_alive()
    
    def test_worker_processes_job(self, temp_queue_db):
        """
        Worker picks up a pending job and marks it done.
        We mock the actual extraction to isolate queue behavior.
        """
        from lace.mcp.queue import enqueue, get_job_status
        
        # Mock the extractor so we don't need Ollama running
        with patch("lace.memory.extractor.should_attempt_extraction", return_value=False):
            with patch("lace.mcp.queue._WORKER_POLL_INTERVAL_SECONDS", 0.1):
                from lace.mcp.queue import start_worker_thread
                thread = start_worker_thread()
                
                job_id = enqueue("test", "response", "global", [])
                
                # Wait up to 2 seconds for the worker to process it
                deadline = time.time() + 2.0
                while time.time() < deadline:
                    job = get_job_status(job_id)
                    if job and job["status"] == "done":
                        break
                    time.sleep(0.05)
                
                job = get_job_status(job_id)
                assert job["status"] == "done"
    
    def test_worker_retries_on_failure(self, temp_queue_db):
        """Worker increments retry count when job processing fails."""
        from lace.mcp.queue import enqueue, get_job_status
        
        # Make extraction always fail
        with patch(
            "lace.memory.extractor.should_attempt_extraction",
            side_effect=RuntimeError("LLM offline"),
        ):
            with patch("lace.mcp.queue._WORKER_POLL_INTERVAL_SECONDS", 0.05):
                from lace.mcp.queue import start_worker_thread
                start_worker_thread()
                
                job_id = enqueue("test", "response", "global", [])
                
                # Wait for at least one retry
                deadline = time.time() + 3.0
                while time.time() < deadline:
                    job = get_job_status(job_id)
                    if job and job["retry_count"] >= 1:
                        break
                    time.sleep(0.1)
                
                job = get_job_status(job_id)
                assert job["retry_count"] >= 1
    
    def test_worker_permanently_fails_after_max_retries(self, temp_queue_db):
        """Jobs exceeding max retries are marked permanently failed."""
        from lace.mcp.queue import enqueue, get_job_status, _MAX_RETRIES
        
        # Insert directly via SQL with high retry_count to avoid race conditions with running daemon threads
        import datetime
        import uuid
        job_id = str(uuid.uuid4())
        conn = sqlite3.connect(str(temp_queue_db))
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO extraction_queue (id, query, response, scope, history_json, status, retry_count, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (job_id, "test", "response", "global", "[]", "pending", _MAX_RETRIES + 1, datetime.datetime.now(datetime.timezone.utc).isoformat())
        )
        conn.commit()
        conn.close()
        
        with patch("lace.mcp.queue._WORKER_POLL_INTERVAL_SECONDS", 0.05):
            from lace.mcp.queue import start_worker_thread
            start_worker_thread()
            
            deadline = time.time() + 2.0
            while time.time() < deadline:
                job = get_job_status(job_id)
                if job and job["status"] == "failed":
                    break
                time.sleep(0.05)
            
            job = get_job_status(job_id)
            assert job["status"] == "failed"
            assert "Max retries" in (job["error_msg"] or "")


# ---------------------------------------------------------------------------
# Context building tests
# ---------------------------------------------------------------------------

class TestBuildContextFromHistory:
    def test_empty_history(self):
        from lace.mcp.queue import _build_context_from_history
        assert _build_context_from_history([]) == ""
    
    def test_single_turn(self):
        from lace.mcp.queue import _build_context_from_history
        history = [{"query": "What is FastAPI?", "response": "It's a web framework."}]
        result = _build_context_from_history(history)
        assert "User: What is FastAPI?" in result
        assert "Assistant: It's a web framework." in result
    
    def test_long_response_truncated(self):
        from lace.mcp.queue import _build_context_from_history
        long_response = "word " * 200  # 1000 words
        history = [{"query": "q", "response": long_response}]
        result = _build_context_from_history(history)
        assert "[truncated for context]" in result
    
    def test_multiple_turns_ordered(self):
        from lace.mcp.queue import _build_context_from_history
        history = [
            {"query": "first q", "response": "first r"},
            {"query": "second q", "response": "second r"},
        ]
        result = _build_context_from_history(history)
        # First turn should appear before second turn
        first_pos = result.index("first q")
        second_pos = result.index("second q")
        assert first_pos < second_pos
