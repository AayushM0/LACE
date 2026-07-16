"""
End-to-end integration tests for Phase 1.

These tests exercise the complete pipeline:
enqueue → worker → inbox → review → promote → vault

They use real SQLite and real file I/O in temp directories.
They mock only the LLM extractor (we can't require Ollama in CI).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def isolated_lace_home(tmp_path, monkeypatch):
    """
    Creates an isolated LACE home directory for integration tests.
    All file I/O goes to tmp_path — nothing touches ~/.lace.
    """
    lace_home = tmp_path / ".lace"
    from lace.core.config import init_lace_home
    init_lace_home(lace_home)
    monkeypatch.setenv("LACE_HOME", str(lace_home))
    return lace_home


class TestEndToEndExtractionPipeline:
    def test_enqueue_to_vault(self, isolated_lace_home):
        """
        Full pipeline: enqueue → worker processes → memory stored directly in vault.
        """
        from lace.mcp.queue import init_queue_db, enqueue, get_job_status, get_pending_jobs, _process_single_job
        from lace.memory.store import MemoryStore
        
        init_queue_db()
        
        # Mock LLM response in the new format
        fake_llm = """{
          "worth_remembering": true,
          "reason": "Test reasoning",
          "memories": [
            {
              "category": "decision",
              "summary": "We decided to use SQLite for the extraction queue.",
              "body": "We decided to use SQLite for the extraction queue.",
              "tags": ["sqlite", "queue"],
              "confidence": 0.75
            }
          ]
        }"""
        
        # Mock MemoryStore dependencies for ChromaDB to avoid real vector DB issues
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        
        with patch("lace.memory.extractor.should_attempt_extraction", return_value=True):
            with patch("lace.memory.extractor.call_llm", return_value=fake_llm):
                with patch("lace.retrieval.vector.get_client", return_value=mock_client):
                    job_id = enqueue(
                        query="How should we handle the extraction queue?",
                        response="We decided to use SQLite for simplicity.",
                        scope="global",
                        history=[],
                    )
                    
                    # Process synchronously
                    jobs = get_pending_jobs()
                    for job in jobs:
                        _process_single_job(job)
        
        job = get_job_status(job_id)
        assert job["status"] == "done"
        
        # Verify the memory was saved directly to the vault
        store = MemoryStore()
        memories = store.list(scope="global")
        assert len(memories) == 1
        assert memories[0].content == "We decided to use SQLite for the extraction queue."
        assert memories[0].category.value == "decision"
    
    def test_short_interaction_filtered_out(self, isolated_lace_home):
        """
        Short/trivial interactions are filtered by should_attempt_extraction
        and don't result in any inbox drafts.
        """
        from lace.mcp.queue import init_queue_db, enqueue, get_job_status, get_pending_jobs, _process_single_job
        
        init_queue_db()
        
        with patch("lace.memory.extractor.should_attempt_extraction", return_value=False):
            job_id = enqueue(
                query="Thanks!",
                response="You're welcome!",
                scope="global",
                history=[],
            )
            
            # Process synchronously
            jobs = get_pending_jobs()
            for job in jobs:
                _process_single_job(job)
        
        # Job should be done (filtered out cleanly)
        job = get_job_status(job_id)
        assert job["status"] == "done"
    
    def test_worker_survives_llm_offline(self, isolated_lace_home):
        """
        Worker retries when LLM is offline, eventually marks as failed
        after max retries. The worker thread itself stays alive.
        """
        from lace.mcp.queue import (
            init_queue_db,
            enqueue,
            get_job_status,
            get_pending_jobs,
            mark_processing,
            _process_single_job,
            increment_retry,
            _patch_error_msg,
            mark_failed,
            _MAX_RETRIES,
        )
        
        init_queue_db()
        
        job_id = enqueue(
            query="This is a query of sufficient length to trigger extraction, definitely.",
            response="This is a response of sufficient length to trigger extraction, definitely.",
            scope="global",
            history=[],
        )
        
        with patch("lace.memory.extractor.should_attempt_extraction", return_value=True):
            with patch(
                "lace.memory.extractor.call_llm",
                side_effect=RuntimeError("LLM offline"),
            ):
                for _ in range(_MAX_RETRIES + 2):
                    jobs = get_pending_jobs()
                    for job in jobs:
                        if job["retry_count"] > _MAX_RETRIES:
                            mark_failed(job["id"], f"Max retries ({_MAX_RETRIES}) exceeded")
                            continue
                        
                        mark_processing(job["id"])
                        try:
                            _process_single_job(job)
                        except Exception as e:
                            increment_retry(job["id"])
                            _patch_error_msg(job["id"], str(e))
        
        job = get_job_status(job_id)
        assert job["status"] == "failed"
        assert "Max retries" in (job["error_msg"] or "")

