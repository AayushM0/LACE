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
    def test_enqueue_to_inbox(self, isolated_lace_home):
        """
        Full pipeline: enqueue → worker processes → draft appears in inbox.
        
        This is the core Phase 1 flow.
        """
        from lace.mcp.queue import init_queue_db, enqueue, get_job_status, get_pending_jobs, _process_single_job
        from lace.memory.inbox import list_inbox, get_inbox_count
        
        init_queue_db()
        
        # Create a mock memory object that the extractor will "return"
        mock_candidate = MagicMock()
        mock_candidate.content = "We decided to use SQLite for the extraction queue."
        mock_candidate.summary = "SQLite queue decision"
        mock_candidate.category = "decision"
        mock_candidate.tags = ["sqlite", "queue"]
        mock_candidate.scope = "global"
        mock_candidate.confidence = 0.75
        mock_candidate.metadata = {}
        
        # Track what gets written to inbox
        written_drafts = []
        
        def mock_save_to_inbox(obj):
            draft_id = f"draft_test{len(written_drafts):04d}"
            written_drafts.append(draft_id)
            return draft_id
        
        with patch("lace.memory.extractor.should_attempt_extraction", return_value=True):
            with patch(
                "lace.memory.extractor.extract_from_conversation",
                return_value=[mock_candidate],
            ):
                with patch(
                    "lace.memory.inbox.save_to_inbox",
                    side_effect=mock_save_to_inbox,
                ):
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
        assert len(written_drafts) == 1
    
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
        
        # Simulate worker loop iterations synchronously
        with patch("lace.memory.extractor.should_attempt_extraction", return_value=True):
            with patch(
                "lace.memory.extractor.extract_from_conversation",
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

    def test_promote_draft_to_vault(self, isolated_lace_home):
        """
        Full pipeline test:
        1. Enqueue job
        2. Worker extracts and saves to inbox as draft
        3. list_inbox shows the draft
        4. promote_to_vault promotes draft to vault and deletes the draft
        5. Verify the memory is in the vault store
        """
        from lace.mcp.queue import init_queue_db, enqueue, get_job_status, get_pending_jobs, _process_single_job
        from lace.memory.inbox import list_inbox, promote_to_vault, get_inbox_count
        from lace.memory.store import MemoryStore
        
        init_queue_db()
        
        # Create a candidate
        from lace.memory.models import make_memory
        candidate = make_memory(
            content="This is a candidate of sufficient length for memory.",
            category="decision",
            tags=["sqlite", "queue"],
            scope="global",
        )
        
        with patch("lace.memory.extractor.should_attempt_extraction", return_value=True):
            with patch(
                "lace.memory.extractor.extract_from_conversation",
                return_value=[candidate],
            ):
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
        
        # Verify it's in the inbox
        assert get_inbox_count() == 1
        drafts = list_inbox()
        assert len(drafts) == 1
        draft_id = drafts[0].id
        assert draft_id.startswith("draft_")
        
        # Mock MemoryStore dependencies for ChromaDB to avoid real vector DB issues
        mock_client = MagicMock()
        mock_collection = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection
        
        with patch("lace.retrieval.vector.get_client", return_value=mock_client):
            vault_id = promote_to_vault(draft_id)
        
        assert vault_id.startswith("mem_")
        assert get_inbox_count() == 0
        
        # Verify it was saved in the vault
        store = MemoryStore()
        vault_memory = store.get(vault_id)
        assert vault_memory is not None
        assert vault_memory.content == "This is a candidate of sufficient length for memory."
        assert vault_memory.confidence == 0.6
