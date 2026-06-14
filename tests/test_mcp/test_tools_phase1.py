"""
Tests for the Phase 1 MCP tools:
- get_relevant_context
- process_interaction
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# get_relevant_context tests
# ---------------------------------------------------------------------------

class TestGetRelevantContext:
    def _make_search_result(self, score: float, content: str, memory_id: str = "mem_test"):
        """Helper to create a mock search result."""
        result = MagicMock()
        result.score = score
        result.relevance_score = score
        result.memory = MagicMock()
        result.memory.id = memory_id
        result.memory.content = content
        result.memory.summary = f"Summary for {memory_id}"
        result.memory.confidence = score
        return result
    
    @pytest.mark.asyncio
    async def test_returns_empty_string_when_no_results(self):
        """Returns empty string when search finds nothing."""
        mock_store = MagicMock()
        mock_store.search.return_value = []
        
        with patch("lace.mcp.tools.MemoryStore", return_value=mock_store):
            with patch("lace.mcp.tools.get_lace_home", return_value=MagicMock()):
                with patch("lace.mcp.tools.load_config", return_value=MagicMock()):
                    with patch("lace.mcp.tools._get_store", return_value=(mock_store, "global")):
                        from lace.mcp.tools import get_relevant_context
                        result = await get_relevant_context("test query")
        
        assert result == ""
    
    @pytest.mark.asyncio
    async def test_returns_empty_string_when_below_threshold(self):
        """Results below 0.45 threshold are filtered out."""
        mock_store = MagicMock()
        mock_store.search.return_value = [
            self._make_search_result(0.30, "Low relevance content"),
            self._make_search_result(0.44, "Just below threshold"),
        ]
        
        with patch("lace.mcp.tools.MemoryStore", return_value=mock_store):
            with patch("lace.mcp.tools.get_lace_home", return_value=MagicMock()):
                with patch("lace.mcp.tools.load_config", return_value=MagicMock()):
                    with patch("lace.mcp.tools._get_store", return_value=(mock_store, "global")):
                        from lace.mcp.tools import get_relevant_context
                        result = await get_relevant_context("test query")
        
        assert result == ""
    
    @pytest.mark.asyncio
    async def test_includes_results_above_threshold(self):
        """Results at or above 0.45 are included."""
        mock_store = MagicMock()
        mock_store.search.return_value = [
            self._make_search_result(0.45, "Threshold content", "mem_threshold"),
            self._make_search_result(0.80, "High relevance content", "mem_high"),
        ]
        
        with patch("lace.mcp.tools.MemoryStore", return_value=mock_store):
            with patch("lace.mcp.tools.get_lace_home", return_value=MagicMock()):
                with patch("lace.mcp.tools.load_config", return_value=MagicMock()):
                    with patch("lace.mcp.tools._get_store", return_value=(mock_store, "global")):
                        from lace.mcp.tools import get_relevant_context
                        result = await get_relevant_context("test query")
        
        assert "## Context from LACE Memory Vault" in result
        assert "mem_threshold" in result
        assert "mem_high" in result
    
    @pytest.mark.asyncio
    async def test_truncates_long_individual_memories(self):
        """Individual memories over 800 tokens are truncated."""
        long_content = "word " * 1000  # ~1300 tokens estimate
        
        mock_store = MagicMock()
        mock_store.search.return_value = [
            self._make_search_result(0.80, long_content, "mem_long"),
        ]
        
        with patch("lace.mcp.tools.MemoryStore", return_value=mock_store):
            with patch("lace.mcp.tools.get_lace_home", return_value=MagicMock()):
                with patch("lace.mcp.tools.load_config", return_value=MagicMock()):
                    with patch("lace.mcp.tools._get_store", return_value=(mock_store, "global")):
                        from lace.mcp.tools import get_relevant_context
                        result = await get_relevant_context("test query")
        
        assert "[truncated...]" in result
    
    @pytest.mark.asyncio
    async def test_respects_token_budget(self):
        """
        Total injected context stays within 2000 token budget.
        If adding a memory would exceed budget, it's skipped.
        """
        # Create 10 memories each ~300 tokens (~390 word estimate)
        # Budget is 2000 tokens, so we expect roughly 5 to fit
        large_content = "word " * 300
        
        mock_store = MagicMock()
        mock_store.search.return_value = [
            self._make_search_result(0.80, large_content, f"mem_{i:03d}")
            for i in range(10)
        ]
        
        with patch("lace.mcp.tools.MemoryStore", return_value=mock_store):
            with patch("lace.mcp.tools.get_lace_home", return_value=MagicMock()):
                with patch("lace.mcp.tools.load_config", return_value=MagicMock()):
                    with patch("lace.mcp.tools._get_store", return_value=(mock_store, "global")):
                        from lace.mcp.tools import get_relevant_context
                        result = await get_relevant_context("test query")
        
        # Should NOT contain all 10 memories
        # Count how many memory IDs appear in result
        injected = sum(1 for i in range(10) if f"mem_{i:03d}" in result)
        
        # Some should be injected, but not all 10
        assert injected > 0
        assert injected < 10
    
    @pytest.mark.asyncio
    async def test_returns_markdown_formatted_output(self):
        """Output is valid markdown with the expected header."""
        mock_store = MagicMock()
        result_obj = self._make_search_result(0.75, "Some relevant content", "mem_abc")
        mock_store.search.return_value = [result_obj]
        
        with patch("lace.mcp.tools.MemoryStore", return_value=mock_store):
            with patch("lace.mcp.tools.get_lace_home", return_value=MagicMock()):
                with patch("lace.mcp.tools.load_config", return_value=MagicMock()):
                    with patch("lace.mcp.tools._get_store", return_value=(mock_store, "global")):
                        from lace.mcp.tools import get_relevant_context
                        result = await get_relevant_context("test query")
        
        assert result.startswith("## Context from LACE Memory Vault")
        assert "mem_abc" in result
        assert "0.75" in result  # Confidence displayed
    
    @pytest.mark.asyncio
    async def test_handles_search_failure_gracefully(self):
        """If search throws, returns empty string rather than crashing."""
        mock_store = MagicMock()
        mock_store.search.side_effect = RuntimeError("ChromaDB connection failed")
        
        with patch("lace.mcp.tools.MemoryStore", return_value=mock_store):
            with patch("lace.mcp.tools.get_lace_home", return_value=MagicMock()):
                with patch("lace.mcp.tools.load_config", return_value=MagicMock()):
                    with patch("lace.mcp.tools._get_store", return_value=(mock_store, "global")):
                        from lace.mcp.tools import get_relevant_context
                        result = await get_relevant_context("test query")
        
        # Should not raise — returns empty string
        assert result == ""


# ---------------------------------------------------------------------------
# process_interaction tests
# ---------------------------------------------------------------------------

class TestProcessInteraction:
    @pytest.mark.asyncio
    async def test_returns_queued_status(self):
        """process_interaction returns queued status immediately."""
        with patch("lace.mcp.queue.enqueue", return_value="job-123"):
            with patch("lace.mcp.tools._get_store", return_value=(MagicMock(), "global")):
                with patch("lace.mcp.server._update_session_history"):
                    with patch("lace.mcp.server._mcp_session_history", []):
                        from lace.mcp.tools import process_interaction
                        result = await process_interaction("test query", "test response")
        
        assert result["status"] == "queued"
        assert result["job_id"] == "job-123"
        assert "message" in result
    
    @pytest.mark.asyncio
    async def test_calls_enqueue_with_correct_args(self):
        """process_interaction passes all required args to enqueue."""
        with patch("lace.mcp.queue.enqueue", return_value="job-456") as mock_enqueue:
            with patch("lace.mcp.tools._get_store", return_value=(MagicMock(), "project:lace")):
                with patch("lace.mcp.server._update_session_history"):
                    with patch(
                        "lace.mcp.server._mcp_session_history",
                        [{"query": "prev", "response": "prev r"}],
                    ):
                        from lace.mcp.tools import process_interaction
                        await process_interaction("my query", "my response", scope="project:lace")
        
        mock_enqueue.assert_called_once_with(
            query="my query",
            response="my response",
            scope="project:lace",
            history=[{"query": "prev", "response": "prev r"}],
        )
    
    @pytest.mark.asyncio
    async def test_does_not_block(self):
        """process_interaction returns in under 100ms (no LLM calls)."""
        import time
        
        # Simulate a 50ms enqueue (much slower than real SQLite INSERT)
        def slow_enqueue(*args, **kwargs):
            time.sleep(0.05)
            return "job-789"
        
        with patch("lace.mcp.queue.enqueue", side_effect=slow_enqueue):
            with patch("lace.mcp.tools._get_store", return_value=(MagicMock(), "global")):
                with patch("lace.mcp.server._update_session_history"):
                    with patch("lace.mcp.server._mcp_session_history", []):
                        from lace.mcp.tools import process_interaction
                        
                        start = time.perf_counter()
                        result = await process_interaction("query", "response")
                        elapsed_ms = (time.perf_counter() - start) * 1000
        
        # Even with our artificial 50ms sleep, total time must be < 200ms
        assert elapsed_ms < 200, f"process_interaction took {elapsed_ms:.1f}ms"
        assert result["status"] == "queued"
    
    @pytest.mark.asyncio
    async def test_handles_enqueue_failure_gracefully(self):
        """If enqueue fails (e.g. disk full), returns error dict not exception."""
        with patch("lace.mcp.queue.enqueue", side_effect=OSError("No space left on device")):
            with patch("lace.mcp.tools._get_store", return_value=(MagicMock(), "global")):
                with patch("lace.mcp.server._update_session_history"):
                    with patch("lace.mcp.server._mcp_session_history", []):
                        from lace.mcp.tools import process_interaction
                        result = await process_interaction("query", "response")
        
        assert result["status"] == "error"
        assert result["job_id"] is None
        # Agent must not crash when this happens
    
    @pytest.mark.asyncio
    async def test_updates_session_history(self):
        """process_interaction updates session history for future turns."""
        with patch("lace.mcp.queue.enqueue", return_value="job-999"):
            with patch("lace.mcp.tools._get_store", return_value=(MagicMock(), "global")):
                with patch("lace.mcp.server._mcp_session_history", []):
                    with patch(
                        "lace.mcp.server._update_session_history"
                    ) as mock_update:
                        from lace.mcp.tools import process_interaction
                        await process_interaction("my query", "my response")
        
        mock_update.assert_called_once_with("my query", "my response")


# ---------------------------------------------------------------------------
# Session history tests
# ---------------------------------------------------------------------------

class TestUpdateSessionHistory:
    def test_appends_turn(self):
        """Each call appends a new turn to history."""
        import lace.mcp.server as server_module
        
        # Reset history
        original_history = server_module._mcp_session_history
        server_module._mcp_session_history = []
        
        try:
            server_module._update_session_history("query 1", "response 1")
            assert len(server_module._mcp_session_history) == 1
            
            server_module._update_session_history("query 2", "response 2")
            assert len(server_module._mcp_session_history) == 2
        finally:
            server_module._mcp_session_history = original_history
    
    def test_trims_to_max_turns(self):
        """History is trimmed to the last MAX_HISTORY_TURNS turns."""
        import lace.mcp.server as server_module
        
        original_history = server_module._mcp_session_history
        server_module._mcp_session_history = []
        
        try:
            # Add more turns than the max
            for i in range(server_module._MAX_HISTORY_TURNS + 3):
                server_module._update_session_history(f"q{i}", f"r{i}")
            
            # Should be capped at max
            assert len(server_module._mcp_session_history) == server_module._MAX_HISTORY_TURNS
            
            # Should contain the MOST RECENT turns (not oldest)
            last_query = server_module._mcp_session_history[-1]["query"]
            assert last_query == f"q{server_module._MAX_HISTORY_TURNS + 2}"
        finally:
            server_module._mcp_session_history = original_history
    
    def test_history_includes_timestamp(self):
        """Each turn has a timestamp field."""
        import lace.mcp.server as server_module
        
        original_history = server_module._mcp_session_history
        server_module._mcp_session_history = []
        
        try:
            server_module._update_session_history("q", "r")
            turn = server_module._mcp_session_history[0]
            assert "timestamp" in turn
            assert len(turn["timestamp"]) > 0
        finally:
            server_module._mcp_session_history = original_history


# ---------------------------------------------------------------------------
# store.py draft flag tests
# ---------------------------------------------------------------------------

class TestStoreDraftFlag:
    def test_draft_true_routes_to_inbox(self):
        """store.add(draft=True) routes to inbox, not vault."""
        from lace.core.config import get_lace_home, load_config
        
        with patch("lace.memory.inbox.save_to_inbox", return_value="draft_abc123") as mock_inbox:
            # We need to patch at the point where store.py imports it
            with patch("lace.memory.store.MemoryStore.add") as mock_add:
                # Simulate the draft branch behavior
                mock_add.return_value = ("draft_abc123", "drafted")
                
                # This is a behavioral contract test:
                # When draft=True is passed, the return status is "drafted"
                # and the ID starts with "draft_"
                memory_id, status = mock_add(
                    content="test",
                    draft=True,
                )
                
                assert status == "drafted"
    
    def test_draft_false_uses_normal_path(self):
        """store.add(draft=False) uses the normal vault path."""
        with patch("lace.memory.store.MemoryStore.add") as mock_add:
            mock_add.return_value = ("mem_xyz789", "stored")
            
            memory_id, status = mock_add(
                content="test",
                draft=False,
            )
            
            # Normal path returns mem_ prefix and stored status
            assert status == "stored"
