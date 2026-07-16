"""
Tests for the CLI extract wiring (main.py run_extract).

Strategy:
  - Test run_extract() with injected dependencies (no Typer, no console)
  - One vertical slice per test

Run with: pytest tests/test_memory/test_extract_cli.py -v
"""

import pytest
from unittest.mock import MagicMock

from lace.core.config import LaceConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SAMPLE_MEMORIES = [
    {
        "category": "debug",
        "summary": "SQLite WAL mode resolves concurrent write errors.",
        "body": "Use PRAGMA journal_mode=WAL.",
        "tags": ["sqlite", "wal"],
        "confidence": 0.85,
    },
    {
        "category": "pattern",
        "summary": "Use lazy imports to avoid circular dependencies.",
        "body": "Import inside function body.",
        "tags": ["python", "imports"],
        "confidence": 0.7,
    },
]

FAKE_PATHS = {
    "vector_db": "/tmp/vector_db",
    "hash_index": "/tmp/hash_index",
    "pipeline_log": "/tmp/pipeline_log",
}


def _run(
    memories=None,
    should_extract=True,
    dry_run=False,
    confirm=False,
    scope="project:LACE",
    dedup_return="mem_new_001",
):
    """Run run_extract with all dependencies mocked."""
    from lace.main import run_extract

    mock_extract = MagicMock(return_value=SAMPLE_MEMORIES if memories is None else memories)
    mock_dedup = MagicMock(return_value=dedup_return)
    mock_should = MagicMock(return_value=should_extract)
    mock_vi = MagicMock()
    mock_store = MagicMock()

    result = run_extract(
        query="How do I fix concurrent writes?",
        response="Set PRAGMA journal_mode=WAL.",
        config=LaceConfig(),
        paths=FAKE_PATHS,
        store=mock_store,
        active_scope=scope,
        dry_run=dry_run,
        confirm=confirm,
        extract_fn=mock_extract,
        dedup_fn=mock_dedup,
        vector_index_cls=mock_vi,
        should_extract_fn=mock_should,
    )

    return result, mock_dedup, mock_extract, mock_should


# ---------------------------------------------------------------------------
# Slice 1: Dry-run shows candidates without storing
# ---------------------------------------------------------------------------


class TestDryRun:

    def test_dry_run_does_not_call_dedup(self):
        """Dry-run returns memories but never stores them."""
        result, mock_dedup, _, _ = _run(dry_run=True)
        mock_dedup.assert_not_called()
        assert len(result["memories"]) == len(SAMPLE_MEMORIES)
        assert result["stored"] == 0

    def test_confirm_behaves_like_dry_run(self):
        """--confirm is equivalent to --dry-run."""
        result, mock_dedup, _, _ = _run(confirm=True)
        mock_dedup.assert_not_called()
        assert len(result["memories"]) == len(SAMPLE_MEMORIES)


# ---------------------------------------------------------------------------
# Slice 2: Store path calls dedup for each memory
# ---------------------------------------------------------------------------


class TestStorePath:

    def test_store_calls_dedup_for_each_memory(self):
        """Each extracted memory goes through dedup_and_store."""
        result, mock_dedup, _, _ = _run(dry_run=False)
        assert mock_dedup.call_count == len(SAMPLE_MEMORIES)
        assert result["stored"] == len(SAMPLE_MEMORIES)

    def test_store_sets_project_scope(self):
        """dedup_and_store receives the memory with project_scope set."""
        _, mock_dedup, _, _ = _run(dry_run=False, scope="global")
        for args, kwargs in mock_dedup.call_args_list:
            candidate = kwargs.get("candidate") or args[0]
            assert candidate["project_scope"] == "global"

    def test_store_passes_vector_index_and_store(self):
        """dedup_and_store receives the vector index and memory store."""
        _, mock_dedup, _, _ = _run(dry_run=False)
        for args, kwargs in mock_dedup.call_args_list:
            assert "vector_index" in kwargs
            assert "memory_store" in kwargs

    def test_dedup_returns_none_counts_as_skipped(self):
        """When dedup_and_store returns None (duplicate), it counts as skipped."""
        result, _, _, _ = _run(dry_run=False, dedup_return=None)
        assert result["skipped"] == len(SAMPLE_MEMORIES)
        assert result["stored"] == 0


# ---------------------------------------------------------------------------
# Slice 3: Empty extraction
# ---------------------------------------------------------------------------


class TestEmptyExtraction:

    def test_empty_memories_skips_store(self):
        """No extracted memories → no dedup calls."""
        result, mock_dedup, _, _ = _run(memories=[], dry_run=False)
        mock_dedup.assert_not_called()
        assert result["stored"] == 0
        assert result["memories"] == []

    def test_pre_filter_blocks_extraction(self):
        """should_attempt_extraction=False → early return."""
        result, mock_dedup, mock_extract, _ = _run(should_extract=False)
        mock_dedup.assert_not_called()
        mock_extract.assert_not_called()
        assert result["blocked"] is True


# ---------------------------------------------------------------------------
# Slice 4: Scope override
# ---------------------------------------------------------------------------


class TestScopeOverride:

    def test_scope_passed_through_to_dedup(self):
        """active_scope flows into candidate project_scope."""
        _, mock_dedup, _, _ = _run(scope="project:other", dry_run=False)
        for args, kwargs in mock_dedup.call_args_list:
            candidate = kwargs.get("candidate") or args[0]
            assert candidate["project_scope"] == "project:other"
