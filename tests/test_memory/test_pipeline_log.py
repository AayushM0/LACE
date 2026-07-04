# tests/test_memory/test_pipeline_log.py
"""
Tests for memory/pipeline_log.py — chunk 5.

Strategy:
  - Test schema initialization (idempotent, all tables/indexes)
  - Test each writer function independently
  - Test _write_event never raises under failure conditions
  - Test each query function returns correct structure and values
  - Integration test: simulate full stress test run, verify funnel
  - Integration test: simulate real interaction, verify full trace
  - Cross-module wiring test: verify queue/extractor/dedup import correctly

Run with: pytest tests/test_memory/test_pipeline_log.py -v
"""

import sqlite3
import pytest
from pathlib import Path
from unittest.mock import patch

from lace.memory.pipeline_log import (
    initialize_pipeline_log_db,
    log_queue_suppressed,
    log_extraction_verdict,
    log_dedup_action,
    query_funnel_summary,
    query_suppressed_hashes,
    query_verdict_reasons,
    query_dedup_score_distribution,
    query_full_trace,
    query_recent_events,
    PIPELINE_LOG_DB_PATH,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_all_rows(db_path: Path) -> list:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM pipeline_log ORDER BY id ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def get_rows_by_event(db_path: Path, event_type: str) -> list:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM pipeline_log WHERE event_type = ? ORDER BY id ASC",
        (event_type,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Schema initialization tests
# ---------------------------------------------------------------------------

class TestInitializePipelineLogDb:

    def test_creates_pipeline_log_table(self, tmp_path):
        db_path = tmp_path / "pipeline_log.db"
        initialize_pipeline_log_db(db_path)

        conn = sqlite3.connect(str(db_path))
        tables = {
            row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert "pipeline_log" in tables

    def test_creates_all_four_indexes(self, tmp_path):
        db_path = tmp_path / "pipeline_log.db"
        initialize_pipeline_log_db(db_path)

        conn = sqlite3.connect(str(db_path))
        indexes = {
            row[1] for row in conn.execute(
                "SELECT type, name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        conn.close()

        assert "idx_pipeline_log_event" in indexes
        assert "idx_pipeline_log_hash" in indexes
        assert "idx_pipeline_log_queue_id" in indexes
        assert "idx_pipeline_log_dedup_action" in indexes

    def test_is_idempotent(self, tmp_path):
        db_path = tmp_path / "pipeline_log.db"
        initialize_pipeline_log_db(db_path)
        initialize_pipeline_log_db(db_path)  # must not raise

    def test_all_columns_present(self, tmp_path):
        db_path = tmp_path / "pipeline_log.db"
        initialize_pipeline_log_db(db_path)

        conn = sqlite3.connect(str(db_path))
        columns = {
            row[1] for row in conn.execute(
                "PRAGMA table_info(pipeline_log)"
            ).fetchall()
        }
        conn.close()

        expected = {
            "id", "event_type", "canonical_hash", "queue_id",
            "memory_id", "worth_remembering", "reason",
            "dedup_action", "similarity_score", "repeat_count", "created_at"
        }
        assert expected.issubset(columns)


# ---------------------------------------------------------------------------
# Writer function tests
# ---------------------------------------------------------------------------

class TestLogQueueSuppressed:

    def test_writes_correct_event_type(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)

        log_queue_suppressed(
            canonical_hash_value="a" * 64,
            queue_id=1,
            repeat_count=5,
            db_path=db_path,
        )

        rows = get_rows_by_event(db_path, "queue_suppressed")
        assert len(rows) == 1
        assert rows[0]["event_type"] == "queue_suppressed"

    def test_stores_canonical_hash(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)

        log_queue_suppressed("b" * 64, 2, 10, db_path=db_path)

        rows = get_rows_by_event(db_path, "queue_suppressed")
        assert rows[0]["canonical_hash"] == "b" * 64

    def test_stores_queue_id(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)

        log_queue_suppressed("c" * 64, 42, 3, db_path=db_path)

        rows = get_rows_by_event(db_path, "queue_suppressed")
        assert rows[0]["queue_id"] == 42

    def test_stores_repeat_count(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)

        log_queue_suppressed("d" * 64, 1, 17, db_path=db_path)

        rows = get_rows_by_event(db_path, "queue_suppressed")
        assert rows[0]["repeat_count"] == 17

    def test_dedup_fields_are_null(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)

        log_queue_suppressed("e" * 64, 1, 5, db_path=db_path)

        rows = get_rows_by_event(db_path, "queue_suppressed")
        assert rows[0]["worth_remembering"] is None
        assert rows[0]["dedup_action"] is None
        assert rows[0]["similarity_score"] is None


class TestLogExtractionVerdict:

    def test_writes_correct_event_type(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)

        log_extraction_verdict(
            queue_id=1,
            worth_remembering=False,
            reason="Repetitive stress test loop",
            memory_count=0,
            db_path=db_path,
        )

        rows = get_rows_by_event(db_path, "extraction_verdict")
        assert len(rows) == 1

    def test_false_verdict_stored_as_zero(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)

        log_extraction_verdict(1, False, "not useful", 0, db_path=db_path)

        rows = get_rows_by_event(db_path, "extraction_verdict")
        assert rows[0]["worth_remembering"] == 0

    def test_true_verdict_stored_as_one(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)

        log_extraction_verdict(2, True, "concrete debug solution", 1,
                               db_path=db_path)

        rows = get_rows_by_event(db_path, "extraction_verdict")
        assert rows[0]["worth_remembering"] == 1

    def test_reason_stored(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)

        log_extraction_verdict(1, False, "test reason here", 0, db_path=db_path)

        rows = get_rows_by_event(db_path, "extraction_verdict")
        assert rows[0]["reason"] == "test reason here"

    def test_memory_count_stored_in_repeat_count(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)

        log_extraction_verdict(1, True, "good", 3, db_path=db_path)

        rows = get_rows_by_event(db_path, "extraction_verdict")
        assert rows[0]["repeat_count"] == 3

    def test_canonical_hash_stored(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)

        log_extraction_verdict(
            queue_id=1,
            worth_remembering=True,
            reason="good",
            memory_count=1,
            canonical_hash_value="f" * 64,
            db_path=db_path,
        )

        rows = get_rows_by_event(db_path, "extraction_verdict")
        assert rows[0]["canonical_hash"] == "f" * 64


class TestLogDedupAction:

    def test_writes_correct_event_type(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)

        log_dedup_action("a" * 64, "store", "mem_001", db_path=db_path)

        rows = get_rows_by_event(db_path, "dedup_action")
        assert len(rows) == 1
        assert rows[0]["event_type"] == "dedup_action"

    def test_store_action(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)

        log_dedup_action("a" * 64, "store", "mem_001", db_path=db_path)

        rows = get_rows_by_event(db_path, "dedup_action")
        assert rows[0]["dedup_action"] == "store"

    def test_skip_action_with_score(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)

        log_dedup_action("b" * 64, "skip", "mem_002",
                         score=0.97, db_path=db_path)

        rows = get_rows_by_event(db_path, "dedup_action")
        assert rows[0]["dedup_action"] == "skip"
        assert abs(rows[0]["similarity_score"] - 0.97) < 1e-6

    def test_merge_hash_action_no_score(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)

        log_dedup_action("c" * 64, "merge_hash", "mem_003",
                         score=None, db_path=db_path)

        rows = get_rows_by_event(db_path, "dedup_action")
        assert rows[0]["similarity_score"] is None

    def test_merge_embedding_action(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)

        log_dedup_action("d" * 64, "merge_embedding", "mem_004",
                         score=0.88, db_path=db_path)

        rows = get_rows_by_event(db_path, "dedup_action")
        assert rows[0]["dedup_action"] == "merge_embedding"

    def test_queue_id_stored(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)

        log_dedup_action("e" * 64, "store", "mem_005",
                         queue_id=99, db_path=db_path)

        rows = get_rows_by_event(db_path, "dedup_action")
        assert rows[0]["queue_id"] == 99

    def test_unknown_action_still_writes(self, tmp_path):
        """
        Unknown action logs a warning but still writes.
        Pipeline must not crash on unexpected action strings.
        """
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)

        log_dedup_action("f" * 64, "unknown_action", db_path=db_path)

        rows = get_rows_by_event(db_path, "dedup_action")
        assert len(rows) == 1


# ---------------------------------------------------------------------------
# Failure resilience tests
# ---------------------------------------------------------------------------

class TestWriterFailureResilience:
    """
    All writers must be silent on failure — never crash the pipeline.
    """

    def test_log_queue_suppressed_bad_path_no_raise(self):
        bad_path = Path("/nonexistent/path/log.db")
        # Must not raise
        log_queue_suppressed("a" * 64, 1, 5, db_path=bad_path)

    def test_log_extraction_verdict_bad_path_no_raise(self):
        bad_path = Path("/nonexistent/path/log.db")
        log_extraction_verdict(1, False, "reason", 0, db_path=bad_path)

    def test_log_dedup_action_bad_path_no_raise(self):
        bad_path = Path("/nonexistent/path/log.db")
        log_dedup_action("a" * 64, "store", db_path=bad_path)

    def test_query_bad_path_returns_empty(self):
        bad_path = Path("/nonexistent/path/log.db")
        # All queries must return empty/default, not raise
        assert query_funnel_summary(db_path=bad_path) is not None
        assert query_suppressed_hashes(db_path=bad_path) == []
        assert query_verdict_reasons(db_path=bad_path) == []
        assert query_dedup_score_distribution(db_path=bad_path) == {}
        assert query_full_trace("a" * 64, db_path=bad_path) == []
        assert query_recent_events(db_path=bad_path) == []


# ---------------------------------------------------------------------------
# Query function tests
# ---------------------------------------------------------------------------

class TestQueryFunnelSummary:

    def test_empty_db_returns_zero_counts(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)

        result = query_funnel_summary(db_path=db_path)

        assert result["queue_suppressed"] == 0
        assert result["extraction_total"] == 0
        assert result["extraction_worthy"] == 0
        assert result["extraction_rejected"] == 0
        assert result["dedup_stored"] == 0

    def test_correct_counts_after_writes(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)

        # 3 suppressions
        for i in range(3):
            log_queue_suppressed(f"{'a' * 60}{i:04}", 1, i + 2, db_path=db_path)

        # 2 extraction verdicts: 1 worthy, 1 rejected
        log_extraction_verdict(1, True, "good", 1, db_path=db_path)
        log_extraction_verdict(2, False, "bad", 0, db_path=db_path)

        # 2 dedup actions: 1 store, 1 skip
        log_dedup_action("b" * 64, "store", "mem_1", db_path=db_path)
        log_dedup_action("c" * 64, "skip", "mem_2", score=0.97, db_path=db_path)

        result = query_funnel_summary(db_path=db_path)

        assert result["queue_suppressed"] == 3
        assert result["extraction_total"] == 2
        assert result["extraction_worthy"] == 1
        assert result["extraction_rejected"] == 1
        assert result["dedup_stored"] == 1
        assert result["dedup_skipped"] == 1

    def test_all_keys_present_in_result(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)

        result = query_funnel_summary(db_path=db_path)

        expected_keys = {
            "queue_suppressed", "extraction_total",
            "extraction_worthy", "extraction_rejected",
            "dedup_stored", "dedup_merge_hash",
            "dedup_merge_embedding", "dedup_skipped",
        }
        assert expected_keys.issubset(set(result.keys()))


class TestQuerySuppressedHashes:

    def test_returns_empty_for_empty_db(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)
        assert query_suppressed_hashes(db_path=db_path) == []

    def test_returns_sorted_by_count_desc(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)

        # Hash A: 3 suppressions
        for i in range(3):
            log_queue_suppressed("a" * 64, 1, i + 2, db_path=db_path)

        # Hash B: 1 suppression
        log_queue_suppressed("b" * 64, 2, 2, db_path=db_path)

        results = query_suppressed_hashes(db_path=db_path)
        assert results[0]["canonical_hash"] == "a" * 64
        assert results[0]["suppression_count"] == 3

    def test_limit_respected(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)

        for i in range(10):
            h = str(i) * 64
            log_queue_suppressed(h[:64], i, 2, db_path=db_path)

        results = query_suppressed_hashes(limit=5, db_path=db_path)
        assert len(results) <= 5


class TestQueryVerdictReasons:

    def test_returns_all_when_filter_is_none(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)

        log_extraction_verdict(1, True, "good reason", 1, db_path=db_path)
        log_extraction_verdict(2, False, "bad reason", 0, db_path=db_path)

        results = query_verdict_reasons(worth_remembering=None, db_path=db_path)
        assert len(results) == 2

    def test_filters_to_false_verdicts(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)

        log_extraction_verdict(1, True, "good", 1, db_path=db_path)
        log_extraction_verdict(2, False, "bad", 0, db_path=db_path)
        log_extraction_verdict(3, False, "also bad", 0, db_path=db_path)

        results = query_verdict_reasons(
            worth_remembering=False, db_path=db_path
        )
        assert len(results) == 2
        assert all(r["worth_remembering"] == 0 for r in results)

    def test_result_has_expected_keys(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)

        log_extraction_verdict(1, False, "test reason", 0, db_path=db_path)

        results = query_verdict_reasons(db_path=db_path)
        assert "queue_id" in results[0]
        assert "worth_remembering" in results[0]
        assert "reason" in results[0]
        assert "memory_count" in results[0]


class TestQueryDedupScoreDistribution:

    def test_returns_empty_for_no_scores(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)

        result = query_dedup_score_distribution(db_path=db_path)
        assert result == {}

    def test_computes_stats_per_action(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)

        # Two skips
        log_dedup_action("a" * 64, "skip", score=0.97, db_path=db_path)
        log_dedup_action("b" * 64, "skip", score=0.96, db_path=db_path)

        # One merge_embedding
        log_dedup_action("c" * 64, "merge_embedding", score=0.88,
                         db_path=db_path)

        # One store
        log_dedup_action("d" * 64, "store", score=0.60, db_path=db_path)

        result = query_dedup_score_distribution(db_path=db_path)

        assert "skip" in result
        assert result["skip"]["count"] == 2
        assert abs(result["skip"]["avg"] - 0.965) < 0.001

        assert "merge_embedding" in result
        assert result["merge_embedding"]["count"] == 1

        assert "store" in result

    def test_merge_hash_excluded_no_score(self, tmp_path):
        """merge_hash has no similarity score — must not appear in distribution."""
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)

        log_dedup_action("a" * 64, "merge_hash", score=None, db_path=db_path)

        result = query_dedup_score_distribution(db_path=db_path)
        assert "merge_hash" not in result


class TestQueryFullTrace:

    def test_returns_all_events_for_hash(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)
        h = "a" * 64

        log_queue_suppressed(h, 1, 5, db_path=db_path)
        log_extraction_verdict(1, False, "bad", 0,
                               canonical_hash_value=h, db_path=db_path)

        results = query_full_trace(h, db_path=db_path)
        assert len(results) == 2

    def test_returns_in_chronological_order(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)
        h = "b" * 64

        log_queue_suppressed(h, 1, 2, db_path=db_path)
        log_extraction_verdict(1, True, "good", 1,
                               canonical_hash_value=h, db_path=db_path)
        log_dedup_action(h, "store", "mem_001", db_path=db_path)

        results = query_full_trace(h, db_path=db_path)
        event_types = [r["event_type"] for r in results]
        assert event_types == [
            "queue_suppressed", "extraction_verdict", "dedup_action"
        ]


# ---------------------------------------------------------------------------
# Integration and Cross-Module Wiring Tests
# ---------------------------------------------------------------------------

class TestIntegrationAndWiring:

    def test_simulate_funnel_stress_run(self, tmp_path):
        """Simulate a stress test run with mixed events and verify funnel numbers."""
        db_path = tmp_path / "funnel.db"
        initialize_pipeline_log_db(db_path)

        # 10 suppressions
        for i in range(10):
            log_queue_suppressed(f"hash_sup_{i}", 100 + i, 2, db_path=db_path)

        # 5 verdicts: 3 worthy, 2 rejected
        for i in range(3):
            log_extraction_verdict(200 + i, True, "worthy reason", 1, db_path=db_path)
        for i in range(2):
            log_extraction_verdict(300 + i, False, "rejected reason", 0, db_path=db_path)

        # 3 dedup actions: 1 store, 1 merge_hash, 1 skip
        log_dedup_action("hash_1", "store", "mem_1", db_path=db_path)
        log_dedup_action("hash_2", "merge_hash", "mem_2", db_path=db_path)
        log_dedup_action("hash_3", "skip", "mem_3", score=0.98, db_path=db_path)

        funnel = query_funnel_summary(db_path=db_path)
        assert funnel["queue_suppressed"] == 10
        assert funnel["extraction_total"] == 5
        assert funnel["extraction_worthy"] == 3
        assert funnel["extraction_rejected"] == 2
        assert funnel["dedup_stored"] == 1
        assert funnel["dedup_merge_hash"] == 1
        assert funnel["dedup_skipped"] == 1

    def test_query_recent_events_filters(self, tmp_path):
        db_path = tmp_path / "recent.db"
        initialize_pipeline_log_db(db_path)

        log_queue_suppressed("hash_a", 1, 2, db_path=db_path)
        log_extraction_verdict(2, True, "good", 1, db_path=db_path)

        all_events = query_recent_events(db_path=db_path)
        assert len(all_events) == 2

        suppressed = query_recent_events(event_type="queue_suppressed", db_path=db_path)
        assert len(suppressed) == 1
        assert suppressed[0]["event_type"] == "queue_suppressed"

    def test_cross_module_imports(self):
        """Verify other modules import successfully from pipeline_log."""
        from lace.mcp.queue import enqueue_interaction
        from lace.memory.extractor import process_queue_item
        from lace.memory.dedup import dedup_and_store

        assert callable(enqueue_interaction)
        assert callable(process_queue_item)
        assert callable(dedup_and_store)
