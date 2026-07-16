"""
Tests for memory/dedup.py chunk 4 changes.

Strategy:
  - Test vault hash index CRUD in isolation
  - Test merge_into() contract exhaustively (pure-ish function)
  - Test dedup_and_store() with mocked vector_index and memory_store
  - Test scope filtering
  - Test protection logic for high-confidence memories
  - Integration test: full two-tier flow

Run with: pytest tests/test_memory/test_dedup.py -v
"""

import pytest
import sqlite3
import json
from pathlib import Path
from unittest.mock import MagicMock, patch, call
from datetime import datetime, timezone

from lace.memory.dedup import (
    initialize_vault_hash_index,
    lookup_by_hash,
    insert_hash_index_entry,
    update_hash_index_merge_time,
    log_dedup_event,
    merge_into,
    cosine_similarity,
    get_scope_filter,
    dedup_and_store,
    _store_new,
)
from lace.memory.extractor import initialize_pipeline_log_db
from lace.core.config import LaceConfig, DedupConfig
from lace.memory.models import MemoryObject


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(
    skip: float = 0.95,
    merge: float = 0.85,
    cooldown: int = 300,
) -> LaceConfig:
    return LaceConfig(
        dedup=DedupConfig(
            skip_threshold=skip,
            merge_threshold=merge,
            hash_cooldown_seconds=cooldown,
        )
    )


def make_memory(
    memory_id: str = "mem_test001",
    summary: str = "SQLite WAL mode resolves concurrent write errors.",
    tags: list = None,
    confidence: float = 0.7,
    lifecycle: str = "captured",
    scope: str = "project:LACE",
    body: str = "Use PRAGMA journal_mode=WAL.",
    access_count: int = 0,
) -> MemoryObject:
    # Map 'body' parameter to the MemoryObject's content field
    from lace.memory.models import MemoryLifecycle
    return MemoryObject(
        id=memory_id,
        content=body,
        summary=summary,
        tags=tags or ["sqlite", "wal"],
        confidence=confidence,
        lifecycle=MemoryLifecycle(lifecycle),
        project_scope=scope,
        category="debug",
        source="auto_extracted",
        access_count=access_count,
        last_accessed=datetime.now(timezone.utc),
        created_at=datetime.now(timezone.utc),
    )


def make_candidate(
    summary: str = "SQLite WAL mode resolves concurrent write errors.",
    body: str = "Enable WAL with PRAGMA journal_mode=WAL before connections.",
    tags: list = None,
    category: str = "debug",
    scope: str = "project:LACE",
    confidence: float = 0.7,
) -> dict:
    return {
        "summary": summary,
        "body": body,
        "tags": tags or ["sqlite", "wal", "concurrency"],
        "category": category,
        "project_scope": scope,
        "confidence": confidence,
    }


def make_vector_result(memory: MemoryObject, distance: float = 0.2):
    result = MagicMock()
    result.memory = memory
    result.distance = distance
    return result


def read_log_rows(db_path: Path, action: str = None) -> list:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    if action:
        rows = conn.execute(
            "SELECT * FROM pipeline_log WHERE dedup_action = ?", (action,)
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT * FROM pipeline_log WHERE event_type = 'dedup_action'"
        ).fetchall()
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# Vault hash index tests
# ---------------------------------------------------------------------------

class TestVaultHashIndex:

    def test_initialize_creates_table(self, tmp_path):
        db_path = tmp_path / "hash_index.db"
        initialize_vault_hash_index(db_path)

        conn = sqlite3.connect(str(db_path))
        tables = {
            row[0] for row in
            conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        conn.close()
        assert "vault_hash_index" in tables

    def test_initialize_creates_indexes(self, tmp_path):
        db_path = tmp_path / "hash_index.db"
        initialize_vault_hash_index(db_path)

        conn = sqlite3.connect(str(db_path))
        indexes = {
            row[1] for row in
            conn.execute(
                "SELECT type, name FROM sqlite_master WHERE type='index'"
            ).fetchall()
        }
        conn.close()
        assert "idx_vault_hash" in indexes
        assert "idx_vault_scope" in indexes

    def test_initialize_is_idempotent(self, tmp_path):
        db_path = tmp_path / "hash_index.db"
        initialize_vault_hash_index(db_path)
        initialize_vault_hash_index(db_path)  # must not raise

    def test_lookup_returns_none_when_empty(self, tmp_path):
        db_path = tmp_path / "hash_index.db"
        initialize_vault_hash_index(db_path)
        result = lookup_by_hash("a" * 64, db_path=db_path)
        assert result is None

    def test_insert_and_lookup(self, tmp_path):
        db_path = tmp_path / "hash_index.db"
        initialize_vault_hash_index(db_path)

        insert_hash_index_entry(
            hash_value="a" * 64,
            memory_id="mem_001",
            summary_preview="Test summary",
            scope="project:LACE",
            db_path=db_path,
        )

        result = lookup_by_hash("a" * 64, db_path=db_path)
        assert result == "mem_001"

    def test_lookup_with_correct_scope(self, tmp_path):
        db_path = tmp_path / "hash_index.db"
        initialize_vault_hash_index(db_path)

        insert_hash_index_entry(
            hash_value="b" * 64,
            memory_id="mem_002",
            summary_preview="Project-scoped memory",
            scope="project:LACE",
            db_path=db_path,
        )

        # Same scope → found
        result = lookup_by_hash(
            "b" * 64, scope="project:LACE", db_path=db_path
        )
        assert result == "mem_002"

    def test_lookup_with_wrong_scope_returns_none(self, tmp_path):
        db_path = tmp_path / "hash_index.db"
        initialize_vault_hash_index(db_path)

        insert_hash_index_entry(
            hash_value="c" * 64,
            memory_id="mem_003",
            summary_preview="Project-scoped memory",
            scope="project:LACE",
            db_path=db_path,
        )

        # Different project → not found
        result = lookup_by_hash(
            "c" * 64, scope="project:OTHER", db_path=db_path
        )
        assert result is None

    def test_global_scope_found_from_any_project(self, tmp_path):
        db_path = tmp_path / "hash_index.db"
        initialize_vault_hash_index(db_path)

        insert_hash_index_entry(
            hash_value="d" * 64,
            memory_id="mem_global",
            summary_preview="Global memory",
            scope="global",
            db_path=db_path,
        )

        # Project scope query finds global entries
        result = lookup_by_hash(
            "d" * 64, scope="project:LACE", db_path=db_path
        )
        assert result == "mem_global"

    def test_insert_or_ignore_is_idempotent(self, tmp_path):
        db_path = tmp_path / "hash_index.db"
        initialize_vault_hash_index(db_path)

        insert_hash_index_entry("e" * 64, "mem_004", "summary", "global",
                                db_path=db_path)
        insert_hash_index_entry("e" * 64, "mem_004", "summary", "global",
                                db_path=db_path)  # must not raise

        conn = sqlite3.connect(str(db_path))
        count = conn.execute(
            "SELECT COUNT(*) FROM vault_hash_index WHERE canonical_hash = ?",
            ("e" * 64,)
        ).fetchone()[0]
        conn.close()
        assert count == 1

    def test_update_merge_time(self, tmp_path):
        db_path = tmp_path / "hash_index.db"
        initialize_vault_hash_index(db_path)

        insert_hash_index_entry("f" * 64, "mem_005", "summary", "global",
                                db_path=db_path)
        update_hash_index_merge_time("f" * 64, db_path=db_path)

        conn = sqlite3.connect(str(db_path))
        row = conn.execute(
            "SELECT last_merged_at FROM vault_hash_index "
            "WHERE canonical_hash = ?", ("f" * 64,)
        ).fetchone()
        conn.close()
        assert row[0] is not None


# ---------------------------------------------------------------------------
# cosine_similarity tests
# ---------------------------------------------------------------------------

class TestCosineSimilarity:

    def test_identical_vectors_return_1(self):
        v = [1.0, 2.0, 3.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors_return_0(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_zero_vector_returns_0(self):
        assert cosine_similarity([0.0, 0.0], [1.0, 2.0]) == 0.0

    def test_length_mismatch_returns_0(self):
        assert cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0

    def test_known_similarity(self):
        a = [1.0, 0.0]
        b = [1.0, 1.0]
        # cos(45°) = 0.7071...
        sim = cosine_similarity(a, b)
        assert abs(sim - 0.7071) < 0.001


# ---------------------------------------------------------------------------
# get_scope_filter tests
# ---------------------------------------------------------------------------

class TestGetScopeFilter:

    def test_project_scope_includes_self_and_global(self):
        result = get_scope_filter("project:LACE")
        assert "project:LACE" in result
        assert "global" in result
        assert len(result) == 2

    def test_global_scope_returns_only_global(self):
        result = get_scope_filter("global")
        assert result == ["global"]

    def test_none_scope_returns_empty(self):
        result = get_scope_filter(None)
        assert result == []


# ---------------------------------------------------------------------------
# merge_into tests
# ---------------------------------------------------------------------------

class TestMergeInto:

    def test_tags_are_unioned(self):
        existing = make_memory(tags=["sqlite", "wal"])
        candidate = make_candidate(tags=["sqlite", "concurrency", "performance"])

        result = merge_into(existing, candidate)

        assert "sqlite" in result.tags
        assert "wal" in result.tags
        assert "concurrency" in result.tags
        assert "performance" in result.tags

    def test_tags_are_deduplicated(self):
        existing = make_memory(tags=["sqlite", "wal"])
        candidate = make_candidate(tags=["sqlite", "wal"])

        result = merge_into(existing, candidate)

        assert result.tags.count("sqlite") == 1
        assert result.tags.count("wal") == 1

    def test_confidence_boosted_by_005(self):
        existing = make_memory(confidence=0.7)
        candidate = make_candidate()

        result = merge_into(existing, candidate)

        assert abs(result.confidence - 0.75) < 1e-6

    def test_confidence_capped_at_1(self):
        existing = make_memory(confidence=0.98)
        candidate = make_candidate()

        result = merge_into(existing, candidate)

        assert result.confidence <= 1.0

    def test_body_appended_when_different(self):
        existing = make_memory(body="Original body content here.")
        candidate = make_candidate(body="Additional context about WAL mode.")

        result = merge_into(existing, candidate)

        assert "Original body content here." in result.content
        assert "Additional context about WAL mode." in result.content

    def test_body_not_duplicated_when_same(self):
        body = "Use PRAGMA journal_mode=WAL."
        existing = make_memory(body=body)
        candidate = make_candidate(body=body)

        result = merge_into(existing, candidate)

        # Body should appear only once
        assert result.content.count(body) == 1

    def test_access_count_incremented(self):
        existing = make_memory(access_count=5)
        candidate = make_candidate()

        result = merge_into(existing, candidate)

        assert result.access_count == 6

    def test_last_accessed_updated(self):
        existing = make_memory()
        old_time = existing.last_accessed
        candidate = make_candidate()

        result = merge_into(existing, candidate)

        assert result.last_accessed != old_time

    def test_lifecycle_not_downgraded(self):
        existing = make_memory(lifecycle="validated")
        candidate = make_candidate()

        result = merge_into(existing, candidate)

        from lace.memory.models import MemoryLifecycle
        assert result.lifecycle == MemoryLifecycle.VALIDATED

    def test_summary_not_changed(self):
        original_summary = "SQLite WAL mode resolves concurrent write errors."
        existing = make_memory(summary=original_summary)
        candidate = make_candidate(summary="Different summary text here.")

        result = merge_into(existing, candidate)

        assert result.summary == original_summary


# ---------------------------------------------------------------------------
# log_dedup_event tests
# ---------------------------------------------------------------------------

class TestLogDedupEvent:

    def test_store_action_logged(self, tmp_path):
        db_path = tmp_path / "pipeline_log.db"
        initialize_pipeline_log_db(db_path)

        log_dedup_event(
            canonical_hash_value="a" * 64,
            action="store",
            target_id="mem_001",
            score=None,
            queue_id=1,
            db_path=db_path,
        )

        rows = read_log_rows(db_path, action="store")
        assert len(rows) == 1
        assert rows[0]["dedup_action"] == "store"
        assert rows[0]["memory_id"] == "mem_001"

    def test_skip_action_logged_with_score(self, tmp_path):
        db_path = tmp_path / "pipeline_log.db"
        initialize_pipeline_log_db(db_path)

        log_dedup_event(
            canonical_hash_value="b" * 64,
            action="skip",
            target_id="mem_002",
            score=0.97,
            db_path=db_path,
        )

        rows = read_log_rows(db_path, action="skip")
        assert len(rows) == 1
        assert abs(rows[0]["similarity_score"] - 0.97) < 1e-6

    def test_merge_hash_action_logged(self, tmp_path):
        db_path = tmp_path / "pipeline_log.db"
        initialize_pipeline_log_db(db_path)

        log_dedup_event(
            canonical_hash_value="c" * 64,
            action="merge_hash",
            target_id="mem_003",
            score=None,
            db_path=db_path,
        )

        rows = read_log_rows(db_path, action="merge_hash")
        assert len(rows) == 1

    def test_logging_failure_does_not_raise(self):
        bad_path = Path("/nonexistent/path/pipeline_log.db")
        # Must not raise
        log_dedup_event(
            canonical_hash_value="a" * 64,
            action="store",
            db_path=bad_path,
        )


# ---------------------------------------------------------------------------
# dedup_and_store tests (mocked dependencies)
# ---------------------------------------------------------------------------

class TestDedupAndStore:

    def _make_mocks(self):
        vector_index = MagicMock()
        memory_store = MagicMock()
        return vector_index, memory_store

    def test_tier1_hash_hit_triggers_merge(self, tmp_path):
        """
        When the hash index has a match, Tier 1 fires:
        merge_into called, vector_index.query NOT called.
        """
        hash_db = tmp_path / "hash.db"
        log_db = tmp_path / "log.db"
        initialize_vault_hash_index(hash_db)
        initialize_pipeline_log_db(log_db)

        existing = make_memory(memory_id="mem_existing")
        vector_index, memory_store = self._make_mocks()
        memory_store.get.return_value = existing

        # Pre-populate hash index with the candidate's hash
        from lace.memory.normalize import canonical_hash
        candidate = make_candidate()
        h = canonical_hash(candidate["summary"])
        insert_hash_index_entry(h, "mem_existing", "preview", "project:LACE",
                                db_path=hash_db)

        config = make_config()

        with patch("lace.memory.dedup.embed_text") as mock_embed:
            result = dedup_and_store(
                candidate=candidate,
                vector_index=vector_index,
                memory_store=memory_store,
                config=config,
                queue_id=1,
                hash_index_db_path=hash_db,
                log_db_path=log_db,
            )

        # Vector index must NOT have been queried (Tier 1 short-circuited)
        vector_index.query.assert_not_called()
        # Embedding must NOT have been computed
        mock_embed.assert_not_called()
        # merge_into result stored
        memory_store.save.assert_called_once()
        assert result == "mem_existing"

        # Log row written with merge_hash action
        rows = read_log_rows(log_db, action="merge_hash")
        assert len(rows) == 1

    def test_tier2_skip_on_high_similarity(self, tmp_path):
        hash_db = tmp_path / "hash.db"
        log_db = tmp_path / "log.db"
        initialize_vault_hash_index(hash_db)
        initialize_pipeline_log_db(log_db)

        existing = make_memory(memory_id="mem_similar")
        vector_index, memory_store = self._make_mocks()

        # distance=0.04 → similarity = 1 - 0.04/2 = 0.98 → above skip_threshold
        vector_index.query.return_value = [make_vector_result(existing, distance=0.04)]

        config = make_config(skip=0.95, merge=0.85)
        candidate = make_candidate(summary="A completely new summary text.")

        with patch("lace.memory.dedup.embed_text", return_value=[0.1] * 384):
            result = dedup_and_store(
                candidate=candidate,
                vector_index=vector_index,
                memory_store=memory_store,
                config=config,
                hash_index_db_path=hash_db,
                log_db_path=log_db,
            )

        # Store not called (skipped)
        memory_store.add.assert_not_called()
        memory_store.save.assert_not_called()
        assert result == "mem_similar"

        rows = read_log_rows(log_db, action="skip")
        assert len(rows) == 1
        assert rows[0]["similarity_score"] >= 0.95

    def test_tier2_merge_on_medium_similarity(self, tmp_path):
        hash_db = tmp_path / "hash.db"
        log_db = tmp_path / "log.db"
        initialize_vault_hash_index(hash_db)
        initialize_pipeline_log_db(log_db)

        existing = make_memory(memory_id="mem_close", confidence=0.7)
        vector_index, memory_store = self._make_mocks()
        memory_store.get.return_value = existing

        # distance=0.20 → similarity = 1 - 0.10 = 0.90 → between merge and skip
        vector_index.query.return_value = [
            make_vector_result(existing, distance=0.20)
        ]

        config = make_config(skip=0.95, merge=0.85)
        candidate = make_candidate(summary="A completely different summary here.")

        with patch("lace.memory.dedup.embed_text", return_value=[0.1] * 384):
            result = dedup_and_store(
                candidate=candidate,
                vector_index=vector_index,
                memory_store=memory_store,
                config=config,
                hash_index_db_path=hash_db,
                log_db_path=log_db,
            )

        memory_store.save.assert_called_once()
        memory_store.add.assert_not_called()
        assert result == "mem_close"

        rows = read_log_rows(log_db, action="merge_embedding")
        assert len(rows) == 1

    def test_tier2_store_on_low_similarity(self, tmp_path):
        hash_db = tmp_path / "hash.db"
        log_db = tmp_path / "log.db"
        initialize_vault_hash_index(hash_db)
        initialize_pipeline_log_db(log_db)

        existing = make_memory(memory_id="mem_distant")
        new_memory = make_memory(memory_id="mem_new_001")
        vector_index, memory_store = self._make_mocks()
        memory_store.add.return_value = new_memory

        # distance=0.80 → similarity = 1 - 0.40 = 0.60 → below merge threshold
        vector_index.query.return_value = [
            make_vector_result(existing, distance=0.80)
        ]

        config = make_config(skip=0.95, merge=0.85)
        candidate = make_candidate(summary="Redis caching reduces database load.")

        with patch("lace.memory.dedup.embed_text", return_value=[0.1] * 384):
            result = dedup_and_store(
                candidate=candidate,
                vector_index=vector_index,
                memory_store=memory_store,
                config=config,
                hash_index_db_path=hash_db,
                log_db_path=log_db,
            )

        memory_store.add.assert_called_once()
        assert result == "mem_new_001"

        rows = read_log_rows(log_db, action="store")
        assert len(rows) == 1

    def test_empty_vault_stores_directly(self, tmp_path):
        hash_db = tmp_path / "hash.db"
        log_db = tmp_path / "log.db"
        initialize_vault_hash_index(hash_db)
        initialize_pipeline_log_db(log_db)

        new_memory = make_memory(memory_id="mem_first")
        vector_index, memory_store = self._make_mocks()
        memory_store.add.return_value = new_memory

        # Empty vault
        vector_index.query.return_value = []

        config = make_config()
        candidate = make_candidate()

        with patch("lace.memory.dedup.embed_text", return_value=[0.1] * 384):
            result = dedup_and_store(
                candidate=candidate,
                vector_index=vector_index,
                memory_store=memory_store,
                config=config,
                hash_index_db_path=hash_db,
                log_db_path=log_db,
            )

        assert result == "mem_first"
        memory_store.add.assert_called_once()

    def test_high_confidence_memory_protected_from_merge(self, tmp_path):
        """
        A validated, high-confidence memory must not be merged into
        by a low-confidence candidate — store as new instead.
        """
        hash_db = tmp_path / "hash.db"
        log_db = tmp_path / "log.db"
        initialize_vault_hash_index(hash_db)
        initialize_pipeline_log_db(log_db)

        existing = make_memory(
            memory_id="mem_validated",
            confidence=0.92,
            lifecycle="validated",
        )
        new_memory = make_memory(memory_id="mem_new_low_conf")
        vector_index, memory_store = self._make_mocks()
        memory_store.add.return_value = new_memory

        # Similarity is in merge range (0.90)
        vector_index.query.return_value = [
            make_vector_result(existing, distance=0.20)
        ]

        config = make_config(skip=0.95, merge=0.85)
        # Low confidence candidate
        candidate = make_candidate(
            summary="Different summary entirely for this candidate.",
            confidence=0.4,  # below 0.6 → triggers protection
        )

        with patch("lace.memory.dedup.embed_text", return_value=[0.1] * 384):
            result = dedup_and_store(
                candidate=candidate,
                vector_index=vector_index,
                memory_store=memory_store,
                config=config,
                hash_index_db_path=hash_db,
                log_db_path=log_db,
            )

        # Should store as new, not merge into validated memory
        memory_store.save.assert_not_called()
        memory_store.add.assert_called_once()

    def test_thresholds_read_from_config(self, tmp_path):
        """
        Verify no hardcoded thresholds exist.
        Use extreme config values and verify behavior changes.
        """
        hash_db = tmp_path / "hash.db"
        log_db = tmp_path / "log.db"
        initialize_vault_hash_index(hash_db)
        initialize_pipeline_log_db(log_db)

        existing = make_memory(memory_id="mem_test")
        new_memory = make_memory(memory_id="mem_stored")
        vector_index, memory_store = self._make_mocks()
        memory_store.add.return_value = new_memory

        # similarity = 0.90
        vector_index.query.return_value = [
            make_vector_result(existing, distance=0.20)
        ]

        # With skip=0.80: 0.90 >= 0.80 → SKIP
        config_low_skip = make_config(skip=0.80, merge=0.70)
        candidate = make_candidate(summary="Different text entirely for threshold test.")

        with patch("lace.memory.dedup.embed_text", return_value=[0.1] * 384):
            result = dedup_and_store(
                candidate=candidate,
                vector_index=vector_index,
                memory_store=memory_store,
                config=config_low_skip,
                hash_index_db_path=hash_db,
                log_db_path=log_db,
            )

        # With skip_threshold=0.80 and similarity=0.90 → skip
        memory_store.add.assert_not_called()
        skip_rows = read_log_rows(log_db, action="skip")
        assert len(skip_rows) == 1


# ---------------------------------------------------------------------------
# Integration test: full two-tier pipeline
# ---------------------------------------------------------------------------

class TestTwoTierIntegration:

    def test_genuine_new_memory_stored_correctly(self, tmp_path):
        """
        A genuinely novel memory goes through both tiers
        and ends up stored with a hash index entry.
        """
        hash_db = tmp_path / "hash.db"
        log_db = tmp_path / "log.db"
        initialize_vault_hash_index(hash_db)
        initialize_pipeline_log_db(log_db)

        new_memory = make_memory(memory_id="mem_stored_001")
        vector_index = MagicMock()
        memory_store = MagicMock()
        memory_store.add.return_value = new_memory
        vector_index.query.return_value = []  # empty vault

        config = make_config()
        candidate = make_candidate(
            summary="ChromaDB persistent client prevents reconnection overhead.",
            body="Use chromadb.PersistentClient() instead of ephemeral Client().",
            tags=["chromadb", "performance", "client"],
        )

        with patch("lace.memory.dedup.embed_text", return_value=[0.5] * 384):
            result = dedup_and_store(
                candidate=candidate,
                vector_index=vector_index,
                memory_store=memory_store,
                config=config,
                queue_id=10,
                hash_index_db_path=hash_db,
                log_db_path=log_db,
            )

        assert result == "mem_stored_001"

        # Verify hash index was updated
        from lace.memory.normalize import canonical_hash
        h = canonical_hash(candidate["summary"])
        stored_id = lookup_by_hash(h, db_path=hash_db)
        assert stored_id == "mem_stored_001"

        # Verify log row
        rows = read_log_rows(log_db, action="store")
        assert len(rows) == 1
        assert rows[0]["queue_id"] == 10

    def test_second_identical_candidate_hits_tier1(self, tmp_path):
        """
        Store a memory, then send an identical candidate.
        Second candidate must hit Tier 1 (hash match) without touching embeddings.
        """
        hash_db = tmp_path / "hash.db"
        log_db = tmp_path / "log.db"
        initialize_vault_hash_index(hash_db)
        initialize_pipeline_log_db(log_db)

        existing = make_memory(memory_id="mem_original")
        new_memory = make_memory(memory_id="mem_original")
        vector_index = MagicMock()
        memory_store = MagicMock()
        memory_store.add.return_value = new_memory
        memory_store.get.return_value = existing
        vector_index.query.return_value = []

        config = make_config()
        candidate = make_candidate(
            summary="SQLite WAL mode resolves concurrent write errors."
        )

        # First store
        with patch("lace.memory.dedup.embed_text", return_value=[0.1] * 384):
            dedup_and_store(
                candidate=candidate,
                vector_index=vector_index,
                memory_store=memory_store,
                config=config,
                hash_index_db_path=hash_db,
                log_db_path=log_db,
            )

        # Reset mocks for second call
        vector_index.reset_mock()
        memory_store.reset_mock()
        memory_store.get.return_value = existing

        # Second identical candidate
        with patch("lace.memory.dedup.embed_text") as mock_embed:
            dedup_and_store(
                candidate=candidate,
                vector_index=vector_index,
                memory_store=memory_store,
                config=config,
                hash_index_db_path=hash_db,
                log_db_path=log_db,
            )

        # Tier 1 must have caught it — no embedding call, no vector query
        mock_embed.assert_not_called()
        vector_index.query.assert_not_called()

        # Merge happened on the existing memory
        memory_store.save.assert_called_once()

        rows = read_log_rows(log_db, action="merge_hash")
        assert len(rows) == 1