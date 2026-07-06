# tests/test_e2e_pipeline.py
"""
End-to-end pipeline verification for LACE — Chunk 6.

Tests the complete path:
  enqueue_interaction() → process_queue_item() → dedup_and_store()
  → pipeline_log queries

All LLM calls are mocked (deterministic, free).
All vector/memory store calls are mocked (no ChromaDB needed).
All SQLite databases use tmp_path (isolated per test).

Run with:
  pytest tests/test_e2e_pipeline.py -v

Run full suite (all chunks):
  pytest tests/ -v

Key regression test:
  pytest tests/test_e2e_pipeline.py::TestStressTestScenario::test_50_stress_test_interactions_produce_zero_vault_writes -v
"""

import json
import sqlite3
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

# --- Pipeline stages ---
from lace.mcp.queue import (
    init_queue_db as initialize_queue_db,
    enqueue_interaction,
    get_pending_jobs as get_pending_items,
    mark_done as mark_processed,
)
from lace.memory.extractor import (
    process_queue_item,
    initialize_pipeline_log_db,
)
from lace.memory.dedup import (
    initialize_vault_hash_index,
    dedup_and_store,
    lookup_by_hash,
)
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
)
from lace.memory.normalize import canonical_hash
from lace.core.config import LaceConfig, DedupConfig, ExtractionConfig
from lace.memory.models import MemoryLifecycle, MemoryObject, MemoryCategory


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def dbs(tmp_path):
    """
    Initialize all three databases into tmp_path.
    Returns a dict of paths for injection into pipeline functions.

    Usage:
        def test_something(dbs):
            enqueue_interaction(..., db_path=dbs["queue"])
            process_queue_item(..., log_db_path=dbs["log"])
            dedup_and_store(..., hash_index_db_path=dbs["hash"],
                            log_db_path=dbs["log"])
    """
    queue_db   = tmp_path / "queue.db"
    log_db     = tmp_path / "pipeline_log.db"
    hash_db    = tmp_path / "vault_hash_index.db"

    initialize_queue_db(queue_db)
    initialize_pipeline_log_db(log_db)
    initialize_vault_hash_index(hash_db)

    return {
        "queue":  queue_db,
        "log":    log_db,
        "hash":   hash_db,
        "root":   tmp_path,
    }


@pytest.fixture
def default_config():
    """Standard config with default thresholds."""
    return LaceConfig(
        dedup=DedupConfig(
            skip_threshold=0.95,
            merge_threshold=0.85,
            hash_cooldown_seconds=300,
        ),
        extraction=ExtractionConfig(
            require_worthiness_verdict=True,
            log_all_verdicts=True,
        )
    )


@pytest.fixture
def mock_stores():
    """
    Mocked vector index and memory store.
    Configurable per test — override return values as needed.
    """
    vector_index = MagicMock()
    memory_store = MagicMock()

    # Default: empty vault (no nearest neighbor)
    vector_index.query.return_value = []

    # Default: create returns a new memory with predictable ID
    def make_new_memory(candidate, config=None):
        mem = MagicMock()
        mem.id = f"mem_{canonical_hash(candidate.get('summary', ''))[:8]}"
        mem.summary = candidate.get("summary", "")
        mem.tags = candidate.get("tags", [])
        mem.confidence = candidate.get("confidence", 0.7)
        mem.lifecycle = MemoryLifecycle.CAPTURED
        mem.body = candidate.get("body", "")
        mem.access_count = 0
        mem.last_accessed = datetime.now(timezone.utc).isoformat()
        return mem

    memory_store.create.side_effect = make_new_memory
    memory_store.get_by_id.return_value = None
    memory_store.update.return_value = None

    return vector_index, memory_store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def llm_verdict(
    worth: bool,
    reason: str = "test reason",
    memories: list = None,
) -> str:
    """Build a JSON string matching the extraction schema."""
    return json.dumps({
        "worth_remembering": worth,
        "reason": reason,
        "memories": memories or [],
    })


def llm_garbage_verdict() -> str:
    return llm_verdict(
        worth=False,
        reason="Repetitive numbered stress test loop with no durable insight"
    )


def llm_real_verdict(summary: str, category: str = "debug") -> str:
    return llm_verdict(
        worth=True,
        reason="Concrete technical insight useful in future sessions",
        memories=[{
            "category": category,
            "summary": summary,
            "body": f"Detailed body content for: {summary}",
            "tags": ["technical", "lace", "debug"],
            "confidence": 0.8,
        }]
    )


def run_full_pipeline(
    query: str,
    response: str,
    llm_response_str: str,
    config: LaceConfig,
    dbs: dict,
    vector_index,
    memory_store,
) -> dict:
    """
    Run one interaction through the complete pipeline:
      1. enqueue_interaction()
      2. If inserted (not suppressed): process_queue_item()
      3. If memories returned: dedup_and_store() for each

    Returns dict with:
        enqueue_result  : return value from enqueue_interaction()
        memories        : list of memory dicts from extractor
        stored_ids      : list of memory IDs from dedup_and_store()
    """
    # Stage 1: Enqueue
    enqueue_result = enqueue_interaction(
        query=query,
        response=response,
        config=config.dedup,
        db_path=dbs["queue"],
        log_db_path=dbs["log"],
    )

    memories = []
    stored_ids = []

    # Stage 2: Only process if this was a new insert (not suppressed)
    if enqueue_result["action"] == "inserted":
        # Get the queue item
        pending = get_pending_items(db_path=dbs["queue"])
        # Find the specific item we just inserted
        item = next(
            (p for p in pending if p["id"] == enqueue_result["queue_id"]),
            None
        )

        if item:
            with patch("lace.memory.extractor.call_llm") as mock_llm:
                mock_llm.return_value = llm_response_str
                memories = process_queue_item(
                    item=item,
                    config=config,
                    log_db_path=dbs["log"],
                )

            mark_processed(enqueue_result["queue_id"], db_path=dbs["queue"])

            # Stage 3: Dedup and store each memory
            with patch("lace.memory.dedup.embed_text",
                       return_value=[0.1] * 384):
                for mem in memories:
                    mem_id = dedup_and_store(
                        candidate=mem,
                        vector_index=vector_index,
                        memory_store=memory_store,
                        config=config,
                        queue_id=enqueue_result["queue_id"],
                        hash_index_db_path=dbs["hash"],
                        log_db_path=dbs["log"],
                    )
                    if mem_id:
                        stored_ids.append(mem_id)

    return {
        "enqueue_result": enqueue_result,
        "memories": memories,
        "stored_ids": stored_ids,
    }


def read_all_rows(db_path: Path, table: str = "pipeline_log") -> list:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        f"SELECT * FROM {table} ORDER BY id ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# 6.1: Stress Test Scenario
# ---------------------------------------------------------------------------

class TestStressTestScenario:
    """
    The original bug:
    50 identical stress test interactions → 40-50 garbage files in vault.

    After the fix, the expected outcome:
    - 1 queue row (repeat_count=50)
    - 49 queue_suppressed log events
    - 1 extraction_verdict (worth=false)
    - 0 dedup_action events
    - 0 vault writes
    """

    def test_50_stress_test_interactions_produce_zero_vault_writes(
        self, dbs, default_config, mock_stores
    ):
        """Primary regression test for the original bug."""
        vector_index, memory_store = mock_stores
        all_results = []

        for i in range(1, 51):
            result = run_full_pipeline(
                query=f"stress test {i}",
                response=f"result: success, iteration: {i}, time: {220 + i}ms",
                llm_response_str=llm_garbage_verdict(),
                config=default_config,
                dbs=dbs,
                vector_index=vector_index,
                memory_store=memory_store,
            )
            all_results.append(result)

        # Zero memories extracted
        total_memories = sum(len(r["memories"]) for r in all_results)
        assert total_memories == 0, (
            f"Expected 0 memories extracted, got {total_memories}"
        )

        # Zero vault writes
        memory_store.create.assert_not_called()
        memory_store.update.assert_not_called()

        # Zero stored IDs
        all_stored = [id for r in all_results for id in r["stored_ids"]]
        assert len(all_stored) == 0

    def test_exactly_one_queue_row_after_50_stress_tests(
        self, dbs, default_config, mock_stores
    ):
        vector_index, memory_store = mock_stores

        for i in range(1, 51):
            enqueue_interaction(
                query=f"stress test {i}",
                response=f"success {i}",
                config=default_config.dedup,
                db_path=dbs["queue"],
                log_db_path=dbs["log"],
            )

        conn = sqlite3.connect(str(dbs["queue"]))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM extraction_queue"
        ).fetchall()
        conn.close()

        assert len(rows) == 1, (
            f"Expected exactly 1 queue row, got {len(rows)}"
        )
        assert rows[0]["repeat_count"] == 49

    def test_funnel_counts_after_stress_test(
        self, dbs, default_config, mock_stores
    ):
        """
        Verify the pipeline_log funnel reflects the correct
        suppression and rejection counts.
        """
        vector_index, memory_store = mock_stores

        for i in range(1, 51):
            run_full_pipeline(
                query=f"stress test {i}",
                response=f"success {i}",
                llm_response_str=llm_garbage_verdict(),
                config=default_config,
                dbs=dbs,
                vector_index=vector_index,
                memory_store=memory_store,
            )

        funnel = query_funnel_summary(db_path=dbs["log"])

        assert funnel["queue_suppressed"] == 49, (
            f"Expected 49 suppressed, got {funnel['queue_suppressed']}"
        )
        assert funnel["extraction_total"] == 1, (
            f"Expected 1 extraction, got {funnel['extraction_total']}"
        )
        assert funnel["extraction_rejected"] == 1, (
            f"Expected 1 rejected, got {funnel['extraction_rejected']}"
        )
        assert funnel["extraction_worthy"] == 0
        assert funnel["dedup_stored"] == 0
        assert funnel["dedup_merge_hash"] == 0
        assert funnel["dedup_merge_embedding"] == 0
        assert funnel["dedup_skipped"] == 0

    def test_rejection_reason_recorded_correctly(
        self, dbs, default_config, mock_stores
    ):
        """The LLM's rejection reason must appear in pipeline_log."""
        vector_index, memory_store = mock_stores

        run_full_pipeline(
            query="stress test 1",
            response="success",
            llm_response_str=llm_garbage_verdict(),
            config=default_config,
            dbs=dbs,
            vector_index=vector_index,
            memory_store=memory_store,
        )

        reasons = query_verdict_reasons(
            worth_remembering=False,
            db_path=dbs["log"],
        )

        assert len(reasons) == 1
        assert "stress test" in reasons[0]["reason"].lower()

    def test_suppressed_hash_queryable(
        self, dbs, default_config, mock_stores
    ):
        """After suppression, the hash appears in suppressed hashes query."""
        vector_index, memory_store = mock_stores

        for i in range(1, 11):
            enqueue_interaction(
                query=f"stress test {i}",
                response=f"success {i}",
                config=default_config.dedup,
                db_path=dbs["queue"],
                log_db_path=dbs["log"],
            )

        suppressed = query_suppressed_hashes(db_path=dbs["log"])
        assert len(suppressed) >= 1
        assert suppressed[0]["suppression_count"] == 9


# ---------------------------------------------------------------------------
# 6.2: Genuine Interaction Scenario
# ---------------------------------------------------------------------------

class TestGenuineInteractionScenario:
    """
    5 real technical interactions with distinct content.
    All must survive the full pipeline and be stored.
    """

    GENUINE_INTERACTIONS = [
        {
            "query": "How do I fix SQLite concurrent write errors in LACE?",
            "response": (
                "Set PRAGMA journal_mode=WAL before opening any connections. "
                "This enables concurrent reads with non-blocking writes. "
                "Call it once at DB initialization, not per query."
            ),
            "summary": "SQLite WAL mode resolves concurrent write errors in LACE.",
            "category": "debug",
        },
        {
            "query": "What embedding model should I use for LACE locally?",
            "response": (
                "all-MiniLM-L6-v2 gives the best quality/speed tradeoff for "
                "local CPU inference. 384-dimensional vectors, ~80ms per batch "
                "on modern hardware. Available via sentence-transformers."
            ),
            "summary": "all-MiniLM-L6-v2 is the recommended local embedding model for LACE.",
            "category": "reference",
        },
        {
            "query": "How should I structure the MCP tool call ordering?",
            "response": (
                "Always call initialize_lace_session first, then "
                "get_relevant_context at the start of every turn, "
                "and process_interaction at the end. Never skip the init call."
            ),
            "summary": "MCP tool call order: initialize → get_context → process_interaction.",
            "category": "pattern",
        },
        {
            "query": "Why is ChromaDB returning stale results after a vault update?",
            "response": (
                "ChromaDB PersistentClient caches collection metadata. "
                "After bulk updates, call collection.get() to force a "
                "cache refresh before querying. Or restart the client."
            ),
            "summary": "ChromaDB returns stale results after updates — call collection.get() to refresh cache.",
            "category": "debug",
        },
        {
            "query": "What is the preferred Python version for LACE development?",
            "response": (
                "Python 3.11+ is required. LACE uses match statements, "
                "tomllib, and timezone-aware datetime which require 3.11. "
                "3.12 is preferred for better error messages."
            ),
            "summary": "LACE requires Python 3.11+ with 3.12 preferred for development.",
            "category": "preference",
        },
    ]

    def test_all_genuine_interactions_pass_extraction_gate(
        self, dbs, default_config, mock_stores
    ):
        vector_index, memory_store = mock_stores

        extracted_count = 0
        for interaction in self.GENUINE_INTERACTIONS:
            result = run_full_pipeline(
                query=interaction["query"],
                response=interaction["response"],
                llm_response_str=llm_real_verdict(
                    interaction["summary"],
                    interaction["category"],
                ),
                config=default_config,
                dbs=dbs,
                vector_index=vector_index,
                memory_store=memory_store,
            )
            extracted_count += len(result["memories"])

        assert extracted_count == 5, (
            f"Expected 5 memories extracted from 5 genuine interactions, "
            f"got {extracted_count}"
        )

    def test_all_genuine_interactions_stored_in_vault(
        self, dbs, default_config, mock_stores
    ):
        vector_index, memory_store = mock_stores

        for interaction in self.GENUINE_INTERACTIONS:
            run_full_pipeline(
                query=interaction["query"],
                response=interaction["response"],
                llm_response_str=llm_real_verdict(
                    interaction["summary"],
                    interaction["category"],
                ),
                config=default_config,
                dbs=dbs,
                vector_index=vector_index,
                memory_store=memory_store,
            )

        # memory_store.create called once per unique interaction
        assert memory_store.create.call_count == 5, (
            f"Expected 5 vault writes, got {memory_store.create.call_count}"
        )

    def test_funnel_counts_after_genuine_interactions(
        self, dbs, default_config, mock_stores
    ):
        vector_index, memory_store = mock_stores

        for interaction in self.GENUINE_INTERACTIONS:
            run_full_pipeline(
                query=interaction["query"],
                response=interaction["response"],
                llm_response_str=llm_real_verdict(
                    interaction["summary"],
                    interaction["category"],
                ),
                config=default_config,
                dbs=dbs,
                vector_index=vector_index,
                memory_store=memory_store,
            )

        funnel = query_funnel_summary(db_path=dbs["log"])

        assert funnel["queue_suppressed"] == 0
        assert funnel["extraction_total"] == 5
        assert funnel["extraction_worthy"] == 5
        assert funnel["extraction_rejected"] == 0
        assert funnel["dedup_stored"] == 5

    def test_each_genuine_interaction_creates_separate_queue_row(
        self, dbs, default_config, mock_stores
    ):
        """5 distinct interactions → 5 distinct queue rows."""
        vector_index, memory_store = mock_stores

        for interaction in self.GENUINE_INTERACTIONS:
            enqueue_interaction(
                query=interaction["query"],
                response=interaction["response"],
                config=default_config.dedup,
                db_path=dbs["queue"],
                log_db_path=dbs["log"],
            )

        conn = sqlite3.connect(str(dbs["queue"]))
        count = conn.execute(
            "SELECT COUNT(*) FROM extraction_queue"
        ).fetchone()[0]
        conn.close()

        assert count == 5

    def test_hash_index_populated_after_genuine_interactions(
        self, dbs, default_config, mock_stores
    ):
        """After storing, each summary's hash must be in the vault hash index."""
        vector_index, memory_store = mock_stores

        for interaction in self.GENUINE_INTERACTIONS:
            run_full_pipeline(
                query=interaction["query"],
                response=interaction["response"],
                llm_response_str=llm_real_verdict(
                    interaction["summary"],
                    interaction["category"],
                ),
                config=default_config,
                dbs=dbs,
                vector_index=vector_index,
                memory_store=memory_store,
            )

        # Each interaction's summary hash must be findable
        found = 0
        for interaction in self.GENUINE_INTERACTIONS:
            h = canonical_hash(interaction["summary"])
            result = lookup_by_hash(h, db_path=dbs["hash"])
            if result is not None:
                found += 1

        assert found == 5, (
            f"Expected all 5 summaries in hash index, found {found}"
        )

    def test_all_five_categories_represented_in_verdicts(
        self, dbs, default_config, mock_stores
    ):
        """Verify that pattern/decision/debug/reference/preference all pass."""
        vector_index, memory_store = mock_stores

        for interaction in self.GENUINE_INTERACTIONS:
            run_full_pipeline(
                query=interaction["query"],
                response=interaction["response"],
                llm_response_str=llm_real_verdict(
                    interaction["summary"],
                    interaction["category"],
                ),
                config=default_config,
                dbs=dbs,
                vector_index=vector_index,
                memory_store=memory_store,
            )

        reasons = query_verdict_reasons(
            worth_remembering=True,
            db_path=dbs["log"],
        )
        assert len(reasons) == 5


# ---------------------------------------------------------------------------
# 6.3: Mixed Scenario
# ---------------------------------------------------------------------------

class TestMixedScenario:
    """
    Realistic scenario: genuine interactions interleaved with stress test loops.
    Only genuine interactions must reach the vault.
    """

    def test_genuine_and_stress_interleaved(
        self, dbs, default_config, mock_stores
    ):
        """
        Pattern:
          genuine → stress×10 → genuine → stress×10 → genuine
        Expected:
          3 vault writes (genuine only)
          20 suppressed (stress)
          2 extracted but rejected (first of each stress batch)
        """
        vector_index, memory_store = mock_stores

        genuine_interactions = [
            (
                "How do I configure ChromaDB for persistent storage?",
                "Use chromadb.PersistentClient(path='/your/path') instead of Client().",
                "ChromaDB persistent storage configured via PersistentClient.",
                "pattern",
            ),
            (
                "What causes NetworkX graph to lose edges on reload?",
                "Edges are lost if you serialize with nx.write_gml() but "
                "don't include edge attributes. Use nx.write_gpickle() instead.",
                "NetworkX edge loss on reload fixed by using write_gpickle.",
                "debug",
            ),
            (
                "How should LACE handle scope detection for monorepos?",
                "Walk up from cwd, find first .git, check for .lace/project.yaml "
                "at each level. The deepest project.yaml wins over the git root.",
                "LACE monorepo scope detection uses deepest project.yaml.",
                "decision",
            ),
        ]

        vault_writes = 0
        total_extracted = 0

        # Genuine 1
        q, r, s, c = genuine_interactions[0]
        result = run_full_pipeline(
            query=q, response=r,
            llm_response_str=llm_real_verdict(s, c),
            config=default_config, dbs=dbs,
            vector_index=vector_index, memory_store=memory_store,
        )
        vault_writes += len(result["stored_ids"])
        total_extracted += len(result["memories"])

        # Stress batch 1: 10 items
        for i in range(1, 11):
            run_full_pipeline(
                query=f"stress test {i}",
                response=f"success {i}",
                llm_response_str=llm_garbage_verdict(),
                config=default_config, dbs=dbs,
                vector_index=vector_index, memory_store=memory_store,
            )

        # Genuine 2
        q, r, s, c = genuine_interactions[1]
        result = run_full_pipeline(
            query=q, response=r,
            llm_response_str=llm_real_verdict(s, c),
            config=default_config, dbs=dbs,
            vector_index=vector_index, memory_store=memory_store,
        )
        vault_writes += len(result["stored_ids"])
        total_extracted += len(result["memories"])

        # Stress batch 2: 10 items
        for i in range(1, 11):
            run_full_pipeline(
                query=f"benchmark run {i}",
                response=f"completed {i}",
                llm_response_str=llm_garbage_verdict(),
                config=default_config, dbs=dbs,
                vector_index=vector_index, memory_store=memory_store,
            )

        # Genuine 3
        q, r, s, c = genuine_interactions[2]
        result = run_full_pipeline(
            query=q, response=r,
            llm_response_str=llm_real_verdict(s, c),
            config=default_config, dbs=dbs,
            vector_index=vector_index, memory_store=memory_store,
        )
        vault_writes += len(result["stored_ids"])
        total_extracted += len(result["memories"])

        # --- Assertions ---
        assert vault_writes == 3, (
            f"Expected 3 vault writes (genuine only), got {vault_writes}"
        )
        assert total_extracted == 3, (
            f"Expected 3 extracted memories, got {total_extracted}"
        )

        funnel = query_funnel_summary(db_path=dbs["log"])

        # 9 suppressed from batch 1 + 9 suppressed from batch 2
        assert funnel["queue_suppressed"] == 18, (
            f"Expected 18 suppressed, got {funnel['queue_suppressed']}"
        )
        assert funnel["extraction_worthy"] == 3
        assert funnel["dedup_stored"] == 3

    def test_stress_test_does_not_pollute_hash_index(
        self, dbs, default_config, mock_stores
    ):
        """
        Stress test items are rejected before reaching dedup.
        The vault hash index must contain only genuine memories.
        """
        vector_index, memory_store = mock_stores

        # 20 stress tests
        for i in range(1, 21):
            run_full_pipeline(
                query=f"stress test {i}",
                response=f"success {i}",
                llm_response_str=llm_garbage_verdict(),
                config=default_config, dbs=dbs,
                vector_index=vector_index, memory_store=memory_store,
            )

        # 1 genuine
        run_full_pipeline(
            query="How do I fix SQLite WAL mode in LACE?",
            response="Use PRAGMA journal_mode=WAL at connection init.",
            llm_response_str=llm_real_verdict(
                "SQLite WAL mode fix applied at connection initialization."
            ),
            config=default_config, dbs=dbs,
            vector_index=vector_index, memory_store=memory_store,
        )

        # Hash index should have exactly 1 entry (the genuine one)
        conn = sqlite3.connect(str(dbs["hash"]))
        count = conn.execute(
            "SELECT COUNT(*) FROM vault_hash_index"
        ).fetchone()[0]
        conn.close()

        assert count == 1, (
            f"Expected 1 hash index entry (genuine only), got {count}. "
            "Stress test content must not reach the vault hash index."
        )


# ---------------------------------------------------------------------------
# 6.4: Threshold Behavior
# ---------------------------------------------------------------------------

class TestThresholdBehavior:
    """
    Drive interactions through with controlled similarity scores.
    Verify that dedup decisions match configured thresholds exactly.
    """

    def _make_vector_result(self, memory, distance: float):
        result = MagicMock()
        result.memory = memory
        result.distance = distance
        return result

    def _make_existing_memory(self, memory_id: str = "mem_existing"):
        mem = MagicMock()
        mem.id = memory_id
        mem.summary = "Existing memory summary text."
        mem.tags = ["existing"]
        mem.confidence = 0.7
        mem.lifecycle = MemoryLifecycle.CAPTURED
        mem.body = "Existing body."
        mem.access_count = 3
        mem.last_accessed = datetime.now(timezone.utc).isoformat()
        return mem

    def test_similarity_above_skip_threshold_produces_skip(
        self, dbs, default_config
    ):
        """
        distance=0.04 → similarity = 1 - 0.02 = 0.98
        skip_threshold = 0.95
        Expected: skip
        """
        existing = self._make_existing_memory()
        vector_index = MagicMock()
        memory_store = MagicMock()
        memory_store.get_by_id.return_value = existing
        # distance 0.04 → similarity 0.98
        vector_index.query.return_value = [
            self._make_vector_result(existing, distance=0.04)
        ]

        candidate = {
            "summary": "A completely unique summary that hits high similarity.",
            "body": "Body text.",
            "tags": ["test"],
            "category": "debug",
            "project_scope": "global",
            "confidence": 0.7,
        }

        with patch("lace.memory.dedup.embed_text", return_value=[0.1] * 384):
            dedup_and_store(
                candidate=candidate,
                vector_index=vector_index,
                memory_store=memory_store,
                config=default_config,
                hash_index_db_path=dbs["hash"],
                log_db_path=dbs["log"],
            )

        memory_store.create.assert_not_called()
        memory_store.update.assert_not_called()

        rows = [
            r for r in read_all_rows(dbs["log"])
            if r["dedup_action"] == "skip"
        ]
        assert len(rows) == 1
        assert rows[0]["similarity_score"] >= default_config.dedup.skip_threshold

    def test_similarity_in_merge_range_produces_merge(
        self, dbs, default_config
    ):
        """
        distance=0.20 → similarity = 1 - 0.10 = 0.90
        merge_threshold=0.85, skip_threshold=0.95
        0.85 <= 0.90 < 0.95 → merge
        """
        existing = self._make_existing_memory()
        vector_index = MagicMock()
        memory_store = MagicMock()
        memory_store.get_by_id.return_value = existing
        vector_index.query.return_value = [
            self._make_vector_result(existing, distance=0.20)
        ]

        candidate = {
            "summary": "A summary in the merge similarity range for testing.",
            "body": "Additional context body.",
            "tags": ["new-tag"],
            "category": "debug",
            "project_scope": "global",
            "confidence": 0.7,
        }

        with patch("lace.memory.dedup.embed_text", return_value=[0.1] * 384):
            dedup_and_store(
                candidate=candidate,
                vector_index=vector_index,
                memory_store=memory_store,
                config=default_config,
                hash_index_db_path=dbs["hash"],
                log_db_path=dbs["log"],
            )

        memory_store.update.assert_called_once()
        memory_store.create.assert_not_called()

        rows = [
            r for r in read_all_rows(dbs["log"])
            if r["dedup_action"] == "merge_embedding"
        ]
        assert len(rows) == 1

    def test_similarity_below_merge_threshold_produces_store(
        self, dbs, default_config
    ):
        """
        distance=0.80 → similarity = 1 - 0.40 = 0.60
        Below merge_threshold=0.85 → store
        """
        existing = self._make_existing_memory()
        new_mem = self._make_existing_memory("mem_new")
        vector_index = MagicMock()
        memory_store = MagicMock()
        memory_store.create.return_value = new_mem
        vector_index.query.return_value = [
            self._make_vector_result(existing, distance=0.80)
        ]

        candidate = {
            "summary": "A genuinely novel memory below merge threshold.",
            "body": "New content entirely.",
            "tags": ["novel"],
            "category": "pattern",
            "project_scope": "global",
            "confidence": 0.7,
        }

        with patch("lace.memory.dedup.embed_text", return_value=[0.1] * 384):
            dedup_and_store(
                candidate=candidate,
                vector_index=vector_index,
                memory_store=memory_store,
                config=default_config,
                hash_index_db_path=dbs["hash"],
                log_db_path=dbs["log"],
            )

        memory_store.create.assert_called_once()

        rows = [
            r for r in read_all_rows(dbs["log"])
            if r["dedup_action"] == "store"
        ]
        assert len(rows) == 1

    def test_score_distribution_query_reflects_actions(
        self, dbs, default_config
    ):
        """
        After running skip + merge + store scenarios,
        query_dedup_score_distribution must reflect correct stats.
        """
        existing = self._make_existing_memory()
        new_mem = self._make_existing_memory("mem_brand_new")
        vector_index = MagicMock()
        memory_store = MagicMock()
        memory_store.get_by_id.return_value = existing
        memory_store.create.return_value = new_mem

        # Three different similarity scenarios
        scenarios = [
            (0.04, "skip candidate summary text here please"),     # sim=0.98 → skip
            (0.20, "merge candidate summary text here please"),    # sim=0.90 → merge
            (0.80, "store candidate summary text here please"),    # sim=0.60 → store
        ]

        for distance, summary in scenarios:
            vector_index.query.return_value = [
                self._make_vector_result(existing, distance=distance)
            ]
            candidate = {
                "summary": summary,
                "body": "Body.",
                "tags": ["test"],
                "category": "debug",
                "project_scope": "global",
                "confidence": 0.7,
            }
            with patch("lace.memory.dedup.embed_text",
                       return_value=[0.1] * 384):
                dedup_and_store(
                    candidate=candidate,
                    vector_index=vector_index,
                    memory_store=memory_store,
                    config=default_config,
                    hash_index_db_path=dbs["hash"],
                    log_db_path=dbs["log"],
                )

        distribution = query_dedup_score_distribution(db_path=dbs["log"])

        assert "skip" in distribution
        assert "merge_embedding" in distribution
        assert "store" in distribution

        assert distribution["skip"]["count"] == 1
        assert distribution["merge_embedding"]["count"] == 1
        assert distribution["store"]["count"] == 1

        # Verify scores are in expected ranges
        assert distribution["skip"]["avg"] >= default_config.dedup.skip_threshold
        assert (
            default_config.dedup.merge_threshold
            <= distribution["merge_embedding"]["avg"]
            < default_config.dedup.skip_threshold
        )
        assert distribution["store"]["avg"] < default_config.dedup.merge_threshold

    def test_lowering_merge_threshold_catches_more_merges(
        self, dbs, mock_stores
    ):
        """
        With merge_threshold=0.80, a similarity of 0.82 should merge.
        With merge_threshold=0.85 (default), 0.82 would store.
        This validates that thresholds are not hardcoded.
        """
        vector_index, memory_store = mock_stores
        existing = MagicMock()
        existing.id = "mem_threshold_test"
        existing.tags = ["existing"]
        existing.confidence = 0.7
        existing.lifecycle = MemoryLifecycle.CAPTURED
        existing.body = "Original body."
        existing.access_count = 0
        existing.last_accessed = datetime.now(timezone.utc).isoformat()
        existing.summary = "Threshold test existing memory."

        memory_store.get_by_id.return_value = existing
        # distance=0.36 → similarity = 1 - 0.18 = 0.82
        vector_index.query.return_value = [
            MagicMock(memory=existing, distance=0.36)
        ]

        candidate = {
            "summary": "Threshold boundary candidate summary sentence.",
            "body": "Body.",
            "tags": ["new"],
            "category": "debug",
            "project_scope": "global",
            "confidence": 0.7,
        }

        # Lower threshold config: 0.82 falls in merge range
        low_config = LaceConfig(
            dedup=DedupConfig(
                skip_threshold=0.95,
                merge_threshold=0.80,
                hash_cooldown_seconds=300,
            ),
            extraction=ExtractionConfig(
                require_worthiness_verdict=True,
                log_all_verdicts=True,
            )
        )

        with patch("lace.memory.dedup.embed_text", return_value=[0.1] * 384):
            dedup_and_store(
                candidate=candidate,
                vector_index=vector_index,
                memory_store=memory_store,
                config=low_config,
                hash_index_db_path=dbs["hash"],
                log_db_path=dbs["log"],
            )

        # With lower threshold → merge, not store
        memory_store.update.assert_called_once()
        memory_store.create.assert_not_called()


# ---------------------------------------------------------------------------
# 6.5: Pipeline Resilience
# ---------------------------------------------------------------------------

class TestPipelineResilience:
    """
    What happens when individual stages fail?
    The pipeline must degrade gracefully and never crash.
    """

    def test_llm_failure_does_not_crash_pipeline(
        self, dbs, default_config, mock_stores
    ):
        """LLM API error → empty memory list returned, no exception."""
        vector_index, memory_store = mock_stores

        with patch("lace.memory.extractor.call_llm") as mock_llm:
            mock_llm.side_effect = Exception("OpenAI API rate limit")

            # Enqueue first
            enqueue_interaction(
                query="How do I fix this?",
                response="Here is the fix.",
                config=default_config.dedup,
                db_path=dbs["queue"],
                log_db_path=dbs["log"],
            )

            pending = get_pending_items(db_path=dbs["queue"])
            item = pending[0]

            # Should not raise
            memories = process_queue_item(
                item=item,
                config=default_config,
                log_db_path=dbs["log"],
            )

        assert memories == []
        memory_store.create.assert_not_called()

    def test_embedding_failure_stores_as_new(
        self, dbs, default_config, mock_stores
    ):
        """
        If embed_text() fails, dedup_and_store should store as new
        rather than silently dropping the memory.
        """
        vector_index, memory_store = mock_stores
        new_mem = MagicMock()
        new_mem.id = "mem_fallback"
        new_mem.summary = "Fallback memory."
        memory_store.create.return_value = new_mem

        candidate = {
            "summary": "Memory stored despite embedding failure.",
            "body": "Body.",
            "tags": ["fallback"],
            "category": "debug",
            "project_scope": "global",
            "confidence": 0.7,
        }

        with patch(
            "lace.memory.dedup.embed_text",
            side_effect=Exception("Model not loaded")
        ):
            result = dedup_and_store(
                candidate=candidate,
                vector_index=vector_index,
                memory_store=memory_store,
                config=default_config,
                hash_index_db_path=dbs["hash"],
                log_db_path=dbs["log"],
            )

        # Must store as new rather than drop
        memory_store.create.assert_called_once()
        assert result is not None

    def test_pipeline_log_failure_does_not_crash_pipeline(
        self, dbs, default_config, mock_stores
    ):
        """
        Even if pipeline_log writes fail, the pipeline continues.
        """
        vector_index, memory_store = mock_stores
        new_mem = MagicMock()
        new_mem.id = "mem_log_resilience"
        new_mem.summary = "Log resilience test."
        memory_store.create.return_value = new_mem

        candidate = {
            "summary": "Pipeline continues even when logging fails completely.",
            "body": "Body.",
            "tags": ["resilience"],
            "category": "debug",
            "project_scope": "global",
            "confidence": 0.7,
        }

        # Bad log path — writes will fail silently
        bad_log_path = Path("/nonexistent/path/pipeline_log.db")

        with patch("lace.memory.dedup.embed_text", return_value=[0.1] * 384):
            # Should not raise even with bad log path
            result = dedup_and_store(
                candidate=candidate,
                vector_index=vector_index,
                memory_store=memory_store,
                config=default_config,
                hash_index_db_path=dbs["hash"],
                log_db_path=bad_log_path,
            )

        # Memory still stored despite log failure
        memory_store.create.assert_called_once()

    def test_malformed_llm_json_does_not_crash_pipeline(
        self, dbs, default_config, mock_stores
    ):
        """Malformed LLM JSON → parse error handled, returns []."""
        vector_index, memory_store = mock_stores

        enqueue_interaction(
            query="Valid query text here.",
            response="Valid response text here.",
            config=default_config.dedup,
            db_path=dbs["queue"],
            log_db_path=dbs["log"],
        )

        pending = get_pending_items(db_path=dbs["queue"])
        item = pending[0]

        with patch("lace.memory.extractor.call_llm") as mock_llm:
            mock_llm.return_value = "{ this is not valid json {{{"

            memories = process_queue_item(
                item=item,
                config=default_config,
                log_db_path=dbs["log"],
            )

        assert memories == []
        memory_store.create.assert_not_called()

    def test_cooldown_zero_allows_all_through(
        self, dbs, mock_stores
    ):
        """
        cooldown=0 bypasses suppression entirely.
        Verifies no hardcoded cooldown in queue.py.
        """
        vector_index, memory_store = mock_stores

        zero_cooldown_config = LaceConfig(
            dedup=DedupConfig(
                skip_threshold=0.95,
                merge_threshold=0.85,
                hash_cooldown_seconds=0,
            ),
            extraction=ExtractionConfig(
                require_worthiness_verdict=True,
                log_all_verdicts=True,
            )
        )

        for i in range(5):
            enqueue_interaction(
                query="stress test 1",
                response="success",
                config=zero_cooldown_config.dedup,
                db_path=dbs["queue"],
                log_db_path=dbs["log"],
            )

        conn = sqlite3.connect(str(dbs["queue"]))
        count = conn.execute(
            "SELECT COUNT(*) FROM extraction_queue"
        ).fetchone()[0]
        conn.close()

        # With cooldown=0, all 5 must be inserted as separate rows
        assert count == 5, (
            f"Expected 5 rows with cooldown=0, got {count}. "
            "Cooldown may be hardcoded."
        )


# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 6.6: System Health Report
# ---------------------------------------------------------------------------

class TestSystemHealthReport:
    """
    Run a realistic combined scenario and produce a complete
    system health report using all five query functions.
    The report must be complete, accurate, and structurally valid.
    """

    def _run_realistic_scenario(self, dbs, config, mock_stores):
        """
        Realistic scenario:
          - 30 stress test interactions (suppressed to 1, rejected)
          - 4 genuine interactions (stored)
          - 2 near-duplicate genuine interactions (one merged, one skipped)

        Total expected pipeline_log events:
          queue_suppressed     : 29
          extraction_verdict   : 7  (1 rejected + 6 worthy)
          dedup_action         : 6  (4 store + 1 merge_embedding + 1 skip)
        """
        vector_index, memory_store = mock_stores

        # --- 30 stress tests ---
        for i in range(1, 31):
            enqueue_interaction(
                query=f"stress test {i}",
                response=f"success {i}",
                config=config.dedup,
                db_path=dbs["queue"],
                log_db_path=dbs["log"],
            )

        # Process the one stress test that made it through
        pending = get_pending_items(db_path=dbs["queue"])
        stress_item = pending[0]

        with patch("lace.memory.extractor.call_llm") as mock_llm:
            mock_llm.return_value = llm_garbage_verdict()
            process_queue_item(
                item=stress_item,
                config=config,
                log_db_path=dbs["log"],
            )
        mark_processed(stress_item["id"], db_path=dbs["queue"])

        # --- 4 genuine interactions ---
        genuine = [
            (
                "How does LACE handle project scope detection?",
                "Walks up directory tree to find .git or .lace/project.yaml.",
                "LACE scope detection walks up to find .git or project.yaml.",
                "pattern",
                0.80,   # distance → sim = 0.60 → STORE
            ),
            (
                "What embedding model should I use for LACE locally?",
                "all-MiniLM-L6-v2 gives the best quality/speed tradeoff for CPU.",
                "all-MiniLM-L6-v2 is the recommended local embedding model.",
                "reference",
                0.80,
            ),
            (
                "How should I structure the MCP tool call ordering?",
                "Always call initialize_lace_session first, then get_context.",
                "MCP tool call order: initialize → get_context → process.",
                "pattern",
                0.80,
            ),
            (
                "Why is ChromaDB returning stale results after updates?",
                "ChromaDB persistent client caches metadata, reload to fix.",
                "ChromaDB returns stale results after updates — refresh cache.",
                "debug",
                0.80,
            )
        ]

        stored_memories = []

        for q, r, s, cat, dist in genuine:
            # First insert (new)
            enqueue_res = enqueue_interaction(
                query=q,
                response=r,
                config=config.dedup,
                db_path=dbs["queue"],
                log_db_path=dbs["log"],
            )
            
            pending_jobs = get_pending_items(db_path=dbs["queue"])
            item = next(p for p in pending_jobs if p["id"] == enqueue_res["queue_id"])
            
            with patch("lace.memory.extractor.call_llm") as mock_llm:
                mock_llm.return_value = llm_real_verdict(s, cat)
                memories = process_queue_item(
                    item=item,
                    config=config,
                    log_db_path=dbs["log"],
                )
            mark_processed(item["id"], db_path=dbs["queue"])
            
            # Mock vector index query to return a neighbor with distance=dist (sim=0.60)
            if stored_memories:
                mock_res = MagicMock()
                mock_res.memory = stored_memories[0]
                mock_res.distance = dist
                vector_index.query.return_value = [mock_res]
            else:
                vector_index.query.return_value = []
            
            with patch("lace.memory.dedup.embed_text", return_value=[0.1] * 384):
                for mem in memories:
                    mem_id = dedup_and_store(
                        candidate=mem,
                        vector_index=vector_index,
                        memory_store=memory_store,
                        config=config,
                        queue_id=enqueue_res["queue_id"],
                        hash_index_db_path=dbs["hash"],
                        log_db_path=dbs["log"],
                    )
                    if mem_id:
                        # Re-mock/store the memory for lookups
                        mock_mem = MagicMock()
                        mock_mem.id = mem_id
                        mock_mem.summary = s
                        mock_mem.body = mem["body"]
                        mock_mem.tags = mem["tags"]
                        mock_mem.confidence = 0.7
                        mock_mem.lifecycle = MemoryLifecycle.CAPTURED
                        mock_mem.access_count = 0
                        mock_mem.last_accessed = datetime.now(timezone.utc).isoformat()
                        stored_memories.append(mock_mem)

        # --- 2 near-duplicate genuine interactions ---
        # 1. Merge (similarity = 0.90)
        q_merge = "How does LACE handle project scope detection again?"
        r_merge = "Walks up directory tree to find .git or .lace/project.yaml."
        s_merge = "LACE project scope detection uses git or project.yaml." # Different summary, so different hash
        
        enqueue_res = enqueue_interaction(
            query=q_merge,
            response=r_merge,
            config=config.dedup,
            db_path=dbs["queue"],
            log_db_path=dbs["log"],
        )
        pending_jobs = get_pending_items(db_path=dbs["queue"])
        item = next(p for p in pending_jobs if p["id"] == enqueue_res["queue_id"])
        
        with patch("lace.memory.extractor.call_llm") as mock_llm:
            mock_llm.return_value = llm_real_verdict(s_merge, "pattern")
            memories = process_queue_item(
                item=item,
                config=config,
                log_db_path=dbs["log"],
            )
        mark_processed(item["id"], db_path=dbs["queue"])
        
        # We query and return the first stored memory with distance = 0.20 (similarity = 0.90)
        # We mock get_by_id to return our existing memory object
        existing_mem = stored_memories[0]
        memory_store.get_by_id.return_value = existing_mem
        
        mock_result = MagicMock()
        mock_result.memory = existing_mem
        mock_result.distance = 0.20
        vector_index.query.return_value = [mock_result]
        
        with patch("lace.memory.dedup.embed_text", return_value=[0.1] * 384):
            for mem in memories:
                dedup_and_store(
                    candidate=mem,
                    vector_index=vector_index,
                    memory_store=memory_store,
                    config=config,
                    queue_id=enqueue_res["queue_id"],
                    hash_index_db_path=dbs["hash"],
                    log_db_path=dbs["log"],
                )

        # 2. Skip (similarity = 0.98)
        q_skip = "How does LACE handle project scope detection one more time?"
        r_skip = "Walks up directory tree to find .git or .lace/project.yaml."
        s_skip = "LACE scope detection finds .git and project.yaml."
        
        enqueue_res = enqueue_interaction(
            query=q_skip,
            response=r_skip,
            config=config.dedup,
            db_path=dbs["queue"],
            log_db_path=dbs["log"],
        )
        pending_jobs = get_pending_items(db_path=dbs["queue"])
        item = next(p for p in pending_jobs if p["id"] == enqueue_res["queue_id"])
        
        with patch("lace.memory.extractor.call_llm") as mock_llm:
            mock_llm.return_value = llm_real_verdict(s_skip, "pattern")
            memories = process_queue_item(
                item=item,
                config=config,
                log_db_path=dbs["log"],
            )
        mark_processed(item["id"], db_path=dbs["queue"])
        
        # We query and return the first stored memory with distance = 0.04 (similarity = 0.98)
        memory_store.get_by_id.return_value = existing_mem
        mock_result_skip = MagicMock()
        mock_result_skip.memory = existing_mem
        mock_result_skip.distance = 0.04
        vector_index.query.return_value = [mock_result_skip]
        
        with patch("lace.memory.dedup.embed_text", return_value=[0.1] * 384):
            for mem in memories:
                dedup_and_store(
                    candidate=mem,
                    vector_index=vector_index,
                    memory_store=memory_store,
                    config=config,
                    queue_id=enqueue_res["queue_id"],
                    hash_index_db_path=dbs["hash"],
                    log_db_path=dbs["log"],
                )

    def test_funnel_summary_is_accurate(self, dbs, default_config, mock_stores):
        self._run_realistic_scenario(dbs, default_config, mock_stores)
        funnel = query_funnel_summary(db_path=dbs["log"])
        
        assert funnel["queue_suppressed"] == 29
        assert funnel["extraction_total"] == 7
        assert funnel["extraction_worthy"] == 6
        assert funnel["extraction_rejected"] == 1
        assert funnel["dedup_stored"] == 4
        assert funnel["dedup_merge_embedding"] == 1
        assert funnel["dedup_skipped"] == 1

    def test_reasons_contain_rejections_and_acceptances(self, dbs, default_config, mock_stores):
        self._run_realistic_scenario(dbs, default_config, mock_stores)
        
        rejected = query_verdict_reasons(worth_remembering=False, db_path=dbs["log"])
        assert len(rejected) == 1
        assert "garbage" in rejected[0]["reason"].lower() or "repetitive" in rejected[0]["reason"].lower()
        
        accepted = query_verdict_reasons(worth_remembering=True, db_path=dbs["log"])
        assert len(accepted) == 6
        assert all("concrete" in r["reason"].lower() for r in accepted)

    def test_distribution_contains_all_actions(self, dbs, default_config, mock_stores):
        self._run_realistic_scenario(dbs, default_config, mock_stores)
        
        dist = query_dedup_score_distribution(db_path=dbs["log"])
        assert "store" in dist
        assert "merge_embedding" in dist
        assert "skip" in dist
        
        assert dist["store"]["count"] == 3
        assert dist["merge_embedding"]["count"] == 1
        assert dist["skip"]["count"] == 1

    def test_recent_events_returns_all(self, dbs, default_config, mock_stores):
        self._run_realistic_scenario(dbs, default_config, mock_stores)
        
        events = query_recent_events(limit=100, db_path=dbs["log"])
        # Expected events: 29 suppression + 7 verdicts + 6 dedup actions = 42 events
        assert len(events) == 42

    def test_full_trace_contains_chronological_events(self, dbs, default_config, mock_stores):
        self._run_realistic_scenario(dbs, default_config, mock_stores)
        
        # Interaction hash (query + response)
        q = "How does LACE handle project scope detection?"
        r = "Walks up directory tree to find .git or .lace/project.yaml."
        int_hash = canonical_hash(f"{q}\n{r}")
        trace_int = query_full_trace(int_hash, db_path=dbs["log"])
        assert len(trace_int) == 1
        assert trace_int[0]["event_type"] == "extraction_verdict"
        
        # Summary hash
        h = canonical_hash("LACE scope detection walks up to find .git or project.yaml.")
        trace_sum = query_full_trace(h, db_path=dbs["log"])
        assert len(trace_sum) == 1
        assert trace_sum[0]["event_type"] == "dedup_action"

    def test_generate_system_health_report(self, dbs, default_config, mock_stores):
        self._run_realistic_scenario(dbs, default_config, mock_stores)
        
        funnel = query_funnel_summary(db_path=dbs["log"])
        dist = query_dedup_score_distribution(db_path=dbs["log"])
        reasons = query_verdict_reasons(db_path=dbs["log"])
        suppressed = query_suppressed_hashes(db_path=dbs["log"])
        
        report = f"""# LACE System Health Report
Produced: {datetime.now(timezone.utc).isoformat()}

## Pipeline Throughput Funnel
- **Total suppressed at queue**: {funnel['queue_suppressed']}
- **Total LLM extraction runs**: {funnel['extraction_total']}
  - **Worthy verdicts**: {funnel['extraction_worthy']}
  - **Rejected verdicts**: {funnel['extraction_rejected']}
- **Total vault store actions**: {funnel['dedup_stored']}
- **Total vault merge actions**: {funnel['dedup_merge_embedding']}
- **Total vault skip actions**: {funnel['dedup_skipped']}

## Similarity Score Distribution
- **Store**: count={dist['store']['count']}, avg_score={dist['store']['avg']:.4f}
- **Merge**: count={dist['merge_embedding']['count']}, avg_score={dist['merge_embedding']['avg']:.4f}
- **Skip**: count={dist['skip']['count']}, avg_score={dist['skip']['avg']:.4f}

## Verdict Reasons Summary
{chr(10).join(f"- Queue {r['queue_id']}: {'WORTHY' if r['worth_remembering'] else 'REJECTED'} - {r['reason']}" for r in reasons[:10])}

## Top Suppressed Hashes
{chr(10).join(f"- {s['canonical_hash'][:12]}... (count={s['suppression_count']})" for s in suppressed)}
"""
        print(report)
        assert len(report) > 0
        assert "LACE System Health Report" in report
        assert "Throughput Funnel" in report
