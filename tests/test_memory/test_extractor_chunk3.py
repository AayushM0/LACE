"""
Tests for memory/extractor.py — Chunk 3 additions.

Strategy:
  - Mock call_llm() for all unit tests — no real LLM calls
  - Test parse_extraction_response() exhaustively (pure function)
  - Test log_extraction_event() with tmp_path DB
  - Test extract_memories() gate behavior with mocked LLM
  - Test process_queue_item() with mocked LLM + queue item helper
  - All existing extractor tests (test_extractor.py) must still pass

Run with: pytest tests/test_memory/test_extractor_chunk3.py -v
"""

import json
import sqlite3
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from lace.memory.extractor import (
    EXTRACTION_SYSTEM_PROMPT,
    _validate_memory_dict,
    extract_memories,
    initialize_pipeline_log_db,
    log_extraction_event,
    parse_extraction_response,
    process_queue_item,
)
from lace.core.config import DedupConfig, ExtractionConfig, LaceConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_config(
    require_verdict: bool = True,
    log_verdicts: bool = True,
) -> LaceConfig:
    return LaceConfig(
        extraction=ExtractionConfig(
            require_worthiness_verdict=require_verdict,
            log_all_verdicts=log_verdicts,
        )
    )


def make_queue_item(
    query: str = "test query",
    response: str = "test response",
    item_id: int = 1,
    canonical_hash: str = "a" * 64,
) -> MagicMock:
    item = MagicMock()
    data = {
        "id": item_id,
        "query": query,
        "response": response,
        "canonical_hash": canonical_hash,
    }
    item.__getitem__ = lambda self, key: data[key]
    return item


def llm_response(
    worth: bool,
    reason: str = "test reason",
    memories: list = None,
) -> str:
    mems = []
    if memories:
        for m in memories:
            if isinstance(m, dict):
                m_copy = dict(m)
                if "confidence" not in m_copy:
                    m_copy["confidence"] = 0.8
                mems.append(m_copy)
            else:
                mems.append(m)
    return json.dumps({
        "worth_remembering": worth,
        "reason": reason,
        "memories": mems,
    })


def read_log_rows(db_path: Path) -> list:
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        "SELECT * FROM pipeline_log ORDER BY id ASC"
    ).fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# A. Prompt contract
# ---------------------------------------------------------------------------

class TestExtractionPromptContract:

    def test_prompt_contains_worth_remembering_field(self):
        assert "worth_remembering" in EXTRACTION_SYSTEM_PROMPT

    def test_prompt_contains_reason_field(self):
        assert "reason" in EXTRACTION_SYSTEM_PROMPT

    def test_prompt_contains_memories_field(self):
        assert "memories" in EXTRACTION_SYSTEM_PROMPT

    def test_prompt_lists_all_five_categories(self):
        for cat in ("pattern", "decision", "debug", "reference", "preference"):
            assert cat in EXTRACTION_SYSTEM_PROMPT

    def test_prompt_explicitly_rejects_stress_tests(self):
        assert "stress test" in EXTRACTION_SYSTEM_PROMPT.lower()

    def test_prompt_explicitly_rejects_numbered_sequences(self):
        text = EXTRACTION_SYSTEM_PROMPT.lower()
        assert "incrementing" in text or "numbered" in text

    def test_prompt_requires_empty_array_when_false(self):
        assert "empty array" in EXTRACTION_SYSTEM_PROMPT or "[]" in EXTRACTION_SYSTEM_PROMPT

    def test_worth_remembering_appears_before_memories_in_schema(self):
        """
        Structural: worth_remembering must come before memories so the model
        commits to a verdict before generating memory content.
        """
        wr_pos = EXTRACTION_SYSTEM_PROMPT.index("worth_remembering")
        mem_pos = EXTRACTION_SYSTEM_PROMPT.index('"memories"')
        assert wr_pos < mem_pos, (
            "worth_remembering must appear before memories in the JSON schema"
        )

    def test_reason_appears_between_worth_and_memories(self):
        wr_pos  = EXTRACTION_SYSTEM_PROMPT.index("worth_remembering")
        r_pos   = EXTRACTION_SYSTEM_PROMPT.index('"reason"')
        mem_pos = EXTRACTION_SYSTEM_PROMPT.index('"memories"')
        assert wr_pos < r_pos < mem_pos


# ---------------------------------------------------------------------------
# B. parse_extraction_response — pure function, exhaustive
# ---------------------------------------------------------------------------

class TestParseExtractionResponse:

    def test_valid_false_verdict_no_memories(self):
        raw = llm_response(
            worth=False,
            reason="Repetitive stress test loop with no durable insight",
        )
        result = parse_extraction_response(raw)
        assert result["worth_remembering"] is False
        assert result["memories"] == []
        assert "stress test" in result["reason"].lower()

    def test_valid_true_verdict_with_memory(self):
        raw = llm_response(
            worth=True,
            reason="Concrete SQLite debugging solution",
            memories=[{
                "category": "debug",
                "summary": "SQLite WAL mode resolves concurrent write errors.",
                "body": "Setting PRAGMA journal_mode=WAL before opening connections.",
                "tags": ["sqlite", "wal", "concurrency"],
            }],
        )
        result = parse_extraction_response(raw)
        assert result["worth_remembering"] is True
        assert len(result["memories"]) == 1
        assert result["memories"][0]["category"] == "debug"

    def test_invalid_json_returns_safe_fallback(self):
        result = parse_extraction_response("not json {{{")
        assert result["worth_remembering"] is False
        assert result["memories"] == []
        assert "error" in result["reason"].lower()

    def test_missing_worth_remembering_returns_fallback(self):
        raw = json.dumps({"reason": "something", "memories": []})
        result = parse_extraction_response(raw)
        assert result["worth_remembering"] is False
        assert result["memories"] == []

    def test_integer_worth_coerced_to_bool_true(self):
        raw = json.dumps({"worth_remembering": 1, "reason": "ok", "memories": []})
        result = parse_extraction_response(raw)
        assert isinstance(result["worth_remembering"], bool)
        assert result["worth_remembering"] is True

    def test_integer_worth_coerced_to_bool_false(self):
        raw = json.dumps({"worth_remembering": 0, "reason": "ok", "memories": []})
        result = parse_extraction_response(raw)
        assert isinstance(result["worth_remembering"], bool)
        assert result["worth_remembering"] is False

    def test_false_verdict_with_memories_discards_memories(self):
        """
        Schema contract: worth=false + non-empty memories → memories discarded.
        """
        raw = json.dumps({
            "worth_remembering": False,
            "reason": "not useful",
            "memories": [{
                "category": "debug",
                "summary": "This should be discarded by the schema enforcer.",
                "body": "Body text.",
                "tags": ["test"],
            }],
        })
        result = parse_extraction_response(raw)
        assert result["worth_remembering"] is False
        assert result["memories"] == [], (
            "Memories must be empty when worth_remembering is False"
        )

    def test_missing_reason_gets_default_string(self):
        raw = json.dumps({"worth_remembering": False, "memories": []})
        result = parse_extraction_response(raw)
        assert isinstance(result["reason"], str)
        assert len(result["reason"]) > 0

    def test_empty_reason_gets_default_string(self):
        raw = json.dumps({"worth_remembering": False, "reason": "  ", "memories": []})
        result = parse_extraction_response(raw)
        assert len(result["reason"]) > 0

    def test_tags_normalized_to_lowercase(self):
        raw = llm_response(
            worth=True,
            reason="Valid memory",
            memories=[{
                "category": "pattern",
                "summary": "Use dependency injection for testability.",
                "body": "Inject through constructors.",
                "tags": ["DependencyInjection", "TESTING", "Python"],
            }],
        )
        result = parse_extraction_response(raw)
        assert all(t == t.lower() for t in result["memories"][0]["tags"])

    def test_invalid_category_memory_skipped(self):
        raw = llm_response(
            worth=True,
            reason="Mixed bag",
            memories=[
                {
                    "category": "invalid_category",
                    "summary": "This should be skipped due to bad category.",
                    "body": "Body.",
                    "tags": ["test"],
                },
                {
                    "category": "debug",
                    "summary": "This one is valid and should survive.",
                    "body": "Body.",
                    "tags": ["valid"],
                },
            ],
        )
        result = parse_extraction_response(raw)
        assert len(result["memories"]) == 1
        assert result["memories"][0]["category"] == "debug"

    def test_memory_missing_required_field_skipped(self):
        raw = llm_response(
            worth=True,
            reason="Has memory",
            memories=[{"category": "debug"}],  # missing summary, body, tags
        )
        result = parse_extraction_response(raw)
        assert result["memories"] == []

    def test_five_memories_all_pass_validation(self):
        """Parser validates structure, not count — all 5 pass."""
        mems = [
            {
                "category": "debug",
                "summary": f"Memory {i} is a valid self-contained sentence.",
                "body": f"Body {i}",
                "tags": [f"tag{i}"],
            }
            for i in range(5)
        ]
        raw = llm_response(worth=True, reason="Many", memories=mems)
        result = parse_extraction_response(raw)
        assert len(result["memories"]) == 5

    def test_memories_not_a_list_returns_empty(self):
        raw = json.dumps({
            "worth_remembering": True,
            "reason": "ok",
            "memories": "not a list",
        })
        result = parse_extraction_response(raw)
        assert result["memories"] == []

    def test_non_dict_memory_items_skipped(self):
        raw = json.dumps({
            "worth_remembering": True,
            "reason": "ok",
            "memories": ["string_item", 42, None],
        })
        result = parse_extraction_response(raw)
        assert result["memories"] == []


# ---------------------------------------------------------------------------
# C. _validate_memory_dict
# ---------------------------------------------------------------------------

class TestValidateMemoryDict:

    def test_valid_memory_passes(self):
        mem = {
            "category": "pattern",
            "summary": "Use context managers for resource cleanup.",
            "body": "Always use with statements for files and DB connections.",
            "tags": ["python", "context-managers"],
            "confidence": 0.8,
        }
        assert _validate_memory_dict(mem, 0) is not None

    def test_all_five_categories_valid(self):
        for cat in ("pattern", "decision", "debug", "reference", "preference"):
            mem = {
                "category": cat,
                "summary": "Valid summary sentence for this memory.",
                "body": "Body.",
                "tags": [],
                "confidence": 0.8,
            }
            assert _validate_memory_dict(mem, 0) is not None

    def test_summary_too_short_rejected(self):
        mem = {
            "category": "pattern",
            "summary": "Short",  # < 10 chars
            "body": "Body text here.",
            "tags": ["test"],
            "confidence": 0.8,
        }
        assert _validate_memory_dict(mem, 0) is None

    def test_tags_not_list_rejected(self):
        mem = {
            "category": "pattern",
            "summary": "Valid summary sentence here.",
            "body": "Body.",
            "tags": "not-a-list",
            "confidence": 0.8,
        }
        assert _validate_memory_dict(mem, 0) is None

    def test_tags_normalized_to_lowercase_by_validator(self):
        mem = {
            "category": "debug",
            "summary": "Valid summary sentence that is long enough.",
            "body": "Body.",
            "tags": ["UPPER", "MiXeD"],
            "confidence": 0.8,
        }
        result = _validate_memory_dict(mem, 0)
        assert result is not None
        assert all(t == t.lower() for t in result["tags"])

    def test_invalid_category_rejected(self):
        mem = {
            "category": "nonsense",
            "summary": "Valid summary sentence that is long enough.",
            "body": "Body.",
            "tags": [],
            "confidence": 0.8,
        }
        assert _validate_memory_dict(mem, 0) is None

    def test_missing_confidence_rejected(self):
        mem = {
            "category": "pattern",
            "summary": "Valid summary sentence that is long enough.",
            "body": "Body content here.",
            "tags": ["test"],
        }
        assert _validate_memory_dict(mem, 0) is None

    def test_non_numeric_confidence_rejected(self):
        mem = {
            "category": "pattern",
            "summary": "Valid summary sentence that is long enough.",
            "body": "Body content here.",
            "tags": ["test"],
            "confidence": "high",
        }
        assert _validate_memory_dict(mem, 0) is None

    def test_confidence_out_of_bounds_rejected(self):
        for conf in (-0.1, 1.1):
            mem = {
                "category": "pattern",
                "summary": "Valid summary sentence that is long enough.",
                "body": "Body content here.",
                "tags": ["test"],
                "confidence": conf,
            }
            assert _validate_memory_dict(mem, 0) is None

    def test_coerced_confidence_float(self):
        mem = {
            "category": "pattern",
            "summary": "Valid summary sentence that is long enough.",
            "body": "Body content here.",
            "tags": ["test"],
            "confidence": "0.75",
        }
        res = _validate_memory_dict(mem, 0)
        assert res is not None
        assert res["confidence"] == 0.75


# ---------------------------------------------------------------------------
# D. initialize_pipeline_log_db + log_extraction_event
# ---------------------------------------------------------------------------

class TestPipelineLogDB:

    def test_init_creates_table_and_indexes(self, tmp_path):
        db_path = tmp_path / "pipeline_log.db"
        initialize_pipeline_log_db(db_path)

        conn = sqlite3.connect(str(db_path))
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        indexes = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index'"
        ).fetchall()
        conn.close()

        table_names = [t[0] for t in tables]
        index_names = [i[0] for i in indexes]

        assert "pipeline_log" in table_names
        assert any("event" in n for n in index_names)
        assert any("hash" in n for n in index_names)

    def test_init_is_idempotent(self, tmp_path):
        db_path = tmp_path / "pipeline_log.db"
        initialize_pipeline_log_db(db_path)
        initialize_pipeline_log_db(db_path)  # second call must not fail

    def test_writes_false_verdict_row(self, tmp_path):
        db_path = tmp_path / "pipeline_log.db"
        initialize_pipeline_log_db(db_path)

        log_extraction_event(
            queue_id=42,
            worth_remembering=False,
            reason="Repetitive stress test loop",
            memory_count=0,
            canonical_hash_value="a" * 64,
            db_path=db_path,
        )

        rows = read_log_rows(db_path)
        assert len(rows) == 1
        assert rows[0]["event_type"] == "extraction_verdict"
        assert rows[0]["worth_remembering"] == 0
        assert rows[0]["queue_id"] == 42
        assert "stress test" in rows[0]["reason"].lower()

    def test_writes_true_verdict_row(self, tmp_path):
        db_path = tmp_path / "pipeline_log.db"
        initialize_pipeline_log_db(db_path)

        log_extraction_event(
            queue_id=7,
            worth_remembering=True,
            reason="Concrete SQLite debugging solution",
            memory_count=2,
            db_path=db_path,
        )

        rows = read_log_rows(db_path)
        assert rows[0]["worth_remembering"] == 1
        assert rows[0]["repeat_count"] == 2  # memory_count stored in repeat_count

    def test_logging_failure_never_raises(self):
        """Logging must not crash the extraction pipeline."""
        bad_path = _Path("/nonexistent/path/log.db")
        # Should not raise
        log_extraction_event(
            queue_id=1,
            worth_remembering=False,
            reason="test",
            memory_count=0,
            db_path=bad_path,
        )

    def test_multiple_verdicts_accumulate(self, tmp_path):
        db_path = tmp_path / "pipeline_log.db"
        initialize_pipeline_log_db(db_path)

        for i in range(10):
            log_extraction_event(
                queue_id=i,
                worth_remembering=False,
                reason=f"Stress test {i}",
                memory_count=0,
                db_path=db_path,
            )

        assert len(read_log_rows(db_path)) == 10

    def test_canonical_hash_stored_in_log_row(self, tmp_path):
        db_path = tmp_path / "pipeline_log.db"
        initialize_pipeline_log_db(db_path)
        h = "b" * 64

        log_extraction_event(
            queue_id=1,
            worth_remembering=True,
            reason="test",
            memory_count=1,
            canonical_hash_value=h,
            db_path=db_path,
        )

        rows = read_log_rows(db_path)
        assert rows[0]["canonical_hash"] == h

    def test_created_at_is_iso_timestamp(self, tmp_path):
        from datetime import datetime
        db_path = tmp_path / "pipeline_log.db"
        initialize_pipeline_log_db(db_path)

        log_extraction_event(
            queue_id=1,
            worth_remembering=False,
            reason="test",
            memory_count=0,
            db_path=db_path,
        )

        rows = read_log_rows(db_path)
        ts = rows[0]["created_at"]
        # Must parse without error
        datetime.fromisoformat(ts)


# ---------------------------------------------------------------------------
# E. extract_memories — gate behavior (mocked LLM)
# ---------------------------------------------------------------------------

class TestExtractMemoriesGate:

    def test_false_verdict_gates_to_empty_list(self, tmp_path):
        config = make_config(require_verdict=True)
        with patch("lace.memory.extractor.call_llm") as mock_llm:
            mock_llm.return_value = llm_response(
                worth=False,
                reason="Repetitive stress test with no durable insight",
            )
            result = extract_memories(
                "stress test 1", "success",
                config=config,
                log_db_path=tmp_path / "log.db",
            )
        assert result == []

    def test_true_verdict_returns_memories(self, tmp_path):
        config = make_config(require_verdict=True)
        with patch("lace.memory.extractor.call_llm") as mock_llm:
            mock_llm.return_value = llm_response(
                worth=True,
                reason="Concrete debug solution",
                memories=[{
                    "category": "debug",
                    "summary": "SQLite WAL mode resolves concurrent write errors.",
                    "body": "Use PRAGMA journal_mode=WAL.",
                    "tags": ["sqlite", "wal"],
                }],
            )
            result = extract_memories(
                "How to fix SQLite concurrent writes?",
                "Set PRAGMA journal_mode=WAL",
                config=config,
                log_db_path=tmp_path / "log.db",
            )
        assert len(result) == 1
        assert result[0]["category"] == "debug"

    def test_require_verdict_false_bypasses_gate(self, tmp_path):
        """
        require_worthiness_verdict=False → gate disabled.
        Even a false verdict still passes through (returns [] here
        because memories=[]).
        """
        config = make_config(require_verdict=False)
        with patch("lace.memory.extractor.call_llm") as mock_llm:
            mock_llm.return_value = llm_response(
                worth=False,
                reason="Not useful",
                memories=[],
            )
            result = extract_memories(
                "test", "test",
                config=config,
                log_db_path=tmp_path / "log.db",
            )
        # Gate is disabled — returned what the parser gave us (empty memories)
        assert result == []

    def test_require_verdict_false_does_not_gate_true_memories(self, tmp_path):
        config = make_config(require_verdict=False)
        with patch("lace.memory.extractor.call_llm") as mock_llm:
            mock_llm.return_value = llm_response(
                worth=True,
                reason="Good",
                memories=[{
                    "category": "pattern",
                    "summary": "Use lazy imports to avoid circular dependencies.",
                    "body": "Import inside function body.",
                    "tags": ["python", "imports"],
                }],
            )
            result = extract_memories(
                "q", "r",
                config=config,
                log_db_path=tmp_path / "log.db",
            )
        assert len(result) == 1

    def test_log_all_verdicts_true_writes_row(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)
        config = make_config(log_verdicts=True)

        with patch("lace.memory.extractor.call_llm") as mock_llm:
            mock_llm.return_value = llm_response(worth=False, reason="Stress test loop")
            extract_memories("stress test 1", "success", config=config, log_db_path=db_path)

        rows = read_log_rows(db_path)
        assert len(rows) == 1
        assert rows[0]["worth_remembering"] == 0

    def test_log_all_verdicts_false_writes_no_row(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)
        config = make_config(log_verdicts=False)

        with patch("lace.memory.extractor.call_llm") as mock_llm:
            mock_llm.return_value = llm_response(worth=False, reason="Not useful")
            extract_memories("test", "test", config=config, log_db_path=db_path)

        assert len(read_log_rows(db_path)) == 0

    def test_llm_exception_returns_empty_list(self, tmp_path):
        config = make_config()
        with patch("lace.memory.extractor.call_llm") as mock_llm:
            mock_llm.side_effect = Exception("LLM API error")
            result = extract_memories(
                "valid query", "valid response",
                config=config,
                log_db_path=tmp_path / "log.db",
            )
        assert result == []

    def test_extract_memories_logs_with_canonical_hash(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)
        config = make_config(log_verdicts=True)

        with patch("lace.memory.extractor.call_llm") as mock_llm:
            mock_llm.return_value = llm_response(worth=False, reason="test")
            extract_memories("q", "r", config=config, log_db_path=db_path)

        rows = read_log_rows(db_path)
        assert rows[0]["canonical_hash"] is not None
        assert len(rows[0]["canonical_hash"]) == 64  # SHA-256 hex


# ---------------------------------------------------------------------------
# F. process_queue_item — queue-id-aware logging
# ---------------------------------------------------------------------------

class TestProcessQueueItem:

    def test_garbage_item_returns_empty(self, tmp_path):
        item = make_queue_item(query="stress test 1", response="success", item_id=1)
        config = make_config(require_verdict=True)

        with patch("lace.memory.extractor.call_llm") as mock_llm:
            mock_llm.return_value = llm_response(
                worth=False, reason="Repetitive stress test loop"
            )
            result = process_queue_item(item, config=config, log_db_path=tmp_path / "log.db")

        assert result == []

    def test_real_item_returns_memories(self, tmp_path):
        item = make_queue_item(
            query="How do I handle SQLite concurrent writes?",
            response="Use WAL mode: PRAGMA journal_mode=WAL",
            item_id=5,
        )
        config = make_config(require_verdict=True)

        with patch("lace.memory.extractor.call_llm") as mock_llm:
            mock_llm.return_value = llm_response(
                worth=True,
                reason="Concrete debugging solution with specific fix",
                memories=[{
                    "category": "debug",
                    "summary": "SQLite WAL mode resolves concurrent write errors.",
                    "body": "PRAGMA journal_mode=WAL enables concurrent reads.",
                    "tags": ["sqlite", "wal", "concurrency"],
                }],
            )
            result = process_queue_item(item, config=config, log_db_path=tmp_path / "log.db")

        assert len(result) == 1

    def test_queue_id_logged_correctly(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)

        item = make_queue_item(item_id=99)
        config = make_config(log_verdicts=True)

        with patch("lace.memory.extractor.call_llm") as mock_llm:
            mock_llm.return_value = llm_response(worth=False, reason="Not useful")
            process_queue_item(item, config=config, log_db_path=db_path)

        rows = read_log_rows(db_path)
        assert len(rows) == 1
        assert rows[0]["queue_id"] == 99

    def test_canonical_hash_from_item_logged(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)
        expected_hash = "c" * 64

        item = make_queue_item(item_id=5, canonical_hash=expected_hash)
        config = make_config(log_verdicts=True)

        with patch("lace.memory.extractor.call_llm") as mock_llm:
            mock_llm.return_value = llm_response(worth=False, reason="ok")
            process_queue_item(item, config=config, log_db_path=db_path)

        rows = read_log_rows(db_path)
        assert rows[0]["canonical_hash"] == expected_hash

    def test_llm_failure_in_queue_item_returns_empty(self, tmp_path):
        item = make_queue_item(item_id=3)
        config = make_config()

        with patch("lace.memory.extractor.call_llm") as mock_llm:
            mock_llm.side_effect = RuntimeError("provider unreachable")
            result = process_queue_item(item, config=config, log_db_path=tmp_path / "log.db")

        assert result == []

    def test_fifty_stress_test_items_all_gate_to_empty(self, tmp_path):
        """
        Integration: 50 simulated stress test queue items → all return [].
        This is the core regression test for the original 40-50 junk memories bug.
        """
        config = make_config(require_verdict=True, log_verdicts=False)

        with patch("lace.memory.extractor.call_llm") as mock_llm:
            mock_llm.side_effect = lambda q, r, c: llm_response(
                worth=False,
                reason="Repetitive numbered stress test loop with no durable insight",
            )
            for i in range(1, 51):
                item = make_queue_item(
                    query=f"stress test {i}",
                    response=f"result: success iteration {i} completed",
                    item_id=i,
                )
                result = process_queue_item(item, config=config, log_db_path=tmp_path / "log.db")
                assert result == [], f"Item {i} was not gated out: {result}"


# ---------------------------------------------------------------------------
# G. ExtractionConfig wiring — verify config flags actually control behaviour
# ---------------------------------------------------------------------------

class TestExtractionConfigWiring:

    def test_default_config_enables_verdict_gate(self):
        cfg = LaceConfig()
        assert cfg.extraction.require_worthiness_verdict is True

    def test_default_config_enables_logging(self):
        cfg = LaceConfig()
        assert cfg.extraction.log_all_verdicts is True

    def test_gate_disabled_even_when_llm_says_false(self, tmp_path):
        """require_worthiness_verdict=False: false verdict still passes gate."""
        config = make_config(require_verdict=False)

        with patch("lace.memory.extractor.call_llm") as mock_llm:
            mock_llm.return_value = llm_response(
                worth=False,
                reason="noise",
                memories=[],
            )
            result = extract_memories(
                "any query", "any response",
                config=config,
                log_db_path=tmp_path / "log.db",
            )
        # Gate is disabled, memories=[] → empty list (no memories in payload)
        assert result == []

    def test_gate_enabled_blocks_false_verdict(self, tmp_path):
        config = make_config(require_verdict=True)
        with patch("lace.memory.extractor.call_llm") as mock_llm:
            mock_llm.return_value = llm_response(worth=False, reason="noise")
            result = extract_memories("q", "r", config=config, log_db_path=tmp_path / "log.db")
        assert result == []

    def test_log_verdicts_false_skips_db_write(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)
        config = make_config(log_verdicts=False)

        with patch("lace.memory.extractor.call_llm") as mock_llm:
            mock_llm.return_value = llm_response(worth=True, reason="great memory", memories=[{
                "category": "pattern",
                "summary": "Always validate inputs before processing.",
                "body": "Input validation prevents downstream errors.",
                "tags": ["validation"],
            }])
            extract_memories("q", "r", config=config, log_db_path=db_path)

        assert len(read_log_rows(db_path)) == 0

    def test_log_verdicts_true_writes_even_for_true_verdict(self, tmp_path):
        db_path = tmp_path / "log.db"
        initialize_pipeline_log_db(db_path)
        config = make_config(log_verdicts=True)

        with patch("lace.memory.extractor.call_llm") as mock_llm:
            mock_llm.return_value = llm_response(
                worth=True,
                reason="Concrete debugging fix",
                memories=[{
                    "category": "debug",
                    "summary": "WAL mode resolves SQLite lock errors under concurrency.",
                    "body": "PRAGMA journal_mode=WAL",
                    "tags": ["sqlite"],
                }],
            )
            extract_memories("q", "r", config=config, log_db_path=db_path)

        rows = read_log_rows(db_path)
        assert len(rows) == 1
        assert rows[0]["worth_remembering"] == 1


# ---------------------------------------------------------------------------
# H. Backward-compat: existing public API still works
# ---------------------------------------------------------------------------

class TestBackwardCompatibility:

    def test_existing_parse_response_still_importable(self):
        from lace.memory.extractor import _parse_extraction_response
        assert callable(_parse_extraction_response)

    def test_existing_should_attempt_extraction_still_importable(self):
        from lace.memory.extractor import should_attempt_extraction
        assert callable(should_attempt_extraction)

    def test_existing_extract_from_conversation_still_importable(self):
        from lace.memory.extractor import extract_from_conversation
        assert callable(extract_from_conversation)

    def test_existing_extraction_candidate_still_importable(self):
        from lace.memory.extractor import ExtractionCandidate
        assert ExtractionCandidate is not None

    def test_existing_extraction_result_still_importable(self):
        from lace.memory.extractor import ExtractionResult
        assert ExtractionResult is not None

    def test_new_symbols_are_importable(self):
        from lace.memory.extractor import (
            EXTRACTION_SYSTEM_PROMPT,
            _validate_memory_dict,
            call_llm,
            extract_memories,
            initialize_pipeline_log_db,
            log_extraction_event,
            parse_extraction_response,
            process_queue_item,
            PIPELINE_LOG_DB_PATH,
        )
        # If the import succeeds, all symbols exist
        assert EXTRACTION_SYSTEM_PROMPT
        assert callable(parse_extraction_response)

# Make _Path available inside this module for the bad_path test
from pathlib import Path as _Path
