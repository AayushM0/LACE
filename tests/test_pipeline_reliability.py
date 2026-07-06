"""Focused regressions for the LACE reliability hardening plan."""

from __future__ import annotations

import pytest


def test_extraction_prompt_requires_confidence_variance():
    from lace.memory.extractor import EXTRACTION_SYSTEM_PROMPT

    assert "Do NOT assign the same confidence value" in EXTRACTION_SYSTEM_PROMPT


def test_validate_memory_dict_warns_on_fallback_confidence(caplog):
    from lace.memory.extractor import _validate_memory_dict

    mem = {
        "category": "debug",
        "summary": "SQLite WAL mode fixes concurrent writer lock failures.",
        "body": "Use PRAGMA journal_mode=WAL for the queue database.",
        "tags": ["sqlite"],
        "confidence": 0.4,
    }

    result = _validate_memory_dict(mem, 0)

    assert result is not None
    assert "fallback confidence 0.4" in caplog.text


def test_pipeline_log_default_matches_resolved_path():
    from lace.core.config import resolve_lace_paths
    from lace.memory.pipeline_log import PIPELINE_LOG_DB_PATH

    assert PIPELINE_LOG_DB_PATH == resolve_lace_paths()["pipeline_log"]


def test_needs_reindex_round_trips_through_markdown(tmp_path):
    from lace.memory.markdown import markdown_to_memory, save_memory_to_file
    from lace.memory.models import MemoryCategory, MemoryObject

    memory = MemoryObject(
        content="Embedding failed but the memory should be recoverable later.",
        category=MemoryCategory.DEBUG,
        needs_reindex=True,
    )

    path = save_memory_to_file(memory, tmp_path)
    loaded = markdown_to_memory(path)

    assert loaded is not None
    assert loaded.needs_reindex is True


@pytest.mark.asyncio
async def test_process_interaction_uses_enqueue_interaction(monkeypatch):
    import lace.mcp.server as mcp_server
    import lace.mcp.tools as tools

    calls = []

    def fake_enqueue_interaction(**kwargs):
        calls.append(kwargs)
        return {
            "status": "queued",
            "job_id": "job-1",
            "queue_id": "job-1",
            "scope_used": kwargs["scope"],
            "action": "inserted",
        }

    monkeypatch.setattr("lace.mcp.queue.enqueue_interaction", fake_enqueue_interaction)
    monkeypatch.setattr(tools, "_mcp_context_project", "project:test")
    monkeypatch.setattr(tools, "_mcp_context_cwd", "")
    monkeypatch.setattr(mcp_server, "_mcp_session_history", [])

    result = await tools.process_interaction(
        query="What did we decide?",
        response="We decided to use the gated extraction path for reliability.",
        scope="auto",
    )

    assert result["status"] == "queued"
    assert result["job_id"] == "job-1"
    assert calls
    assert calls[0]["scope"] == "project:test"
    assert calls[0]["history"][0]["query"] == "What did we decide?"


def test_doctor_checks_wal_mode(tmp_path, monkeypatch):
    from typer.testing import CliRunner
    from lace.main import app

    # Force LACE_HOME to temp path to avoid polluting real setup
    monkeypatch.setenv("LACE_HOME", str(tmp_path))

    runner = CliRunner()
    result = runner.invoke(app, ["doctor"])
    # The output should check for WAL journal mode on both SQLite databases
    assert "journal:queue_db" in result.output
    assert "journal:pipeline_log" in result.output


def test_worker_recovers_stuck_processing_jobs(tmp_path, monkeypatch):
    import sqlite3
    from lace.mcp.queue import init_queue_db, get_queue_db_path, get_pending_jobs

    # Force LACE_HOME to temp path to avoid polluting real setup
    monkeypatch.setenv("LACE_HOME", str(tmp_path))

    # Initialize queue DB to create the tables
    init_queue_db()

    # Manually insert a job with status='processing'
    db_path = get_queue_db_path()
    conn = sqlite3.connect(str(db_path))
    conn.execute(
        """
        INSERT INTO extraction_queue (id, query, response, scope, status, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        ("stuck-job-1", "Query content", "Response content", "global", "processing", "2026-07-06T12:00:00Z")
    )
    conn.commit()
    conn.close()

    # Before recovery init, get_pending_jobs shouldn't return this job
    pending = get_pending_jobs()
    assert not any(j["id"] == "stuck-job-1" for j in pending)

    # Re-run init_queue_db which should reset stuck processing jobs to pending
    init_queue_db()

    # After recovery init, get_pending_jobs should now return the job
    pending = get_pending_jobs()
    assert any(j["id"] == "stuck-job-1" for j in pending)


def test_search_filters_by_min_confidence(tmp_path, monkeypatch):
    from lace.memory.store import MemoryStore
    from lace.core.config import load_config
    import unittest.mock as mock

    # Force LACE_HOME to temp path to avoid polluting real setup
    monkeypatch.setenv("LACE_HOME", str(tmp_path))

    store = MemoryStore(lace_home=tmp_path)
    # Mock embed to avoid calling live embeddings API
    monkeypatch.setattr(store, "_embed", lambda text: [0.1] * 384)

    # Add two mock memories
    mem_high = store.add(
        content="This is high confidence memory that should be returned.",
        confidence=0.9,
    )
    mem_low = store.add(
        content="This is low confidence memory that should be filtered.",
        confidence=0.2,
    )

    # Initialize store retriever components
    store.initialize()

    # Search with min_confidence=0.5
    # Expect only high confidence memory to be returned
    results = store.search(
        query="confidence",
        min_confidence=0.5,
    )

    assert any(r.memory.id == mem_high.id for r in results)
    assert not any(r.memory.id == mem_low.id for r in results)


def test_load_config_project_overrides(tmp_path):
    import yaml
    from lace.core.config import load_config

    # Create a simulated project directory structure
    project_dir = tmp_path / "my_project"
    lace_subdir = project_dir / ".lace"
    lace_subdir.mkdir(parents=True)

    project_yaml_content = {
        "project_name": "rescuemesh",
        "scope": "project:rescuemesh",
        "extraction": {
            "require_worthiness_verdict": False,
            "noise_profile": "high",
        },
        "dedup": {
            "merge_threshold": 0.89,  # overrides noise profile's 0.90
        }
    }

    with open(lace_subdir / "project.yaml", "w") as f:
        yaml.dump(project_yaml_content, f)

    # Load configuration passing the simulated project directory as cwd
    config = load_config(lace_home=tmp_path, cwd=project_dir)

    # Check that overrides have been applied correctly
    assert config.extraction.require_worthiness_verdict is False
    # Check that high noise profile set skip_threshold to 0.98 and hash_cooldown to 900
    assert config.dedup.skip_threshold == 0.98
    assert config.dedup.hash_cooldown_seconds == 900
    # Check that the explicit merge_threshold overrides the profile's 0.90
    assert config.dedup.merge_threshold == 0.89




def test_process_queue_once_executes_synchronously(tmp_path, monkeypatch):
    import json
    from unittest.mock import patch
    from lace.mcp.queue import init_queue_db, enqueue_interaction, get_job_status, process_queue_once

    # Force LACE_HOME to temp path to avoid polluting real setup
    monkeypatch.setenv("LACE_HOME", str(tmp_path))

    init_queue_db()

    # Enqueue a mock job
    enqueue_result = enqueue_interaction(
        query="Explain TDD loops",
        response="TDD is red-green-refactor.",
        scope="global",
    )
    job_id = enqueue_result["job_id"]

    # Mock the LLM to return memories
    fake_llm = json.dumps({
        "worth_remembering": True,
        "reason": "Explain loops",
        "memories": [{
            "category": "pattern",
            "summary": "TDD is red-green-refactor.",
            "body": "TDD process verifies behavior through public seams.",
            "tags": ["tdd"],
            "confidence": 0.9,
        }],
    })

    with patch("lace.memory.extractor.call_llm", return_value=fake_llm):
        # Run worker processing synchronously
        process_queue_once()

    # Verify that the job is marked as done
    status = get_job_status(job_id)
    assert status["status"] == "done"


def test_doctor_cleanup_prevents_pollution(tmp_path, monkeypatch):
    import os
    import sqlite3
    from typer.testing import CliRunner
    from lace.main import app
    from lace.core.config import resolve_lace_paths

    # Setup isolated LACE_HOME
    monkeypatch.setenv("LACE_HOME", str(tmp_path))
    paths = resolve_lace_paths(tmp_path)

    runner = CliRunner()
    # First init the home
    runner.invoke(app, ["init"])

    # Record baseline state of memory vault directory
    vault_dir = paths["vault"]
    vault_files_before = os.listdir(vault_dir) if vault_dir.exists() else []

    # Run the doctor command
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0

    # Check that vault files did not change (mock memories cleaned up)
    vault_files_after = os.listdir(vault_dir) if vault_dir.exists() else []
    assert len(vault_files_after) == len(vault_files_before)

    # Check hash index database has no doctor entries
    if paths["hash_index"].exists():
        with sqlite3.connect(str(paths["hash_index"])) as conn:
            c = conn.execute("SELECT COUNT(*) FROM vault_hash_index").fetchone()[0]
            assert c == 0


def test_extractor_resolves_path_dynamically(tmp_path, monkeypatch):
    from lace.memory.extractor import extract_memories
    from lace.core.config import resolve_lace_paths

    # Set LACE_HOME
    monkeypatch.setenv("LACE_HOME", str(tmp_path))

    # Mock call_llm to return immediately
    import json
    from unittest.mock import patch
    fake_llm = json.dumps({
        "worth_remembering": False,
        "reason": "Test dynamic path resolution",
        "memories": []
    })

    with patch("lace.memory.extractor.call_llm", return_value=fake_llm):
        extract_memories(query="test", response="test")

    # The log file should have been created in the new LACE_HOME
    expected_path = resolve_lace_paths(tmp_path)["pipeline_log"]
    assert expected_path.exists()







