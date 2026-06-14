"""Tests for MCP tool implementations."""

import pytest
from pathlib import Path
from lace.core.config import LaceConfig, init_lace_home
from lace.memory.store import MemoryStore


@pytest.fixture
def lace_env(tmp_path, monkeypatch):
    """Set up a temporary LACE environment."""
    lace_home = tmp_path / ".lace"
    init_lace_home(lace_home)

    # Point LACE_HOME at temp dir
    monkeypatch.setenv("LACE_HOME", str(lace_home))

    # Copy config files
    import shutil
    config_src = Path(__file__).parent.parent.parent / "config"
    if config_src.exists():
        for f in config_src.glob("*"):
            shutil.copy2(f, lace_home / "config" / f.name)

    return lace_home


@pytest.fixture
def store(lace_env):
    from lace.core.config import load_config
    config = LaceConfig()
    return MemoryStore(lace_home=lace_env, config=config)


@pytest.mark.asyncio
async def test_search_memory_returns_results(store, lace_env, monkeypatch):
    """search_memory returns relevant memories."""
    monkeypatch.setenv("LACE_HOME", str(lace_env))

    store.add("Use connection pooling with asyncpg", tags=["db"])
    store.add("Use Redis for session caching", tags=["cache"])

    from lace.mcp.tools import search_memory
    results = await search_memory("database connection", scope="global")

    assert isinstance(results, list)
    assert len(results) >= 1
    assert "id" in results[0]
    assert "relevance_score" in results[0]
    assert "content" in results[0]


@pytest.mark.asyncio
async def test_remember_stores_memory(store, lace_env, monkeypatch):
    """remember() creates a new memory."""
    monkeypatch.setenv("LACE_HOME", str(lace_env))

    from lace.mcp.tools import remember
    result = await remember(
        content="Always validate inputs with Pydantic",
        category="pattern",
        tags=["pydantic", "validation"],
        scope="global",
    )

    assert result["stored"] is True
    assert result["id"].startswith("mem_")
    assert result["category"] == "pattern"

    # Verify it was actually saved
    memory = store.get(result["id"])
    assert memory is not None
    assert memory.content == "Always validate inputs with Pydantic"


@pytest.mark.asyncio
async def test_list_memories_returns_list(store, lace_env, monkeypatch):
    """list_memories() returns a list of memory summaries."""
    monkeypatch.setenv("LACE_HOME", str(lace_env))

    store.add("Pattern memory 1", category="pattern")
    store.add("Pattern memory 2", category="pattern")
    store.add("Decision memory", category="decision")

    from lace.mcp.tools import list_memories
    results = await list_memories(category="pattern", scope="global")

    assert isinstance(results, list)
    assert all(r["category"] == "pattern" for r in results)


@pytest.mark.asyncio
async def test_forget_memory_archives(store, lace_env, monkeypatch):
    """forget_memory() archives the specified memory."""
    monkeypatch.setenv("LACE_HOME", str(lace_env))

    memory = store.add("Memory to forget")

    from lace.mcp.tools import forget_memory
    result = await forget_memory(memory.id)

    assert result["success"] is True
    assert result["lifecycle"] == "archived"

    # Verify it's archived
    retrieved = store.get(memory.id)
    assert retrieved is not None
    assert not retrieved.is_active()


@pytest.mark.asyncio
async def test_forget_memory_unknown_id(lace_env, monkeypatch):
    """forget_memory() returns error for unknown ID."""
    monkeypatch.setenv("LACE_HOME", str(lace_env))

    from lace.mcp.tools import forget_memory
    result = await forget_memory("mem_doesnotexist")

    assert result["success"] is False
    assert "error" in result


@pytest.mark.asyncio
async def test_get_project_context_returns_identity(lace_env, monkeypatch):
    """get_project_context() returns identity and preferences."""
    monkeypatch.setenv("LACE_HOME", str(lace_env))

    from lace.mcp.tools import get_project_context
    result = await get_project_context()

    assert "identity" in result
    assert "preferences" in result
    assert "scope" in result
    assert isinstance(result["identity"], str)
    assert len(result["identity"]) > 0


@pytest.mark.asyncio
async def test_get_relevant_context(store, lace_env, monkeypatch):
    """get_relevant_context returns relevant memories as markdown."""
    monkeypatch.setenv("LACE_HOME", str(lace_env))

    store.add("Always use connection pooling with asyncpg", category="pattern", confidence=0.9)

    from lace.mcp.tools import get_relevant_context
    result = await get_relevant_context("asyncpg connection pooling", scope="global")

    assert "Context from LACE Memory Vault" in result
    assert "Always use connection pooling with asyncpg" in result
    assert "Confidence: 0.90" in result


@pytest.mark.asyncio
async def test_process_interaction(lace_env, monkeypatch):
    """process_interaction updates session history and enqueues job."""
    monkeypatch.setenv("LACE_HOME", str(lace_env))

    from lace.mcp.tools import process_interaction
    from lace.mcp.queue import init_queue_db, get_pending_jobs
    import lace.mcp.server as mcp_server

    init_queue_db()

    mcp_server._mcp_session_history = []

    res = await process_interaction(
        query="test query 123",
        response="test response abc",
        scope="global",
    )

    assert res["status"] == "queued"
    assert res["job_id"] is not None

    assert len(mcp_server._mcp_session_history) == 1
    assert mcp_server._mcp_session_history[0]["query"] == "test query 123"

    pending = get_pending_jobs()
    assert len(pending) == 1
    assert pending[0]["id"] == res["job_id"]
    assert pending[0]["query"] == "test query 123"