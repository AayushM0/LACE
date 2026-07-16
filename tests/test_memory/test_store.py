"""Tests for MemoryStore CRUD operations."""

import pytest
from pathlib import Path
from lace.core.config import LaceConfig, init_lace_home
from lace.memory.store import MemoryStore
from lace.memory.models import MemoryCategory, MemoryLifecycle


@pytest.fixture
def store(tmp_path):
    """A MemoryStore pointed at a temporary directory."""
    lace_home = tmp_path / ".lace"
    init_lace_home(lace_home)
    config = LaceConfig()
    return MemoryStore(lace_home=lace_home, config=config)


def test_add_and_get(store):
    """Memory can be added and retrieved by ID."""
    memory = store.add("Use connection pooling", tags=["db"])
    retrieved = store.get(memory.id)
    assert retrieved is not None
    assert retrieved.content == "Use connection pooling"
    assert retrieved.tags == ["db"]


def test_add_creates_file(store, tmp_path):
    """Adding a memory creates a .md file in the vault."""
    memory = store.add("Test memory")
    assert memory.file_path is not None
    assert Path(memory.file_path).exists()


def test_list_returns_all_active(store):
    """list() returns all non-archived memories."""
    store.add("Memory 1")
    store.add("Memory 2")
    store.add("Memory 3")
    memories = store.list()
    assert len(memories) == 3


def test_list_excludes_archived(store):
    """list() excludes archived memories by default."""
    m1 = store.add("Active memory")
    m2 = store.add("To be archived")
    store.forget(m2.id)

    memories = store.list()
    ids = {m.id for m in memories}
    assert m1.id in ids
    assert m2.id not in ids


def test_list_includes_archived_when_requested(store):
    """list(include_archived=True) includes archived memories."""
    m1 = store.add("Active")
    m2 = store.add("Archived")
    store.forget(m2.id)

    memories = store.list(include_archived=True)
    ids = {m.id for m in memories}
    assert m1.id in ids
    assert m2.id in ids


def test_list_filter_by_category(store):
    """list() can filter by category."""
    store.add("Pattern memory", category="pattern")
    store.add("Decision memory", category="decision")

    patterns = store.list(category="pattern")
    assert all(m.category == MemoryCategory.PATTERN for m in patterns)
    assert len(patterns) == 1


def test_forget_archives_memory(store):
    """forget() archives a memory rather than deleting it."""
    memory = store.add("Temporary memory")
    result = store.forget(memory.id)
    assert result is True

    # Memory still exists but is archived
    retrieved = store.get(memory.id)
    assert retrieved is not None
    assert retrieved.lifecycle == MemoryLifecycle.ARCHIVED


def test_forget_unknown_id_returns_false(store):
    """forget() returns False for non-existent IDs."""
    result = store.forget("mem_doesnotexist")
    assert result is False


def test_keyword_search_finds_match(store):
    """search_keyword returns memories containing the query."""
    store.add("Use connection pooling with asyncpg for PostgreSQL")
    store.add("Use Redis for session caching")
    store.add("Always write unit tests")

    results = store.search_keyword("connection pooling")
    assert len(results) >= 1
    assert any("connection pooling" in m.content for m in results)


def test_keyword_search_excludes_archived(store):
    """search_keyword does not return archived memories."""
    m = store.add("This should be archived pooling content")
    store.forget(m.id)

    results = store.search_keyword("pooling")
    ids = {r.id for r in results}
    assert m.id not in ids


def test_stats_returns_correct_counts(store):
    """stats() returns accurate counts."""
    store.add("Pattern 1", category="pattern")
    store.add("Pattern 2", category="pattern")
    store.add("Decision 1", category="decision")
    m = store.add("To archive")
    store.forget(m.id)

    stats = store.stats()
    assert stats["total"] == 4
    assert stats["active"] == 3
    assert stats["archived"] == 1
    assert stats["by_category"]["pattern"] == 3
    assert stats["by_category"]["decision"] == 1


# ── Inbox removal (Finding 6) ────────────────────────────────────────────────
# Proves: inbox module deleted, store.add() stores directly to vault, no draft param.

import inspect


def test_inbox_module_not_importable():
    """inbox.py was removed — importing it must fail."""
    with pytest.raises(ModuleNotFoundError):
        import importlib
        importlib.import_module("lace.memory.inbox")


def test_store_add_has_no_draft_parameter():
    """store.add() signature has no 'draft' parameter."""
    sig = inspect.signature(MemoryStore.add)
    assert "draft" not in sig.parameters


def test_store_add_stores_directly_to_vault(store):
    """add() writes to vault immediately — no intermediate inbox/draft state."""
    memory = store.add("Direct to vault", tags=["test"])
    # Immediately retrievable from vault
    retrieved = store.get(memory.id)
    assert retrieved is not None
    assert retrieved.content == "Direct to vault"
    # File exists on disk
    assert Path(retrieved.file_path).exists()


def test_store_add_no_inbox_import_in_store_module():
    """store.py does not import anything from the inbox module."""
    import lace.memory.store as store_mod
    source = inspect.getsource(store_mod)
    assert "inbox" not in source.lower()