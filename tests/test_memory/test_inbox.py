"""Tests for LACE Inbox system."""

import pytest
from pathlib import Path
from lace.core.config import LaceConfig, init_lace_home
from lace.memory.models import make_memory
from lace.memory.store import MemoryStore
from lace.memory.inbox import (
    get_inbox_path,
    save_to_inbox,
    list_inbox,
    promote_to_vault,
    purge_from_inbox,
    get_inbox_count,
)

@pytest.fixture
def inbox_env(tmp_path, monkeypatch):
    """Set up LACE environment with a temporary directory."""
    lace_home = tmp_path / ".lace"
    init_lace_home(lace_home)
    monkeypatch.setenv("LACE_HOME", str(lace_home))
    return lace_home

def test_save_to_inbox_mutates_and_writes(inbox_env):
    """Verify save_to_inbox sets metadata, ID prefix, confidence=0.0, and saves file."""
    memory_obj = make_memory(
        content="This is a test candidate extracted from LLM.",
        category="decision",
        tags=["pytest", "test"],
        scope="global",
        confidence=0.9,
    )
    
    draft_id = save_to_inbox(memory_obj)
    
    assert draft_id.startswith("draft_")
    assert memory_obj.id == draft_id
    assert memory_obj.confidence == 0.0
    assert memory_obj.metadata["verified"] is False
    assert memory_obj.metadata["source"] == "auto_extracted"
    assert memory_obj.metadata["inbox"] is True
    
    # Check that file exists in inbox
    inbox_dir = get_inbox_path()
    file_path = inbox_dir / f"{draft_id}.md"
    assert file_path.exists()

def test_list_inbox(inbox_env):
    """Verify list_inbox and get_inbox_count correctly return drafts."""
    assert get_inbox_count() == 0
    assert list_inbox() == []
    
    m1 = make_memory("Candidate 1 content of sufficient length", category="pattern")
    m2 = make_memory("Candidate 2 content of sufficient length", category="decision")
    
    save_to_inbox(m1)
    save_to_inbox(m2)
    
    assert get_inbox_count() == 2
    drafts = list_inbox()
    assert len(drafts) == 2
    
    draft_ids = {d.id for d in drafts}
    assert m1.id in draft_ids
    assert m2.id in draft_ids

def test_promote_to_vault(inbox_env):
    """Verify promote_to_vault moves draft to vault, changes confidence/metadata, and deletes draft."""
    memory_obj = make_memory(
        content="Candidate content of sufficient length for memory.",
        category="pattern",
        tags=["tag"],
        scope="global",
    )
    
    draft_id = save_to_inbox(memory_obj)
    
    # Promote it
    vault_id = promote_to_vault(draft_id)
    
    assert vault_id.startswith("mem_")
    # Original draft file should be gone
    assert not (get_inbox_path() / f"{draft_id}.md").exists()
    assert get_inbox_count() == 0
    
    # Verify it was saved in the vault with correct properties
    store = MemoryStore()
    vault_memory = store.get(vault_id)
    assert vault_memory is not None
    assert vault_memory.content == "Candidate content of sufficient length for memory."
    assert vault_memory.confidence == 0.6
    assert "inbox" not in vault_memory.metadata
    assert "source" not in vault_memory.metadata
    assert "verified" not in vault_memory.metadata

def test_purge_from_inbox(inbox_env):
    """Verify purge_from_inbox unlinks the draft."""
    memory_obj = make_memory("Content to delete", category="pattern")
    draft_id = save_to_inbox(memory_obj)
    
    assert get_inbox_count() == 1
    purge_from_inbox(draft_id)
    assert get_inbox_count() == 0
    assert not (get_inbox_path() / f"{draft_id}.md").exists()

def test_store_add_draft_routing(inbox_env):
    """Verify that store.add(..., draft=True) correctly routes to inbox."""
    store = MemoryStore()
    
    # Add with draft=True
    memory = store.add(
        content="This is a candidate to route to inbox.",
        category="pattern",
        tags=["pytest"],
        scope="global",
        draft=True,
    )
    
    # Should return a MemoryObject with a draft_ ID
    assert memory.id.startswith("draft_")
    assert get_inbox_count() == 1
    
    # The memory should NOT be present in normal store search or vault files
    assert store.get(memory.id) is None
