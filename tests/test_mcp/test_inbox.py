"""
Tests for the inbox system.

Key behaviors to verify:
- Inbox items are isolated from vault (no ChromaDB)
- Promote correctly transfers to vault and deletes draft
- Purge correctly deletes without touching ChromaDB
- list_inbox handles corrupt files gracefully
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


@pytest.fixture
def temp_inbox(tmp_path, monkeypatch):
    """Redirects inbox to a temp directory for test isolation."""
    lace_home = tmp_path / ".lace"
    from lace.core.config import init_lace_home
    init_lace_home(lace_home)
    monkeypatch.setenv("LACE_HOME", str(lace_home))
    
    from lace.memory.inbox import get_inbox_path
    return get_inbox_path()


@pytest.fixture
def mock_memory_object():
    """Creates a minimal mock MemoryObject for testing."""
    obj = MagicMock()
    obj.id = None
    obj.content = "We decided to use SQLite for the queue because it's simple."
    obj.summary = "SQLite queue decision"
    obj.category = "decision"
    obj.tags = ["sqlite", "queue"]
    obj.scope = "project:lace"
    obj.confidence = 0.75
    obj.metadata = {}
    return obj


class TestSaveToInbox:
    def test_generates_draft_id(self, temp_inbox, mock_memory_object):
        """save_to_inbox assigns a draft_ prefixed ID."""
        with patch("lace.memory.markdown.write_memory_markdown"):
            from lace.memory.inbox import save_to_inbox
            draft_id = save_to_inbox(mock_memory_object)
        
        assert draft_id.startswith("draft_")
        assert mock_memory_object.id == draft_id
    
    def test_sets_confidence_to_zero(self, temp_inbox, mock_memory_object):
        """Inbox items always have confidence=0.0 (unverified)."""
        with patch("lace.memory.markdown.write_memory_markdown"):
            from lace.memory.inbox import save_to_inbox
            save_to_inbox(mock_memory_object)
        
        assert mock_memory_object.confidence == 0.0
    
    def test_sets_inbox_metadata_flags(self, temp_inbox, mock_memory_object):
        """Inbox items have verified=False, source=auto_extracted, inbox=True."""
        with patch("lace.memory.markdown.write_memory_markdown"):
            from lace.memory.inbox import save_to_inbox
            save_to_inbox(mock_memory_object)
        
        assert mock_memory_object.metadata["verified"] is False
        assert mock_memory_object.metadata["source"] == "auto_extracted"
        assert mock_memory_object.metadata["inbox"] is True
    
    def test_writes_to_inbox_directory(self, temp_inbox, mock_memory_object):
        """save_to_inbox writes the file to the inbox directory."""
        written_paths = []
        
        def capture_write(obj, path):
            written_paths.append(path)
        
        with patch("lace.memory.markdown.write_memory_markdown", side_effect=capture_write):
            from lace.memory.inbox import save_to_inbox
            draft_id = save_to_inbox(mock_memory_object)
        
        assert len(written_paths) == 1
        assert written_paths[0].parent == temp_inbox
        assert written_paths[0].name == f"{draft_id}.md"
    
    def test_unique_draft_ids(self, temp_inbox):
        """Multiple calls generate unique IDs."""
        from lace.memory.inbox import save_to_inbox
        
        ids = set()
        for _ in range(10):
            obj = MagicMock()
            obj.metadata = {}
            with patch("lace.memory.markdown.write_memory_markdown"):
                draft_id = save_to_inbox(obj)
            ids.add(draft_id)
        
        assert len(ids) == 10


class TestListInbox:
    def test_empty_inbox_returns_empty_list(self, temp_inbox):
        from lace.memory.inbox import list_inbox
        assert list_inbox() == []
    
    def test_returns_parsed_memories(self, temp_inbox):
        """list_inbox parses all .md files and returns MemoryObjects."""
        # Create fake .md files in the inbox
        (temp_inbox / "draft_aaaaaaaa.md").write_text("# Draft 1")
        (temp_inbox / "draft_bbbbbbbb.md").write_text("# Draft 2")
        
        mock_obj = MagicMock()
        
        with patch("lace.memory.markdown.markdown_to_memory", return_value=mock_obj):
            from lace.memory.inbox import list_inbox
            results = list_inbox()
        
        assert len(results) == 2
    
    def test_skips_corrupt_files_gracefully(self, temp_inbox):
        """A corrupt file doesn't crash list_inbox — it's logged and skipped."""
        (temp_inbox / "draft_good.md").write_text("# Good")
        (temp_inbox / "draft_bad.md").write_text("# Bad")
        
        good_obj = MagicMock()
        
        def mock_parse(path):
            if "bad" in str(path):
                raise ValueError("Corrupt file")
            return good_obj
        
        with patch("lace.memory.markdown.markdown_to_memory", side_effect=mock_parse):
            from lace.memory.inbox import list_inbox
            results = list_inbox()
        
        # Should return only the good file, not crash
        assert len(results) == 1
        assert results[0] == good_obj
    
    def test_ignores_non_md_files(self, temp_inbox):
        """Non-.md files in the inbox directory are ignored."""
        (temp_inbox / "draft_good.md").write_text("# Good")
        (temp_inbox / "README.txt").write_text("Not a draft")
        (temp_inbox / ".hidden").write_text("Hidden")
        
        mock_obj = MagicMock()
        
        with patch("lace.memory.markdown.markdown_to_memory", return_value=mock_obj):
            from lace.memory.inbox import list_inbox
            results = list_inbox()
        
        assert len(results) == 1


class TestPromoteToVault:
    def test_raises_if_draft_not_found(self, temp_inbox):
        from lace.memory.inbox import promote_to_vault
        
        with pytest.raises(FileNotFoundError, match="draft_nonexistent"):
            promote_to_vault("draft_nonexistent")
    
    def test_sets_baseline_confidence(self, temp_inbox):
        """Promoted memories get confidence=0.6 (user-verified baseline)."""
        # Create a draft file
        (temp_inbox / "draft_test1234.md").write_text("# Draft")
        
        mock_obj = MagicMock()
        mock_obj.metadata = {"inbox": True, "source": "auto_extracted", "verified": False}
        mock_obj.id = None
        mock_obj.content = "test content"
        mock_obj.category = "decision"
        mock_obj.tags = []
        mock_obj.scope = "global"
        mock_obj.summary = "test"
        
        mock_store = MagicMock()
        mock_store.add.return_value = MagicMock(id="mem_abc123")
        
        with patch("lace.memory.markdown.markdown_to_memory", return_value=mock_obj):
            with patch("lace.memory.store.MemoryStore", return_value=mock_store):
                from lace.memory.inbox import promote_to_vault
                vault_id = promote_to_vault("draft_test1234")
        
        # Confidence should be set to 0.6 baseline
        assert mock_obj.confidence == 0.6
        assert vault_id == "mem_abc123"
    
    def test_removes_inbox_metadata_flags(self, temp_inbox):
        """Inbox flags are stripped before promoting to vault."""
        (temp_inbox / "draft_cleanme.md").write_text("# Draft")
        
        mock_obj = MagicMock()
        mock_obj.metadata = {
            "inbox": True,
            "source": "auto_extracted",
            "verified": False,
            "keep_this": "preserved",
        }
        mock_obj.id = None
        mock_obj.content = "content"
        mock_obj.category = "decision"
        mock_obj.tags = []
        mock_obj.scope = "global"
        mock_obj.summary = "test"
        
        mock_store = MagicMock()
        mock_store.add.return_value = MagicMock(id="mem_xyz789")
        
        with patch("lace.memory.markdown.markdown_to_memory", return_value=mock_obj):
            with patch("lace.memory.store.MemoryStore", return_value=mock_store):
                from lace.memory.inbox import promote_to_vault
                promote_to_vault("draft_cleanme")
        
        # Inbox-specific flags should be gone
        assert "inbox" not in mock_obj.metadata
        assert "source" not in mock_obj.metadata
        assert "verified" not in mock_obj.metadata
        # Other metadata should be preserved
        assert mock_obj.metadata.get("keep_this") == "preserved"
    
    def test_deletes_draft_after_successful_promotion(self, temp_inbox):
        """Draft file is deleted from inbox after vault write succeeds."""
        draft_file = temp_inbox / "draft_deleteme.md"
        draft_file.write_text("# Draft")
        assert draft_file.exists()
        
        mock_obj = MagicMock()
        mock_obj.metadata = {}
        mock_obj.id = None
        mock_obj.content = "content"
        mock_obj.category = "decision"
        mock_obj.tags = []
        mock_obj.scope = "global"
        mock_obj.summary = "test"
        
        mock_store = MagicMock()
        mock_store.add.return_value = MagicMock(id="mem_done")
        
        with patch("lace.memory.markdown.markdown_to_memory", return_value=mock_obj):
            with patch("lace.memory.store.MemoryStore", return_value=mock_store):
                from lace.memory.inbox import promote_to_vault
                promote_to_vault("draft_deleteme")
        
        # Draft should be gone from inbox
        assert not draft_file.exists()
    
    def test_draft_preserved_if_vault_write_fails(self, temp_inbox):
        """If vault write fails, the draft stays in inbox for retry."""
        draft_file = temp_inbox / "draft_keepme.md"
        draft_file.write_text("# Draft")
        
        mock_obj = MagicMock()
        mock_obj.metadata = {}
        mock_obj.id = None
        mock_obj.content = "content"
        mock_obj.category = "decision"
        mock_obj.tags = []
        mock_obj.scope = "global"
        mock_obj.summary = "test"
        
        mock_store = MagicMock()
        mock_store.add.side_effect = RuntimeError("ChromaDB unavailable")
        
        with patch("lace.memory.markdown.markdown_to_memory", return_value=mock_obj):
            with patch("lace.memory.store.MemoryStore", return_value=mock_store):
                from lace.memory.inbox import promote_to_vault
                with pytest.raises(RuntimeError, match="ChromaDB unavailable"):
                    promote_to_vault("draft_keepme")
        
        # Draft should still exist — not lost on failure
        assert draft_file.exists()


class TestPurgeFromInbox:
    def test_deletes_draft_file(self, temp_inbox):
        draft_file = temp_inbox / "draft_purgeme.md"
        draft_file.write_text("# Draft")
        assert draft_file.exists()
        
        from lace.memory.inbox import purge_from_inbox
        purge_from_inbox("draft_purgeme")
        
        assert not draft_file.exists()
    
    def test_raises_if_not_found(self, temp_inbox):
        from lace.memory.inbox import purge_from_inbox
        with pytest.raises(FileNotFoundError):
            purge_from_inbox("draft_doesnotexist")
    
    def test_does_not_touch_chromadb(self, temp_inbox):
        """Purge is file-only — ChromaDB is never involved."""
        draft_file = temp_inbox / "draft_nochroma.md"
        draft_file.write_text("# Draft")
        
        with patch("lace.retrieval.vector.get_client") as mock_chroma:
            from lace.memory.inbox import purge_from_inbox
            purge_from_inbox("draft_nochroma")
        
        # ChromaDB client was never opened
        mock_chroma.assert_not_called()


class TestGetInboxCount:
    def test_empty_inbox(self, temp_inbox):
        from lace.memory.inbox import get_inbox_count
        assert get_inbox_count() == 0
    
    def test_counts_md_files_only(self, temp_inbox):
        (temp_inbox / "draft_1.md").write_text("# 1")
        (temp_inbox / "draft_2.md").write_text("# 2")
        (temp_inbox / "README.txt").write_text("Not a draft")
        
        from lace.memory.inbox import get_inbox_count
        assert get_inbox_count() == 2
