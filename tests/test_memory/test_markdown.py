"""Tests for markdown serialization."""

import pytest
from pathlib import Path
from lace.memory.models import make_memory, MemoryCategory
from lace.memory.markdown import (
    memory_to_markdown,
    save_memory_to_file,
    markdown_to_memory,
    load_all_memories,
)


def test_roundtrip_global_memory(tmp_path):
    """Memory can be written and read back identically."""
    memory = make_memory(
        "Use connection pooling with asyncpg",
        category=MemoryCategory.PATTERN,
        tags=["db", "performance"],
    )

    saved_path = save_memory_to_file(memory, tmp_path)
    assert saved_path.exists()

    loaded = markdown_to_memory(saved_path)
    assert loaded is not None
    assert loaded.id == memory.id
    assert loaded.content == memory.content
    assert loaded.category == memory.category
    assert loaded.tags == memory.tags
    assert loaded.project_scope == "global"


def test_global_memory_saved_in_correct_location(tmp_path):
    """Global pattern memory goes to vault/global/pattern/."""
    memory = make_memory("Test pattern", category=MemoryCategory.PATTERN)
    path = save_memory_to_file(memory, tmp_path)
    assert "global/pattern" in path.as_posix()
    assert path.name == f"{memory.id}.md"


def test_project_memory_saved_in_correct_location(tmp_path):
    """Project memory goes to vault/projects/<name>/."""
    memory = make_memory("Test", scope="project:my-api")
    path = save_memory_to_file(memory, tmp_path)
    assert "projects/my-api" in path.as_posix()



def test_markdown_file_has_frontmatter(tmp_path):
    """Saved file has YAML frontmatter with required fields."""
    memory = make_memory("Test content", tags=["test"])
    path = save_memory_to_file(memory, tmp_path)
    raw = path.read_text()

    assert raw.startswith("---")
    assert "id:" in raw
    assert "category:" in raw
    assert "confidence:" in raw
    assert "tags:" in raw


def test_invalid_file_returns_none(tmp_path):
    """Files without 'id' in frontmatter return None."""
    bad_file = tmp_path / "bad.md"
    bad_file.write_text("# Just a regular markdown file\n\nNo frontmatter.")
    result = markdown_to_memory(bad_file)
    assert result is None


def test_load_all_memories(tmp_path):
    """load_all_memories returns all valid memory files."""
    m1 = make_memory("Memory 1", category=MemoryCategory.PATTERN)
    m2 = make_memory("Memory 2", category=MemoryCategory.DECISION)
    m3 = make_memory("Memory 3", scope="project:test")

    save_memory_to_file(m1, tmp_path)
    save_memory_to_file(m2, tmp_path)
    save_memory_to_file(m3, tmp_path)

    # Add a non-memory file that should be ignored
    (tmp_path / "random.md").write_text("# Not a memory")

    memories = load_all_memories(tmp_path)
    assert len(memories) == 3
    ids = {m.id for m in memories}
    assert m1.id in ids
    assert m2.id in ids
    assert m3.id in ids