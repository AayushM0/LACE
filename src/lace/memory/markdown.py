"""Read and write memory objects as markdown files with YAML frontmatter."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import frontmatter  # python-frontmatter

from lace.memory.models import (
    MemoryCategory,
    MemoryLifecycle,
    MemoryObject,
    MemorySource,
)


# ── Constants ─────────────────────────────────────────────────────────────────

DATE_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


# ── Serialization helpers ─────────────────────────────────────────────────────

def _dt_to_str(dt: datetime) -> str:
    """Convert datetime to ISO string for frontmatter."""
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt.strftime(DATE_FORMAT)


def _str_to_dt(s: str | None) -> datetime:
    """Parse ISO string from frontmatter to datetime."""
    if not s:
        return datetime.now(timezone.utc)
    try:
        dt = datetime.strptime(str(s), DATE_FORMAT)
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        # Fallback for other formats
        return datetime.now(timezone.utc)


# ── Write ─────────────────────────────────────────────────────────────────────

def memory_to_markdown(memory: MemoryObject) -> str:
    """Serialize a MemoryObject to a markdown string with YAML frontmatter."""
    post = frontmatter.Post(
        content=memory.content,
        **{
            "id":            memory.id,
            "category":      memory.category.value,
            "source":        memory.source.value,
            "lifecycle":     memory.lifecycle.value,
            "confidence":    round(memory.confidence, 4),
            "project_scope": memory.project_scope,
            "scope": memory.project_scope,
            "tags":          memory.tags,
            "created_at":    _dt_to_str(memory.created_at),
            "last_accessed": _dt_to_str(memory.last_accessed),
            "access_count":  memory.access_count,
            "related_ids":   memory.related_ids,
            **({"summary": memory.summary} if memory.summary else {}),
            **(memory.metadata if memory.metadata else {}),
        },
    )
    return frontmatter.dumps(post)


def save_memory_to_file(memory: MemoryObject, vault_path: Path) -> Path:
    """Write a MemoryObject to its markdown file in the vault.

    File location is determined by:
      - global memories  → vault/global/<category>/<id>.md
      - project memories → vault/projects/<project>/<category>/<id>.md

    Returns the path where the file was written.
    Updates memory.file_path in place.
    """
    if memory.project_scope == "global":
        target_dir = vault_path / "global" / memory.category.value
    else:
        # "project:my-api" → "my-api"
        project_name = memory.project_scope.removeprefix("project:")
        target_dir = vault_path / "projects" / project_name / memory.category.value

    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / f"{memory.id}.md"

    file_path.write_text(memory_to_markdown(memory), encoding="utf-8")
    memory.file_path = str(file_path)

    return file_path


def write_memory_markdown(memory: MemoryObject, file_path: Path) -> None:
    """Write a MemoryObject as a markdown string with frontmatter directly to a file."""
    file_path.write_text(memory_to_markdown(memory), encoding="utf-8")
    memory.file_path = str(file_path)


# ── Read ──────────────────────────────────────────────────────────────────────

def markdown_to_memory(file_path: Path) -> MemoryObject | None:
    """Parse a markdown file into a MemoryObject.

    Returns None if the file is not a valid memory file.
    """
    try:
        post = frontmatter.load(str(file_path))
    except Exception:
        return None

    meta = post.metadata

    # Must have an id to be a valid memory file
    if "id" not in meta:
        return None

    try:
        memory = MemoryObject(
            id=meta["id"],
            content=post.content.strip(),
            category=MemoryCategory(meta.get("category", "pattern")),
            source=MemorySource(meta.get("source", "manual")),
            lifecycle=MemoryLifecycle(meta.get("lifecycle", "captured")),
            confidence=float(meta.get("confidence", 0.8)),
            project_scope=(
                meta.get("project_scope")
                or meta.get("scope")
                or "global"
            ),
            tags=list(meta.get("tags", [])),
            created_at=_str_to_dt(meta.get("created_at")),
            last_accessed=_str_to_dt(meta.get("last_accessed")),
            access_count=int(meta.get("access_count", 0)),
            related_ids=list(meta.get("related_ids", [])),
            summary=meta.get("summary"),
            file_path=str(file_path),
            metadata={
                k: v for k, v in meta.items()
                if k not in {
                    "id", "category", "source", "lifecycle", "confidence",
                    "project_scope", "scope", "tags", "created_at", "last_accessed",
                    "access_count", "related_ids", "summary",
                }
            },
        )
        return memory
    except (ValueError, KeyError):
        return None


def load_all_memories(vault_path: Path) -> list[MemoryObject]:
    """Load every memory file from the vault directory.

    Walks the entire vault recursively and parses every .md file.
    Skips files that are not valid memory files (no 'id' in frontmatter).
    """
    memories: list[MemoryObject] = []

    for md_file in sorted(vault_path.rglob("*.md")):
        memory = markdown_to_memory(md_file)
        if memory is not None:
            memories.append(memory)

    return memories