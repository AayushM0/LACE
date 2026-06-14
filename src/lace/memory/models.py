"""Core data models for LACE memory system."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ── Enums ─────────────────────────────────────────────────────────────────────

class MemorySource(str, Enum):
    CONVERSATION    = "conversation"
    USER_CORRECTION = "user_correction"
    MANUAL          = "manual"
    INGESTION       = "ingestion"
    MCP             = "mcp"
    AUTO_EXTRACTED  = "auto_extracted"


class MemoryCategory(str, Enum):
    PATTERN    = "pattern"
    DECISION   = "decision"
    DEBUG      = "debug"
    REFERENCE  = "reference"
    PREFERENCE = "preference"


class MemoryLifecycle(str, Enum):
    CAPTURED     = "captured"
    VALIDATED    = "validated"
    CONSOLIDATED = "consolidated"
    ARCHIVED     = "archived"


# ── MemoryObject ──────────────────────────────────────────────────────────────

@dataclass
class MemoryObject:
    """A single unit of persistent memory in LACE."""

    content: str
    category: MemoryCategory
    source: MemorySource                    = MemorySource.MANUAL
    project_scope: str                      = "global"
    id: str                                 = field(default_factory=lambda: f"mem_{uuid.uuid4().hex[:12]}")
    summary: str | None                     = None
    confidence: float                       = 0.8
    lifecycle: MemoryLifecycle              = MemoryLifecycle.CAPTURED
    created_at: datetime                    = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime                 = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int                       = 0
    embedding: list[float] | None          = None
    related_ids: list[str]                  = field(default_factory=list)
    tags: list[str]                         = field(default_factory=list)
    metadata: dict[str, Any]               = field(default_factory=dict)
    file_path: str | None                   = None

    def __post_init__(self) -> None:
        """Validate fields after init."""
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence must be between 0.0 and 1.0, got {self.confidence}")
        if not self.content.strip():
            raise ValueError("content cannot be empty")

    def touch(self) -> None:
        """Update access tracking — call every time this memory is retrieved."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1

    def archive(self) -> None:
        """Move to archived lifecycle."""
        self.lifecycle = MemoryLifecycle.ARCHIVED

    def validate(self) -> None:
        """Promote to validated lifecycle."""
        self.lifecycle = MemoryLifecycle.VALIDATED
        self.confidence = min(1.0, self.confidence + 0.1)

    def is_active(self) -> bool:
        """Return True if this memory should appear in normal search results."""
        return self.lifecycle != MemoryLifecycle.ARCHIVED

    def short_id(self) -> str:
        """Return last 8 chars of ID for display."""
        return self.id[-8:]

    def display_summary(self) -> str:
        """Return the summary if set, otherwise truncate content."""
        if self.summary:
            return self.summary
        return self.content[:80] + "..." if len(self.content) > 80 else self.content


# ── RetrievalResult ───────────────────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """A memory returned from a search, with its relevance score."""
    memory: MemoryObject
    relevance_score: float
    match_type: str   # "vector", "keyword", "graph", "hybrid"
    rank: int


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_memory(
    content: str,
    category: str | MemoryCategory = MemoryCategory.PATTERN,
    tags: list[str] | None = None,
    scope: str = "global",
    source: str | MemorySource = MemorySource.MANUAL,
    confidence: float = 0.8,
) -> MemoryObject:
    """Convenience constructor for creating a new MemoryObject."""
    return MemoryObject(
        content=content,
        category=MemoryCategory(category),
        tags=tags or [],
        project_scope=scope,
        source=MemorySource(source),
        confidence=confidence,
    )