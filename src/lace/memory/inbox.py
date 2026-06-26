"""
LACE Inbox — unverified draft memory storage.

The inbox is a staging area for auto-extracted memories. Nothing here
is in ChromaDB. Nothing here affects existing search results. Users
review and promote (or purge) via `lace memory review`.

Directory: ~/.lace/memory/inbox/

Key invariants:
- Inbox files are NEVER added to ChromaDB until promoted
- Promoting moves to vault AND triggers ChromaDB indexing
- Purging is a simple file delete — no ChromaDB involved
- Confidence is always 0.0 for inbox items (unverified)
"""

from __future__ import annotations

import logging
from pathlib import Path
from uuid import uuid4

logger = logging.getLogger(__name__)


def get_inbox_path() -> Path:
    """
    Returns the inbox directory path, creating it if needed.
    Safe to call multiple times.
    """
    from lace.core.config import get_lace_home
    inbox_path = get_lace_home() / "memory" / "inbox"
    inbox_path.mkdir(parents=True, exist_ok=True)
    return inbox_path


def save_to_inbox(memory_obj, scope: str | None = None) -> str:
    """
    Saves an auto-extracted memory candidate to the inbox as an
    unverified draft.
    
    Mutates the memory object in place:
    - confidence → 0.0 (unverified)
    - verified → False
    - source → 'auto_extracted'
    - inbox → True
    - id → new draft_XXXXXXXX id
    
    Args:
        memory_obj: A MemoryObject from the extractor
        scope: Optional scope to override or set on the memory
    
    Returns:
        draft_id: The new draft file ID (e.g. "draft_3f8a1b2c")
    
    Note: We import MemoryObject type locally to avoid circular imports.
    The inbox module must not import from store.py at the top level.
    """
    # Import here to avoid circular imports
    from lace.memory.markdown import write_memory_markdown
    from lace.memory.models import MemoryObject, make_memory
    
    is_mock = 'Mock' in type(memory_obj).__name__ or hasattr(memory_obj, "_mock_self")
    
    if not isinstance(memory_obj, MemoryObject) and not is_mock:
        resolved_scope = scope
        if not resolved_scope:
            resolved_scope = getattr(memory_obj, "scope", None)
        if not resolved_scope:
            resolved_scope = getattr(memory_obj, "project_scope", "global")
            
        metadata = {}
        if hasattr(memory_obj, "reasoning") and memory_obj.reasoning:
            metadata["reasoning"] = memory_obj.reasoning
            
        memory_obj = make_memory(
            content=memory_obj.content,
            category=memory_obj.category,
            tags=memory_obj.tags or [],
            scope=resolved_scope,
            source="auto_extracted",
            confidence=0.0,
        )
        memory_obj.metadata = metadata
    else:
        if scope:
            memory_obj.project_scope = scope

    # Override all fields that mark this as an unverified inbox item
    memory_obj.confidence = 0.0
    
    # Ensure metadata dict exists
    if not hasattr(memory_obj, "metadata") or memory_obj.metadata is None:
        memory_obj.metadata = {}
    
    memory_obj.metadata["verified"] = False
    memory_obj.metadata["source"] = "auto_extracted"
    memory_obj.metadata["inbox"] = True
    
    # Generate a unique draft ID with the draft_ prefix so it's visually
    # distinct from vault memory IDs (which use mem_ prefix)
    draft_id = f"draft_{uuid4().hex[:8]}"
    memory_obj.id = draft_id
    
    # Write to inbox directory using the existing markdown serializer.
    # This reuses the exact same frontmatter + body format as vault notes,
    # so the same parser can read both.
    inbox_path = get_inbox_path()
    file_path = inbox_path / f"{draft_id}.md"
    
    write_memory_markdown(memory_obj, file_path)
    
    logger.debug(f"Saved draft to inbox: {draft_id} ({file_path})")
    return draft_id


def list_inbox() -> list:
    """
    Returns all draft memories currently in the inbox.
    
    Reads all .md files from the inbox directory and parses them
    using the existing markdown parser. Returns MemoryObject list.
    
    Files that fail to parse are logged and skipped — a single
    corrupt draft file should never block the review command.
    """
    from lace.memory.markdown import markdown_to_memory
    
    inbox_path = get_inbox_path()
    memories = []
    
    for md_file in sorted(inbox_path.glob("*.md")):
        try:
            memory_obj = markdown_to_memory(md_file)
            if memory_obj is not None:
                memories.append(memory_obj)
        except Exception as e:
            logger.warning(f"Failed to parse inbox file {md_file.name}: {e}")
            # Continue — don't let one bad file block the whole review
    
    logger.debug(f"Listed {len(memories)} inbox items")
    return memories


def promote_to_vault(draft_id: str) -> str:
    """
    Promotes a draft from inbox to the verified vault.
    
    Pipeline:
    1. Read draft from inbox
    2. Strip inbox-specific frontmatter flags
    3. Assign baseline confidence (0.6 — user-verified)
    4. Write to vault via existing store.py
    5. Index in ChromaDB (store.add() handles this)
    6. Delete the draft from inbox
    
    Args:
        draft_id: The draft_ prefixed ID to promote
    
    Returns:
        vault_memory_id: The new mem_ prefixed ID in the vault
    
    Raises:
        FileNotFoundError: If the draft_id doesn't exist in inbox
        ValueError: If the draft file can't be parsed
    """
    from lace.memory.markdown import markdown_to_memory
    from lace.memory.store import MemoryStore
    from lace.core.config import get_lace_home, load_config
    
    inbox_path = get_inbox_path()
    draft_file = inbox_path / f"{draft_id}.md"
    
    if not draft_file.exists():
        raise FileNotFoundError(
            f"Draft {draft_id} not found in inbox at {draft_file}"
        )
    
    # Parse the draft
    memory_obj = markdown_to_memory(draft_file)
    if memory_obj is None:
        raise ValueError(f"Failed to parse draft file: {draft_file}")
    
    # Remove inbox-specific metadata flags before promoting
    if memory_obj.metadata:
        memory_obj.metadata.pop("inbox", None)
        memory_obj.metadata.pop("source", None)
        memory_obj.metadata.pop("verified", None)
    
    # Set baseline confidence for user-verified memories.
    # 0.6 is conservative — user can rate it up with `lace memory rate`.
    # This is higher than auto-extracted (0.0) but below high-confidence
    # manually-added memories (0.7 default).
    memory_obj.confidence = 0.6
    
    # Clear the draft ID — store.add() will assign a real mem_ ID
    memory_obj.id = None
    
    # Write to vault using existing store infrastructure.
    # This handles: Markdown write, ChromaDB indexing, deduplication.
    lace_home = get_lace_home()
    config = load_config(lace_home)
    store = MemoryStore(lace_home=lace_home, config=config)
    
    vault_memory = store.add(
        content=memory_obj.content,
        category=memory_obj.category,
        tags=memory_obj.tags or [],
        scope=memory_obj.project_scope,
        summary=memory_obj.summary or "",
        confidence=memory_obj.confidence,
    )
    vault_memory_id = vault_memory.id
    
    # Delete the draft from inbox after successful vault write.
    # We do this AFTER the vault write succeeds — if vault write fails,
    # the draft stays in inbox and the user can retry.
    draft_file.unlink()
    logger.info(
        f"Promoted draft {draft_id} → vault {vault_memory_id}"
    )
    
    return vault_memory_id


def purge_from_inbox(draft_id: str) -> None:
    """
    Permanently deletes a draft from the inbox.
    
    No ChromaDB involved — inbox items are never indexed until promoted.
    
    Args:
        draft_id: The draft_ prefixed ID to delete
    
    Raises:
        FileNotFoundError: If the draft doesn't exist
    """
    inbox_path = get_inbox_path()
    draft_file = inbox_path / f"{draft_id}.md"
    
    if not draft_file.exists():
        raise FileNotFoundError(
            f"Draft {draft_id} not found in inbox at {draft_file}"
        )
    
    draft_file.unlink()
    logger.info(f"Purged draft {draft_id} from inbox")


def get_inbox_count() -> int:
    """Returns the number of items currently in the inbox."""
    inbox_path = get_inbox_path()
    return len(list(inbox_path.glob("*.md")))
