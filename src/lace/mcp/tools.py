"""MCP tool implementations — the actual functions exposed to AI agents."""

from __future__ import annotations
import sys
import asyncio
from datetime import datetime, timezone

from lace.core.config import get_lace_home, load_config
from lace.core.scope import get_active_scope, get_projects
from lace.memory.models import MemoryCategory
from lace.memory.store import MemoryStore

def _debug_log(msg: str) -> None:
    """Debug logging to stderr (MCP uses stdout for JSON-RPC)."""
    print(f"[LACE DEBUG] {msg}", file=sys.stderr, flush=True)


# ── Context state (set by set_context tool) ───────────────────────────────────

_mcp_context_cwd: str | None = None
_mcp_context_project: str | None = None


# ── Store factory ─────────────────────────────────────────────────────────────

def _get_store(scope: str | None = None) -> tuple[MemoryStore, str]:
    """Return a configured MemoryStore and resolved scope."""
    lace_home = get_lace_home()
    config = load_config(lace_home)
    store = MemoryStore(lace_home=lace_home, config=config)

    if scope is None or scope == "auto":
        # Use MCP context if set, otherwise default to global
        if _mcp_context_project:
            resolved_scope = _mcp_context_project
        else:
            resolved_scope = "global"
    else:
        resolved_scope = scope

    return store, resolved_scope


def _multi_scope_search(
    store: MemoryStore,
    query: str,
    primary_scope: str,
    max_results: int,
) -> list:
    """Search across multiple scopes intelligently."""
    all_results = []
    
    if primary_scope.startswith("session:"):
        lace_home = get_lace_home()
        projects = get_projects(lace_home)
        
        for project in projects:
            project_results = store.search(
                query=query,
                scope=project["scope"],
                max_results=max_results,
            )
            all_results.extend(project_results)
        
        global_results = store.search(
            query=query,
            scope="global",
            max_results=max_results,
        )
        all_results.extend(global_results)
    
    elif primary_scope.startswith("project:"):
        primary_results = store.search(
            query=query,
            scope=primary_scope,
            max_results=max_results,
        )
        all_results.extend(primary_results)
        
        global_results = store.search(
            query=query,
            scope="global",
            max_results=max_results,
        )
        all_results.extend(global_results)
    
    else:
        global_results = store.search(
            query=query,
            scope="global",
            max_results=max_results,
        )
        all_results.extend(global_results)
    
    # Deduplicate
    seen = set()
    unique_results = []
    for result in all_results:
        if result.memory.id not in seen:
            seen.add(result.memory.id)
            unique_results.append(result)
    
    unique_results.sort(key=lambda r: r.relevance_score, reverse=True)
    
    return unique_results[:max_results]

def _populate_embeddings(
    memories: list,
    lace_home,
) -> None:
    """Load embeddings from ChromaDB into memory objects in-place.
    
    Memories loaded from markdown files don't have embeddings.
    This fetches them from ChromaDB so dedup can compare them.
    """
    from lace.retrieval.vector import get_client, _scope_to_collection_name
    from pathlib import Path

    vector_db_path = Path(lace_home) / "memory" / "vector_db"
    
    try:
        client = get_client(vector_db_path)
    except Exception:
        return

    # Group memories by scope to minimize ChromaDB calls
    by_scope: dict[str, list] = {}
    for m in memories:
        by_scope.setdefault(m.project_scope, []).append(m)

    for scope, scope_memories in by_scope.items():
        try:
            collection_name = _scope_to_collection_name(scope)
            collection = client.get_collection(collection_name)
            
            ids = [m.id for m in scope_memories]
            result = collection.get(
                ids=ids,
                include=["embeddings"],
            )
            
            if not result or not result.get("ids"):
                continue

            # Check if embeddings exist (don't use if directly on array)
            embeddings = result.get("embeddings")
            if embeddings is None or len(embeddings) == 0:  # ← FIX
                continue

            # Map id → embedding
            embedding_map = {}
            for i, mem_id in enumerate(result["ids"]):
                if i < len(embeddings):
                    embedding_map[mem_id] = embeddings[i]

            # Assign back to memory objects
            for m in scope_memories:
                if m.id in embedding_map:
                    m.embedding = embedding_map[m.id]

        except Exception as e:
            import sys
            print(f"[LACE] _populate_embeddings error for scope {scope}: {e}", file=sys.stderr)
            continue


# ── Tool implementations ──────────────────────────────────────────────────────

async def set_context(
    working_directory: str,
    project_name: str | None = None,
    **kwargs,
) -> dict:
    """Set the MCP server's working directory context."""
    global _mcp_context_cwd, _mcp_context_project
    from lace.core.scope import detect_current_project

    _mcp_context_cwd = working_directory

    if project_name:
        _mcp_context_project = f"project:{project_name}"
    else:
        # Auto-detect from the provided cwd
        _mcp_context_project = detect_current_project(cwd=working_directory)

    return {
        "context_set": True,
        "working_directory": _mcp_context_cwd,
        "detected_scope": _mcp_context_project or "global",
    }


async def search_memory(
    query: str,
    scope: str = "auto",
    max_results: int = 5,
    category: str = "all",
    **kwargs
) -> list[dict]:
    """Search your knowledge base for memories relevant to a query."""
    import time
    t_start = time.monotonic()

    store, resolved_scope = _get_store(scope)

    # ── Determine which scopes to search ────────────────────────────────────
    # When scope is "auto" and a project context is set, search both:
    #   1. global (always)
    #   2. the active project (if set)
    # This allows cross-project knowledge while maintaining isolation.
    
    scopes_to_search = []
    
    if scope == "auto":
        scopes_to_search.append("global")
        if _mcp_context_project and _mcp_context_project != "global":
            scopes_to_search.append(_mcp_context_project)
    else:
        # Explicit scope provided
        scopes_to_search.append(resolved_scope)
    
    # ── Search across all target scopes ─────────────────────────────────────
    all_results = []
    seen_ids = set()
    
    for search_scope in scopes_to_search:
        scope_results = _multi_scope_search(
            store=store,
            query=query,
            primary_scope=search_scope,
            max_results=max_results * 2,  # Get more candidates for merging
        )
        
        # Deduplicate across scopes
        for r in scope_results:
            if r.memory.id not in seen_ids:
                seen_ids.add(r.memory.id)
                all_results.append(r)
    
    # ── Re-rank combined results ────────────────────────────────────────────
    # Sort by relevance score (already computed by _multi_scope_search)
    all_results.sort(key=lambda r: r.relevance_score, reverse=True)
    results = all_results[:max_results]

    # ── Apply category filter ───────────────────────────────────────────────
    if category != "all":
        try:
            cat = MemoryCategory(category)
            results = [r for r in results if r.memory.category == cat]
        except ValueError:
            pass

    # ── Record access for each retrieved memory ─────────────────────────────
    for r in results:
        try:
            store.record_access(r.memory.id)
        except Exception:
            pass

    # ── Log retrieval ───────────────────────────────────────────────────────
    try:
        latency_ms = (time.monotonic() - t_start) * 1000
        lace_home = get_lace_home()
        from lace.utils.logging import RetrievalLogger
        logger = RetrievalLogger(lace_home)
        
        # Log the primary scope that was searched
        log_scope = resolved_scope if scope != "auto" else f"auto({','.join(scopes_to_search)})"
        
        logger.log_retrieval(
            query=query,
            scope=log_scope,
            results=results,
            latency_ms=latency_ms,
        )
    except Exception:
        pass  # Logging must never break search

    return [
        {
            "id": r.memory.id,
            "content": r.memory.content,
            "summary": r.memory.display_summary(),
            "category": r.memory.category.value,
            "tags": r.memory.tags,
            "scope": r.memory.project_scope,
            "confidence": r.memory.confidence,
            "relevance_score": round(r.relevance_score, 3),
            "created_at": r.memory.created_at.isoformat(),
            "access_count": r.memory.access_count,
        }
        for r in results
    ]


async def get_project_context(
    project_name: str | None = None,
    **kwargs
) -> dict:
    """Get memories and metadata for a specific project."""
    store, active_scope = _get_store()

    if project_name:
        scope = f"project:{project_name}"
    else:
        scope = active_scope if active_scope.startswith("project:") else "global"

    memories = store.list(scope=scope, limit=50, include_archived=False)
    patterns = [m for m in memories if m.category == MemoryCategory.PATTERN][:10]
    decisions = [m for m in memories if m.category == MemoryCategory.DECISION][:10]

    # Load identity and preferences
    from lace.core.identity import compose_identity
    from lace.core.config import get_lace_home
    lace_home = get_lace_home()
    identity_text, preferences = compose_identity(lace_home, scope=scope)

    return {
        "scope": scope,
        "total_memories": len(memories),
        "identity": identity_text or "",
        "preferences": preferences or {},
        "patterns": [
            {"id": m.id, "summary": m.display_summary(), "tags": m.tags, "confidence": m.confidence}
            for m in patterns
        ],
        "decisions": [
            {"id": m.id, "summary": m.display_summary(), "tags": m.tags, "confidence": m.confidence}
            for m in decisions
        ],
    }


async def remember(
    content: str,
    category: str = "pattern",
    tags: list[str] | None = None,
    confidence: float = 0.7,
    scope: str = "auto",
    **kwargs
) -> dict:
    """Store a new memory from this interaction."""
    store, resolved_scope = _get_store(scope)

    # Sessions are ephemeral — store in global instead
    if resolved_scope.startswith("session:"):
        resolved_scope = "global"

    try:
        cat = MemoryCategory(category)
    except ValueError:
        cat = MemoryCategory.PATTERN

    # ── Dedup check before storing ─────────────────────────────────────────
    try:
        from lace.memory.dedup import check_duplicate, DedupAction
        from lace.memory.models import make_memory
        from lace.retrieval.embeddings import embed_text
        from lace.retrieval.vector import get_collection

        candidate = make_memory(
            content=content,
            category=cat.value,
            tags=tags or [],
            scope=resolved_scope,
            source="mcp",
            confidence=max(0.0, min(1.0, confidence)),
        )
        candidate.embedding = embed_text(content)

        # Load all memories and populate their embeddings from ChromaDB
        lace_home = get_lace_home()
        existing_memories = store.list(include_archived=False, limit=500)
        
        # Populate embeddings from ChromaDB for each memory
        _populate_embeddings(existing_memories, lace_home)

        dedup = check_duplicate(candidate, existing_memories)

        if dedup.action == DedupAction.SKIP:
            return {
                "stored": False,
                "status": "duplicate",
                "reason": "Similar memory already exists",
                "existing_id": dedup.existing.id if dedup.existing else None,
            }

        if dedup.action == DedupAction.MERGE and dedup.existing:
            from lace.memory.dedup import merge_memories
            merged = merge_memories(dedup.existing, candidate)
            store.save(merged)
            return {
                "stored": True,
                "status": "merged",
                "id": merged.id,
                "scope": merged.project_scope,
                "category": merged.category.value,
            }

    except Exception as e:
        _debug_log(f"Dedup check failed, storing anyway: {e}")

    # ── Store the memory ───────────────────────────────────────────────────
    memory = store.add(
        content=content,
        category=cat,
        tags=tags or [],
        scope=resolved_scope,
        source="mcp",
        confidence=max(0.0, min(1.0, confidence)),
    )

    # ── Log interaction ────────────────────────────────────────────────────
    try:
        lace_home = get_lace_home()
        from lace.utils.logging import RetrievalLogger
        logger = RetrievalLogger(lace_home)
        logger.log_interaction(
            query=content[:200],
            response_length=len(content),
            provider="mcp",
            model="antigravity",
            memories_used=0,
            latency_ms=0,
        )
    except Exception:
        pass

    return {
        "stored": True,
        "status": "stored",
        "id": memory.id,
        "scope": memory.project_scope,
        "category": memory.category.value,
    }


async def list_memories(
    scope: str = "auto",
    category: str = "all",
    limit: int = 20,
    **kwargs  # Accept lifecycle and other args
) -> list[dict]:
    """List recent memories, optionally filtered by category or scope."""
    store, resolved_scope = _get_store(scope)

    cat_filter = None
    if category != "all":
        try:
            cat_filter = MemoryCategory(category)
        except ValueError:
            pass

    # Handle lifecycle filter (ignore if "all" or invalid)
    lifecycle_filter = None
    if "lifecycle" in kwargs and kwargs["lifecycle"] != "all":
        try:
            from lace.memory.models import MemoryLifecycle
            lifecycle_filter = MemoryLifecycle(kwargs["lifecycle"])
        except (ValueError, KeyError):
            pass
    
    memories = store.list(
        category=cat_filter,
        scope=resolved_scope if resolved_scope != "global" else None,
        limit=limit,
        lifecycle=lifecycle_filter,
        include_archived=False,
    )

    return [
        {
            "id": m.id,
            "summary": m.display_summary(),
            "category": m.category.value,
            "tags": m.tags,
            "scope": m.project_scope,
            "confidence": m.confidence,
            "last_accessed": m.last_accessed.isoformat(),
        }
        for m in memories
    ]


async def forget_memory(
    memory_id: str,
    **kwargs
) -> dict:
    """Archive a memory."""
    store, _ = _get_store()
    success = store.forget(memory_id)

    return {
        "success": success,
        "status": "archived" if success else "not_found",
        "lifecycle": "archived" if success else "not_found",
        "id": memory_id,
        **({"error": f"Memory {memory_id} not found"} if not success else {}),
    }


async def get_related_concepts(
    concept: str,
    depth: int = 2,
    **kwargs  # Accept memories_only and other args
) -> list[dict]:
    """Find memories and concepts related to a given concept via the knowledge graph."""
    from lace.core.engine import GraphManager
    from lace.graph.traversal import find_memories_near_concept

    lace_home = get_lace_home()
    manager = GraphManager(lace_home=lace_home)
    G = manager.get_graph()

    if G.number_of_nodes() == 0:
        return []

    related = find_memories_near_concept(G, concept, depth=min(depth, 3))

    return [
        {
            "type": node["type"],
            "id": node["id"],
            "label": node.get("label", ""),
            "distance": node["distance"],
        }
        for node in related[:20]
    ]
