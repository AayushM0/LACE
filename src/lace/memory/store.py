"""Memory CRUD operations — the interface between the CLI/MCP and the vault."""

from __future__ import annotations

from pathlib import Path

from lace.core.config import LaceConfig, get_lace_home, load_config
from lace.core.scope import get_active_scope
from lace.memory.markdown import (
    load_all_memories,
    markdown_to_memory,
    save_memory_to_file,
)
from lace.memory.models import (
    MemoryCategory,
    MemoryLifecycle,
    MemoryObject,
    MemorySource,
    RetrievalResult,
    make_memory,
)


class MemoryStore:
    """Primary interface for reading and writing memories."""

    def __init__(
        self,
        lace_home: Path | None = None,
        config: LaceConfig | None = None,
        active_scope: str | None = None,
    ) -> None:
        self.lace_home   = lace_home or get_lace_home()
        self.config      = config or load_config(self.lace_home)
        self.vault_path  = self.config.vault_path(self.lace_home)
        self.vector_db_path = self.lace_home / "memory" / "vector_db"
        self.active_scope   = active_scope or "global"
        self._logger        = None   # lazy loaded

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _get_model_name(self) -> str:
        return self.config.embeddings.model

    def _embed(self, text: str) -> list[float]:
        """Embed text using the configured model."""
        from lace.retrieval.embeddings import embed_text
        return embed_text(text, model_name=self._get_model_name())

    def _upsert_to_vector_store(self, memory: MemoryObject) -> None:
        """Store memory embedding in ChromaDB. Silent on failure."""
        try:
            from lace.retrieval.vector import upsert_memory
            upsert_memory(memory, self.vector_db_path)
        except Exception:
            pass  # Vector store is optional — markdown is source of truth

    # ── Write ─────────────────────────────────────────────────────────────────

    def add(
        self,
        content: str,
        category: str | MemoryCategory = MemoryCategory.PATTERN,
        tags: list[str] | None = None,
        scope: str = "global",
        source: str | MemorySource = MemorySource.MANUAL,
        confidence: float = 0.8,
        summary: str | None = None,
        draft: bool = False,  # NEW: routes to inbox instead of vault
    ) -> MemoryObject:
        """Create and persist a new memory with embedding."""
        if draft:
            # Route to inbox — do not touch ChromaDB or vault
            from lace.memory.inbox import save_to_inbox
            
            memory = make_memory(
                content=content,
                category=category,
                tags=tags or [],
                scope=scope,
                source=source,
                confidence=confidence,
            )
            if summary:
                memory.summary = summary
                
            draft_id = save_to_inbox(memory)
            memory.id = draft_id
            return memory

        memory = make_memory(
            content=content,
            category=category,
            tags=tags or [],
            scope=scope,
            source=source,
            confidence=confidence,
        )
        if summary:
            memory.summary = summary

        # Generate embedding
        try:
            memory.embedding = self._embed(content)
        except Exception:
            memory.embedding = None  # Graceful degradation

        # Write markdown file (source of truth)
        save_memory_to_file(memory, self.vault_path)

        # Write to vector store
        self._upsert_to_vector_store(memory)

        return memory

    def save(self, memory: MemoryObject) -> Path:
        """Persist an existing MemoryObject (update file + vector store)."""
        path = save_memory_to_file(memory, self.vault_path)
        if memory.embedding is None:
            try:
                memory.embedding = self._embed(memory.content)
            except Exception:
                pass
        self._upsert_to_vector_store(memory)
        return path

    def forget(self, memory_id: str) -> bool:
        """Archive a memory — removes from search, never deletes file."""
        memory = self.get(memory_id)
        if memory is None:
            return False

        memory.archive()
        save_memory_to_file(memory, self.vault_path)

        # Update vector store metadata to reflect archived state
        self._upsert_to_vector_store(memory)
        return True

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, memory_id: str) -> MemoryObject | None:
        """Fetch a single memory by ID."""
        for md_file in self.vault_path.rglob(f"{memory_id}.md"):
            return markdown_to_memory(md_file)
        return None

    def record_access(self, memory_id: str) -> None:
        """Increment access count and update last_accessed timestamp.
        
        Called every time a memory is retrieved via MCP or search.
        Used for frequency scoring in multi-signal ranking.
        """
        from datetime import datetime, timezone
        
        memory = self.get(memory_id)
        if memory is None:
            return
        
        memory.access_count += 1
        memory.last_accessed = datetime.now(timezone.utc)
        self.save(memory)

    def list(
        self,
        category: str | MemoryCategory | None = None,
        scope: str | None = None,
        lifecycle: str | MemoryLifecycle | None = None,
        include_archived: bool = False,
        limit: int = 100,
    ) -> list[MemoryObject]:
        """Return memories with optional filtering."""
        memories = load_all_memories(self.vault_path)

        if not include_archived:
            memories = [m for m in memories if m.is_active()]

        if category is not None:
            cat = MemoryCategory(category) if isinstance(category, str) else category
            memories = [m for m in memories if m.category == cat]

        if scope is not None:
            memories = [m for m in memories if m.project_scope == scope]

        if lifecycle is not None:
            lc = MemoryLifecycle(lifecycle) if isinstance(lifecycle, str) else lifecycle
            memories = [m for m in memories if m.lifecycle == lc]

        memories.sort(key=lambda m: m.last_accessed, reverse=True)
        return memories[:limit]

    # ── Search ────────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        scope: str | None = None,
        max_results: int | None = None,
        threshold: float | None = None,
    ) -> list[RetrievalResult]:
        """Semantic search using vector similarity + multi-signal ranking."""
        import time

        cfg        = self.config.retrieval
        _max       = max_results or cfg.max_results
        _threshold = threshold or cfg.relevance_threshold
        _scope     = scope or self.active_scope

        start = time.perf_counter()

        try:
            results    = self._vector_search(query, _scope, _max, _threshold)
            match_type = "vector"
        except Exception:
            keyword_results = self.search_keyword(query, limit=_max)
            results = [
                RetrievalResult(
                    memory=m,
                    relevance_score=0.5,
                    match_type="keyword",
                    rank=i + 1,
                )
                for i, m in enumerate(keyword_results)
            ]
            match_type = "keyword"

        latency_ms = (time.perf_counter() - start) * 1000

        # Log every retrieval — silent on failure
        logger = self._get_logger()
        if logger:
            logger.log_retrieval(
                query=query,
                scope=_scope,
                results=results,
                latency_ms=latency_ms,
                match_type=match_type,
            )

        return results

    def _get_logger(self):
        """Lazy-load the retrieval logger."""
        if self._logger is None:
            if self.config.logging.retrieval_logs:
                from lace.utils.logging import RetrievalLogger
                self._logger = RetrievalLogger(self.lace_home)
        return self._logger

    def _vector_search(
        self,
        query: str,
        scope: str,
        max_results: int,
        threshold: float,
    ) -> list[RetrievalResult]:
        """Internal: perform vector search + ranking."""
        from lace.retrieval.embeddings import embed_text
        from lace.retrieval.vector import multi_scope_vector_search
        from lace.retrieval.ranking import rank_candidates, RankingWeights

        # Determine which scopes to search
        if scope == "global":
            scopes = ["global"]
        else:
            scopes = [scope, "global"]  # Project first, then global

        # Embed the query
        query_embedding = embed_text(query, model_name=self._get_model_name())

        # Search vector store
        raw_results = multi_scope_vector_search(
            query_embedding=query_embedding,
            scopes=scopes,
            vector_db_path=self.vector_db_path,
            n_results=max_results * 2,  # Over-fetch for ranking to filter
        )

        if not raw_results:
            return []

        # Load full MemoryObjects from markdown (vector store has metadata only)
        candidates: list[tuple[MemoryObject, float]] = []
        for result in raw_results:
            memory = self.get(result["id"])
            if memory is not None and memory.is_active():
                candidates.append((memory, result["distance"]))

        # Rank and filter
        weights = RankingWeights(
            semantic_similarity=self.config.retrieval.weights.semantic_similarity,
            recency=self.config.retrieval.weights.recency,
            frequency=self.config.retrieval.weights.frequency,
            confidence=self.config.retrieval.weights.confidence,
            scope=self.config.retrieval.weights.scope,
        )

        return rank_candidates(
            candidates=candidates,
            active_scope=scope,
            weights=weights,
            threshold=threshold,
            max_results=max_results,
        )

    def search_keyword(self, query: str, limit: int = 20) -> list[MemoryObject]:
        """Fallback keyword search when vector store is unavailable."""
        query_lower = query.lower()
        memories = self.list(include_archived=False, limit=10_000)

        matches: list[tuple[MemoryObject, int]] = []
        for memory in memories:
            score = 0
            text = (
                memory.content + " " +
                " ".join(memory.tags) + " " +
                memory.category.value
            ).lower()

            if query_lower in text:
                score += 10
            for word in query_lower.split():
                if len(word) > 2 and word in text:
                    score += 1

            if score > 0:
                matches.append((memory, score))

        matches.sort(key=lambda x: x[1], reverse=True)
        return [m for m, _ in matches[:limit]]

    def reindex_all(self) -> tuple[int, int]:
        """Re-embed and re-index all memories into the vector store.

        Useful after switching embedding models or if vector store
        gets out of sync with the markdown vault.

        Returns:
            (success_count, failure_count)
        """
        from lace.retrieval.vector import upsert_memory
        from lace.retrieval.embeddings import embed_text

        memories = load_all_memories(self.vault_path)
        success = 0
        failure = 0

        for memory in memories:
            try:
                memory.embedding = embed_text(
                    memory.content,
                    model_name=self._get_model_name(),
                )
                upsert_memory(memory, self.vector_db_path)
                success += 1
            except Exception as e:
                import traceback
                traceback.print_exc()
                failure += 1

        return success, failure

    def stats(self) -> dict[str, int | dict]:
        """Return memory statistics."""
        all_memories = load_all_memories(self.vault_path)

        by_category: dict[str, int] = {}
        by_lifecycle: dict[str, int] = {}
        by_scope: dict[str, int] = {}

        for memory in all_memories:
            by_category[memory.category.value] = by_category.get(memory.category.value, 0) + 1
            by_lifecycle[memory.lifecycle.value] = by_lifecycle.get(memory.lifecycle.value, 0) + 1
            by_scope[memory.project_scope] = by_scope.get(memory.project_scope, 0) + 1

        return {
            "total": len(all_memories),
            "active": sum(1 for m in all_memories if m.is_active()),
            "archived": sum(1 for m in all_memories if not m.is_active()),
            "by_category": by_category,
            "by_lifecycle": by_lifecycle,
            "by_scope": by_scope,
        }

    def rate(self, memory_id: str, signal: str) -> bool:
        """Update memory confidence based on explicit user feedback.

        Args:
            memory_id: The ID of the memory to rate.
            signal: "helpful", "outdated", or "wrong".

        Returns:
            True if memory was found and rated, False otherwise.
        """
        from datetime import datetime, timezone

        memory = self.get(memory_id)
        if memory is None:
            return False

        now = datetime.now(timezone.utc)

        if signal == "helpful":
            memory.confidence = min(1.0, memory.confidence + 0.05)
            memory.access_count += 1
            memory.last_accessed = now
            memory.metadata.pop("flagged", None)

        elif signal == "outdated":
            memory.confidence = max(0.1, memory.confidence * 0.5)
            memory.metadata["flagged"] = "outdated"
            memory.last_accessed = now

        elif signal == "wrong":
            memory.confidence = max(0.05, memory.confidence * 0.2)
            memory.metadata["flagged"] = "incorrect"
            memory.last_accessed = now

        else:
            return False

        self.save(memory)
        return True

    def get_review_candidates(
        self,
        min_confidence: float = 0.7,
        include_zero_access: bool = True,
        limit: int = 50,
    ) -> list[MemoryObject]:
        """Return memories that need attention for review.

        Criteria:
        - Confidence < min_confidence
        - OR never accessed (access_count == 0) and still "captured"
        - Excludes already archived memories
        """
        memories = self.list(include_archived=False, limit=10_000)

        candidates = []
        for m in memories:
            if m.confidence < min_confidence:
                candidates.append(m)
            elif include_zero_access and m.access_count == 0 and m.lifecycle.value == "captured":
                candidates.append(m)

        # Sort by confidence ascending (lowest first), then by creation date
        candidates.sort(key=lambda x: (x.confidence, x.created_at))
        return candidates[:limit]