"""Memory CRUD operations — the interface between the CLI/MCP and the vault."""

from __future__ import annotations

from pathlib import Path

from lace.core.config import LaceConfig, get_lace_home, load_config, resolve_lace_paths
from lace.core.scope import get_active_scope
from lace.memory.markdown import (
    load_all_memories,
    markdown_to_memory,
    save_memory_to_file,
)
from lace.memory.models import (
    Confidence,
    MemoryCategory,
    MemoryLifecycle,
    MemoryObject,
    MemorySource,
    RetrievalResult,
    make_memory,
)

# ── NEW: Multi-signal retrieval imports ──────────────────────────────────────
from lace.retrieval.tag_index import TagIndex
from lace.retrieval.graph import MemoryGraph, EdgeType as GraphEdgeType
from lace.retrieval.co_occurrence import CoRetrievalTracker
from lace.retrieval.unified import UnifiedRetriever, UnifiedWeights


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
        paths            = resolve_lace_paths(self.lace_home)
        self.vault_path  = self.config.vault_path(self.lace_home)
        self.vector_db_path = paths["vector_db"]
        self.active_scope   = active_scope or "global"
        self._logger        = None   # lazy loaded

        # ── NEW: Multi-signal retrieval components ──────────────────────────
        self._tag_index   = TagIndex()
        self._graph       = MemoryGraph()
        self._co_tracker  = CoRetrievalTracker(graph=self._graph)
        self._unified     = None
        self._initialized = False

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
        except Exception as e:
            import logging
            logging.getLogger("lace.store").error(
                f"Failed to upsert memory {memory.id} to vector store: {e}",
                exc_info=True
            )

    # ── NEW: Initialization ───────────────────────────────────────────────────

    def initialize(self) -> dict[str, int | float | str]:
        """
        Build tag index, load graph, restore co-retrievals, create unified retriever.

        Must be called after construction, before the first search() call.
        """
        import time
        import logging

        start = time.perf_counter()

        # 1. Load active memories
        all_memories = load_all_memories(self.vault_path)
        active = [m for m in all_memories if m.is_active()]
        memory_count = len(active)

        # 2. Build tag index
        self._tag_index.build(active)
        tag_count = self._tag_index.tag_count()

        # 3. Load persisted graph
        paths = resolve_lace_paths(self.lace_home)
        graph_path = paths["graph"]
        loaded_edges = self._graph.load(graph_path)

        # 4. Run batch graph builder
        new_edges = self._graph.build_from_memories(active, self._tag_index)

        # 5. Save the now-complete graph
        self._graph.save(graph_path)
        total_edge_count = self._graph.edge_count()

        # 6. Load persisted co-retrieval tracker
        co_path = paths["co_retrieval"]
        self._co_tracker.load(co_path)

        # 7. Set co-tracker save path
        self._co_tracker.set_save_path(co_path)

        # 8. Create UnifiedRetriever
        def vector_search_wrapper(query: str, n_results: int, scope: str) -> list[dict]:
            try:
                from lace.retrieval.embeddings import embed_text
                from lace.retrieval.vector import multi_scope_vector_search

                scopes = self._get_search_scopes(scope)
                query_embedding = embed_text(
                    query, model_name=self._get_model_name()
                )
                return multi_scope_vector_search(
                    query_embedding=query_embedding,
                    scopes=scopes,
                    vector_db_path=self.vector_db_path,
                    n_results=n_results,
                )
            except Exception:
                import logging as _log
                _log.getLogger("lace.store").error(
                    "vector_search_wrapper failed — vector results unavailable for this query. "
                    "Memories stored while embeddings are broken will not appear in vector search "
                    "until 'lace memory reindex' is run.",
                    exc_info=True,
                )
                return []

        retrieval_weights = UnifiedWeights()
        if self.config and hasattr(self.config, "retrieval") and hasattr(self.config.retrieval, "weights"):
            cfg_w = self.config.retrieval.weights
            try:
                retrieval_weights = UnifiedWeights(
                    vector=cfg_w.vector,
                    tag=cfg_w.tag,
                    graph=cfg_w.graph,
                    co_retrieval=cfg_w.co_retrieval,
                    recency=cfg_w.recency,
                    confidence=cfg_w.confidence
                )
            except Exception as e:
                import logging as _log
                _log.getLogger("lace.store").warning(
                    f"Invalid weights configuration, using defaults: {e}"
                )

        self._unified = UnifiedRetriever(
            vector_search_fn=vector_search_wrapper,
            get_memory_fn=self.get,
            tag_index=self._tag_index,
            graph=self._graph,
            co_tracker=self._co_tracker,
            weights=retrieval_weights,
        )

        # 9. Mark as initialized
        self._initialized = True

        elapsed = time.perf_counter() - start

        co_stats = self._co_tracker.stats()
        results = {
            "indexed_memories":    memory_count,
            "distinct_tags":       tag_count,
            "graph_edges":         total_edge_count,
            "edges_loaded":        loaded_edges,
            "edges_discovered":    (
                new_edges.get("explicit", 0)
                + new_edges.get("tag_similarity", 0)
                + new_edges.get("temporal", 0)
            ),
            "co_pairs_loaded":     co_stats.get("tracked_pairs", 0),
            "initialization_ms":   round(elapsed * 1000, 1),
        }

        logging.getLogger("lace.store").info(
            f"MemoryStore initialized: "
            f"{memory_count} memories, {tag_count} tags, "
            f"{total_edge_count} graph edges "
            f"({elapsed:.1f}s)"
        )

        return results

    def _get_search_scopes(self, scope: str) -> list[str]:
        """Determine which ChromaDB collections to search based on scope."""
        if scope == "global":
            return ["global"]
        return [scope, "global"]

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
        draft: bool = False,
    ) -> MemoryObject:
        """Create and persist a new memory with embedding."""
        if draft:
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
        except Exception as e:
            import logging
            logging.getLogger("lace.store").error(
                f"Embedding generation failed for memory {memory.id}: {e} — "
                "memory will be stored in the vault but will NOT appear in vector search. "
                "Set needs_reindex=True and run 'lace memory reindex' to recover.",
                exc_info=True,
            )
            memory.embedding = None
            memory.needs_reindex = True

        # Write markdown file (source of truth)
        save_memory_to_file(memory, self.vault_path)

        # Write to vector store
        self._upsert_to_vector_store(memory)

        # ── NEW: Update indices for multi-signal retrieval ──────────────────
        if self._initialized:
            self._update_indices_for_new_memory(memory)

        return memory

    def _update_indices_for_new_memory(self, memory: MemoryObject) -> None:
        """Update tag index, graph, and persistence after storing a new memory."""
        # Update tag index
        self._tag_index.add(memory)

        # Create EXPLICIT edges from related_ids
        for related_id in memory.related_ids:
            related = self.get(related_id)
            if related is not None and related.is_active():
                self._graph.add_edge(
                    source=memory.id,
                    target=related_id,
                    edge_type=GraphEdgeType.EXPLICIT,
                    weight=1.0,
                    bidirectional=True,
                )

        # Create TAG_SIMILARITY edges from shared tags
        tag_neighbors = self._tag_index.find_tag_neighbors(
            source_memory_ids=[memory.id],
            max_results=10,
            min_score=0.3,
        )
        for neighbor_id, _ in tag_neighbors:
            shared = self._tag_index.shared_tags(memory.id, neighbor_id)
            if len(shared) >= self._graph.MIN_SHARED_TAGS:
                jaccard = self._tag_index.jaccard_similarity(memory.id, neighbor_id)
                self._graph.add_edge(
                    source=memory.id,
                    target=neighbor_id,
                    edge_type=GraphEdgeType.TAG_SIMILARITY,
                    weight=jaccard,
                    bidirectional=True,
                )

        # Save the updated graph
        paths = resolve_lace_paths(self.lace_home)
        graph_path = paths["graph"]
        self._graph.save(graph_path)

    def save(self, memory: MemoryObject) -> Path:
        """Persist an existing MemoryObject (update file + vector store)."""
        path = save_memory_to_file(memory, self.vault_path)
        if memory.embedding is None:
            try:
                memory.embedding = self._embed(memory.content)
            except Exception as e:
                import logging
                logging.getLogger("lace.store").error(
                    f"Embedding generation failed for memory {memory.id} on save: {e}",
                    exc_info=True
                )
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

        # ── NEW: Clean up indices ─────────────────────────────────────────
        if self._initialized:
            self._clean_indices_for_archived_memory(memory_id)

        return True

    def _clean_indices_for_archived_memory(self, memory_id: str) -> None:
        """Remove archived memory from tag index, graph, and co-retrieval tracker."""
        self._tag_index.remove(memory_id)
        self._graph.remove_node(memory_id)
        self._co_tracker.purge_memory(memory_id)

        # Force save graph and co-tracker after destructive operation
        paths = resolve_lace_paths(self.lace_home)
        graph_path = paths["graph"]
        self._graph.save(graph_path)

        co_path = paths["co_retrieval"]
        self._co_tracker.save(co_path)

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, memory_id: str) -> MemoryObject | None:
        """Fetch a single memory by ID."""
        for md_file in self.vault_path.rglob(f"{memory_id}.md"):
            return markdown_to_memory(md_file)
        return None

    def record_access(self, memory_id: str) -> None:
        """Increment access count and update last_accessed timestamp."""
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
        min_confidence: Confidence | float | None = None,
    ) -> list[RetrievalResult]:
        """Semantic search using vector similarity + multi-signal ranking."""
        import time

        cfg        = self.config.retrieval
        _max       = max_results or cfg.max_results
        _threshold = threshold or cfg.relevance_threshold
        _scope     = scope or self.active_scope

        start = time.perf_counter()

        if self._initialized and self._unified is not None:
            # ── Use unified multi-signal retriever ────────────────────────
            try:
                results = self._unified.retrieve(
                    query=query,
                    max_results=_max,
                    threshold=_threshold,
                    active_scope=_scope,
                    min_confidence=min_confidence,
                )
                match_type = "unified"
            except Exception:
                import logging as _log
                _log.getLogger("lace.store").warning(
                    "UnifiedRetriever.retrieve() failed — falling back to vector-only search.",
                    exc_info=True,
                )
                # Fallback to vector-only on error
                results = self._fallback_search(query, _scope, _max, _threshold, min_confidence)
                match_type = self._detect_match_type(results)
        else:
            # ── Original behavior (not initialized) ───────────────────────
            results = self._fallback_search(query, _scope, _max, _threshold, min_confidence)
            match_type = self._detect_match_type(results)

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

    def _fallback_search(
        self,
        query: str,
        scope: str,
        max_results: int,
        threshold: float,
        min_confidence: Confidence | float | None = None,
    ) -> list[RetrievalResult]:
        """
        Original vector search + keyword fallback logic.
        Preserves exact original behavior.
        """
        try:
            results = self._vector_search(query, scope, max_results, threshold)
        except Exception:
            import logging as _log
            _log.getLogger("lace.store").warning(
                "_vector_search() failed inside _fallback_search — degrading to keyword-only.",
                exc_info=True,
            )
            keyword_results = self.search_keyword(query, limit=max_results)
            results = [
                RetrievalResult(
                    memory=m,
                    relevance_score=0.5,
                    match_type="keyword",
                    rank=i + 1,
                )
                for i, m in enumerate(keyword_results)
            ]
        if min_confidence is not None:
            results = [r for r in results if r.memory.confidence >= float(min_confidence)]
        return results

    def _detect_match_type(self, results: list[RetrievalResult]) -> str:
        """Derive match_type from results if available, else default."""
        if results and hasattr(results[0], "match_type"):
            return results[0].match_type
        return "vector"

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
        scopes = self._get_search_scopes(scope)

        # Embed the query
        query_embedding = embed_text(query, model_name=self._get_model_name())

        # Search vector store
        raw_results = multi_scope_vector_search(
            query_embedding=query_embedding,
            scopes=scopes,
            vector_db_path=self.vector_db_path,
            n_results=max_results * 2,
        )

        if not raw_results:
            return []

        # Load full MemoryObjects from markdown
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
        """Re-embed and re-index all memories into the vector store."""
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
                import logging
                logging.getLogger("lace.store").error(
                    f"Reindex failed for memory {memory.id}: {e}",
                    exc_info=True,
                )
                memory.needs_reindex = True
                save_memory_to_file(memory, self.vault_path)
                failure += 1

        return success, failure

    def stats(self) -> dict[str, int | dict]:
        """Return memory statistics including multi-signal index health."""
        all_memories = load_all_memories(self.vault_path)

        by_category: dict[str, int] = {}
        by_lifecycle: dict[str, int] = {}
        by_scope: dict[str, int] = {}

        for memory in all_memories:
            by_category[memory.category.value] = by_category.get(memory.category.value, 0) + 1
            by_lifecycle[memory.lifecycle.value] = by_lifecycle.get(memory.lifecycle.value, 0) + 1
            by_scope[memory.project_scope] = by_scope.get(memory.project_scope, 0) + 1

        result: dict = {
            "total": len(all_memories),
            "active": sum(1 for m in all_memories if m.is_active()),
            "archived": sum(1 for m in all_memories if not m.is_active()),
            "by_category": by_category,
            "by_lifecycle": by_lifecycle,
            "by_scope": by_scope,
        }

        # ── NEW: Add index stats if initialized ───────────────────────────
        if self._initialized:
            result["tag_index"]      = self._tag_index.stats()
            result["graph"]          = self._graph.stats()
            result["co_retrieval"]   = self._co_tracker.stats()
            result["retrieval_mode"] = "unified"

            if self._unified is not None:
                result["weights"] = self._unified.get_weights().to_dict()
        else:
            result["retrieval_mode"] = "classic"

        return result

    def rate(self, memory_id: str, signal: str) -> bool:
        """Update memory confidence based on explicit user feedback."""
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
        min_confidence: Confidence | float = 0.7,
        include_zero_access: bool = True,
        limit: int = 50,
    ) -> list[MemoryObject]:
        """Return memories that need attention for review."""
        memories = self.list(include_archived=False, limit=10_000)

        candidates = []
        for m in memories:
            if m.confidence < float(min_confidence):
                candidates.append(m)
            elif include_zero_access and m.access_count == 0 and m.lifecycle.value == "captured":
                candidates.append(m)

        candidates.sort(key=lambda x: (x.confidence, x.created_at))
        return candidates[:limit]
