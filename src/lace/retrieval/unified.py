"""
Unified multi-signal retriever for LACE.

Orchestrates tag index, memory graph, and co-retrieval tracker into a
single retrieval pipeline that extends vector search with structural
and learned signals.

Pipeline (7 steps):
    1. Vector search  — existing ChromaDB cosine similarity
    2. Tag expansion  — candidates from query tag matching
    3. Graph expansion — neighbors of top vector results
    4. Co-retrieval boost — learned usage pattern scores
    5. Load MemoryObjects — fetch full objects for expanded candidates
    6. Score everything — weighted combination of all signals
    7. Filter, sort, rank, record, return

Output: list[RetrievalResult] — identical type to existing search().
Match types: "vector", "tag_expansion", "graph", "hybrid"

This file does not modify any memory, write to disk, or change
ChromaDB. It is a pure read-and-rank layer on top of existing systems.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Callable

from lace.memory.models import MemoryObject, RetrievalResult

if TYPE_CHECKING:
    from lace.retrieval.tag_index import TagIndex
    from lace.retrieval.graph import MemoryGraph
    from lace.retrieval.co_occurrence import CoRetrievalTracker


# ── Internal candidate representation ────────────────────────────────────────

@dataclass
class RetrievalCandidate:
    """
    Internal representation of a memory during the retrieval pipeline.

    Holds all signal scores separately so the final scoring step can
    combine them with configurable weights. Also tracks why each
    candidate entered the pool (for match_type and explainability).

    Not exposed outside this file — converted to RetrievalResult at
    the end of the pipeline.
    """
    memory_id: str

    # The full MemoryObject — may be None until Step 5 loads it
    memory: MemoryObject | None = None

    # Individual signal scores — all default 0.0
    vector_score:       float = 0.0
    tag_score:          float = 0.0
    graph_score:        float = 0.0
    co_retrieval_score: float = 0.0
    recency_score:      float = 0.0

    # Final combined score — computed in Step 6
    final_score: float = 0.0

    # How this candidate entered the pool
    # Values: "vector", "tag_expansion", "graph", combinations → "hybrid"
    sources: list[str] = field(default_factory=list)

    def add_source(self, source: str) -> None:
        """Record that this candidate entered via a given signal."""
        if source not in self.sources:
            self.sources.append(source)

    def match_type(self) -> str:
        """
        Derive RetrievalResult.match_type from sources list.

        Single source → that source name.
        Multiple sources → "hybrid".
        """
        if not self.sources:
            return "unknown"
        if len(self.sources) == 1:
            return self.sources[0]
        return "hybrid"


# ── Weight configuration ──────────────────────────────────────────────────────

@dataclass
class UnifiedWeights:
    """
    Weights for the five retrieval signals.

    Must sum to 1.0. Validated on construction.
    Defaults represent a reasonable starting point —
    vector search stays dominant, structural signals augment it.

    Can be overridden by passing custom weights to UnifiedRetriever.
    In Chunk 5, these are read from LaceConfig when available.
    """
    vector:       float = 0.45
    tag:          float = 0.15
    graph:        float = 0.15
    co_retrieval: float = 0.10
    recency:      float = 0.10
    confidence:   float = 0.05

    def __post_init__(self) -> None:
        total = (
            self.vector + self.tag + self.graph
            + self.co_retrieval + self.recency + self.confidence
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(
                f"UnifiedWeights must sum to 1.0, got {total:.3f}"
            )

    def to_dict(self) -> dict[str, float]:
        return {
            "vector":       self.vector,
            "tag":          self.tag,
            "graph":        self.graph,
            "co_retrieval": self.co_retrieval,
            "recency":      self.recency,
            "confidence":   self.confidence,
        }


# ── Main retriever ────────────────────────────────────────────────────────────

class UnifiedRetriever:
    """
    Multi-signal memory retriever.

    Combines vector similarity, tag structure, graph relationships,
    and learned co-retrieval patterns into a single ranked result list.

    Requires:
        - A callable that performs vector search and returns raw results
        - A callable that fetches a single MemoryObject by ID
        - TagIndex from Chunk 1
        - MemoryGraph from Chunk 2
        - CoRetrievalTracker from Chunk 3

    The store callables are passed as functions rather than the store
    itself to avoid circular imports (store.py will import this file
    in Chunk 5, so this file must not import store.py).
    """

    # How many top vector results to use as graph expansion seeds
    GRAPH_SEED_COUNT: int = 5

    # Maximum candidates to add via tag expansion
    MAX_TAG_CANDIDATES: int = 20

    # Maximum candidates to add via graph expansion
    MAX_GRAPH_CANDIDATES: int = 20

    # Recency half-life in days (matches existing ranking.py constant)
    RECENCY_HALF_LIFE_DAYS: int = 30

    def __init__(
        self,
        vector_search_fn: Callable[[str, int], list[dict]],
        get_memory_fn: Callable[[str], MemoryObject | None],
        tag_index: TagIndex,
        graph: MemoryGraph,
        co_tracker: CoRetrievalTracker,
        weights: UnifiedWeights | None = None,
    ) -> None:
        """
        Create a UnifiedRetriever.

        Args:
            vector_search_fn: Callable(query, n_results) → list of dicts
                              with keys "id" and "distance".
                              This wraps the existing _vector_search logic
                              in MemoryStore.
            get_memory_fn: Callable(memory_id) → MemoryObject or None.
                           Wraps MemoryStore.get().
            tag_index: Built TagIndex instance from Chunk 1.
            graph: Built MemoryGraph instance from Chunk 2.
            co_tracker: CoRetrievalTracker instance from Chunk 3.
            weights: Signal weights. Uses UnifiedWeights defaults if None.
        """
        self._vector_search = vector_search_fn
        self._get_memory    = get_memory_fn
        self._tag_index     = tag_index
        self._graph         = graph
        self._co_tracker    = co_tracker
        self._weights       = weights or UnifiedWeights()

    # ── Main pipeline ─────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        max_results: int = 10,
        threshold: float = 0.35,
        active_scope: str = "global",
    ) -> list[RetrievalResult]:
        """
        Run the full multi-signal retrieval pipeline.

        This is the only public method callers need. Called by
        MemoryStore.search() in Chunk 5 when initialized.

        Args:
            query: Raw query string from the user.
            max_results: Maximum memories to return.
            threshold: Minimum final_score to include in results.
                       Matches the existing relevance_threshold in config.
            active_scope: Current project scope for scope scoring.

        Returns:
            list[RetrievalResult] — identical type to existing search().
            Sorted by relevance_score descending. Ranked from 1.
        """
        # Candidate pool: memory_id → RetrievalCandidate
        pool: dict[str, RetrievalCandidate] = {}

        # ── Step 1: Vector Search ─────────────────────────────────────────────
        # Over-fetch to give ranking room to reorder
        raw_vector = self._vector_search(query, max_results * 3)

        for raw in raw_vector:
            mid = raw["id"]
            distance = raw["distance"]

            # Convert ChromaDB cosine distance to similarity score
            # distance = 1 - cosine_similarity, so similarity = 1 - distance
            # Divide by 2 to normalize to [0, 1] (max distance is 2.0)
            vector_score = max(0.0, 1.0 - (distance / 2.0))

            candidate = RetrievalCandidate(
                memory_id=mid,
                memory=raw.get("memory"),  # may be pre-loaded or None
                vector_score=vector_score,
            )
            candidate.add_source("vector")
            pool[mid] = candidate

        # ── Step 2: Tag Expansion ─────────────────────────────────────────────
        matched_tags = self._tag_index.extract_query_tags(query)

        if matched_tags:
            tag_candidate_ids = self._tag_index.get_tag_candidates(
                query, max_results=self.MAX_TAG_CANDIDATES
            )

            for mid in tag_candidate_ids:
                if mid not in pool:
                    # New candidate from tag expansion
                    pool[mid] = RetrievalCandidate(memory_id=mid)
                    pool[mid].add_source("tag_expansion")
                else:
                    # Already in pool from vector — mark as hybrid
                    pool[mid].add_source("tag_expansion")

        # ── Step 3: Graph Expansion ───────────────────────────────────────────
        # Seed from top vector results only (not the full pool)
        top_vector_ids = sorted(
            [mid for mid, c in pool.items() if "vector" in c.sources],
            key=lambda mid: pool[mid].vector_score,
            reverse=True,
        )[:self.GRAPH_SEED_COUNT]

        graph_candidate_scores: dict[str, float] = {}

        for source_id in top_vector_ids:
            neighbors = self._graph.get_neighbors(
                memory_id=source_id,
                max_depth=2,
                min_weight=0.1,
            )
            for neighbor_id, weight, depth in neighbors:
                # Take the maximum weight if reachable from multiple seeds
                current = graph_candidate_scores.get(neighbor_id, 0.0)
                graph_candidate_scores[neighbor_id] = max(current, weight)

        # Add graph candidates to pool (cap at MAX_GRAPH_CANDIDATES)
        graph_added = 0
        for mid, graph_score in sorted(
            graph_candidate_scores.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            if graph_added >= self.MAX_GRAPH_CANDIDATES:
                break

            if mid not in pool:
                pool[mid] = RetrievalCandidate(memory_id=mid)
                pool[mid].add_source("graph")
                graph_added += 1
            else:
                pool[mid].add_source("graph")

            # Set or update graph score — take max across sources
            pool[mid].graph_score = max(
                pool[mid].graph_score, graph_score
            )

        # ── Step 4: Co-retrieval Boost ────────────────────────────────────────
        # Check each candidate against the top vector results
        for mid, candidate in pool.items():
            candidate.co_retrieval_score = (
                self._co_tracker.get_co_retrieval_boost(
                    candidate_id=mid,
                    top_result_ids=top_vector_ids,
                )
            )

        # ── Step 5: Load Full MemoryObjects ───────────────────────────────────
        # Vector candidates may already have memory loaded (depends on
        # how vector_search_fn is implemented). Tag and graph candidates
        # only have IDs. Load everything that is still None.
        ids_to_drop: list[str] = []

        for mid, candidate in pool.items():
            if candidate.memory is None:
                candidate.memory = self._get_memory(mid)

            if candidate.memory is None:
                # Memory was deleted or corrupt — remove from pool
                ids_to_drop.append(mid)
                continue

            if not candidate.memory.is_active():
                # Archived memory — should not appear in results
                ids_to_drop.append(mid)

        for mid in ids_to_drop:
            del pool[mid]

        if not pool:
            return []

        # ── Step 6: Score Everything ──────────────────────────────────────────
        now = datetime.now(timezone.utc)

        for mid, candidate in pool.items():
            # Tag score: fraction of query tags that appear in this memory's tags
            candidate.tag_score = self._compute_tag_score(
                candidate.memory, matched_tags
            )

            # Recency score: exponential decay from last_accessed
            candidate.recency_score = self._compute_recency_score(
                candidate.memory.last_accessed, now
            )

            # Scope bonus (carried from existing ranking logic)
            scope_bonus = self._compute_scope_score(
                candidate.memory.project_scope, active_scope
            )

            # Combine confidence with scope bonus for the confidence slot
            # (scope was a separate signal in old ranking — we fold it in here)
            effective_confidence = (
                candidate.memory.confidence * 0.7 + scope_bonus * 0.3
            )

            # Final weighted score
            candidate.final_score = (
                self._weights.vector       * candidate.vector_score
                + self._weights.tag        * candidate.tag_score
                + self._weights.graph      * candidate.graph_score
                + self._weights.co_retrieval * candidate.co_retrieval_score
                + self._weights.recency    * candidate.recency_score
                + self._weights.confidence * effective_confidence
            )

        # ── Step 7: Filter, Sort, Rank, Record, Return ────────────────────────
        # Filter below threshold
        passing = [
            c for c in pool.values()
            if c.final_score >= threshold
        ]

        # Sort by final score descending
        passing.sort(key=lambda c: c.final_score, reverse=True)

        # Cap at max_results
        final_candidates = passing[:max_results]

        # Convert to RetrievalResult (existing public type)
        results: list[RetrievalResult] = []
        for rank, candidate in enumerate(final_candidates, start=1):
            results.append(RetrievalResult(
                memory=candidate.memory,
                relevance_score=round(candidate.final_score, 4),
                match_type=candidate.match_type(),
                rank=rank,
            ))

        # Record this retrieval for the co-retrieval tracker
        # MUST happen after we decide what to return, not before
        returned_ids = [r.memory.id for r in results]
        if returned_ids:
            self._co_tracker.record_retrieval(returned_ids)

        return results

    # ── Explainability ────────────────────────────────────────────────────────

    def explain(
        self,
        query: str,
        memory_id: str,
        active_scope: str = "global",
    ) -> dict | None:
        """
        Return a detailed signal breakdown for a specific memory.

        Used for debugging retrieval behavior. Not called during normal
        retrieval — only when explicitly requested.

        Args:
            query: The query to explain retrieval for.
            memory_id: Which memory to explain.
            active_scope: Current scope.

        Returns:
            Dict with all signal scores and their weighted contributions.
            None if the memory was not retrieved for this query.
        """
        # Run the full pipeline but look for specific memory
        results = self.retrieve(
            query=query,
            max_results=50,
            threshold=0.0,  # No threshold — include everything
            active_scope=active_scope,
        )

        for result in results:
            if result.memory.id == memory_id:
                return {
                    "memory_id":      memory_id,
                    "query":          query,
                    "final_score":    result.relevance_score,
                    "match_type":     result.match_type,
                    "rank":           result.rank,
                    "weights_used":   self._weights.to_dict(),
                }

        return None

    def get_weights(self) -> UnifiedWeights:
        """Return the current weight configuration."""
        return self._weights

    def set_weights(self, weights: UnifiedWeights) -> None:
        """
        Update signal weights at runtime.

        Useful for experimentation without restarting.
        Weights are validated by UnifiedWeights.__post_init__().
        """
        self._weights = weights

    # ── Signal score helpers ──────────────────────────────────────────────────

    def _compute_tag_score(
        self,
        memory: MemoryObject,
        matched_tags: list[str],
    ) -> float:
        """
        Compute how well a memory's tags match the query tags.

        Score = number of query tags found in memory tags
                / total number of query tags

        A memory with tags [jwt, auth, api] and query tags [jwt, auth]
        scores 2/2 = 1.0. A memory with tags [jwt, reference] scores
        1/2 = 0.5.

        Returns 0.0 if no query tags were found (empty matched_tags).

        Args:
            memory: The MemoryObject being scored.
            matched_tags: Tags extracted from the query by TagIndex.

        Returns:
            Float between 0.0 and 1.0.
        """
        if not matched_tags:
            return 0.0

        memory_tags_lower = {t.lower().strip() for t in memory.tags}
        hits = sum(
            1 for tag in matched_tags
            if tag.lower() in memory_tags_lower
        )
        return hits / len(matched_tags)

    def _compute_recency_score(
        self,
        last_accessed: datetime,
        now: datetime,
    ) -> float:
        """
        Exponential decay score based on time since last access.

        Uses the same half-life formula as existing ranking.py:
            score = 0.5 ^ (days_elapsed / half_life_days)

        Accessed today → 1.0
        Accessed 30 days ago → 0.5
        Accessed 60 days ago → 0.25

        Args:
            last_accessed: When the memory was last retrieved.
            now: Current time (passed in to avoid repeated datetime.now()).

        Returns:
            Float between 0.0 and 1.0.
        """
        # Handle naive datetimes
        if last_accessed.tzinfo is None:
            last_accessed = last_accessed.replace(tzinfo=timezone.utc)

        days_elapsed = max(0, (now - last_accessed).days)
        return 0.5 ** (days_elapsed / self.RECENCY_HALF_LIFE_DAYS)

    def _compute_scope_score(
        self,
        memory_scope: str,
        active_scope: str,
    ) -> float:
        """
        Score based on scope match between memory and active context.

        Mirrors the scope_score function in existing ranking.py exactly.
        Kept here so the unified retriever is self-contained.

        Args:
            memory_scope: The memory's project_scope field.
            active_scope: The current active scope.

        Returns:
            1.0 exact match, 0.5 global memory, 0.2 different project.
        """
        if memory_scope == active_scope:
            return 1.0
        elif memory_scope == "global":
            return 0.5
        elif (
            memory_scope.startswith("project:")
            and active_scope.startswith("project:")
        ):
            return 0.2
        else:
            return 0.3
