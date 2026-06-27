"""
Inverted tag index for LACE memory retrieval.

Builds an in-memory index of tag → memory IDs and memory ID → tags.
Used by the unified retriever (Chunk 4) to expand search candidates
beyond what vector similarity alone can find.

This file has no side effects. It does not modify memories, write to
disk, or interact with ChromaDB. It is a pure read-only data structure
built from MemoryObject instances that already exist.

Lifecycle:
    1. TagIndex() — create empty index
    2. build(memories) — populate from all existing memories at startup
    3. add(memory) — update incrementally when a memory is stored
    4. remove(memory_id) — update incrementally when a memory is archived
    5. find_tag_neighbors() / get_tag_candidates() — used during retrieval
"""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lace.memory.models import MemoryObject


class TagIndex:
    """
    In-memory inverted index: tag → set[memory_id] and memory_id → set[tag].

    Both directions are maintained simultaneously so that:
    - Tag lookups are O(1)
    - Memory removal is O(tags_on_memory), not O(all_memories)

    Thread safety: not thread-safe. LACE is single-process; this is fine.
    """

    def __init__(self) -> None:
        # Forward index: tag → set of memory IDs
        self._tag_to_ids: dict[str, set[str]] = defaultdict(set)
        # Reverse index: memory ID → set of tags it carries
        self._id_to_tags: dict[str, set[str]] = {}

    # ── Build ─────────────────────────────────────────────────────────────────

    def build(self, memories: list[MemoryObject]) -> None:
        """
        Build the index from scratch from a list of MemoryObjects.

        Called once at startup after load_all_memories(). Replaces any
        existing index state — safe to call multiple times.

        Args:
            memories: All active memories loaded from the vault.
                      Archived memories should be excluded by the caller
                      (MemoryStore.initialize() filters them before passing here).
        """
        # Reset both indices
        self._tag_to_ids = defaultdict(set)
        self._id_to_tags = {}

        for memory in memories:
            self._index_memory(memory)

    # ── Incremental updates ───────────────────────────────────────────────────

    def add(self, memory: MemoryObject) -> None:
        """
        Add a single memory to the index.

        Called by MemoryStore.add() immediately after save_memory_to_file().
        If the memory is already in the index (e.g., called twice), this is
        a no-op because set.add() is idempotent.

        Args:
            memory: The newly stored MemoryObject.
        """
        self._index_memory(memory)

    def remove(self, memory_id: str) -> None:
        """
        Remove a memory from the index.

        Called by MemoryStore.forget() when a memory is archived.
        Uses the reverse index to find affected tag buckets — O(tags_on_memory).

        Args:
            memory_id: The ID of the memory being archived.
        """
        tags = self._id_to_tags.pop(memory_id, set())
        for tag in tags:
            self._tag_to_ids[tag].discard(memory_id)
            # Clean up empty tag buckets to prevent memory leak
            if not self._tag_to_ids[tag]:
                del self._tag_to_ids[tag]

    def update(self, memory: MemoryObject) -> None:
        """
        Update the index when a memory's tags change.

        Called if a memory's tags are edited after creation. Removes old
        state and re-indexes with new tags.

        Args:
            memory: The updated MemoryObject with new tags.
        """
        self.remove(memory.id)
        self._index_memory(memory)

    # ── Lookup ────────────────────────────────────────────────────────────────

    def get_memories_by_tag(self, tag: str) -> set[str]:
        """
        Return all memory IDs carrying a given tag.

        Args:
            tag: Exact tag string to look up.

        Returns:
            Set of memory IDs. Empty set if tag is not in index.
        """
        return frozenset(self._tag_to_ids.get(tag, set()))

    def get_tags_for_memory(self, memory_id: str) -> set[str]:
        """
        Return all tags on a given memory.

        Args:
            memory_id: The memory ID to look up.

        Returns:
            Set of tag strings. Empty set if memory is not in index.
        """
        return frozenset(self._id_to_tags.get(memory_id, set()))

    def get_all_tags(self) -> list[str]:
        """
        Return all known tags in the index, sorted alphabetically.

        Useful for autocomplete, debugging, and the tag neighbor algorithm
        which needs to iterate over all known tags.
        """
        return sorted(self._tag_to_ids.keys())

    def tag_count(self) -> int:
        """Return the number of distinct tags in the index."""
        return len(self._tag_to_ids)

    def memory_count(self) -> int:
        """Return the number of memories in the index."""
        return len(self._id_to_tags)

    # ── Specificity scoring ───────────────────────────────────────────────────

    def tag_specificity(self, tag: str) -> float:
        """
        Compute the specificity weight for a tag: 1.0 / number_of_members.

        Rare tags score high (specific). Common tags score low (noisy).
        A tag on 2 memories → 0.500. On 10 → 0.100. On 100 → 0.010.

        Args:
            tag: The tag to score.

        Returns:
            Float specificity weight. Returns 0.0 if tag is unknown.
        """
        members = self._tag_to_ids.get(tag)
        if not members:
            return 0.0
        return 1.0 / len(members)

    # ── Query tag extraction ──────────────────────────────────────────────────

    def extract_query_tags(self, query: str) -> list[str]:
        """
        Find index tags that match words or phrases in a query string.

        Three matching strategies applied in order of specificity:
        1. Exact word match — "jwt" in query words matches tag "jwt"
        2. Substring match — tag "rate-limiting" appears in query text
        3. Prefix match — query word "authenticate" and tag "auth" share
           first 4+ characters (only for words/tags >= 4 chars)

        Does not use NLP or stemming. Simple string heuristics that catch
        the obvious cases without external dependencies.

        Args:
            query: Raw query string from the user.

        Returns:
            List of matched tag strings from the index. May be empty.
            Deduped — each tag appears at most once.
        """
        if not query.strip():
            return []

        query_lower = query.lower().strip()
        # Split on whitespace and common punctuation
        query_words = set(
            word.strip("?.,!:;\"'()")
            for word in query_lower.split()
            if word.strip("?.,!:;\"'()")
        )

        matched: set[str] = set()

        for tag in self._tag_to_ids.keys():
            tag_lower = tag.lower()

            # Strategy 1: exact word match
            if tag_lower in query_words:
                matched.add(tag)
                continue

            # Strategy 2: substring match (for multi-word/hyphenated tags)
            if len(tag_lower) >= 3 and tag_lower in query_lower:
                matched.add(tag)
                continue

            # Strategy 3: prefix match (4+ char prefix overlap)
            if len(tag_lower) >= 4:
                tag_prefix = tag_lower[:4]
                for word in query_words:
                    if len(word) >= 4 and (
                        word.startswith(tag_prefix)
                        or tag_lower.startswith(word[:4])
                    ):
                        matched.add(tag)
                        break

        return sorted(matched)  # Sorted for deterministic behavior

    def get_tag_candidates(
        self,
        query: str,
        max_results: int = 20,
    ) -> list[str]:
        """
        Return memory IDs matching tags found in the query.

        Combines extract_query_tags() with get_memories_by_tag() to
        produce a flat list of candidate memory IDs for retrieval expansion.

        Args:
            query: Raw query string.
            max_results: Cap on returned IDs.

        Returns:
            List of memory IDs (deduplicated). Order is not meaningful —
            these are candidates for scoring, not ranked results.
        """
        tags = self.extract_query_tags(query)
        if not tags:
            return []

        candidate_ids: set[str] = set()
        for tag in tags:
            candidate_ids.update(self._tag_to_ids.get(tag, set()))

        return list(candidate_ids)[:max_results]

    # ── Tag neighbor finding ──────────────────────────────────────────────────

    def find_tag_neighbors(
        self,
        source_memory_ids: list[str],
        max_results: int = 15,
        min_score: float = 0.1,
    ) -> list[tuple[str, float]]:
        """
        Find memories that share tags with a set of source memories.

        Used in two places:
        1. During retrieval (Chunk 4) — expand candidates beyond vector results
        2. During add() (Chunk 5) — find related memories to link in the graph

        Algorithm:
            1. Collect all tags from all source memories
            2. For each tag, find all memories that carry it
            3. Skip any memory already in the source set
            4. Score each candidate by summing specificity weights of shared tags
            5. Normalize scores to [0, 1] range
            6. Filter by min_score and return top max_results

        Specificity weighting means that sharing the tag "jwt" (on 3 memories)
        scores higher than sharing "general" (on 200 memories).

        Args:
            source_memory_ids: Memory IDs to find neighbors for.
                               Typically the top-N vector search results.
            max_results: Maximum candidates to return.
            min_score: Minimum normalized score to include. Filters noise.

        Returns:
            List of (memory_id, score) sorted by score descending.
            Score is normalized to [0, 1]. Does not include source IDs.
        """
        if not source_memory_ids:
            return []

        source_set = set(source_memory_ids)

        # Step 1: collect all tags from source memories
        source_tags: set[str] = set()
        for mid in source_memory_ids:
            source_tags.update(self._id_to_tags.get(mid, set()))

        if not source_tags:
            return []

        # Step 2 + 3 + 4: find candidates and score them
        candidate_scores: dict[str, float] = defaultdict(float)

        for tag in source_tags:
            weight = self.tag_specificity(tag)
            if weight == 0.0:
                continue

            for member_id in self._tag_to_ids.get(tag, set()):
                if member_id not in source_set:
                    candidate_scores[member_id] += weight

        if not candidate_scores:
            return []

        # Step 5: normalize to [0, 1]
        max_score = max(candidate_scores.values())
        if max_score > 0:
            normalized = {
                mid: score / max_score
                for mid, score in candidate_scores.items()
            }
        else:
            normalized = dict(candidate_scores)

        # Step 6: filter, sort, cap
        results = [
            (mid, score)
            for mid, score in normalized.items()
            if score >= min_score
        ]
        results.sort(key=lambda x: x[1], reverse=True)

        return results[:max_results]

    def jaccard_similarity(self, memory_id_a: str, memory_id_b: str) -> float:
        """
        Compute Jaccard similarity between two memories' tag sets.

        Jaccard = |intersection| / |union|
        Range: 0.0 (no shared tags) to 1.0 (identical tag sets).

        Used by the graph builder (Chunk 2) to compute TAG_SIMILARITY
        edge weights between memory pairs.

        Args:
            memory_id_a: First memory ID.
            memory_id_b: Second memory ID.

        Returns:
            Jaccard similarity as a float. Returns 0.0 if either memory
            is not in the index or if both have empty tag sets.
        """
        tags_a = self._id_to_tags.get(memory_id_a, set())
        tags_b = self._id_to_tags.get(memory_id_b, set())

        if not tags_a or not tags_b:
            return 0.0

        intersection = len(tags_a & tags_b)
        union = len(tags_a | tags_b)

        return intersection / union if union > 0 else 0.0

    def shared_tags(self, memory_id_a: str, memory_id_b: str) -> set[str]:
        """
        Return the set of tags shared between two memories.

        Used by the graph builder to decide whether a TAG_SIMILARITY
        edge should be created (requires >= 2 shared specific tags).

        Args:
            memory_id_a: First memory ID.
            memory_id_b: Second memory ID.

        Returns:
            Set of shared tag strings. Empty set if no overlap.
        """
        tags_a = self._id_to_tags.get(memory_id_a, set())
        tags_b = self._id_to_tags.get(memory_id_b, set())
        return tags_a & tags_b

    # ── Diagnostics ───────────────────────────────────────────────────────────

    def stats(self) -> dict[str, int | list]:
        """
        Return diagnostic information about the index.

        Used by MemoryStore.stats() to include tag index health in
        the overall system stats report.

        Returns:
            Dict with counts and top tags by frequency.
        """
        tag_sizes = {
            tag: len(members)
            for tag, members in self._tag_to_ids.items()
        }
        top_tags = sorted(
            tag_sizes.items(), key=lambda x: x[1], reverse=True
        )[:10]

        return {
            "total_tags": len(self._tag_to_ids),
            "total_indexed_memories": len(self._id_to_tags),
            "top_tags": [{"tag": t, "count": c} for t, c in top_tags],
        }

    def debug_dump(self) -> dict[str, list[str]]:
        """
        Return the full forward index as a plain dict for debugging.

        Keys are tags, values are sorted lists of memory IDs.
        Only call this in development — can be large.
        """
        return {
            tag: sorted(ids)
            for tag, ids in self._tag_to_ids.items()
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _index_memory(self, memory: MemoryObject) -> None:
        """
        Add a single memory to both indices.

        Internal method — not called directly. Used by build() and add().
        Normalizes tags to lowercase for consistent matching.

        Args:
            memory: The MemoryObject to index.
        """
        # Normalize: lowercase, strip whitespace, skip empty strings
        normalized_tags = {
            tag.lower().strip()
            for tag in memory.tags
            if tag.strip()
        }

        if not normalized_tags:
            # Memory has no tags — still track it in reverse index
            # so remove() works correctly (it's a no-op for empty sets)
            self._id_to_tags[memory.id] = set()
            return

        # Update reverse index
        self._id_to_tags[memory.id] = normalized_tags

        # Update forward index
        for tag in normalized_tags:
            self._tag_to_ids[tag].add(memory.id)
