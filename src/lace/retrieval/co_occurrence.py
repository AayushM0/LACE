"""
Co-retrieval tracker for LACE memory system.

Tracks which memories are retrieved together and computes statistical
associations using Normalized Pointwise Mutual Information (NPMI).

Over time this builds a learned picture of which memories tend to be
useful in the same context — without any manual configuration.

When a pair's NPMI score crosses a threshold, a CO_RETRIEVED edge is
created in the MemoryGraph (Chunk 2), making the association permanent
and usable by graph traversal in the UnifiedRetriever (Chunk 4).

Lifecycle:
    1. CoRetrievalTracker(graph) — create with graph reference
    2. load(filepath) — restore from JSON at startup
    3. record_retrieval(memory_ids) — called after every retrieval
    4. get_pmi_score(id_a, id_b) — used during retrieval scoring
    5. save(filepath) — called periodically (every 10 retrievals)
    6. apply_decay() — called every 100 retrievals automatically
    7. purge_memory(memory_id) — called when a memory is archived
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lace.retrieval.graph import MemoryGraph


class CoRetrievalTracker:
    """
    Tracks co-retrieval patterns and computes NPMI-based associations.

    Storage:
        _co_counts[(id_a, id_b)] = float  (always id_a < id_b lexicographically)
        _retrieval_counts[id]    = float  (float to support decay arithmetic)
        _total_retrievals        = float

    Counts are floats (not ints) because decay multiplies them by 0.95.
    Integer counts would lose precision after repeated decay cycles.

    Thread safety: not thread-safe. LACE is single-process.
    """

    # NPMI threshold to create a CO_RETRIEVED graph edge
    GRAPH_EDGE_THRESHOLD: float = 0.3

    # Minimum co-occurrence count before NPMI is trusted
    MIN_CO_COUNT: float = 3.0

    # Auto-save every N retrievals
    SAVE_EVERY: int = 10

    # Apply decay every N retrievals
    DECAY_EVERY: int = 100

    # Decay multiplier applied to all counts
    DECAY_FACTOR: float = 0.95

    # Prune pairs with co_count below this after decay
    PRUNE_THRESHOLD: float = 0.5

    def __init__(self, graph: MemoryGraph | None = None) -> None:
        """
        Create a new co-retrieval tracker.

        Args:
            graph: The MemoryGraph instance from Chunk 2. When provided,
                   CO_RETRIEVED edges are automatically created when NPMI
                   crosses GRAPH_EDGE_THRESHOLD. If None, tracking still
                   works but no graph edges are created.
        """
        # Pair co-occurrence counts. Key: (id_a, id_b) where id_a < id_b.
        self._co_counts: dict[tuple[str, str], float] = defaultdict(float)

        # Individual memory retrieval counts.
        self._retrieval_counts: dict[str, float] = defaultdict(float)

        # Total number of record_retrieval() calls.
        self._total_retrievals: float = 0.0

        # Counts since last save (triggers auto-save at SAVE_EVERY).
        self._dirty_count: int = 0

        # Counts since last decay (triggers decay at DECAY_EVERY).
        self._retrieval_count_since_decay: int = 0

        # Reference to graph for creating CO_RETRIEVED edges.
        self._graph: MemoryGraph | None = graph

        # Path for auto-save (set by MemoryStore when wiring in Chunk 5).
        self._save_path: Path | None = None

    # ── Core tracking ─────────────────────────────────────────────────────────

    def record_retrieval(self, memory_ids: list[str]) -> None:
        """
        Record a retrieval event — call after every search that returns results.

        Updates:
        - Individual retrieval counts for each returned memory
        - Co-occurrence counts for every pair in the result set
        - Total retrieval counter
        - Triggers auto-save every SAVE_EVERY calls
        - Triggers decay every DECAY_EVERY calls
        - Creates/updates CO_RETRIEVED graph edges for strong pairs

        Args:
            memory_ids: List of memory IDs returned by the retrieval pipeline.
                        Order does not matter — all pairs are recorded.
                        Empty list or single-item list: only counts updated,
                        no pairs to record.
        """
        if not memory_ids:
            return

        # Update individual counts
        for mid in memory_ids:
            self._retrieval_counts[mid] += 1.0

        # Update co-occurrence counts for all pairs
        pairs_in_this_retrieval: list[tuple[str, str]] = []
        for i in range(len(memory_ids)):
            for j in range(i + 1, len(memory_ids)):
                pair = self._make_pair_key(memory_ids[i], memory_ids[j])
                self._co_counts[pair] += 1.0
                pairs_in_this_retrieval.append(pair)

        self._total_retrievals += 1.0
        self._dirty_count += 1
        self._retrieval_count_since_decay += 1

        # Check pairs for graph edge creation
        if self._graph is not None:
            self._update_graph_edges(pairs_in_this_retrieval)

        # Auto-decay
        if self._retrieval_count_since_decay >= self.DECAY_EVERY:
            self.apply_decay()
            self._retrieval_count_since_decay = 0

        # Auto-save
        if self._dirty_count >= self.SAVE_EVERY and self._save_path is not None:
            self.save(self._save_path)
            self._dirty_count = 0

    # ── PMI scoring ───────────────────────────────────────────────────────────

    def get_pmi_score(self, id_a: str, id_b: str) -> float:
        """
        Compute the Normalized PMI score between two memories.

        NPMI formula:
            P(a,b) = co_count(a,b) / total_retrievals
            P(a)   = count(a) / total_retrievals
            P(b)   = count(b) / total_retrievals
            PMI    = log2(P(a,b) / (P(a) * P(b)))
            NPMI   = PMI / -log2(P(a,b))

        NPMI range: -1 (never together) to 1 (always together).
        We return only positive values — negative means independent or
        negatively associated, which is not useful for boosting retrieval.

        Returns 0.0 if:
        - Either memory has never been retrieved
        - The pair has never co-occurred
        - Co-occurrence count is below MIN_CO_COUNT (noisy estimate)
        - Total retrievals is too low for reliable statistics (< 5)

        Args:
            id_a: First memory ID.
            id_b: Second memory ID.

        Returns:
            NPMI score between 0.0 and 1.0.
        """
        if self._total_retrievals < 5:
            return 0.0

        pair = self._make_pair_key(id_a, id_b)
        co_count = self._co_counts.get(pair, 0.0)

        if co_count < self.MIN_CO_COUNT:
            return 0.0

        count_a = self._retrieval_counts.get(id_a, 0.0)
        count_b = self._retrieval_counts.get(id_b, 0.0)

        if count_a == 0.0 or count_b == 0.0:
            return 0.0

        total = self._total_retrievals

        # Compute probabilities
        p_ab = co_count / total
        p_a = count_a / total
        p_b = count_b / total

        # Guard against log(0)
        if p_ab <= 0 or p_a <= 0 or p_b <= 0:
            return 0.0

        try:
            pmi = math.log2(p_ab / (p_a * p_b))
            # Normalize: divide by -log2(P(a,b))
            normalizer = -math.log2(p_ab)
            if normalizer <= 0:
                return 0.0
            npmi = pmi / normalizer
        except (ValueError, ZeroDivisionError):
            return 0.0

        # Clamp to [0, 1] — we only care about positive association
        return max(0.0, min(1.0, npmi))

    def get_associated_memories(
        self,
        memory_id: str,
        min_score: float = 0.1,
        max_results: int = 10,
    ) -> list[tuple[str, float]]:
        """
        Get memories most frequently co-retrieved with a given memory.

        Used by the UnifiedRetriever (Chunk 4) during the co-retrieval
        boost step and by the graph builder when creating CO_RETRIEVED edges.

        Args:
            memory_id: The memory to find associations for.
            min_score: Minimum NPMI score to include. Filters noise.
            max_results: Cap on returned results.

        Returns:
            List of (memory_id, npmi_score) sorted by score descending.
            Does not include the input memory_id.
        """
        results: list[tuple[str, float]] = []

        for pair, co_count in self._co_counts.items():
            if memory_id not in pair:
                continue
            if co_count < self.MIN_CO_COUNT:
                continue

            other_id = pair[0] if pair[1] == memory_id else pair[1]
            score = self.get_pmi_score(memory_id, other_id)

            if score >= min_score:
                results.append((other_id, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:max_results]

    def get_co_retrieval_boost(
        self,
        candidate_id: str,
        top_result_ids: list[str],
    ) -> float:
        """
        Compute the co-retrieval boost for a candidate memory.

        Called during retrieval scoring (Chunk 4, Step 4). Checks the
        candidate's NPMI score against each of the top vector results.
        Returns the maximum score found — if the candidate is strongly
        associated with any top result, it gets the full boost.

        Args:
            candidate_id: The memory being scored.
            top_result_ids: The top N vector search results (typically 5).

        Returns:
            Maximum NPMI score between candidate and any top result.
            Returns 0.0 if no association found.
        """
        if not top_result_ids:
            return 0.0

        max_score = 0.0
        for top_id in top_result_ids:
            if top_id == candidate_id:
                continue
            score = self.get_pmi_score(candidate_id, top_id)
            if score > max_score:
                max_score = score

        return max_score

    def should_create_graph_edge(self, id_a: str, id_b: str) -> bool:
        """
        Return True if this pair's NPMI score warrants a graph edge.

        Called after record_retrieval() to check newly-updated pairs.
        The caller (record_retrieval) already knows which pairs were
        involved in this retrieval batch.

        Args:
            id_a: First memory ID.
            id_b: Second memory ID.

        Returns:
            True if NPMI >= GRAPH_EDGE_THRESHOLD.
        """
        return self.get_pmi_score(id_a, id_b) >= self.GRAPH_EDGE_THRESHOLD

    # ── Maintenance ───────────────────────────────────────────────────────────

    def apply_decay(self) -> int:
        """
        Multiply all counts by DECAY_FACTOR and prune weak pairs.

        Called automatically every DECAY_EVERY retrievals.
        Can also be called manually for testing or maintenance.

        Decay keeps the tracker from being dominated by old patterns.
        Pruning keeps memory usage bounded.

        Returns:
            Number of pairs pruned from co_counts.
        """
        # Decay all co-occurrence counts
        to_prune: list[tuple[str, str]] = []
        for pair in list(self._co_counts.keys()):
            self._co_counts[pair] *= self.DECAY_FACTOR
            if self._co_counts[pair] < self.PRUNE_THRESHOLD:
                to_prune.append(pair)

        # Prune weak pairs
        for pair in to_prune:
            del self._co_counts[pair]

        # Decay individual retrieval counts
        for mid in list(self._retrieval_counts.keys()):
            self._retrieval_counts[mid] *= self.DECAY_FACTOR
            # Prune memories that have essentially faded to zero
            if self._retrieval_counts[mid] < 0.1:
                del self._retrieval_counts[mid]

        # Decay total (keeps ratios consistent)
        self._total_retrievals *= self.DECAY_FACTOR

        return len(to_prune)

    def purge_memory(self, memory_id: str) -> None:
        """
        Remove all tracking data for an archived memory.

        Called by MemoryStore.forget() after archiving. Ensures archived
        memories do not continue to inflate NPMI scores or appear in
        get_associated_memories() results.

        Args:
            memory_id: The ID of the memory being archived.
        """
        # Remove from individual counts
        self._retrieval_counts.pop(memory_id, None)

        # Remove all pairs involving this memory
        pairs_to_remove = [
            pair for pair in self._co_counts
            if memory_id in pair
        ]
        for pair in pairs_to_remove:
            del self._co_counts[pair]

    # ── Persistence ───────────────────────────────────────────────────────────

    def set_save_path(self, filepath: str | Path) -> None:
        """
        Set the path for auto-save.

        Called by MemoryStore.initialize() (Chunk 5) after loading.
        Once set, the tracker auto-saves every SAVE_EVERY retrievals.

        Args:
            filepath: Path to co_retrieval.json.
        """
        self._save_path = Path(filepath)

    def save(self, filepath: str | Path | None = None) -> None:
        """
        Persist tracker state to JSON.

        Format:
        {
            "version": 1,
            "total_retrievals": 234.5,
            "retrieval_counts": {"mem_abc": 12.3, ...},
            "co_counts": {"mem_abc|mem_def": 5.7, ...}
        }

        Pair keys use "|" as separator (IDs use "_" and hex, no "|" possible).

        Args:
            filepath: Path to write. Uses self._save_path if None.
                      Creates parent directories if needed.
        """
        target = Path(filepath) if filepath else self._save_path
        if target is None:
            return

        target.parent.mkdir(parents=True, exist_ok=True)

        # Serialize pair tuple keys to "id_a|id_b" strings
        co_counts_serialized = {
            f"{pair[0]}|{pair[1]}": round(count, 6)
            for pair, count in self._co_counts.items()
            if count >= self.PRUNE_THRESHOLD  # Only save meaningful counts
        }

        retrieval_counts_serialized = {
            mid: round(count, 6)
            for mid, count in self._retrieval_counts.items()
            if count >= 0.1
        }

        data = {
            "version":          1,
            "total_retrievals": round(self._total_retrievals, 6),
            "retrieval_counts": retrieval_counts_serialized,
            "co_counts":        co_counts_serialized,
        }

        target.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

    def load(self, filepath: str | Path) -> bool:
        """
        Load tracker state from JSON.

        Called at startup before any retrievals occur.
        Safe to call on a missing file — returns False and leaves
        tracker in empty state.

        Args:
            filepath: Path to co_retrieval.json.

        Returns:
            True if loaded successfully. False if file missing or corrupt.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            return False

        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False

        try:
            self._total_retrievals = float(
                data.get("total_retrievals", 0.0)
            )

            self._retrieval_counts = defaultdict(float, {
                mid: float(count)
                for mid, count in data.get("retrieval_counts", {}).items()
            })

            # Deserialize "id_a|id_b" keys back to tuple pairs
            self._co_counts = defaultdict(float)
            for key, count in data.get("co_counts", {}).items():
                parts = key.split("|")
                if len(parts) == 2:
                    pair = self._make_pair_key(parts[0], parts[1])
                    self._co_counts[pair] = float(count)

        except (ValueError, KeyError, TypeError):
            # Corrupt data — reset to empty rather than crash
            self._co_counts = defaultdict(float)
            self._retrieval_counts = defaultdict(float)
            self._total_retrievals = 0.0
            return False

        return True

    # ── Stats & diagnostics ───────────────────────────────────────────────────

    def stats(self) -> dict:
        """
        Return diagnostic information about the tracker.

        Used by MemoryStore.stats() to include co-retrieval health
        in the overall system stats report.

        Returns:
            Dict with retrieval counts, pair counts, and top associations.
        """
        # Find top co-retrieved pairs by raw count
        top_pairs = sorted(
            self._co_counts.items(),
            key=lambda x: x[1],
            reverse=True,
        )[:5]

        top_pairs_display = [
            {
                "pair":     f"{p[0]}...{p[1][-6:]}",
                "co_count": round(count, 2),
                "npmi":     round(self.get_pmi_score(p[0], p[1]), 3),
            }
            for p, count in top_pairs
        ]

        # Count pairs with strong NPMI (would have graph edges)
        strong_pairs = sum(
            1 for pair in self._co_counts
            if self.get_pmi_score(pair[0], pair[1]) >= self.GRAPH_EDGE_THRESHOLD
        )

        return {
            "total_retrievals":       int(self._total_retrievals),
            "tracked_memories":       len(self._retrieval_counts),
            "tracked_pairs":          len(self._co_counts),
            "strong_associations":    strong_pairs,
            "top_pairs":              top_pairs_display,
        }

    def debug_scores(self, memory_ids: list[str]) -> dict[str, float]:
        """
        Return all NPMI scores between a list of memory IDs.

        Used during development to inspect learned associations.
        Only call in development — O(n²) for the input list.

        Args:
            memory_ids: List of memory IDs to compare pairwise.

        Returns:
            Dict mapping "id_a|id_b" to NPMI score.
        """
        scores: dict[str, float] = {}
        for i in range(len(memory_ids)):
            for j in range(i + 1, len(memory_ids)):
                key = f"{memory_ids[i]}|{memory_ids[j]}"
                scores[key] = self.get_pmi_score(memory_ids[i], memory_ids[j])
        return scores

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_pair_key(self, id_a: str, id_b: str) -> tuple[str, str]:
        """
        Create a canonical pair key with the smaller ID always first.

        This ensures (mem_abc, mem_def) and (mem_def, mem_abc) map to
        the same dict entry regardless of the order they appear in a
        retrieval result.

        Args:
            id_a: First memory ID.
            id_b: Second memory ID.

        Returns:
            Tuple (smaller_id, larger_id) by lexicographic comparison.
        """
        if id_a <= id_b:
            return (id_a, id_b)
        return (id_b, id_a)

    def _update_graph_edges(
        self,
        pairs: list[tuple[str, str]],
    ) -> None:
        """
        Check recently-updated pairs and create/update graph edges.

        Called internally after record_retrieval() updates counts.
        Only evaluates pairs that were part of the current retrieval
        batch — not all known pairs.

        Args:
            pairs: Canonical pair tuples from the current retrieval.
        """
        if self._graph is None:
            return

        from lace.retrieval.graph import EdgeType

        for pair in pairs:
            npmi = self.get_pmi_score(pair[0], pair[1])
            if npmi >= self.GRAPH_EDGE_THRESHOLD:
                self._graph.add_edge(
                    source=pair[0],
                    target=pair[1],
                    edge_type=EdgeType.CO_RETRIEVED,
                    weight=npmi,
                    bidirectional=True,
                )
