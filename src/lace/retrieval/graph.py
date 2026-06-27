"""
Memory relationship graph for LACE.

Builds and maintains a directed weighted graph over memory IDs.
Nodes are memory IDs. Edges represent discovered or explicit relationships.

This file has no side effects. It does not modify memories, call ChromaDB,
or interact with the tag index directly (the caller passes tag index data in).

Edge types:
    EXPLICIT      — from related_ids field. Weight 1.0. Strongest signal.
    TAG_SIMILARITY — auto-created from shared tags. Weight = Jaccard score.
    CO_RETRIEVED  — created by co-occurrence tracker. Weight = NPMI score.
    TEMPORAL      — memories created within 30 minutes. Weight <= 0.3.
    SUPERSEDES    — directional. Newer memory points to older on same topic.

Lifecycle:
    1. MemoryGraph() — create empty graph
    2. load(filepath) — restore from JSON if file exists (startup)
    3. build_from_memories(memories, tag_index) — fill in missing edges
    4. save(filepath) — persist current state
    5. add_edge() — incremental updates when memories are stored
    6. remove_node() — incremental updates when memories are archived
    7. get_neighbors() — used during retrieval by UnifiedRetriever
"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lace.memory.models import MemoryObject
    from lace.retrieval.tag_index import TagIndex


# ── Edge types ────────────────────────────────────────────────────────────────

class EdgeType(Enum):
    """
    The five relationship types between memories.

    Ordered here from strongest to weakest signal:
    EXPLICIT > TAG_SIMILARITY > CO_RETRIEVED > TEMPORAL > SUPERSEDES

    SUPERSEDES is not weaker in terms of confidence — it is directional
    and serves a different purpose (preference, not similarity).
    """
    EXPLICIT        = "explicit"
    TAG_SIMILARITY  = "tag_similarity"
    CO_RETRIEVED    = "co_retrieved"
    TEMPORAL        = "temporal"
    SUPERSEDES      = "supersedes"


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Edge:
    """
    A single directed relationship between two memories.

    All edges are stored as directed. Bidirectional relationships
    are represented as two Edge instances pointing in each direction.
    """
    source: str       # memory ID
    target: str       # memory ID
    edge_type: EdgeType
    weight: float     # 0.0 to 1.0

    def to_dict(self) -> dict:
        """Serialize to JSON-safe dict."""
        return {
            "source":    self.source,
            "target":    self.target,
            "type":      self.edge_type.value,
            "weight":    round(self.weight, 6),
        }

    @classmethod
    def from_dict(cls, data: dict) -> Edge:
        """Deserialize from JSON dict."""
        return cls(
            source=data["source"],
            target=data["target"],
            edge_type=EdgeType(data["type"]),
            weight=float(data["weight"]),
        )


# ── Main graph class ──────────────────────────────────────────────────────────

class MemoryGraph:
    """
    Directed weighted graph over memory IDs.

    Storage: adjacency dict of dicts.
        self._adj[source_id][target_id] = Edge

    This stores directed edges. For bidirectional relationships,
    both directions are stored:
        _adj["mem_a"]["mem_b"] = Edge(a→b)
        _adj["mem_b"]["mem_a"] = Edge(b→a)

    All lookups are O(1). Removal is O(degree of node).
    Memory usage: ~200 bytes per edge, scales to tens of thousands.
    """

    # Minimum Jaccard score to create a TAG_SIMILARITY edge
    MIN_TAG_JACCARD: float = 0.2

    # Minimum number of shared tags to create a TAG_SIMILARITY edge
    MIN_SHARED_TAGS: int = 2

    # Temporal window for TEMPORAL edges
    TEMPORAL_WINDOW_MINUTES: int = 30

    # Maximum weight for TEMPORAL edges (deliberately weak signal)
    MAX_TEMPORAL_WEIGHT: float = 0.3

    # Weight decay factor per hop during BFS traversal
    DEPTH_DECAY: float = 0.5

    def __init__(self) -> None:
        # adjacency[source_id][target_id] = Edge
        self._adj: dict[str, dict[str, Edge]] = defaultdict(dict)

    # ── Edge operations ───────────────────────────────────────────────────────

    def add_edge(
        self,
        source: str,
        target: str,
        edge_type: EdgeType,
        weight: float,
        bidirectional: bool = True,
    ) -> None:
        """
        Add or update an edge between two memory IDs.

        If an edge of the same type between the same pair already exists,
        the weight is updated to the maximum of old and new. This prevents
        repeated build passes from weakening existing strong edges.

        Bidirectional edges are stored as two directed edges. The weight
        is the same in both directions.

        Args:
            source: Source memory ID.
            target: Target memory ID.
            edge_type: The type of relationship.
            weight: Strength of the relationship, 0.0 to 1.0.
            bidirectional: If True, add edge in both directions.
                           Set to False for SUPERSEDES edges.
        """
        if source == target:
            return  # No self-loops

        weight = max(0.0, min(1.0, weight))  # Clamp to [0, 1]

        self._set_edge(source, target, edge_type, weight)
        if bidirectional:
            self._set_edge(target, source, edge_type, weight)

    def remove_node(self, memory_id: str) -> None:
        """
        Remove a memory node and all edges connected to it.

        Called when a memory is archived. After this call, the memory ID
        will not appear in any get_neighbors() result.

        Removes:
        - All outgoing edges from memory_id
        - All incoming edges to memory_id (by scanning neighbors' entries)

        Args:
            memory_id: The memory ID being archived.
        """
        # Find all nodes that point to this one
        targets_of_source = list(self._adj.get(memory_id, {}).keys())

        # Remove outgoing edges (delete the source entry entirely)
        self._adj.pop(memory_id, None)

        # Remove incoming edges from all neighbors
        for neighbor_id in targets_of_source:
            self._adj[neighbor_id].pop(memory_id, None)
            # Clean up empty adjacency entries
            if not self._adj[neighbor_id]:
                del self._adj[neighbor_id]

    def has_edge(self, source: str, target: str) -> bool:
        """Check if any edge exists from source to target."""
        return target in self._adj.get(source, {})

    def get_edge(self, source: str, target: str) -> Edge | None:
        """Return the edge from source to target, or None."""
        return self._adj.get(source, {}).get(target)

    # ── Traversal ─────────────────────────────────────────────────────────────

    def get_neighbors(
        self,
        memory_id: str,
        max_depth: int = 2,
        edge_types: set[EdgeType] | None = None,
        min_weight: float = 0.1,
    ) -> list[tuple[str, float, int]]:
        """
        BFS traversal from a memory node.

        Starting from memory_id, traverse the graph up to max_depth hops.
        Weight decays by DEPTH_DECAY (0.5) per hop:
            depth 1: weight * 1.0
            depth 2: weight * 0.5
            depth 3: weight * 0.25

        If a node is reachable via multiple paths, the highest accumulated
        weight is kept. The starting node is never included in results.

        Args:
            memory_id: Starting node for traversal.
            max_depth: Maximum hops from the starting node. Default 2.
            edge_types: If provided, only follow edges of these types.
                        If None, follow all edge types.
            min_weight: Skip edges below this weight during traversal.
                        Prevents very weak edges from adding noise.

        Returns:
            List of (memory_id, accumulated_weight, depth) tuples.
            Sorted by accumulated_weight descending.
            Does not include the starting memory_id.
        """
        if memory_id not in self._adj and memory_id not in self._get_all_targets():
            return []

        # best_weight[node] = highest weight path found so far
        best_weight: dict[str, float] = {}
        best_depth: dict[str, int] = {}

        # BFS queue: (current_id, accumulated_weight, current_depth)
        queue: deque[tuple[str, float, int]] = deque()
        queue.append((memory_id, 1.0, 0))
        visited_at_depth: dict[str, int] = {memory_id: 0}

        while queue:
            current_id, current_weight, depth = queue.popleft()

            if depth >= max_depth:
                continue

            for neighbor_id, edge in self._adj.get(current_id, {}).items():
                # Filter by edge type if specified
                if edge_types and edge.edge_type not in edge_types:
                    continue

                # Skip weak edges
                if edge.weight < min_weight:
                    continue

                # Compute propagated weight with depth decay
                propagated = current_weight * edge.weight * self.DEPTH_DECAY

                next_depth = depth + 1

                # Only process if we found a better path or haven't visited
                if (
                    neighbor_id not in visited_at_depth
                    or propagated > best_weight.get(neighbor_id, 0)
                ):
                    visited_at_depth[neighbor_id] = next_depth
                    best_weight[neighbor_id] = propagated
                    best_depth[neighbor_id] = next_depth

                    if next_depth < max_depth:
                        queue.append((neighbor_id, propagated, next_depth))

        # Build result list — exclude the starting node
        results = [
            (nid, weight, best_depth[nid])
            for nid, weight in best_weight.items()
            if nid != memory_id
        ]
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    # ── Batch build ───────────────────────────────────────────────────────────

    def build_from_memories(
        self,
        memories: list[MemoryObject],
        tag_index: TagIndex | None = None,
    ) -> dict[str, int]:
        """
        Discover and create edges from a list of memories.

        Runs three passes in order. Each pass only creates edges that do
        not already exist — safe to call multiple times on the same data.
        Called at startup after load() to fill in any missing edges.

        Pass 1 — EXPLICIT edges from related_ids field.
        Pass 2 — TAG_SIMILARITY edges from shared tags (requires tag_index).
        Pass 3 — TEMPORAL edges from creation timestamps.

        Args:
            memories: All active memories from the vault.
            tag_index: Built TagIndex from Chunk 1. Required for Pass 2.
                       If None, Pass 2 is skipped.

        Returns:
            Dict of counts: {"explicit": N, "tag_similarity": N, "temporal": N}
        """
        counts = {"explicit": 0, "tag_similarity": 0, "temporal": 0}

        # ── Pass 1: Explicit edges from related_ids ───────────────────────────
        memory_id_set = {m.id for m in memories}  # For validating related_ids

        for memory in memories:
            for related_id in memory.related_ids:
                # Only link to memories that actually exist in the vault
                if related_id not in memory_id_set:
                    continue
                if self.has_edge(memory.id, related_id):
                    continue  # Already exists — skip

                self.add_edge(
                    source=memory.id,
                    target=related_id,
                    edge_type=EdgeType.EXPLICIT,
                    weight=1.0,
                    bidirectional=True,
                )
                counts["explicit"] += 1

        # ── Pass 2: Tag similarity edges ──────────────────────────────────────
        if tag_index is not None:
            counts["tag_similarity"] = self._build_tag_similarity_edges(
                memories, tag_index
            )

        # ── Pass 3: Temporal edges ────────────────────────────────────────────
        counts["temporal"] = self._build_temporal_edges(memories)

        return counts

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, filepath: str | Path) -> None:
        """
        Save the graph to a JSON file.

        Only unique edges are saved — bidirectional edges are stored once
        (as source < target alphabetically) and reconstructed on load.
        This halves the file size for bidirectional relationships.

        Args:
            filepath: Path to write graph.json. Created if missing.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Collect unique edges (deduplicate bidirectional pairs)
        seen: set[tuple[str, str, str]] = set()
        edges_to_save: list[dict] = []

        for source_id, targets in self._adj.items():
            for target_id, edge in targets.items():
                # Canonical key: smaller ID first to deduplicate bidirectional
                if edge.edge_type == EdgeType.SUPERSEDES:
                    # SUPERSEDES is directional — always save as-is
                    canonical_key = (source_id, target_id, edge.edge_type.value)
                else:
                    a, b = sorted([source_id, target_id])
                    canonical_key = (a, b, edge.edge_type.value)

                if canonical_key not in seen:
                    seen.add(canonical_key)
                    edges_to_save.append(edge.to_dict())

        data = {
            "version": 1,
            "edge_count": len(edges_to_save),
            "edges": edges_to_save,
        }

        filepath.write_text(
            json.dumps(data, indent=2),
            encoding="utf-8",
        )

    def load(self, filepath: str | Path) -> int:
        """
        Load graph edges from a JSON file.

        Reconstructs bidirectional edges from the stored single representation.
        Edges are added on top of any existing edges — does not clear the graph
        first. This means load() + build_from_memories() is additive and safe.

        Args:
            filepath: Path to graph.json.

        Returns:
            Number of edges loaded. 0 if file does not exist.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            return 0

        try:
            data = json.loads(filepath.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return 0

        loaded = 0
        for edge_data in data.get("edges", []):
            try:
                edge = Edge.from_dict(edge_data)
            except (KeyError, ValueError):
                continue  # Skip malformed edges

            bidirectional = edge.edge_type != EdgeType.SUPERSEDES

            self.add_edge(
                source=edge.source,
                target=edge.target,
                edge_type=edge.edge_type,
                weight=edge.weight,
                bidirectional=bidirectional,
            )
            loaded += 1

        return loaded

    # ── Stats ─────────────────────────────────────────────────────────────────

    def node_count(self) -> int:
        """Number of memory nodes in the graph."""
        all_nodes: set[str] = set()
        for source, targets in self._adj.items():
            all_nodes.add(source)
            all_nodes.update(targets.keys())
        return len(all_nodes)

    def edge_count(self) -> int:
        """
        Number of unique undirected edges.

        Bidirectional edges count as one. SUPERSEDES edges count as one
        directed edge.
        """
        seen: set[tuple[str, str, str]] = set()
        for source, targets in self._adj.items():
            for target, edge in targets.items():
                if edge.edge_type == EdgeType.SUPERSEDES:
                    seen.add((source, target, edge.edge_type.value))
                else:
                    a, b = sorted([source, target])
                    seen.add((a, b, edge.edge_type.value))
        return len(seen)

    def stats(self) -> dict:
        """
        Return diagnostic information about the graph.

        Used by MemoryStore.stats() to include graph health in the
        overall system stats report.

        Returns:
            Dict with node count, edge count, and breakdown by edge type.
        """
        type_counts: dict[str, int] = defaultdict(int)
        seen: set[tuple[str, str, str]] = set()

        for source, targets in self._adj.items():
            for target, edge in targets.items():
                if edge.edge_type == EdgeType.SUPERSEDES:
                    key = (source, target, edge.edge_type.value)
                else:
                    a, b = sorted([source, target])
                    key = (a, b, edge.edge_type.value)

                if key not in seen:
                    seen.add(key)
                    type_counts[edge.edge_type.value] += 1

        return {
            "node_count":  self.node_count(),
            "edge_count":  self.edge_count(),
            "by_type":     dict(type_counts),
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _set_edge(
        self,
        source: str,
        target: str,
        edge_type: EdgeType,
        weight: float,
    ) -> None:
        """
        Internal: set a single directed edge, taking max weight on collision.
        """
        existing = self._adj[source].get(target)
        if existing is not None and existing.edge_type == edge_type:
            # Same type — keep the stronger weight
            if weight <= existing.weight:
                return
        self._adj[source][target] = Edge(source, target, edge_type, weight)

    def _get_all_targets(self) -> set[str]:
        """Return all node IDs that appear as targets (have incoming edges)."""
        targets: set[str] = set()
        for neighbors in self._adj.values():
            targets.update(neighbors.keys())
        return targets

    def _build_tag_similarity_edges(
        self,
        memories: list[MemoryObject],
        tag_index: TagIndex,
    ) -> int:
        """
        Pass 2: create TAG_SIMILARITY edges between memories sharing tags.

        Uses the tag index to efficiently find candidate pairs —
        only compares memories that share at least one tag bucket.
        This avoids the O(n²) full pairwise comparison.

        Threshold: Jaccard >= MIN_TAG_JACCARD AND shared_count >= MIN_SHARED_TAGS.

        Returns:
            Number of new edges created (counting bidirectional as one).
        """
        count = 0

        # Group memories by tag to find candidate pairs efficiently
        # tag → list of memory IDs that have this tag
        tag_to_ids: dict[str, list[str]] = defaultdict(list)
        for memory in memories:
            for tag in memory.tags:
                tag_to_ids[tag.lower().strip()].append(memory.id)

        # Find all pairs that share at least one tag
        already_checked: set[tuple[str, str]] = set()

        for tag, ids_with_tag in tag_to_ids.items():
            if len(ids_with_tag) < 2:
                continue  # Only one memory has this tag — no pairs

            for i in range(len(ids_with_tag)):
                for j in range(i + 1, len(ids_with_tag)):
                    id_a = ids_with_tag[i]
                    id_b = ids_with_tag[j]

                    # Canonical pair key to avoid checking same pair twice
                    pair_key = (min(id_a, id_b), max(id_a, id_b))
                    if pair_key in already_checked:
                        continue
                    already_checked.add(pair_key)

                    # Check if edge already exists
                    if self.has_edge(id_a, id_b):
                        continue

                    # Require minimum number of shared tags
                    shared = tag_index.shared_tags(id_a, id_b)
                    if len(shared) < self.MIN_SHARED_TAGS:
                        continue

                    # Require minimum Jaccard similarity
                    jaccard = tag_index.jaccard_similarity(id_a, id_b)
                    if jaccard < self.MIN_TAG_JACCARD:
                        continue

                    self.add_edge(
                        source=id_a,
                        target=id_b,
                        edge_type=EdgeType.TAG_SIMILARITY,
                        weight=jaccard,
                        bidirectional=True,
                    )
                    count += 1

        return count

    def _build_temporal_edges(
        self,
        memories: list[MemoryObject],
    ) -> int:
        """
        Pass 3: create TEMPORAL edges between memories created close in time.

        Sorts memories by created_at, then uses a sliding window of
        TEMPORAL_WINDOW_MINUTES to find pairs. Weight is proportional
        to how close in time — capped at MAX_TEMPORAL_WEIGHT (0.3).

        Returns:
            Number of new edges created (counting bidirectional as one).
        """
        if len(memories) < 2:
            return 0

        window = timedelta(minutes=self.TEMPORAL_WINDOW_MINUTES)
        count = 0

        # Sort by creation time
        sorted_mems = sorted(memories, key=lambda m: m.created_at)

        for i, mem_a in enumerate(sorted_mems):
            # Only look forward within the window
            for mem_b in sorted_mems[i + 1:]:
                # Ensure both datetimes are timezone-aware for subtraction
                a_time = mem_a.created_at
                b_time = mem_b.created_at

                if a_time.tzinfo is None:
                    a_time = a_time.replace(tzinfo=timezone.utc)
                if b_time.tzinfo is None:
                    b_time = b_time.replace(tzinfo=timezone.utc)

                delta = b_time - a_time

                # Stop scanning once outside the window
                if delta > window:
                    break

                # Skip if edge already exists
                if self.has_edge(mem_a.id, mem_b.id):
                    continue

                # Weight: linear decay within window, capped at MAX_TEMPORAL_WEIGHT
                closeness = 1.0 - (
                    delta.total_seconds()
                    / window.total_seconds()
                )
                weight = closeness * self.MAX_TEMPORAL_WEIGHT

                if weight < 0.05:
                    continue  # Too weak to be worth storing

                self.add_edge(
                    source=mem_a.id,
                    target=mem_b.id,
                    edge_type=EdgeType.TEMPORAL,
                    weight=weight,
                    bidirectional=True,
                )
                count += 1

        return count
