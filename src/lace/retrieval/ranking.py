"""Multi-signal ranking for retrieved memory candidates.

Every candidate memory gets a composite relevance score combining:
  - Semantic similarity  (40%) — how similar is the content to the query?
  - Recency             (20%) — was this memory accessed recently?
  - Frequency           (15%) — is this memory accessed often?
  - Confidence          (15%) — how reliable is this memory?
  - Scope bonus         (10%) — is this memory scoped to the active project?

The final score is between 0.0 and 1.0.
Memories below RELEVANCE_THRESHOLD are filtered out.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from dataclasses import dataclass

from lace.memory.models import MemoryObject, RetrievalResult


# ── Constants (match lace.yaml defaults) ──────────────────────────────────────

HALF_LIFE_DAYS = 30
MAX_EXPECTED_COUNT = 100
RELEVANCE_THRESHOLD = 0.35
MAX_RESULTS = 20


# ── Weight configuration ──────────────────────────────────────────────────────

@dataclass
class RankingWeights:
    semantic_similarity: float = 0.40
    recency: float             = 0.20
    frequency: float           = 0.15
    confidence: float          = 0.15
    scope: float               = 0.10

    def validate(self) -> None:
        total = (
            self.semantic_similarity
            + self.recency
            + self.frequency
            + self.confidence
            + self.scope
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Ranking weights must sum to 1.0, got {total:.3f}")


# ── Individual signal functions ───────────────────────────────────────────────

def semantic_score(distance: float) -> float:
    """Convert ChromaDB cosine distance to a similarity score [0, 1].

    ChromaDB cosine distance = 1 - cosine_similarity.
    So distance 0.0 = identical, distance 2.0 = opposite.
    We convert to similarity: score = 1 - (distance / 2).
    """
    return max(0.0, 1.0 - (distance / 2.0))


def recency_score(last_accessed: datetime, half_life_days: int = HALF_LIFE_DAYS) -> float:
    """Exponential decay based on days since last access.

    Score = 0.5 ^ (days_elapsed / half_life)
    - Accessed today → 1.0
    - Accessed half_life_days ago → 0.5
    - Accessed 2*half_life_days ago → 0.25
    """
    now = datetime.now(timezone.utc)

    # Handle naive datetimes
    if last_accessed.tzinfo is None:
        last_accessed = last_accessed.replace(tzinfo=timezone.utc)

    days_elapsed = max(0, (now - last_accessed).days)
    return 0.5 ** (days_elapsed / half_life_days)


def frequency_score(access_count: int, max_count: int = MAX_EXPECTED_COUNT) -> float:
    """Log-scaled frequency score [0, 1].

    Uses log scale so the difference between 1 and 10 accesses
    matters more than the difference between 90 and 100.
    """
    return math.log(access_count + 1) / math.log(max_count + 1)


def scope_score(memory_scope: str, active_scope: str) -> float:
    """Score based on how well the memory scope matches the active context.

    Session scope → 1.0  (most relevant)
    Project scope → 0.8  (relevant to current project)
    Global scope  → 0.5  (generally useful)
    Other project → 0.2  (probably not relevant)
    """
    if memory_scope == active_scope:
        return 1.0
    elif memory_scope == "global":
        return 0.5
    elif active_scope != "global" and memory_scope == "global":
        return 0.5
    elif memory_scope.startswith("project:") and active_scope.startswith("project:"):
        # Different project
        return 0.2
    else:
        return 0.3


# ── Main ranking function ─────────────────────────────────────────────────────

def compute_relevance_score(
    memory: MemoryObject,
    distance: float,
    active_scope: str = "global",
    weights: RankingWeights | None = None,
    half_life_days: int = HALF_LIFE_DAYS,
) -> float:
    """Compute the composite relevance score for a memory candidate.

    Args:
        memory: The memory object being scored.
        distance: ChromaDB cosine distance from the query.
        active_scope: The current project/session scope.
        weights: Custom weight configuration (uses defaults if None).
        half_life_days: Recency decay rate.

    Returns:
        A float between 0.0 and 1.0.
    """
    if weights is None:
        weights = RankingWeights()

    sem   = semantic_score(distance)
    rec   = recency_score(memory.last_accessed, half_life_days)
    freq  = frequency_score(memory.access_count)
    conf  = memory.confidence
    scope = scope_score(memory.project_scope, active_scope)

    return (
        weights.semantic_similarity * sem  +
        weights.recency             * rec  +
        weights.frequency           * freq +
        weights.confidence          * conf +
        weights.scope               * scope
    )


def rank_candidates(
    candidates: list[tuple[MemoryObject, float]],
    active_scope: str = "global",
    weights: RankingWeights | None = None,
    threshold: float = RELEVANCE_THRESHOLD,
    max_results: int = MAX_RESULTS,
    half_life_days: int = HALF_LIFE_DAYS,
    min_semantic_score: float = 0.45,   # ← ADD THIS
) -> list[RetrievalResult]:
    """Rank candidates and filter by threshold."""
    scored: list[RetrievalResult] = []

    for memory, distance in candidates:
        # HARD GATE: distance > 0.80 means semantically irrelevant
        # Raw distances: relevant=0.27-0.77, irrelevant=0.91-0.95
        # This gate runs BEFORE composite scoring
        if distance > 0.80:
            continue

        score = compute_relevance_score(
            memory=memory,
            distance=distance,
            active_scope=active_scope,
            weights=weights,
            half_life_days=half_life_days,
        )

        if score >= threshold:
            scored.append(RetrievalResult(
                memory=memory,
                relevance_score=round(score, 4),
                match_type="vector",
                rank=0,
            ))

    scored.sort(key=lambda r: r.relevance_score, reverse=True)
    results = scored[:max_results]
    for i, result in enumerate(results):
        result.rank = i + 1

    return results


