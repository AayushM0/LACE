"""Deduplication logic for LACE memory system.

Before storing a new memory, we check if it's too similar
to an existing one. Three outcomes:

  - STORE  → novel content, add it
  - MERGE  → very similar, update existing
  - SKIP   → nearly identical, discard

Thresholds (configurable):
  > 0.95 cosine similarity → SKIP  (nearly identical)
  > 0.85 cosine similarity → MERGE (combine into one)
  ≤ 0.85                  → STORE (novel)
"""

from __future__ import annotations

import logging
import sqlite3
from enum import Enum
from pathlib import Path
from typing import Optional, TYPE_CHECKING
from datetime import datetime, timezone

from lace.core.config import LaceConfig, resolve_lace_paths
from lace.memory.models import MemoryObject, MemoryLifecycle, MemoryCategory
from lace.retrieval.embeddings import embed_text
from lace.memory.normalize import canonical_hash
from lace.memory.pipeline_log import (
    PIPELINE_LOG_DB_PATH,
    log_dedup_action as log_dedup_event,
)

if TYPE_CHECKING:
    from lace.memory.store import MemoryStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Backward Compatibility Enum and Classes
# ---------------------------------------------------------------------------

class DedupAction(str, Enum):
    STORE = "store"   # Novel — add it
    MERGE = "merge"   # Similar — update existing
    SKIP  = "skip"    # Duplicate — discard


from dataclasses import dataclass


@dataclass
class DedupResult:
    action:          DedupAction
    candidate:       MemoryObject
    existing:        MemoryObject | None = None
    similarity:      float               = 0.0
    reason:          str                 = ""

# ---------------------------------------------------------------------------
# Cosine Similarity & Legacy dedup check
# ---------------------------------------------------------------------------

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(vec_a) != len(vec_b):
        logger.warning(
            f"[Dedup] Vector length mismatch: {len(vec_a)} vs {len(vec_b)}. "
            "Returning 0.0"
        )
        return 0.0

    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = sum(a ** 2 for a in vec_a) ** 0.5
    mag_b = sum(b ** 2 for b in vec_b) ** 0.5

    if mag_a == 0.0 or mag_b == 0.0:
        return 0.0

    return dot / (mag_a * mag_b)


def check_duplicate(
    candidate: MemoryObject,
    existing_memories: list[MemoryObject],
    skip_threshold:  float = 0.95,
    merge_threshold: float = 0.85,
) -> DedupResult:
    if candidate.embedding is None:
        return DedupResult(
            action=DedupAction.STORE,
            candidate=candidate,
            reason="no embedding available for comparison",
        )

    best_similarity = 0.0
    best_match: MemoryObject | None = None

    for existing in existing_memories:
        # Explicitly skip archived memories — they are dead
        if existing.lifecycle.value == "archived":
            continue
        if existing.embedding is None:
            continue
        if existing.category != candidate.category:
            continue

        sim = cosine_similarity(candidate.embedding, existing.embedding)
        if sim > best_similarity:
            best_similarity = sim
            best_match = existing

    if best_similarity > skip_threshold:
        return DedupResult(
            action=DedupAction.SKIP,
            candidate=candidate,
            existing=best_match,
            similarity=best_similarity,
            reason=f"nearly identical to existing memory (sim={best_similarity:.3f})",
        )

    if best_similarity > merge_threshold:
        return DedupResult(
            action=DedupAction.MERGE,
            candidate=candidate,
            existing=best_match,
            similarity=best_similarity,
            reason=f"very similar to existing memory (sim={best_similarity:.3f})",
        )

    return DedupResult(
        action=DedupAction.STORE,
        candidate=candidate,
        existing=best_match,
        similarity=best_similarity,
        reason="novel content",
    )


def merge_memories(
    existing: MemoryObject,
    candidate: MemoryObject,
) -> MemoryObject:
    """Merge candidate into existing memory."""
    merged_tags = list(set(existing.tags + candidate.tags))

    if candidate.content.strip() not in existing.content:
        existing.content = (
            existing.content.rstrip()
            + "\n\n"
            + candidate.content.strip()
        )

    existing.tags = merged_tags
    existing.confidence = min(1.0, existing.confidence + 0.05)
    existing.last_accessed = datetime.now(timezone.utc)
    existing.metadata["merged_from"] = existing.metadata.get(
        "merged_from", []
    ) + [candidate.id]

    return existing


# ---------------------------------------------------------------------------
# CHUNK 4 — Two-Tier Deduplication Additions
# ---------------------------------------------------------------------------

VAULT_HASH_INDEX_PATH = resolve_lace_paths()["hash_index"]

# ---------------------------------------------------------------------------
# VectorIndex interface and production adapter
# ---------------------------------------------------------------------------

from dataclasses import dataclass as _dc


@_dc
class VectorSearchResult:
    """Minimal result type returned by VectorIndex.query()."""
    memory: "MemoryObject"
    distance: float


class VectorIndex:
    """Duck-typed interface expected by dedup_and_store().

    Production callers should use StoreBackedVectorIndex.
    Tests may use MagicMock() with .query.return_value = [].
    """
    def query(self, embedding, n_results: int, scope_filter=None):
        """Return list[VectorSearchResult], nearest neighbours first."""
        raise NotImplementedError

    def add(self, memory: "MemoryObject", embedding: list[float]) -> None:
        """Persist a new memory+embedding to the vector store."""
        raise NotImplementedError


class StoreBackedVectorIndex(VectorIndex):
    """Production VectorIndex that reads/writes ChromaDB via store's vector_db_path.

    This is the canonical adapter for use in _process_single_job and any
    production caller of dedup_and_store().  Do not use MagicMock() in
    production code — tests should patch embed_text instead.
    """

    def __init__(self, vector_db_path: Path) -> None:
        self._vector_db_path = vector_db_path

    def query(self, embedding, n_results: int, scope_filter=None):
        from lace.retrieval.vector import get_client, _scope_to_collection_name
        client = get_client(self._vector_db_path)
        collections = {c.name: c for c in client.list_collections()}

        results: list[VectorSearchResult] = []
        for name, collection in collections.items():
            if scope_filter and not self._scope_matches(name, scope_filter):
                continue
            try:
                raw = collection.query(
                    query_embeddings=[embedding],
                    n_results=min(n_results, collection.count() or 1),
                    include=["documents", "metadatas", "distances"],
                )
            except Exception:
                logger.warning(
                    "[Dedup] StoreBackedVectorIndex query failed for collection %s",
                    name,
                    exc_info=True,
                )
                continue
            ids = raw.get("ids", [[]])[0]
            distances = raw.get("distances", [[]])[0]
            metadatas = raw.get("metadatas", [[]])[0]
            documents = raw.get("documents", [[]])[0]
            for i, mem_id in enumerate(ids):
                # Build a minimal MemoryObject stub for dedup scoring
                meta = metadatas[i] if i < len(metadatas) else {}
                doc = documents[i] if i < len(documents) else ""
                from lace.memory.models import MemoryObject
                try:
                    mem = MemoryObject.from_chromadb(mem_id, doc, meta)
                except Exception:
                    logger.warning(
                        "[Dedup] Could not parse vector result %s from collection %s",
                        mem_id,
                        name,
                        exc_info=True,
                    )
                    continue
                dist = distances[i] if i < len(distances) else 1.0
                results.append(VectorSearchResult(memory=mem, distance=dist))

        results.sort(key=lambda r: r.distance)
        return results[:n_results]

    def add(self, memory: "MemoryObject", embedding: list[float]) -> None:
        from lace.retrieval.vector import upsert_memory
        memory.embedding = embedding
        upsert_memory(memory, self._vector_db_path)

    @staticmethod
    def _scope_matches(collection_name: str, scope_filter) -> bool:
        """Return True if the collection name belongs to any scope in the filter."""
        if isinstance(scope_filter, dict):
            scopes = scope_filter.get("project_scope", {}).get("$in", [])
        elif isinstance(scope_filter, list):
            scopes = scope_filter
        else:
            return True  # no filter — include everything
        from lace.retrieval.vector import _scope_to_collection_name
        allowed = {_scope_to_collection_name(s) for s in scopes}
        return not allowed or collection_name in allowed


def _get_hash_index_connection(
    db_path: Path = VAULT_HASH_INDEX_PATH,
) -> sqlite3.Connection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def initialize_vault_hash_index(
    db_path: Path = VAULT_HASH_INDEX_PATH,
) -> None:
    """
    Create vault_hash_index table and indexes if they don't exist.
    Called once at LACE startup. Idempotent.
    """
    with _get_hash_index_connection(db_path) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS vault_hash_index (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                canonical_hash  TEXT    NOT NULL UNIQUE,
                memory_id       TEXT    NOT NULL,
                summary_preview TEXT,
                scope           TEXT,
                created_at      TEXT    NOT NULL,
                last_merged_at  TEXT
            )
        """)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_vault_hash "
            "ON vault_hash_index(canonical_hash)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_vault_scope "
            "ON vault_hash_index(scope)"
        )
        conn.commit()
    logger.debug(f"[Dedup] Vault hash index ready at {db_path}")


def lookup_by_hash(
    hash_value: str,
    scope: Optional[str] = None,
    db_path: Path = VAULT_HASH_INDEX_PATH,
) -> Optional[str]:
    """
    Look up a memory_id by canonical hash.
    Optionally filter to same scope or global scope.
    """
    with _get_hash_index_connection(db_path) as conn:
        if scope:
            row = conn.execute(
                """
                SELECT memory_id FROM vault_hash_index
                WHERE  canonical_hash = ?
                AND    (scope = ? OR scope = 'global')
                LIMIT  1
                """,
                (hash_value, scope)
            ).fetchone()
        else:
            row = conn.execute(
                """
                SELECT memory_id FROM vault_hash_index
                WHERE  canonical_hash = ?
                LIMIT  1
                """,
                (hash_value,)
            ).fetchone()

    return row["memory_id"] if row else None


def insert_hash_index_entry(
    hash_value: str,
    memory_id: str,
    summary_preview: str,
    scope: str,
    db_path: Path = VAULT_HASH_INDEX_PATH,
) -> None:
    """
    Add a new entry to the vault hash index.
    Called after store_new_memory() succeeds.
    Uses INSERT OR IGNORE — safe if called twice with same hash.
    """
    now = datetime.now(timezone.utc).isoformat()
    with _get_hash_index_connection(db_path) as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO vault_hash_index
                (canonical_hash, memory_id, summary_preview, scope, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (hash_value, memory_id, summary_preview[:120], scope, now)
        )
        conn.commit()
    logger.debug(
        f"[Dedup] Hash index entry added | "
        f"memory_id={memory_id} | hash={hash_value[:12]}..."
    )


def update_hash_index_merge_time(
    hash_value: str,
    db_path: Path = VAULT_HASH_INDEX_PATH,
) -> None:
    """Update last_merged_at timestamp when a hash match triggers a merge."""
    now = datetime.now(timezone.utc).isoformat()
    with _get_hash_index_connection(db_path) as conn:
        conn.execute(
            "UPDATE vault_hash_index SET last_merged_at = ? "
            "WHERE canonical_hash = ?",
            (now, hash_value)
        )
        conn.commit()
# ---------------------------------------------------------------------------
# Scope-aware candidate filtering
# ---------------------------------------------------------------------------

def get_scope_filter(candidate_scope: Optional[str]) -> list[str]:
    """
    Return the list of scopes to compare against for dedup.
    """
    if candidate_scope is None:
        return []
    if candidate_scope == "global":
        return ["global"]
    return [candidate_scope, "global"]


# ---------------------------------------------------------------------------
# merge_into — the merge operation
# ---------------------------------------------------------------------------

def merge_into(
    existing: MemoryObject,
    candidate: dict,
    reason: str = "embedding_similarity",
) -> MemoryObject:
    """
    Merge a candidate memory dict into an existing MemoryObject.
    """
    now = datetime.now(timezone.utc)

    # --- Tags: union ---
    existing_tags = set(existing.tags or [])
    candidate_tags = set(candidate.get("tags", []))
    merged_tags = sorted(existing_tags | candidate_tags)

    original_tag_count = len(existing_tags)
    new_tags = candidate_tags - existing_tags

    # --- Content / Body: append if novel ---
    existing_content = (existing.content or "").strip()
    candidate_content = (
        candidate.get("body") or candidate.get("content") or ""
    ).strip()

    body_updated = False
    if candidate_content and candidate_content.lower() not in existing_content.lower():
        existing.content = (
            f"{existing_content}\n\n"
            f"---\n"
            f"*Additional context ({now.strftime('%Y-%m-%d')}):*\n"
            f"{candidate_content}"
        )
        body_updated = True

    # --- Confidence: boost, cap at 1.0 ---
    old_confidence = existing.confidence
    existing.confidence = min(1.0, (existing.confidence or 0.5) + 0.05)

    # --- Recency signals ---
    existing.last_accessed = now
    existing.access_count = (existing.access_count or 0) + 1
    existing.tags = merged_tags

    logger.info(
        f"[Dedup] merge_into | id={existing.id} | reason={reason} | "
        f"tags: {original_tag_count}→{len(merged_tags)} "
        f"(+{len(new_tags)} new) | "
        f"confidence: {old_confidence:.2f}→{existing.confidence:.2f} | "
        f"body_updated={body_updated}"
    )

    return existing


# ---------------------------------------------------------------------------
# Main two-tier entry point
# ---------------------------------------------------------------------------

def dedup_and_store(
    candidate: dict,
    vector_index: VectorIndex,
    memory_store,
    config: Optional[LaceConfig] = None,
    queue_id: Optional[int] = None,
    hash_index_db_path: Path = VAULT_HASH_INDEX_PATH,
    log_db_path: Path = PIPELINE_LOG_DB_PATH,
) -> Optional[str]:
    """
    Run two-tier dedup and store or merge a candidate memory.
    """
    if config is None:
        config = LaceConfig()

    dedup_cfg = config.dedup

    # Compute candidate hash from summary
    summary = candidate.get("summary", "")
    candidate_hash = canonical_hash(summary)
    candidate_scope = candidate.get("project_scope", "global")

    logger.debug(
        f"[Dedup] Checking candidate | "
        f"hash={candidate_hash[:12]}... | "
        f"scope={candidate_scope} | "
        f"summary='{summary[:60]}'"
    )

    initialize_vault_hash_index(hash_index_db_path)

    # -----------------------------------------------------------------------
    # TIER 1: Hash lookup — no embedding call
    # -----------------------------------------------------------------------
    existing_id = lookup_by_hash(
        candidate_hash,
        scope=candidate_scope,
        db_path=hash_index_db_path,
    )

    if existing_id:
        # Load existing memory
        existing = memory_store.get(existing_id)

        if existing is None:
            logger.warning(
                f"[Dedup] Hash index points to missing memory {existing_id}. "
                "Falling through to Tier 2."
            )
        else:
            updated = merge_into(existing, candidate, reason="hash_match")
            memory_store.save(updated)
                
            update_hash_index_merge_time(candidate_hash, db_path=hash_index_db_path)

            log_dedup_event(
                canonical_hash_value=candidate_hash,
                action="merge_hash",
                target_id=existing_id,
                score=None,
                queue_id=queue_id,
                db_path=log_db_path,
            )

            logger.info(
                f"[Dedup] MERGE_HASH | "
                f"candidate merged into {existing_id}"
            )
            return existing_id

    # -----------------------------------------------------------------------
    # TIER 2: Embedding similarity
    # -----------------------------------------------------------------------

    # Generate embedding for the candidate summary
    try:
        candidate_embedding = embed_text(summary)
    except Exception as e:
        logger.error(
            f"[Dedup] Embedding failed: {e}. Storing as new with needs_reindex=True.",
            exc_info=True,
        )
        candidate["needs_reindex"] = True
        return _store_new(
            candidate, candidate_hash, vector_index,
            memory_store, queue_id, config,
            hash_index_db_path, log_db_path
        )

    # Query ChromaDB for nearest neighbor within scope
    scope_filter = get_scope_filter(candidate_scope)
    nearest_results = vector_index.query(
        embedding=candidate_embedding,
        n_results=1,
        scope_filter=scope_filter if scope_filter else None,
    )

    if not nearest_results:
        # Vault is empty — store directly
        return _store_new(
            candidate, candidate_hash, vector_index,
            memory_store, queue_id, config,
            hash_index_db_path, log_db_path
        )

    nearest = nearest_results[0]
    similarity = 1.0 - (nearest.distance / 2.0)

    logger.debug(
        f"[Dedup] Tier 2 | nearest={nearest.memory.id} | "
        f"similarity={similarity:.4f} | "
        f"skip_threshold={dedup_cfg.skip_threshold} | "
        f"merge_threshold={dedup_cfg.merge_threshold}"
    )

    # --- SKIP ---
    if similarity >= dedup_cfg.skip_threshold:
        log_dedup_event(
            canonical_hash_value=candidate_hash,
            action="skip",
            target_id=nearest.memory.id,
            score=similarity,
            queue_id=queue_id,
            db_path=log_db_path,
        )
        logger.info(
            f"[Dedup] SKIP | similarity={similarity:.4f} >= "
            f"skip_threshold={dedup_cfg.skip_threshold}"
        )
        return nearest.memory.id

    # --- MERGE ---
    elif similarity >= dedup_cfg.merge_threshold:
        # Protect high-quality memories from low-confidence merges
        is_protected = (
            nearest.memory.lifecycle.value in ("validated", "consolidated") and
            (nearest.memory.confidence or 0.0) > 0.85 and
            (candidate.get("confidence", 0.5) or 0.5) < 0.6
        )

        if is_protected:
            logger.info(
                f"[Dedup] STORE (protected) | high-confidence validated memory "
                f"{nearest.memory.id} protected from low-confidence merge. "
                f"Storing candidate as new."
            )
            return _store_new(
                candidate, candidate_hash, vector_index,
                memory_store, queue_id, config,
                hash_index_db_path, log_db_path,
                score=similarity
            )

        existing = memory_store.get(nearest.memory.id)

        if existing is None:
            logger.warning(
                f"[Dedup] MERGE target {nearest.memory.id} not found in store. "
                "Storing as new."
            )
            return _store_new(
                candidate, candidate_hash, vector_index,
                memory_store, queue_id, config,
                hash_index_db_path, log_db_path,
                score=similarity
            )

        updated = merge_into(existing, candidate, reason="embedding_similarity")
        memory_store.save(updated)

        log_dedup_event(
            canonical_hash_value=candidate_hash,
            action="merge_embedding",
            target_id=existing.id,
            score=similarity,
            queue_id=queue_id,
            db_path=log_db_path,
        )

        logger.info(
            f"[Dedup] MERGE_EMBEDDING | similarity={similarity:.4f} | "
            f"merged into {existing.id}"
        )
        return existing.id

    # --- STORE ---
    else:
        return _store_new(
            candidate, candidate_hash, vector_index,
            memory_store, queue_id, config,
            hash_index_db_path, log_db_path,
            score=similarity
        )


# ---------------------------------------------------------------------------
# Store helper
# ---------------------------------------------------------------------------

def _store_new(
    candidate: dict,
    candidate_hash: str,
    vector_index: VectorIndex,
    memory_store,
    queue_id: Optional[int],
    config: LaceConfig,
    hash_index_db_path: Path,
    log_db_path: Path,
    score: Optional[float] = None,
) -> Optional[str]:
    """
    Store a candidate as a brand new memory.
    """
    try:
        new_memory = memory_store.add(
            content=candidate.get("body") or candidate.get("content") or candidate.get("summary", ""),
            category=candidate.get("category", "pattern"),
            tags=candidate.get("tags", []),
            scope=candidate.get("project_scope", "global"),
            source="auto_extracted",
            confidence=candidate.get("confidence", 0.4),
            summary=candidate.get("summary"),
        )
        if candidate.get("needs_reindex") and not new_memory.needs_reindex:
            new_memory.needs_reindex = True
            memory_store.save(new_memory)

        # Update hash index
        insert_hash_index_entry(
            hash_value=candidate_hash,
            memory_id=new_memory.id,
            summary_preview=candidate.get("summary", ""),
            scope=candidate.get("project_scope", "global"),
            db_path=hash_index_db_path,
        )

        log_dedup_event(
            canonical_hash_value=candidate_hash,
            action="store",
            target_id=new_memory.id,
            score=score,
            queue_id=queue_id,
            db_path=log_db_path,
        )

        logger.info(
            f"[Dedup] STORE | new memory_id={new_memory.id} | "
            f"summary='{candidate.get('summary', '')[:60]}'"
        )
        return new_memory.id

    except Exception as e:
        logger.error(f"[Dedup] STORE failed: {e}", exc_info=True)
        return None
