"""ChromaDB vector store operations for LACE.

ChromaDB is embedded (no server needed), stores data locally,
and provides fast approximate nearest-neighbor search.

One ChromaDB collection per scope:
  - "global"       → collection "lace_global"
  - "project:myapi"→ collection "lace_project_myapi"
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings

from lace.memory.models import MemoryObject


# ── Client singleton ──────────────────────────────────────────────────────────

_client: chromadb.PersistentClient | None = None
_client_path: str | None = None


def get_client(vector_db_path: Path) -> chromadb.PersistentClient:
    """Return a ChromaDB persistent client, creating it if needed."""
    global _client, _client_path

    path_str = str(vector_db_path)
    if _client is None or _client_path != path_str:
        vector_db_path.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=path_str,
            settings=Settings(anonymized_telemetry=False),
        )
        _client_path = path_str

    return _client


def _scope_to_collection_name(scope: str) -> str:
    """Convert a scope string to a valid ChromaDB collection name.

    ChromaDB collection names must be 3-63 chars, alphanumeric + hyphens,
    start and end with alphanumeric.

    Examples:
        "global"          → "lace-global"
        "project:my-api"  → "lace-project-my-api"
        "project:My API"  → "lace-project-my-api"
    """
    clean = scope.replace(":", "-").replace(" ", "-").replace("_", "-")
    clean = re.sub(r"[^a-z0-9-]", "", clean.lower())
    clean = re.sub(r"-+", "-", clean).strip("-")
    name = f"lace-{clean}"
    # ChromaDB max length is 63
    return name[:63]


class DummyEmbeddingFunction:
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:
        return [[0.0]] * len(input)


def get_collection(
    scope: str,
    vector_db_path: Path,
) -> chromadb.Collection:
    """Get or create a ChromaDB collection for a given scope."""
    client = get_client(vector_db_path)
    collection_name = _scope_to_collection_name(scope)
    return client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
        embedding_function=DummyEmbeddingFunction(),
    )


# ── Write operations ──────────────────────────────────────────────────────────

def upsert_memory(
    memory: MemoryObject,
    vector_db_path: Path,
) -> None:
    """Add or update a memory's embedding in the vector store.

    If the memory already exists (same ID), it is updated.
    If memory.embedding is None, this is a no-op.
    """
    if memory.embedding is None:
        return

    collection = get_collection(memory.project_scope, vector_db_path)

    collection.upsert(
        ids=[memory.id],
        embeddings=[memory.embedding],
        documents=[memory.content],
        metadatas=[{
            "category":      memory.category.value,
            "source":        memory.source.value,
            "lifecycle":     memory.lifecycle.value,
            "confidence":    memory.confidence,
            "project_scope": memory.project_scope,
            "tags":          ",".join(memory.tags),
            "access_count":  memory.access_count,
        }],
    )


def delete_from_vector_store(
    memory_id: str,
    scope: str,
    vector_db_path: Path,
) -> None:
    """Remove a memory from the vector store."""
    collection = get_collection(scope, vector_db_path)
    try:
        collection.delete(ids=[memory_id])
    except Exception:
        pass  # Not in vector store — that's fine


# ── Search operations ─────────────────────────────────────────────────────────

def vector_search(
    query_embedding: list[float],
    scope: str,
    vector_db_path: Path,
    n_results: int = 20,
    exclude_archived: bool = True,
) -> list[dict[str, Any]]:
    """Search a collection for the most similar memories.

    Args:
        query_embedding: The embedded query vector.
        scope: Which collection to search ("global", "project:my-api").
        vector_db_path: Path to ChromaDB storage.
        n_results: Maximum candidates to return before ranking.
        exclude_archived: If True, skip archived memories.

    Returns:
        List of dicts with keys: id, distance, document, metadata.
        Sorted by distance ascending (lower = more similar in cosine space).
    """
    collection = get_collection(scope, vector_db_path)

    # Check if collection has any documents
    count = collection.count()
    if count == 0:
        return []

    # Don't ask for more results than exist
    actual_n = min(n_results, count)

    where_filter: dict | None = None
    if exclude_archived:
        where_filter = {"lifecycle": {"$ne": "archived"}}

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=actual_n,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )
    except Exception:
        return []

    # Unpack ChromaDB's nested response format
    output: list[dict[str, Any]] = []
    if not results["ids"] or not results["ids"][0]:
        return []

    for i, memory_id in enumerate(results["ids"][0]):
        output.append({
            "id":       memory_id,
            "distance": results["distances"][0][i],
            "document": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
        })

    return output


def multi_scope_vector_search(
    query_embedding: list[float],
    scopes: list[str],
    vector_db_path: Path,
    n_results: int = 20,
) -> list[dict[str, Any]]:
    """Search across multiple scopes and merge results.

    Used when active scope is a project — we search both the project
    collection and the global collection.

    Returns deduplicated results sorted by distance.
    """
    seen_ids: set[str] = set()
    all_results: list[dict[str, Any]] = []

    for scope in scopes:
        results = vector_search(
            query_embedding=query_embedding,
            scope=scope,
            vector_db_path=vector_db_path,
            n_results=n_results,
        )
        for result in results:
            if result["id"] not in seen_ids:
                result["search_scope"] = scope
                all_results.append(result)
                seen_ids.add(result["id"])

    # Sort by distance (ascending — lower is more similar)
    all_results.sort(key=lambda r: r["distance"])
    return all_results


def get_collection_stats(scope: str, vector_db_path: Path) -> dict[str, int]:
    """Return basic stats for a collection."""
    collection = get_collection(scope, vector_db_path)
    return {"count": collection.count()}