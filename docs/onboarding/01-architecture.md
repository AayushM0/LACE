# LACE Architecture & Data Flow

This document details the system design, core modules, ingestion pipeline, and multi-signal retrieval pipeline of LACE (**Local AI Context Engine**).

---

## 1. System Shape

The LACE architecture is built on a clean separation between **Ingestion (Write Path)**, **Retrieval (Read Path)**, and **Obsidian Synchronization (Sync)**.

```
AI client (Cursor / Claude Desktop / `lace ask`)
        │  MCP stdio (JSON-RPC)        │  CLI (Typer)
        ▼                               ▼
   mcp/server.py  ◄── tools ──►   main.py  (Typer app, 9 sub-apps)
        │                               │
        └──────────► MemoryStore ◄──────┘   (the seam: CRUD + retrieval)
                          │
        ┌─────────────────┼──────────────────────┐
        ▼                 ▼                      ▼
   memory/            retrieval/              vault/
   extractor          unified (7-step)        sync (Obsidian)
   dedup              vector (ChromaDB)       state (mtime tracking)
   pipeline_log       graph (NetworkX)
   queue (SQLite)     co_occurrence
   markdown           tag_index
   models             embeddings (local)
```

### Core Architecture Principles:
- **Local-First & Offline**: Embeddings are computed locally using HuggingFace sentence-transformers (`all-MiniLM-L6-v2` by default), vector store runs locally via ChromaDB, and the knowledge graph is processed in-memory using NetworkX.
- **Markdown Source of Truth**: The Markdown files in `~/.lace/memory/vault` represent the canonical state. The database indexes (SQLite queues, vector DB collections, graph files) are *derivative* and can be rebuilt entirely from the Markdown notes via `lace memory reindex`.
- **Ephemerality of Sessions**: EPHEMERAL sessions store memories in memory/vault/global if they represent reusable patterns, but separate active scopes to prevent contamination.

---

## 2. Ingestion Pipeline (The Write Path)

Ingestion is asynchronous to ensure that AI interactions remain fast (<100ms path for the client), while the slow LLM extraction (5–30s) runs in the background.

```
AI Client Interaction
       │
       ▼
process_interaction()
       │
       ▼
[SQLite: extraction_queue.db]  <--- Enqueued in <5ms
       │
  (background thread worker polls every 30s)
       │
       ▼
Pre-filter: should_attempt_extraction()  <--- Cheap heuristics (greetings/errors/short responses filtered out)
       │
       ▼
LLM Extraction: extract_memories()  <--- Worthiness verdict (commit to boolean worth_remembering)
       │
       ├──► Gated Out / Rejected ──► Log to pipeline_log.db (if configured)
       │
       ▼ (If Worthy)
Two-Tier Deduplication: dedup_and_store()
       │
       ├──► Cosine Similarity > 95% ──► Skip (Suppressed Duplicate)
       │
       ├──► Cosine Similarity 85% - 95% ──► Merge content/tags into existing note
       │
       └──► Cosine Similarity < 85% ──► Store as New Note
                                            │
                                            ├──► Write Markdown file (~/.lace/memory/vault)
                                            ├──► Index in ChromaDB Vector Store
                                            └──► Insert Wikilinks / Update Graph
```

### Worthiness Verdict & Logging
The worthiness verdict is a critical security and noise guard. The LLM commits to whether a turn is `worth_remembering` and provides a `reason`.
- **Verdict Logs**: Every verdict (including rejections) is written to `pipeline_log.db` to enable auditing of what got filtered out.
- **Canonical Hash Suppression**: In `enqueue()`, LACE computes a canonical hash of the interaction. If the same hash is sent within `hash_cooldown_seconds` (default: 300s) to the same scope, the queue merges the job by incrementing the `repeat_count` and suppressing redundant LLM invocations.

---

## 3. The 7-Step Retrieval Pipeline (The Read Path)

LACE replaces simple vector search with a composite multi-signal ranking system. This is orchestrating in [unified.py](file:///home/aayush/Projects/lace/LACE/src/lace/retrieval/unified.py).

```
          Query String
               │
               ▼
┌──────────────────────────────┐
│  Step 1: Vector Search       │ ◄── Cosine similarity from ChromaDB
└──────────────┬───────────────┘
               │ (Seed Candidates)
               ▼
┌──────────────────────────────┐
│  Step 2: Tag Expansion       │ ◄── Scan query for tags, fetch tagged notes
└──────────────┬───────────────┘
               │ (Augmented Candidates)
               ▼
┌──────────────────────────────┐
│  Step 3: Graph Expansion     │ ◄── Fetch graph neighbors of top vector seeds
└──────────────┬───────────────┘
               │ (Structural Candidates)
               ▼
┌──────────────────────────────┐
│  Step 4: Co-Retrieval Boost  │ ◄── Apply boost for learned co-retrieval patterns
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Step 5: Load MemoryObjects  │ ◄── Lazy-load full files for expanded candidates
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Step 6: Score Everything    │ ◄── Combined weighted formula (5 signals)
└──────────────┬───────────────┘
               │
               ▼
┌──────────────────────────────┐
│  Step 7: Filter & Rank       │ ◄── Threshold check, sort, cap, log search metrics
└──────────────────────────────┘
```

### Detail of Scoring (Step 6)
The final relevance score of a memory is computed as:

$$\text{Score} = w_{\text{vector}} \cdot S_{\text{vector}} + w_{\text{tag}} \cdot S_{\text{tag}} + w_{\text{graph}} \cdot S_{\text{graph}} + w_{\text{co\_retrieval}} \cdot S_{\text{co\_retrieval}} + w_{\text{recency}} \cdot S_{\text{recency}} + w_{\text{confidence}} \cdot S_{\text{confidence\_effective}}$$

Where:
- **$S_{\text{vector}}$**: Normalized cosine similarity $(1.0 - \text{distance}/2.0)$.
- **$S_{\text{tag}}$**: Ratio of matched query tags present in the memory tags.
- **$S_{\text{graph}}$**: Node transition weight from NetworkX.
- **$S_{\text{co\_retrieval}}$**: Frequency-based usage pattern boost.
- **$S_{\text{recency}}$**: Exponential decay based on time elapsed since `last_accessed` (default half-life: 30 days).
- **$S_{\text{confidence\_effective}}$**: Combination of the memory's `confidence` rating (70%) and a matching `scope_bonus` (30%).

---

## 4. Vault Synchronization (The Sync Path)

Real-time bidirectional synchronization mirrors LACE memories to Obsidian vaults:
- Sync watches file systems using `watchdog` to catch changes in Obsidian.
- File modification times (`mtime`) are checked. If Obsidian `mtime > lace_mtime + 1.0s`, changes are pulled into LACE and ChromaDB is updated. If `lace_mtime > obs_mtime + 1.0s`, changes are copied to Obsidian.
- A central SQLite database `vault_hash_index.db` tracks file hashes and modification timestamps to avoid sync loops.
