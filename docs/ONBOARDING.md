# LACE Developer Onboarding Guide

Welcome to the LACE (**Local AI Context Engine**) codebase! This document provides the architecture, directory layouts, configuration gotchas, runbooks, and debugging tools needed to work on LACE safely and productively.

---

## 1. System Architecture & Data Flow

LACE is structured around three key pathways: **Ingestion (Write)**, **Retrieval (Read)**, and **Synchronization (Sync)**.

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

### 1.1 Ingestion Pipeline (Write Path)
Ingestion runs asynchronously to keep client interactions fast ($<100\text{ms}$). Slow LLM extraction ($5\text{s} - 30\text{s}$) runs in a background thread worker:
1. **`process_interaction`**: Invoked by client, enqueues turn to SQLite `extraction_queue.db` in $<5\text{ms}$.
2. **Pre-filter**: The worker parses the job using `should_attempt_extraction()`. Responses under 100 characters, greetings, or command errors are discarded early.
3. **LLM Extraction**: Evaluates worthiness. Gated/rejected interactions are logged to `pipeline_log.db` for debugging.
4. **Two-Tier Deduplication**: Evaluates similarity with existing vault memories:
   * **Cosine Similarity $> 95\%$**: Discard candidate (duplicate suppressed).
   * **Cosine Similarity $85\% - 95\%$**: Merge candidate contents, tags, and update existing memory.
   * **Cosine Similarity $< 85\%$**: Create new Markdown vault file, index in ChromaDB, and link network graph.

### 1.2 The 7-Step Unified Retrieval Pipeline (Read Path)
Instead of simple vector lookup, LACE scores and ranks memories through a composite relevance algorithm:
1. **Vector Search**: Computes cosine similarities in ChromaDB to seed candidates.
2. **Tag Expansion**: Scans the query for tags, fetching matching vault notes.
3. **Graph Expansion**: Fetches topological neighbors from the NetworkX graph.
4. **Co-Retrieval Boost**: Boosts items based on historical usage co-occurrence.
5. **Lazy-Load**: Hydrates full `MemoryObject` models for expanded candidates.
6. **Multi-Signal Score**: Ranks elements using a weighted formula:
   $$\text{Score} = w_{\text{vector}} \cdot S_{\text{vector}} + w_{\text{tag}} \cdot S_{\text{tag}} + w_{\text{graph}} \cdot S_{\text{graph}} + w_{\text{co\_retrieval}} \cdot S_{\text{co\_retrieval}} + w_{\text{recency}} \cdot S_{\text{recency}} + w_{\text{confidence}} \cdot S_{\text{confidence}}$$
7. **Filter & Rank**: Filters out items below relevance thresholds and sorts the results.

### 1.3 Sync Path (Obsidian Integration)
Uses `watchdog` to monitor filesystems. Relies on modification times (`mtime`) and file hashes inside `vault_hash_index.db` to synchronize updates bidirectionally and prevent file write cycles.

---

## 2. Directory Layout & State Locations

All persistent states default to the user's home directory `~/.lace/` (overrideable via `LACE_HOME` environment variable):

```
~/.lace/
├── config/
│   ├── lace.yaml               <--- Global configuration settings
│   ├── identity.md             <--- Persona definitions for the LLM
│   └── preferences.yaml        <--- Ingested user preference overrides
│
├── memory/
│   ├── vault/                  <--- Source of truth (Markdown files)
│   │   ├── global/
│   │   └── projects/
│   ├── vector_db/              <--- ChromaDB collections / embeddings database
│   ├── vault_hash_index.db     <--- SQLite tracking of synced file mtimes/hashes
│   ├── graph.json              <--- Serialized NetworkX concept graph
│   └── co_retrieval.json       <--- Frequency-based retrieval co-occurrences
│
├── queue/
│   ├── extraction_queue.db     <--- SQLite async interaction queue
│   └── pipeline_log.db         <--- SQLite extraction worthiness verdicts & rejections
│
└── logs/
    ├── retrieval/              <--- Unified retrieval scoring & latency logs
    └── interactions/           <--- Client interaction logs
```

### 2.1 Codebase File Map
The most critical files in `src/lace/` are:
* **`core/config.py`**: Configuration schema and the path resolver `resolve_lace_paths()`.
* **`memory/models.py`**: Pydantic models for core memory objects.
* **`memory/store.py`**: high-level CRUD seam orchestrating vault files, ChromaDB, and NetworkX.
* **`mcp/tools.py`**: Core MCP tools (e.g. `get_relevant_context`, `remember`).
* **`retrieval/unified.py`**: Coordinates the 7-step retrieval pipeline.
* **`mcp/queue.py`**: Implements the SQLite background extraction queue worker thread.
* **`memory/extractor.py`**: Defines LLM extraction prompts and worthiness gates.
* **`memory/dedup.py`**: Evaluates cosine similarity thresholds.
* **`vault/sync.py`**: Mirroring files to Obsidian vaults.

---

## 3. Local Setup & Gotchas

### 3.1 Prerequisites
- **Python**: `>= 3.11` (Python `3.13` or `3.14` recommended)
- **uv**: Speedy package manager.
- **Ollama**: Running locally with the `llama3.2` model downloaded (`ollama pull llama3.2`).

### 3.2 Step-by-Step Installation
```bash
git clone https://github.com/AayushM0/lace.git
cd lace
uv pip install -e .
lace init
pytest
```

### 3.3 Key Developer Gotchas
1. **Path Resolvers**: Never compute paths manually. Always import and call `resolve_lace_paths(lace_home)`.
2. **ID Constraints**: Memory IDs are structured as `mem_<12 hex>`. The sync engine relies on this structure; changing it breaks synchronization.
3. **Database Timeout**: To prevent sqlite concurrency blockages, initialize connections with `timeout=10.0` and ensure connections are closed inside `finally` blocks.
4. **Stale ChromaDB/Graph**: If Markdown files are edited or deleted manually, the vector DB and graph will go out of sync. Rebuild them by running:
   ```bash
   lace memory reindex
   ```

---

## 4. Developer Runbooks

### Runbook 1: Add a New MCP Tool
1. **Implement tool**: Open `src/lace/mcp/tools.py`, create your async tool, and accept `**kwargs` to prevent runtime dictionary matching crashes:
   ```python
   async def list_unique_tags(scope: str = "auto", **kwargs) -> list[str]:
       store, resolved_scope = _get_store(scope)
       memories = store.list(scope=resolved_scope, limit=1000)
       return list({tag for m in memories for tag in m.tags})
   ```
2. **Register tool**: Open `src/lace/mcp/server.py`, import the function, and register it via `@mcp.tool()`.
3. **Verify**: Run `lace mcp start` to test the stdio daemon.

### Runbook 2: Write and Run Tests
* Place tests under the `tests/` directory matching the module name.
* **Run all tests**: `pytest`
* **Run specific test file**: `pytest tests/test_memory/test_dedup.py`

### Runbook 3: Change Retrieval Weights
* **Via CLI (Permanent)**:
  ```bash
  lace config set retrieval.weights.semantic_similarity 0.50
  lace config set retrieval.weights.recency 0.10
  ```
  *(Note: All weights must sum to exactly `1.0`)*
* **Programmatically (Runtime)**:
  ```python
  from lace.retrieval.unified import UnifiedWeights
  retriever.set_weights(UnifiedWeights(vector=0.50, tag=0.10, graph=0.15, co_retrieval=0.10, recency=0.10, confidence=0.05))
  ```

---

## 5. Diagnostic Commands

### Inspect background worker queue:
```bash
sqlite3 ~/.lace/queue/extraction_queue.db "SELECT id, status, retry_count, error_msg FROM extraction_queue ORDER BY created_at DESC LIMIT 5;"
```

### Audit LLM worthiness verdicts:
```bash
sqlite3 ~/.lace/queue/pipeline_log.db "SELECT created_at, worth_remembering, reason, memory_count FROM pipeline_logs ORDER BY created_at DESC LIMIT 5;"
```

### Check graph health:
```bash
lace graph stats
lace memory stats
```
