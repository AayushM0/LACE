# LACE Developer Onboarding Guide

Welcome to the LACE (**Local AI Context Engine**) codebase. This guide is designed to get a new engineer from `git clone` to a fully verified local development environment in **under 10 minutes**, followed by an architectural breakdown, key file map, developer runbooks, and debugging runbooks.

---

## 1. 10-Minute Local Setup

### Prerequisites

| Tool | Required Version | Purpose | Installation / Check Command |
| :--- | :--- | :--- | :--- |
| **Python** | `>= 3.11` (3.13+ recommended) | Runtime environment | `python --version` |
| **uv** | `>= 0.4.0` (recommended) or `pip` | Package & virtualenv manager | `uv --version` (or `pip install uv`) |
| **Git** | `>= 2.30` | Version control & scope detection | `git --version` |
| **Ollama** | Optional (for local LLM extraction) | Local LLM inference server | `ollama --version` (`ollama pull llama3.2`) |

---

### Step-by-Step Installation

#### Step 1: Clone the repository (1 min)
```bash
git clone https://github.com/AayushM0/LACE.git
cd LACE
```

#### Step 2: Create virtual environment & install dependencies (2 min)
Using `uv` (recommended):
```bash
uv venv
# On Windows PowerShell:
.venv\Scripts\Activate.ps1
# On macOS / Linux:
source .venv/bin/activate

uv pip install -e ".[dev]"
```

#### Step 3: Initialize LACE home environment (30 sec)
This provisions default directories and configuration templates in `~/.lace/`:
```bash
lace init
```
To verify:
```bash
lace config show
```

#### Step 4: Run the test suite (1.5 min)
Execute the complete test suite to confirm environment integrity:
```bash
pytest
```
*Expected result: `584 passed` across all test modules.*

#### Step 5: Test the CLI memory workflow (1 min)
Verify that embedding generation, storage, and retrieval operate correctly:
```bash
# Add a test memory note
lace memory add "Testing local onboarding setup for LACE" --tag setup --category tech-stack

# Semantically query the memory vault
lace memory search "onboarding setup" --scores
```

---

### Setup Verification Checklist

- [ ] `lace --help` outputs Typer commands without Python traceback.
- [ ] `lace config show` prints valid YAML configuration.
- [ ] `pytest` passes 584 tests with 0 failures.
- [ ] `~/.lace/memory/vault/global/` exists on disk.
- [ ] `lace memory search` returns scored results.

---

## 2. System Architecture & Data Flow

LACE is structured as an **embedded, local-first engine with decoupled background subsystems**. Rather than incurring network latency and privacy leaks through cloud microservices, LACE coordinates local processes via stdio IPC, background worker threads, and local filesystem watchers.

```
AI Client (Claude Desktop / Cursor / `lace ask`)
        │
        ├── Standard I/O (JSON-RPC over stdio)
        ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ src/lace/mcp/server.py                                      │
   │  - Exposes MCP tools & resources                            │
   │  - Dispatches calls to MemoryStore                          │
   └──────────────────────────────┬──────────────────────────────┘
                                  │
                                  ▼
   ┌─────────────────────────────────────────────────────────────┐
   │ src/lace/memory/store.py (High-level CRUD & Retrieval Seam) │
   └───────┬──────────────────────┬──────────────────────┬───────┘
           │                      │                      │
           ▼                      ▼                      ▼
  [ Ingestion Pipeline ]  [ 7-Step Retrieval ]   [ Obsidian Sync Daemon ]
   mcp/queue.py            retrieval/unified.py   vault/sync.py
   memory/extractor.py     retrieval/vector.py    vault/state.py
   memory/dedup.py         retrieval/graph.py     (`watchdog` + mtime)
           │                      │                      │
           ▼                      ▼                      ▼
  SQLite Queue DB         ChromaDB & Graph        Local Markdown Vault
  (~/.lace/queue/)        (~/.lace/memory/)       (~/.lace/memory/vault/)
```

### Ingestion Pathway (Write Path)
* **`<5ms` Enqueue**: `process_interaction` enqueues conversation turns into `~/.lace/queue/extraction_queue.db`.
* **Pre-filter**: Background worker discards trivial strings, greetings, or error outputs ($<100$ characters).
* **LLM Extraction**: Evaluates worthiness; logs rejections to `pipeline_log.db`.
* **Two-Tier Deduplication**:
  * Cosine similarity $> 95\%$: Drop (suppressed duplicate).
  * Cosine similarity $85\% - 95\%$: Merge content and tags into existing note.
  * Cosine similarity $< 85\%$: Create new `.md` file in vault, index into ChromaDB, update NetworkX graph.

### Unified Retrieval Pathway (Read Path)
Composite scoring formula balancing 6 distinct signals:
$$\text{Score} = w_{\text{vector}} S_{\text{vector}} + w_{\text{tag}} S_{\text{tag}} + w_{\text{graph}} S_{\text{graph}} + w_{\text{co}} S_{\text{co}} + w_{\text{recency}} S_{\text{recency}} + w_{\text{confidence}} S_{\text{confidence}}$$

---

## 3. Key File Map & Critical Code Paths

### Priority Files to Read First

| Priority | File Path | Component | Responsibility |
| :--- | :--- | :--- | :--- |
| **P1** | [`src/lace/memory/models.py`](file:///d:/Projects/LACE/src/lace/memory/models.py) | Data Models | `MemoryObject` schema, validation, lifecycle states |
| **P2** | [`src/lace/memory/store.py`](file:///d:/Projects/LACE/src/lace/memory/store.py) | Memory Seam | Core CRUD orchestrating Markdown, ChromaDB, and NetworkX |
| **P3** | [`src/lace/mcp/queue.py`](file:///d:/Projects/LACE/src/lace/mcp/queue.py) | Ingestion | SQLite worker thread, queue polling, retry backoff |
| **P4** | [`src/lace/retrieval/unified.py`](file:///d:/Projects/LACE/src/lace/retrieval/unified.py) | Retrieval | 7-step hybrid ranking algorithm and weight normalization |
| **P5** | [`src/lace/mcp/tools.py`](file:///d:/Projects/LACE/src/lace/mcp/tools.py) | MCP Protocol | Tool implementations (`get_relevant_context`, `remember`) |
| **P6** | [`src/lace/vault/sync.py`](file:///d:/Projects/LACE/src/lace/vault/sync.py) | Sync Engine | Filesystem watcher, mtime comparisons, hash tracking |
| **P7** | [`src/lace/core/config.py`](file:///d:/Projects/LACE/src/lace/core/config.py) | Configuration | Settings schema and `resolve_lace_paths()` |

### Dangerous Files — Coordinate Before Modifying

| File Path | Risk Level | Reason & Blast Radius |
| :--- | :--- | :--- |
| `src/lace/memory/models.py` | **CRITICAL** | Changing `id` format (`mem_<12hex>`) breaks vault indexing, ChromaDB primary keys, and Obsidian sync tracking. |
| `src/lace/mcp/queue.py` | **HIGH** | Concurrency handling for SQLite. Improper connection lifecycles cause `database is locked` errors during multi-agent tool execution. |
| `src/lace/retrieval/unified.py` | **HIGH** | Weight summation must equal `1.0`. Modifying step order or normalization logic skews retrieval relevance scores across all consumers. |
| `src/lace/vault/sync.py` | **HIGH** | File deletion or bidirectional sync logic. Bugs can cause file clobbering or infinite write recursion between LACE and external editors. |

---

## 4. Developer Runbooks

### Runbook 1: Add a New MCP Tool
1. **Define the tool logic** in [`src/lace/mcp/tools.py`](file:///d:/Projects/LACE/src/lace/mcp/tools.py). Always accept `**kwargs` to prevent runtime argument mismatch crashes:
   ```python
   async def export_vault_summary(scope: str = "auto", **kwargs) -> dict:
       store, resolved_scope = _get_store(scope)
       memories = store.list(scope=resolved_scope, limit=500)
       return {"count": len(memories), "scope": resolved_scope}
   ```
2. **Register the tool** in [`src/lace/mcp/server.py`](file:///d:/Projects/LACE/src/lace/mcp/server.py):
   ```python
   @mcp.tool()
   async def export_vault_summary(scope: str = "auto") -> str:
       """Export a high-level summary of active vault memories."""
       return json.dumps(await tools.export_vault_summary(scope=scope))
   ```
3. **Write tests** in `tests/test_mcp/test_tools.py`.
4. **Verify stdio daemon**: Run `lace mcp start` and send a test JSON-RPC request.

---

### Runbook 2: Adjust Retrieval Weights
Retrieval weights control how much influence vector similarity vs. graph connectivity vs. recency has on memory ranking:

* **Permanent adjustment via CLI**:
  ```bash
  lace config set retrieval.weights.vector 0.45
  lace config set retrieval.weights.graph 0.20
  lace config set retrieval.weights.recency 0.15
  ```
  *(Note: All 6 weights in `retrieval.weights` must sum to exactly `1.0`)*

* **Runtime testing in code**:
  ```python
  from lace.retrieval.unified import UnifiedWeights
  retriever.set_weights(UnifiedWeights(
      vector=0.45, tag=0.10, graph=0.20,
      co_retrieval=0.10, recency=0.10, confidence=0.05
  ))
  ```

---

### Runbook 3: Rebuilding Out-of-Sync Indexes
If Markdown vault files are manually added, edited, or removed outside LACE:
```bash
# Rebuild ChromaDB vector embeddings from scratch:
lace memory reindex

# Reconstruct NetworkX concept graph:
lace graph build

# Verify index counts match:
lace memory stats
lace graph stats
```

---

## 5. Debugging Guide & Real Solutions

### Error 1: `sqlite3.OperationalError: database is locked`
* **Symptom**: MCP server or CLI commands fail with a database locked exception.
* **Root Cause**: An unclosed connection in a concurrent thread or a long-running transaction blocked SQLite.
* **Fix**: Ensure all SQLite connection objects specify `timeout=10.0` upon initialization and are encapsulated in `try ... finally: conn.close()` or context managers.
* **Verification**:
  ```bash
  sqlite3 ~/.lace/queue/extraction_queue.db "PRAGMA journal_mode;"
  # Should output: wal
  ```

### Error 2: `ChromaDB collection count != Markdown file count`
* **Symptom**: Searching returns stale memories that were deleted from disk.
* **Root Cause**: Files were deleted from `~/.lace/memory/vault/` via File Explorer/Finder without using `lace memory forget`.
* **Fix**: Run `lace memory reindex` to synchronize ChromaDB collections with the physical `.md` files on disk.

### Error 3: Windows Virtual Environment File Lock (`os error 5`)
* **Symptom**: Running `uv run pytest` fails with `failed to remove directory ... .venv\Lib: Access is denied`.
* **Root Cause**: An active Python process (e.g. background MCP server or IDE terminal) has `.venv` DLLs locked in memory.
* **Fix**: Run tests directly via the virtualenv binary instead of invoking uv rebuild:
  ```powershell
  .venv\Scripts\pytest.exe
  ```

### Error 4: `OllamaConnectionError: Failed to connect to localhost:11434`
* **Symptom**: Background extraction worker reports connection refused.
* **Root Cause**: Ollama server is not running locally.
* **Fix**: Start the server with `ollama serve` and ensure the extraction model is pulled: `ollama pull llama3.2`. Alternatively, switch provider in config: `lace config set provider.default "openai"`.

---

## 6. Diagnostic Commands

Inspect SQLite background worker queue:
```bash
sqlite3 ~/.lace/queue/extraction_queue.db "SELECT id, status, retry_count, error_msg, created_at FROM extraction_queue ORDER BY created_at DESC LIMIT 5;"
```

Audit LLM worthiness verdicts and filter rejections:
```bash
sqlite3 ~/.lace/queue/pipeline_log.db "SELECT created_at, worth_remembering, reason, memory_count FROM pipeline_logs ORDER BY created_at DESC LIMIT 5;"
```

Inspect vault synchronization tracking index:
```bash
sqlite3 ~/.lace/memory/vault_hash_index.db "SELECT file_path, mtime, hash FROM hash_index LIMIT 5;"
```

Check graph and vector health:
```bash
lace graph stats
lace memory stats
```
