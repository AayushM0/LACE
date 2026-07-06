# Codebase Debugging Guide

This guide describes how to inspect state, locate logs, and resolve common development errors within LACE.

---

## 1. Where State and Indexes Live

LACE is local-first, meaning all state is stored under your user home directory (usually `~/.lace/` unless overridden via the `LACE_HOME` environment variable).

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

---

## 2. Where Logs Live

- **CLI / Stdout & Stderr**: When running commands via the Typer CLI, standard logs are printed to `stderr` since stdout is reserved for CLI data or pipe outputs.
- **MCP Server Stderr**: The MCP server writes trace logs to `stderr`. LLM clients like Cursor capture this stderr. You can read these logs in your client's developer console or output panes.
- **Structured Logs**:
  * Retrieval performance and query metrics are logged in JSON lines format under `~/.lace/logs/retrieval/`.
  * Interaction records are stored in `~/.lace/logs/interactions/`.

---

## 3. Useful Diagnostic Commands

### Inspecting the Ingestion Queue
To view the status of enqueued background extraction jobs:
```bash
sqlite3 ~/.lace/queue/extraction_queue.db "SELECT id, status, retry_count, error_msg FROM extraction_queue ORDER BY created_at DESC LIMIT 5;"
```

### Inspecting Pipeline Worthiness Verdicts
To see which interactions were accepted or rejected by the worthiness LLM gate:
```bash
sqlite3 ~/.lace/queue/pipeline_log.db "SELECT created_at, worth_remembering, reason, memory_count FROM pipeline_logs ORDER BY created_at DESC LIMIT 5;"
```

### Checking Vault and Graph Health
Use the CLI to check health metrics:
```bash
# Check memory storage statistics and indexing latency
.venv/bin/lace memory stats

# Check node and edge count in the NetworkX graph
.venv/bin/lace graph stats
```

---

## 4. Common Errors and Fixes

### Error: `connect ECONNREFUSED 127.0.0.1:11434`
- **Cause**: Ollama is not running, or is running on a port other than the default `11434`.
- **Fix**: Run `ollama serve` in a separate terminal. If running on a different port, update the configuration:
  ```bash
  .venv/bin/lace config set provider.ollama.host "http://localhost:your-port"
  ```

### Error: `sqlite3.OperationalError: database is locked`
- **Cause**: SQLite databases are accessed concurrently by the main MCP thread and the background worker daemon, and a thread is holding a transaction open too long.
- **Fix**: LACE initializes database connections with a `timeout=10.0` parameters to wait for locks to clear. Ensure you close all database cursors and connections inside `finally` blocks.

### Issue: Search results mismatch (Index is stale)
- **Cause**: Markdown files in the vault were edited or deleted manually, but the derivative vector store (ChromaDB) and graph index did not update.
- **Fix**: Trigger a full reindex to rebuild the vector database and graph directly from the Markdown notes:
  ```bash
  .venv/bin/lace memory reindex
  ```
