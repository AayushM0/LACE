# LACE Domain Context & Glossary

This document establishes the canonical domain language and technical concepts used across the LACE codebase, documentation, and architectural decisions.

---

## 1. Core Architectural Concepts

### Vault
The local filesystem directory tree (`~/.lace/memory/vault`) serving as the authoritative single source of truth for all stored knowledge. Every memory is stored as an individual, human-readable Markdown file (`.md`) with standardized YAML frontmatter.

### Memory Object (`MemoryObject`)
The core domain model defined in `src/lace/memory/models.py`. Encapsulates:
* `id`: Unique identifier formatted as `mem_<12 hex digits>` (e.g. `mem_a1b2c3d4e5f6`).
* `title`: Concise, descriptive name for the memory.
* `content`: The raw text containing the insight, pattern, convention, or decision.
* `category`: Broad classification (`tech-stack`, `decision`, `pattern`, `preference`, `debugging`).
* `tags`: Array of semantic keywords used for exact tag expansion.
* `scope`: Target visibility domain (`global`, project-specific, or session-scoped).
* `confidence`: A floating-point rating `[0.0 - 1.0]` representing reliability, refined through user feedback (`lace memory rate`).
* `created_at` / `updated_at`: ISO-8601 UTC timestamps used for recency decay calculation.

### Ingestion Queue & Worker
The decoupled producer-consumer pipeline implemented in `src/lace/mcp/queue.py`:
* **Producer (`process_interaction`)**: Receives conversation turns from the AI client and enqueues them into an embedded SQLite database (`~/.lace/queue/extraction_queue.db`) in $<5\text{ms}$.
* **Consumer (Background Thread)**: Polls the SQLite queue, performs pre-filtering, calls the extraction model, executes deduplication, and writes new Markdown notes without blocking the AI client.

### Worthiness Gate
A two-stage filtering layer inside `src/lace/memory/extractor.py` that defends the memory vault against noise:
1. **Deterministic Pre-filter**: Discards short strings ($<100$ characters), greetings ("thanks", "ok"), shell command error traces, and repeated queries.
2. **LLM Evaluation**: Evaluates whether the interaction contains reusable architectural decisions, preferences, or debugging insights. Rejected turns are logged to `pipeline_log.db` for auditability.

### Two-Tier Deduplication
The similarity analysis mechanism inside `src/lace/memory/dedup.py` that prevents vector database bloat during ingestion:
* **Cosine Similarity $> 95\%$**: Discard candidate. The memory is already fully known.
* **Cosine Similarity $85\% - 95\%$**: Merge candidate. Existing Markdown note is updated, appending new details and unioning tag arrays.
* **Cosine Similarity $< 85\%$**: Create candidate. A new Markdown file is written to the vault and indexed in ChromaDB.

---

## 2. Retrieval & Knowledge Graph

### 7-Step Unified Retrieval Pipeline
The composite search and scoring algorithm defined in `src/lace/retrieval/unified.py`:
1. **Vector Search**: Queries ChromaDB for semantic cosine distance candidates.
2. **Tag Expansion**: Extracts keywords from the query and fetches exact-matching vault notes.
3. **Graph Expansion**: Traverses NetworkX topological neighbors for concept-linked notes.
4. **Co-Retrieval Boost**: Increases scores for notes frequently retrieved together in previous sessions.
5. **Lazy Hydration**: Resolves full `MemoryObject` models for all candidate IDs.
6. **Multi-Signal Scoring**: Evaluates a weighted linear combination:
   $$\text{Score} = w_{\text{vector}} S_{\text{vector}} + w_{\text{tag}} S_{\text{tag}} + w_{\text{graph}} S_{\text{graph}} + w_{\text{co}} S_{\text{co}} + w_{\text{recency}} S_{\text{recency}} + w_{\text{confidence}} S_{\text{confidence}}$$
   *(Constraint: $\sum w_i = 1.0$)*
7. **Threshold Filtering & Ranking**: Discards results below `relevance_threshold` and sorts descending.

### Knowledge Graph
A directed concept and entity network maintained via NetworkX (`~/.lace/memory/graph.json`). Nodes represent concepts or memory IDs; edges represent semantic co-occurrence or Obsidian-style `[[wikilinks]]`.

---

## 3. Scoping & Protocols

### Scopes
The hierarchy used to partition context and avoid cross-project contamination:
* **`global`**: Shared knowledge accessible across all directories and tools.
* **`project`**: Project-specific context resolved dynamically by walking parent directories for a Git root or `.lace/project.yaml`.
* **`session`**: Ephemeral, short-lived memory trees isolated to a specific debugging session or task.

### Model Context Protocol (MCP) Stdio
The JSON-RPC standard used by LACE to communicate directly with IDEs and AI coding tools (Claude Desktop, Cursor). Running over standard input/output streams (`stdio`), it eliminates network configuration and external port bindings.

### Obsidian Sync Daemon
A background filesystem synchronization service (`src/lace/vault/sync.py`) using Python's `watchdog`. It monitors external edits in an Obsidian vault, indexing changes into ChromaDB while using modification timestamps (`mtime`) and SHA-256 hashes in `vault_hash_index.db` to break recursive write loops.
