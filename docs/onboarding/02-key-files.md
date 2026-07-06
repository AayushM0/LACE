# Key Files Map

This document lists the ~20 most critical files in the LACE codebase, mapping out what they do, when to read them, and which ones require caution before modifying.

---

## 1. Priority Files — Read These First

Read these files in your first week to understand the core behaviors and interfaces of LACE:

| Priority | Path | What It Does | When to Read |
|----------|------|-------------|-------------|
| 1 | [`src/lace/core/config.py`](file:///home/aayush/Projects/lace/LACE/src/lace/core/config.py) | Configuration management and the canonical path resolver `resolve_lace_paths()`. All other modules must use this resolver to calculate file and index locations. | Day 1 |
| 2 | [`src/lace/memory/models.py`](file:///home/aayush/Projects/lace/LACE/src/lace/memory/models.py) | Pydantic models for core abstractions: `MemoryObject` (frontmatter format), `RetrievalResult`, and `ExtractionCandidate`. | Day 1 |
| 3 | [`src/lace/mcp/tools.py`](file:///home/aayush/Projects/lace/LACE/src/lace/mcp/tools.py) | The implementation layer for MCP tools exposed to LLM clients (e.g. Cursor/Claude Desktop). This includes the `get_relevant_context` and `remember` tools. | Day 2 |
| 4 | [`src/lace/memory/store.py`](file:///home/aayush/Projects/lace/LACE/src/lace/memory/store.py) | High-level orchestration seam. Performs CRUD operations, coordinates the Markdown vault files, vector databases, and graphs. | Day 2 |
| 5 | [`src/lace/retrieval/unified.py`](file:///home/aayush/Projects/lace/LACE/src/lace/retrieval/unified.py) | The 7-step unified retrieval and ranking pipeline. Combines vector search, tag matching, graph neighbors, co-occurrence, and decay. | Day 3 |
| 6 | [`src/lace/mcp/queue.py`](file:///home/aayush/Projects/lace/LACE/src/lace/mcp/queue.py) | SQLite async queue and worker daemon thread. Manages how turns are queued and analyzed by the extractor. | Day 3 |
| 7 | [`src/lace/memory/extractor.py`](file:///home/aayush/Projects/lace/LACE/src/lace/memory/extractor.py) | LLM worthiness verdict rules, prompts, and parser. Defines the criteria for what LACE considers worth remembering. | Day 4 |
| 8 | [`src/lace/memory/dedup.py`](file:///home/aayush/Projects/lace/LACE/src/lace/memory/dedup.py) | The two-tier deduplication check (skip vs. merge vs. store) using cosine similarity. | Day 4 |
| 9 | [`src/lace/vault/sync.py`](file:///home/aayush/Projects/lace/LACE/src/lace/vault/sync.py) | Bidirectional synchronization logic that mirrors files between the LACE vault and your Obsidian vault. | Day 5 |

---

## 2. Dangerous Files — Coordinate Before Modifying

Some files contain load-bearing invariants or affect the system widely. Coordinate changes with the team before modifying these:

| Path | Risk / Invariant | Coordination Required |
|------|------------------|----------------------|
| [`src/lace/core/config.py`](file:///home/aayush/Projects/lace/LACE/src/lace/core/config.py) | Changing default directories or path calculation will break existing installations and break paths. Any path addition must use `resolve_lace_paths()`. | PR review and design alignment. |
| [`src/lace/vault/sync.py`](file:///home/aayush/Projects/lace/LACE/src/lace/vault/sync.py) | Uses regex to extract Memory IDs `mem_<12 hex>` from filenames. If you change the ID format or filename structure, Obsidian sync will fail and potentially duplicate files. | PR review, validation with file watcher enabled. |
| [`src/lace/memory/dedup.py`](file:///home/aayush/Projects/lace/LACE/src/lace/memory/dedup.py) | Enforces the invariant `merge_threshold < skip_threshold` via Pydantic validator `DedupConfig.validate_thresholds()`. Bypassing this will cause deduplication logical errors. | Design review before altering dedup thresholds. |
| [`src/lace/mcp/queue.py`](file:///home/aayush/Projects/lace/LACE/src/lace/mcp/queue.py) | Background daemon worker runs continuously. Any unhandled exception can kill the thread, stopping memory extraction. The SQLite database write must return in <5ms. | Code owner review. Verify error handling with tests. |
| [`src/lace/main.py`](file:///home/aayush/Projects/lace/LACE/src/lace/main.py) | Extremely large file (2,123 lines). Directly manages the Typer CLI and command registration. High risk of merge conflicts and command shadowing. | PR review. Splitting this file is a planned refactor. |
