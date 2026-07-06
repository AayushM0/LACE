---
Type: task
Status: ready-for-agent
Category: enhancement
---

# Issue 01: Standardized Canonical Path Centralization

## What to build

Centralize all default path calculations in LACE. Implement a single path resolver (`resolve_lace_paths`) inside the core configuration module that returns absolute paths for the vault, vector store, SQLite queue database, pipeline log database, vault sync hash index, and co-retrieval tracker file. Update all store, queue, and deduplication modules to call this canonical resolver instead of constructing paths locally.

## Acceptance criteria

- [ ] `resolve_lace_paths()` returns all 6 canonical LACE paths resolved under a single home directory.
- [ ] `MemoryStore` uses the centralized path resolver for its vault, vector store, graph, and co-retrieval paths.
- [ ] Background worker queue logic uses config-resolved paths for queue and pipeline log databases.
- [ ] Deduplication hash index uses the config-resolved database path.
- [ ] Existing tests pass with the new centralized configuration paths.

## Blocked by

None - can start immediately
