# LACE Onboarding (Senior Engineer)

Audience: senior engineer joining the LACE codebase.
Scope: enough architecture, ownership, and operational knowledge to make changes safely — without re-deriving it from source.

LACE (**Local AI Context Engine**) is a persistent, local-first memory layer for AI tools. It ingests conversation turns, extracts durable knowledge via an LLM, deduplicates it, stores it as human-readable Markdown in a local vault, indexes it with local vector embeddings, links it into a NetworkX knowledge graph, and serves it back to AI clients over the Model Context Protocol (MCP). Everything runs on the user's machine; no server, no API keys for the default path.

> **Already documented elsewhere.** This onboarding focuses on *how to work in the code*. For the product vision see `lace_product_prd.md`; for the phase-by-phase build plan see `implementation_plan.md`; for the older technical reference see `architecture.md` (note: largely superseded by recent phases — trust the code first).

## Read in this order

1. [`01-architecture.md`](01-architecture.md) — system shape, data flow, the extraction pipeline, the 7-step retrieval pipeline. **Start here.**
2. [`02-key-files.md`](02-key-files.md) — the ~20 files that matter, why they matter, and which ones are dangerous to touch.
3. [`03-setup.md`](03-setup-known-gotchas.md) — prerequisites, install, init, and the **known local gotchas** (Python version mismatch, `uv` not on PATH).
4. [`04-runbooks.md`](04-runbooks.md) — common dev tasks: add an MCP tool, run/write tests, change the retrieval weights, add a config field.
5. [`05-debugging.md`](05-debugging.md) — where state lives, where logs live, common errors with exact fixes.

## The 30-second mental model

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

- **Write path** (ingestion): `process_interaction` → SQLite queue → background worker → LLM extractor → dedup (skip/merge/store) → Markdown file + ChromaDB upsert + graph update.
- **Read path** (retrieval): query → 7-step unified pipeline (vector → tag → graph → co-retrieval → score → rank) → Markdown context returned to the client.
- **Two persistence layers, by design**: Markdown vault is the source of truth (human-readable, Obsidian-syncable); ChromaDB + SQLite indexes are *rebuildable* from it (`lace memory reindex`).

## Critical invariants (do not break these)

These are load-bearing assumptions the code makes. Changing them silently will corrupt user data.

1. **`resolve_lace_paths()` is the single path resolver.** No module computes its own default path. Every new path goes through `core/config.py`.
2. **Memory IDs are `mem_<12 hex>`** and are parsed by regex in `vault/sync.py`. If you change the ID format, sync breaks.
3. **Dedup thresholds must satisfy `merge_threshold < skip_threshold`** — enforced by `DedupConfig.validate_thresholds()`. Don't bypass it.
4. **The extraction queue must never block the caller.** `enqueue()` writes to SQLite and returns in `<5ms`; the slow LLM work happens on a daemon worker thread. See `mcp/queue.py` header.
5. **Markdown frontmatter is the canonical memory format.** Vector store and indexes are derived. Never mutate a memory by editing ChromaDB directly — edit the Markdown, then reindex.
6. **`process_interaction` is the only sanctioned ingestion entrypoint** for conversation turns. Bypassing it skips hash suppression, worthiness verdicts, and dedup — which is the entire noise defense.

## Known technical debt & sharp edges

- **`main.py` is 2,123 lines** and holds 9 Typer sub-apps. There is no `commands/` package yet. Splitting it is a known, unstarted refactor.
- **Two `memory search` commands are registered** (`@memory_app.command("search")` appears twice — lines 535 and 679). The second registration shadows the first in Typer. This is a latent bug; investigate before relying on either.
- **`architecture.md` exists but is stale** relative to Phase 2 (graph + co-retrieval + unified pipeline). Trust source over that doc until it's refreshed.
- **Python version drift.** `pyproject.toml` requires `>=3.11`, `.python-version` pins `3.13`, but the checked-in `.venv` is `3.14.6`. Tests may behave differently across these. See `03-setup-known-gotchas.md`.
- **No CI.** No `.github/workflows/`, no GitLab CI. Tests run only locally. Don't trust that "green locally" means "green everywhere."
- **Scratch dirs leak.** `scratch_test_db/` and `verify_lace_fixes.py` are untracked at the repo root and look like ad-hoc verification cruft. `.gitignore` ignores `scratch/` but not these.
