# LACE (Local AI Context Engine)

[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Protocol: MCP](https://img.shields.io/badge/Protocol-MCP%20stdio-purple.svg)](https://modelcontextprotocol.io/)
[![Tests: 584 Passed](https://img.shields.io/badge/Tests-584%20Passed-brightgreen.svg)](tests/)
[![Architecture: Local--First](https://img.shields.io/badge/Architecture-Local--First%20Decoupled-orange.svg)](docs/adr/0001-local-first-decoupled-architecture.md)

**LACE (Local AI Context Engine)** is a local-first context and persistent memory engine designed for AI coding agents and developer environments (Claude Desktop, Cursor, and custom agent harnesses). It stores memories as human-readable Markdown notes in a local vault, maintains an embedded vector index and a concept graph, and exposes them over the Model Context Protocol (MCP).

By running locally and decoupling the write path via an embedded SQLite queue, LACE provides persistent cross-session memory without transmitting proprietary code to cloud databases or stalling interactive agent loops.

---

## Architecture & System Design

Rather than introducing cloud microservices overhead, LACE is engineered as an **embedded, local-first system with decoupled background subsystems**. Inter-process communication (IPC) runs over standard I/O (`stdio`), and write latency is bounded by a producer-consumer SQLite worker.

```
                  AI Client (Claude Desktop / Cursor / IDE)
                                      │
                         Model Context Protocol (JSON-RPC over stdio)
                                      ▼
                        ┌───────────────────────────┐
                        │   src/lace/mcp/server.py  │  ◄── Typer CLI (`lace`)
                        └─────────────┬─────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │   src/lace/memory/store.py│  (CRUD & Retrieval Seam)
                        └──────┬──────────────┬─────┘
                               │              │
        ┌──────────────────────┘              └─────────────────────┐
        ▼                                                           ▼
[ Ingestion Pipeline ]                                     [ Retrieval Engine ]
 1. Write to SQLite queue (<5ms)                            1. ChromaDB Cosine Search
 2. Worker pre-filters (<100 chars)                         2. Tag Scan Expansion
 3. LLM evaluates worthiness                                3. NetworkX Topological Walk
 4. Two-Tier Deduplication:                                 4. Co-Retrieval Boost
    • >95% similarity: Drop duplicate                       5. Recency Decay Half-Life
    • 85–95% similarity: Merge into note                    6. User Confidence Weighting
    • <85% similarity: Write new note                       7. Filter & Rank (Score >= threshold)
        │                                                           │
        ▼                                                           ▼
~/.lace/queue/extraction_queue.db                          Compressed Markdown Context Block
        │
        ▼
~/.lace/memory/vault/ (Markdown Source of Truth) ◄──► Obsidian Sync Daemon (`watchdog` + mtime)
```

For the formal evaluation of why this local-first architecture was selected over remote cloud clusters (e.g. Kubernetes/SaaS), see [ADR-0001: Decoupled Local Subsystem Architecture](docs/adr/0001-local-first-decoupled-architecture.md).

---

## Core Engineering Highlights

* **Decoupled Asynchronous Write Path**: Interactive agent turns require sub-second responses. LACE accepts incoming conversation turns via `process_interaction`, enqueues them into an embedded SQLite database in **$<5\text{ms}$**, and returns immediately. A detached background worker thread handles slow LLM extraction ($5\text{s} - 30\text{s}$) asynchronously.
* **Deterministic Concurrency Management**: SQLite database connections enforce parameterized queries (`?`) and `timeout=10.0` locks running in WAL mode, eliminating database contention between concurrent CLI commands and MCP daemon threads.
* **Two-Tier Vector Deduplication**: Prevents index bloat and duplicate memory creation during agent sessions:
  * **$> 95\%$ Cosine Similarity**: Candidate dropped (suppressed).
  * **$85\% - 95\%$ Cosine Similarity**: Candidate merged into the existing Markdown file (appends new details and unions tags).
  * **$< 85\%$ Cosine Similarity**: Stored as a new memory note.
* **7-Step Multi-Signal Retrieval**: Evaluates candidate notes through a weighted scoring equation:
  $$\text{Score} = w_{\text{vector}} S_{\text{vector}} + w_{\text{tag}} S_{\text{tag}} + w_{\text{graph}} S_{\text{graph}} + w_{\text{co}} S_{\text{co}} + w_{\text{recency}} S_{\text{recency}} + w_{\text{confidence}} S_{\text{confidence}}$$
* **Multi-Threaded Vault Ingestion**: Uses Python's `concurrent.futures.ThreadPoolExecutor` to parallelize disk I/O when reading and indexing thousands of notes during cold starts.
* **Bidirectional Obsidian Sync**: Employs `watchdog` to monitor filesystem changes, using modification times (`mtime`) and SHA-256 hashes in `vault_hash_index.db` to synchronize edits without infinite write loops.

---

## Quickstart (Under 2 Minutes)

### 1. Prerequisites
* Python `>= 3.11`
* [uv](https://github.com/astral-sh/uv) (recommended) or `pip`

### 2. Installation
```bash
git clone https://github.com/AayushM0/LACE.git
cd LACE
uv pip install -e .
```

### 3. Initialize Environment
Provisions the default configuration (`~/.lace/config/lace.yaml`) and vault directories (`~/.lace/memory/vault/`):
```bash
lace init
```

### 4. Wire into AI Clients (MCP Setup)

LACE communicates with any MCP-compliant client via standard I/O. Add the following entry to your client configuration:

#### For Claude Desktop
Edit `claude_desktop_config.json`:
* **macOS**: `~/Library/Application Support/Claude/claude_desktop_config.json`
* **Windows**: `%APPDATA%\Claude\claude_desktop_config.json`
* **Linux**: `~/.config/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "lace": {
      "command": "uv",
      "args": ["run", "--directory", "/absolute/path/to/LACE", "lace", "mcp", "start"]
    }
  }
}
```

#### For Cursor
Add to Cursor Settings > Features > MCP > Add New MCP Server:
* **Name**: `lace`
* **Type**: `command`
* **Command**: `uv run --directory /absolute/path/to/LACE lace mcp start`

---

## Verified CLI Command Reference

### System & Configuration
| Command | Description |
| :--- | :--- |
| `lace init [--home <PATH>]` | Initialize default directory tree and YAML configuration |
| `lace version` | Print installed package version |
| `lace config show` | Print resolved configuration settings |
| `lace config set <key> <value>` | Update a specific configuration parameter using dot notation |

### Scope Management
| Command | Description |
| :--- | :--- |
| `lace project detect` | Automatically resolve project scope from the active Git root |
| `lace project create <name>` | Explicitly initialize a named project scope |
| `lace project switch <name>` | Set the active project scope |
| `lace project list` | List all registered project scopes |
| `lace session start` | Begin an isolated, ephemeral session memory tree |
| `lace session stop` | Terminate active ephemeral session and restore baseline scope |

### Memory CRUD & Search
| Command | Description |
| :--- | :--- |
| `lace memory add "<content>"` | Manually add a memory note with optional `--tag` and `--category` |
| `lace memory search "<query>"` | Semantically search memories with composite scoring (`--scores`) |
| `lace memory list` | Display tabular list of active memories in scope |
| `lace memory show <id>` | Print memory body, frontmatter metadata, and history |
| `lace memory forget <id>` | Soft-delete a memory note (archives file, removes from vector index) |
| `lace memory rate <id> <score>` | Provide feedback (`helpful`, `outdated`, `wrong`) to adjust confidence |
| `lace memory reindex` | Re-embed all Markdown vault files into ChromaDB |
| `lace memory stats` | Output collection sizes, memory counts, and retrieval latency |

### Knowledge Graph & Obsidian Sync
| Command | Description |
| :--- | :--- |
| `lace graph build` | Reconstruct NetworkX concept network from vault links and tags |
| `lace graph stats` | Display node, edge, and density statistics |
| `lace graph related <concept>` | Perform Breadth-First Search (BFS) for related concepts |
| `lace wikilink inject` | Automatically inject `[[wikilinks]]` into matching vault notes |
| `lace vault sync` | Execute bidirectional synchronization with Obsidian vault |
| `lace vault watch` | Launch real-time background filesystem monitor for Obsidian vault |

---

## Directory & State Layout

All persistent application data resides in `~/.lace/` (overrideable via `LACE_HOME`):

```
~/.lace/
├── config/
│   ├── lace.yaml               # Global configuration file
│   ├── identity.md             # Injected agent persona definition
│   └── preferences.yaml        # User-defined tool and coding preferences
├── memory/
│   ├── vault/                  # Single source of truth (human-readable Markdown)
│   │   ├── global/             # Global memories
│   │   └── projects/           # Scoped project directories
│   ├── vector_db/              # Embedded ChromaDB collections
│   ├── vault_hash_index.db     # SQLite index tracking file mtimes and SHA-256 hashes
│   ├── graph.json              # Serialized NetworkX concept graph
│   └── co_retrieval.json       # Frequency co-occurrence matrix
├── queue/
│   ├── extraction_queue.db     # Asynchronous worker job queue
│   └── pipeline_log.db         # Extraction worthiness verdicts & audit log
└── logs/
    ├── retrieval/              # Query scoring traces and latency benchmarks
    └── interactions/           # Logged conversation turns
```

---

## Testing & Quality Verification

LACE enforces strict automated testing across data models, SQLite queue concurrency, vector retrieval, graph parsing, and MCP tool protocols.

```bash
# Run the complete test suite
pytest

# Run with coverage report
pytest --cov=lace tests/

# Test specific subsystems
pytest tests/test_mcp/
pytest tests/test_retrieval/
pytest tests/test_memory/
```

**Verification Results**:
* **584 passing tests** across 33 test suites.
* Zero external cloud services required to execute unit or integration tests.

---

## Technical Documentation Index

For deeper architectural breakdowns, runbooks, and domain terminology:

* [**Developer Onboarding Guide**](docs/ONBOARDING.md) — 10-minute setup, key file map, developer runbooks, and real error debugging.
* [**Domain Context & Glossary**](CONTEXT.md) — Ubiquitous vocabulary (Vault, Scopes, 7-Step Retrieval, Two-Tier Dedup).
* [**ADR-0001: Local-First Decoupled Architecture**](docs/adr/0001-local-first-decoupled-architecture.md) — Design decision record on why LACE uses decoupled local subsystems over cloud microservices.
* [**MCP API Reference**](docs/API.md) — Detailed tool signatures, parameters, and MCP resource URI schemes.

---

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
