# ADR-0001: Decoupled Local Subsystem Architecture vs. Cloud Microservices

## Status

Accepted

## Date

2026-09-16

## Context

Modern AI coding agents (such as Claude Desktop, Cursor, and custom agentic frameworks) require persistent, cross-session memory to recall project conventions, architectural decisions, and debugging runbooks. Without persistent memory, agents repeat mistakes, lose context between IDE sessions, and require manual prompt re-injection.

Two broad architectural paths exist for implementing an agent context engine:

1. **Remote Cloud Microservice (SaaS)**: A networked backend deploying distributed microservices (e.g., in a Kubernetes cluster or AWS ECS), using remote managed vector databases (Pinecone, Qdrant Cloud), external message queues (Kafka, AWS SQS), and REST/gRPC endpoints.
2. **Decoupled Local Subsystem Engine**: An embedded, local-first engine executing directly on the developer's host machine, interfacing via the Model Context Protocol (MCP) over standard I/O streams (`stdio`), using embedded stores (SQLite, ChromaDB, local filesystem), and running asynchronous background threads for worker tasks.

## Decision Drivers

* **Data Sovereignty & Privacy**: Source code, proprietary architectural patterns, and conversation transcripts must never be transmitted to third-party databases without explicit consent.
* **Latency Budget**: The write-path acknowledgement from the context engine must complete in under $100\text{ms}$ to prevent stalling interactive agent turn cycles.
* **Zero Infrastructure Overhead**: Developers should not require Docker daemons, cloud provider accounts, or network configuration to use persistent memory in their editor.
* **Human Inspectability**: The primary storage medium must be non-opaque, editable, and portable (e.g., Markdown notes readable in Obsidian or VS Code).
* **High-Throughput Local Scaling**: The engine must scale to thousands of memory notes without CPU or disk I/O bottlenecks during vault loading or re-indexing.

## Considered Options

### Option 1: Remote Cloud Microservice Cluster (Kubernetes / Managed Cloud)
* **Architecture**: Distributed microservices communicating over HTTP/gRPC, backed by cloud vector databases and central relational databases.
* **Pros**:
  * Multi-user shared state is trivial (team-wide memory sharing).
  * Heavy embedding and extraction computation is offloaded to remote GPU nodes.
* **Cons**:
  * **Data Privacy Barrier**: Transmits proprietary codebase context to external servers, violating enterprise IP policies.
  * **Network Latency**: Network round-trips ($20\text{ms} - 150\text{ms}$) add unnecessary delay to local tool execution.
  * **Operational Cost & Complexity**: Demands cloud hosting, authentication infra, network ingress, and recurring cloud bills.
  * **Offline Failure**: Agent cannot function without active internet connectivity.

### Option 2: Monolithic Synchronous In-Process Memory Store
* **Architecture**: A single synchronous Python script storing context in a monolithic JSON or SQLite file, running LLM extraction inline during tool calls.
* **Pros**:
  * Minimal code complexity.
* **Cons**:
  * **Unacceptable Ingestion Latency**: LLM summarization and extraction take $5\text{s} - 30\text{s}$. Running synchronously blocks the agent tool-call, degrading developer experience.
  * **Concurrency Contention**: File locks during ingestion block concurrent read queries from other editor instances.

### Option 3: Decoupled Local Subsystem Architecture (Selected)
* **Architecture**: A local-first, multi-subsystem engine executing as a background daemon or embedded process:
  * **Interface Layer**: Model Context Protocol (MCP) server communicating over `stdio` via JSON-RPC.
  * **Ingestion Pipeline**: Fast producer-consumer model where `process_interaction` enqueues jobs to an embedded SQLite queue in $<5\text{ms}$, returning immediately. A background worker thread processes LLM extraction and deduplication asynchronously.
  * **Storage Engine**: Human-readable Markdown vault as the single source of truth, mirrored bidirectionally to Obsidian via a `watchdog` filesystem watcher.
  * **Index Layer**: Embedded ChromaDB vector collections alongside an in-memory NetworkX concept graph.
  * **Parallel Loading**: Multi-core parsing with `concurrent.futures.ThreadPoolExecutor` for high-speed cold-starts.

## Decision

We will implement **Option 3: Decoupled Local Subsystem Architecture**.

LACE will run entirely on the developer's local machine, exposing an MCP stdio server to AI clients while maintaining separate execution threads for background queue processing and filesystem monitoring.

## Rationale

1. **Zero Data Egress**: Notes, vector embeddings, and queue state live in `~/.lace/`. Sensitive tokens and proprietary architecture never leave the local environment.
2. **Sub-Millisecond Inter-Process Communication (IPC)**: MCP over `stdio` eliminates TCP/HTTP connection overhead.
3. **Decoupled Write-Path Latency**: Separating the enqueue operation ($<5\text{ms}$) from the LLM extraction worker ($5\text{s} - 30\text{s}$) maintains a $<100\text{ms}$ overall agent tool-call latency.
4. **Anti-Vendor Lock-In**: Because the storage source of truth is Markdown with YAML frontmatter, users retain full access to their knowledge base even if LACE is uninstalled.

## Consequences

### Positive
* **Developer Experience**: Zero setup friction—installable via `uv pip install -e .` and initialized via `lace init`.
* **Resource Efficiency**: Idles at $<150\text{MB}$ RAM with zero external cloud dependencies when using local embedding models (`all-MiniLM-L6-v2`) and local Ollama.
* **Deterministic Concurrency**: SQLite configured with `timeout=10.0` and parameterized queries prevents database locks between concurrent CLI and MCP operations.

### Negative / Trade-offs
* **Local Compute Contention**: Running local extraction models (e.g. Ollama `llama3.2`) consumes local CPU/GPU resources alongside developer workloads.
* **Single-Node Boundary**: Real-time cross-developer memory synchronization is not built-in; synchronization between developers requires Git repository tracking of the Markdown vault.

## Implementation Details

* **Queue Worker**: Implemented in `src/lace/mcp/queue.py` using SQLite database `~/.lace/queue/extraction_queue.db`.
* **Concurrency Protection**: SQLite transactions enforce explicit `timeout=10.0` and parameterized queries (`?`) to prevent thread deadlocks.
* **Parallel I/O**: `src/lace/vault/` uses `ThreadPoolExecutor` to parallelize disk reads across multiple CPU cores during vault re-indexing.
* **Filesystem Sync**: `src/lace/vault/sync.py` uses `watchdog` with file modification time (`mtime`) and content hash tracking (`vault_hash_index.db`) to prevent infinite write loops with external Markdown editors.
