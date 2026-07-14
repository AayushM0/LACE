# LACE Model Context Protocol (MCP) API Reference

This document details the tools, schemas, and resource definitions exposed by LACE over the Model Context Protocol (MCP) stdio interface.

---

## 1. MCP Tools

### `initialize_lace_session`
Initializes a new LACE session, resolves the active project scope, and triggers context warm-up.

* **Arguments**:
  * `working_directory` (string, required): The absolute path of the active project workspace root.
* **Returns**:
  ```json
  {
    "status": "active",
    "project": "project_name",
    "cwd": "path",
    "message": "LACE session active..."
  }
  ```

---

### `get_relevant_context`
Queries the database and vault to compile and format relevant memories into a unified markdown prompt injection block.

* **Arguments**:
  * `query` (string, required): The user message or question context to search against.
* **Returns**: A formatted Markdown string containing relevant guidelines, decisions, preferences, and code patterns.

---

### `process_interaction`
Asynchronously enqueues a completed conversation turn to be analyzed by the background worker for memory extraction.

* **Arguments**:
  * `query` (string, required): The user prompt.
  * `response` (string, required): The assistant's response.
  * `context_hint` (string, optional): One of `debugging_insight`, `architectural_decision`, `user_preference`, `repeated_action`, `general_knowledge`.
* **Returns**:
  ```json
  {
    "status": "queued",
    "job_id": "uuid",
    "queue_id": "uuid",
    "scope_used": "scope",
    "action": "inserted|suppressed"
  }
  ```

---

### `remember`
Explicitly writes a new memory note into the markdown vault and indexes it into the local vector database.

* **Arguments**:
  * `content` (string, required): The complete knowledge or information to store.
  * `category` (string, required): One of `decision`, `pattern`, `preference`, `reference`, `debug`.
  * `tags` (array of strings, required): Tags for structural graphing and search indexing.
  * `scope` (string, optional): Specific scope (e.g. `global` or `project:name`).
  * `confidence` (number, optional, default: `0.7`): Score indicating information reliability.
* **Returns**:
  ```json
  {
    "status": "stored",
    "memory_id": "mem_uuid"
  }
  ```

---

### `search_memory`
Performs a composite five-signal relevance search on memories.

* **Arguments**:
  * `query` (string, required): Search query string.
  * `scope` (string, optional): Scope to restrict query search.
  * `max_results` (integer, optional, default: `10`): Max search hits.
* **Returns**: Array of retrieval results showing composite score breakdowns.

---

### `list_memories`
Lists existing memories in the active scope matching filters.

* **Arguments**:
  * `scope` (string, optional): Scope filter.
  * `category` (string, optional): Category filter.

---

### `forget_memory`
Archives a memory by moving its lifecycle state to `archived`, which skips it during retrieval but preserves the file.

* **Arguments**:
  * `memory_id` (string, required): ID of the memory note to forget.

---

### `get_project_context`
Returns structured workspace context, including architecture files and local preferences.

* **Arguments**:
  * `project_name` (string, optional): Target project name.

---

### `get_related_concepts`
Traverses the NetworkX concept graph using Breadth-First Search (BFS) starting from a specific node.

* **Arguments**:
  * `concept` (string, required): Starting node name.

---

### `set_context`
Sets session or project scope overrides explicitly.

* **Arguments**:
  * `session_id` (string, optional): Target session ID.
  * `project_name` (string, optional): Target project name.

---

## 2. MCP Resources

LACE exposes standard static resources under the `memory://` URI protocol scheme.

### `memory://instructions`
Retrieves the LACE Active Memory Protocol rules, detailing tool sequences, metadata checks, and ingestion validation boundaries.

### `memory://project-context`
Retrieves resolved project-scoped context rules, local directory shapes, and ADR summaries.

### `memory://patterns`
Retrieves repeating development conventions, formatting shapes, and coding structures.

### `memory://decisions`
Retrieves architectural ADR notes and design choices.

### `memory://debug-log`
Retrieves troubleshooting logs, stack traces, and runbooks.
