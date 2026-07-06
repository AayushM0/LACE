---
Status: ready-for-agent
---

# Product Requirements Document (PRD): LACE Core Reliability Fixes

## Problem Statement

Developers using LACE (Local AI Context Engine) as a persistent long-term memory layer frequently run into silent failure states:
1. Real bugs and connection errors occur during background LLM memory extraction without any log output, tracebacks, or visible indicators, making diagnosing problems impossible.
2. If the local LLM server (Ollama) is temporarily offline or unreachable, extraction fails silently and marks the queue job as `done` instead of retrying. This leads to silent and permanent memory loss.
3. Path resolution is fragmented across modules, with various defaults constructed independently. This leads to data drift, synchronization loops, or missing vector embeddings when running in custom LACE home paths.
4. Codebase contributors and end-users have no unified way to check if their LACE installation (path permissions, pipeline logs, embedding generation, database connectivity) is fully functional.

---

## Solution

Implement a comprehensive reliability hardening of the LACE ingestion, extraction, and retrieval systems:
1. **Error Propagation and Retries**: Pass a `raise_on_llm_error` flag to the gated extraction processor, ensuring LLM connection failures propagate to the background worker loop, triggering retries and eventual failures rather than silent completions.
2. **Canonical Path Centralization**: Standardize all path calculations across the vault, vector DB, queue database, pipeline log database, hash index, and co-retrieval files under a single resolver (`resolve_lace_paths`).
3. **Observability and Diagnostics**: Implement a CLI `doctor` subcommand that performs a suite of diagnostic checks, including path access verification, pipeline log auditing, confidence spread detection, reindex backlog counting, and a mocked LLM round-trip wiring test.
4. **Enhanced Data Contracts**: Expose granular signal scores (vector, tag, graph, co-retrieval, recency, and confidence) on `RetrievalResult` objects to make the multi-signal ranking explainable.

---

## User Stories

1. As a developer using LACE, I want background extraction errors to be propagated and logged, so that I can debug why a memory was not stored.
2. As a developer, I want transient LLM connection errors to trigger queue retries, so that my memories are not lost when Ollama is temporarily offline.
3. As a developer, I want all LACE database and file paths to be resolved from a single configuration resolver, so that I do not run into file conflicts or stale path defaults.
4. As an AI client, I want retrieval results to contain granular signal scores (vector, tag, graph, co-retrieval, recency, confidence), so that I can inspect and explain the relevance ranking.
5. As a developer, I want a `lace doctor` CLI command to check my installation health, so that I can quickly verify path access, backlog sizes, and pipeline wiring.
6. As a developer, I want the queue worker to run synchronously during tests and diagnostic checks, so that I can verify end-to-end functionality without asynchronous race conditions.
7. As a developer, I want the `lace doctor` tool to run a mocked LLM round-trip check, so that I can verify storage and retrieval wiring without consuming live API tokens.
8. As a developer, I want a warning if the LLM extraction continuously returns a flat default confidence rating (like 0.4), so that I know the LLM is ignoring the scoring rubric.
9. As a developer, I want a reindex backlog indicator in the health check, so that I can see if any memories in the vault are missing vector embeddings.
10. As a codebase contributor, I want a dedicated integration test suite covering pipeline reliability, so that I can make changes without breaking the ingestion and retrieval flows.

---

## Implementation Decisions

### Modified Modules
- **Core Configuration**: Standardize path calculations using a single resolver mapping the markdown vault, vector store, SQLite queue, pipeline log, sync hash index, and co-retrieval tracker.
- **Extraction Queue & Worker**:
  - Implement strict error propagation from the extraction processor to the worker retry loop to retry transient errors.
  - Avoid silent swallowing of queue insertion errors.
  - *Prototype reference:* The background worker's `_process_single_job` handles transient LLM errors by propagating exceptions through the gated extraction processor:
    ```python
    memories = process_queue_item(
        item=job,
        config=config,
        log_db_path=Path(log_db_path) if isinstance(log_db_path, str) else log_db_path,
        raise_on_llm_error=True,
    )
    ```
- **Memory Store CRUD**: Ensure that embedding generation failures flag the memory note for reindexing (setting `needs_reindex = True` on the Markdown frontmatter) rather than failing silently, and log errors with full tracebacks.
- **Unified Retrieval**: Add fields to the `RetrievalResult` schema to capture individual signal scores.
- **CLI Subcommand**: Add a `doctor` subcommand implementing a Table console readout for system diagnostics.

---

## Testing Decisions

- **Test Boundaries**: Tests must verify external behaviors (queue retries on LLM offline, scope propagation, confidence spreads, vector store write verification, retrieval ranking scoring) rather than internal implementation details.
- **Tested Components**: Gated extraction queue, background worker retries, scope threading, confidence variance, vector store reach, and retrieval round-trip.
- **Test Infrastructure**: Standardize integration test suites under isolated local LACE home paths, mocking external API calls (LLM and embedding models) while validating sqlite, vault files, and vector indices.

---

## Out of Scope

- Live LLM prompt quality tuning (the `doctor` command only tests wiring and prompt structure, not the semantic quality of the extraction).
- Automated CI/CD pipelines (testing remains local-first for development verification).

---

## Further Notes

- A complete test suite covering these reliability fixes has been implemented at `tests/test_pipeline_reliability.py`. All tests have been executed and passed.
