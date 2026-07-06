---
Type: task
Status: ready-for-agent
Category: bug
---

# Issue 02: Gated Extraction Error Propagation & Retry Loop

## What to build

Ensure transient LLM connection errors propagate to the background worker thread so they can be retried instead of failing silently. Add a `raise_on_llm_error` boolean parameter to the gated extraction process (`process_queue_item`), defaulting to `False` for backwards compatibility. When calling this processor from the worker's queue loop, set `raise_on_llm_error=True` to let failures bubble up to the worker's retry handler.

## Acceptance criteria

- [ ] `process_queue_item` accepts a `raise_on_llm_error` parameter and propagates exceptions if set to `True`.
- [ ] The background worker thread loop calls `process_queue_item` with `raise_on_llm_error=True`.
- [ ] LLM connection errors during queue processing correctly increment retry counts and re-queue jobs.
- [ ] If LLM failures persist past the maximum retry limit, jobs are marked as permanently failed in the database.

## Blocked by

- [Issue 01: Standardized Canonical Path Centralization](file:///home/aayush/Projects/lace/LACE/.scratch/core-reliability/issues/01-path-centralization.md)
