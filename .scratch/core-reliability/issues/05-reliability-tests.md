---
Type: task
Status: ready-for-agent
Category: enhancement
---

# Issue 05: Pipeline Integration & Reliability Test Suite

## What to build

Create a regression test suite under `tests/test_pipeline_reliability.py` to protect the reliability improvements. The suite must cover the full pipeline end-to-end with config-resolved database paths, mocked LLM and embedding functions, and explicit scopes. Verify:
1. Gated extraction worker uses the correct new pipeline and retry/failure behavior on LLM connection errors.
2. Active scopes are threaded properly and enqueued project-scoped jobs store project-scoped memories.
3. Gated out/rejected extractions are logged as verdicts to `pipeline_log.db` without writing to the vault.
4. Confidence scores differ between extracted memories (rubric is respected).
5. Stored memories can be recalled semantically with correct signal score breakdowns.

## Acceptance criteria

- [ ] All tests in `tests/test_pipeline_reliability.py` pass.
- [ ] Test suite executes quickly using mocked endpoints.
- [ ] No test creates files or directories outside a temporary isolated test home path.

## Blocked by

- [Issue 02: Gated Extraction Error Propagation & Retry Loop](file:///home/aayush/Projects/lace/LACE/.scratch/core-reliability/issues/02-error-propagation.md)
- [Issue 03: Explainable Multi-Signal Retrieval Results](file:///home/aayush/Projects/lace/LACE/.scratch/core-reliability/issues/03-retrieval-scores.md)
- [Issue 04: LACE Doctor Diagnostic CLI Command](file:///home/aayush/Projects/lace/LACE/.scratch/core-reliability/issues/04-doctor-command.md)
