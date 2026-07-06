---
Type: task
Status: ready-for-agent
Category: enhancement
---

# Issue 04: LACE Doctor Diagnostic CLI Command

## What to build

Implement a CLI command `lace doctor` to verify LACE installation and wiring health. The command must run the following checks and print a status table:
1. Verify LACE_HOME and configured database paths exist and are non-empty.
2. Assert `PIPELINE_LOG_DB_PATH` matches the configuration resolver path.
3. Check for recent extraction events and verify at least one worthiness verdict is logged.
4. Detect abnormal confidence spreads (flag if >90% of notes share a single confidence value).
5. Check if any memories in the vault are flagged as needing reindexing.
6. Run a mock LLM round-trip test enqueuing a check item, running extraction synchronously, storing it, and verifying vector search recall.

## Acceptance criteria

- [ ] `lace doctor` CLI command is registered and outputs a status table.
- [ ] Command fails or flags errors if paths are missing, or log databases are unreachable.
- [ ] Round-trip verification uses a mocked LLM and does not execute real API calls.
- [ ] Warnings are displayed for flat confidence spreads and reindex backlogs.

## Blocked by

- [Issue 01: Standardized Canonical Path Centralization](file:///home/aayush/Projects/lace/LACE/.scratch/core-reliability/issues/01-path-centralization.md)
- [Issue 02: Gated Extraction Error Propagation & Retry Loop](file:///home/aayush/Projects/lace/LACE/.scratch/core-reliability/issues/02-error-propagation.md)
- [Issue 03: Explainable Multi-Signal Retrieval Results](file:///home/aayush/Projects/lace/LACE/.scratch/core-reliability/issues/03-retrieval-scores.md)
