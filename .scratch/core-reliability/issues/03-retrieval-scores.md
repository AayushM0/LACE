---
Type: task
Status: ready-for-agent
Category: enhancement
---

# Issue 03: Explainable Multi-Signal Retrieval Results

## What to build

Expose individual retrieval scoring metrics on the `RetrievalResult` data model. Add float fields for `vector_score`, `tag_score`, `graph_score`, `co_retrieval_score`, `recency_score`, and `confidence_score`. Modify the unified retriever ranking step to populate these values, providing explainability for the final combined relevance ranking.

## Acceptance criteria

- [ ] `RetrievalResult` schema contains individual score fields for all 5 signals.
- [ ] Unified retriever maps internal candidate signal scores to output result fields.
- [ ] Retrieval results returned to AI clients contain granular scores.
- [ ] Search ranking tests assert that signal scores are populated.

## Blocked by

- [Issue 01: Standardized Canonical Path Centralization](file:///home/aayush/Projects/lace/LACE/.scratch/core-reliability/issues/01-path-centralization.md)
