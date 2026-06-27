# LACE Multi-Signal Retrieval — Implementation Plan

## Background

LACE currently has one retrieval signal: cosine vector similarity. Tags are stored and ignored. `related_ids` is stored and ignored. Categories are stored and ignored. The retrieval ranking has five factors but four of them — recency, frequency, confidence, scope — are metadata adjustments on top of a single candidate set that vector search alone produces. If a memory doesn't make it through vector search, nothing else can save it.

This plan upgrades that to a six-signal pipeline where vector search remains primary but four additional signals can bring in memories vector search missed, boost memories that have proven useful, and eventually learn which memories belong together.

---

## The Four Failures Being Fixed

**Failure 1 — The Threshold Cliff.** Vector similarity is a hard gate. A memory scoring 0.29 against a 0.30 threshold disappears completely even if it's explicitly linked to the top-scoring memory. There's no mechanism to pull it in via that relationship.

**Failure 2 — No Structural Knowledge.** The system doesn't know "jwt" and "auth" and "token-expiry" belong to the same conceptual cluster. It only compares sentence embeddings. Two memories about the same topic written differently can have surprisingly low cosine similarity and the system treats them as unrelated.

**Failure 3 — No Learning.** Every query starts from zero. If Memory A and Memory B have been retrieved together successfully 50 times, the system has no record of that. It recomputes from scratch every time.

**Failure 4 — related_ids is Decoration.** Someone manually linked Memory A to Memory B. That relationship is stored in the markdown file and completely ignored during retrieval. The human signal is discarded.

---

## Guiding Principles

**Vector search stays primary.** It is not being replaced — it is being augmented. Every new signal adds candidates or adjusts scores on top of what vector search returns. Vector search is still 45% of the final score.

**Everything must be explainable.** Every returned memory should carry a record of why it was returned: which signals contributed, what each signal score was, whether it came from vector search or graph expansion or tag matching. This makes the system debuggable.

**Signals should learn from usage.** The best indicator of which memories belong together is which memories have actually been useful together. The system should track this and use it.

---

## Phase 1 — Tag Index

### What it is

Right now "tags" is a comma-joined string sitting in ChromaDB metadata, never queried. This phase builds an actual inverted index in memory: a structure that maps from each tag to the set of memory IDs that have it, and from each memory ID to the set of tags it has.

This is not a database. It is a Python dictionary, built at startup by scanning all existing memory files, and updated incrementally as memories are added or archived.

### What it enables

**Tag-based candidate expansion.** When a query arrives, the system can extract words from the query text that match known tags, then immediately pull every memory with those tags into the candidate pool. These candidates don't automatically get returned — they enter the pool and compete in the final ranking like everything else. This is the difference between "search only what vector search found" and "also look at everything explicitly tagged with what you're asking about."

**Tag specificity weighting.** Not all tags are equal. A tag like "jwt" that appears on 3 memories is a strong signal when two memories share it. A tag like "general" that appears on 180 memories is noise. The index lets you compute this: each tag's contribution to a match is weighted by `1 / number_of_memories_with_that_tag`. Rare tags carry more signal. This is the same principle as TF-IDF in classical information retrieval, applied to memory tags.

**Tag-based similarity between memories.** Two memories sharing three specific tags are clearly related even if their sentence embeddings don't happen to be similar. The index lets you compute this overlap efficiently without scanning all memory pairs.

### Query-to-tag matching

Matching query text to tags is not pure exact string matching. It also handles obvious variations: partial matches ("authentication" → "auth"), plural/singular ("tokens" → "token"), and common prefix overlap for 4+ character words ("authenticate" → "auth"). This is simple string heuristics, not full NLP — the goal is catching obvious matches without adding complex dependencies.

### Lifecycle

The index is built once at startup from all memory files. It is updated incrementally: when a new memory is stored, its tags are added to the index. When a memory is archived, its tags are removed. It never needs to be fully rebuilt during a running session because incremental updates are exact — there is no approximation involved.

---

## Phase 2 — The Memory Graph

### What it is

A graph where each node is a memory and each edge represents a typed, weighted relationship between two memories. The graph lives in memory at runtime. It is persisted as a JSON file alongside the memory store and rebuilt at startup.

### Edge types

**Explicit edges** come from the existing `related_ids` field. For the first time, when someone sets `related_ids` on a memory, that relationship actually does something. It becomes a high-weight edge (weight 1.0) in the graph. This directly fixes Failure 4.

**Tag similarity edges** are automatically created between memories sharing two or more specific tags. Two memories sharing one tag might be coincidental. Two sharing three specific tags are almost certainly related. Edge weight is computed using Jaccard similarity: the size of the tag intersection divided by the size of the tag union, giving a score between 0 and 1. The tag index from Phase 1 is used to compute this efficiently at startup.

**Co-retrieval edges** are created and strengthened over time as memories are retrieved together. These originate from Phase 3. The graph stores them but doesn't generate them itself.

**Temporal edges** connect memories created within a short time window of each other (default: 30 minutes). If five memories were created in a single working session, they were probably related to the same context. These are low-weight edges (around 0.3) because temporal co-creation is a weak signal — two memories might have been created in the same session for unrelated reasons.

**Supersedes edges** are directional. They point from a newer memory to an older one that covers the same ground. If Memory B has the same category, overlapping tags, and equal or higher confidence than Memory A, and was created later, Memory B probably represents updated information about the same topic. This edge type lets the system prefer newer memories and identify when old information has been superseded.

### Graph traversal during retrieval

When vector search returns its top results, those results become starting nodes in the graph. The traversal fans out from each starting node — following all edge types up to a configurable depth, typically one or two hops — and adds discovered neighbors to the candidate pool.

Weight decays with distance: a direct neighbor of a top result carries the full edge weight as its graph score. A two-hop neighbor gets half of that. A three-hop neighbor gets a quarter. This prevents graph expansion from pulling in distantly-related memories at full strength.

This directly fixes Failure 1. Memory B, which scored 0.29 against a 0.30 threshold, is a direct graph neighbor of Memory A (which scored 0.85). Graph traversal adds Memory B to the candidate pool. Its final combined score — vector 0.29 × 0.45 weight plus graph 0.85 × 0.15 weight — clears the threshold. The explicit relationship actually matters now.

### Automatic graph building at startup

The graph is not purely manual. At startup the graph builder runs three passes:

First pass: explicit edges from all `related_ids` fields across all memories.

Second pass: tag similarity edges — for every pair of memories sharing two or more specific tags, compute Jaccard similarity and create an edge if it exceeds a minimum threshold (around 0.2). The tag index makes this efficient: you only compare memories that share at least one tag.

Third pass: temporal clustering — for memories created within the time window, create weak temporal edges.

This means the graph has real structure immediately, even before any usage and before anyone has manually set `related_ids`.

### Persistence

The graph is saved as a JSON file listing all edges with their source, target, type, and weight. It is loaded at startup and the in-memory graph is reconstructed. New edges discovered during runtime are appended. The graph grows over time as the system learns more relationships.

---

## Phase 3 — Co-Retrieval Tracking

### The core idea

Every time retrieval returns a set of memories, the system records which memories were returned together. Over time this builds a statistical picture of which memories tend to be useful in the same context. This is collaborative filtering applied to memories.

The measure used is Normalized Pointwise Mutual Information (NPMI). Without the math: NPMI measures how much more often two things appear together than you'd expect by chance. If Memory A and Memory B are retrieved together far more often than chance predicts, their NPMI is high. NPMI ranges from -1 to 1; you only care about positive values (memories that co-occur more than expected).

### What this learns automatically

The first time "jwt auth" is queried, Memory A and Memory B both happen to appear — A from vector search, B from graph expansion. The co-retrieval tracker records this. After ten similar queries, both memories have been returned together repeatedly. Their NPMI rises above the threshold. Now when a query brings in Memory A via vector search, Memory B gets a co-retrieval boost even if the graph connection between them is weak. The system has learned these two memories tend to be useful together — with zero manual configuration.

### What gets tracked

Two things: how many times each individual memory has been retrieved (the baseline), and how many times each pair of memories has been retrieved together (the co-occurrence count). These two numbers are enough to compute NPMI.

### When co-retrieval creates graph edges

When the NPMI score between two memories rises above a threshold (around 0.3), a co-retrieval edge is created in the graph. This means the graph gets richer over time — frequently co-retrieved memories become explicitly connected, making future traversal more reliable and the relationship more durable.

### The caveat

Co-retrieval can encode bad patterns. If a query consistently returns two memories together but one is wrong or outdated, the system reinforces that pairing. The mitigation: co-retrieval is 10% of the final score, not primary. Bad patterns are corrected by archiving the wrong memory or explicitly rating it wrong — both of which zero out its co-retrieval counts.

### Persistence and decay

The tracker is saved as a JSON file containing all co-occurrence counts and individual retrieval counts. It is loaded at startup. It grows over time. A decay mechanism halves all counts on a weekly or monthly schedule so old patterns fade without being lost entirely — recent usage matters more than patterns from six months ago.

---

## Phase 4 — The Unified Retriever

### Architecture

The unified retriever replaces the current single-step retrieval with a six-step pipeline:

**Step 1 — Vector search.** Unchanged from current behavior. The query is embedded, ChromaDB returns the top N candidates by cosine distance, over-fetched to leave room for reranking. Each candidate gets a vector score.

**Step 2 — Tag expansion.** The query analyzer extracts tags from the query text. All memories with those tags are added to the candidate pool if not already present from vector search. Each candidate in the pool gets a tag overlap score — the Jaccard similarity between the query's matched tags and the memory's tags.

**Step 3 — Graph expansion.** The top five candidates from vector search become graph traversal starting points. BFS traversal to depth one or two adds their graph neighbors to the candidate pool. Each graph-expanded candidate gets a graph score equal to the edge weight multiplied by the depth decay factor (0.5 per hop).

**Step 4 — Co-retrieval boost.** For each candidate in the pool, the system checks its NPMI score against each of the top-five vector results. The maximum NPMI among those pairs becomes the candidate's co-retrieval score.

**Step 5 — Final scoring.** Each candidate's final score is a weighted sum:

| Signal | Weight | Why |
|---|---|---|
| Vector similarity | 45% | Most semantically aware — knows meaning not just keywords |
| Tag overlap | 15% | Flat but precise — shared tags are explicit author signals |
| Graph proximity | 15% | Graduated — direct neighbors matter more than two-hop ones |
| Co-retrieval (NPMI) | 10% | Probabilistic, can encode noise — supporting signal only |
| Recency | 10% | Newer memories have a slight edge — knowledge evolves |
| Confidence | 5% | Weakest because currently set inconsistently |

**Step 6 — Filter and return.** Candidates below the minimum score threshold are removed. The remainder are sorted by final score and the top N are returned. After returning, the retriever records the retrieved memory IDs with the co-retrieval tracker.

### Explainability

Every returned memory carries a structured record of its retrieval: which signals contributed to its final score, what each individual signal score was, and whether it came from vector search, tag expansion, or graph expansion. You can look at any result and answer "why is this here?"

---

## Phase 5 — Automatic Graph Building (Enhancement Layer)

This phase makes the graph richer automatically during normal operation, beyond what the startup batch pass builds.

### Temporal clustering

When memories are created in bursts — multiple memories within a 30-minute window — the graph builder connects them with weak temporal edges. These are created as part of the `store.add()` operation so they happen immediately when a new memory is stored.

### Supersedes detection

The builder groups memories by category and leading tags. Within each group it looks for newer memories covering the same ground as older ones. When a newer memory has the same category, overlapping tags, and equal or higher confidence, a directional SUPERSEDES edge is created from the newer to the older. During retrieval, if both are in the candidate pool, the retriever can prefer the source of the SUPERSEDES edge. Future consolidation passes can use SUPERSEDES edges to identify merge candidates.

### Incremental vs batch

The batch pass runs at startup over all memories — O(n²) in the worst case for tag similarity, but bounded by the tag index so only memories sharing at least one tag are compared. The incremental pass runs for each new memory when it is stored — it checks tag similarity against existing memories, looks for temporal neighbors, runs supersedes detection, and creates edges. This should complete in milliseconds.

---

## Phase 6 — Index and Graph Maintenance

Indices degrade. Memories get archived but their nodes linger in the graph. Tags get changed but old edges persist. This phase defines the maintenance operations and where they hook in.

### Tag index maintenance

When a memory is archived, its tags are immediately removed from the index. When a memory's tags are edited, the old tags are removed and new ones added. Both happen as part of the existing store operations — no separate maintenance job needed.

### Graph maintenance

When a memory is archived, its node and all connected edges are removed from the graph immediately. This is critical: archived memories must not continue to pull other memories into the candidate pool via graph traversal. The graph file is pruned periodically to remove orphaned entries.

### Co-retrieval maintenance

Retrieval counts for archived memories are zeroed out so they don't continue to inflate NPMI scores with active memories. The weekly/monthly count decay handles gradual staleness. If a reset is needed (for example, after bulk archiving), the tracker can be rebuilt from scratch by replaying recent retrieval logs.

### Startup sequence

On startup, in order:

1. Load all memory files and build the in-memory cache
2. Build the tag index from all memories
3. Load the graph from disk, then run the batch graph builder to discover relationships not yet in the graph file
4. Load the co-retrieval tracker from disk
5. Initialize the unified retriever with all of the above
6. System is ready

The batch graph builder in step 3 acts as a repair pass — if the graph file is missing or corrupt, it rebuilds from the memory files themselves. This ensures startup always produces a consistent state.

---

## Files to Create

### New files

| File | Purpose |
|---|---|
| `src/lace/retrieval/tag_index.py` | In-memory inverted index, tag extraction from queries |
| `src/lace/retrieval/graph.py` | MemoryGraph, EdgeType, BFS traversal, JSON persistence |
| `src/lace/retrieval/graph_builder.py` | Automatic edge discovery (tag similarity, temporal, supersedes) |
| `src/lace/retrieval/co_occurrence.py` | NPMI computation, co-retrieval tracking, persistence |
| `src/lace/retrieval/unified.py` | UnifiedRetriever — the 6-step pipeline |

### Modified files

| File | Change |
|---|---|
| `src/lace/memory/store.py` | Wire tag index updates into `add()`, `save()`, `forget()`. Replace `_vector_search()` with unified retriever. |
| `src/lace/main.py` | Initialize and pass the new components through to the store on startup |
| `src/lace/mcp/tools.py` | Ensure `get_relevant_context` calls `record_retrieval()` after every search |

---

## Implementation Order and Rationale

**Phase 1 first (Tag Index).** Lowest complexity. Can be built and tested independently of everything else. Provides immediate improvement to retrieval for queries containing tag words. Also required by Phase 2 — the graph builder uses the tag index for tag similarity edges.

**Phase 2 second (Graph).** Depends on the tag index for automatic edge creation. Makes `related_ids` functional for the first time. Significant improvement because it enables non-obvious connections between memories.

**Phase 3 third (Co-retrieval).** Needs a working retrieval pipeline to track. Add to the retrieval loop after Phases 1 and 2 are working. Starts producing signal immediately but becomes genuinely useful after a few hundred queries.

**Phase 4 fourth (Unified Retriever).** The integration phase. Connects Phases 1–3 into a single pipeline. Build after the individual components are tested because it orchestrates all of them.

**Phase 5 fifth (Automatic Graph Building).** Enhancement on top of a working system. Not strictly required for the unified retriever to work — the batch startup pass gives a good baseline graph. Add after Phase 4 is stable.

**Phase 6 sixth (Maintenance).** Build maintenance hooks incrementally alongside the other phases rather than all at the end. Add the archive cleanup to Phase 2's archive operation. Add the co-retrieval decay to Phase 3's persistence logic. Deferring all maintenance to the end is consistently regretted.

---

## Risk Assessment

**Performance.** Graph traversal and tag expansion add latency per query. Both the tag index and graph are in-memory so individual lookups are fast. The risk is traversal depth on a dense graph. Mitigations: cap traversal depth at 2 hops, cap graph-expanded candidates at 20, traverse only from the top 5 vector results.

**Noise amplification.** Bad graph edges (spurious tag similarity connections, incorrect temporal clusters) pull irrelevant memories into the candidate pool. Mitigation: graph-expanded candidates must still clear the final score threshold — a single weak edge doesn't guarantee an irrelevant memory makes the cut. Conservative edge weights for automatically created edges (tag similarity requires Jaccard ≥ 0.2, temporal edges capped at weight 0.5).

**Cold start.** The co-retrieval tracker starts empty. The graph starts sparse. For a new installation, the system behaves like the current system until data accumulates. This is acceptable — the fallback is current behavior, not a failure. The tag index and explicit graph edges provide immediate value before usage patterns accumulate.

**Staleness.** The tag index and graph are built at startup and updated incrementally. If an incremental update fails (crash, error), they might be slightly out of sync with the memory files. Mitigation: the startup batch rebuild corrects any drift. It runs on every startup so the worst case is one session of partial inconsistency.

---

## Success Criteria

The implementation is complete and working when:

- A query for "jwt token expiry" retrieves Memory B directly and also retrieves Memory A via graph expansion from B, with Memory A's retrieval source logged as `graph_from_B`
- After 20 queries co-retrieving Memories A and B, their NPMI score exceeds 0.3 and a co-retrieval edge exists in the graph
- Setting `related_ids` on a memory causes the linked memory to appear in graph traversal results for queries that hit the source memory
- Archiving a memory removes it from the tag index, removes its node from the graph, and zeroes its co-retrieval counts
- The explain output for any returned memory shows which signals contributed to its score and from which source it was retrieved
- Total retrieval latency (all six steps) stays under 200ms for a store with up to 10,000 memories

---

## What This Does Not Change (Yet)

The Option A auto-store pipeline discussed separately (confidence at 0.4, confidence rises through `record_access()`, prune stale auto-extracted memories) is independent of this plan. That changes how memories enter the system. This plan changes how they are retrieved. Both should be implemented, in either order, without conflict.

The Obsidian file format, the markdown vault structure, and the ChromaDB storage layout are unchanged. This plan adds new files and new in-memory structures alongside what exists — it does not replace or migrate the storage layer.
