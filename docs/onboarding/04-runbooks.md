# Developer Task Runbooks

This document contains step-by-step guides for common development tasks within the LACE codebase.

---

## Runbook 1: Add a New MCP Tool

LACE exposes tools to LLM clients (Cursor, Claude Desktop, etc.) via MCP. To add a new tool:

### Step 1: Implement the tool logic
1. Open [`src/lace/mcp/tools.py`](file:///home/aayush/Projects/lace/LACE/src/lace/mcp/tools.py).
2. Implement your async tool function. Make sure it accepts `**kwargs` to ignore extra parameters cleanly:
   ```python
   async def get_memory_tags(scope: str = "auto", **kwargs) -> list[str]:
       """List all unique tags in the memory vault."""
       store, resolved_scope = _get_store(scope)
       memories = store.list(scope=resolved_scope, limit=1000)
       tags = set()
       for m in memories:
           tags.update(m.tags)
       return list(tags)
   ```

### Step 2: Register the tool with the MCP Server
1. Open [`src/lace/mcp/server.py`](file:///home/aayush/Projects/lace/LACE/src/lace/mcp/server.py).
2. Look for the `@mcp.tool()` registrations or similar declarations.
3. Import your tool function from `lace.mcp.tools` and declare it.
4. Verify the MCP server starts correctly:
   ```bash
   .venv/bin/lace mcp start
   ```

---

## Runbook 2: Write and Run Tests

LACE relies heavily on tests to verify deduplication, search ranking, and sync logic.

### Step 1: Write a test
1. Create or open a test file under `tests/` corresponding to your changes (e.g. `tests/test_retrieval/test_custom.py`).
2. Implement your test function prefixed with `test_`. Use standard `pytest` fixtures.

### Step 2: Run the test suite
- **Run all tests**:
  ```bash
  .venv/bin/pytest
  ```
- **Run specific test directory**:
  ```bash
  .venv/bin/pytest tests/test_memory/
  ```
- **Run a single test file**:
  ```bash
  .venv/bin/pytest tests/test_retrieval/test_ranking.py
  ```
- **Run a single test method**:
  ```bash
  .venv/bin/pytest tests/test_memory/test_dedup.py -k "test_check_duplicate"
  ```

---

## Runbook 3: Change Retrieval Weights

Retrieval weights dictate how LACE combines the 5 signals (vector, tag, graph, co-retrieval, recency, confidence) into the final relevance score.

### Method 1: Using the CLI (Permanent Change)
You can adjust the Pydantic config values directly, which updates `~/.lace/config/lace.yaml`:
```bash
# Set vector similarity weight to 50%
.venv/bin/lace config set retrieval.weights.semantic_similarity 0.50

# Reduce recency weight to 10%
.venv/bin/lace config set retrieval.weights.recency 0.10
```
> [!IMPORTANT]
> The weights must sum to exactly `1.0`. Pydantic will validate this when you next load or save configuration.

### Method 2: Programmatically (Runtime/Test changes)
You can update weights at runtime in tests or experimental code:
```python
from lace.retrieval.unified import UnifiedWeights

new_weights = UnifiedWeights(
    vector=0.50,
    tag=0.10,
    graph=0.15,
    co_retrieval=0.10,
    recency=0.10,
    confidence=0.05
)
retriever.set_weights(new_weights)
```

---

## Runbook 4: Add a New Config Field

To add a new configuration parameter to LACE (e.g. configuring a new LLM provider model):

### Step 1: Update the config model
1. Open [`src/lace/core/config.py`](file:///home/aayush/Projects/lace/LACE/src/lace/core/config.py).
2. Locate the appropriate model (e.g. `ExtractionConfig` or `MemoryConfig`).
3. Define your field with type annotation, default value, and description:
   ```python
   class ExtractionConfig(BaseModel):
       # Existing fields...
       debug_mode: bool = Field(
           default=False,
           description="Enable debug logs during extraction"
       )
   ```

### Step 2: Use the field in code
1. Load config using `load_config()`:
   ```python
   from lace.core.config import load_config, get_lace_home
   
   config = load_config(get_lace_home())
   if config.extraction.debug_mode:
       print("Debug mode active")
   ```
2. The config set parser dot-notation works automatically. You can now configure this from CLI:
   ```bash
   .venv/bin/lace config set extraction.debug_mode "true"
   ```
