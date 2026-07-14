---
trigger: always_on
---

# LACE Active Memory Protocol Rules

LACE (Local AI Context Engine) persistent memory is connected to this agent via Model Context Protocol (MCP).

This protocol defines how the agent must interact with LACE each turn, and how each project can tune LACE's behavior via a local config file. Server-side gating (worthiness verdicts, hash suppression, dedup) is LACE's primary defense against noise — but the agent is the first checkpoint, and lazy or careless calls here still cost LLM spend and pollute the queue before server-side filters ever run.

---

## 1. Session Initialization & Warm-up (Turn 1 only)

Call `initialize_lace_session(working_directory)` automatically at the start of a new conversation session.

- **`working_directory`**: Absolute path of the current workspace root.
- **Scope Fallback**: On init, LACE resolves project scope by walking up for a Git root or `.lace/project.yaml`. If neither exists, LACE falls back to `global` scope. If you detect this fallback, mention it to the user once; do not repeat the warning every turn.
- **Context Warm-up**: Immediately after initialization, read the `memory://project-context` resource if available to align on local project rules, preferences, and ADR outlines.

---

## 2. Retrieve Context & Leverage Resources (Every turn, before responding)

1. **Context Retrieval**: Call `get_relevant_context(query)` passing the user's exact message.
   - Inject the returned markdown context into your system context before generating a response.
   - Do not skip this even for turns you expect to skip logging on (Section 4) — retrieval and logging are independent decisions.
2. **Specialized MCP Resources**: In addition to standard semantic retrieval, if the user asks questions regarding custom developer patterns, debugging history, or architecture details, utilize the corresponding LACE MCP resources:
   - `memory://patterns`: Coding conventions and structures.
   - `memory://decisions`: Architectural ADRs and choices.
   - `memory://debug-log`: Debugging runbooks and history.

---

## 3. Sensitive Data Guard (Before Ingestion)

Prior to calling `process_interaction`, inspect the user's query and the generated response for secrets, credentials (tokens, API keys, passwords, connection strings, auth headers, or certificates).
- **Sanitize**: Replace secrets with a safe placeholder (e.g. `<REDACTED_API_KEY>`) in the parameters passed to the queue.
- **Skip**: If the turn is centered around managing sensitive variables or secrets and cannot be cleanly sanitized, skip the `process_interaction` call entirely for that turn to prevent credential leaks to the local Markdown database.

---

## 4. Log Decisions & Discoveries (Every turn, after responding)

Call `process_interaction(query, response, context_hint=None)` for nearly every turn. **Do not make a unilateral decision to skip logging based on judgment about whether the turn "seems repetitive" or "seems noisy"** — that judgment is inconsistent across turns and models, and a wrongly-skipped turn is permanent memory loss with no server-side backstop.

**Only skip the call outright for:**
- Pure greetings or acknowledgments with no content ("hi", "thanks", "ok got it").

**For everything else, call it — but attach `context_hint` to categorize the interaction:**
- `context_hint="debugging_insight"` — The turn resolved a compilation error, package installation block, or runtime bug.
- `context_hint="architectural_decision"` — The turn resolved a design pattern, system structure, library choice, or ADR mapping.
- `context_hint="user_preference"` — The turn captured coding style preferences, formatting rules, or custom user guidelines.
- `context_hint="repeated_action"` — Part of a stress test, retry loop, or batch execution.
- `context_hint="general_knowledge"` — The turn was answered from standard pre-trained knowledge with no interaction with the user's codebase, config, or logic.
- Leave `context_hint=None` for ordinary development interactions.

---

## 5. Proactive Project Config & Health Recommendations

1. **Config Tuning**: If you notice a high frequency of repetitive tasks, load tests, or automated pipeline scripts, recommend the user create or tune their `.lace/project.yaml` to raise dedup and cooldown thresholds (using a `high` noise profile).
2. **Interactive Optimization**: If the user corrects your code, notes a memory gap, or rejects a pattern, suggest they run `lace memory review` or rate the memory via `/learn` / `/grill-me` to ensure LACE's derivative indexes remain accurate.