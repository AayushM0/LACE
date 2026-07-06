---
trigger: always_on
---

# LACE Active Memory Protocol Rules

LACE (Local AI Context Engine) persistent memory is connected to this agent via Model Context Protocol (MCP).

This protocol defines how the agent must interact with LACE each turn, and how each project can tune LACE's
behavior via a local config file. Server-side gating (worthiness verdicts, hash suppression, dedup) is LACE's
primary defense against noise — but the agent is the first checkpoint, and lazy or careless calls here still cost
LLM spend and pollute the queue before server-side filters ever run.

---

## 1. Session Initialization (Turn 1 only)

Call `initialize_lace_session(working_directory)` automatically at the start of a new conversation session.

- `working_directory`: absolute path of the current workspace root.
- On init, LACE resolves project scope by walking up for a Git root or `.lace/project.yaml`. If neither exists,
  LACE falls back to `global` scope — this is usually a signal the project hasn't been configured yet (see
  Section 4). If you detect this fallback, mention it to the user once, don't repeat the warning every turn.

---

## 2. Retrieve Context (every turn, before responding)

Call `get_relevant_context(query)` passing the user's exact message.

- Inject the returned markdown context into your system context before generating a response.
- Do not skip this even for turns you expect to skip logging on (Section 3) — retrieval and logging are
  independent decisions.

---

## 3. Log Decisions & Discoveries (every turn, after responding)

Call `process_interaction(query, response, context_hint=None)` for nearly every turn. **Do not make a unilateral
decision to skip logging based on judgment about whether the turn "seems repetitive" or "seems noisy"** — that
judgment is inconsistent across turns and models, and a wrongly-skipped turn is permanent memory loss with no
server-side backstop. LACE's deterministic pipeline (hash suppression at insert, worthiness verdict during
extraction, tiered dedup) is the actual filter and costs almost nothing to run — a queue insert is `<5ms`, and
the worthiness check happens inside the extraction call LACE already makes, with no extra round-trip. Skipping
the call yourself saves LACE no meaningful latency, while risking silently dropping something useful.

**Only skip the call outright for:**
- Pure greetings or acknowledgments with no content ("hi", "thanks", "ok got it").

**For everything else, call it — but attach `context_hint` when you notice something LACE's automated pipeline
can't see from the text alone:**
- `context_hint="repeated_action"` — you recognize this as one of many similar/looped actions (e.g. part of a
  stress test, retry loop, batch run). This is a *signal*, not a skip: the extractor weighs it when producing its
  worthiness verdict, and hash suppression still runs independently regardless of the hint.
- `context_hint="general_knowledge"` — the turn was answered from general knowledge with no interaction with the
  user's actual project, codebase, or decisions.
- Leave `context_hint=None` for ordinary turns — decisions, bug fixes, preferences, project-specific patterns.

This shifts judgment calls from "agent unilaterally decides and the information is gone if wrong" to "agent
informs, deterministic pipeline still decides" — the hint improves the server's accuracy without making the
agent a single point of failure.

---

## 4. Per-Project Configuration

Each project can override global LACE defaults via `.lace/project.yaml` in the project root. This lets noisy
projects (load-testing tools, data pipelines with repetitive runs) tune more aggressively without affecting
quieter projects (writing, design, small scripts).

```yaml
# .lace/project.yaml
project_name: rescuemesh
scope: project:rescuemesh

extraction:
  require_worthiness_verdict: true       # keep true unless you have a strong reason not to
  noise_profile: high                    # low | medium | high — see below

dedup:
  skip_threshold: 0.95                   # >= this cosine similarity: discard as duplicate
  merge_threshold: 0.85                  # >= this: merge into existing memory
  hash_cooldown_seconds: 300             # window for collapsing repeat interactions at insert time
```

**`noise_profile` presets** (adjust the underlying thresholds automatically if set, individual keys above still
override):

| Profile | Typical project type | Effect |
| :--- | :--- | :--- |
| `low` | Writing, design, docs-heavy work | Wider cooldown window, lower merge threshold (0.80) — fewer repeats expected, so bias toward capturing nuance. |
| `medium` (default) | General dev work | Defaults as shown above. |
| `high` | Load/stress testing, data pipelines, CI-heavy repos | Shorter cooldown isn't needed (repeats are frequent and fast), but merge threshold rises to ~0.90 and hash-suppression cooldown extends to ~900s, since templated near-duplicates are the norm, not the exception. |

If `.lace/project.yaml` is missing, the project runs on global defaults and inherits whatever noise your global
config tolerates — for any project you expect to generate repetitive or scripted interactions (like stress
testing), create this file up front rather than tuning after the fact.

---

## 5. What Changed From the Previous Protocol, and Why

- Section 3's skip list previously only excluded greetings/clarifications — this is why repetitive test runs
  (e.g. "stress test 1" through "stress test 50") were reaching the extraction pipeline and being stored as
  near-duplicate files. The agent is now expected to recognize obvious loops itself, ahead of server-side
  filtering.
- Section 4 is new. Previously all projects shared one global dedup/extraction configuration, which meant tuning
  for a noisy project (loosening thresholds to reduce junk) would have made a quiet project's dedup too
  aggressive, and vice versa. Per-project config removes that tradeoff.