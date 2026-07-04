---
trigger: always_on
---

LACE Active Memory Protocol Rules

LACE (Local AI Context Engine) persistent memory is connected to this agent via Model Context Protocol (MCP).

This protocol defines how the agent must interact with LACE each turn, and how each project can tune LACE's
behavior via a local config file. Server-side gating (worthiness verdicts, hash suppression, dedup) is LACE's
primary defense against noise — but the agent is the first checkpoint, and lazy or careless calls here still cost
LLM spend and pollute the queue before server-side filters ever run.


1. Session Initialization (Turn 1 only)

Call initialize_lace_session(working_directory) automatically at the start of a new conversation session.


working_directory: absolute path of the current workspace root.
On init, LACE resolves project scope by walking up for a Git root or .lace/project.yaml. If neither exists,
LACE falls back to global scope — this is usually a signal the project hasn't been configured yet (see
Section 4). If you detect this fallback, mention it to the user once, don't repeat the warning every turn.



2. Retrieve Context (every turn, before responding)

Call get_relevant_context(query) passing the user's exact message.


Inject the returned markdown context into your system context before generating a response.
Do not skip this even for turns you expect to skip logging on (Section 3) — retrieval and logging are
independent decisions.



3. Log Decisions & Discoveries (every turn, after responding)

Call process_interaction(query, response) only when the turn contains something a future session would
benefit from knowing. LACE's server-side pipeline (worthiness verdict → hash suppression → tiered dedup) is the
main filter, but every call still costs a queue write and eventual LLM extraction pass — don't rely on the
backend to clean up what you can recognize as noise yourself.

Skip logging when the turn is:


A basic greeting, acknowledgment, or simple clarifying question.
A generic explanation with no project-specific decision, pattern, or bug fix in it.
Part of an obvious repetitive loop — the same or near-identical action run repeatedly (stress tests, retry
loops, batch scripts, "run it again," incrementing test numbers). Log the first occurrence if it revealed
something new; skip the rest of the loop entirely. Recognizing "this is iteration N of the same thing" is an
agent-side judgment call LACE's hash suppression also catches, but catching it before the call saves the
round-trip entirely.
A question you answered purely from general knowledge, with no interaction with the user's actual codebase,
data, or decisions.


Log when the turn contains:


A decision made (architecture choice, tradeoff resolved, approach picked over an alternative).
A bug found and fixed, especially with a non-obvious root cause.
A durable preference the user stated (style, tooling, workflow).
A pattern or convention specific to this project that future sessions would need to know.


When in doubt, log it — false positives are cheap for LACE to filter server-side; false negatives (skipping
something genuinely useful) are permanent memory loss.


4. Per-Project Configuration

Each project can override global LACE defaults via .lace/project.yaml in the project root. This lets noisy
projects (load-testing tools, data pipelines with repetitive runs) tune more aggressively without affecting
quieter projects (writing, design, small scripts).

yaml# .lace/project.yaml
project_name: rescuemesh
scope: project:rescuemesh

extraction:
  require_worthiness_verdict: true       # keep true unless you have a strong reason not to
  noise_profile: high                    # low | medium | high — see below

dedup:
  skip_threshold: 0.95                   # >= this cosine similarity: discard as duplicate
  merge_threshold: 0.85                  # >= this: merge into existing memory
  hash_cooldown_seconds: 300             # window for collapsing repeat interactions at insert time

noise_profile presets (adjust the underlying thresholds automatically if set, individual keys above still
override):

ProfileTypical project typeEffectlowWriting, design, docs-heavy workWider cooldown window, lower merge threshold (0.80) — fewer repeats expected, so bias toward capturing nuance.medium (default)General dev workDefaults as shown above.highLoad/stress testing, data pipelines, CI-heavy reposShorter cooldown isn't needed (repeats are frequent and fast), but merge threshold rises to ~0.90 and hash-suppression cooldown extends to ~900s, since templated near-duplicates are the norm, not the exception.

If .lace/project.yaml is missing, the project runs on global defaults and inherits whatever noise your global
config tolerates — for any project you expect to generate repetitive or scripted interactions (like stress
testing), create this file up front rather than tuning after the fact.