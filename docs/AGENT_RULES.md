# LACE Agent Rules and Conventions

This document consolidates guidelines, triage labels, and issue tracking conventions used by engineering agents within this repository.

---

## 1. Domain Documentation & Glossary Usage

AI agents must consult the domain and architecture documents to maintain alignment with project vocabulary and concepts.

### 1.1 Before Exploring a Topic, check:
* **`CONTEXT.md`** at the repo root, or
* **`CONTEXT-MAP.md`** at the repo root (which references topic-specific `CONTEXT.md` files).
* **`docs/adr/`** — Architectural Decision Records mapping to the feature areas you are working in.

> **Note on Absence:** If any of these files do not exist, proceed silently. Do not suggest creating them upfront. They are created lazily as terms and decisions are resolved.

### 1.2 Glossary Consistency
When naming domain concepts in issues, code changes, refactors, or tests, use terms exactly as defined in `CONTEXT.md`. Do not invent synonyms.

### 1.3 ADR Conflicts
If a proposed change contradicts an existing ADR, flag it explicitly in the plan or output:
> *Contradicts ADR-XXXX (Event Sourced Orders) — but worth reopening because...*

---

## 2. Local Markdown Issue Tracker

Issues and PRDs live as markdown files under the `.scratch/` directory.

> **Note on Absence:** The `.scratch/` directory and its issue files are created dynamically or lazily when feature scopes are triaged or active development begins. If `.scratch/` does not exist, the workspace is in a clean baseline state; proceed silently.

### 2.1 Directory Structure & Conventions
* **Feature Scope**: One directory per feature: `.scratch/<feature-slug>/`
* **Product Requirements**: PRDs are saved at `.scratch/<feature-slug>/PRD.md`
* **Implementation Issues**: Numbered sequentially: `.scratch/<feature-slug>/issues/<NN>-<slug>.md` (starting at `01`)
* **Status Log**: Recorded via a `Status:` line near the top of the issue file.
* **History**: Conversation logs append under a `## Comments` heading at the bottom of the file.

### 2.2 Wayfinder Operations
Wayfinder runs using the following conventions:
* **Map**: `.scratch/<effort>/map.md` (stores notes and locked decisions).
* **Child Ticket**: `.scratch/<effort>/issues/NN-<slug>.md` (uses `Type: research|prototype|grilling|task` and `Status: claimed|resolved` headers).
* **Blocking**: Indicated via `Blocked by: NN, NN`. A ticket is unblocked when all listed dependencies are marked `resolved`.
* **Claiming**: Mark `Status: claimed` before beginning any task work.
* **Resolving**: Append the outcome under an `## Answer` heading, set `Status: resolved`, and link to decisions in `map.md`.

---

## 3. Triage Status Labels

Triage statuses are mapped to canonical role labels. In the local markdown tracker, these are set on the `Status:` line in the issue files.

| Canonical Triage Role | Tracker Status String | Description |
| :--- | :--- | :--- |
| `needs-triage` | `needs-triage` | Proposes new issue for maintainer evaluation |
| `needs-info` | `needs-info` | Blocked waiting on reporter/user clarifications |
| `ready-for-agent` | `ready-for-agent` | Fully specified task ready for autonomous agent execution |
| `ready-for-human` | `ready-for-human` | Requires manual implementation/review by human developer |
| `wontfix` | `wontfix` | Issue will not be actioned or resolved |
