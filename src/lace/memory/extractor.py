"""LLM-assisted memory extraction pipeline.

After a conversation turn, this module:
  1. Analyzes the query + response
  2. Extracts knowledge worth storing
  3. Runs dedup against existing memories
  4. Stores novel insights automatically

This is what makes LACE learn automatically.

Extraction is:
  - Async and non-blocking
  - Conservative (most turns produce NO extraction)
  - Configurable (can require confirmation)
  - Capped (max 3 extractions per turn)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lace.memory.store import MemoryStore


# ── Extraction result ─────────────────────────────────────────────────────────

@dataclass
class ExtractionCandidate:
    """A piece of knowledge extracted from a conversation."""
    content:    str
    category:   str          # pattern, decision, debug, reference, preference
    tags:       list[str]
    confidence: float
    reasoning:  str          # why this was extracted


@dataclass
class ExtractionResult:
    """Result of an extraction attempt."""
    candidates:   list[ExtractionCandidate]
    stored:       list[str]   # memory IDs of stored memories
    merged:       list[str]   # memory IDs of merged memories
    skipped:      int         # count of skipped duplicates
    error:        str | None  # error message if extraction failed


# ── Extraction prompt ─────────────────────────────────────────────────────────

EXTRACTION_SYSTEM_PROMPT = """You are a knowledge extraction assistant.
Your job is to identify insights worth storing as persistent memories.

Extract knowledge ONLY if it meets ALL of these criteria:
1. SPECIFIC and ACTIONABLE — not vague or generic
2. REUSABLE — useful in future conversations
3. NON-OBVIOUS — not common knowledge
4. TECHNICAL — relates to code, architecture, debugging, or decisions

Before extracting, ask these three questions (SPECIFICITY TEST):
1. Is this specific to THIS project or could it apply to any codebase?
   If any codebase → DISCARD
2. Would a developer need to re-explain this to an AI in a future session?
   If no → DISCARD  
3. Does this represent an actual decision made or pattern established?
   If it's just information discussed → DISCARD

NEVER extract the following — discard immediately:
- Any information about LACE itself, its tools, or how it works
- File paths, directory paths, or local machine locations
- Generic documentation that applies to any project universally
- Information that is freely available in official docs
- Anything that references "get_relevant_context" or "process_interaction"
- Duplicate concepts already covered by a previous extraction in this turn
- Conversational filler
- Questions without clear answers
- Temporary context
- Anything about the AI stopping, pausing, or changing its behavior
- Anything describing what the AI assistant did or will do
- Memories that reference the AI in first or third person
- Anything not directly related to the active project's technical decisions

Output a JSON array of extractions. Each extraction has:
{
  "content": "The specific knowledge to store",
  "category": "pattern|decision|debug|reference|preference",
  "tags": ["tag1", "tag2"],
  "confidence": 0.0-1.0,
  "reasoning": "Why this is worth storing"
}

If nothing is worth storing, return an empty array: []

Maximum 3 extractions per conversation turn.
Be conservative — most turns should produce 0 extractions."""


def _build_extraction_prompt(query: str, response: str, context: str = "") -> str:
    """Build the user prompt for extraction."""
    prompt = "Conversation turn to analyze:\n\n"
    if context:
        prompt += f"CONVERSATION HISTORY:\n{context}\n\n"
    prompt += (
        f"USER QUERY:\n{query}\n\n"
        f"ASSISTANT RESPONSE:\n{response}\n\n"
        f"Extract any knowledge worth storing as persistent memories.\n"
        f"Return a JSON array (can be empty [])."
    )
    return prompt


# ── Extraction engine ─────────────────────────────────────────────────────────

def extract_from_conversation(
    query: str,
    response: str,
    store: "MemoryStore" | None = None,
    scope: str = "global",
    max_extractions: int = 3,
    require_confirmation: bool = False,
    provider=None,
    context: str = "",
    source: str = "conversation",
    confidence_cap: float | None = None,
) -> ExtractionResult:
    # Never store extracted memories under a session scope
    # Sessions are ephemeral — extracted knowledge should persist
    if scope.startswith("session:"):
        scope = "global"

    # All imports here — avoids circular import issues
    from lace.memory.dedup import check_duplicate, merge_memories, DedupAction
    from lace.memory.models import make_memory

    # Step 1 — Get LLM provider
    if provider is None:
        try:
            from lace.core.config import load_config, get_lace_home
            from lace.utils.providers import get_provider
            config = load_config(get_lace_home())
            provider = get_provider(config)
        except Exception as e:
            return ExtractionResult(
                candidates=[], stored=[], merged=[],
                skipped=0, error=f"Could not load provider: {e}",
            )

    # Step 2 — Run LLM extraction
    try:
        raw = provider.complete(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_message=_build_extraction_prompt(query, response, context),
        )
    except Exception as e:
        return ExtractionResult(
            candidates=[], stored=[], merged=[],
            skipped=0, error=f"LLM extraction failed: {e}",
        )

    # Step 3 — Parse JSON response
    candidates = _parse_extraction_response(raw, max_extractions)

    if not candidates:
        return ExtractionResult(
            candidates=[], stored=[], merged=[],
            skipped=0, error=None,
        )

    # Step 4 — If confirmation required or store is None, return without storing
    if require_confirmation or store is None:
        return ExtractionResult(
            candidates=candidates,
            stored=[],
            merged=[],
            skipped=0,
            error=None,
        )

    # Step 5 — Load existing memories for dedup check
    # Load only active memories for dedup — archived memories are dead
    existing = [
        m for m in store.list(include_archived=False, limit=500)
        if m.lifecycle.value != "archived"
    ]

    # Step 6 — Dedup and store
    stored: list[str]  = []
    merged: list[str]  = []
    skipped: int       = 0

    for candidate in candidates:
        try:
            from lace.memory.models import make_memory
            memory = make_memory(
                content=candidate.content,
                category=candidate.category,
                tags=candidate.tags,
                scope=scope,
                source="conversation",
                confidence=candidate.confidence,
            )

            # Generate embedding
            try:
                from lace.retrieval.embeddings import embed_text
                memory.embedding = embed_text(memory.content)
            except Exception as e:
                memory.embedding = None

            # Dedup check
            dedup = check_duplicate(memory, existing)

            if dedup.action == DedupAction.SKIP:
                skipped += 1
                continue

            elif dedup.action == DedupAction.MERGE and dedup.existing:
                merged_memory = merge_memories(dedup.existing, memory)
                store.save(merged_memory)
                merged.append(merged_memory.id)
                existing = [
                    merged_memory if m.id == merged_memory.id else m
                    for m in existing
                ]

            else:  # STORE
                stored_confidence = (
                    min(candidate.confidence, confidence_cap)
                    if confidence_cap is not None
                    else candidate.confidence
                )
                saved = store.add(
                    content=candidate.content,
                    category=candidate.category,
                    tags=candidate.tags,
                    scope=scope,
                    source=source,
                    confidence=stored_confidence,
                )
                stored.append(saved.id)
                existing.append(saved)

        except Exception as e:
            import sys
            print(f"[LACE extractor] candidate failed: {e}", file=sys.stderr)
            continue

    return ExtractionResult(
        candidates=candidates,
        stored=stored,
        merged=merged,
        skipped=skipped,
        error=None,
    )


def _parse_extraction_response(
    raw: str,
    max_extractions: int = 3,
) -> list[ExtractionCandidate]:
    """Parse LLM JSON response into ExtractionCandidate objects.

    Handles messy LLM output — extracts JSON even if surrounded by text.
    """
    if not raw or not raw.strip():
        return []

    # Find JSON array in response (LLMs sometimes wrap it in prose)
    start = raw.find("[")
    end   = raw.rfind("]")

    if start == -1 or end == -1 or start >= end:
        return []

    json_str = raw[start:end + 1]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return []

    if not isinstance(data, list):
        return []

    candidates: list[ExtractionCandidate] = []

    valid_categories = {
        "pattern", "decision", "debug", "reference", "preference"
    }

    for item in data[:max_extractions]:
        if not isinstance(item, dict):
            continue

        content = str(item.get("content", "")).strip()
        if not content or len(content) < 20:
            continue

        category = str(item.get("category", "pattern")).lower()
        if category not in valid_categories:
            category = "pattern"

        tags = item.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        tags = [str(t).lower() for t in tags if t][:5]

        confidence = float(item.get("confidence", 0.7))
        confidence = max(0.1, min(1.0, confidence))

        reasoning = str(item.get("reasoning", "")).strip()

        candidates.append(ExtractionCandidate(
            content=content,
            category=category,
            tags=tags,
            confidence=confidence,
            reasoning=reasoning,
        ))

    return candidates


# ── Quality filters ───────────────────────────────────────────────────────────

def should_attempt_extraction(query: str, response: str) -> bool:
    """Quick pre-filter — should we even attempt extraction?

    Returns False for turns that clearly won't produce useful memories:
    - Very short responses
    - Pure greetings or meta-conversation
    - Error responses
    """
    # Too short to contain useful knowledge
    if len(response) < 100:
        return False

    # Pure greetings
    greetings = {"hello", "hi", "hey", "thanks", "thank you", "bye", "goodbye"}
    query_words = set(query.lower().split())
    if query_words.issubset(greetings):
        return False

    # Error responses from LLM
    error_indicators = ["[error:", "connection refused", "model not found"]
    response_lower = response.lower()
    if any(e in response_lower for e in error_indicators):
        return False

    return True