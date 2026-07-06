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

# ═══════════════════════════════════════════════════════════════════════════════
# CHUNK 3 — Worthiness-Gated Extraction Pipeline
# ═══════════════════════════════════════════════════════════════════════════════
# The functions below are a NEW, independent extraction pipeline that adds:
#   - A structured worthiness verdict (worth_remembering + reason)
#   - Pipeline log DB for full audit trail
#   - ExtractionConfig wiring (require_worthiness_verdict, log_all_verdicts)
#
# The existing extract_from_conversation() API above is PRESERVED for
# backward compatibility with main.py, mcp/queue.py, and utils/ask.py.
# ═══════════════════════════════════════════════════════════════════════════════

import logging as _logging
from datetime import datetime as _datetime, timezone as _timezone
from pathlib import Path as _Path
from typing import Optional as _Optional

from lace.core.config import ExtractionConfig as _ExtractionConfig
from lace.memory.pipeline_log import (
    PIPELINE_LOG_DB_PATH,
    initialize_pipeline_log_db,
    log_extraction_verdict as log_extraction_event,  # Keep the same name locally for ease
)

_logger = _logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Worthiness-gated extraction prompt
#
# Design: worth_remembering is the FIRST field so the model commits to
# a verdict before generating memory content (left-to-right token order).
# ---------------------------------------------------------------------------

EXTRACTION_SYSTEM_PROMPT = """You extract durable, useful memory from a single developer-AI interaction.

Respond ONLY with JSON matching this exact schema — no prose, no markdown, no extra keys:
{
  "worth_remembering": <boolean>,
  "reason": "<one sentence: why this IS worth remembering, or why it is NOT>",
  "memories": [
    {
      "category": "<pattern|decision|debug|reference|preference>",
      "summary": "<one clear sentence a developer could read in isolation>",
      "body": "<full context, details, code snippets if relevant>",
      "tags": ["<tag1>", "<tag2>"],
      "confidence": <float: 0.1-1.0 representing certainty/confidence>
    }
  ]
}

RULES FOR worth_remembering:

Set worth_remembering to FALSE if the interaction is ANY of:
  - A test, benchmark, stress test, ping, or synthetic loop
  - Incrementing numbered sequences (test 1, test 2, test N)
  - A simple greeting, acknowledgment, or one-word exchange
  - A restatement of a question with no answer or decision
  - Pure tool/command output with no insight (e.g. "success", "done", "ok")
  - Vague or generic enough that it would apply to any project
  - Something a developer would never need to recall in a future session

Set worth_remembering to TRUE only if the interaction contains:
  - A specific technical decision with reasoning
  - A concrete debugging solution (problem + fix + why it works)
  - A reusable pattern or approach with enough detail to apply again
  - Explicit user preference about how they work
  - A reference to a specific API, library, or system behavior

RULES FOR memories array:
  - If worth_remembering is false: memories MUST be an empty array []
  - If worth_remembering is true: extract 1-3 memories maximum
  - Each memory must be self-contained — readable with no other context
  - summary must be a complete sentence, not a fragment
  - tags must be lowercase, single words or hyphenated-phrases
  - Do NOT assign the same confidence value (e.g. 0.8) to all memories. Vary it dynamically based on how certain you are (0.1 to 1.0).

CATEGORY DEFINITIONS:
  pattern    — a reusable approach or technique
  decision   — an architectural or design choice made in this session
  debug      — a specific bug identified and resolved
  reference  — factual information about an API, library, or system
  preference — how this developer prefers to work or structure things"""


# ---------------------------------------------------------------------------
# Response parsing and validation
# ---------------------------------------------------------------------------

VALID_CATEGORIES = {"pattern", "decision", "debug", "reference", "preference"}


def _validate_memory_dict(mem: dict, index: int) -> _Optional[dict]:
    """
    Validate a single memory dict from the LLM response.
    Returns the validated dict, or None if structurally invalid.
    """
    required_fields = {"category", "summary", "body", "tags"}
    missing = required_fields - set(mem.keys())
    if missing:
        _logger.warning(
            f"[Extractor] Memory[{index}] missing fields: {missing}. Skipping."
        )
        return None

    if mem["category"] not in VALID_CATEGORIES:
        _logger.warning(
            f"[Extractor] Memory[{index}] invalid category: "
            f"'{mem['category']}'. Skipping."
        )
        return None

    if not isinstance(mem["summary"], str) or len(mem["summary"].strip()) < 10:
        _logger.warning(
            f"[Extractor] Memory[{index}] summary too short or invalid. Skipping."
        )
        return None

    if not isinstance(mem["tags"], list):
        _logger.warning(
            f"[Extractor] Memory[{index}] tags must be a list. Skipping."
        )
        return None

    # Normalize: lowercase tags, strip whitespace
    mem["tags"] = [
        str(t).lower().strip()
        for t in mem["tags"]
        if str(t).strip()
    ]

    confidence = mem.get("confidence")
    if confidence is None:
        _logger.warning(f"[Extractor] Memory[{index}] missing confidence. Skipping.")
        return None

    try:
        conf_val = float(confidence)
    except (ValueError, TypeError):
        _logger.warning(f"[Extractor] Memory[{index}] non-numeric confidence. Skipping.")
        return None

    if not (0.0 <= conf_val <= 1.0):
        _logger.warning(f"[Extractor] Memory[{index}] confidence out of bounds. Skipping.")
        return None

    mem["confidence"] = conf_val

    if conf_val == 0.4:
        _logger.warning(
            f"[Extractor] Memory[{index}] has fallback confidence 0.4."
        )
    return mem


def parse_extraction_response(raw: str) -> dict:
    """
    Parse and validate the new-format LLM JSON response.

    Expected schema::

        {
            "worth_remembering": bool,
            "reason": str,
            "memories": [...]
        }

    Returns a dict guaranteed to have all three keys.
    On any failure, returns a safe fallback (worth_remembering=False).
    Never raises.
    """
    _fallback = {
        "worth_remembering": False,
        "reason": "Parse error — defaulting to not worth remembering",
        "memories": [],
    }

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        _logger.error(f"[Extractor] JSON decode failed: {e}. Raw: {raw[:200]}")
        _fallback["reason"] = f"JSON decode error: {e}"
        return _fallback

    if "worth_remembering" not in parsed:
        _logger.error(
            "[Extractor] Response missing 'worth_remembering'. "
            f"Keys found: {list(parsed.keys())}"
        )
        _fallback["reason"] = "Missing worth_remembering field in LLM response"
        return _fallback

    # Coerce 0/1 to bool (some models emit integers)
    if not isinstance(parsed["worth_remembering"], bool):
        parsed["worth_remembering"] = bool(parsed["worth_remembering"])

    # Reason field
    reason = parsed.get("reason", "")
    if not isinstance(reason, str) or not reason.strip():
        reason = "No reason provided by extraction model"
    parsed["reason"] = reason.strip()

    # Memories array
    raw_memories = parsed.get("memories", [])
    if not isinstance(raw_memories, list):
        _logger.warning("[Extractor] 'memories' field is not a list. Using [].")
        parsed["memories"] = []
        return parsed

    # Enforce schema contract: worth=false → memories must be empty
    if not parsed["worth_remembering"] and raw_memories:
        _logger.warning(
            "[Extractor] worth_remembering=false but memories non-empty. "
            f"Discarding {len(raw_memories)} memories — schema contract enforced."
        )
        parsed["memories"] = []
        return parsed

    # Validate each memory dict individually
    validated = []
    for i, mem in enumerate(raw_memories):
        if not isinstance(mem, dict):
            _logger.warning(f"[Extractor] Memory[{i}] is not a dict. Skipping.")
            continue
        result = _validate_memory_dict(mem, i)
        if result is not None:
            validated.append(result)

    parsed["memories"] = validated
    return parsed


# ---------------------------------------------------------------------------
# LLM caller (gated pipeline only)
# ---------------------------------------------------------------------------

def call_llm(
    query: str,
    response: str,
    config: "LaceConfig",  # type: ignore[name-defined]  # forward ref
) -> str:
    """
    Call the configured LLM with EXTRACTION_SYSTEM_PROMPT.
    Returns raw JSON string. Supports openai, anthropic, local.
    """
    from lace.core.config import LaceConfig as _LaceConfig

    user_content = (
        f"Extract memory from this interaction:\n\n"
        f"QUERY:\n{query}\n\n"
        f"RESPONSE:\n{response}"
    )

    # Resolve provider: use config.provider.default, model from config.extraction
    provider_name = config.provider.default.lower()
    model = config.extraction.extraction_model

    if provider_name in ("openai", "local"):
        from openai import OpenAI
        client_kwargs: dict = {}
        if provider_name == "local":
            import os as _os
            client_kwargs["base_url"] = _os.getenv(
                "LACE_LOCAL_LLM_URL", "http://localhost:11434/v1"
            )
            client_kwargs["api_key"] = "local"

        client = OpenAI(**client_kwargs)
        result = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=1024,
        )
        return result.choices[0].message.content

    elif provider_name == "anthropic":
        import anthropic
        client = anthropic.Anthropic()
        result = client.messages.create(
            model=model,
            max_tokens=1024,
            system=EXTRACTION_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.1,
        )
        return result.content[0].text

    elif provider_name == "ollama":
        from openai import OpenAI
        import os as _os
        ollama_host = config.provider.ollama.host
        client = OpenAI(base_url=f"{ollama_host}/v1", api_key="ollama")
        result = client.chat.completions.create(
            model=config.provider.ollama.model,
            messages=[
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
                {"role": "user", "content": user_content},
            ],
            temperature=0.1,
            max_tokens=1024,
        )
        return result.choices[0].message.content

    else:
        raise ValueError(
            f"[Extractor] Unknown LLM provider: '{provider_name}'. "
            "Expected 'openai', 'anthropic', 'ollama', or 'local'."
        )


# ---------------------------------------------------------------------------
# Main entry points
# ---------------------------------------------------------------------------

def extract_memories(
    query: str,
    response: str,
    config=None,
    log_db_path: _Path | None = None,
) -> list[dict]:
    """
    Full worthiness-gated extraction for a single query+response pair.

    Steps:
      1. Call LLM with EXTRACTION_SYSTEM_PROMPT
      2. Parse + validate JSON → {worth_remembering, reason, memories}
      3. Log verdict to pipeline_log.db (if log_all_verdicts=True)
      4. Gate on worth_remembering (if require_worthiness_verdict=True)
      5. Return validated memory dicts

    Returns
    -------
    list[dict] — ready for dedup_and_store(); empty if gated out or error.
    """
    from lace.core.config import LaceConfig as _LaceConfig, resolve_lace_paths
    if config is None:
        config = _LaceConfig()

    if log_db_path is None:
        log_db_path = resolve_lace_paths(getattr(config, "lace_home", None))["pipeline_log"]

    extraction_cfg: _ExtractionConfig = config.extraction

    try:
        raw = call_llm(query, response, config)
    except Exception as e:
        _logger.error(f"[Extractor] LLM call failed: {e}")
        return []

    parsed = parse_extraction_response(raw)
    worth = parsed["worth_remembering"]
    reason = parsed["reason"]
    memories = parsed["memories"]

    _logger.info(
        f"[Extractor] Verdict: worth={worth} | "
        f"memories={len(memories)} | reason='{reason[:80]}'"
    )

    if extraction_cfg.log_all_verdicts:
        from lace.memory.normalize import canonical_hash as _ch
        log_extraction_event(
            queue_id=-1,
            worth_remembering=worth,
            reason=reason,
            memory_count=len(memories),
            canonical_hash_value=_ch(f"{query}\n{response}"),
            db_path=log_db_path,
        )

    if extraction_cfg.require_worthiness_verdict and not worth:
        _logger.debug(f"[Extractor] Gated out — reason: {reason}")
        return []

    return memories


def process_queue_item(
    item,
    config=None,
    log_db_path: _Path | None = None,
    raise_on_llm_error: bool = False,
) -> list[dict]:
    """
    Process a single extraction_queue row through the gated pipeline.

    Unlike extract_memories(), this function has access to item["id"]
    (the queue_id) so pipeline_log rows carry the correct correlation key.

    Returns
    -------
    list[dict] — validated memory dicts (may be empty if gated out).
    """
    from lace.core.config import LaceConfig as _LaceConfig, resolve_lace_paths
    if config is None:
        config = _LaceConfig()

    if log_db_path is None:
        log_db_path = resolve_lace_paths(getattr(config, "lace_home", None))["pipeline_log"]

    extraction_cfg: _ExtractionConfig = config.extraction

    query: str = item["query"]
    response: str = item["response"]
    queue_id: int = item["id"]
    item_hash: str = item["canonical_hash"]

    try:
        raw = call_llm(query, response, config)
    except Exception as e:
        _logger.error(
            f"[Extractor] LLM call failed for queue_id={queue_id}: {e}"
        )
        if raise_on_llm_error:
            raise
        return []

    parsed = parse_extraction_response(raw)
    worth = parsed["worth_remembering"]
    reason = parsed["reason"]
    memories = parsed["memories"]

    _logger.info(
        f"[Extractor] queue_id={queue_id} | worth={worth} | "
        f"memories={len(memories)} | reason='{reason[:80]}'"
    )

    if extraction_cfg.log_all_verdicts:
        log_extraction_event(
            queue_id=queue_id,
            worth_remembering=worth,
            reason=reason,
            memory_count=len(memories),
            canonical_hash_value=item_hash,
            db_path=log_db_path,
        )

    if extraction_cfg.require_worthiness_verdict and not worth:
        _logger.debug(
            f"[Extractor] queue_id={queue_id} gated out. Reason: {reason}"
        )
        return []

    return memories
