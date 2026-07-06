"""Core ask engine — retrieval + prompt construction + LLM call.

This is what powers `lace ask`. It:
  1. Resolves active scope
  2. Retrieves relevant memories (unless --no-memory)
  3. Loads identity and preferences
  4. Constructs a prompt with memory context injected
  5. Streams the response from the configured LLM
"""



from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator
from typing import Iterator

from lace.core.config import LaceConfig, get_lace_home, load_config
from lace.core.identity import compose_identity
from lace.core.scope import get_active_scope
from lace.memory.models import RetrievalResult
from lace.memory.store import MemoryStore
from lace.utils.providers import LLMProvider, get_provider
from lace.utils.tokens import estimate_tokens, truncate_to_token_limit


# ── Ask result ────────────────────────────────────────────────────────────────

@dataclass
class AskResult:
    """Result from a lace ask call."""
    query: str
    response: str
    memories_used: list[RetrievalResult]
    scope: str
    provider: str
    model: str
    total_tokens_estimate: int


# ── Prompt construction ───────────────────────────────────────────────────────

def build_system_prompt(
    identity: str,
    preferences: dict,
    memories: list[RetrievalResult],
    scope: str,
) -> str:
    """Build the system prompt with identity + memory context.

    Structure:
      [Identity]
      [Preferences summary]
      [Retrieved memories — if any]
      [Scope context]
    """
    sections: list[str] = []

    # Identity
    sections.append(identity.strip())

    # Preferences summary (compact)
    if preferences:
        coding = preferences.get("coding", {})
        if coding:
            pref_lines = []
            if coding.get("language"):
                pref_lines.append(f"Language: {coding['language']} {coding.get('version', '')}")
            if coding.get("style"):
                pref_lines.append(f"Style: {coding['style']}")
            if coding.get("testing_framework"):
                pref_lines.append(f"Testing: {coding['testing_framework']}")
            if coding.get("type_hints"):
                pref_lines.append(f"Type hints: {coding['type_hints']}")

            if pref_lines:
                sections.append("## Active Preferences\n" + "\n".join(f"- {p}" for p in pref_lines))

    # Retrieved memories
    if memories:
        memory_lines = [
            "## Context from Your Knowledge Base",
            "",
            "Use these memories ONLY if they directly apply to the question.",
            "If a memory is not relevant, ignore it completely.",
            "Never mention a memory just because it exists in the list.",
            "",
        ]
        for i, result in enumerate(memories, 1):
            m = result.memory
            memory_lines.append(f"### [{i}] {m.display_summary()}")
            memory_lines.append(m.content)
            if m.tags:
                memory_lines.append(f"*Tags: {', '.join(m.tags)}*")
            memory_lines.append("")

        sections.append("\n".join(memory_lines))


    # Scope context
    if scope != "global":
        sections.append(f"## Current Scope\nYou are working in: {scope}")

    return "\n\n---\n\n".join(sections)


def build_user_message(
    query: str,
    memories: list[RetrievalResult],
) -> str:
    """Build the user message.

    If memories were retrieved, remind the model to use them.
    """
    if memories:
        return (
            f"{query}\n\n"
            f"[Note: {len(memories)} relevant memories were retrieved from your knowledge base. "
            f"Please reference them in your response where applicable.]"
        )
    return query


# ── Main ask function ─────────────────────────────────────────────────────────

def ask(
    query: str,
    use_memory: bool = True,
    scope: str | None = None,
    max_memories: int = 5,
    lace_home=None,
    config: LaceConfig | None = None,
) -> tuple[list[RetrievalResult], Iterator[str], LLMProvider]:
    """Core ask function — returns memories, response stream, and provider.

    Args:
        query: The user's question.
        use_memory: Whether to retrieve and inject memories.
        scope: Override active scope (None = auto-detect).
        max_memories: Max memories to retrieve and inject.
        lace_home: Override LACE home directory.
        config: Override config (loads from disk if None).

    Returns:
        Tuple of (retrieved_memories, response_stream, provider)
    """
    if lace_home is None:
        lace_home = get_lace_home()
    if config is None:
        config = load_config(lace_home)

    # Resolve scope
    resolved_scope = scope or get_active_scope(lace_home)

    # Load identity
    identity, preferences = compose_identity(lace_home, scope=resolved_scope, config=config)

    # Retrieve memories
    memories: list[RetrievalResult] = []
    if use_memory:
        store = MemoryStore(lace_home=lace_home, config=config)

        # Initialize multi-signal retrieval indices.
        try:
            store.initialize()
        except Exception as e:
            import logging
            logging.getLogger("lace.utils.ask").warning(
                f"MemoryStore.initialize() failed, falling back to classic search: {e}"
            )

        memories = store.search(
            query=query,
            scope=resolved_scope,
            max_results=max_memories,
        )

        # Update access counts only for memories we actually use
        for result in memories:
            result.memory.touch()
            store.save(result.memory)

    # Build prompts
    context_window = _get_context_window(config)
    max_memory_tokens = context_window // 4  # Reserve 25% of context for memories

    system_prompt = build_system_prompt(
        identity=identity,
        preferences=preferences,
        memories=memories,
        scope=resolved_scope,
    )

    # Safety: truncate if system prompt is too large
    system_prompt = truncate_to_token_limit(system_prompt, max_memory_tokens)

    user_message = build_user_message(query, memories)

    # Get provider and stream
    provider = get_provider(config)
    stream = provider.stream_response(system_prompt, user_message)

    # Wrap stream to log interaction after completion
    def _logged_stream() -> Iterator[str]:
        import time
        chunks: list[str] = []
        stream_start = time.perf_counter()

        for chunk in stream:
            chunks.append(chunk)
            yield chunk

        response_text  = "".join(chunks)
        total_latency  = (time.perf_counter() - stream_start) * 1000

        # Log interaction
        try:
            from lace.utils.logging import RetrievalLogger
            logger = RetrievalLogger(lace_home)
            logger.log_interaction(
                query=query,
                response_length=len(response_text),
                provider=provider.provider_name,
                model=provider.model_name,
                memories_used=len(memories),
                latency_ms=total_latency,
            )
        except Exception:
            pass

        # Auto-extraction (if enabled in config)
        try:
            if config.memory.auto_extract and use_memory:
                from lace.memory.extractor import (
                    extract_and_store_memories,
                    should_attempt_extraction,
                )
                if should_attempt_extraction(query, response_text):
                    store = MemoryStore(lace_home=lace_home, config=config)
                    from lace.core.config import resolve_lace_paths
                    result = extract_and_store_memories(
                        query=query,
                        response=response_text,
                        store=store,
                        scope=resolved_scope,
                        config=config,
                        dry_run=config.memory.require_confirmation,
                        log_db_path=resolve_lace_paths(lace_home)["pipeline_log"],
                    )
                    if result.stored:
                        # Print to stderr so it doesn't corrupt stdout
                        import sys
                        print(
                            f"\n[LACE] Auto-extracted {len(result.stored)} memories",
                            file=sys.stderr,
                        )
        except Exception:
            pass  # Extraction never breaks the main flow

    return memories, _logged_stream(), provider


def _get_context_window(config: LaceConfig) -> int:
    """Get the context window size for the active provider."""
    provider_name = config.provider.default
    if provider_name == "ollama":
        return config.provider.ollama.context_window
    elif provider_name == "openai":
        return config.provider.openai.context_window
    elif provider_name == "anthropic":
        return config.provider.anthropic.context_window
    return 8192  # Safe default
