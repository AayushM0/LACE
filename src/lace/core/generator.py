from __future__ import annotations
import re
from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from lace.memory.store import MemoryStore
    from lace.memory.models import MemoryObject

SYNTHESIS_PROMPT = """
You are generating a project context file for AI coding assistants.
Below are memories captured from real development sessions on this project.
Synthesize them into a clean, structured markdown document.

Rules:
- Write in present tense, declarative style
- Group related items under clear headings
- Do not mention LACE, memory IDs, or confidence scores
- Do not include generic advice that applies to any project
- Only include information specific and actionable for THIS project
- Merge duplicate or overlapping memories into single statements
- Use bullet points under each heading
- Maximum 500 words total
- If a section has no memories, omit that section entirely

Project: {project_name}

DECISIONS MADE:
{decisions}

PATTERNS ESTABLISHED:
{patterns}

DEBUG FIXES AND KNOWN ISSUES:
{debug}

REFERENCE INFORMATION:
{reference}

RELEVANT GLOBAL KNOWLEDGE:
{global_knowledge}

Generate the context file now.
Start directly with the markdown, beginning with # {project_name}
Do not add any preamble, explanation, or closing remarks.
"""

FORMAT_MENU = """
Select output format:
  [1] AGENTS.md        — works with Claude Code, Cursor, Codex, Windsurf
  [2] CLAUDE.md        — Claude Code specific
  [3] LACE.context.md  — generic, no tool assumptions
  [4] All three simultaneously

Choice [1]: """

FORMAT_MAP = {
    1: ["AGENTS.md"],
    2: ["CLAUDE.md"],
    3: ["LACE.context.md"],
    4: ["AGENTS.md", "CLAUDE.md", "LACE.context.md"],
}


def _extract_project_tags(memories: list) -> list[str]:
    """Extract most frequent tags from a list of memories."""
    all_tags = []
    for memory in memories:
        all_tags.extend(memory.tags or [])
    counter = Counter(all_tags)
    return [tag for tag, _ in counter.most_common(5)]


def _get_project_root() -> Path:
    """Walk up from cwd to find git root. Falls back to cwd."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        if (parent / ".git").exists():
            return parent
    return current


def _ask_format_choice() -> int:
    """Show format menu and return validated choice 1-4."""
    while True:
        try:
            raw = input(FORMAT_MENU).strip()
            if raw == "":
                return 1
            choice = int(raw)
            if choice in FORMAT_MAP:
                return choice
            print("  Please enter 1, 2, 3, or 4.")
        except (ValueError, KeyboardInterrupt):
            print("\nAborted.")
            raise SystemExit(0)


def _get_filenames(choice: int) -> list[str]:
    """Return list of filenames for the given format choice."""
    return FORMAT_MAP.get(choice, ["AGENTS.md"])


def _format_memories_for_prompt(memories: list) -> str:
    """Format a list of MemoryObject into a numbered string."""
    if not memories:
        return "None"
    lines = []
    for i, memory in enumerate(memories, 1):
        content = memory.content.strip()
        lines.append(f"{i}. {content}")
    return "\n".join(lines)


def load_memories_for_generation(
    project_scope: str,
    store: MemoryStore,
    min_confidence: float = 0.65,
    global_relevance_threshold: float = 0.55,
) -> dict[str, list]:
    """
    Load and group memories for context generation.
    Returns dict with keys: decision, pattern, debug, 
    reference, global
    """
    # Load all active memories for this project scope
    all_memories = store.list(
        scope=project_scope,
        include_archived=False,
    )

    # Filter by minimum confidence
    project_memories = [
        m for m in all_memories
        if (m.confidence or 0.0) >= min_confidence
    ]

    # Extract top tags from project memories for global search
    project_tags = _extract_project_tags(project_memories)

    # Find relevant global memories using project tags as queries
    global_memories = []
    seen_ids = {m.id for m in project_memories}

    for tag in project_tags:
        try:
            results = store.search(
                query=tag,
                scope="global",
                max_results=3,
            )
            for result in results:
                mem = result.memory
                if (
                    mem.id not in seen_ids
                    and result.relevance_score >= global_relevance_threshold
                ):
                    global_memories.append(mem)
                    seen_ids.add(mem.id)
        except Exception:
            continue

    # Group project memories by category
    return {
        "decision": [
            m for m in project_memories
            if m.category == "decision"
        ],
        "pattern": [
            m for m in project_memories
            if m.category == "pattern"
        ],
        "debug": [
            m for m in project_memories
            if m.category == "debug"
        ],
        "reference": [
            m for m in project_memories
            if m.category == "reference"
        ],
        "global": global_memories,
    }


def synthesize_context(
    grouped: dict[str, list],
    project_name: str,
    provider_config: dict | None = None,
) -> str:
    """
    Call the configured LLM provider to synthesize memories
    into a structured markdown context file.
    Returns the raw markdown string.
    """
    from lace.utils.providers import get_provider
    from lace.core.config import load_config, get_lace_home

    prompt = SYNTHESIS_PROMPT.format(
        project_name=project_name,
        decisions=_format_memories_for_prompt(
            grouped.get("decision", [])
        ),
        patterns=_format_memories_for_prompt(
            grouped.get("pattern", [])
        ),
        debug=_format_memories_for_prompt(
            grouped.get("debug", [])
        ),
        reference=_format_memories_for_prompt(
            grouped.get("reference", [])
        ),
        global_knowledge=_format_memories_for_prompt(
            grouped.get("global", [])
        ),
    )

    config = provider_config
    if config is None or isinstance(config, dict):
        config = load_config(get_lace_home())

    provider = get_provider(config)
    response = provider.complete(
        system_prompt="You are a technical documentation writer.",
        user_message=prompt,
    )
    return response.strip()
