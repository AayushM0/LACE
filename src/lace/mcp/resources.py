"""MCP resource definitions for LACE.

Resources are read-only data exposed to the MCP client.
Unlike tools (which are called with arguments), resources
are accessed by URI and return structured content.
"""

from __future__ import annotations

from lace.core.config import get_lace_home
from lace.core.identity import compose_identity
from lace.core.scope import get_active_scope
from lace.memory.store import MemoryStore


def _get_store() -> tuple[MemoryStore, str]:
    """Return store and active scope."""
    lace_home = get_lace_home()
    from lace.core.config import load_config
    config = load_config(lace_home)
    store = MemoryStore(lace_home=lace_home, config=config)
    scope = get_active_scope(lace_home)
    return store, scope


async def get_patterns_resource() -> str:
    """Return all pattern memories as markdown."""
    store, scope = _get_store()
    memories = store.list(category="pattern", limit=50)

    if not memories:
        return "# Patterns\n\nNo patterns stored yet."

    lines = ["# Stored Patterns\n"]
    for m in memories:
        lines.append(f"## {m.display_summary()}")
        lines.append(f"**Confidence:** {m.confidence:.2f} | **Tags:** {', '.join(m.tags)}")
        lines.append(f"\n{m.content}\n")

    return "\n".join(lines)


async def get_decisions_resource() -> str:
    """Return all decision memories as markdown."""
    store, scope = _get_store()
    memories = store.list(category="decision", limit=50)

    if not memories:
        return "# Decisions\n\nNo decisions stored yet."

    lines = ["# Architectural Decisions\n"]
    for m in memories:
        lines.append(f"## {m.display_summary()}")
        lines.append(f"**Confidence:** {m.confidence:.2f} | **Tags:** {', '.join(m.tags)}")
        lines.append(f"\n{m.content}\n")

    return "\n".join(lines)


async def get_project_context_resource() -> str:
    """Return current project context as markdown."""
    lace_home = get_lace_home()
    scope = get_active_scope(lace_home)
    identity, preferences = compose_identity(lace_home, scope=scope)

    import yaml
    prefs_str = yaml.dump(preferences, default_flow_style=False)

    return (
        f"# Project Context\n\n"
        f"**Active Scope:** {scope}\n\n"
        f"## Identity\n\n{identity}\n\n"
        f"## Preferences\n\n```yaml\n{prefs_str}```"
    )


async def get_debug_log_resource() -> str:
    """Return all debug memories as markdown."""
    store, scope = _get_store()
    memories = store.list(category="debug", limit=50)

    if not memories:
        return "# Debug Log\n\nNo debug insights stored yet."

    lines = ["# Debug Log\n"]
    for m in memories:
        lines.append(f"## {m.display_summary()}")
        lines.append(f"**Confidence:** {m.confidence:.2f} | **Tags:** {', '.join(m.tags)}")
        lines.append(f"\n{m.content}\n")

    return "\n".join(lines)


async def get_instructions_resource() -> str:
    """Return LACE active memory protocol instructions."""
    return """# LACE Active Memory Protocol

You are operating with persistent memory via LACE MCP.

MANDATORY BEHAVIOR — every turn, no exceptions:
1. Call get_relevant_context(query) BEFORE generating any response
2. Generate your response using the injected context  
3. Call process_interaction(query, response) AFTER your response

These are not optional. Start from turn 1. 
Do not wait to be reminded. This is your default operating mode.
"""