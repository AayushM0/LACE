"""LACE MCP Server.

Exposes LACE memory operations as MCP tools and resources.
Communicates via stdio (JSON-RPC) — no HTTP server needed.

Usage:
    lace mcp start         # Started by MCP client (Antigravity/Cursor)
    lace mcp start --debug # With debug logging
"""

from __future__ import annotations

import os
import asyncio
import warnings

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"


import logging
import sys




from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from lace.mcp.tools import (
    search_memory,
    get_project_context,
    remember,
    list_memories,
    forget_memory,
    get_related_concepts,
    set_context,
    get_relevant_context,
    process_interaction,
    initialize_lace_session,
)
from lace.mcp.resources import (
    get_patterns_resource,
    get_decisions_resource,
    get_project_context_resource,
    get_debug_log_resource,
    get_instructions_resource,
)


# NEW: Session history for multi-turn extraction context
_mcp_session_history: list[dict] = []
_MAX_HISTORY_TURNS: int = 5


def _update_session_history(query: str, response: str) -> None:
    """
    Appends the current turn to session history and trims to last N turns.
    
    Called by process_interaction before enqueuing.
    The history snapshot is passed to the queue so the worker has
    multi-turn context for extraction.
    
    Thread safety note: MCP tools run in the main thread (stdio is
    sequential), so no lock needed here.
    """
    global _mcp_session_history
    from datetime import datetime, timezone
    
    _mcp_session_history.append({
        "query": query,
        "response": response,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    
    # Trim to last N turns — we don't need the full history, just enough
    # context for the extractor to understand multi-turn decisions
    if len(_mcp_session_history) > _MAX_HISTORY_TURNS:
        _mcp_session_history = _mcp_session_history[-_MAX_HISTORY_TURNS:]


# ── Server setup ──────────────────────────────────────────────────────────────

def create_server() -> Server:
    """Create and configure the LACE MCP server."""
    server = Server("lace")

    # ── Tool definitions ──────────────────────────────────────────────────────

    @server.list_tools()
    async def list_tools() -> list[types.Tool]:
        return [
            types.Tool(
                name="search_memory",
                description=(
                    "Search your knowledge base for memories relevant to a query. "
                    "Use when the user asks about something they might have encountered, "
                    "decided, or learned before. Returns memories ranked by relevance."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "What to search for (natural language).",
                        },
                        "scope": {
                            "type": "string",
                            "description": "auto, global, or project:<name>",
                            "default": "auto",
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Max memories to return (default 5).",
                            "default": 5,
                        },
                        "category": {
                            "type": "string",
                            "description": "all, pattern, decision, debug, reference, preference",
                            "default": "all",
                        },
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="get_project_context",
                description=(
                    "Get the current project's identity, preferences, rules, and conventions. "
                    "Use at the start of a conversation to understand the user's context, "
                    "coding standards, and project-specific decisions."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {},
                    "required": [],
                },
            ),
            types.Tool(
                name="remember",
                description=(
                    "Store a new piece of knowledge for future retrieval. "
                    "Use when the user discovers something worth remembering: "
                    "a pattern, a decision rationale, or a debugging insight."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "content": {
                            "type": "string",
                            "description": "The knowledge to store (be specific and actionable).",
                        },
                        "category": {
                            "type": "string",
                            "description": "pattern, decision, debug, reference, preference",
                            "default": "pattern",
                        },
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Optional tags for filtering.",
                            "default": [],
                        },
                        "scope": {
                            "type": "string",
                            "description": "auto, global, or project:<name>",
                            "default": "auto",
                        },
                    },
                    "required": ["content"],
                },
            ),
            types.Tool(
                name="list_memories",
                description=(
                    "List stored memories with optional filtering. "
                    "Use when the user wants to browse what the system remembers. "
                    "For searching by content, use search_memory instead."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "description": "all, pattern, decision, debug, reference, preference",
                            "default": "all",
                        },
                        "scope": {
                            "type": "string",
                            "description": "auto, global, or project:<name>",
                            "default": "auto",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max memories to return.",
                            "default": 20,
                        },
                    },
                    "required": [],
                },
            ),
            types.Tool(
                name="forget_memory",
                description=(
                    "Archive a memory so it no longer appears in search results. "
                    "The memory is NOT deleted — just deprioritized. "
                    "Use when the user says a memory is outdated or wrong."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "memory_id": {
                            "type": "string",
                            "description": "The ID of the memory to archive (e.g. mem_abc123).",
                        },
                    },
                    "required": ["memory_id"],
                },
            ),
            types.Tool(
                name="set_context",
                description=(
                    "Set the current working directory context for LACE. "
                    "Call this when switching to a different project folder. "
                    "Affects scope resolution for future remember/search calls."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "working_directory": {
                            "type": "string",
                            "description": "Absolute path to the current working directory",
                        },
                        "project_name": {
                            "type": "string",
                            "description": "Optional explicit project name (overrides git detection)",
                        },
                    },
                    "required": ["working_directory"],
                },
            ),
            types.Tool(
                name="get_related_concepts",
                description=(
                    "Find concepts and memories related to a given concept "
                    "by traversing the knowledge graph. Use when you need "
                    "broader context around a topic beyond semantic similarity."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "concept": {
                            "type": "string",
                            "description": "Concept to find related nodes for.",
                        },
                        "depth": {
                            "type": "integer",
                            "description": "Hop depth (1=direct neighbors, 2=extended).",
                            "default": 2,
                        },
                        "memories_only": {
                            "type": "boolean",
                            "description": "Return only memory nodes.",
                            "default": False,
                        },
                    },
                    "required": ["concept"],
                },
            ),
            types.Tool(
                name="get_relevant_context",
                description=(
                    "ALWAYS call this tool at the start of EVERY conversation turn, "
                    "BEFORE generating any response. Pass the user's exact message as "
                    "the query. Returns relevant memories from the LACE vault formatted "
                    "as markdown. Inject the returned content into your system context "
                    "before responding — this gives you persistent memory of past "
                    "decisions, patterns, and debug insights. If nothing relevant is "
                    "found, returns empty string. Never skip this call."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The user's exact message.",
                        },
                        "scope": {
                            "type": "string",
                            "description": "auto, global, or project:<name>",
                            "default": "auto",
                        },
                    },
                    "required": ["query"],
                },
            ),
            types.Tool(
                name="process_interaction",
                description=(
                    "ALWAYS call this tool immediately AFTER generating every response, "
                    "before the conversation turn ends. Pass the user's original query "
                    "and your complete response. LACE analyzes the conversation in the "
                    "background and automatically extracts important decisions, patterns, "
                    "debug insights, and architectural choices into persistent memory. "
                    "This call is instant — it never blocks. Do not skip this call even "
                    "for short responses. Consistent calling builds the memory base that "
                    "makes get_relevant_context more useful over time."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The user's original query.",
                        },
                        "response": {
                            "type": "string",
                            "description": "The agent's complete response.",
                        },
                        "scope": {
                            "type": "string",
                            "description": "auto, global, or project:<name>",
                            "default": "auto",
                        },
                        "context_hint": {
                            "type": "string",
                            "description": "Optional category: debugging_insight, architectural_decision, user_preference, repeated_action, general_knowledge",
                        },
                    },
                    "required": ["query", "response"],
                },
            ),
            types.Tool(
                name="initialize_lace_session",
                description=(
                    "Call this tool AUTOMATICALLY at the start of every new conversation "
                    "before the user sends any message. This initializes the LACE memory session, "
                    "sets the active project context, and confirms memory tools are active."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "working_directory": {
                            "type": "string",
                            "description": "The full absolute path to the project's working directory."
                        }
                    },
                    "required": [],
                },
            ),
        ]

    # ── Tool call handler ─────────────────────────────────────────────────────

    @server.call_tool()
    async def call_tool(
        name: str,
        arguments: dict,
    ) -> list[types.TextContent]:
        import json

        try:
            if name == "search_memory":
                result = await search_memory(
                    query=arguments["query"],
                    scope=arguments.get("scope", "auto"),
                    max_results=arguments.get("max_results", 5),
                    category=arguments.get("category", "all"),
                )
            elif name == "get_project_context":
                result = await get_project_context()
            elif name == "remember":
                result = await remember(
                    content=arguments["content"],
                    category=arguments.get("category", "pattern"),
                    tags=arguments.get("tags", []),
                    scope=arguments.get("scope", "auto"),
                )
            elif name == "list_memories":
                result = await list_memories(
                    category=arguments.get("category", "all"),
                    scope=arguments.get("scope", "auto"),
                    limit=arguments.get("limit", 20),
                    lifecycle=arguments.get("lifecycle", "all"),
                )
            elif name == "forget_memory":
                result = await forget_memory(
                    memory_id=arguments["memory_id"],
                )
            elif name == "get_related_concepts":
                result = await get_related_concepts(
                    concept=arguments["concept"],
                    depth=arguments.get("depth", 2),
                    memories_only=arguments.get("memories_only", False),
                )
            elif name == "set_context":
                result = await set_context(
                    working_directory=arguments["working_directory"],
                    project_name=arguments.get("project_name"),
                )
            elif name == "get_relevant_context":
                result = await get_relevant_context(
                    query=arguments["query"],
                    scope=arguments.get("scope", "auto"),
                )
            elif name == "process_interaction":
                result = await process_interaction(
                    query=arguments["query"],
                    response=arguments["response"],
                    scope=arguments.get("scope", "auto"),
                    context_hint=arguments.get("context_hint"),
                )
            elif name == "initialize_lace_session":
                result = await initialize_lace_session(
                    working_directory=arguments.get("working_directory", "")
                )
            else:
                result = {"error": f"Unknown tool: {name}"}

        except Exception as e:
            result = {"error": str(e), "tool": name}

        return [types.TextContent(
            type="text",
            text=json.dumps(result, indent=2, default=str),
        )]

    # ── Resource definitions ──────────────────────────────────────────────────

    @server.list_resources()
    async def list_resources() -> list[types.Resource]:
        return [
            types.Resource(
                uri="memory://patterns",
                name="Stored Patterns",
                description="All stored coding patterns and best practices.",
                mimeType="text/markdown",
            ),
            types.Resource(
                uri="memory://decisions",
                name="Architectural Decisions",
                description="All stored architectural decisions and rationale.",
                mimeType="text/markdown",
            ),
            types.Resource(
                uri="memory://project-context",
                name="Project Context",
                description="Current project identity, preferences, and rules.",
                mimeType="text/markdown",
            ),
            types.Resource(
                uri="memory://debug-log",
                name="Debug Log",
                description="Past debugging insights and solutions.",
                mimeType="text/markdown",
            ),
            types.Resource(
                uri="memory://instructions",
                name="LACE Memory Protocol Instructions",
                description="Instructions on how to operate with persistent LACE memory via MCP.",
                mimeType="text/markdown",
            ),
        ]

    @server.read_resource()
    async def read_resource(uri: str) -> str:
        uri_str = str(uri)
        if uri_str == "memory://patterns":
            return await get_patterns_resource()
        elif uri_str == "memory://decisions":
            return await get_decisions_resource()
        elif uri_str == "memory://project-context":
            return await get_project_context_resource()
        elif uri_str == "memory://debug-log":
            return await get_debug_log_resource()
        elif uri_str == "memory://instructions":
            return await get_instructions_resource()
        else:
            return f"Unknown resource: {uri_str}"

    return server


# ── Entry point ───────────────────────────────────────────────────────────────

async def run_server(debug: bool = False) -> None:
    """Run the MCP server over stdio."""
    if debug:
        logging.basicConfig(level=logging.DEBUG, stream=sys.stderr)
    else:
        logging.basicConfig(level=logging.WARNING, stream=sys.stderr)

    # Initialize queue DB and start worker thread (daemon)
    try:
        from lace.mcp.queue import init_queue_db, start_worker_thread
        init_queue_db()
        start_worker_thread()
        print("[LACE] SQLite queue and background worker initialized.", file=sys.stderr, flush=True)
    except Exception as e:
        print(f"[LACE] Warning: could not initialize extraction queue: {e}", file=sys.stderr, flush=True)

    # ── Pre-warm embedding model in background ─────────────────────────────
    # Load the model in a background thread so the client's handshake doesn't timeout.
    def pre_warm():
        try:
            from lace.core.config import load_config, get_lace_home
            from lace.retrieval.embeddings import get_model
            config = load_config(get_lace_home())
            model_name = config.embeddings.model
            print(f"[LACE] Pre-warming embedding model: {model_name}", file=sys.stderr, flush=True)
            get_model(model_name)
            print("[LACE] Embedding model ready.", file=sys.stderr, flush=True)
        except Exception as e:
            print(f"[LACE] Warning: could not pre-warm model: {e}", file=sys.stderr, flush=True)

    import threading
    threading.Thread(target=pre_warm, daemon=True).start()

    server = create_server()

    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options(),
        )