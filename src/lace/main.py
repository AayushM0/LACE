"""LACE CLI entry point."""

from __future__ import annotations

import os
import warnings

# Suppress ALL noisy warnings — must be before any other imports
warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = os.path.expanduser("~/.cache/sentence_transformers")

from pathlib import Path
from typing import Annotated, Optional



import typer
import yaml
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import print as rprint


from lace.core.config import (
    get_lace_home,
    init_lace_home,
    load_config,
    set_config_value,
)
from lace.core.scope import (
    get_active_scope,
    detect_current_project,
    get_projects,
    create_project,
    set_project_last_used,
    get_active_session,
    create_new_session,
)
from lace.core.identity import compose_identity

from lace.core.generator import (
    load_memories_for_generation,
    synthesize_context,
    _ask_format_choice,
    _get_filenames,
    _get_project_root,
)

app = typer.Typer(
    name="lace",
    help="LACE — Local AI Context Engine",
    add_completion=False,
    rich_markup_mode="rich",
)

config_app = typer.Typer(help="Manage LACE configuration.")
app.add_typer(config_app, name="config")

memory_app = typer.Typer(help="Manage memories.")
app.add_typer(memory_app, name="memory")

project_app = typer.Typer(help="Manage projects.")
app.add_typer(project_app, name="project")

mcp_app = typer.Typer(help="MCP server management.")
app.add_typer(mcp_app, name="mcp")

console = Console()


# ── lace init ─────────────────────────────────────────────────────────────────

@app.command()
def init(
    home: Annotated[
        Optional[str],
        typer.Option("--home", help="Custom LACE home directory."),
    ] = None,
) -> None:
    """Initialize LACE — create ~/.lace directory structure."""
    lace_home = Path(home).expanduser() if home else get_lace_home()

    with console.status("[bold green]Initializing LACE...[/bold green]"):
        path, already_existed = init_lace_home(lace_home)
        from lace.mcp.queue import init_queue_db
        init_queue_db()

    if already_existed:
        console.print(
            Panel(
                f"[yellow]LACE was already initialized.[/yellow]\n\n"
                f"Home: [bold]{path}[/bold]\n\n"
                f"Any missing files/directories have been created.",
                title="[bold yellow]LACE[/bold yellow]",
                border_style="yellow",
            )
        )
    else:
        console.print(
            Panel(
                f"[bold green]✓ LACE initialized successfully![/bold green]\n\n"
                f"Home: [bold]{path}[/bold]\n\n"
                f"[dim]Next steps:[/dim]\n"
                f"  1. Edit [bold]{path}/config/identity.md[/bold]\n"
                f"  2. Edit [bold]{path}/config/preferences.yaml[/bold]\n"
                f"  3. Run [bold]lace memory add \"your first memory\"[/bold]\n"
                f"  4. Run [bold]lace mcp start[/bold] — connect to Antigravity",
                title="[bold green]LACE[/bold green]",
                border_style="green",
            )
        )


# ── lace version ──────────────────────────────────────────────────────────────

@app.command()
def version() -> None:
    """Show LACE version."""
    from lace import __version__
    console.print(f"[bold]LACE[/bold] v{__version__}")


@app.command()
def doctor() -> None:
    """Run LACE diagnostic checks."""
    import sqlite3
    from collections import Counter
    from datetime import datetime, timedelta, timezone
    from unittest.mock import patch
    import json
    from lace.core.config import resolve_lace_paths, load_config
    from lace.memory.pipeline_log import initialize_pipeline_log_db
    from lace.memory.store import load_all_memories

    lace_home = get_lace_home()
    paths = resolve_lace_paths(lace_home)

    rows: list[list[str]] = []

    def add_check(name: str, passed: bool, detail: str = "") -> None:
        status_str = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        rows.append([name, status_str, detail])

    # Check database WAL mode
    def check_db_wal(db_path: Path, init_fn, conn_fn, check_name: str) -> None:
        try:
            init_fn(db_path)
            conn = conn_fn(db_path)
            mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
            conn.close()
            add_check(check_name, mode == "wal", f"mode={mode}")
        except Exception as e:
            add_check(check_name, False, f"error: {e}")

    from lace.mcp.queue import init_queue_db, _get_connection as _get_queue_conn
    from lace.memory.pipeline_log import _get_connection as _get_log_conn

    check_db_wal(paths["queue_db"], init_queue_db, _get_queue_conn, "journal:queue_db")
    check_db_wal(paths["pipeline_log"], initialize_pipeline_log_db, _get_log_conn, "journal:pipeline_log")

    # Wiring check
    try:
        from lace.memory.dedup import StoreBackedVectorIndex, dedup_and_store
        from lace.memory.extractor import process_queue_item
        from lace.memory.store import MemoryStore
        from lace.retrieval.embeddings import embed_text

        config = load_config(lace_home)
        store = MemoryStore(lace_home=lace_home, config=config)
        store.initialize()
        body = "Doctor round-trip marker for LACE pipeline wiring."
        fake_item = {
            "id": "doctor-roundtrip",
            "query": "doctor pipeline check",
            "response": body,
            "scope": "global",
            "canonical_hash": f"doctor-{datetime.now(timezone.utc).timestamp()}",
        }
        fake_llm = json.dumps({
            "worth_remembering": True,
            "reason": "Pipeline wiring check with mocked LLM output.",
            "memories": [{
                "category": "debug",
                "summary": body,
                "body": body,
                "tags": ["doctor"],
                "confidence": 0.8,
            }],
        })
        with patch("lace.memory.extractor.call_llm", return_value=fake_llm):
            extracted = process_queue_item(fake_item, config=config, log_db_path=paths["pipeline_log"])
        vector_index = StoreBackedVectorIndex(paths["vector_db"])
        stored_ids: list[str] = []
        for candidate in extracted:
            candidate["project_scope"] = "global"
            new_id = dedup_and_store(
                candidate=candidate,
                vector_index=vector_index,
                memory_store=store,
                config=config,
                queue_id=fake_item["id"],
                hash_index_db_path=paths["hash_index"],
                log_db_path=paths["pipeline_log"],
            )
            if new_id:
                stored_ids.append(new_id)
        query_embedding = embed_text(body, model_name=config.embeddings.model)
        vector_hits = vector_index.query(query_embedding, n_results=3, scope_filter=["global"])
        found = any(body in hit.memory.content for hit in vector_hits)
        add_check(
            "pipeline wiring",
            found,
            "mocked LLM round trip; prompt quality not checked here",
        )

        # Cleanup mock doctor data to prevent production pollution
        for mem_id in stored_ids:
            try:
                store.delete(mem_id)
            except Exception:
                pass
            try:
                with sqlite3.connect(str(paths["hash_index"])) as conn:
                    conn.execute("DELETE FROM vault_hash_index WHERE memory_id = ?", (mem_id,))
                    conn.commit()
            except Exception:
                pass
        try:
            with sqlite3.connect(str(paths["pipeline_log"])) as conn:
                conn.execute("DELETE FROM pipeline_log WHERE queue_id = ?", (fake_item["id"],))
                conn.commit()
        except Exception:
            pass
    except Exception as e:
        add_check("pipeline wiring", False, f"{e}; prompt quality not checked here")

    table = Table(title="LACE Doctor", show_header=True, header_style="bold cyan", expand=True)
    table.add_column("Check", no_wrap=True)
    table.add_column("Status", no_wrap=True)
    table.add_column("Detail")
    for row in rows:
        table.add_row(*row)
    console.print(table)


# ── config commands ───────────────────────────────────────────────────────────

@config_app.command("show")
def config_show() -> None:
    """Show current LACE configuration."""
    lace_home = get_lace_home()
    config = load_config(lace_home)

    table = Table(title="LACE Configuration", show_header=True, header_style="bold cyan")
    table.add_column("Key", style="bold")
    table.add_column("Value")

    def flatten(d: dict, prefix: str = "") -> list[tuple[str, str]]:
        rows = []
        for k, v in d.items():
            full_key = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                rows.extend(flatten(v, full_key))
            else:
                rows.append((full_key, str(v)))
        return rows

    for key, value in flatten(config.model_dump()):
        table.add_row(key, value)

    console.print(table)
    console.print(f"\n[dim]Config file: {lace_home}/config/lace.yaml[/dim]")


@config_app.command("set")
def config_set(
    key: Annotated[str, typer.Argument(help="Config key (e.g. memory.decay_half_life_days)")],
    value: Annotated[str, typer.Argument(help="New value")],
) -> None:
    """Set a configuration value."""
    try:
        set_config_value(key, value)
        console.print(f"[green]✓[/green] Set [bold]{key}[/bold] = [bold]{value}[/bold]")
    except KeyError as e:
        console.print(f"[red]✗ Unknown config key:[/red] {e}")
        raise typer.Exit(1)
    except ValueError as e:
        console.print(f"[red]✗ Invalid value:[/red] {e}")
        raise typer.Exit(1)


# ── memory commands ───────────────────────────────────────────────────────────

def _get_store(scope: str | None = None):
    """Get a MemoryStore instance with active scope."""
    from lace.memory.store import MemoryStore
    store = MemoryStore()
    # Set active scope for store to use in searches
    if scope is None:
        scope = get_active_scope()

    # Initialize multi-signal retrieval indices.
    try:
        store.initialize()
    except Exception as e:
        import logging
        logging.getLogger("lace.main").warning(
            f"MemoryStore.initialize() failed, falling back to classic search: {e}"
        )

    return store


@memory_app.command("add")
def memory_add(
    content: Annotated[str, typer.Argument(help="The memory content to store.")],
    tag: Annotated[
        Optional[list[str]],
        typer.Option("--tag", "-t", help="Tags (repeatable: --tag=pattern --tag=db)"),
    ] = None,
    category: Annotated[
        str,
        typer.Option("--category", "-c", help="Category: pattern, decision, debug, reference, preference"),
    ] = "pattern",
    scope: Annotated[
        str,
        typer.Option("--scope", "-s", help="Scope: global or project:<name>"),
    ] = "global",
    summary: Annotated[
        Optional[str],
        typer.Option("--summary", help="One-line summary for display."),
    ] = None,
) -> None:
    """Store a new memory."""
    store = _get_store()

    try:
        memory = store.add(
            content=content,
            category=category,
            tags=tag or [],
            scope=scope,
            summary=summary,
        )
    except ValueError as e:
        console.print(f"[red]✗ Error:[/red] {e}")
        raise typer.Exit(1)

    console.print(
        Panel(
            f"[bold green]✓ Memory stored[/bold green]\n\n"
            f"ID:       [bold]{memory.id}[/bold]\n"
            f"Category: {memory.category.value}\n"
            f"Scope:    {memory.project_scope}\n"
            f"Tags:     {', '.join(memory.tags) if memory.tags else '[dim]none[/dim]'}\n\n"
            f"[dim]{memory.display_summary()}[/dim]",
            title="[bold]Memory Added[/bold]",
            border_style="green",
        )
    )


@memory_app.command("list")
def memory_list(
    category: Annotated[
        Optional[str],
        typer.Option("--category", "-c", help="Filter by category."),
    ] = None,
    scope: Annotated[
        Optional[str],
        typer.Option("--scope", "-s", help="Filter by scope."),
    ] = None,
    include_archived: Annotated[
        bool,
        typer.Option("--archived", help="Include archived memories."),
    ] = False,
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Max results to show."),
    ] = 20,
) -> None:
    """List stored memories."""
    store = _get_store()
    memories = store.list(
        category=category,
        scope=scope,
        include_archived=include_archived,
        limit=limit,
    )

    if not memories:
        console.print("[yellow]No memories found.[/yellow]")
        console.print("[dim]Try: lace memory add \"your first memory\"[/dim]")
        return

    table = Table(
        title=f"Memories ({len(memories)} shown)",
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("ID", style="dim", width=16)
    table.add_column("Category", width=10)
    table.add_column("Scope", width=14)
    table.add_column("Conf", width=5)
    table.add_column("Tags", width=20)
    table.add_column("Summary")

    for memory in memories:
        lifecycle_color = {
            "captured": "white",
            "validated": "green",
            "consolidated": "blue",
            "archived": "red",
        }.get(memory.lifecycle.value, "white")

        table.add_row(
            memory.id,
            memory.category.value,
            memory.project_scope,
            f"{memory.confidence:.2f}",
            ", ".join(memory.tags[:3]) if memory.tags else "[dim]—[/dim]",
            Text(memory.display_summary(), style=lifecycle_color),
        )

    console.print(table)


@memory_app.command("show")
def memory_show(
    memory_id: Annotated[str, typer.Argument(help="Memory ID to show.")],
) -> None:
    """Show full details of a memory."""
    store = _get_store()
    memory = store.get(memory_id)

    if memory is None:
        console.print(f"[red]✗ Memory not found:[/red] {memory_id}")
        raise typer.Exit(1)

    console.print(
        Panel(
            f"[bold]{memory.display_summary()}[/bold]\n\n"
            f"{memory.content}\n\n"
            f"[dim]─────────────────────────────────[/dim]\n"
            f"ID:           [bold]{memory.id}[/bold]\n"
            f"Category:     {memory.category.value}\n"
            f"Source:       {memory.source.value}\n"
            f"Lifecycle:    {memory.lifecycle.value}\n"
            f"Confidence:   {memory.confidence:.2f}\n"
            f"Scope:        {memory.project_scope}\n"
            f"Tags:         {', '.join(memory.tags) if memory.tags else '[dim]none[/dim]'}\n"
            f"Access count: {memory.access_count}\n"
            f"Created:      {memory.created_at.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"Last access:  {memory.last_accessed.strftime('%Y-%m-%d %H:%M UTC')}\n"
            f"File:         [dim]{memory.file_path}[/dim]",
            title=f"[bold]Memory — {memory.id}[/bold]",
            border_style="cyan",
        )
    )


@memory_app.command("forget")
def memory_forget(
    memory_id: Annotated[str, typer.Argument(help="Memory ID to archive.")],
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation."),
    ] = False,
) -> None:
    """Archive a memory (removes from search, never deletes)."""
    store = _get_store()

    memory = store.get(memory_id)
    if memory is None:
        console.print(f"[red]✗ Memory not found:[/red] {memory_id}")
        raise typer.Exit(1)

    if not yes:
        console.print(f"Archive memory: [bold]{memory.display_summary()}[/bold]")
        confirmed = typer.confirm("This will remove it from search results. Continue?")
        if not confirmed:
            console.print("[dim]Cancelled.[/dim]")
            return

    store.forget(memory_id)
    console.print(f"[green]✓[/green] Memory [bold]{memory_id}[/bold] archived.")



@memory_app.command("reindex")
def memory_reindex() -> None:
    """Re-embed all memories into the vector store."""
    store = _get_store()
    with console.status("[bold green]Re-indexing all memories...[/bold green]"):
        success, failure = store.reindex_all()
    console.print(f"[green]✓[/green] Indexed {success} memories. Failures: {failure}")



@memory_app.command("stats")
def memory_stats(
    days: Annotated[
        int,
        typer.Option("--days", "-d", help="Days of logs to analyze."),
    ] = 7,
) -> None:
    """Show memory statistics and retrieval quality dashboard."""
    from lace.utils.logging import compute_retrieval_stats, compute_storage_stats

    store    = _get_store()
    lace_home = get_lace_home()

    stats      = store.stats()
    retrieval  = compute_retrieval_stats(lace_home / "logs" / "retrieval", days=days)
    storage    = compute_storage_stats(lace_home)

    by_cat = stats["by_category"]
    by_lc  = stats["by_lifecycle"]

    # ── Memory panel ──────────────────────────────────────────────────────────
    mem_lines = [
        f"[bold]Total:[/bold] {stats['total']}  "
        f"Active: {stats['active']}  Archived: {stats['archived']}",
        "",
        "[bold]By category:[/bold]",
    ]
    for cat, count in sorted(by_cat.items()):
        bar = "█" * min(count, 20)
        mem_lines.append(f"  {cat:<14} {bar} {count}")

    mem_lines += ["", "[bold]By lifecycle:[/bold]"]
    for lc, count in sorted(by_lc.items()):
        mem_lines.append(f"  {lc:<16} {count}")

    console.print(Panel(
        "\n".join(mem_lines),
        title="[bold cyan]Memory[/bold cyan]",
        border_style="cyan",
    ))

    # ── Retrieval quality panel ───────────────────────────────────────────────
    if retrieval["total_searches"] > 0:
        ret_lines = [
            f"[bold]Searches (last {days} days):[/bold] {retrieval['total_searches']}",
            f"  Avg results/search:  {retrieval['avg_results']}",
            f"  Zero-result rate:    {retrieval['zero_result_rate']}%",
            "",
            "[bold]Latency:[/bold]",
            f"  Average: {retrieval['avg_latency_ms']}ms",
            f"  P95:     {retrieval['p95_latency_ms']}ms",
            "",
            "[bold]Quality:[/bold]",
            f"  Avg top-result score: {retrieval['avg_relevance_score']}",
        ]
        if retrieval["top_queries"]:
            ret_lines += ["", "[bold]Top search terms:[/bold]"]
            for term in retrieval["top_queries"]:
                ret_lines.append(f"  {term}")
    else:
        ret_lines = [
            f"[dim]No retrieval data for the last {days} days.[/dim]",
            "[dim]Run some searches to build up stats.[/dim]",
        ]

    console.print(Panel(
        "\n".join(ret_lines),
        title="[bold cyan]Retrieval Quality[/bold cyan]",
        border_style="cyan",
    ))

    # ── Storage panel ─────────────────────────────────────────────────────────
    console.print(Panel(
        f"[bold]Vault:[/bold]     {storage['vault']}\n"
        f"[bold]Vector DB:[/bold] {storage['vector_db']}\n"
        f"[bold]Logs:[/bold]      {storage['logs']}\n"
        f"[bold]Total:[/bold]     {storage['total']}",
        title="[bold cyan]Storage[/bold cyan]",
        border_style="cyan",
    ))

@memory_app.command("search")
def memory_search(
    query: Annotated[str, typer.Argument(help="Search query.")],
    limit: Annotated[int, typer.Option("--limit", "-n")] = 10,
    scope: Annotated[str, typer.Option("--scope", "-s")] = "global",
    show_scores: Annotated[bool, typer.Option("--scores")] = False,
) -> None:
    """Semantic search across memories."""
    store = _get_store()

    with console.status(f"[bold green]Searching for:[/bold green] {query}"):
        results = store.search(query, scope=scope, max_results=limit)

    if not results:
        console.print(f"[yellow]No memories found for:[/yellow] {query}")
        return

    table = Table(
        title=f"Search: '{query}' ({len(results)} results)",
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("Rank", width=4)
    table.add_column("ID", style="dim", width=16)
    table.add_column("Category", width=10)
    table.add_column("Tags", width=20)
    if show_scores:
        table.add_column("Score", width=6)
    table.add_column("Summary")

    for result in results:
        m = result.memory
        row = [
            str(result.rank),
            m.id,
            m.category.value,
            ", ".join(m.tags[:3]) if m.tags else "[dim]—[/dim]",
        ]
        if show_scores:
            row.append(f"{result.relevance_score:.3f}")
        row.append(m.display_summary())
        table.add_row(*row)

    console.print(table)
    console.print(f"[dim]Match type: {results[0].match_type if results else '—'}[/dim]")


def run_extract(
    *,
    query: str,
    response: str,
    config,
    paths: dict,
    store,
    active_scope: str,
    dry_run: bool = False,
    confirm: bool = False,
    extract_fn=None,
    dedup_fn=None,
    vector_index_cls=None,
    should_extract_fn=None,
) -> dict:
    """Core extract logic — testable without Typer.

    Returns {"stored": int, "skipped": int, "memories": list[dict]}.
    """
    from lace.memory.extractor import extract_memories, should_attempt_extraction as _default_should
    from lace.memory.dedup import StoreBackedVectorIndex as _defaultVICls, dedup_and_store as _default_dedup

    should_extract = should_extract_fn or _default_should
    _extract = extract_fn or extract_memories
    _dedup = dedup_fn or _default_dedup
    _vi_cls = vector_index_cls or _defaultVICls

    if not should_extract(query, response):
        return {"stored": 0, "skipped": 0, "memories": [], "blocked": True}

    memories = _extract(query=query, response=response, config=config)

    if not memories:
        return {"stored": 0, "skipped": 0, "memories": [], "blocked": False}

    if dry_run or confirm:
        return {"stored": 0, "skipped": 0, "memories": memories, "blocked": False}

    vector_index = _vi_cls(paths["vector_db"])
    stored, skipped = 0, 0

    for mem in memories:
        mem["project_scope"] = active_scope
        result_id = _dedup(
            candidate=mem,
            vector_index=vector_index,
            memory_store=store,
            config=config,
            queue_id=None,
            hash_index_db_path=paths["hash_index"],
            log_db_path=paths["pipeline_log"],
        )
        if result_id:
            stored += 1
        else:
            skipped += 1

    return {"stored": stored, "skipped": skipped, "memories": memories, "blocked": False}


@memory_app.command("extract")
def memory_extract(
    query: Annotated[str, typer.Argument(help="The query from the conversation.")],
    response: Annotated[str, typer.Argument(help="The response from the conversation.")],
    scope: Annotated[
        Optional[str],
        typer.Option("--scope", "-s", help="Scope to store under."),
    ] = None,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be extracted without storing."),
    ] = False,
    confirm: Annotated[
        bool,
        typer.Option("--confirm", help="Preview extractions without storing (same as --dry-run)."),
    ] = False,
) -> None:
    """Extract and store knowledge from a conversation turn."""
    lace_home = get_lace_home()
    from lace.core.config import load_config, resolve_lace_paths
    config    = load_config(lace_home)
    paths     = resolve_lace_paths(lace_home)
    store     = _get_store()
    active_scope = scope or get_active_scope()

    result = run_extract(
        query=query, response=response, config=config, paths=paths,
        store=store, active_scope=active_scope, dry_run=dry_run, confirm=confirm,
    )

    if result["blocked"]:
        console.print("[yellow]This conversation turn doesn't appear to contain extractable knowledge.[/yellow]")
        return

    if not result["memories"]:
        console.print("[yellow]No knowledge worth extracting from this turn.[/yellow]")
        return

    if dry_run or confirm:
        for i, mem in enumerate(result["memories"], 1):
            console.print(Panel(
                f"[bold]Summary:[/bold] {mem.get('summary', '')}\n"
                f"[bold]Category:[/bold] {mem.get('category', '')}\n"
                f"[bold]Tags:[/bold] {', '.join(mem.get('tags', []))}\n"
                f"[bold]Confidence:[/bold] {mem.get('confidence', 0):.2f}\n"
                f"[dim]Dry run — not stored.[/dim]",
                title=f"[bold]Extraction [{i}][/bold]",
                border_style="cyan",
            ))
        return

    console.print(f"\n[green]+[/green] Stored: {result['stored']} | Skipped (duplicate): {result['skipped']}")


@memory_app.command("rate")
def memory_rate(
    memory_id: Annotated[str, typer.Argument(help="Memory ID to rate.")],
    signal: Annotated[
        str,
        typer.Argument(help="Rating: helpful, outdated, or wrong."),
    ],
) -> None:
    """Rate a memory to improve future retrieval."""
    if signal not in ("helpful", "outdated", "wrong"):
        console.print("[red]✗ Invalid signal.[/red] Use: helpful, outdated, wrong")
        raise typer.Exit(1)

    store = _get_store()
    memory = store.get(memory_id)
    if not memory:
        console.print(f"[red]✗ Memory not found:[/red] {memory_id}")
        raise typer.Exit(1)

    old_conf = memory.confidence
    success = store.rate(memory_id, signal)

    if success:
        memory = store.get(memory_id)  # reload updated version
        direction = "↑" if signal == "helpful" else "↓"
        console.print(
            Panel(
                f"[bold]Memory:[/bold] {memory.display_summary()[:60]}\n"
                f"[bold]Signal:[/bold] {signal} {direction}\n"
                f"[bold]Confidence:[/bold] {old_conf:.2f} → [bold]{memory.confidence:.2f}[/bold]",
                title="[bold green]✓ Memory Rated[/bold green]",
                border_style="green",
            )
        )
    else:
        console.print("[red]✗ Failed to rate memory.[/red]")
        raise typer.Exit(1)


@memory_app.command("review")
def memory_review(
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Max memories to review."),
    ] = 20,
) -> None:
    """
    Interactive review of low-confidence vault memories.

    Auto-extracted memories start at confidence 0.4 and appear here
    until they earn higher confidence through use (record_access) or
    explicit rating. Rate them helpful/outdated/wrong to adjust confidence.
    """
    # Counters for final summary
    marked_helpful = 0
    marked_outdated = 0
    marked_wrong = 0
    vault_skipped = 0

    typer.echo(f"\n{'='*60}")
    typer.echo("  VAULT — Low-confidence memory review")
    typer.echo(f"{'='*60}\n")

    store = _get_store()
    candidates = store.get_review_candidates(limit=limit)

    if not candidates:
        console.print("[green]✓[/green] All memories look healthy. Nothing to review.")
    else:
        console.print(
            f"[bold]Reviewing {len(candidates)} memories[/bold] "
            "(low confidence or unaccessed)\n"
        )

        for memory in candidates:
            source_label = (
                " [auto]" if memory.source.value == "auto_extracted" else ""
            )
            console.print("\n" + "─" * 60)
            console.print(
                f"[bold]ID:[/bold] {memory.id}{source_label}  |  "
                f"[bold]Conf:[/bold] {memory.confidence:.2f}  |  "
                f"[bold]Accesses:[/bold] {memory.access_count}  |  "
                f"[bold]Scope:[/bold] {memory.project_scope}"
            )
            console.print(f"[dim]{memory.content[:120]}{'...' if len(memory.content) > 120 else ''}[/dim]")

            raw = typer.prompt(
                "Action",
                default="",
                prompt_suffix=" [Enter=helpful / o=outdated / w=wrong / s=skip]: ",
            ).strip().lower()

            if raw in ("", "helpful"):
                store.rate(memory.id, "helpful")
                typer.echo(typer.style("  ✓ Marked helpful", fg=typer.colors.GREEN))
                marked_helpful += 1
            elif raw in ("o", "outdated"):
                store.rate(memory.id, "outdated")
                typer.echo(typer.style("  ✓ Marked outdated", fg=typer.colors.YELLOW))
                marked_outdated += 1
            elif raw in ("w", "wrong"):
                store.rate(memory.id, "wrong")
                typer.echo(typer.style("  ✓ Marked wrong", fg=typer.colors.RED))
                marked_wrong += 1
            else:
                vault_skipped += 1

    console.print(
        f"\n[green]✓[/green] Review complete: "
        f"{marked_helpful} marked helpful, "
        f"{marked_outdated} marked outdated, "
        f"{marked_wrong} marked wrong, "
        f"{vault_skipped} skipped"
    )


@memory_app.command("recent")
def memory_recent(
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Number of recent memories to show."),
    ] = 20,
    auto_only: Annotated[
        bool,
        typer.Option("--auto", help="Show only auto-extracted memories."),
    ] = False,
) -> None:
    """
    Show recently stored memories, newest first.

    Use --auto to filter to only auto-extracted memories — useful for
    auditing what the background extractor has been storing.
    """
    from rich.table import Table

    store = _get_store()
    memories = store.list(include_archived=False, limit=limit * 3)

    # Sort by created_at descending
    memories.sort(key=lambda m: m.created_at, reverse=True)

    if auto_only:
        memories = [m for m in memories if m.source.value == "auto_extracted"]

    memories = memories[:limit]

    if not memories:
        console.print("[dim]No memories found.[/dim]")
        return

    table = Table(title=f"Recent Memories ({'auto-extracted only' if auto_only else 'all'})")
    table.add_column("ID", style="dim", no_wrap=True)
    table.add_column("Source", no_wrap=True)
    table.add_column("Conf", justify="right")
    table.add_column("Scope")
    table.add_column("Tags")
    table.add_column("Content", max_width=60)

    for m in memories:
        source_style = "yellow" if m.source.value == "auto_extracted" else "green"
        table.add_row(
            m.id,
            f"[{source_style}]{m.source.value}[/{source_style}]",
            f"{m.confidence:.2f}",
            m.project_scope,
            ", ".join(m.tags[:3]) or "(none)",
            m.content[:80] + ("..." if len(m.content) > 80 else ""),
        )

    console.print(table)


@memory_app.command("prune")
def memory_prune(
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would be pruned without archiving."),
    ] = False,
    days: Annotated[
        int,
        typer.Option("--days", help="Minimum age in days before pruning."),
    ] = 30,
    max_confidence: Annotated[
        float,
        typer.Option("--max-confidence", help="Only prune memories below this confidence."),
    ] = 0.5,
) -> None:
    """
    Archive stale auto-extracted memories nobody has used.

    Targets auto-extracted memories that are: older than --days, below
    --max-confidence, and have zero accesses. These are memories the
    system extracted but that have never proven useful.
    """
    from datetime import datetime, timezone

    store = _get_store()
    memories = store.list(include_archived=False, limit=10_000)

    now = datetime.now(timezone.utc)
    pruned = 0
    candidates = []

    for m in memories:
        if m.source.value != "auto_extracted":
            continue
        if m.access_count > 0:
            continue
        if m.confidence >= max_confidence:
            continue

        # Ensure created_at is timezone-aware
        created = m.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=timezone.utc)

        age_days = (now - created).days
        if age_days < days:
            continue

        candidates.append((m, age_days))

    if not candidates:
        console.print(
            f"[green]✓[/green] Nothing to prune "
            f"(no auto-extracted memories older than {days}d with 0 accesses and conf < {max_confidence})."
        )
        return

    console.print(
        f"[bold]{'[DRY RUN] ' if dry_run else ''}Found {len(candidates)} memories to prune:[/bold]\n"
    )

    for m, age_days in candidates:
        console.print(
            f"  [dim]{m.id}[/dim]  conf={m.confidence:.2f}  "
            f"age={age_days}d  {m.content[:60]}{'...' if len(m.content) > 60 else ''}"
        )
        if not dry_run:
            store.forget(m.id)
            pruned += 1

    if dry_run:
        console.print(f"\n[dim]Dry run — nothing archived. Run without --dry-run to prune.[/dim]")
    else:
        console.print(f"\n[green]✓[/green] Pruned {pruned} stale auto-extracted memories.")



# Session commands

session_app = typer.Typer(help="Session management.")
app.add_typer(session_app, name="session")


@session_app.command("start")
def session_start() -> None:
    """Start a new session (temporary memory scope)."""
    session_id = create_new_session()
    console.print(f"[green]✓[/green] Started session: [bold]{session_id}[/bold]")


@session_app.command("info")
def session_info() -> None:
    """Show current active session."""
    session = get_active_session()
    if session:
        console.print(f"[bold]Active session:[/bold] {session}")
    else:
        console.print("[yellow]No active session.[/yellow]")


@session_app.command("stop")
def session_stop() -> None:
    """Stop current active session."""
    lace_home = get_lace_home()
    session_file = lace_home / "sessions" / "active"
    if session_file.exists():
        session_file.unlink()
        console.print("[green]✓[/green] Stopped active session.")
    else:
        console.print("[yellow]No active session to stop.[/yellow]")


# ── logs commands ─────────────────────────────────────────────────────────────

logs_app = typer.Typer(help="View and manage retrieval logs.")
app.add_typer(logs_app, name="logs")


@logs_app.command("show")
def logs_show(
    days: Annotated[
        int,
        typer.Option("--days", "-d", help="How many days back to show."),
    ] = 1,
    limit: Annotated[
        int,
        typer.Option("--limit", "-n", help="Max entries to show."),
    ] = 20,
    log_type: Annotated[
        str,
        typer.Option("--type", "-t", help="retrieval or interaction"),
    ] = "retrieval",
) -> None:
    """Show recent retrieval or interaction logs."""
    from lace.utils.logging import read_recent_logs

    lace_home = get_lace_home()
    log_dir   = lace_home / "logs" / (
        "retrieval" if log_type == "retrieval" else "interactions"
    )

    entries = read_recent_logs(log_dir, days=days, log_type=log_type)[:limit]

    if not entries:
        console.print(
            f"[yellow]No {log_type} logs found for the last {days} day(s).[/yellow]"
        )
        return

    table = Table(
        title=f"Recent {log_type} logs ({len(entries)} shown)",
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )

    if log_type == "retrieval":
        table.add_column("Time",      width=20)
        table.add_column("Query",     width=35)
        table.add_column("Scope",     width=18)
        table.add_column("Results",   width=7)
        table.add_column("Latency",   width=10)
        table.add_column("Top Score", width=9)

        for e in entries:
            ts      = e.get("timestamp", "")[:19].replace("T", " ")
            results = e.get("results", [])
            top     = f"{results[0]['relevance_score']:.3f}" if results else "—"
            lat     = f"{e.get('latency_ms', 0):.0f}ms"

            table.add_row(
                ts,
                e.get("query", "")[:35],
                e.get("scope", ""),
                str(e.get("total_results", 0)),
                lat,
                top,
            )
    else:
        table.add_column("Time",     width=20)
        table.add_column("Query",    width=35)
        table.add_column("Provider", width=10)
        table.add_column("Memories", width=8)
        table.add_column("Latency",  width=10)

        for e in entries:
            ts = e.get("timestamp", "")[:19].replace("T", " ")
            table.add_row(
                ts,
                e.get("query", "")[:35],
                e.get("provider", ""),
                str(e.get("memories_used", 0)),
                f"{e.get('latency_ms', 0):.0f}ms",
            )

    console.print(table)


@logs_app.command("stats")
def logs_stats(
    days: Annotated[
        int,
        typer.Option("--days", "-d", help="Days to analyze."),
    ] = 7,
) -> None:
    """Show retrieval quality statistics."""
    from lace.utils.logging import compute_retrieval_stats

    lace_home = get_lace_home()
    stats     = compute_retrieval_stats(lace_home / "logs" / "retrieval", days=days)

    if stats["total_searches"] == 0:
        console.print(
            f"[yellow]No retrieval logs found for the last {days} days.[/yellow]"
        )
        return

    lines = [
        f"[bold]Period:[/bold]         Last {days} days",
        f"[bold]Total searches:[/bold] {stats['total_searches']}",
        "",
        "[bold]Results:[/bold]",
        f"  Avg per search:   {stats['avg_results']}",
        f"  Zero-result rate: {stats['zero_result_rate']}%",
        "",
        "[bold]Latency:[/bold]",
        f"  Average: {stats['avg_latency_ms']}ms",
        f"  P95:     {stats['p95_latency_ms']}ms",
        "",
        "[bold]Quality:[/bold]",
        f"  Avg top score: {stats['avg_relevance_score']}",
    ]

    if stats["top_queries"]:
        lines += ["", "[bold]Top search terms:[/bold]"]
        for t in stats["top_queries"]:
            lines.append(f"  {t}")

    console.print(Panel(
        "\n".join(lines),
        title="[bold cyan]Retrieval Stats[/bold cyan]",
        border_style="cyan",
    ))


@logs_app.command("clear")
def logs_clear(
    yes: Annotated[
        bool,
        typer.Option("--yes", "-y", help="Skip confirmation."),
    ] = False,
    older_than: Annotated[
        int,
        typer.Option("--older-than", help="Only clear logs older than N days."),
    ] = 90,
) -> None:
    """Clear old log files."""
    from lace.utils.logging import clean_old_logs

    lace_home = get_lace_home()

    if not yes:
        confirmed = typer.confirm(
            f"Delete log files older than {older_than} days?"
        )
        if not confirmed:
            console.print("[dim]Cancelled.[/dim]")
            return

    r = clean_old_logs(lace_home / "logs" / "retrieval",    retention_days=older_than)
    i = clean_old_logs(lace_home / "logs" / "interactions", retention_days=older_than)
    console.print(f"[green]✓[/green] Deleted {r + i} log files.")




# ── wikilink commands ─────────────────────────────────────────────────────────

wikilink_app = typer.Typer(help="Inject wikilinks into memory files for Obsidian visualization.")
app.add_typer(wikilink_app, name="wikilink")


@wikilink_app.command("inject")
def wikilink_inject() -> None:
    """Inject wikilinks into all memory files based on knowledge graph.
    
    This creates [[concept]] links in your memory markdown files, allowing
    Obsidian to render an interactive knowledge graph visualization.
    """
    from lace.graph.wikilinks import inject_wikilinks_all
    
    console.print("[cyan]Injecting wikilinks into memory files...[/cyan]")
    
    result = inject_wikilinks_all()
    
    console.print(f"[green]✅ Updated {result['updated']}/{result['total']} memory files[/green]")
    console.print(f"[dim]Wikilinks added based on knowledge graph relationships[/dim]")


@wikilink_app.command("status")
def wikilink_status() -> None:
    """Show wikilink injection status."""
    from lace.core.config import get_lace_home, load_config
    
    lace_home = get_lace_home()
    config = load_config(lace_home)
    vault_path = config.vault_path(lace_home)
    
    from lace.graph.wikilinks import extract_existing_wikilinks
    
    files_with_links = 0
    total_links = 0
    total_files = 0
    
    # Find all memory files
    for md_path in vault_path.rglob("mem_*.md"):
        total_files += 1
        content = md_path.read_text()
        links = extract_existing_wikilinks(content)
        if links:
            files_with_links += 1
            total_links += len(links)
    
    console.print("[bold]Wikilink Status[/bold]")
    console.print(f"  Total memory files: {total_files}")
    console.print(f"  Files with wikilinks: {files_with_links}")
    console.print(f"  Total wikilinks: {total_links}")
    if files_with_links > 0:
        console.print(f"  Average links per file: {total_links / files_with_links:.1f}")
    console.print("")
    console.print("[dim]Open your vault in Obsidian to see the interactive graph![/dim]")

# ── vault commands ────────────────────────────────────────────────────────────

vault_app = typer.Typer(help="Obsidian vault sync operations.")
app.add_typer(vault_app, name="vault")


def _get_obs_vault(obs_vault_arg: str | None, lace_home: "Path") -> "Path | None":
    """Resolve Obsidian vault path from arg or config."""
    from lace.core.config import load_config
    if obs_vault_arg:
        p = Path(obs_vault_arg).expanduser().resolve()
        if not p.exists():
            console.print(f"[red]✗ Path does not exist:[/red] {p}")
            return None
        return p
    # Try reading from saved state
    from lace.vault.state import SyncState
    state = SyncState.load(lace_home)
    if state.obsidian_vault:
        p = Path(state.obsidian_vault)
        if p.exists():
            return p
        console.print(f"[red]✗ Saved Obsidian vault no longer exists:[/red] {p}")
        return None
    console.print(
        "[red]✗ No Obsidian vault configured.[/red]\n"
        "[dim]Pass --vault /path/to/obsidian  or run lace vault sync --vault ... once to save it.[/dim]"
    )
    return None


@vault_app.command("sync")
def vault_sync(
    obs_vault: Annotated[
        Optional[str],
        typer.Option("--vault", "-v", help="Path to your Obsidian vault root."),
    ] = None,
    no_reindex: Annotated[
        bool,
        typer.Option("--no-reindex", help="Skip re-embedding pulled files."),
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option("--dry-run", help="Show what would change without doing it."),
    ] = False,
) -> None:
    """Full bidirectional sync between LACE vault and Obsidian."""
    from lace.core.config import load_config
    from lace.vault.sync import full_sync, get_sync_status

    lace_home = get_lace_home()
    config = load_config(lace_home)
    lace_vault = config.vault_path(lace_home)

    obs_path = _get_obs_vault(obs_vault, lace_home)
    if obs_path is None:
        raise typer.Exit(1)

    if dry_run:
        console.print(Panel(
            f"[bold]LACE vault:[/bold]    {lace_vault}\n"
            f"[bold]Obsidian vault:[/bold] {obs_path}\n"
            f"[bold]Reindex:[/bold]        {not no_reindex}\n\n"
            "[dim]Dry run — no files will be changed.[/dim]",
            title="[bold cyan]Sync Preview[/bold cyan]",
            border_style="cyan",
        ))
        # Count what would change
        lace_files = list(lace_vault.rglob("*.md"))
        console.print(f"[dim]LACE vault has {len(lace_files)} memory files.[/dim]")
        return

    with console.status("[bold green]Syncing vaults...[/bold green]"):
        result = full_sync(
            lace_vault=lace_vault,
            obs_vault=obs_path,
            lace_home=lace_home,
            reindex=not no_reindex,
        )

    # ── Report ────────────────────────────────────────────────────────────────
    lines = []

    if result.lace_to_obs:
        lines.append(f"[green]→ Pushed to Obsidian:[/green]  {len(result.lace_to_obs)} files")
        for f in result.lace_to_obs[:5]:
            lines.append(f"    {f}")
        if len(result.lace_to_obs) > 5:
            lines.append(f"    [dim]... and {len(result.lace_to_obs) - 5} more[/dim]")

    if result.obs_to_lace:
        lines.append(f"[blue]← Pulled from Obsidian:[/blue] {len(result.obs_to_lace)} files")
        for f in result.obs_to_lace[:5]:
            lines.append(f"    {f}")
        if len(result.obs_to_lace) > 5:
            lines.append(f"    [dim]... and {len(result.obs_to_lace) - 5} more[/dim]")

    if result.reindexed:
        lines.append(f"[cyan]⟳ Re-indexed:[/cyan]          {len(result.reindexed)} memories")

    if result.errors:
        lines.append(f"[red]✗ Errors:[/red]              {len(result.errors)}")
        for e in result.errors:
            lines.append(f"    [red]{e}[/red]")

    if not result.lace_to_obs and not result.obs_to_lace and not result.errors:
        lines.append("[dim]Already in sync — no changes needed.[/dim]")

    lines.append("")
    lines.append(f"[dim]Skipped {len(result.skipped)} unchanged files[/dim]")
    lines.append(f"[dim]Obsidian vault: {obs_path}[/dim]")

    border = "red" if result.errors else "green"
    title = "[bold red]Sync Complete (with errors)[/bold red]" if result.errors else "[bold green]Sync Complete[/bold green]"

    console.print(Panel("\n".join(lines), title=title, border_style=border))


@vault_app.command("watch")
def vault_watch(
    obs_vault: Annotated[
        Optional[str],
        typer.Option("--vault", "-v", help="Path to your Obsidian vault root."),
    ] = None,
    interval: Annotated[
        float,
        typer.Option("--interval", "-i", help="Poll interval in seconds."),
    ] = 1.0,
) -> None:
    """Watch both vaults for changes and sync automatically. Press Ctrl+C to stop."""
    from lace.core.config import load_config
    from lace.vault.sync import sync_single_file
    from lace.vault.watcher import start_watcher

    lace_home = get_lace_home()
    config = load_config(lace_home)
    lace_vault = config.vault_path(lace_home)

    obs_path = _get_obs_vault(obs_vault, lace_home)
    if obs_path is None:
        raise typer.Exit(1)

    console.print(Panel(
        f"[bold]LACE vault:[/bold]    {lace_vault}\n"
        f"[bold]Obsidian vault:[/bold] {obs_path}\n"
        f"[bold]Poll interval:[/bold]  {interval}s\n\n"
        "[dim]Watching for changes... Press Ctrl+C to stop.[/dim]",
        title="[bold cyan]Vault Watcher Active[/bold cyan]",
        border_style="cyan",
    ))

    change_count = 0

    def on_change(changed_file: "Path", direction: str) -> None:
        nonlocal change_count
        change_count += 1
        arrow = "→ Obs" if direction == "lace_to_obs" else "← LACE"
        console.print(f"[dim]{arrow}[/dim] [green]{changed_file.name}[/green]", end=" ")
        try:
            result = sync_single_file(
                changed_file=changed_file,
                lace_vault=lace_vault,
                obs_vault=obs_path,
                lace_home=lace_home,
            )
            if result.errors:
                console.print(f"[red]✗ {result.errors[0]}[/red]")
            elif result.reindexed:
                console.print(f"[cyan]✓ synced + reindexed[/cyan]")
            else:
                console.print(f"[green]✓ synced[/green]")
        except Exception as e:
            console.print(f"[red]✗ {e}[/red]")

    try:
        start_watcher(
            lace_vault=lace_vault,
            obs_vault=obs_path,
            lace_home=lace_home,
            on_change=on_change,
            poll_interval=interval,
        )
    except KeyboardInterrupt:
        pass

    console.print(f"\n[dim]Watcher stopped. {change_count} change(s) synced.[/dim]")


@vault_app.command("status")
def vault_status() -> None:
    """Show current vault sync status."""
    from lace.core.config import load_config
    from lace.vault.sync import get_sync_status

    lace_home = get_lace_home()
    config = load_config(lace_home)
    lace_vault = config.vault_path(lace_home)

    status = get_sync_status(lace_home)
    lace_files = list(lace_vault.rglob("*.md"))

    # Count by scope
    global_files  = [f for f in lace_files if "global"   in f.parts]
    project_files = [f for f in lace_files if "projects" in f.parts]

    obs_line = (
        f"[green]{status['obsidian_vault']}[/green]"
        if status["configured"]
        else "[yellow]Not configured[/yellow]"
    )

    last_sync = (
        status["last_full_sync"].replace("T", " ").replace("Z", " UTC")
        if status["last_full_sync"]
        else "[yellow]Never[/yellow]"
    )

    lines = [
        f"[bold]LACE vault:[/bold]      {lace_vault}",
        f"[bold]Obsidian vault:[/bold]  {obs_line}",
        f"[bold]Last full sync:[/bold]  {last_sync}",
        "",
        "[bold]LACE vault contents:[/bold]",
        f"  Total .md files:   {len(lace_files)}",
        f"  Global memories:   {len(global_files)}",
        f"  Project memories:  {len(project_files)}",
        "",
        "[bold]Sync state:[/bold]",
        f"  LACE files tracked:    {status['lace_files_tracked']}",
        f"  Obsidian files tracked: {status['obs_files_tracked']}",
    ]

    if not status["configured"]:
        lines += [
            "",
            "[dim]Run: lace vault sync --vault /path/to/obsidian[/dim]",
        ]

    console.print(Panel(
        "\n".join(lines),
        title="[bold cyan]Vault Sync Status[/bold cyan]",
        border_style="cyan",
    ))


# ── graph commands ────────────────────────────────────────────────────────────

graph_app = typer.Typer(help="Knowledge graph operations.")
app.add_typer(graph_app, name="graph")


@graph_app.command("build")
def graph_build() -> None:
    """Build the knowledge graph from the vault."""
    from lace.core.engine import GraphManager
    from lace.graph.graph import get_graph_stats

    lace_home = get_lace_home()
    manager = GraphManager(lace_home=lace_home)

    with console.status("[bold green]Building knowledge graph...[/bold green]"):
        G = manager.rebuild()

    stats = get_graph_stats(G)
    console.print(Panel(
        f"[bold]Nodes:[/bold]   {stats['total_nodes']}\n"
        f"  Memory nodes:  {stats['memory_nodes']}\n"
        f"  Concept nodes: {stats['concept_nodes']}\n\n"
        f"[bold]Edges:[/bold]   {stats['total_edges']}\n"
        + "\n".join(
            f"  {rel}: {count}"
            for rel, count in stats["edge_types"].items()
        ),
        title="[bold cyan]Knowledge Graph Built[/bold cyan]",
        border_style="cyan",
    ))


@graph_app.command("stats")
def graph_stats() -> None:
    """Show knowledge graph statistics."""
    from lace.core.engine import GraphManager
    from lace.graph.graph import get_graph_stats

    lace_home = get_lace_home()
    manager = GraphManager(lace_home=lace_home)
    G = manager.get_graph()
    stats = get_graph_stats(G)

    if stats["is_empty"]:
        console.print("[yellow]Graph is empty. Run:[/yellow] lace graph build")
        return

    console.print(Panel(
        f"[bold]Total nodes:[/bold]   {stats['total_nodes']}\n"
        f"  Memory nodes:  {stats['memory_nodes']}\n"
        f"  Concept nodes: {stats['concept_nodes']}\n\n"
        f"[bold]Total edges:[/bold]   {stats['total_edges']}\n"
        + "\n".join(
            f"  {rel}: {count}"
            for rel, count in stats["edge_types"].items()
        ),
        title="[bold cyan]Knowledge Graph[/bold cyan]",
        border_style="cyan",
    ))


@graph_app.command("related")
def graph_related(
    concept: Annotated[str, typer.Argument(help="Concept or tag to find related memories for.")],
    depth: Annotated[
        int,
        typer.Option("--depth", "-d", help="Hop depth for traversal."),
    ] = 2,
    memories_only: Annotated[
        bool,
        typer.Option("--memories", help="Show only memory nodes."),
    ] = False,
) -> None:
    """Find concepts and memories related to a given concept."""
    from lace.core.engine import GraphManager
    from lace.graph.traversal import get_neighbors, find_memories_near_concept

    lace_home = get_lace_home()
    manager = GraphManager(lace_home=lace_home)
    G = manager.get_graph()

    if G.number_of_nodes() == 0:
        console.print("[yellow]Graph is empty. Run:[/yellow] lace graph build")
        return

    if memories_only:
        results = find_memories_near_concept(G, concept, depth=depth)
    else:
        # Find the concept node
        concept_normalized = concept.lower().replace(" ", "-")
        results = get_neighbors(G, concept_normalized, depth=depth)

    if not results:
        console.print(f"[yellow]No related nodes found for:[/yellow] {concept}")
        console.print(
            "[dim]Try: lace graph build  — then add [[wikilinks]] to your memories[/dim]"
        )
        return

    table = Table(
        title=f"Related to '{concept}' (depth={depth})",
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("Hop",      width=4)
    table.add_column("Type",     width=8)
    table.add_column("ID/Name",  width=20)
    table.add_column("Relation", width=14)
    table.add_column("Label")

    for node in results:
        node_type = node["type"]
        color = "green" if node_type == "memory" else "cyan"
        table.add_row(
            str(node["distance"]),
            f"[{color}]{node_type}[/{color}]",
            node["id"][:20],
            node.get("relation", "—"),
            node.get("label", "")[:50],
        )

    console.print(table)


@graph_app.command("show")
def graph_show(
    memory_id: Annotated[str, typer.Argument(help="Memory ID to show connections for.")],
    depth: Annotated[
        int,
        typer.Option("--depth", "-d", help="Hop depth."),
    ] = 1,
) -> None:
    """Show all graph connections for a specific memory."""
    from lace.core.engine import GraphManager
    from lace.graph.traversal import get_neighbors

    lace_home = get_lace_home()
    manager = GraphManager(lace_home=lace_home)
    G = manager.get_graph()

    if memory_id not in G:
        console.print(f"[yellow]Memory {memory_id} not in graph. Run:[/yellow] lace graph build")
        return

    neighbors = get_neighbors(G, memory_id, depth=depth)
    memory = _get_store().get(memory_id)

    console.print(Panel(
        f"[bold]{memory.display_summary() if memory else memory_id}[/bold]\n\n"
        + "\n".join(
            f"  {'→' if n['distance'] == 1 else '⇒'} [{n['type']}] {n['id'][:30]} "
            f"({n.get('relation', '?')})"
            for n in neighbors
        ),
        title=f"[bold cyan]Graph connections: {memory_id}[/bold cyan]",
        border_style="cyan",
    ))


# ── project commands ───────────────────────────────────────────────────────────

@project_app.command("create")
def project_create(
    name: Annotated[str, typer.Argument(help="Project name.")],
    description: Annotated[
        Optional[str],
        typer.Option("--description", "-d", help="Project description."),
    ] = None,
) -> None:
    """Create a new project."""
    lace_home = get_lace_home()
    created = create_project(name, description, lace_home)

    if created:
        console.print(
            Panel(
                f"[bold green]✓ Project created[/bold green]\n\n"
                f"Name:        [bold]{name}[/bold]\n"
                f"Scope:       project:{name}\n"
                f"Description: {description or '[dim]none[/dim]'}\n\n"
                f"[dim]Add project-specific memories with:[/dim]\n"
                f"  lace memory add \"...\" --scope=project:{name}",
                title="[bold]Project Created[/bold]",
                border_style="green",
            )
        )
    else:
        console.print(f"[yellow]Project [bold]{name}[/bold] already exists.[/yellow]")


@project_app.command("list")
def project_list() -> None:
    """List all configured projects."""
    projects = get_projects()

    if not projects:
        console.print("[yellow]No projects found.[/yellow]")
        console.print("[dim]Try: lace project create \"my-api\"[/dim]")
        return

    table = Table(
        title=f"Projects ({len(projects)} total)",
        show_header=True,
        header_style="bold cyan",
        expand=True,
    )
    table.add_column("Name", style="bold")
    table.add_column("Scope", style="dim")
    table.add_column("Description", width=40)
    table.add_column("Last Used", width=20)

    for project in sorted(projects, key=lambda p: p.get("last_used") or "", reverse=True):
        last_used = project.get("last_used")
        if last_used:
            last_used = last_used.split("T")[0]  # Just date, not time
        else:
            last_used = "[dim]never[/dim]"

        table.add_row(
            project["name"],
            project["scope"],
            project["description"] or "[dim]none[/dim]",
            last_used,
        )

    console.print(table)


@project_app.command("switch")
def project_switch(
    name: Annotated[str, typer.Argument(help="Project name to switch to.")],
) -> None:
    """Switch to a project as active scope."""
    lace_home = get_lace_home()
    projects = get_projects()
    project_names = {p["name"] for p in projects}

    if name not in project_names:
        console.print(f"[red]✗ Project not found:[/red] {name}")
        raise typer.Exit(1)

    set_project_last_used(name, lace_home)
    console.print(f"[green]✓[/green] Switched to project: [bold]{name}[/bold]")


@project_app.command("info")
def project_info() -> None:
    """Show current active project info."""
    active_scope = get_active_scope()
    if active_scope == "global":
        console.print("[bold]Active scope:[/bold] global")
        return

    if active_scope.startswith("session:"):
        session_id = active_scope.removeprefix("session:")
        console.print(f"[bold]Active scope:[/bold] session:{session_id}")
        return

    if active_scope.startswith("project:"):
        project_name = active_scope.removeprefix("project:")
        lace_home = get_lace_home()
        projects = get_projects()
        project = next((p for p in projects if p["name"] == project_name), None)

        if project:
            console.print(
                Panel(
                    f"[bold]Project:[/bold] {project['name']}\n"
                    f"[bold]Scope:[/bold] {project['scope']}\n"
                    f"[bold]Description:[/bold] {project['description'] or '[dim]none[/dim]'}\n"
                    f"[bold]Created:[/bold] {project.get('created_at', '[dim]unknown[/dim]').split('T')[0]}\n"
                    f"[bold]Last Used:[/bold] {project.get('last_used', '[dim]never[/dim]').split('T')[0]}",
                    title="[bold]Active Project[/bold]",
                    border_style="cyan",
                )
            )
        else:
            console.print(f"[bold]Active scope:[/bold] {active_scope}")
        return


@project_app.command("detect")
def project_detect() -> None:
    """Auto-detect current project from working directory."""
    detected = detect_current_project()
    if detected:
        console.print(f"[green]✓ Detected project:[/green] [bold]{detected}[/bold]")
    else:
        console.print("[yellow]No project detected in current directory.[/yellow]")


# ── mcp placeholder ───────────────────────────────────────────────────────────

@mcp_app.command("start")
def mcp_start(
    debug: Annotated[
        bool,
        typer.Option("--debug", help="Enable debug logging."),
    ] = False,
) -> None:
    """Start the LACE MCP server (stdio mode for Antigravity/Cursor)."""
    import asyncio
    import warnings
    import os

    # Suppress all warnings in MCP mode — they corrupt stdio JSON-RPC
    warnings.filterwarnings("ignore")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    from lace.mcp.server import run_server
    asyncio.run(run_server(debug=debug))



# ── lace ask ──────────────────────────────────────────────────────────────────

@app.command()
def ask(
    query: Annotated[str, typer.Argument(help="Your question.")],
    show_context: Annotated[
        bool,
        typer.Option("--show-context", help="Show retrieved memories before response."),
    ] = False,
    no_memory: Annotated[
        bool,
        typer.Option("--no-memory", help="Skip memory retrieval entirely."),
    ] = False,
    scope: Annotated[
        Optional[str],
        typer.Option("--scope", "-s", help="Override active scope."),
    ] = None,
    max_memories: Annotated[
        int,
        typer.Option("--max-memories", "-m", help="Max memories to inject."),
    ] = 5,
    provider: Annotated[
        Optional[str],
        typer.Option("--provider", "-p", help="Override provider: ollama, openai, anthropic."),
    ] = None,
) -> None:
    """Ask a question with your memory injected automatically."""
    import time
    import warnings
    warnings.filterwarnings("ignore")

    from lace.core.config import load_config, get_lace_home
    from lace.utils.ask import ask as ask_engine

    lace_home = get_lace_home()
    config = load_config(lace_home)

    # Override provider if specified
    if provider:
        config.provider.default = provider

    start_time = time.time()

    # Run the ask engine
    try:
        memories, stream, llm_provider = ask_engine(
            query=query,
            use_memory=not no_memory,
            scope=scope,
            max_memories=max_memories,
            lace_home=lace_home,
            config=config,
        )
    except ValueError as e:
        console.print(f"[red]✗ Configuration error:[/red] {e}")
        raise typer.Exit(1)

    # Show context panel if requested
    if show_context:
        if memories:
            memory_lines = []
            for i, result in enumerate(memories, 1):
                m = result.memory
                memory_lines.append(
                    f"  [{i}] [bold]{m.display_summary()[:60]}[/bold]\n"
                    f"      scope: {m.project_scope} | "
                    f"conf: {m.confidence:.2f} | "
                    f"score: {result.relevance_score:.3f}"
                )

            retrieval_time = int((time.time() - start_time) * 1000)
            console.print(
                Panel(
                    "\n".join(memory_lines) +
                    f"\n\n[dim]Retrieved {len(memories)} memories in {retrieval_time}ms[/dim]",
                    title="[bold cyan]Context Retrieved[/bold cyan]",
                    border_style="cyan",
                )
            )
        else:
            if no_memory:
                console.print(Panel(
                    "[dim]Memory retrieval disabled (--no-memory)[/dim]",
                    title="[bold cyan]Context[/bold cyan]",
                    border_style="dim",
                ))
            else:
                console.print(Panel(
                    "[dim]No relevant memories found for this query.[/dim]",
                    title="[bold cyan]Context Retrieved[/bold cyan]",
                    border_style="dim",
                ))

    # Stream the response
    console.print()
    response_chunks: list[str] = []

    try:
        for chunk in stream:
            print(chunk, end="", flush=True)
            response_chunks.append(chunk)
    except KeyboardInterrupt:
        console.print("\n[dim]Interrupted.[/dim]")
        return

    # Footer
    total_time = int((time.time() - start_time) * 1000)
    full_response = "".join(response_chunks)
    token_estimate = len(full_response) // 4

    console.print(f"\n\n[dim]Provider: {llm_provider.provider_name} | "
                  f"Model: {llm_provider.model_name} | "
                  f"~{token_estimate} tokens | "
                  f"{total_time}ms total[/dim]")


# ── generate-context ──────────────────────────────────────────────────────────

@app.command("generate-context")
def generate_context(
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Print output to terminal without writing any file.",
    ),
    scope: str = typer.Option(
        "auto",
        "--scope", "-s",
        help="Project scope to generate context for. "
             "Defaults to auto-detecting from git.",
    ),
):
    """
    Synthesize vault memories into a project context file.
    Outputs AGENTS.md, CLAUDE.md, or LACE.context.md 
    at the project root.
    """
    from lace.core.scope import detect_current_project
    from lace.memory.store import MemoryStore

    # Resolve project scope
    if scope == "auto":
        resolved_scope = detect_current_project()
    else:
        resolved_scope = scope

    if not resolved_scope or resolved_scope == "global":
        typer.echo(
            "✗ No active project detected.\n"
            "  Run this command from inside a git repository,\n"
            "  or pass --scope project:yourproject"
        )
        raise typer.Exit(1)

    project_name = resolved_scope.replace("project:", "")

    # Load memories
    store = MemoryStore()

    # Initialize multi-signal retrieval indices.
    try:
        store.initialize()
    except Exception as e:
        import logging
        logging.getLogger("lace.main").warning(
            f"MemoryStore.initialize() failed, falling back to classic search: {e}"
        )

    typer.echo(f"Loading memories for project: {project_name}")

    grouped = load_memories_for_generation(
        project_scope=resolved_scope,
        store=store,
    )

    total = sum(len(v) for v in grouped.values())

    if total == 0:
        typer.echo(
            f"✗ No memories found for {project_name}.\n"
            "  Have some AI conversations first, then run:\n"
            "  lace memory review"
        )
        raise typer.Exit(1)

    typer.echo(f"Found {total} memories. Synthesizing...")

    # Call LLM synthesis
    try:
        content = synthesize_context(grouped, project_name)
    except Exception as e:
        typer.echo(f"✗ Synthesis failed: {e}")
        raise typer.Exit(1)

    # Dry run — print and exit
    if dry_run:
        typer.echo("\n" + "─" * 60)
        typer.echo(content)
        typer.echo("─" * 60)
        typer.echo("\n[Dry run — no file written]")
        return

    # Ask user for format choice
    format_choice = _ask_format_choice()
    filenames = _get_filenames(format_choice)
    project_root = _get_project_root()

    # Check for existing files and warn
    existing = [
        f for f in filenames
        if (project_root / f).exists()
    ]

    if existing:
        typer.echo("\n⚠ These files already exist:")
        for f in existing:
            typer.echo(f"  {project_root / f}")
        confirm = typer.confirm(
            "Overwrite?",
            default=False,
        )
        if not confirm:
            typer.echo("Aborted.")
            raise typer.Exit(0)

    # Write files
    typer.echo("")
    for filename in filenames:
        output_path = project_root / filename
        output_path.write_text(content, encoding="utf-8")
        typer.echo(f"✓ Written: {output_path}")

    # Print commit suggestion
    typer.echo(
        "\nCommit this file to share context with your team:"
    )
    for filename in filenames:
        typer.echo(f"  git add {filename}")
    typer.echo(
        "  git commit -m 'chore: update AI context via LACE'"
    )



# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app()