"""
SQLite-backed async extraction queue for LACE Phase 1.

Design principles:
- enqueue() must NEVER block — it only writes to SQLite
- Worker thread must NEVER crash — double try/except everywhere
- Worker writes to inbox only — never directly to vault
- Daemon thread dies cleanly with the MCP server process

The queue decouples process_interaction (fast path, < 100ms) from
actual LLM extraction work (slow path, 5-30 seconds).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from uuid import uuid4

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def get_queue_db_path() -> Path:
    """
    Returns ~/.lace/queue/extraction_queue.db (or LACE_HOME equivalent).
    Creates the directory if it does not exist.
    
    We keep the queue database separate from the main memory store
    so a corrupted queue never touches the vault.
    """
    from lace.core.config import resolve_lace_paths
    queue_db = resolve_lace_paths()["queue_db"]
    queue_db.parent.mkdir(parents=True, exist_ok=True)
    return queue_db


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS extraction_queue (
    id            TEXT PRIMARY KEY,
    query         TEXT NOT NULL,
    response      TEXT NOT NULL,
    scope         TEXT NOT NULL,
    history_json  TEXT DEFAULT '[]',
    status        TEXT DEFAULT 'pending',
    created_at    TEXT NOT NULL,
    processed_at  TEXT,
    retry_count   INTEGER DEFAULT 0,
    error_msg     TEXT,
    canonical_hash TEXT,
    repeat_count   INTEGER DEFAULT 0
);
"""

# Index for the hot path: worker polling for pending jobs ordered by age.
# Without this index, every poll is a full table scan.
_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_status_created
    ON extraction_queue (status, created_at ASC);
"""

_MAX_RETRIES = 3
_WORKER_POLL_INTERVAL_SECONDS = 30
_WORKER_BATCH_SIZE = 5


def init_queue_db(db_path: Optional[Path] = None) -> None:
    """
    Creates the extraction_queue table and index if they don't exist.
    Safe to call multiple times (IF NOT EXISTS guards).
    Called once at MCP server startup before the worker thread starts.
    """
    if db_path is None:
        db_path = get_queue_db_path()
    
    # Use check_same_thread=False because the worker thread and main thread
    # both need access. We manage our own connection-per-operation discipline
    # below to avoid actual concurrency issues.
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    try:
        conn.execute(_CREATE_TABLE_SQL)
        # Migrate old schema if canonical_hash column is missing
        try:
            conn.execute("ALTER TABLE extraction_queue ADD COLUMN canonical_hash TEXT")
            conn.execute("ALTER TABLE extraction_queue ADD COLUMN repeat_count INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass # Columns already exist
        conn.execute(_CREATE_INDEX_SQL)
        # Recover stuck jobs from previous crashed runs
        conn.execute(
            "UPDATE extraction_queue SET status = 'pending' WHERE status = 'processing'"
        )
        conn.commit()
        logger.debug(f"Queue DB initialized at {db_path}")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def _get_connection(db_path: Optional[Path] = None) -> sqlite3.Connection:
    """
    Opens a fresh connection to the queue DB.
    
    We open/close per operation rather than keeping a persistent connection
    because:
    1. SQLite handles concurrent readers fine with short-lived connections
    2. No risk of a stale connection state causing silent data corruption
    3. The worker poll interval is 30 seconds — connection overhead is irrelevant
    """
    if db_path is None:
        db_path = get_queue_db_path()
    conn = sqlite3.connect(
        str(db_path),
        check_same_thread=False,
        timeout=10.0,  # Wait up to 10 seconds for locks to clear
    )
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    except Exception:
        pass

    
    # Always ensure tables and indexes exist. This guarantees LACE recovers
    # gracefully if the database file is deleted/reset while the server is running.
    try:
        conn.execute(_CREATE_TABLE_SQL)
        # Migrate old schema if canonical_hash column is missing
        try:
            conn.execute("ALTER TABLE extraction_queue ADD COLUMN canonical_hash TEXT")
            conn.execute("ALTER TABLE extraction_queue ADD COLUMN repeat_count INTEGER DEFAULT 0")
        except sqlite3.OperationalError:
            pass # Columns already exist
        conn.execute(_CREATE_INDEX_SQL)
        conn.commit()
    except Exception as e:
        logger.error(f"Failed to auto-initialize queue DB tables: {e}")
        
    conn.row_factory = sqlite3.Row  # Allow dict-style access: row['id']
    return conn


# ---------------------------------------------------------------------------
# Write operations
# ---------------------------------------------------------------------------

def enqueue(
    query: str,
    response: str,
    scope: str,
    history: list[dict],
    config = None,
    log_db_path = None,
    db_path: Optional[Path] = None,
) -> str:
    """
    Inserts a new extraction job with status=pending, implementing canonical hash suppression.
    """
    from lace.memory.normalize import canonical_hash
    from lace.core.config import load_config, get_lace_home
    from lace.memory.pipeline_log import log_queue_suppressed, PIPELINE_LOG_DB_PATH
    
    if config is None:
        try:
            config = load_config(get_lace_home())
        except Exception:
            logger.warning(
                "enqueue(): could not load LACE config, using defaults.",
                exc_info=True,
            )
            from lace.core.config import LaceConfig
            config = LaceConfig()
    if hasattr(config, "dedup"):
        cooldown = config.dedup.hash_cooldown_seconds
    else:
        cooldown = getattr(config, "hash_cooldown_seconds", 300)
    h = canonical_hash(f"{query}\n{response}")
    
    conn = _get_connection(db_path)
    try:
        now = datetime.now(timezone.utc)
        
        cursor = conn.execute(
            """
            SELECT id, repeat_count, created_at FROM extraction_queue
            WHERE canonical_hash = ? AND status != 'failed' AND scope = ?
            ORDER BY created_at DESC LIMIT 1
            """,
            (h, scope),
        )
        existing = cursor.fetchone()
        
        if existing:
            try:
                created_dt = datetime.fromisoformat(existing["created_at"])
                age = (now - created_dt).total_seconds()
            except Exception:
                logger.debug(
                    f"[Queue] Could not parse created_at for job {existing['id']}, "
                    "treating as within cooldown window.",
                    exc_info=True,
                )
                age = 999999.0
                
            if age <= cooldown:
                new_repeat_count = (existing["repeat_count"] or 0) + 1
                conn.execute(
                    """
                    UPDATE extraction_queue
                    SET repeat_count = ?
                    WHERE id = ?
                    """,
                    (new_repeat_count, existing["id"]),
                )
                conn.commit()
                
                # Log queue suppressed event
                actual_log_path = log_db_path if log_db_path is not None else PIPELINE_LOG_DB_PATH
                log_queue_suppressed(
                    canonical_hash_value=h,
                    queue_id=existing["id"],
                    repeat_count=new_repeat_count,
                    db_path=actual_log_path,
                )
                
                logger.debug(f"[Queue] SUPPRESSED | hash={h[:12]}... | repeat_count={new_repeat_count}")
                return existing["id"]

        # Insert new pending job
        job_id = str(uuid4())
        created_at = now.isoformat()
        history_json = json.dumps(history, ensure_ascii=False)
        conn.execute(
            """
            INSERT INTO extraction_queue
                (id, query, response, scope, history_json, status, created_at, canonical_hash, repeat_count)
            VALUES
                (?, ?, ?, ?, ?, 'pending', ?, ?, 0)
            """,
            (job_id, query, response, scope, history_json, created_at, h),
        )
        conn.commit()
        logger.debug(f"Enqueued job {job_id} (scope={scope})")
        return job_id
    except Exception as e:
        logger.error(f"Failed to enqueue job: {e}")
        raise
    finally:
        conn.close()


def enqueue_interaction(
    query: str,
    response: str,
    config = None,
    db_path: Optional[Path] = None,
    log_db_path: Optional[Path] = None,
    scope: Optional[str] = None,
    history: Optional[list[dict]] = None,
) -> dict:
    """
    Compatibility wrapper matching the enqueue_interaction interface.

    Scope resolution (in priority order):
      1. ``scope`` argument — explicit, always wins.
      2. ``_mcp_context_project`` — set by initialize_lace_session() in MCP context.
      3. Falls back to ``"global"``.

    Calling this outside an active MCP session should pass ``scope`` explicitly
    to avoid silent fallback to global.
    """
    from lace.mcp.tools import _mcp_context_project
    if scope is None:
        scope = _mcp_context_project or "global"
    if not scope.startswith("project:") and scope != "global":
        scope = f"project:{scope}"

    from lace.memory.pipeline_log import PIPELINE_LOG_DB_PATH
    actual_log_path = log_db_path if log_db_path is not None else PIPELINE_LOG_DB_PATH

    enqueue_kwargs = {
        "query": query,
        "response": response,
        "scope": scope,
        "history": history or [],
    }
    if config is not None:
        enqueue_kwargs["config"] = config
    if log_db_path is not None:
        enqueue_kwargs["log_db_path"] = actual_log_path
    if db_path is not None:
        enqueue_kwargs["db_path"] = db_path
    job_id = enqueue(**enqueue_kwargs)

    # Determine action (inserted vs suppressed) based on repeat_count in queue DB.
    # Failures here are returned as a structured error — this is the pipeline entry
    # point; a silent failure means no queue row, no log row, nothing to investigate.
    action = "inserted"
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT repeat_count FROM extraction_queue WHERE id = ?",
            (job_id,)
        )
        row = cursor.fetchone()
        if row and row["repeat_count"] > 0:
            action = "suppressed"
    except Exception as e:
        logger.error(
            f"enqueue_interaction: could not read repeat_count for job {job_id}: {e}",
            exc_info=True,
        )
        return {
            "status": "error",
            "error": f"Job enqueued ({job_id}) but repeat_count check failed: {e}",
            "job_id": job_id,
            "queue_id": job_id,
            "scope_used": scope,
            "action": "inserted",  # safe assumption — job IS in the queue
        }
    finally:
        conn.close()

    return {
        "status": "queued",
        "job_id": job_id,
        "queue_id": job_id,
        "scope_used": scope,
        "action": action,
    }


def mark_processing(job_id: str, db_path: Optional[Path] = None) -> None:
    """Transitions a job from pending → processing."""
    _update_status(job_id, "processing", db_path)


def mark_done(job_id: str, db_path: Optional[Path] = None) -> None:
    """Transitions a job to done and records completion time."""
    processed_at = datetime.now(timezone.utc).isoformat()
    conn = _get_connection(db_path)
    try:
        conn.execute(
            """
            UPDATE extraction_queue
            SET status = 'done', processed_at = ?
            WHERE id = ?
            """,
            (processed_at, job_id),
        )
        conn.commit()
    finally:
        conn.close()


def mark_failed(job_id: str, error: str, db_path: Optional[Path] = None) -> None:
    """
    Transitions a job to failed and records the error message.
    
    Note: This does NOT prevent retry — increment_retry() handles that.
    A job is only permanently failed when retry_count > MAX_RETRIES
    and we explicitly call mark_failed at that point.
    """
    processed_at = datetime.now(timezone.utc).isoformat()
    conn = _get_connection(db_path)
    try:
        conn.execute(
            """
            UPDATE extraction_queue
            SET status = 'failed', processed_at = ?, error_msg = ?
            WHERE id = ?
            """,
            (processed_at, error[:2000], job_id),  # Cap error length
        )
        conn.commit()
    finally:
        conn.close()


def increment_retry(job_id: str, db_path: Optional[Path] = None) -> None:
    """
    Increments retry count and resets status to pending so the
    worker picks it up again on the next poll cycle.
    """
    conn = _get_connection(db_path)
    try:
        conn.execute(
            """
            UPDATE extraction_queue
            SET retry_count = retry_count + 1,
                status = 'pending',
                error_msg = NULL
            WHERE id = ?
            """,
            (job_id,),
        )
        conn.commit()
    finally:
        conn.close()


def _update_status(job_id: str, status: str, db_path: Optional[Path] = None) -> None:
    """Internal helper for simple status transitions."""
    conn = _get_connection(db_path)
    try:
        conn.execute(
            "UPDATE extraction_queue SET status = ? WHERE id = ?",
            (status, job_id),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Read operations
# ---------------------------------------------------------------------------

def get_pending_jobs(limit: int = _WORKER_BATCH_SIZE, db_path: Optional[Path] = None) -> list[dict]:
    """
    Returns up to `limit` pending jobs, oldest first.
    
    We order by created_at ASC to process jobs in arrival order.
    This prevents newer jobs from jumping the queue if old jobs
    are repeatedly failing and being retried.
    """
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            """
            SELECT id, query, response, scope, history_json,
                   status, created_at, processed_at,
                   retry_count, error_msg, canonical_hash, repeat_count
            FROM extraction_queue
            WHERE status = 'pending'
            ORDER BY created_at ASC
            LIMIT ?
            """,
            (limit,),
        )
        # Convert sqlite3.Row objects to plain dicts for easier handling
        return [dict(row) for row in cursor.fetchall()]
    finally:
        conn.close()


def get_job_status(job_id: str, db_path: Optional[Path] = None) -> dict | None:
    """
    Returns the full job record for a given ID.
    Used in tests and debugging — not called in the hot path.
    """
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            "SELECT * FROM extraction_queue WHERE id = ?",
            (job_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Worker thread
# ---------------------------------------------------------------------------

def _build_context_from_history(history: list[dict]) -> str:
    """
    Converts the last N conversation turns into a plain text context string
    that the extractor can use to understand multi-turn decisions.
    
    Format:
        User: <query>
        Assistant: <response>
        [repeated for each turn]
    
    We keep this simple — the extractor prompt already handles the heavy
    lifting of interpretation.
    """
    if not history:
        return ""
    
    lines: list[str] = []
    for turn in history:
        query = turn.get("query", "").strip()
        response = turn.get("response", "").strip()
        if query:
            lines.append(f"User: {query}")
        if response:
            # Truncate very long responses in context to avoid token bloat
            if len(response) > 500:
                response = response[:500] + "...[truncated for context]"
            lines.append(f"Assistant: {response}")
    
    return "\n".join(lines)


def _process_single_job(job: dict) -> None:
    """
    Processes one extraction job. Called by the worker loop.

    Pipeline:
      1. should_attempt_extraction() — cheap pre-filter, no LLM call
      2. process_queue_item()        — worthiness gate + pipeline_log verdict
      3. dedup_and_store()           — two-tier dedup, ChromaDB + vault write

    All exceptions are caught by the caller — this function may raise
    and the worker loop handles retry/fail logic.
    """
    from lace.memory.extractor import (
        process_queue_item,
        should_attempt_extraction,
    )
    from lace.memory.dedup import (
        StoreBackedVectorIndex,
        dedup_and_store,
    )
    from lace.memory.store import MemoryStore
    from lace.memory.pipeline_log import PIPELINE_LOG_DB_PATH
    from lace.core.config import get_lace_home, load_config, resolve_lace_paths

    job_id    = job["id"]
    job_scope = job.get("scope", "global")

    # Step 1 — Fast pre-filter: cheap, no LLM call.
    if not should_attempt_extraction(job["query"], job["response"]):
        logger.debug(f"Job {job_id}: pre-filter rejected, marking done")
        mark_done(job_id)
        return

    # Step 2 — Load config and build MemoryStore so dedup_and_store can persist.
    try:
        lace_home = get_lace_home()
        paths     = resolve_lace_paths(lace_home)
        config    = load_config(lace_home)
        store     = MemoryStore(lace_home=lace_home, config=config)
        store.initialize()
    except Exception as e:
        logger.error(
            f"Job {job_id}: could not build/initialize MemoryStore: {e}",
            exc_info=True,
        )
        raise

    # Step 3 — Worthiness-gated extraction (logs verdict to pipeline_log).
    #
    # process_queue_item() carries job_id for log correlation — this is why
    # we call it instead of extract_memories(), which uses queue_id=-1.
    log_db_path = job.get("log_db_path") or PIPELINE_LOG_DB_PATH
    memories = process_queue_item(
        item=job,
        config=config,
        log_db_path=Path(log_db_path) if isinstance(log_db_path, str) else log_db_path,
        raise_on_llm_error=True,
    )

    if not memories:
        logger.debug(f"Job {job_id}: gated out or nothing worth remembering")
        mark_done(job_id)
        return

    # Step 4 — Two-tier dedup + store each returned memory dict.
    #
    # StoreBackedVectorIndex wraps the same ChromaDB instance that MemoryStore
    # uses for upserts — guaranteed to be the same path, not a divergent copy.
    vector_index = StoreBackedVectorIndex(paths["vector_db"])

    stored_count = 0
    merged_count = 0
    skipped_count = 0

    for candidate in memories:
        # Thread scope from the job row into the candidate dict.
        # process_queue_item() does not set project_scope; the worker owns it.
        if "project_scope" not in candidate or not candidate["project_scope"]:
            candidate["project_scope"] = job_scope

        result_id = dedup_and_store(
            candidate=candidate,
            vector_index=vector_index,
            memory_store=store,
            config=config,
            queue_id=job_id,
            hash_index_db_path=paths["hash_index"],
            log_db_path=Path(log_db_path) if isinstance(log_db_path, str) else log_db_path,
        )
        if result_id:
            stored_count += 1
        else:
            skipped_count += 1

    logger.info(
        f"Job {job_id}: completed — "
        f"{stored_count} stored/merged, {skipped_count} skipped"
    )
    mark_done(job_id)






def _worker_loop() -> None:
    """
    Background extraction worker. Runs forever in a daemon thread.
    
    Design:
    - Polls every 30 seconds for pending jobs
    - Processes up to 5 jobs per cycle
    - Never crashes — all exceptions are caught at multiple levels
    - If LLM is offline, jobs fail and retry up to MAX_RETRIES times
    
    The outer try/except catches failures in the polling/DB logic itself.
    The inner try/except catches failures in individual job processing.
    The finally block ensures we always sleep between cycles, even after
    catastrophic failures.
    """
    logger.info("LACE extraction worker started")
    
    while True:
        try:
            jobs = get_pending_jobs(limit=_WORKER_BATCH_SIZE)
            
            if jobs:
                logger.debug(f"Worker found {len(jobs)} pending jobs")
            
            for job in jobs:
                job_id = job["id"]
                
                # Permanently fail jobs that have exceeded max retries.
                # These sit in the queue as 'failed' for audit purposes.
                if job["retry_count"] > _MAX_RETRIES:
                    logger.warning(
                        f"Job {job_id} exceeded max retries "
                        f"({_MAX_RETRIES}), marking permanently failed"
                    )
                    mark_failed(job_id, f"Max retries ({_MAX_RETRIES}) exceeded")
                    continue
                
                mark_processing(job_id)
                
                try:
                    _process_single_job(job)
                    
                except Exception as e:
                    # Job-level failure: increment retry, reset to pending.
                    # The job will be picked up again on the next poll cycle.
                    error_msg = str(e)
                    logger.error(
                        f"Job {job_id} failed "
                        f"(retry {job['retry_count']}/{_MAX_RETRIES}): "
                        f"{error_msg}"
                    )
                    increment_retry(job_id)
                    # Also record the error message for debugging
                    # (increment_retry resets to pending, so we patch error_msg)
                    _patch_error_msg(job_id, error_msg)
        
        except Exception as e:
            # Outer safety net: catches DB connection failures, import errors,
            # or any other catastrophic issue. The worker MUST NOT die here.
            logger.error(f"Worker loop error (outer): {e}", exc_info=True)
        
        finally:
            # Always sleep between cycles, even after a catastrophic error.
            # This prevents a tight loop hammering the DB if something is
            # fundamentally broken.
            time.sleep(_WORKER_POLL_INTERVAL_SECONDS)


def _patch_error_msg(job_id: str, error_msg: str) -> None:
    """
    Records the error message on a pending job after increment_retry().
    increment_retry resets status to pending and clears error_msg,
    so we patch it back for debugging visibility.
    """
    conn = _get_connection()
    try:
        conn.execute(
            "UPDATE extraction_queue SET error_msg = ? WHERE id = ?",
            (error_msg[:2000], job_id),
        )
        conn.commit()
    except Exception:
        logger.debug(
            "Could not patch error_msg for failed queue job %s",
            job_id,
            exc_info=True,
        )
    finally:
        conn.close()


def start_worker_thread() -> threading.Thread:
    """
    Starts the background extraction worker as a daemon thread.
    
    Daemon threads are automatically killed when the main process exits,
    so we don't need explicit shutdown logic. The MCP server dying takes
    the worker with it cleanly.
    
    Returns the thread reference. Store this in server.py if you want to
    check thread health (though with daemon=True, it's mainly for logging).
    """
    thread = threading.Thread(
        target=_worker_loop,
        name="lace-extraction-worker",
        daemon=True,  # Dies with the MCP server process
    )
    thread.start()
    logger.info(f"Worker thread started: {thread.name} (daemon={thread.daemon})")
    return thread


def process_queue_once(db_path: Optional[Path] = None) -> list[dict]:
    """
    Process pending jobs in the queue synchronously in a single cycle.
    Used for tests, CLI commands, and synchronous environments.
    """
    logger.info("Processing queue once synchronously")
    jobs = get_pending_jobs(limit=_WORKER_BATCH_SIZE, db_path=db_path)
    for job in jobs:
        job_id = job["id"]
        if job["retry_count"] > _MAX_RETRIES:
            logger.warning(
                f"Job {job_id} exceeded max retries "
                f"({_MAX_RETRIES}), marking permanently failed"
            )
            mark_failed(job_id, f"Max retries ({_MAX_RETRIES}) exceeded", db_path)
            continue

        mark_processing(job_id, db_path)
        try:
            _process_single_job(job)
        except Exception as e:
            error_msg = str(e)
            logger.error(
                f"Job {job_id} failed "
                f"(retry {job['retry_count']}/{_MAX_RETRIES}): "
                f"{error_msg}"
            )
            increment_retry(job_id, db_path)
            _patch_error_msg(job_id, error_msg)
    return jobs
