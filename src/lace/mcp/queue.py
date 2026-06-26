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
    from lace.core.config import get_lace_home
    queue_dir = get_lace_home() / "queue"
    queue_dir.mkdir(parents=True, exist_ok=True)
    return queue_dir / "extraction_queue.db"


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
    error_msg     TEXT
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


def init_queue_db() -> None:
    """
    Creates the extraction_queue table and index if they don't exist.
    Safe to call multiple times (IF NOT EXISTS guards).
    Called once at MCP server startup before the worker thread starts.
    """
    db_path = get_queue_db_path()
    
    # Use check_same_thread=False because the worker thread and main thread
    # both need access. We manage our own connection-per-operation discipline
    # below to avoid actual concurrency issues.
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    try:
        conn.execute(_CREATE_TABLE_SQL)
        conn.execute(_CREATE_INDEX_SQL)
        conn.commit()
        logger.debug(f"Queue DB initialized at {db_path}")
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def _get_connection() -> sqlite3.Connection:
    """
    Opens a fresh connection to the queue DB.
    
    We open/close per operation rather than keeping a persistent connection
    because:
    1. SQLite handles concurrent readers fine with short-lived connections
    2. No risk of a stale connection state causing silent data corruption
    3. The worker poll interval is 30 seconds — connection overhead is irrelevant
    """
    db_path = get_queue_db_path()
    conn = sqlite3.connect(
        str(db_path),
        check_same_thread=False,
        timeout=10.0,  # Wait up to 10 seconds for locks to clear
    )
    
    # Always ensure tables and indexes exist. This guarantees LACE recovers
    # gracefully if the database file is deleted/reset while the server is running.
    try:
        conn.execute(_CREATE_TABLE_SQL)
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
) -> str:
    """
    Inserts a new extraction job with status=pending.
    
    Returns the job ID immediately. This function must complete in
    < 5ms — it only does a single SQLite INSERT, no LLM calls,
    no embeddings, no file I/O beyond the DB write.
    
    Args:
        query:    The user's original message
        response: The agent's complete response
        scope:    Resolved scope string (e.g. "global", "project:lace")
        history:  Last N conversation turns for context
    
    Returns:
        job_id: UUID string identifying this job
    """
    job_id = str(uuid4())
    created_at = datetime.now(timezone.utc).isoformat()
    history_json = json.dumps(history, ensure_ascii=False)
    
    conn = _get_connection()
    try:
        conn.execute(
            """
            INSERT INTO extraction_queue
                (id, query, response, scope, history_json, status, created_at)
            VALUES
                (?, ?, ?, ?, ?, 'pending', ?)
            """,
            (job_id, query, response, scope, history_json, created_at),
        )
        conn.commit()
        logger.debug(f"Enqueued job {job_id} (scope={scope})")
        return job_id
    except Exception as e:
        logger.error(f"Failed to enqueue job: {e}")
        # Re-raise because if we can't write to the queue, the caller
        # should know — but this should never fail in practice
        raise
    finally:
        conn.close()


def mark_processing(job_id: str) -> None:
    """Transitions a job from pending → processing."""
    _update_status(job_id, "processing")


def mark_done(job_id: str) -> None:
    """Transitions a job to done and records completion time."""
    processed_at = datetime.now(timezone.utc).isoformat()
    conn = _get_connection()
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


def mark_failed(job_id: str, error: str) -> None:
    """
    Transitions a job to failed and records the error message.
    
    Note: This does NOT prevent retry — increment_retry() handles that.
    A job is only permanently failed when retry_count > MAX_RETRIES
    and we explicitly call mark_failed at that point.
    """
    processed_at = datetime.now(timezone.utc).isoformat()
    conn = _get_connection()
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


def increment_retry(job_id: str) -> None:
    """
    Increments retry count and resets status to pending so the
    worker picks it up again on the next poll cycle.
    """
    conn = _get_connection()
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


def _update_status(job_id: str, status: str) -> None:
    """Internal helper for simple status transitions."""
    conn = _get_connection()
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

def get_pending_jobs(limit: int = _WORKER_BATCH_SIZE) -> list[dict]:
    """
    Returns up to `limit` pending jobs, oldest first.
    
    We order by created_at ASC to process jobs in arrival order.
    This prevents newer jobs from jumping the queue if old jobs
    are repeatedly failing and being retried.
    """
    conn = _get_connection()
    try:
        cursor = conn.execute(
            """
            SELECT id, query, response, scope, history_json,
                   status, created_at, processed_at,
                   retry_count, error_msg
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


def get_job_status(job_id: str) -> dict | None:
    """
    Returns the full job record for a given ID.
    Used in tests and debugging — not called in the hot path.
    """
    conn = _get_connection()
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
    
    All exceptions are caught by the caller — this function may raise
    and the worker loop handles retry/fail logic.
    """
    # Import here to avoid circular imports at module load time.
    # These imports happen inside the worker thread, which is fine.
    from lace.memory.extractor import (
        extract_from_conversation,
        should_attempt_extraction,
    )
    from lace.memory.inbox import save_to_inbox
    
    job_id = job["id"]
    history = json.loads(job.get("history_json", "[]"))
    context = _build_context_from_history(history)
    
    # Fast pre-filter: cheap, no LLM call.
    # If this returns False, the turn isn't worth extracting from.
    if not should_attempt_extraction(job["query"], job["response"]):
        logger.debug(f"Job {job_id}: pre-filter rejected, marking done")
        mark_done(job_id)
        return
    
    # Call the existing LLM extractor.
    # This is the slow part (5-30 seconds depending on hardware).
    result = extract_from_conversation(
        query=job["query"],
        response=job["response"],
        context=context,
        scope=job["scope"],
    )
    
    if hasattr(result, "candidates"):
        candidates = result.candidates
    else:
        candidates = result
    
    if not candidates:
        logger.debug(f"Job {job_id}: extractor returned no candidates")
        mark_done(job_id)
        return
    
    # Write each candidate to inbox — never directly to vault.
    # Inbox items are unverified drafts awaiting user review.
    saved_count = 0
    for candidate in candidates:
        try:
            draft_id = save_to_inbox(candidate, scope=job["scope"])
            logger.info(f"Job {job_id}: saved draft {draft_id}")
            saved_count += 1
        except Exception as e:
            # Don't fail the whole job if one candidate fails to save
            logger.warning(f"Job {job_id}: failed to save candidate: {e}")
    
    logger.info(f"Job {job_id}: completed, {saved_count} drafts saved")
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
        pass  # Non-critical — don't let this kill the worker
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
