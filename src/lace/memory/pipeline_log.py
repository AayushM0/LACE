# src/lace/memory/pipeline_log.py
"""
Pipeline logging for LACE — single source of truth.

All pipeline events write to ~/.lace/queue/pipeline_log.db
via this module. No other module defines log writers or
initializes the schema.

Event types and their column usage:

  queue_suppressed:
    canonical_hash  ✓
    queue_id        ✓  (the existing row being updated)
    repeat_count    ✓  (new repeat_count after increment)

  extraction_verdict:
    canonical_hash    ✓
    queue_id          ✓
    worth_remembering ✓
    reason            ✓
    repeat_count      ✓  (used as memory_count here)

  dedup_action:
    canonical_hash  ✓
    queue_id        ✓
    memory_id       ✓
    dedup_action    ✓
    similarity_score ✓

Columns left NULL for a given event type carry no meaning
and must not be read for that event type.

Import pattern for other modules:
    from lace.memory.pipeline_log import (
        initialize_pipeline_log_db,
        log_queue_suppressed,
        log_extraction_verdict,
        log_dedup_action,
    )

Query pattern (post-run analysis):
    from lace.memory.pipeline_log import (
        query_funnel_summary,
        query_suppressed_hashes,
        query_verdict_reasons,
        query_dedup_score_distribution,
        query_full_trace,
    )
"""

import sqlite3
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path — single definition, imported by queue.py / extractor.py / dedup.py
# ---------------------------------------------------------------------------

PIPELINE_LOG_DB_PATH = Path(
    "~/.lace/queue/pipeline_log.db"
).expanduser()


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

def _get_connection(
    db_path: Path = PIPELINE_LOG_DB_PATH,
) -> sqlite3.Connection:
    """
    Open a WAL-mode connection to pipeline_log.db.
    Creates parent directory if it does not exist.
    db_path is injectable for testing.
    """
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------

def initialize_pipeline_log_db(
    db_path: Path = PIPELINE_LOG_DB_PATH,
) -> None:
    """
    Create the pipeline_log table and all indexes.

    Safe to call multiple times — all statements use IF NOT EXISTS.
    This is the ONLY place the pipeline_log schema is defined.
    extractor.py and dedup.py must NOT define their own schema.

    Called once at LACE startup, before any pipeline activity.
    """
    with _get_connection(db_path) as conn:

        # Single table for all event types
        conn.execute("""
            CREATE TABLE IF NOT EXISTS pipeline_log (
                id                INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type        TEXT    NOT NULL,
                canonical_hash    TEXT,
                queue_id          INTEGER,
                memory_id         TEXT,
                worth_remembering INTEGER,
                reason            TEXT,
                dedup_action      TEXT,
                similarity_score  REAL,
                repeat_count      INTEGER,
                created_at        TEXT    NOT NULL
            )
        """)

        # Index: filter by event type (funnel queries)
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pipeline_log_event "
            "ON pipeline_log(event_type)"
        )

        # Index: trace one hash across all stages
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pipeline_log_hash "
            "ON pipeline_log(canonical_hash)"
        )

        # Index: join back to queue items
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pipeline_log_queue_id "
            "ON pipeline_log(queue_id)"
        )

        # Index: filter dedup decisions by action type
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_pipeline_log_dedup_action "
            "ON pipeline_log(dedup_action)"
        )

        conn.commit()

    logger.debug(f"[PipelineLog] DB ready at {db_path}")


# ---------------------------------------------------------------------------
# Internal write helper
# ---------------------------------------------------------------------------

def _write_event(
    event_type: str,
    canonical_hash: Optional[str] = None,
    queue_id: Optional[int] = None,
    memory_id: Optional[str] = None,
    worth_remembering: Optional[bool] = None,
    reason: Optional[str] = None,
    dedup_action: Optional[str] = None,
    similarity_score: Optional[float] = None,
    repeat_count: Optional[int] = None,
    db_path: Path = PIPELINE_LOG_DB_PATH,
) -> None:
    """
    Internal writer — all public log functions call this.

    Wrapped in try/except so logging NEVER crashes the pipeline.
    On failure: logs a warning and returns silently.

    Parameters map directly to pipeline_log columns.
    Pass None for columns not relevant to this event type.
    """
    now = datetime.now(timezone.utc).isoformat()

    try:
        with _get_connection(db_path) as conn:
            conn.execute(
                """
                INSERT INTO pipeline_log (
                    event_type, canonical_hash, queue_id,
                    memory_id, worth_remembering, reason,
                    dedup_action, similarity_score,
                    repeat_count, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event_type,
                    canonical_hash,
                    queue_id,
                    memory_id,
                    (1 if worth_remembering else 0)
                    if worth_remembering is not None else None,
                    reason,
                    dedup_action,
                    similarity_score,
                    repeat_count,
                    now,
                )
            )
            conn.commit()

    except Exception as e:
        # Never propagate — logging must not crash the pipeline
        logger.warning(
            f"[PipelineLog] Write failed for event_type='{event_type}': {e}"
        )


# ---------------------------------------------------------------------------
# Public writer functions — one per event type
# ---------------------------------------------------------------------------

def log_queue_suppressed(
    canonical_hash_value: str,
    queue_id: int,
    repeat_count: int,
    db_path: Path = PIPELINE_LOG_DB_PATH,
) -> None:
    """
    Write a queue_suppressed event.

    Called by enqueue_interaction() when an interaction is suppressed
    within the cooldown window instead of inserting a new row.

    Parameters
    ----------
    canonical_hash_value : Hash of the suppressed interaction
    queue_id             : ID of the existing row being updated
    repeat_count         : New repeat_count after this suppression
    db_path              : Injectable for testing
    """
    _write_event(
        event_type="queue_suppressed",
        canonical_hash=canonical_hash_value,
        queue_id=queue_id,
        repeat_count=repeat_count,
        db_path=db_path,
    )

    logger.debug(
        f"[PipelineLog] queue_suppressed | "
        f"queue_id={queue_id} | "
        f"repeat_count={repeat_count} | "
        f"hash={canonical_hash_value[:12]}..."
    )


def log_extraction_verdict(
    queue_id: int,
    worth_remembering: bool,
    reason: str,
    memory_count: int,
    canonical_hash_value: Optional[str] = None,
    db_path: Path = PIPELINE_LOG_DB_PATH,
) -> None:
    """
    Write an extraction_verdict event.

    Called by process_queue_item() after the LLM returns a verdict.
    Written regardless of worth_remembering value when
    log_all_verdicts=True in ExtractionConfig.

    Parameters
    ----------
    queue_id              : ID of the queue item being processed
    worth_remembering     : LLM verdict boolean
    reason                : LLM one-sentence explanation
    memory_count          : Number of memories extracted (0 if false verdict)
    canonical_hash_value  : Hash for cross-table correlation (optional)
    db_path               : Injectable for testing
    """
    _write_event(
        event_type="extraction_verdict",
        canonical_hash=canonical_hash_value,
        queue_id=queue_id,
        worth_remembering=worth_remembering,
        reason=reason,
        repeat_count=memory_count,  # reuse repeat_count col for memory_count
        db_path=db_path,
    )

    logger.debug(
        f"[PipelineLog] extraction_verdict | "
        f"queue_id={queue_id} | "
        f"worth={worth_remembering} | "
        f"memories={memory_count} | "
        f"reason='{reason[:60]}'"
    )


def log_dedup_action(
    canonical_hash_value: str,
    action: str,
    target_id: Optional[str] = None,
    score: Optional[float] = None,
    queue_id: Optional[int] = None,
    db_path: Path = PIPELINE_LOG_DB_PATH,
) -> None:
    """
    Write a dedup_action event.

    Parameters
    ----------
    canonical_hash_value : Hash of the candidate summary
    action               : Dedup action taken
    target_id            : memory_id acted on
    score                : Cosine similarity
    queue_id             : Originating queue item ID
    db_path              : Injectable for testing
    """
    valid_actions = {"merge_hash", "merge_embedding", "skip", "store"}
    if action not in valid_actions:
        logger.warning(
            f"[PipelineLog] Unknown dedup action '{action}'. "
            f"Expected one of {valid_actions}. Writing anyway."
        )

    _write_event(
        event_type="dedup_action",
        canonical_hash=canonical_hash_value,
        queue_id=queue_id,
        memory_id=target_id,
        dedup_action=action,
        similarity_score=score,
        db_path=db_path,
    )

    logger.debug(
        f"[PipelineLog] dedup_action | "
        f"action={action} | "
        f"target={target_id} | "
        f"score={score} | "
        f"hash={canonical_hash_value[:12]}..."
    )


# ---------------------------------------------------------------------------
# Query / reader functions — post-run analysis
# ---------------------------------------------------------------------------

def query_funnel_summary(
    db_path: Path = PIPELINE_LOG_DB_PATH,
) -> dict:
    """
    Return counts for each stage of the pipeline funnel.
    """
    result = {
        "queue_suppressed": 0,
        "extraction_total": 0,
        "extraction_worthy": 0,
        "extraction_rejected": 0,
        "dedup_stored": 0,
        "dedup_merge_hash": 0,
        "dedup_merge_embedding": 0,
        "dedup_skipped": 0,
    }

    try:
        with _get_connection(db_path) as conn:

            # Queue suppression count
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM pipeline_log "
                "WHERE event_type = 'queue_suppressed'"
            ).fetchone()
            result["queue_suppressed"] = row["cnt"]

            # Extraction totals
            rows = conn.execute(
                """
                SELECT worth_remembering, COUNT(*) as cnt
                FROM   pipeline_log
                WHERE  event_type = 'extraction_verdict'
                GROUP  BY worth_remembering
                """
            ).fetchall()
            for row in rows:
                if row["worth_remembering"] == 1:
                    result["extraction_worthy"] = row["cnt"]
                else:
                    result["extraction_rejected"] = row["cnt"]
            result["extraction_total"] = (
                result["extraction_worthy"] + result["extraction_rejected"]
            )

            # Dedup action counts
            rows = conn.execute(
                """
                SELECT dedup_action, COUNT(*) as cnt
                FROM   pipeline_log
                WHERE  event_type = 'dedup_action'
                GROUP  BY dedup_action
                """
            ).fetchall()
            action_map = {
                "store":            "dedup_stored",
                "merge_hash":       "dedup_merge_hash",
                "merge_embedding":  "dedup_merge_embedding",
                "skip":             "dedup_skipped",
            }
            for row in rows:
                key = action_map.get(row["dedup_action"])
                if key:
                    result[key] = row["cnt"]

    except Exception as e:
        logger.warning(f"[PipelineLog] query_funnel_summary failed: {e}")

    return result


def query_suppressed_hashes(
    limit: int = 50,
    db_path: Path = PIPELINE_LOG_DB_PATH,
) -> list[dict]:
    """
    Return the most-suppressed canonical hashes.
    """
    results = []
    try:
        with _get_connection(db_path) as conn:
            rows = conn.execute(
                """
                SELECT
                    canonical_hash,
                    COUNT(*)   as suppression_count,
                    MAX(created_at) as last_seen
                FROM   pipeline_log
                WHERE  event_type = 'queue_suppressed'
                GROUP  BY canonical_hash
                ORDER  BY suppression_count DESC
                LIMIT  ?
                """,
                (limit,)
            ).fetchall()
            results = [dict(row) for row in rows]
    except Exception as e:
        logger.warning(f"[PipelineLog] query_suppressed_hashes failed: {e}")
    return results


def query_verdict_reasons(
    worth_remembering: Optional[bool] = None,
    limit: int = 100,
    db_path: Path = PIPELINE_LOG_DB_PATH,
) -> list[dict]:
    """
    Return extraction verdict reasons for analysis.
    """
    results = []
    try:
        with _get_connection(db_path) as conn:
            if worth_remembering is None:
                rows = conn.execute(
                    """
                    SELECT
                        queue_id,
                        worth_remembering,
                        reason,
                        repeat_count as memory_count,
                        created_at
                    FROM   pipeline_log
                    WHERE  event_type = 'extraction_verdict'
                    ORDER  BY created_at DESC
                    LIMIT  ?
                    """,
                    (limit,)
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT
                        queue_id,
                        worth_remembering,
                        reason,
                        repeat_count as memory_count,
                        created_at
                    FROM   pipeline_log
                    WHERE  event_type = 'extraction_verdict'
                    AND    worth_remembering = ?
                    ORDER  BY created_at DESC
                    LIMIT  ?
                    """,
                    (1 if worth_remembering else 0, limit)
                ).fetchall()

            results = [dict(row) for row in rows]
    except Exception as e:
        logger.warning(f"[PipelineLog] query_verdict_reasons failed: {e}")
    return results


def query_dedup_score_distribution(
    db_path: Path = PIPELINE_LOG_DB_PATH,
) -> dict:
    """
    Return similarity score statistics for Tier 2 dedup decisions.
    """
    result = {}
    try:
        with _get_connection(db_path) as conn:
            rows = conn.execute(
                """
                SELECT
                    dedup_action,
                    COUNT(*)              as count,
                    MIN(similarity_score) as min_score,
                    MAX(similarity_score) as max_score,
                    AVG(similarity_score) as avg_score
                FROM   pipeline_log
                WHERE  event_type = 'dedup_action'
                AND    similarity_score IS NOT NULL
                AND    dedup_action IN ('skip', 'merge_embedding', 'store')
                GROUP  BY dedup_action
                """,
            ).fetchall()

            for row in rows:
                result[row["dedup_action"]] = {
                    "count":     row["count"],
                    "min":       round(row["min_score"], 4) if row["min_score"] else None,
                    "max":       round(row["max_score"], 4) if row["max_score"] else None,
                    "avg":       round(row["avg_score"], 4) if row["avg_score"] else None,
                }
    except Exception as e:
        logger.warning(
            f"[PipelineLog] query_dedup_score_distribution failed: {e}"
        )
    return result


def query_full_trace(
    canonical_hash_value: str,
    db_path: Path = PIPELINE_LOG_DB_PATH,
) -> list[dict]:
    """
    Return all pipeline events for a single canonical hash.
    """
    results = []
    try:
        with _get_connection(db_path) as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM   pipeline_log
                WHERE  canonical_hash = ?
                ORDER  BY created_at ASC
                """,
                (canonical_hash_value,)
            ).fetchall()
            results = [dict(row) for row in rows]
    except Exception as e:
        logger.warning(f"[PipelineLog] query_full_trace failed: {e}")
    return results


def query_recent_events(
    limit: int = 50,
    event_type: Optional[str] = None,
    db_path: Path = PIPELINE_LOG_DB_PATH,
) -> list[dict]:
    """
    Return the most recent pipeline events.
    """
    results = []
    try:
        with _get_connection(db_path) as conn:
            if event_type:
                rows = conn.execute(
                    """
                    SELECT * FROM pipeline_log
                    WHERE  event_type = ?
                    ORDER  BY created_at DESC
                    LIMIT  ?
                    """,
                    (event_type, limit)
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT * FROM pipeline_log
                    ORDER  BY created_at DESC
                    LIMIT  ?
                    """,
                    (limit,)
                ).fetchall()
            results = [dict(row) for row in rows]
    except Exception as e:
        logger.warning(f"[PipelineLog] query_recent_events failed: {e}")
    return results
