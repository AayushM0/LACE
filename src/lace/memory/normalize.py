"""
normalize.py — Canonical text normalization for LACE deduplication.

Used at two pipeline stages:
1. Queue insert: canonicalize(query + response) → hash for suppression
2. Vault dedup:  canonicalize(memory.summary) → hash for tier-1 lookup

Both use the same normalization so volatile tokens (numbers, UUIDs,
timestamps) collapse identically regardless of call site.

Pipeline order:
  1. Lowercase
  2. Replace ISO timestamps   → <ts>
  3. Replace UUIDs            → <uuid>
  4. Replace LACE memory IDs  → <memid>
  5. Replace numbers          → <n>
  6. Strip punctuation        (preserves angle-bracket placeholders)
  7. Collapse whitespace

Punctuation is stripped AFTER placeholders are inserted so that the
`<...>` markers are not destroyed by the punct regex.  The regex for
step 6 only removes characters that are not word chars, whitespace,
or the two angle bracket characters used as placeholder delimiters.
"""

import hashlib
import re


# ─── Compiled regex patterns (compiled once at module load) ───────────────────

# ISO 8601 timestamps: 2024-01-15T13:45:00Z, 2024-01-15 13:45:00+05:30
# re.IGNORECASE so the pattern matches after text.lower() converts 'T' → 't'
_ISO_TS_RE = re.compile(
    r'\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}(\.\d+)?(Z|[+-]\d{2}:\d{2})?',
    re.IGNORECASE,
)

# UUIDs: 550e8400-e29b-41d4-a716-446655440000
_UUID_RE = re.compile(
    r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}',
    re.IGNORECASE,
)

# Memory IDs: mem_a8b9c1d2e3f4
_MEM_ID_RE = re.compile(r'mem_[0-9a-f]{8,}')

# Numbers: integers or decimals, optionally followed by unit letters (e.g. 230ms, 1.5x).
# Using (?<!\d) / (?!\d) avoids matching inside larger numbers while still
# catching numbers embedded in tokens like '230ms' where \b is absent between
# the digit and the following alphabetic unit character.
_NUM_RE = re.compile(r'(?<!\d)\d+(?:\.\d+)?(?!\d)')

# Strip punctuation but PRESERVE < and > so placeholder tokens survive.
# Matches any char that is not: word char (\w), whitespace, <, >
_PUNCT_RE = re.compile(r'[^\w\s<>]')

# Collapse multiple spaces / newlines into a single space
_WHITESPACE_RE = re.compile(r'\s+')


def canonicalize(text: str) -> str:
    """
    Normalize text for near-duplicate detection.

    Transforms volatile tokens so that structurally identical text
    with different numeric/identifier values hashes identically.

    Transformations applied in order:
    1. Lowercase
    2. Replace ISO timestamps → <ts>
    3. Replace UUIDs → <uuid>
    4. Replace LACE memory IDs → <memid>
    5. Replace numbers → <n>
    6. Strip punctuation (angle brackets preserved for placeholders)
    7. Collapse whitespace

    Examples:
        "stress test 1"  → "stress test <n>"
        "stress test 47" → "stress test <n>"    ← same hash
        "2024-01-15T13:00:00Z mem_abc123 completed"
            → "<ts> <memid> completed"

    Args:
        text: Raw text to normalize. Can be multi-line.

    Returns:
        Normalized string. Deterministic for identical semantic content.
    """
    if not text:
        return ""

    t = text.lower()
    t = _ISO_TS_RE.sub('<ts>', t)
    t = _UUID_RE.sub('<uuid>', t)
    t = _MEM_ID_RE.sub('<memid>', t)
    t = _NUM_RE.sub('<n>', t)
    t = _PUNCT_RE.sub('', t)
    t = _WHITESPACE_RE.sub(' ', t).strip()
    return t


def canonical_hash(text: str) -> str:
    """
    SHA-256 hash of the canonical form of text.

    Stable across runs. Two texts with the same canonical form
    produce the same hash. Used for O(1) exact-match dedup lookup.

    Args:
        text: Raw text (query+response for queue, summary for vault).

    Returns:
        64-character hex string (SHA-256).
    """
    normalized = canonicalize(text)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def is_likely_noise(text: str, min_meaningful_words: int = 5) -> bool:
    """
    Quick heuristic check: is this text too low-signal to bother extracting?

    Checks word count AFTER removing volatile tokens. If what remains
    is fewer than min_meaningful_words, it's almost certainly noise
    (e.g., "stress test 1", "ok 200", "run benchmark 3").

    This is NOT the worthiness gate — that's the LLM's job.
    This is a cheap pre-flight that saves an LLM call on obvious junk.

    Args:
        text: Combined query+response text.
        min_meaningful_words: Threshold below which text is noise.

    Returns:
        True if text is likely noise (should skip).
    """
    normalized = canonicalize(text)
    # After normalization, count non-placeholder tokens
    words = [
        w for w in normalized.split()
        if w not in ('<ts>', '<uuid>', '<memid>', '<n>')
        and len(w) > 2  # skip very short tokens
    ]
    return len(words) < min_meaningful_words
