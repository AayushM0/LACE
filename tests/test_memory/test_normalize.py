"""
Tests for src/lace/memory/normalize.py
Covers: canonicalize, canonical_hash, is_likely_noise
"""

import pytest
from lace.memory.normalize import canonicalize, canonical_hash, is_likely_noise


# ─────────────────────────────────────────────────────────────────────────────
# TestCanonicalize
# ─────────────────────────────────────────────────────────────────────────────

class TestCanonicalize:
    """Test normalization transformations."""

    def test_lowercase(self):
        assert canonicalize("STRESS TEST") == "stress test"

    def test_numbers_replaced(self):
        assert canonicalize("stress test 1") == "stress test <n>"
        assert canonicalize("stress test 47") == "stress test <n>"

    def test_numbers_produce_same_output(self):
        """The whole point: different numbers → same canonical form."""
        assert canonicalize("stress test 1") == canonicalize("stress test 47")
        assert canonicalize("benchmark run 100") == canonicalize("benchmark run 999")

    def test_decimal_numbers_replaced(self):
        assert canonicalize("similarity score 0.87") == "similarity score <n>"

    def test_uuid_replaced(self):
        result = canonicalize("id: 550e8400-e29b-41d4-a716-446655440000")
        assert "<uuid>" in result
        assert "550e8400" not in result

    def test_uuid_case_insensitive(self):
        lower = canonicalize("id: 550e8400-e29b-41d4-a716-446655440000")
        upper = canonicalize("id: 550E8400-E29B-41D4-A716-446655440000")
        assert lower == upper

    def test_iso_timestamp_replaced(self):
        result = canonicalize("created at 2024-01-15T13:45:00Z")
        assert "<ts>" in result
        assert "2024" not in result

    def test_iso_timestamp_with_offset(self):
        result = canonicalize("time: 2024-06-01 10:30:00+05:30")
        assert "<ts>" in result

    def test_memory_id_replaced(self):
        result = canonicalize("related to mem_a8b9c1d2e3f4")
        assert "<memid>" in result
        assert "mem_a8b9c1d2e3f4" not in result

    def test_punctuation_stripped(self):
        assert canonicalize("hello, world!") == "hello world"
        assert canonicalize("key: value.") == "key value"

    def test_whitespace_collapsed(self):
        assert canonicalize("hello   world") == "hello world"
        assert canonicalize("hello\n\nworld") == "hello world"
        assert canonicalize("  leading and trailing  ") == "leading and trailing"

    def test_empty_string(self):
        assert canonicalize("") == ""

    def test_multiline_interaction(self):
        combined  = "stress test 1\ncompleted in 230ms"
        combined2 = "stress test 2\ncompleted in 245ms"
        # Both stress test items should normalize to the same form
        assert canonicalize(combined) == canonicalize(combined2)

    def test_real_technical_content_not_over_normalized(self):
        """Real content should survive normalization meaningfully."""
        result = canonicalize(
            "Use SQLite WAL mode for concurrent write performance"
        )
        assert "sqlite" in result
        assert "wal" in result
        assert "concurrent" in result


# ─────────────────────────────────────────────────────────────────────────────
# TestCanonicalHash
# ─────────────────────────────────────────────────────────────────────────────

class TestCanonicalHash:
    """Test hash stability and collision properties."""

    def test_identical_text_same_hash(self):
        assert canonical_hash("hello world") == canonical_hash("hello world")

    def test_stress_test_variants_same_hash(self):
        """Core requirement: numbered variants hash identically."""
        h1 = canonical_hash("stress test 1\ncompleted in 230ms")
        h2 = canonical_hash("stress test 47\ncompleted in 891ms")
        assert h1 == h2

    def test_different_content_different_hash(self):
        """Different meaningful content must NOT hash identically."""
        h1 = canonical_hash("SQLite WAL mode improves concurrency")
        h2 = canonical_hash("Redis is better for caching than Memcached")
        assert h1 != h2

    def test_hash_is_64_char_hex(self):
        h = canonical_hash("any text")
        assert len(h) == 64
        assert all(c in "0123456789abcdef" for c in h)

    def test_hash_is_deterministic_across_calls(self):
        text = "MCP server uses stdio for transport"
        hashes = [canonical_hash(text) for _ in range(10)]
        assert len(set(hashes)) == 1  # all identical


# ─────────────────────────────────────────────────────────────────────────────
# TestIsLikelyNoise
# ─────────────────────────────────────────────────────────────────────────────

class TestIsLikelyNoise:
    """Test the pre-flight noise heuristic."""

    def test_stress_test_is_noise(self):
        assert is_likely_noise("stress test 1\ncompleted") is True

    def test_ping_is_noise(self):
        assert is_likely_noise("ping\npong") is True

    def test_pure_numbers_is_noise(self):
        assert is_likely_noise("200\nok") is True

    def test_real_content_is_not_noise(self):
        text = (
            "Why is my ChromaDB query timing out under load?\n"
            "You should increase the batch size and enable persistent mode "
            "to reduce connection overhead in ChromaDB."
        )
        assert is_likely_noise(text) is False

    def test_threshold_respected(self):
        """Custom threshold changes behavior."""
        text = "test one two three four"
        assert is_likely_noise(text, min_meaningful_words=3) is False
        assert is_likely_noise(text, min_meaningful_words=10) is True


# ─────────────────────────────────────────────────────────────────────────────
# TestIntegrationWithRealPatterns
# ─────────────────────────────────────────────────────────────────────────────

class TestIntegrationWithRealPatterns:
    """
    Integration: test the patterns that caused the original bug.
    These are the EXACT inputs that were creating 40-50 junk memories.
    """

    def test_all_stress_test_variants_collapse(self):
        """Simulate 50 stress test queue items — all must hash identically."""
        hashes = set()
        for i in range(1, 51):
            text = f"stress test {i}\nresult: success iteration {i} completed"
            hashes.add(canonical_hash(text))

        assert len(hashes) == 1, (
            f"Expected 1 unique hash, got {len(hashes)}. "
            f"Stress test variants are NOT being collapsed."
        )

    def test_real_memories_dont_collapse(self):
        """Different real memories must NOT collapse to the same hash."""
        memories = [
            "SQLite WAL mode improves concurrent write performance",
            "ChromaDB requires persistent mode for production use",
            "MCP server communicates over stdio using JSON-RPC",
            "Obsidian sync uses mtime comparison for conflict resolution",
        ]
        hashes = [canonical_hash(m) for m in memories]
        assert len(set(hashes)) == 4, "Real memories are incorrectly collapsing"
