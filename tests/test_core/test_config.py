"""Tests for LACE configuration system."""

import pytest
from pathlib import Path
import yaml

from lace.core.config import (
    LaceConfig,
    MemoryConfig,
    RetrievalConfig,
    DedupConfig,
    ExtractionConfig,
    init_lace_home,
    load_config,
    save_config,
    set_config_value,
    get_lace_home,
)


def test_default_config_is_valid():
    """LaceConfig initializes with correct defaults."""
    config = LaceConfig()
    assert config.version == "1.0"
    assert config.memory.decay_half_life_days == 30
    assert config.retrieval.relevance_threshold == 0.35
    assert config.retrieval.max_results == 20
    assert config.embeddings.provider == "local"


def test_config_weights_sum_to_one():
    """Retrieval weights should sum to 1.0."""
    config = LaceConfig()
    weights = config.retrieval.weights
    total = (
        weights.vector
        + weights.tag
        + weights.graph
        + weights.co_retrieval
        + weights.recency
        + weights.confidence
    )
    assert abs(total - 1.0) < 0.001


def test_config_invalid_weights_sum():
    """Retrieval weights that do not sum to 1.0 should raise validation error."""
    from pydantic import ValidationError
    from lace.core.config import RetrievalWeights
    with pytest.raises(ValidationError):
        RetrievalWeights(vector=0.2, tag=0.2)


def test_config_migrate_old_keys():
    """RetrievalWeights should migrate old config keys successfully."""
    from lace.core.config import RetrievalWeights
    old_data = {
        "semantic_similarity": 0.40,
        "recency": 0.20,
        "frequency": 0.15,
        "confidence": 0.15,
        "scope": 0.10
    }
    weights = RetrievalWeights.model_validate(old_data)
    assert weights.vector == 0.40
    assert weights.recency == 0.20
    assert weights.co_retrieval == 0.15
    assert weights.confidence == 0.15
    assert weights.tag == 0.05
    assert weights.graph == 0.05


def test_init_creates_directory_structure(tmp_path):
    """lace init creates all required directories."""
    lace_home = tmp_path / ".lace"
    path, already_existed = init_lace_home(lace_home)

    assert not already_existed
    assert (lace_home / "memory" / "vault" / "global" / "patterns").exists()
    assert (lace_home / "memory" / "vault" / "global" / "decisions").exists()
    assert (lace_home / "memory" / "vault" / "global" / "debug-log").exists()
    assert (lace_home / "memory" / "vault" / "global" / "references").exists()
    assert (lace_home / "memory" / "vault" / "projects").exists()
    assert (lace_home / "logs" / "retrieval").exists()
    assert (lace_home / "logs" / "interactions").exists()
    assert (lace_home / "sessions").exists()
    assert (lace_home / "config" / "projects").exists()


def test_init_is_idempotent(tmp_path):
    """Running init twice doesn't fail or overwrite existing files."""
    lace_home = tmp_path / ".lace"

    _, first = init_lace_home(lace_home)
    assert not first

    _, second = init_lace_home(lace_home)
    assert second  # already existed on second run


def test_load_config_missing_file_returns_defaults(tmp_path):
    """Loading config when file doesn't exist returns defaults."""
    lace_home = tmp_path / ".lace"
    lace_home.mkdir()

    config = load_config(lace_home)
    assert isinstance(config, LaceConfig)
    assert config.memory.dedup_threshold == 0.85


def test_save_and_load_roundtrip(tmp_path):
    """Config can be saved and loaded back identically."""
    lace_home = tmp_path / ".lace"
    lace_home.mkdir()
    (lace_home / "config").mkdir()

    config = LaceConfig()
    config.memory.decay_half_life_days = 60
    config.retrieval.max_results = 15

    save_config(config, lace_home)
    loaded = load_config(lace_home)

    assert loaded.memory.decay_half_life_days == 60
    assert loaded.retrieval.max_results == 15


def test_set_config_value_int(tmp_path):
    """set_config_value correctly sets an integer value."""
    lace_home = tmp_path / ".lace"
    init_lace_home(lace_home)

    set_config_value("memory.decay_half_life_days", "60", lace_home)
    config = load_config(lace_home)
    assert config.memory.decay_half_life_days == 60


def test_set_config_value_bool(tmp_path):
    """set_config_value correctly sets a boolean value."""
    lace_home = tmp_path / ".lace"
    init_lace_home(lace_home)

    set_config_value("memory.auto_extract", "true", lace_home)
    config = load_config(lace_home)
    assert config.memory.auto_extract is True


def test_set_config_value_unknown_key_raises(tmp_path):
    """set_config_value raises KeyError for unknown keys."""
    lace_home = tmp_path / ".lace"
    init_lace_home(lace_home)

    with pytest.raises(KeyError):
        set_config_value("memory.nonexistent_key", "value", lace_home)


def test_vault_path_default(tmp_path):
    """vault_path returns default when not configured."""
    lace_home = tmp_path / ".lace"
    config = LaceConfig()
    assert config.vault_path(lace_home) == lace_home / "memory" / "vault"


def test_vault_path_custom():
    """vault_path respects custom path when set."""
    config = LaceConfig()
    config.vault.path = "/custom/vault"
    assert config.vault_path(Path("/anything")) == Path("/custom/vault").resolve()



# ── Chunk 1: DedupConfig tests ────────────────────────────────────────────────


class TestDedupConfig:
    """Unit tests for DedupConfig validation."""

    def test_defaults_are_sensible(self):
        cfg = DedupConfig()
        assert cfg.skip_threshold == 0.95
        assert cfg.merge_threshold == 0.85
        assert cfg.hash_cooldown_seconds == 300

    def test_threshold_validation_passes(self):
        """merge < skip is valid."""
        cfg = DedupConfig(merge_threshold=0.80, skip_threshold=0.95)
        cfg.validate_thresholds()  # should not raise

    def test_threshold_validation_fails_when_merge_gte_skip(self):
        """merge >= skip should raise."""
        cfg = DedupConfig(merge_threshold=0.95, skip_threshold=0.90)
        with pytest.raises(ValueError, match="merge_threshold"):
            cfg.validate_thresholds()

    def test_float_bounds(self):
        """Thresholds must be 0.0-1.0."""
        with pytest.raises(Exception):
            DedupConfig(skip_threshold=1.5)
        with pytest.raises(Exception):
            DedupConfig(merge_threshold=-0.1)


# ── Chunk 1: ExtractionConfig tests ──────────────────────────────────────────


class TestExtractionConfig:
    """Unit tests for ExtractionConfig."""

    def test_defaults(self):
        cfg = ExtractionConfig()
        assert cfg.require_worthiness_verdict is True
        assert cfg.log_all_verdicts is True
        assert cfg.max_memories_per_interaction == 5

    def test_max_memories_minimum(self):
        with pytest.raises(Exception):
            ExtractionConfig(max_memories_per_interaction=0)


# ── Chunk 1: LaceConfig nested blocks integration tests ──────────────────────


class TestLaceConfigNewBlocks:
    """Integration tests for dedup/extraction nested blocks via load_config/save_config."""

    def test_defaults_when_no_file(self, tmp_path):
        """Missing config file → all defaults apply."""
        lace_home = tmp_path / ".lace"
        cfg = load_config(lace_home)
        assert cfg.dedup.skip_threshold == 0.95
        assert cfg.extraction.require_worthiness_verdict is True

    def test_save_and_reload(self, tmp_path):
        """Save config then reload — new block values must survive round-trip."""
        lace_home = tmp_path / ".lace"
        (lace_home / "config").mkdir(parents=True)

        original = LaceConfig(
            dedup=DedupConfig(
                skip_threshold=0.92,
                merge_threshold=0.78,
                hash_cooldown_seconds=600,
            ),
            extraction=ExtractionConfig(
                require_worthiness_verdict=False,
                log_all_verdicts=True,
            ),
        )
        save_config(original, lace_home)

        reloaded = load_config(lace_home)
        assert reloaded.dedup.skip_threshold == 0.92
        assert reloaded.dedup.merge_threshold == 0.78
        assert reloaded.dedup.hash_cooldown_seconds == 600
        assert reloaded.extraction.require_worthiness_verdict is False

    def test_partial_yaml_uses_defaults(self, tmp_path):
        """YAML with only some dedup fields → missing fields use defaults."""
        lace_home = tmp_path / ".lace"
        config_file = lace_home / "config" / "lace.yaml"
        config_file.parent.mkdir(parents=True)
        config_file.write_text(yaml.dump({"dedup": {"skip_threshold": 0.90}}))

        cfg = load_config(lace_home)
        assert cfg.dedup.skip_threshold == 0.90
        assert cfg.dedup.merge_threshold == 0.85       # default
        assert cfg.dedup.hash_cooldown_seconds == 300  # default

    def test_no_nested_blocks_uses_all_defaults(self, tmp_path):
        """YAML with no dedup/extraction blocks → all defaults."""
        lace_home = tmp_path / ".lace"
        config_file = lace_home / "config" / "lace.yaml"
        config_file.parent.mkdir(parents=True)
        config_file.write_text(yaml.dump({"version": "1.0"}))

        cfg = load_config(lace_home)
        assert cfg.dedup.skip_threshold == 0.95
        assert cfg.extraction.log_all_verdicts is True

    def test_lace_config_dedup_defaults(self):
        """LaceConfig() constructor wires dedup/extraction with correct defaults."""
        cfg = LaceConfig()
        assert cfg.dedup.skip_threshold == 0.95
        assert cfg.dedup.merge_threshold == 0.85
        assert cfg.dedup.hash_cooldown_seconds == 300
        assert cfg.extraction.require_worthiness_verdict is True
        assert cfg.extraction.log_all_verdicts is True
        assert cfg.extraction.extraction_model == "gpt-4o-mini"
        assert cfg.extraction.max_memories_per_interaction == 5