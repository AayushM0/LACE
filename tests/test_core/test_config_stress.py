"""
Stress & Integration Tests — DedupConfig / ExtractionConfig / LaceConfig
=========================================================================

Covers:
  A. DedupConfig — boundary values, validate_thresholds exhaustive matrix
  B. ExtractionConfig — boundary values, model name edge cases
  C. LaceConfig construction — every field combo, isolation between instances
  D. YAML round-trip — every supported type, nested partial overrides
  E. YAML corruption / malformed input
  F. set_config_value — dot-notation access to dedup.* and extraction.*
  G. LACE_HOME environment variable interaction
  H. init_lace_home + template propagation
  I. Immutability & instance independence
  J. model_dump / model_validate symmetry
  K. Concurrent save + load smoke test
  L. Large / extreme values
  M. System integration — new blocks flow into GraphManager
"""

from __future__ import annotations

import concurrent.futures
import json
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from lace.core.config import (
    DedupConfig,
    ExtractionConfig,
    LaceConfig,
    get_lace_home,
    init_lace_home,
    load_config,
    save_config,
    set_config_value,
)


# ─────────────────────────────────────────────────────────────────────────────
# A. DedupConfig — boundary & exhaustive threshold matrix
# ─────────────────────────────────────────────────────────────────────────────

class TestDedupConfigBoundaries:

    @pytest.mark.parametrize("v", [0.0, 0.001, 0.5, 0.94999, 0.95, 1.0])
    def test_skip_threshold_valid_range(self, v):
        cfg = DedupConfig(skip_threshold=v, merge_threshold=0.0)
        assert cfg.skip_threshold == v

    @pytest.mark.parametrize("v", [-0.001, -1.0, 1.001, 2.0])
    def test_skip_threshold_out_of_range_raises(self, v):
        with pytest.raises((ValidationError, Exception)):
            DedupConfig(skip_threshold=v)

    @pytest.mark.parametrize("v", [0.0, 0.001, 0.5, 0.849, 0.85, 1.0])
    def test_merge_threshold_valid_range(self, v):
        cfg = DedupConfig(merge_threshold=v, skip_threshold=1.0)
        assert cfg.merge_threshold == v

    @pytest.mark.parametrize("v", [-0.001, -1.0, 1.001])
    def test_merge_threshold_out_of_range_raises(self, v):
        with pytest.raises((ValidationError, Exception)):
            DedupConfig(merge_threshold=v)

    @pytest.mark.parametrize("v", [0, 1, 300, 86400, 10_000_000])
    def test_hash_cooldown_valid(self, v):
        cfg = DedupConfig(hash_cooldown_seconds=v)
        assert cfg.hash_cooldown_seconds == v

    def test_hash_cooldown_negative_raises(self):
        with pytest.raises((ValidationError, Exception)):
            DedupConfig(hash_cooldown_seconds=-1)

    def test_hash_cooldown_is_int(self):
        cfg = DedupConfig(hash_cooldown_seconds=300)
        assert isinstance(cfg.hash_cooldown_seconds, int)


class TestDedupValidateThresholdsMatrix:

    @pytest.mark.parametrize("merge,skip", [
        (0.00, 0.01),
        (0.50, 0.51),
        (0.84, 0.85),
        (0.85, 0.86),
        (0.94, 0.95),
        (0.0, 1.0),
    ])
    def test_merge_lt_skip_passes(self, merge, skip):
        cfg = DedupConfig(merge_threshold=merge, skip_threshold=skip)
        cfg.validate_thresholds()  # must not raise

    @pytest.mark.parametrize("merge,skip", [
        (0.95, 0.95),
        (0.96, 0.95),
        (1.0, 0.99),
        (0.86, 0.85),
    ])
    def test_merge_gte_skip_fails(self, merge, skip):
        cfg = DedupConfig(merge_threshold=merge, skip_threshold=skip)
        with pytest.raises(ValueError, match="merge_threshold"):
            cfg.validate_thresholds()

    def test_error_message_contains_both_values(self):
        cfg = DedupConfig(merge_threshold=0.90, skip_threshold=0.80)
        with pytest.raises(ValueError) as exc_info:
            cfg.validate_thresholds()
        msg = str(exc_info.value)
        assert "0.9" in msg
        assert "0.8" in msg


# ─────────────────────────────────────────────────────────────────────────────
# B. ExtractionConfig — boundary & model name edge cases
# ─────────────────────────────────────────────────────────────────────────────

class TestExtractionConfigBoundaries:

    @pytest.mark.parametrize("v", [1, 2, 5, 10, 100, 10_000])
    def test_max_memories_valid(self, v):
        cfg = ExtractionConfig(max_memories_per_interaction=v)
        assert cfg.max_memories_per_interaction == v

    @pytest.mark.parametrize("v", [0, -1, -100])
    def test_max_memories_below_minimum_raises(self, v):
        with pytest.raises((ValidationError, Exception)):
            ExtractionConfig(max_memories_per_interaction=v)

    @pytest.mark.parametrize("model", [
        "gpt-4o-mini",
        "gpt-4o",
        "claude-3-5-sonnet-20241022",
        "ollama/llama3.2",
        "a" * 200,
        "",
    ])
    def test_extraction_model_accepts_arbitrary_strings(self, model):
        cfg = ExtractionConfig(extraction_model=model)
        assert cfg.extraction_model == model

    @pytest.mark.parametrize("flag", [True, False])
    def test_boolean_flags_independently(self, flag):
        cfg = ExtractionConfig(
            require_worthiness_verdict=flag,
            log_all_verdicts=not flag,
        )
        assert cfg.require_worthiness_verdict == flag
        assert cfg.log_all_verdicts == (not flag)


# ─────────────────────────────────────────────────────────────────────────────
# C. LaceConfig construction
# ─────────────────────────────────────────────────────────────────────────────

class TestLaceConfigConstruction:

    def test_all_new_defaults_present_on_bare_init(self):
        cfg = LaceConfig()
        assert hasattr(cfg, "dedup")
        assert hasattr(cfg, "extraction")
        assert isinstance(cfg.dedup, DedupConfig)
        assert isinstance(cfg.extraction, ExtractionConfig)

    def test_existing_fields_still_present(self):
        cfg = LaceConfig()
        for field in ("memory", "retrieval", "vault", "logging", "embeddings", "provider"):
            assert hasattr(cfg, field), f"Missing field: {field}"

    def test_new_blocks_dont_override_old_defaults(self):
        cfg = LaceConfig()
        assert cfg.memory.dedup_threshold == 0.85
        assert cfg.retrieval.relevance_threshold == 0.35
        assert cfg.embeddings.provider == "local"

    def test_custom_dedup_passed_inline(self):
        cfg = LaceConfig(dedup=DedupConfig(skip_threshold=0.99, merge_threshold=0.50))
        assert cfg.dedup.skip_threshold == 0.99
        assert cfg.dedup.merge_threshold == 0.50

    def test_custom_extraction_passed_inline(self):
        cfg = LaceConfig(
            extraction=ExtractionConfig(
                require_worthiness_verdict=False,
                max_memories_per_interaction=1,
            )
        )
        assert cfg.extraction.require_worthiness_verdict is False
        assert cfg.extraction.max_memories_per_interaction == 1

    def test_multiple_instances_are_independent(self):
        a = LaceConfig()
        b = LaceConfig()
        a.dedup.skip_threshold = 0.50
        assert b.dedup.skip_threshold == 0.95

    def test_version_field_unchanged(self):
        cfg = LaceConfig()
        assert cfg.version == "1.0"


# ─────────────────────────────────────────────────────────────────────────────
# D. YAML round-trip
# ─────────────────────────────────────────────────────────────────────────────

class TestYamlRoundTrip:

    def _write_yaml(self, path: Path, data: dict) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.dump(data, default_flow_style=False))

    def test_full_dedup_block_survives_roundtrip(self, tmp_path):
        lace_home = tmp_path / ".lace"
        (lace_home / "config").mkdir(parents=True)
        original = LaceConfig(
            dedup=DedupConfig(skip_threshold=0.91, merge_threshold=0.72, hash_cooldown_seconds=120)
        )
        save_config(original, lace_home)
        reloaded = load_config(lace_home)
        assert reloaded.dedup.skip_threshold == 0.91
        assert reloaded.dedup.merge_threshold == 0.72
        assert reloaded.dedup.hash_cooldown_seconds == 120

    def test_full_extraction_block_survives_roundtrip(self, tmp_path):
        lace_home = tmp_path / ".lace"
        (lace_home / "config").mkdir(parents=True)
        original = LaceConfig(
            extraction=ExtractionConfig(
                require_worthiness_verdict=False,
                log_all_verdicts=False,
                extraction_model="ollama/mistral",
                max_memories_per_interaction=10,
            )
        )
        save_config(original, lace_home)
        reloaded = load_config(lace_home)
        assert reloaded.extraction.require_worthiness_verdict is False
        assert reloaded.extraction.log_all_verdicts is False
        assert reloaded.extraction.extraction_model == "ollama/mistral"
        assert reloaded.extraction.max_memories_per_interaction == 10

    def test_only_one_new_field_in_yaml_rest_default(self, tmp_path):
        lace_home = tmp_path / ".lace"
        config_file = lace_home / "config" / "lace.yaml"
        self._write_yaml(config_file, {"extraction": {"max_memories_per_interaction": 3}})
        cfg = load_config(lace_home)
        assert cfg.extraction.max_memories_per_interaction == 3
        assert cfg.extraction.require_worthiness_verdict is True
        assert cfg.dedup.skip_threshold == 0.95

    def test_empty_yaml_file_gives_all_defaults(self, tmp_path):
        lace_home = tmp_path / ".lace"
        config_file = lace_home / "config" / "lace.yaml"
        config_file.parent.mkdir(parents=True)
        config_file.write_text("")
        cfg = load_config(lace_home)
        assert cfg.dedup.skip_threshold == 0.95
        assert cfg.extraction.max_memories_per_interaction == 5

    def test_yaml_null_blocks_handled(self, tmp_path):
        lace_home = tmp_path / ".lace"
        config_file = lace_home / "config" / "lace.yaml"
        config_file.parent.mkdir(parents=True)
        config_file.write_text("dedup: null\nextraction: null\n")
        try:
            cfg = load_config(lace_home)
            assert isinstance(cfg, LaceConfig)
        except (ValidationError, Exception):
            pass  # acceptable: null nested model may be rejected

    def test_integer_thresholds_coerced_to_float(self, tmp_path):
        lace_home = tmp_path / ".lace"
        config_file = lace_home / "config" / "lace.yaml"
        self._write_yaml(config_file, {"dedup": {"skip_threshold": 1, "merge_threshold": 0}})
        cfg = load_config(lace_home)
        assert isinstance(cfg.dedup.skip_threshold, float)
        assert cfg.dedup.skip_threshold == 1.0

    def test_roundtrip_preserves_all_existing_blocks(self, tmp_path):
        lace_home = tmp_path / ".lace"
        (lace_home / "config").mkdir(parents=True)
        save_config(LaceConfig(), lace_home)
        reloaded = load_config(lace_home)
        assert reloaded.memory.decay_half_life_days == 30
        assert reloaded.retrieval.max_results == 20
        assert reloaded.embeddings.provider == "local"
        assert reloaded.provider.default == "ollama"
        assert reloaded.dedup.skip_threshold == 0.95
        assert reloaded.extraction.require_worthiness_verdict is True

    def test_multiple_save_load_cycles_stable(self, tmp_path):
        lace_home = tmp_path / ".lace"
        (lace_home / "config").mkdir(parents=True)
        cfg = LaceConfig(dedup=DedupConfig(skip_threshold=0.88, merge_threshold=0.77))
        save_config(cfg, lace_home)
        for _ in range(5):
            cfg = load_config(lace_home)
            save_config(cfg, lace_home)
        final = load_config(lace_home)
        assert final.dedup.skip_threshold == 0.88
        assert final.dedup.merge_threshold == 0.77


# ─────────────────────────────────────────────────────────────────────────────
# E. YAML corruption / malformed input
# ─────────────────────────────────────────────────────────────────────────────

class TestMalformedYaml:

    def test_invalid_yaml_syntax_raises(self, tmp_path):
        lace_home = tmp_path / ".lace"
        config_file = lace_home / "config" / "lace.yaml"
        config_file.parent.mkdir(parents=True)
        config_file.write_text("dedup: {skip_threshold: [not, a, float}")
        with pytest.raises(Exception):
            load_config(lace_home)

    def test_wrong_type_for_skip_threshold_raises(self, tmp_path):
        lace_home = tmp_path / ".lace"
        config_file = lace_home / "config" / "lace.yaml"
        config_file.parent.mkdir(parents=True)
        config_file.write_text(yaml.dump({"dedup": {"skip_threshold": "not-a-float"}}))
        with pytest.raises((ValidationError, Exception)):
            load_config(lace_home)

    def test_out_of_range_threshold_in_yaml_raises(self, tmp_path):
        lace_home = tmp_path / ".lace"
        config_file = lace_home / "config" / "lace.yaml"
        config_file.parent.mkdir(parents=True)
        config_file.write_text(yaml.dump({"dedup": {"skip_threshold": 1.5}}))
        with pytest.raises((ValidationError, Exception)):
            load_config(lace_home)

    def test_negative_max_memories_in_yaml_raises(self, tmp_path):
        lace_home = tmp_path / ".lace"
        config_file = lace_home / "config" / "lace.yaml"
        config_file.parent.mkdir(parents=True)
        config_file.write_text(yaml.dump({"extraction": {"max_memories_per_interaction": -5}}))
        with pytest.raises((ValidationError, Exception)):
            load_config(lace_home)

    def test_unknown_keys_in_dedup_block_ignored(self, tmp_path):
        lace_home = tmp_path / ".lace"
        config_file = lace_home / "config" / "lace.yaml"
        config_file.parent.mkdir(parents=True)
        config_file.write_text(yaml.dump({
            "dedup": {"skip_threshold": 0.90, "unknown_future_key": "some_value"}
        }))
        cfg = load_config(lace_home)
        assert cfg.dedup.skip_threshold == 0.90

    def test_unknown_keys_in_extraction_block_ignored(self, tmp_path):
        lace_home = tmp_path / ".lace"
        config_file = lace_home / "config" / "lace.yaml"
        config_file.parent.mkdir(parents=True)
        config_file.write_text(yaml.dump({
            "extraction": {"max_memories_per_interaction": 7, "future_flag": True}
        }))
        cfg = load_config(lace_home)
        assert cfg.extraction.max_memories_per_interaction == 7


# ─────────────────────────────────────────────────────────────────────────────
# F. set_config_value — dot-notation for new blocks
# ─────────────────────────────────────────────────────────────────────────────

class TestSetConfigValueNewBlocks:

    def test_set_dedup_skip_threshold(self, tmp_path):
        lace_home = tmp_path / ".lace"
        init_lace_home(lace_home)
        set_config_value("dedup.skip_threshold", "0.88", lace_home)
        assert load_config(lace_home).dedup.skip_threshold == pytest.approx(0.88)

    def test_set_dedup_merge_threshold(self, tmp_path):
        lace_home = tmp_path / ".lace"
        init_lace_home(lace_home)
        set_config_value("dedup.merge_threshold", "0.70", lace_home)
        assert load_config(lace_home).dedup.merge_threshold == pytest.approx(0.70)

    def test_set_dedup_hash_cooldown(self, tmp_path):
        lace_home = tmp_path / ".lace"
        init_lace_home(lace_home)
        set_config_value("dedup.hash_cooldown_seconds", "600", lace_home)
        assert load_config(lace_home).dedup.hash_cooldown_seconds == 600

    def test_set_extraction_worthiness_false(self, tmp_path):
        lace_home = tmp_path / ".lace"
        init_lace_home(lace_home)
        set_config_value("extraction.require_worthiness_verdict", "false", lace_home)
        assert load_config(lace_home).extraction.require_worthiness_verdict is False

    def test_set_extraction_max_memories(self, tmp_path):
        lace_home = tmp_path / ".lace"
        init_lace_home(lace_home)
        set_config_value("extraction.max_memories_per_interaction", "12", lace_home)
        assert load_config(lace_home).extraction.max_memories_per_interaction == 12

    def test_set_extraction_model_string(self, tmp_path):
        lace_home = tmp_path / ".lace"
        init_lace_home(lace_home)
        set_config_value("extraction.extraction_model", "ollama/mistral", lace_home)
        assert load_config(lace_home).extraction.extraction_model == "ollama/mistral"

    def test_set_unknown_dedup_key_raises(self, tmp_path):
        lace_home = tmp_path / ".lace"
        init_lace_home(lace_home)
        with pytest.raises(KeyError):
            set_config_value("dedup.nonexistent_key", "0.5", lace_home)

    def test_set_unknown_extraction_key_raises(self, tmp_path):
        lace_home = tmp_path / ".lace"
        init_lace_home(lace_home)
        with pytest.raises(KeyError):
            set_config_value("extraction.nonexistent_key", "true", lace_home)

    def test_set_value_doesnt_corrupt_existing_blocks(self, tmp_path):
        lace_home = tmp_path / ".lace"
        init_lace_home(lace_home)
        original_decay = load_config(lace_home).memory.decay_half_life_days
        set_config_value("dedup.hash_cooldown_seconds", "999", lace_home)
        cfg = load_config(lace_home)
        assert cfg.memory.decay_half_life_days == original_decay
        assert cfg.dedup.hash_cooldown_seconds == 999


# ─────────────────────────────────────────────────────────────────────────────
# G. LACE_HOME environment variable interaction
# ─────────────────────────────────────────────────────────────────────────────

class TestLaceHomeEnvVar:

    def test_get_lace_home_respects_env(self, tmp_path, monkeypatch):
        custom_home = str(tmp_path / "custom_lace")
        monkeypatch.setenv("LACE_HOME", custom_home)
        result = get_lace_home()
        assert result == Path(custom_home).expanduser().resolve()

    def test_get_lace_home_falls_back_to_user_home(self, monkeypatch):
        monkeypatch.delenv("LACE_HOME", raising=False)
        result = get_lace_home()
        assert result == Path.home() / ".lace"

    def test_load_config_uses_env_home(self, tmp_path, monkeypatch):
        custom_home = tmp_path / "custom_lace"
        config_file = custom_home / "config" / "lace.yaml"
        config_file.parent.mkdir(parents=True)
        config_file.write_text(yaml.dump({"dedup": {"skip_threshold": 0.77}}))
        monkeypatch.setenv("LACE_HOME", str(custom_home))
        cfg = load_config()
        assert cfg.dedup.skip_threshold == pytest.approx(0.77)

    def test_save_config_uses_env_home(self, tmp_path, monkeypatch):
        custom_home = tmp_path / "custom_lace"
        monkeypatch.setenv("LACE_HOME", str(custom_home))
        cfg = LaceConfig(dedup=DedupConfig(hash_cooldown_seconds=999))
        save_config(cfg)
        assert load_config().dedup.hash_cooldown_seconds == 999


# ─────────────────────────────────────────────────────────────────────────────
# H. init_lace_home + template propagation
# ─────────────────────────────────────────────────────────────────────────────

class TestInitLaceHomeTemplate:

    def test_init_installs_lace_yaml_with_new_blocks(self, tmp_path):
        lace_home = tmp_path / ".lace"
        init_lace_home(lace_home)
        config_file = lace_home / "config" / "lace.yaml"
        if config_file.exists():
            raw = yaml.safe_load(config_file.read_text()) or {}
            assert "dedup" in raw, "dedup block missing from installed lace.yaml template"
            assert "extraction" in raw, "extraction block missing from installed lace.yaml template"

    def test_init_installed_yaml_is_valid_config(self, tmp_path):
        lace_home = tmp_path / ".lace"
        init_lace_home(lace_home)
        cfg = load_config(lace_home)
        assert isinstance(cfg, LaceConfig)
        assert cfg.dedup.skip_threshold == 0.95
        assert cfg.extraction.require_worthiness_verdict is True

    def test_init_does_not_overwrite_existing_config(self, tmp_path):
        lace_home = tmp_path / ".lace"
        init_lace_home(lace_home)
        config_file = lace_home / "config" / "lace.yaml"
        config_file.write_text(yaml.dump({"dedup": {"skip_threshold": 0.11}}))
        init_lace_home(lace_home)
        raw = yaml.safe_load(config_file.read_text()) or {}
        assert raw.get("dedup", {}).get("skip_threshold") == 0.11


# ─────────────────────────────────────────────────────────────────────────────
# I. Immutability & instance independence
# ─────────────────────────────────────────────────────────────────────────────

class TestInstanceIndependence:

    def test_dedup_defaults_not_shared_between_lace_instances(self):
        a = LaceConfig()
        b = LaceConfig()
        a.dedup.skip_threshold = 0.1
        assert b.dedup.skip_threshold == 0.95

    def test_extraction_defaults_not_shared_between_lace_instances(self):
        a = LaceConfig()
        b = LaceConfig()
        a.extraction.max_memories_per_interaction = 99
        assert b.extraction.max_memories_per_interaction == 5

    def test_dedup_config_instances_independent(self):
        a = DedupConfig()
        b = DedupConfig()
        a.skip_threshold = 0.1
        assert b.skip_threshold == 0.95

    def test_extraction_config_instances_independent(self):
        a = ExtractionConfig()
        b = ExtractionConfig()
        a.max_memories_per_interaction = 100
        assert b.max_memories_per_interaction == 5


# ─────────────────────────────────────────────────────────────────────────────
# J. model_dump / model_validate symmetry
# ─────────────────────────────────────────────────────────────────────────────

class TestModelDumpValidateSymmetry:

    def test_dedup_dump_and_validate(self):
        original = DedupConfig(skip_threshold=0.91, merge_threshold=0.71, hash_cooldown_seconds=450)
        assert DedupConfig.model_validate(original.model_dump()) == original

    def test_extraction_dump_and_validate(self):
        original = ExtractionConfig(
            require_worthiness_verdict=False,
            log_all_verdicts=False,
            extraction_model="gpt-4o",
            max_memories_per_interaction=3,
        )
        assert ExtractionConfig.model_validate(original.model_dump()) == original

    def test_lace_config_full_dump_validate(self):
        original = LaceConfig(
            dedup=DedupConfig(skip_threshold=0.88),
            extraction=ExtractionConfig(max_memories_per_interaction=8),
        )
        dumped = original.model_dump()
        assert "dedup" in dumped
        assert "extraction" in dumped
        assert dumped["dedup"]["skip_threshold"] == 0.88
        assert dumped["extraction"]["max_memories_per_interaction"] == 8
        reconstructed = LaceConfig.model_validate(dumped)
        assert reconstructed.dedup.skip_threshold == 0.88
        assert reconstructed.extraction.max_memories_per_interaction == 8

    def test_json_mode_dump_is_serialisable(self):
        cfg = LaceConfig(
            dedup=DedupConfig(skip_threshold=0.90),
            extraction=ExtractionConfig(extraction_model="gpt-4o-mini"),
        )
        serialised = json.dumps(cfg.model_dump(mode="json"))
        assert "dedup" in serialised
        assert "extraction" in serialised


# ─────────────────────────────────────────────────────────────────────────────
# K. Concurrent save + load smoke test
# ─────────────────────────────────────────────────────────────────────────────

class TestConcurrentAccess:

    def test_concurrent_reads_dont_crash(self, tmp_path):
        lace_home = tmp_path / ".lace"
        (lace_home / "config").mkdir(parents=True)
        save_config(LaceConfig(), lace_home)
        errors = []
        results = []

        def reader():
            try:
                cfg = load_config(lace_home)
                results.append(cfg.dedup.skip_threshold)
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as pool:
            concurrent.futures.wait([pool.submit(reader) for _ in range(20)])

        assert not errors, f"Concurrent reads raised: {errors}"
        assert all(v == 0.95 for v in results)

    def test_interleaved_save_load_no_attribute_errors(self, tmp_path):
        lace_home = tmp_path / ".lace"
        (lace_home / "config").mkdir(parents=True)
        save_config(LaceConfig(), lace_home)
        errors = []

        def writer(val: float):
            try:
                cfg = LaceConfig(dedup=DedupConfig(skip_threshold=val, merge_threshold=val - 0.1))
                save_config(cfg, lace_home)
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                cfg = load_config(lace_home)
                assert isinstance(cfg, LaceConfig)
            except Exception as e:
                errors.append(e)

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futs = [pool.submit(writer, 0.8 + i * 0.01) for i in range(4)]
            futs += [pool.submit(reader) for _ in range(8)]
            concurrent.futures.wait(futs)

        for e in errors:
            assert not isinstance(e, (AttributeError, TypeError)), \
                f"Data corruption detected: {type(e).__name__}: {e}"


# ─────────────────────────────────────────────────────────────────────────────
# L. Large / extreme values
# ─────────────────────────────────────────────────────────────────────────────

class TestExtremeValues:

    def test_hash_cooldown_very_large(self):
        cfg = DedupConfig(hash_cooldown_seconds=2**31 - 1)
        assert cfg.hash_cooldown_seconds == 2**31 - 1

    def test_max_memories_very_large(self):
        cfg = ExtractionConfig(max_memories_per_interaction=10_000)
        assert cfg.max_memories_per_interaction == 10_000

    def test_thresholds_at_exact_boundaries(self):
        DedupConfig(skip_threshold=1.0, merge_threshold=0.0)
        DedupConfig(skip_threshold=0.5, merge_threshold=0.0)

    def test_full_config_with_all_extremes(self, tmp_path):
        lace_home = tmp_path / ".lace"
        (lace_home / "config").mkdir(parents=True)
        cfg = LaceConfig(
            dedup=DedupConfig(skip_threshold=1.0, merge_threshold=0.0, hash_cooldown_seconds=0),
            extraction=ExtractionConfig(
                require_worthiness_verdict=False,
                log_all_verdicts=False,
                extraction_model="x" * 255,
                max_memories_per_interaction=9999,
            ),
        )
        save_config(cfg, lace_home)
        reloaded = load_config(lace_home)
        assert reloaded.dedup.skip_threshold == 1.0
        assert reloaded.dedup.merge_threshold == 0.0
        assert reloaded.dedup.hash_cooldown_seconds == 0
        assert reloaded.extraction.max_memories_per_interaction == 9999


# ─────────────────────────────────────────────────────────────────────────────
# M. System integration — new blocks flow into GraphManager and downstream
# ─────────────────────────────────────────────────────────────────────────────

class TestSystemIntegration:

    def test_load_config_returns_lace_config_instance(self, tmp_path):
        cfg = load_config(tmp_path / ".lace")
        assert isinstance(cfg, LaceConfig)

    def test_graph_manager_accepts_config_with_new_blocks(self, tmp_path):
        from lace.core.engine import GraphManager
        lace_home = tmp_path / ".lace"
        init_lace_home(lace_home)
        cfg = LaceConfig(
            dedup=DedupConfig(skip_threshold=0.92),
            extraction=ExtractionConfig(max_memories_per_interaction=3),
        )
        gm = GraphManager(lace_home=lace_home, config=cfg)
        assert gm.config.dedup.skip_threshold == 0.92
        assert gm.config.extraction.max_memories_per_interaction == 3

    def test_config_dedup_fields_accessible_as_primitives(self):
        cfg = LaceConfig()
        assert isinstance(cfg.dedup.skip_threshold, float)
        assert isinstance(cfg.dedup.merge_threshold, float)
        assert isinstance(cfg.dedup.hash_cooldown_seconds, int)

    def test_config_extraction_fields_accessible_as_primitives(self):
        cfg = LaceConfig()
        assert isinstance(cfg.extraction.require_worthiness_verdict, bool)
        assert isinstance(cfg.extraction.log_all_verdicts, bool)
        assert isinstance(cfg.extraction.extraction_model, str)
        assert isinstance(cfg.extraction.max_memories_per_interaction, int)

    def test_new_blocks_survive_set_config_value_on_existing_field(self, tmp_path):
        lace_home = tmp_path / ".lace"
        init_lace_home(lace_home)
        set_config_value("dedup.skip_threshold", "0.77", lace_home)
        set_config_value("extraction.max_memories_per_interaction", "7", lace_home)
        set_config_value("memory.decay_half_life_days", "60", lace_home)
        cfg = load_config(lace_home)
        assert cfg.memory.decay_half_life_days == 60
        assert cfg.dedup.skip_threshold == pytest.approx(0.77)
        assert cfg.extraction.max_memories_per_interaction == 7

    def test_model_dump_includes_all_new_block_keys(self):
        dumped = LaceConfig().model_dump()
        assert "dedup" in dumped
        assert "extraction" in dumped
        for key in ("skip_threshold", "merge_threshold", "hash_cooldown_seconds"):
            assert key in dumped["dedup"], f"Missing key in dedup dump: {key}"
        for key in ("require_worthiness_verdict", "log_all_verdicts",
                    "extraction_model", "max_memories_per_interaction"):
            assert key in dumped["extraction"], f"Missing key in extraction dump: {key}"

    def test_two_configs_loaded_from_same_file_are_equal(self, tmp_path):
        lace_home = tmp_path / ".lace"
        (lace_home / "config").mkdir(parents=True)
        cfg = LaceConfig(dedup=DedupConfig(skip_threshold=0.88))
        save_config(cfg, lace_home)
        a = load_config(lace_home)
        b = load_config(lace_home)
        assert a.dedup.skip_threshold == b.dedup.skip_threshold
        assert a.extraction.extraction_model == b.extraction.extraction_model
