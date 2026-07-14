"""Configuration management for LACE."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings


# ── Path resolution ──────────────────────────────────────────────────────────

def get_lace_home() -> Path:
    """Return the LACE home directory.
    
    Checks LACE_HOME env var first, falls back to ~/.lace
    """
    env_home = os.environ.get("LACE_HOME")
    if env_home:
        return Path(env_home).expanduser().resolve()
    return Path.home() / ".lace"


def get_config_dir() -> Path:
    """Return the default config templates directory (inside the package)."""
    return Path(__file__).parent.parent.parent.parent / "config"


def resolve_lace_paths(lace_home: Path | None = None) -> dict[str, Path]:
    """Return all paths LACE uses, resolved from a single source of truth.

    This is the canonical path resolver. No module should compute its own
    default path independently — all call sites should use this function.

    Parameters
    ----------
    lace_home:
        The LACE home directory. Defaults to ``get_lace_home()`` if not given.

    Returns
    -------
    dict with keys:
        vault         — markdown memory files
        vector_db     — ChromaDB persistent store
        queue_db      — extraction_queue.db (SQLite)
        pipeline_log  — pipeline_log.db (SQLite)
        hash_index    — vault_hash_index.db (SQLite)
        co_retrieval  — co_retrieval.json
        config_file   — lace.yaml
        graph         — graph.json
    """
    if lace_home is None:
        lace_home = get_lace_home()
    return {
        "vault":         lace_home / "memory" / "vault",
        "vector_db":     lace_home / "memory" / "vector_db",
        "queue_db":      lace_home / "queue" / "extraction_queue.db",
        "pipeline_log":  lace_home / "queue" / "pipeline_log.db",
        "hash_index":    lace_home / "memory" / "vault_hash_index.db",
        "co_retrieval":  lace_home / "memory" / "co_retrieval.json",
        "config_file":   lace_home / "config" / "lace.yaml",
        "graph":         lace_home / "memory" / "graph.json",
    }


# ── Config models ─────────────────────────────────────────────────────────────

class MemoryConfig(BaseModel):
    auto_extract: bool = False
    extraction_threshold: float = 0.6
    require_confirmation: bool = False
    max_extractions_per_turn: int = 3
    dedup_threshold: float = 0.85
    decay_half_life_days: int = 30
    consolidation_schedule: str = "weekly"


class RetrievalWeights(BaseModel):
    vector: float = 0.45
    tag: float = 0.15
    graph: float = 0.15
    co_retrieval: float = 0.10
    recency: float = 0.10
    confidence: float = 0.05

    @model_validator(mode="before")
    @classmethod
    def migrate_old_keys(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Check if any old keys exist and map them
            if "semantic_similarity" in data:
                data["vector"] = data.pop("semantic_similarity")
            if "frequency" in data:
                data["co_retrieval"] = data.pop("frequency")
            if "scope" in data:
                scope_val = data.pop("scope")
                # Split scope value between tag and graph
                data["tag"] = scope_val / 2.0
                data["graph"] = scope_val / 2.0
        return data

    @model_validator(mode="after")
    def validate_sum(self) -> RetrievalWeights:
        total = (
            self.vector + self.tag + self.graph
            + self.co_retrieval + self.recency + self.confidence
        )
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Retrieval weights must sum to 1.0, got {total:.3f}")
        return self


class RetrievalConfig(BaseModel):
    relevance_threshold: float = 0.35
    max_results: int = 20
    weights: RetrievalWeights = Field(default_factory=RetrievalWeights)


class VaultConfig(BaseModel):
    obsidian_compatible: bool = True
    file_watcher: bool = False
    path: str | None = None  # None = use default ~/.lace/memory/vault


class LoggingConfig(BaseModel):
    retrieval_logs: bool = True
    interaction_logs: bool = True
    log_retention_days: int = 90


class EmbeddingsConfig(BaseModel):
    provider: str = "local"              # "local" or "openai"
    model: str = "all-MiniLM-L6-v2"     # local default

class OllamaProviderConfig(BaseModel):
    host: str = "http://localhost:11434"
    model: str = "llama3.2"
    temperature: float = 0.7
    context_window: int = 8192


class OpenAIProviderConfig(BaseModel):
    model: str = "gpt-4o"
    temperature: float = 0.7
    context_window: int = 128000


class AnthropicProviderConfig(BaseModel):
    model: str = "claude-3-5-sonnet-20241022"
    temperature: float = 0.7
    context_window: int = 200000


class ProviderConfig(BaseModel):
    default: str = "ollama"
    ollama: OllamaProviderConfig = Field(default_factory=OllamaProviderConfig)
    openai: OpenAIProviderConfig = Field(default_factory=OpenAIProviderConfig)
    anthropic: AnthropicProviderConfig = Field(default_factory=AnthropicProviderConfig)


class DedupConfig(BaseModel):
    """
    Controls deduplication behavior at two levels:
    1. Queue-level hash suppression (before LLM extraction)
    2. Vault-level semantic dedup (after extraction)

    Thresholds:
    - skip_threshold:  cosine sim above this → discard candidate entirely
    - merge_threshold: cosine sim above this → merge into existing memory
    - below merge_threshold → store as new memory
    """
    skip_threshold: float = Field(
        default=0.95,
        ge=0.0,
        le=1.0,
        description="Cosine similarity above which a candidate is a duplicate and discarded",
    )
    merge_threshold: float = Field(
        default=0.85,
        ge=0.0,
        le=1.0,
        description="Cosine similarity above which a candidate is merged into existing",
    )
    hash_cooldown_seconds: int = Field(
        default=300,
        ge=0,
        description="Seconds within which identical canonical hashes are suppressed at queue insert",
    )

    def validate_thresholds(self) -> None:
        """Ensure merge < skip — catches misconfiguration early."""
        if self.merge_threshold >= self.skip_threshold:
            raise ValueError(
                f"merge_threshold ({self.merge_threshold}) must be "
                f"less than skip_threshold ({self.skip_threshold})"
            )


class ExtractionConfig(BaseModel):
    """
    Controls the LLM extraction pipeline behavior.

    require_worthiness_verdict: if True, every item must get a
    worth_remembering judgment before any memory is stored.

    log_all_verdicts: if True, even rejected items are written
    to pipeline_log.db so you can audit what got filtered.
    """
    require_worthiness_verdict: bool = Field(
        default=True,
        description="Require LLM to judge worth_remembering before extracting",
    )
    log_all_verdicts: bool = Field(
        default=True,
        description="Log all LLM verdicts including rejections to pipeline_log.db",
    )
    extraction_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model used for extraction",
    )
    max_memories_per_interaction: int = Field(
        default=5,
        ge=1,
        description="Maximum number of memories extracted from a single interaction",
    )

class LaceConfig(BaseModel):
    """Root configuration model for LACE."""
    version: str = "1.0"
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)
    vault: VaultConfig = Field(default_factory=VaultConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    embeddings: EmbeddingsConfig = Field(default_factory=EmbeddingsConfig)
    provider: ProviderConfig = Field(default_factory=ProviderConfig)
    dedup: DedupConfig = Field(
        default_factory=DedupConfig,
        description="Deduplication configuration",
    )
    extraction: ExtractionConfig = Field(
        default_factory=ExtractionConfig,
        description="Extraction pipeline configuration",
    )

    def vault_path(self, lace_home: Path) -> Path:
        """Resolve the vault path."""
        if self.vault.path:
            return Path(self.vault.path).expanduser().resolve()
        return lace_home / "memory" / "vault"


# ── Config loader ─────────────────────────────────────────────────────────────

def load_config(lace_home: Path | None = None, cwd: Path | str | None = None) -> LaceConfig:
    """Load configuration from ~/.lace/config/lace.yaml, then merge project.yaml overrides.
    
    Falls back to defaults if file doesn't exist.
    """
    if lace_home is None:
        lace_home = get_lace_home()

    config_file = lace_home / "config" / "lace.yaml"

    if not config_file.exists():
        config = LaceConfig()
    else:
        with open(config_file) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
        config = LaceConfig.model_validate(raw)

    # Walk up from cwd to find .lace/project.yaml
    if cwd is None:
        cwd = os.getcwd()
    cwd_path = Path(cwd).resolve()
    project_file = None
    for parent in [cwd_path] + list(cwd_path.parents):
        candidate = parent / ".lace" / "project.yaml"
        if candidate.exists():
            project_file = candidate
            break

    if project_file:
        try:
            with open(project_file) as f:
                proj_data = yaml.safe_load(f) or {}
            
            # Apply noise profile if set
            noise_profile = (
                proj_data.get("extraction", {}).get("noise_profile") or
                proj_data.get("dedup", {}).get("noise_profile")
            )
            if noise_profile == "low":
                config.dedup.merge_threshold = 0.80
                config.dedup.skip_threshold = 0.90
                config.dedup.hash_cooldown_seconds = 600
            elif noise_profile == "medium":
                config.dedup.merge_threshold = 0.85
                config.dedup.skip_threshold = 0.95
                config.dedup.hash_cooldown_seconds = 300
            elif noise_profile == "high":
                config.dedup.merge_threshold = 0.90
                config.dedup.skip_threshold = 0.98
                config.dedup.hash_cooldown_seconds = 900

            # Apply individual overrides
            ext_data = proj_data.get("extraction", {})
            if "require_worthiness_verdict" in ext_data:
                config.extraction.require_worthiness_verdict = bool(ext_data["require_worthiness_verdict"])

            dedup_data = proj_data.get("dedup", {})
            if "skip_threshold" in dedup_data:
                config.dedup.skip_threshold = float(dedup_data["skip_threshold"])
            if "merge_threshold" in dedup_data:
                config.dedup.merge_threshold = float(dedup_data["merge_threshold"])
            if "hash_cooldown_seconds" in dedup_data:
                config.dedup.hash_cooldown_seconds = int(dedup_data["hash_cooldown_seconds"])
        except Exception as e:
            import logging as _log
            _log.getLogger("lace.core.config").warning(
                f"Failed to load project config overrides from {project_file}: {e}",
                exc_info=True,
            )

    return config


def save_config(config: LaceConfig, lace_home: Path | None = None) -> None:
    """Write config back to lace.yaml."""
    if lace_home is None:
        lace_home = get_lace_home()

    config_file = lace_home / "config" / "lace.yaml"
    config_file.parent.mkdir(parents=True, exist_ok=True)

    with open(config_file, "w") as f:
        yaml.dump(config.model_dump(), f, default_flow_style=False, sort_keys=False)


def set_config_value(key_path: str, value: str, lace_home: Path | None = None) -> None:
    """Set a nested config value using dot notation.
    
    Example: set_config_value("memory.decay_half_life_days", "60")
    """
    if lace_home is None:
        lace_home = get_lace_home()

    config = load_config(lace_home)
    data = config.model_dump()

    keys = key_path.split(".")
    target = data
    for key in keys[:-1]:
        if key not in target:
            raise KeyError(f"Unknown config key: {key_path}")
        target = target[key]

    final_key = keys[-1]
    if final_key not in target:
        raise KeyError(f"Unknown config key: {key_path}")

    # Type coercion — preserve the original type
    original = target[final_key]
    if isinstance(original, bool):
        target[final_key] = value.lower() in ("true", "1", "yes")
    elif isinstance(original, int):
        target[final_key] = int(value)
    elif isinstance(original, float):
        target[final_key] = float(value)
    else:
        target[final_key] = value

    updated = LaceConfig.model_validate(data)
    save_config(updated, lace_home)


# ── Init system ───────────────────────────────────────────────────────────────

VAULT_SUBDIRS = [
    "global/patterns",
    "global/decisions",
    "global/debug-log",
    "global/references",
    "projects",
]

OTHER_DIRS = [
    "logs/retrieval",
    "logs/interactions",
    "sessions",
]


def init_lace_home(lace_home: Path | None = None) -> tuple[Path, bool]:
    """Create the ~/.lace directory structure.
    
    Returns (lace_home_path, was_already_initialized).
    """
    if lace_home is None:
        lace_home = get_lace_home()

    already_existed = lace_home.exists()

    # Create all directories
    for subdir in VAULT_SUBDIRS:
        (lace_home / "memory" / "vault" / subdir).mkdir(parents=True, exist_ok=True)

    for subdir in OTHER_DIRS:
        (lace_home / subdir).mkdir(parents=True, exist_ok=True)

    (lace_home / "config" / "projects").mkdir(parents=True, exist_ok=True)

    # Copy default config templates (only if they don't already exist)
    templates_dir = get_config_dir()
    config_dest = lace_home / "config"

    for template_file in ["lace.yaml", "identity.md", "preferences.yaml"]:
        src = templates_dir / template_file
        dst = config_dest / template_file
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    return lace_home, already_existed