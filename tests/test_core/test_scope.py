"""Tests for scope management."""

import pytest
from pathlib import Path
from lace.core.scope import (
    validate_scope,
    normalize_scope,
    detect_current_project,
    get_active_scope,
)


def test_validate_scope_global():
    assert validate_scope("global") is True


def test_validate_scope_project():
    assert validate_scope("project:my-api") is True
    assert validate_scope("project:My API") is True


def test_validate_scope_session():
    assert validate_scope("session:abc123") is True


def test_validate_scope_invalid():
    assert validate_scope("invalid") is False
    assert validate_scope("") is False
    assert validate_scope("project:") is False
    assert validate_scope("project:../escape") is False
    assert validate_scope("project:subfolder/escape") is False
    assert validate_scope("project:subfolder\\escape") is False
    assert validate_scope("session:../escape") is False


def test_normalize_scope_global():
    assert normalize_scope("global") == "global"


def test_normalize_scope_project():
    assert normalize_scope("my-api") == "project:my-api"
    assert normalize_scope("project:my-api") == "project:my-api"


def test_normalize_scope_session():
    assert normalize_scope("session_abc123") == "session:abc123"
    assert normalize_scope("session:abc123") == "session:abc123"


def test_detect_current_project_not_in_repo(tmp_path):
    """Return None when not in a Git repo."""
    result = detect_current_project(tmp_path)
    assert result is None


def test_get_active_scope_global(tmp_path):
    """Return global when no project or session detected."""
    lace_home = tmp_path / ".lace"
    result = get_active_scope(lace_home, tmp_path)
    assert result == "global"