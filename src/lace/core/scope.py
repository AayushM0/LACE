"""Scope management for LACE — project detection and session tracking.

This module handles:
  1. Auto-detecting current project from working directory
  2. Session creation and tracking (temporary memory)
  3. Scope validation and normalization
  4. Active scope storage in ~/.lace/sessions/active
"""

from __future__ import annotations

import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from git import Repo


# ── Session management ─────────────────────────────────────────────────────────

def get_active_session(lace_home: Path | None = None) -> str | None:
    """Return the current active session ID.

    Session ID is stored in ~/.lace/sessions/active
    If file doesn't exist or is empty, returns None.
    """
    if lace_home is None:
        from lace.core.config import get_lace_home
        lace_home = get_lace_home()

    session_file = lace_home / "sessions" / "active"
    if not session_file.exists():
        return None

    try:
        return session_file.read_text().strip()
    except Exception:
        return None


def set_active_session(session_id: str, lace_home: Path | None = None) -> None:
    """Set the current active session ID."""
    if lace_home is None:
        from lace.core.config import get_lace_home
        lace_home = get_lace_home()

    session_dir = lace_home / "sessions"
    session_dir.mkdir(parents=True, exist_ok=True)
    session_file = session_dir / "active"
    session_file.write_text(session_id)


def create_new_session(lace_home: Path | None = None) -> str:
    """Create a new session ID and set it as active."""
    session_id = f"session_{uuid.uuid4().hex[:12]}"
    set_active_session(session_id, lace_home)
    return session_id


def get_session_path(session_id: str, lace_home: Path | None = None) -> Path:
    """Return the path to a session's directory."""
    if lace_home is None:
        from lace.core.config import get_lace_home
        lace_home = get_lace_home()
    return lace_home / "sessions" / session_id


# ── Project detection ──────────────────────────────────────────────────────────

def find_git_root(start_path: Path | str = os.getcwd()) -> Path | None:
    """Find the root of the current Git repository.

    Returns None if not in a Git repo.
    """
    from git import Repo, InvalidGitRepositoryError, NoSuchPathError

    try:
        repo = Repo(start_path, search_parent_directories=True)
        return Path(repo.working_tree_dir)
    except (InvalidGitRepositoryError, NoSuchPathError):
        return None


def get_project_name_from_git(root: Path) -> str | None:
    """Extract project name from git repo.

    Uses directory name of repo root.
    Remote URL parsing removed — unreliable for
    project identification when remote name differs
    from local directory name.
    """
    return root.name


def detect_current_project(
    cwd: Path | str = os.getcwd(),
    lace_home: Path | None = None,
) -> str | None:
    """Auto-detect the current project from the working directory.

    Returns:
        A project scope string like "project:my-api" or None if no project detected.

    Detection steps:
      1. Check if cwd is in a Git repo → use repo name
      2. Check for .lace/project.yaml in cwd or parent dirs → use name from file
      3. Fallback to None (no project detected)
    """
    if lace_home is None:
        from lace.core.config import get_lace_home
        lace_home = get_lace_home()

    # Step 1: Git repo detection
    git_root = find_git_root(cwd)
    if git_root:
        project_name = get_project_name_from_git(git_root)
        if project_name:
            return f"project:{project_name}"

    # Step 2: .lace/project.yaml detection
    cwd_path = Path(cwd).resolve()
    for parent in [cwd_path] + list(cwd_path.parents):
        project_file = parent / ".lace" / "project.yaml"
        if project_file.exists():
            try:
                import yaml
                with open(project_file) as f:
                    data = yaml.safe_load(f) or {}
                name = data.get("name")
                if name:
                    return f"project:{name}"
            except Exception:
                pass

    return None


# ── Scope validation ───────────────────────────────────────────────────────────

def validate_scope(scope: str) -> bool:
    """Return True if scope string is valid.

    Valid scopes:
      - "global"
      - "session:<id>"
      - "project:<name>"
    """
    if scope == "global":
        return True
    if scope.startswith("session:") and len(scope) > 8:
        return True
    if scope.startswith("project:") and len(scope) > 8:
        return True
    return False


def normalize_scope(scope: str) -> str:
    """Normalize scope string to standard format.

    Examples:
      - "my-api" → "project:my-api"
      - "session_abc123" → "session:abc123"
      - "global" → "global"
    """
    if validate_scope(scope):
        return scope
    if scope.startswith("session_"):
        return f"session:{scope[8:]}"
    return f"project:{scope}"


def get_active_scope(
    lace_home: Path | None = None,
    cwd: Path | str = os.getcwd(),
) -> str:
    """Return the current active scope.

    Priority:
      1. Session scope if active session exists
      2. Detected project scope
      3. "global"
    """
    session = get_active_session(lace_home)
    if session:
        return f"session:{session}"

    project = detect_current_project(cwd, lace_home)
    if project:
        return project

    return "global"


# ── Project management ─────────────────────────────────────────────────────────

def get_projects(lace_home: Path | None = None) -> list[dict]:
    """List all configured projects."""
    if lace_home is None:
        from lace.core.config import get_lace_home
        lace_home = get_lace_home()

    projects_dir = lace_home / "config" / "projects"
    projects_dir.mkdir(parents=True, exist_ok=True)

    projects: list[dict] = []

    for project_file in projects_dir.glob("*.yaml"):
        project_name = project_file.stem
        try:
            import yaml
            with open(project_file) as f:
                data = yaml.safe_load(f) or {}
            projects.append({
                "name": project_name,
                "scope": f"project:{project_name}",
                "description": data.get("description", ""),
                "created_at": data.get("created_at"),
                "last_used": data.get("last_used"),
            })
        except Exception:
            pass

    return projects


def create_project(
    name: str,
    description: str | None = None,
    lace_home: Path | None = None,
) -> bool:
    """Create a new project configuration file.

    Returns True if created, False if already exists.
    """
    if lace_home is None:
        from lace.core.config import get_lace_home
        lace_home = get_lace_home()

    projects_dir = lace_home / "config" / "projects"
    projects_dir.mkdir(parents=True, exist_ok=True)
    project_file = projects_dir / f"{name}.yaml"

    if project_file.exists():
        return False

    import yaml
    data = {
        "name": name,
        "description": description or "",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "last_used": None,
    }

    with open(project_file, "w") as f:
        yaml.dump(data, f, default_flow_style=False)

    return True


def set_project_last_used(name: str, lace_home: Path | None = None) -> None:
    """Update last_used timestamp for a project."""
    if lace_home is None:
        from lace.core.config import get_lace_home
        lace_home = get_lace_home()

    project_file = lace_home / "config" / "projects" / f"{name}.yaml"
    if not project_file.exists():
        return

    import yaml
    with open(project_file) as f:
        data = yaml.safe_load(f) or {}

    data["last_used"] = datetime.now(timezone.utc).isoformat()

    with open(project_file, "w") as f:
        yaml.dump(data, f, default_flow_style=False)