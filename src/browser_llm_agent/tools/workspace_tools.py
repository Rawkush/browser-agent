"""
Workspace-level tools for Codex-style coding workflows.

These tools complement the small file/search/bash primitives with operations
that coding agents need constantly: a compact project snapshot, batched reads,
line-range edits, appends, and multi-file unified patches.
"""

import difflib
import os
import subprocess
from pathlib import Path

from browser_llm_agent.tools.registry import tool

_SKIP_DIRS = {
    ".git",
    ".hg",
    ".svn",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    ".code-review-graph",
    "node_modules",
    "dist",
    "build",
    "target",
    "__pypackages__",
}


def _expand(path: str, cwd: str | None = None) -> str:
    path = os.path.expanduser(path)
    if cwd and not os.path.isabs(path):
        path = os.path.join(os.path.expanduser(cwd), path)
    return os.path.abspath(path)


def _make_diff(original: str, modified: str, path: str) -> str:
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    return "".join(diff)


def _numbered(path: str, offset: int = 1, limit: int = 120) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().splitlines()
    start = max(0, offset - 1)
    end = len(lines) if limit <= 0 else min(len(lines), start + limit)
    body = "\n".join(f"{i + 1}: {lines[i]}" for i in range(start, end))
    if end < len(lines):
        body += f"\n... ({len(lines) - end} more lines, {len(lines)} total)"
    return body


@tool("workspace_snapshot", "Return a compact tree of the workspace, skipping dependency/build directories.", {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Workspace root (default: .)"},
        "max_depth": {"type": "integer", "description": "Maximum directory depth (default: 3, max: 6)"},
        "max_entries": {"type": "integer", "description": "Maximum entries to return (default: 200, max: 1000)"},
    },
    "required": [],
})
def workspace_snapshot(path: str = ".", max_depth: int = 3, max_entries: int = 200) -> str:
    root = _expand(path)
    if not os.path.isdir(root):
        return f"Error: not a directory: {root}"

    max_depth = min(max(max_depth, 1), 6)
    max_entries = min(max(max_entries, 1), 1000)
    root_path = Path(root)
    lines = [f"{root_path.name}/"]
    count = 0

    for current, dirs, files in os.walk(root):
        rel = os.path.relpath(current, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        if depth >= max_depth:
            dirs[:] = []
        dirs[:] = sorted(d for d in dirs if d not in _SKIP_DIRS and not d.endswith(".egg-info"))
        files = sorted(files)

        if rel == ".":
            prefix = ""
        else:
            prefix = "  " * depth
            lines.append(f"{prefix}{os.path.basename(current)}/")
            count += 1

        if depth < max_depth:
            for filename in files:
                if count >= max_entries:
                    lines.append(f"... ({max_entries} entry limit reached)")
                    return "\n".join(lines)
                lines.append(f"{prefix}  {filename}")
                count += 1

    return "\n".join(lines)


@tool("read_many_files", "Read several files in one call with line numbers.", {
    "type": "object",
    "properties": {
        "paths": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Files to read",
        },
        "offset": {"type": "integer", "description": "Start line for every file (default: 1)"},
        "limit": {"type": "integer", "description": "Max lines per file (default: 120, 0 = all)"},
    },
    "required": ["paths"],
})
def read_many_files(paths: list, offset: int = 1, limit: int = 120) -> str:
    if not paths:
        return "Error: paths must not be empty"
    if len(paths) > 20:
        return "Error: read_many_files accepts at most 20 files"

    sections = []
    for raw_path in paths:
        path = _expand(str(raw_path))
        if not os.path.exists(path):
            sections.append(f"--- {raw_path} ---\nError: file not found")
            continue
        if not os.path.isfile(path):
            sections.append(f"--- {raw_path} ---\nError: not a file")
            continue
        try:
            sections.append(f"--- {raw_path} ---\n{_numbered(path, offset, limit)}")
        except Exception as e:
            sections.append(f"--- {raw_path} ---\nError: {e}")
    return "\n\n".join(sections)


@tool("replace_lines", "Replace an inclusive 1-based line range in a file. Returns unified diff.", {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "start_line": {"type": "integer", "description": "First line to replace, 1-based"},
        "end_line": {"type": "integer", "description": "Last line to replace, inclusive"},
        "content": {"type": "string", "description": "Replacement text"},
    },
    "required": ["path", "start_line", "end_line", "content"],
})
def replace_lines(path: str, start_line: int, end_line: int, content: str) -> str:
    path = _expand(path)
    if not os.path.exists(path):
        return f"Error: file not found: {path}"
    if start_line < 1 or end_line < start_line:
        return "Error: invalid line range"

    with open(path, "r", encoding="utf-8", errors="replace") as f:
        original = f.read()
    lines = original.splitlines(keepends=True)
    if end_line > len(lines):
        return f"Error: end_line {end_line} exceeds file length {len(lines)}"

    replacement = content
    if replacement and not replacement.endswith("\n"):
        replacement += "\n"
    replacement_lines = replacement.splitlines(keepends=True)
    updated = "".join(lines[:start_line - 1] + replacement_lines + lines[end_line:])

    with open(path, "w", encoding="utf-8") as f:
        f.write(updated)

    diff = _make_diff(original, updated, path)
    return f"Replaced lines {start_line}-{end_line} in {path}\n{diff}"


@tool("append_file", "Append content to a file, creating it if needed. Returns unified diff.", {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "content": {"type": "string"},
    },
    "required": ["path", "content"],
})
def append_file(path: str, content: str) -> str:
    path = _expand(path)
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    original = ""
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            original = f.read()

    addition = content
    if addition and not addition.endswith("\n"):
        addition += "\n"
    updated = original + addition

    with open(path, "w", encoding="utf-8") as f:
        f.write(updated)

    diff = _make_diff(original, updated, path)
    return f"Appended to {path}\n{diff}"


@tool("apply_patch", "Apply a standard unified diff patch across one or more files using git apply.", {
    "type": "object",
    "properties": {
        "patch": {"type": "string", "description": "Unified diff patch text"},
        "cwd": {"type": "string", "description": "Working directory (default: .)"},
    },
    "required": ["patch"],
})
def apply_patch(patch: str, cwd: str = ".") -> str:
    cwd = _expand(cwd)
    if not os.path.isdir(cwd):
        return f"Error: cwd is not a directory: {cwd}"
    if not patch.strip():
        return "Error: patch is empty"

    check = subprocess.run(
        ["git", "apply", "--check", "-"],
        input=patch,
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=30,
    )
    if check.returncode != 0:
        return f"Error: patch check failed\n{check.stderr.strip() or check.stdout.strip()}"

    applied = subprocess.run(
        ["git", "apply", "-"],
        input=patch,
        capture_output=True,
        text=True,
        cwd=cwd,
        timeout=30,
    )
    if applied.returncode != 0:
        return f"Error: patch apply failed\n{applied.stderr.strip() or applied.stdout.strip()}"

    return "Patch applied."
