import difflib
import os
import re

from browser_llm_agent.tools.registry import tool


def _make_diff(original: str, modified: str, path: str) -> str:
    """Generate a unified diff between original and modified content."""
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile=f"a/{path}",
        tofile=f"b/{path}",
    )
    return "".join(diff)


@tool("read_file", "Read a file with line numbers. Use offset/limit for large files.", {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Absolute or relative file path"},
        "offset": {"type": "integer", "description": "Start from this line number (1-based, default: 1)"},
        "limit": {"type": "integer", "description": "Max lines to return (0 = all, default: 0)"},
    },
    "required": ["path"],
})
def read_file(path: str, offset: int = 1, limit: int = 0) -> str:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return f"Error: file not found: {path}"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    lines = content.splitlines()
    total = len(lines)

    # Apply offset (1-based)
    start = max(0, offset - 1)
    if limit > 0:
        end = start + limit
    else:
        end = total

    selected = lines[start:end]
    numbered = "\n".join(f"{start + i + 1}: {line}" for i, line in enumerate(selected))

    if end < total:
        numbered += f"\n\n... ({total - end} more lines, {total} total)"

    return numbered


@tool("write_file", "Create or completely overwrite a file with new content.", {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "content": {"type": "string"},
    },
    "required": ["path", "content"],
})
def write_file(path: str, content: str) -> str:
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Written: {path}"


@tool("edit_file", "Replace an exact string in a file. old must be unique. Returns unified diff.", {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "old": {"type": "string", "description": "Exact text to replace"},
        "new": {"type": "string", "description": "Replacement text"},
    },
    "required": ["path", "old", "new"],
})
def edit_file(path: str, old: str, new: str) -> str:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return f"Error: file not found: {path}"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if old not in content:
        return f"Error: string not found in {path}"
    updated = content.replace(old, new, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(updated)
    diff = _make_diff(content, updated, path)
    return f"Edited: {path}\n{diff}" if diff else f"Edited: {path}"


@tool("multi_edit", "Apply multiple edits to a file atomically. All old strings must exist or none are applied.", {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "edits": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "old": {"type": "string"},
                    "new": {"type": "string"},
                },
                "required": ["old", "new"],
            },
            "description": "List of {old, new} replacement pairs",
        },
    },
    "required": ["path", "edits"],
})
def multi_edit(path: str, edits: list) -> str:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return f"Error: file not found: {path}"
    with open(path, "r", encoding="utf-8") as f:
        original = f.read()

    # Verify ALL old strings exist before applying any
    for i, edit in enumerate(edits):
        old = edit.get("old", "")
        if old not in original:
            return f"Error: edit #{i+1} string not found in {path}: {old[:60]}..."

    # Apply edits sequentially
    content = original
    for edit in edits:
        content = content.replace(edit["old"], edit["new"], 1)

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    diff = _make_diff(original, content, path)
    return f"Applied {len(edits)} edits to {path}\n{diff}" if diff else f"Applied {len(edits)} edits to {path}"


@tool("regex_edit", "Replace text matching a regex pattern in a file. Returns unified diff.", {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "pattern": {"type": "string", "description": "Regex pattern to match"},
        "replacement": {"type": "string", "description": "Replacement string (supports \\1 backrefs)"},
        "count": {"type": "integer", "description": "Max replacements (0 = all, default: 0)"},
    },
    "required": ["path", "pattern", "replacement"],
})
def regex_edit(path: str, pattern: str, replacement: str, count: int = 0) -> str:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return f"Error: file not found: {path}"
    with open(path, "r", encoding="utf-8") as f:
        original = f.read()

    try:
        updated = re.sub(pattern, replacement, original, count=count)
    except re.error as e:
        return f"Error: invalid regex: {e}"

    if updated == original:
        return f"No matches for pattern in {path}"

    with open(path, "w", encoding="utf-8") as f:
        f.write(updated)

    diff = _make_diff(original, updated, path)
    return f"Regex edited: {path}\n{diff}" if diff else f"Regex edited: {path}"


@tool("insert_at", "Insert text before a specific line number in a file.", {
    "type": "object",
    "properties": {
        "path": {"type": "string"},
        "line": {"type": "integer", "description": "Line number to insert before (1-based)"},
        "content": {"type": "string", "description": "Text to insert"},
    },
    "required": ["path", "line", "content"],
})
def insert_at(path: str, line: int, content: str) -> str:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return f"Error: file not found: {path}"
    with open(path, "r", encoding="utf-8") as f:
        original = f.read()

    lines = original.splitlines(keepends=True)
    # Clamp to valid range
    idx = max(0, min(line - 1, len(lines)))

    # Ensure content ends with newline for clean insertion
    if content and not content.endswith("\n"):
        content += "\n"

    new_lines = lines[:idx] + [content] + lines[idx:]
    updated = "".join(new_lines)

    with open(path, "w", encoding="utf-8") as f:
        f.write(updated)

    diff = _make_diff(original, updated, path)
    return f"Inserted at line {line} in {path}\n{diff}" if diff else f"Inserted at line {line} in {path}"


@tool("list_dir", "List the contents of a directory.", {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Directory path (default: .)"},
    },
    "required": [],
})
def list_dir(path: str = ".") -> str:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return f"Error: path not found: {path}"
    entries = []
    for item in sorted(os.listdir(path)):
        full = os.path.join(path, item)
        prefix = "d" if os.path.isdir(full) else "f"
        entries.append(f"[{prefix}] {item}")
    return "\n".join(entries) if entries else "(empty)"
