"""
Git tools with safety guardrails.

Design choices:
- No git_push tool — push stays in bash so the user sees it explicitly.
- git_commit never amends or skips hooks.
- git_branch refuses to delete main/master.
"""

import subprocess

from browser_llm_agent.tools.registry import tool

_PROTECTED_BRANCHES = {"main", "master"}


def _git(args: list[str], cwd: str = ".") -> str:
    """Run a git command and return output."""
    try:
        result = subprocess.run(
            ["git"] + args,
            capture_output=True, text=True, timeout=30, cwd=cwd,
        )
        output = result.stdout.strip()
        if result.returncode != 0:
            err = result.stderr.strip()
            return f"Error (exit {result.returncode}): {err or output}"
        return output or "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: git command timed out"
    except Exception as e:
        return f"Error: {e}"


def _truncate(text: str, max_chars: int = 8000) -> str:
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    omitted = len(text) - max_chars
    return f"{text[:half]}\n\n... ({omitted} chars omitted) ...\n\n{text[-half:]}"


@tool("git_status", "Show git status: current branch, staged/unstaged/untracked files.", {
    "type": "object",
    "properties": {
        "cwd": {"type": "string", "description": "Repository directory (default: .)"},
    },
    "required": [],
})
def git_status(cwd: str = ".") -> str:
    branch = _git(["branch", "--show-current"], cwd)
    status = _git(["status", "--short"], cwd)
    return f"Branch: {branch}\n\n{status}"


@tool("git_diff", "Show git diff. target: 'staged', 'unstaged' (default), a branch/commit, or a file path.", {
    "type": "object",
    "properties": {
        "cwd": {"type": "string", "description": "Repository directory (default: .)"},
        "target": {"type": "string", "description": "'staged', 'unstaged', branch name, or commit hash"},
        "path": {"type": "string", "description": "Optional file path filter"},
    },
    "required": [],
})
def git_diff(cwd: str = ".", target: str = "unstaged", path: str = None) -> str:
    args = ["diff"]
    if target == "staged":
        args.append("--staged")
    elif target != "unstaged":
        args.append(target)
    if path:
        args += ["--", path]
    return _truncate(_git(args, cwd))


@tool("git_log", "Show recent git commits.", {
    "type": "object",
    "properties": {
        "cwd": {"type": "string", "description": "Repository directory (default: .)"},
        "count": {"type": "integer", "description": "Number of commits to show (default: 10)"},
        "oneline": {"type": "boolean", "description": "Compact one-line format (default: true)"},
    },
    "required": [],
})
def git_log(cwd: str = ".", count: int = 10, oneline: bool = True) -> str:
    count = min(max(count, 1), 100)
    if oneline:
        return _git(["log", f"-{count}", "--oneline"], cwd)
    return _git(["log", f"-{count}", "--format=%h %s (%an, %ar)"], cwd)


@tool("git_commit", "Stage files and create a git commit. Never amends. Never skips hooks.", {
    "type": "object",
    "properties": {
        "message": {"type": "string", "description": "Commit message"},
        "cwd": {"type": "string", "description": "Repository directory (default: .)"},
        "files": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Files to stage before committing. If empty, commits whatever is already staged.",
        },
    },
    "required": ["message"],
})
def git_commit(message: str, cwd: str = ".", files: list[str] = None) -> str:
    # Stage specific files if provided
    if files:
        for f in files:
            result = _git(["add", f], cwd)
            if result.startswith("Error"):
                return result

    # Check there's something to commit
    status = _git(["diff", "--staged", "--name-only"], cwd)
    if status == "(no output)" or status.startswith("Error"):
        return "Error: nothing staged to commit. Provide files to stage, or stage manually."

    return _git(["commit", "-m", message], cwd)


@tool("git_branch", "List, create, switch, or delete branches.", {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["list", "create", "switch", "delete"],
            "description": "Branch action",
        },
        "branch_name": {"type": "string", "description": "Branch name (required for create/switch/delete)"},
        "cwd": {"type": "string", "description": "Repository directory (default: .)"},
    },
    "required": ["action"],
})
def git_branch(action: str, branch_name: str = None, cwd: str = ".") -> str:
    if action == "list":
        return _git(["branch", "-a"], cwd)

    if not branch_name:
        return f"Error: branch_name required for '{action}'"

    if action == "create":
        return _git(["checkout", "-b", branch_name], cwd)
    elif action == "switch":
        return _git(["checkout", branch_name], cwd)
    elif action == "delete":
        if branch_name in _PROTECTED_BRANCHES:
            return f"Error: refusing to delete protected branch '{branch_name}'"
        return _git(["branch", "-d", branch_name], cwd)
    else:
        return f"Error: unknown action '{action}'. Use: list, create, switch, delete"


@tool("git_stash", "Stash or restore uncommitted changes.", {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["push", "pop", "list", "drop"],
            "description": "Stash action",
        },
        "cwd": {"type": "string", "description": "Repository directory (default: .)"},
        "message": {"type": "string", "description": "Stash message (for push)"},
    },
    "required": ["action"],
})
def git_stash(action: str, cwd: str = ".", message: str = None) -> str:
    if action == "push":
        args = ["stash", "push"]
        if message:
            args += ["-m", message]
        return _git(args, cwd)
    elif action == "pop":
        return _git(["stash", "pop"], cwd)
    elif action == "list":
        return _git(["stash", "list"], cwd)
    elif action == "drop":
        return _git(["stash", "drop"], cwd)
    else:
        return f"Error: unknown action '{action}'. Use: push, pop, list, drop"
