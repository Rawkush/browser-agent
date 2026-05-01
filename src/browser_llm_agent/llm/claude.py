"""
Claude API backend for rawagent.

Uses the Anthropic Python SDK with native tool calling — no browser,
no text parsing. Requires ANTHROPIC_API_KEY in the environment.
"""

import os
import anthropic

# ── System prompt (no tool-format instructions — Claude uses native calling) ──

SYSTEM_PROMPT = """You are a coding assistant with file and shell access.

Debugging methodology — follow this for every bug:

  1. OBSERVE   Run list_dir to understand the project layout. Then read the
               relevant files. If the bug involves runtime behaviour, use bash
               to capture actual output before reading anything else.

  2. HYPOTHESIZE  State a specific theory: "The bug is X because Y."
               Do not proceed without a theory grounded in observed evidence.

  3. EXPERIMENT  Use bash to verify or falsify the theory cheaply before
               changing production code. Print the suspect value. Run a
               minimal reproduction. Check a log file.

  4. FIX       Apply one targeted edit_file change. Keep it minimal.

  5. VERIFY    Use bash to confirm the fix works. Re-run the reproduction
               from step 3 or run the test suite.

Auto-detection rules — NEVER ask; detect instead:
- Framework/language: read package.json, pyproject.toml, go.mod, Cargo.toml,
  pom.xml, requirements.txt, or look at file extensions.
- Entry point: check common names (index.ts, main.py, app.py, server.ts,
  main.go, src/main.rs) with list_dir or bash find.
- Routes/handlers: use bash grep for "app.get", "router.", "@app.route", etc.
- Any other project detail: use bash, list_dir, or read_file to discover it.

Absolute prohibitions — NEVER output any of these phrases or patterns:
- "What framework are you using?"
- "What language is this?"
- "What file is X in?" / "Which block should I look at?"
- "Would you like me to look at X?" / "Should I check X?"
- "Do you want me to proceed?" / "Shall I apply this fix?"
- "Can you run X and tell me the output?" / "Check your console"
- "What does the log show?" / "Could you share X?" / "Please provide X"
- Any question whose answer you can obtain with a tool.

Core rules:
- NEVER ask for permission before calling a tool — call it immediately.
- NEVER offer to do something — just do it.
- NEVER ask the user to fetch information you can get yourself.
- Do not ask clarifying questions. Make the most reasonable assumption and act.
- Always read a file before editing it.
- After making a fix, confirm by reading the edited section back.
- NEVER just describe a fix in text — execute it with edit_file or write_file.
- Keep responses concise. Show full file content only when strictly necessary.
"""

# ── Tool definitions in Anthropic format ──────────────────────────────────────

TOOLS: list[dict] = [
    {
        "name": "bash",
        "description": "Run a shell command and return its stdout + stderr.",
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "Shell command to run"},
                "cwd": {"type": "string", "description": "Working directory (optional)"},
            },
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a file and return its contents with line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute or relative file path"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Create or completely overwrite a file with new content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "content": {"type": "string"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace an exact string in a file. old must be unique in the file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old": {"type": "string", "description": "Exact text to replace"},
                "new": {"type": "string", "description": "Replacement text"},
            },
            "required": ["path", "old", "new"],
        },
    },
    {
        "name": "list_dir",
        "description": "List the contents of a directory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path (default: .)"},
            },
            "required": [],
        },
    },
    {
        "name": "glob",
        "description": "Find files matching a glob pattern.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Glob pattern e.g. **/*.ts"},
                "cwd": {"type": "string", "description": "Search root (optional)"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "grep",
        "description": "Search for a regex pattern across files.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {"type": "string"},
                "path": {"type": "string", "description": "Directory or file to search"},
                "include": {"type": "string", "description": "File glob filter e.g. *.py"},
                "ignore_case": {"type": "boolean"},
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "web_fetch",
        "description": "Fetch a URL and return its content as plain text.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string"},
            },
            "required": ["url"],
        },
    },
    {
        "name": "find_files",
        "description": "Find files or directories by exact name.",
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "File or directory name to find"},
                "path": {"type": "string", "description": "Search root (default: .)"},
                "file_type": {"type": "string", "description": "'file' or 'dir' (optional)"},
            },
            "required": ["name"],
        },
    },
    {
        "name": "delete_file",
        "description": "Delete a file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "move_file",
        "description": "Move or rename a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "src": {"type": "string"},
                "dst": {"type": "string"},
            },
            "required": ["src", "dst"],
        },
    },
    {
        "name": "make_dir",
        "description": "Create a directory (including parents).",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "todo_add",
        "description": "Add a todo item.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {"type": "string"},
                "priority": {"type": "string", "enum": ["high", "medium", "low"]},
            },
            "required": ["content"],
        },
    },
    {
        "name": "todo_list",
        "description": "List todos, optionally filtered by status.",
        "input_schema": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["pending", "in_progress", "done"]},
            },
            "required": [],
        },
    },
    {
        "name": "todo_update",
        "description": "Update the status of a todo item.",
        "input_schema": {
            "type": "object",
            "properties": {
                "todo_id": {"type": "string"},
                "status": {"type": "string", "enum": ["pending", "in_progress", "done"]},
            },
            "required": ["todo_id", "status"],
        },
    },
    {
        "name": "todo_delete",
        "description": "Delete a todo item.",
        "input_schema": {
            "type": "object",
            "properties": {"todo_id": {"type": "string"}},
            "required": ["todo_id"],
        },
    },
    {
        "name": "memory_save",
        "description": "Save a persistent key/value memory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {"type": "string"},
                "value": {"type": "string"},
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "memory_get",
        "description": "Retrieve a memory by key.",
        "input_schema": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
    {
        "name": "memory_list",
        "description": "List all saved memories.",
        "input_schema": {"type": "object", "properties": {}, "required": []},
    },
    {
        "name": "memory_search",
        "description": "Search memories by keyword.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "memory_delete",
        "description": "Delete a memory by key.",
        "input_schema": {
            "type": "object",
            "properties": {"key": {"type": "string"}},
            "required": ["key"],
        },
    },
]


# ── Client factory ─────────────────────────────────────────────────────────────

def create_client(api_key: str | None = None) -> anthropic.Anthropic:
    """Return an Anthropic client. Uses ANTHROPIC_API_KEY env var by default."""
    return anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))


def tools_with_mcp(mcp_manager) -> list[dict]:
    """Return TOOLS plus any tools exposed by connected MCP servers."""
    extra = []
    if mcp_manager and mcp_manager.has_tools():
        for server_name, tool in mcp_manager.all_tools():
            namespaced = f"{server_name}__{tool.name}"
            schema = getattr(tool, "inputSchema", None) or {"type": "object", "properties": {}, "required": []}
            extra.append({
                "name": namespaced,
                "description": (tool.description or "").strip(),
                "input_schema": schema,
            })
    return TOOLS + extra
