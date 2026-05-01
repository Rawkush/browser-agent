#!/usr/bin/env python3
"""
rawagent MCP server — exposes rawagent tools to Claude Code CLI.

Add to Claude Code once:
    claude mcp add rawagent -- python -m browser_llm_agent.mcp_server

Then every `claude` session has rawagent's tools available automatically.
"""

from mcp.server.fastmcp import FastMCP

from browser_llm_agent.tools.bash_tools import run_bash
from browser_llm_agent.tools.file_tools import read_file, write_file, edit_file, list_dir
from browser_llm_agent.tools.search_tools import (
    glob, grep, web_fetch, find_files, delete_file, move_file, make_dir
)
from browser_llm_agent.tools.todo_tools import todo_add, todo_list, todo_update, todo_delete
from browser_llm_agent.tools.memory_tools import (
    memory_save, memory_get, memory_list, memory_search, memory_delete
)

mcp = FastMCP("rawagent")


# ── Shell ─────────────────────────────────────────────────────────────────────

@mcp.tool()
def bash(command: str, cwd: str = ".") -> str:
    """Run a shell command and return combined stdout + stderr."""
    return run_bash(command, cwd=cwd)


# ── File operations ───────────────────────────────────────────────────────────

@mcp.tool()
def read_file_tool(path: str) -> str:
    """Read a file and return its contents with line numbers."""
    return read_file(path)


@mcp.tool()
def write_file_tool(path: str, content: str) -> str:
    """Create or completely overwrite a file."""
    return write_file(path, content)


@mcp.tool()
def edit_file_tool(path: str, old: str, new: str) -> str:
    """Replace an exact string in a file. `old` must be unique in the file."""
    return edit_file(path, old, new)


@mcp.tool()
def list_dir_tool(path: str = ".") -> str:
    """List directory contents."""
    return list_dir(path)


# ── Search ────────────────────────────────────────────────────────────────────

@mcp.tool()
def glob_tool(pattern: str, cwd: str = ".") -> str:
    """Find files matching a glob pattern, e.g. **/*.ts"""
    return glob(pattern, cwd=cwd)


@mcp.tool()
def grep_tool(pattern: str, path: str = ".", include: str = "", ignore_case: bool = False) -> str:
    """Search for a regex pattern across files."""
    return grep(pattern, path=path, include=include or None, ignore_case=ignore_case)


@mcp.tool()
def web_fetch_tool(url: str) -> str:
    """Fetch a URL and return its content as plain text."""
    return web_fetch(url)


@mcp.tool()
def find_files_tool(name: str, path: str = ".", file_type: str = "") -> str:
    """Find files or directories by exact name."""
    return find_files(name, path=path, file_type=file_type or None)


@mcp.tool()
def delete_file_tool(path: str) -> str:
    """Delete a file."""
    return delete_file(path)


@mcp.tool()
def move_file_tool(src: str, dst: str) -> str:
    """Move or rename a file."""
    return move_file(src, dst)


@mcp.tool()
def make_dir_tool(path: str) -> str:
    """Create a directory (including parents)."""
    return make_dir(path)


# ── Todo ──────────────────────────────────────────────────────────────────────

@mcp.tool()
def todo_add_tool(content: str, priority: str = "medium") -> str:
    """Add a todo item. priority: high | medium | low"""
    return todo_add(content, priority=priority)


@mcp.tool()
def todo_list_tool(status: str = "") -> str:
    """List todos. status: pending | in_progress | done (empty = all)"""
    return todo_list(status=status or None)


@mcp.tool()
def todo_update_tool(todo_id: str, status: str) -> str:
    """Update a todo's status. status: pending | in_progress | done"""
    return todo_update(todo_id, status)


@mcp.tool()
def todo_delete_tool(todo_id: str) -> str:
    """Delete a todo item."""
    return todo_delete(todo_id)


# ── Persistent memory ─────────────────────────────────────────────────────────

@mcp.tool()
def memory_save_tool(key: str, value: str) -> str:
    """Save a persistent key/value memory that survives across sessions."""
    return memory_save(key, value)


@mcp.tool()
def memory_get_tool(key: str) -> str:
    """Retrieve a memory by key."""
    return memory_get(key)


@mcp.tool()
def memory_list_tool() -> str:
    """List all saved memories."""
    return memory_list()


@mcp.tool()
def memory_search_tool(query: str) -> str:
    """Search memories by keyword."""
    return memory_search(query)


@mcp.tool()
def memory_delete_tool(key: str) -> str:
    """Delete a memory by key."""
    return memory_delete(key)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
