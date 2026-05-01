"""
MCP client layer for rawagent.

Each configured MCP server runs in a background thread with its own asyncio
event loop so it doesn't conflict with Playwright's sync API. Tool calls are
dispatched via an asyncio queue and results returned through concurrent.futures.

Config file: ~/.llm-agent/mcp_servers.json  (Claude Desktop format)
"""

import asyncio
import json
import os
import threading
from concurrent.futures import Future as CFuture
from typing import Any

MCP_AVAILABLE = False
try:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client, StdioServerParameters
    MCP_AVAILABLE = True
except ImportError:
    pass

CONFIG_PATH = os.path.expanduser("~/.llm-agent/mcp_servers.json")
DEFAULT_CONFIG: dict = {"mcpServers": {}}


class MCPServerConnection:
    """One MCP server running in a background thread."""

    def __init__(self, name: str, command: str, args: list[str], env: dict | None = None):
        self.name = name
        self.command = command
        self.args = args
        self.env = env
        self.tools: list = []

        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._loop.run_forever, daemon=True)
        self._queue: asyncio.Queue | None = None
        self._ready = threading.Event()
        self._error: Exception | None = None

    def start(self) -> bool:
        """Launch background thread and wait for the server to be ready."""
        self._thread.start()
        asyncio.run_coroutine_threadsafe(self._run(), self._loop)
        self._ready.wait(timeout=15)
        return self._error is None and self._queue is not None

    async def _run(self):
        try:
            # Inherit parent environment so subprocess can find binaries (npx, python, etc.)
            merged_env = {**os.environ, **(self.env or {})}
            params = StdioServerParameters(
                command=self.command,
                args=self.args,
                env=merged_env,
            )
            async with stdio_client(params) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    tools_result = await session.list_tools()
                    self.tools = tools_result.tools

                    self._queue = asyncio.Queue()
                    self._ready.set()

                    # Serve tool-call requests from the main thread indefinitely.
                    while True:
                        item = await self._queue.get()
                        if item is None:
                            break
                        tool_name, arguments, cf = item
                        try:
                            result = await session.call_tool(tool_name, arguments)
                            parts = [
                                c.text if hasattr(c, "text") else str(c)
                                for c in result.content
                            ]
                            cf.set_result("\n".join(parts))
                        except Exception as exc:
                            cf.set_exception(exc)
        except Exception as exc:
            self._error = exc
            self._ready.set()

    def call_tool(self, tool_name: str, arguments: dict) -> str:
        if self._queue is None:
            return f"Error: MCP server '{self.name}' is not connected"

        cf: CFuture[str] = CFuture()
        asyncio.run_coroutine_threadsafe(
            self._queue.put((tool_name, arguments, cf)), self._loop
        )
        try:
            return cf.result(timeout=30)
        except TimeoutError:
            return f"Error: tool call to '{tool_name}' timed out after 30 s"
        except Exception as exc:
            return f"Error: {exc}"

    def stop(self):
        if self._queue is not None:
            asyncio.run_coroutine_threadsafe(self._queue.put(None), self._loop)


class MCPManager:
    """Manages connections to all configured MCP servers."""

    def __init__(self):
        self._servers: dict[str, MCPServerConnection] = {}
        # Maps "servername__toolname" -> server name for fast routing
        self._namespaced: dict[str, str] = {}

    def load_and_connect(self, status_cb=None) -> int:
        """
        Read ~/.llm-agent/mcp_servers.json and connect to every server.
        Returns the number of servers that connected successfully.
        status_cb(msg) is called for each server outcome if provided.
        """
        if not MCP_AVAILABLE:
            if status_cb:
                status_cb("  MCP: 'mcp' package not installed — run: pip install mcp")
            return 0

        config = self._load_config()
        connected = 0

        for name, cfg in config.get("mcpServers", {}).items():
            command = cfg.get("command", "")
            if not command:
                if status_cb:
                    status_cb(f"  MCP '{name}': no command specified, skipping")
                continue

            conn = MCPServerConnection(
                name=name,
                command=command,
                args=cfg.get("args", []),
                env=cfg.get("env") or None,
            )
            ok = conn.start()

            if ok:
                self._servers[name] = conn
                for tool in conn.tools:
                    self._namespaced[f"{name}__{tool.name}"] = name
                connected += 1
                if status_cb:
                    status_cb(f"  MCP '{name}': connected  ({len(conn.tools)} tools)")
            else:
                if status_cb:
                    status_cb(f"  MCP '{name}': failed — {conn._error}")

        return connected

    def _load_config(self) -> dict:
        if not os.path.exists(CONFIG_PATH):
            os.makedirs(os.path.dirname(CONFIG_PATH), exist_ok=True)
            with open(CONFIG_PATH, "w") as f:
                json.dump(DEFAULT_CONFIG, f, indent=2)
            return DEFAULT_CONFIG
        with open(CONFIG_PATH) as f:
            return json.load(f)

    # ── Accessors ──────────────────────────────────────────────────────────────

    def all_tools(self) -> list[tuple[str, Any]]:
        """Return (server_name, tool) for every tool across all servers."""
        return [
            (server_name, tool)
            for server_name, conn in self._servers.items()
            for tool in conn.tools
        ]

    def has_tools(self) -> bool:
        return bool(self._namespaced)

    def is_mcp_tool(self, name: str) -> bool:
        return name in self._namespaced

    def call_tool(self, namespaced_name: str, arguments: dict) -> str:
        server_name = self._namespaced.get(namespaced_name)
        if not server_name:
            return f"Error: unknown MCP tool '{namespaced_name}'"
        # Strip "servername__" prefix to get the original tool name
        tool_name = namespaced_name[len(server_name) + 2:]
        return self._servers[server_name].call_tool(tool_name, arguments)

    def stop_all(self):
        for conn in self._servers.values():
            conn.stop()

    # ── System prompt snippet ─────────────────────────────────────────────────

    def prompt_section(self) -> str:
        """Return a block of tool docs ready to append to the system prompt."""
        if not self.has_tools():
            return ""

        lines = ["\n\nMCP server tools (call using the namespaced name 'server__toolname'):"]
        for server_name, tool in self.all_tools():
            namespaced = f"{server_name}__{tool.name}"
            desc = (tool.description or "").strip().splitlines()[0]  # first line only

            # Build a minimal example from inputSchema required fields
            schema = getattr(tool, "inputSchema", {}) or {}
            required = schema.get("required", [])
            props = schema.get("properties", {})
            example_args: dict = {}
            for k in required:
                prop = props.get(k, {})
                example_args[k] = f"<{prop.get('type', 'string')}>"

            example = json.dumps({"name": namespaced, **example_args}, ensure_ascii=False)
            lines.append(f"- {namespaced}: {desc}  {example}")

        return "\n".join(lines)
