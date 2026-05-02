#!/usr/bin/env python3
"""
rawagent MCP server — exposes rawagent tools to Claude Code CLI.

Add to Claude Code once:
    claude mcp add rawagent -- python -m browser_llm_agent.mcp_server

Then every `claude` session has rawagent's tools available automatically.

Tools are auto-generated from the centralized registry.
"""

from mcp.server.fastmcp import FastMCP

# Import tools package to trigger @tool decorator registration
import browser_llm_agent.tools  # noqa: F401

from browser_llm_agent.tools.registry import TOOL_REGISTRY

mcp = FastMCP("rawagent")


def _register_all():
    """Auto-register all tools from the registry as MCP tools."""
    for td in TOOL_REGISTRY.values():
        # Create a closure to capture the current tool def
        def make_wrapper(tool_def):
            def wrapper(**kwargs):
                return tool_def.fn(**kwargs)
            wrapper.__name__ = tool_def.name + "_tool"
            wrapper.__doc__ = tool_def.description
            return wrapper

        wrapper = make_wrapper(td)

        # Build parameter annotations from JSON schema for FastMCP
        props = td.parameters.get("properties", {})
        required = set(td.parameters.get("required", []))

        # FastMCP uses the function signature, so we register via the decorator
        # For simplicity, we register the raw function and let FastMCP inspect it
        mcp.tool()(wrapper)


_register_all()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
