"""
Centralized tool registry for browser-llm-agent.

Register a tool once with @tool(...) and it becomes available everywhere:
  - cli.py execute_tool dispatch
  - Claude API tool definitions
  - Browser-backend system prompt docs
  - MCP server wrappers
"""

import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class ToolDef:
    name: str
    description: str
    parameters: dict          # JSON Schema for input_schema
    fn: Callable[..., str]
    # Maps JSON schema param names to function kwarg names (if different)
    param_map: dict = field(default_factory=dict)


TOOL_REGISTRY: dict[str, ToolDef] = {}


def tool(name: str, description: str, parameters: dict, param_map: dict | None = None):
    """
    Decorator that registers a tool function in TOOL_REGISTRY.

    Usage:
        @tool("bash", "Run a shell command.", {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        })
        def run_bash(command: str, cwd: str = None) -> str:
            ...
    """
    def decorator(fn: Callable[..., str]) -> Callable[..., str]:
        TOOL_REGISTRY[name] = ToolDef(
            name=name,
            description=description,
            parameters=parameters,
            fn=fn,
            param_map=param_map or {},
        )
        return fn
    return decorator


def execute_tool(call: dict, mcp_manager=None) -> str:
    """
    Unified tool dispatch. Checks MCP tools first, then the registry.
    `call` is a dict like {"name": "bash", "command": "ls"}.
    """
    name = call.get("name", "")

    # MCP tools take priority (namespaced as server__toolname)
    if mcp_manager and mcp_manager.is_mcp_tool(name):
        args = {k: v for k, v in call.items() if k != "name"}
        return mcp_manager.call_tool(name, args)

    tool_def = TOOL_REGISTRY.get(name)
    if not tool_def:
        return f"Error: unknown tool '{name}'"

    missing = [
        param for param in tool_def.parameters.get("required", [])
        if param not in call
    ]
    if missing:
        return f"Error calling '{name}': missing required field(s): {', '.join(missing)}"

    # Build kwargs from the call dict, applying param_map if needed
    kwargs = {}
    props = tool_def.parameters.get("properties", {})
    for param_name in props:
        if param_name in call:
            fn_arg = tool_def.param_map.get(param_name, param_name)
            kwargs[fn_arg] = call[param_name]

    try:
        return tool_def.fn(**kwargs)
    except TypeError as e:
        return f"Error calling '{name}': {e}"


def get_claude_tools(mcp_manager=None) -> list[dict]:
    """Generate Anthropic-format tool definitions from the registry + MCP tools."""
    tools = []
    for td in TOOL_REGISTRY.values():
        tools.append({
            "name": td.name,
            "description": td.description,
            "input_schema": td.parameters,
        })

    # Append MCP tools
    if mcp_manager and mcp_manager.has_tools():
        for server_name, t in mcp_manager.all_tools():
            namespaced = f"{server_name}__{t.name}"
            schema = getattr(t, "inputSchema", None) or {
                "type": "object", "properties": {}, "required": []
            }
            tools.append({
                "name": namespaced,
                "description": (t.description or "").strip(),
                "input_schema": schema,
            })

    return tools


def get_prompt_tools() -> str:
    """Generate text tool documentation for browser-backend system prompts."""
    lines = []
    for td in TOOL_REGISTRY.values():
        props = td.parameters.get("properties", {})
        required = set(td.parameters.get("required", []))
        # Build a compact example
        example_args = {}
        for pname, pdef in props.items():
            if pname in required:
                example_args[pname] = "..."
            else:
                example_args[pname] = f"optional"
        example = json.dumps({"name": td.name, **example_args}, ensure_ascii=False)
        lines.append(f"- {td.name}: {td.description}  {example}")
    return "\n".join(lines)
