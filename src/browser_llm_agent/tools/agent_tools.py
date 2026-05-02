"""
Agent spawning tools — run sub-agents for parallel work.

Sub-agents get their own Claude API client and message history.
Only works with Claude API backend (browser backends can't handle
concurrent conversations).
"""

import concurrent.futures
import os
import threading

from browser_llm_agent.tools.registry import tool, TOOL_REGISTRY, get_claude_tools

# Module-level flag set by cli.py when Claude mode is active
_CLAUDE_AVAILABLE = False
_API_KEY: str | None = None


def _configure(api_key: str):
    """Called by cli.py to enable agent spawning."""
    global _CLAUDE_AVAILABLE, _API_KEY
    _API_KEY = api_key
    _CLAUDE_AVAILABLE = True


def _run_subagent(task: str, timeout: int = 120) -> str:
    """Run a sub-agent with its own Claude client and conversation."""
    import anthropic
    from browser_llm_agent.llm.claude import SYSTEM_PROMPT

    client = anthropic.Anthropic(api_key=_API_KEY)

    # Get all tools except spawn_agent/spawn_agents (prevent recursion)
    tools = [
        {"name": td.name, "description": td.description, "input_schema": td.parameters}
        for td in TOOL_REGISTRY.values()
        if td.name not in ("spawn_agent", "spawn_agents")
    ]

    messages = [{"role": "user", "content": task}]

    for _ in range(20):
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8096,
            system=SYSTEM_PROMPT + "\n\nYou are a sub-agent. Complete the task and return a concise result.",
            tools=tools,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        text_blocks = [b for b in response.content if b.type == "text"]
        tool_uses = [b for b in response.content if b.type == "tool_use"]

        if not tool_uses:
            texts = [b.text for b in text_blocks if b.text.strip()]
            return "\n".join(texts) if texts else "(no response)"

        # Execute tools
        from browser_llm_agent.tools.registry import execute_tool
        tool_results = []
        for tu in tool_uses:
            call = {"name": tu.name, **tu.input}
            result = execute_tool(call)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})

    return "(sub-agent reached max tool turns)"


@tool("spawn_agent", "Spawn a sub-agent to work on a task independently. Claude API only.", {
    "type": "object",
    "properties": {
        "task": {"type": "string", "description": "Task description for the sub-agent"},
        "timeout": {"type": "integer", "description": "Timeout in seconds (default: 120, max: 300)"},
    },
    "required": ["task"],
})
def spawn_agent(task: str, timeout: int = 120) -> str:
    if not _CLAUDE_AVAILABLE:
        return "Error: spawn_agent only works with Claude API backend (--llm claude). Browser backends cannot run concurrent conversations."

    timeout = min(max(timeout, 10), 300)

    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(_run_subagent, task, timeout)
            return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        return f"Error: sub-agent timed out after {timeout}s"
    except Exception as e:
        return f"Error: sub-agent failed: {e}"


@tool("spawn_agents", "Spawn multiple sub-agents in parallel. Claude API only.", {
    "type": "object",
    "properties": {
        "tasks": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of task descriptions, one per sub-agent",
        },
        "timeout": {"type": "integer", "description": "Timeout in seconds for all agents (default: 120, max: 300)"},
    },
    "required": ["tasks"],
})
def spawn_agents(tasks: list, timeout: int = 120) -> str:
    if not _CLAUDE_AVAILABLE:
        return "Error: spawn_agents only works with Claude API backend (--llm claude)."

    if not tasks:
        return "Error: no tasks provided"

    timeout = min(max(timeout, 10), 300)
    max_agents = min(len(tasks), 5)  # Cap at 5 parallel agents

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_agents) as executor:
        futures = {
            executor.submit(_run_subagent, task, timeout): i
            for i, task in enumerate(tasks[:max_agents])
        }

        for future in concurrent.futures.as_completed(futures, timeout=timeout):
            idx = futures[future]
            try:
                result = future.result()
                results.append(f"[Agent {idx+1}] {result}")
            except Exception as e:
                results.append(f"[Agent {idx+1}] Error: {e}")

    # Add skipped tasks if any
    for i in range(max_agents, len(tasks)):
        results.append(f"[Agent {i+1}] Skipped (max 5 parallel agents)")

    return "\n\n---\n\n".join(results)
