#!/usr/bin/env python3
"""
browser-llm-agent: interactive coding shell powered by ChatGPT, Gemini, or Claude.

Usage:
    python -m browser_llm_agent.cli --llm claude    # Anthropic API (no browser needed)
    python -m browser_llm_agent.cli --llm chatgpt
    python -m browser_llm_agent.cli --llm gemini
"""

import argparse
import json
import os
import readline
import re
import sys
import time

# Import tools package to trigger @tool decorator registration
import browser_llm_agent.tools  # noqa: F401

from browser_llm_agent.tools.registry import execute_tool, get_prompt_tools, get_claude_tools
from browser_llm_agent.mcp_client import MCPManager
from browser_llm_agent.prompts import build_browser_system_prompt

# Claude backend (only imported when --llm claude is used)
try:
    from browser_llm_agent.llm.claude import (
        create_client as claude_create_client,
        SYSTEM_PROMPT as CLAUDE_SYSTEM_PROMPT,
    )
    CLAUDE_AVAILABLE = True
except ImportError:
    CLAUDE_AVAILABLE = False


# ── ANSI colors ──────────────────────────────────────────────────────────────

RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"
GREEN  = "\033[32m"
YELLOW = "\033[33m"
CYAN   = "\033[36m"
RED    = "\033[31m"
BLUE   = "\033[34m"
GRAY   = "\033[90m"


def c(text, *codes):
    return "".join(codes) + str(text) + RESET


def build_system_prompt(mcp_manager: MCPManager | None = None) -> str:
    """Build the full system prompt with auto-generated tool docs + project context."""
    from browser_llm_agent.tools.project_tools import project_detect

    ctx = project_detect(".")
    mcp_section = mcp_manager.prompt_section() if mcp_manager else ""
    return build_browser_system_prompt(get_prompt_tools(), ctx, mcp_section)


# ── Tool parsing + execution ──────────────────────────────────────────────────

def _extract_json_objects(text: str) -> list[dict]:
    """Bracket-match every top-level { ... } in text and try to parse as JSON."""
    results = []
    i = 0
    while i < len(text):
        if text[i] == '{':
            depth = 0
            in_string = False
            escape = False
            for j in range(i, len(text)):
                ch = text[j]
                if escape:
                    escape = False
                elif ch == '\\' and in_string:
                    escape = True
                elif ch == '"':
                    in_string = not in_string
                elif not in_string:
                    if ch == '{':
                        depth += 1
                    elif ch == '}':
                        depth -= 1
                        if depth == 0:
                            candidate = text[i:j+1]
                            try:
                                obj = json.loads(candidate)
                                if isinstance(obj, dict):
                                    results.append(obj)
                            except json.JSONDecodeError:
                                pass
                            i = j + 1
                            break
            else:
                i += 1
        else:
            i += 1
    return results


def parse_tool_calls(text: str) -> list[dict]:
    calls = []

    # 1. explicit ```tool ... ``` blocks
    for m in re.findall(r"```tool\s*(.*?)```", text, re.DOTALL):
        try:
            calls.append(json.loads(m.strip()))
        except json.JSONDecodeError:
            pass

    # 2. any fenced block containing "name":
    if not calls:
        for m in re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL):
            stripped = m.strip()
            if '"name"' in stripped:
                try:
                    obj = json.loads(stripped)
                    if isinstance(obj, dict) and "name" in obj:
                        calls.append(obj)
                except json.JSONDecodeError:
                    pass

    # 3. bracket-match any bare JSON object with a "name" key
    if not calls:
        for obj in _extract_json_objects(text):
            if "name" in obj:
                calls.append(obj)

    return calls


def strip_tool_blocks(text: str) -> str:
    return re.sub(r"```tool\s*.*?```", "", text, flags=re.DOTALL).strip()


# ── Display helpers ───────────────────────────────────────────────────────────

def print_llm(text: str):
    clean = strip_tool_blocks(text)
    if clean:
        print(f"\n{c(clean, GREEN)}\n")


def print_tool_call(call: dict):
    name = call.get("name", "?")
    detail = {k: v for k, v in call.items() if k != "name"}
    detail_str = json.dumps(detail, ensure_ascii=False)
    print(c(f"  ⚙  {name}  {detail_str}", YELLOW))


def print_tool_result(result: str):
    lines = result.splitlines()
    # show max 30 lines of output to avoid flooding
    preview = "\n".join(lines[:30])
    if len(lines) > 30:
        preview += f"\n{c(f'  ... ({len(lines) - 30} more lines)', GRAY)}"
    print(c(preview, GRAY))
    print()


def print_header(llm_name: str):
    print(c("─" * 60, BLUE))
    print(c(f"  browser-llm-agent  [{llm_name}]", BOLD + BLUE))
    print(c("─" * 60, BLUE))
    print(c("  /new    start fresh conversation", DIM))
    print(c("  /switch switch to other LLM", DIM))
    print(c("  /help   show this help", DIM))
    print(c("  /quit   exit", DIM))
    print(c("─" * 60, BLUE))
    print()


# ── Agent turn ────────────────────────────────────────────────────────────────

def agent_turn(send_fn, user_message: str, is_first_message: bool,
               system_prompt: str, mcp_manager: MCPManager | None = None):
    """Send a message and handle the full tool loop until LLM stops calling tools."""
    if is_first_message:
        full_message = f"{system_prompt}\n\n{user_message}"
    else:
        full_message = user_message

    print(c("  thinking...", DIM), end="\r")
    response = send_fn(full_message)
    print("              ", end="\r")  # clear "thinking..."

    max_tool_turns = 20
    turn = 0

    while turn < max_tool_turns:
        tool_calls = parse_tool_calls(response)

        # print any prose in the response
        print_llm(response)

        if not tool_calls:
            break

        # execute each tool call
        results = []
        for call in tool_calls:
            print_tool_call(call)
            result = execute_tool(call, mcp_manager)
            print_tool_result(result)
            results.append({"tool": call["name"], "result": result})

        # feed results back
        result_text = "\n".join(
            f"Tool `{r['tool']}` result:\n```\n{r['result']}\n```"
            for r in results
        )
        time.sleep(1)
        print(c("  thinking...", DIM), end="\r")
        response = send_fn(result_text)
        print("              ", end="\r")
        turn += 1

    if turn >= max_tool_turns:
        print(c("  [reached max tool turns]", RED))


# ── Claude native tool-use loop ───────────────────────────────────────────────

def claude_agent_turn(client, tools: list, messages: list, user_message: str,
                      mcp_manager: MCPManager | None = None):
    """One user turn using Claude's native tool calling. Mutates `messages` in place."""
    messages.append({"role": "user", "content": user_message})

    max_turns = 20
    for _ in range(max_turns):
        print(c("  thinking...", DIM), end="\r", flush=True)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8096,
            system=CLAUDE_SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )
        print("              ", end="\r", flush=True)

        # Add assistant response to history
        messages.append({"role": "assistant", "content": response.content})

        # Split content into text and tool_use blocks
        text_blocks = [b for b in response.content if b.type == "text"]
        tool_uses  = [b for b in response.content if b.type == "tool_use"]

        # Print any prose
        for block in text_blocks:
            if block.text.strip():
                print(f"\n{c(block.text.strip(), GREEN)}\n")

        if not tool_uses:
            break

        # Execute tools and collect results
        tool_results = []
        for tu in tool_uses:
            call = {"name": tu.name, **tu.input}
            print_tool_call(call)
            result = execute_tool(call, mcp_manager)
            print_tool_result(result)
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": result,
            })

        messages.append({"role": "user", "content": tool_results})
    else:
        print(c("  [reached max tool turns]", RED))


def claude_shell(mcp_manager: MCPManager | None = None):
    """Interactive shell backed by the Claude API — no browser required."""
    if not CLAUDE_AVAILABLE:
        print(c("  Error: 'anthropic' package not installed. Run: pip install anthropic", RED))
        return

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print(c("  Error: ANTHROPIC_API_KEY environment variable not set.", RED))
        return

    # Enable agent spawning for Claude mode
    from browser_llm_agent.tools.agent_tools import _configure as _configure_agents
    _configure_agents(api_key)

    client = claude_create_client(api_key)
    tools = get_claude_tools(mcp_manager)
    messages: list = []

    print_header("claude")

    history_file = os.path.expanduser("~/.llm-agent/history")
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    try:
        readline.read_history_file(history_file)
    except FileNotFoundError:
        pass
    readline.set_history_length(1000)

    while True:
        try:
            user_input = input("you [claude]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        finally:
            readline.write_history_file(history_file)

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.lower()
            if cmd in ("/quit", "/exit", "/q"):
                break
            elif cmd == "/new":
                messages.clear()
                print(c("  Started fresh conversation.\n", DIM))
            elif cmd == "/help":
                print_header("claude")
            else:
                print(c(f"  Unknown command: {user_input}\n", RED))
            continue

        claude_agent_turn(client, tools, messages, user_input, mcp_manager)


# ── Main interactive shell ────────────────────────────────────────────────────

def interactive_shell(pages: dict, send_fns: dict, new_conv_fns: dict, start_llm: str,
                      mcp_manager: MCPManager | None = None):
    current_llm = start_llm
    send_fn = send_fns[current_llm]
    new_conv_fn = new_conv_fns[current_llm]
    is_first_message = True

    # Build system prompt once — auto-generates tool docs from registry
    system_prompt = build_system_prompt(mcp_manager)

    print_header(current_llm)

    # persist input history across sessions
    history_file = os.path.expanduser("~/.llm-agent/history")
    os.makedirs(os.path.dirname(history_file), exist_ok=True)
    try:
        readline.read_history_file(history_file)
    except FileNotFoundError:
        pass
    readline.set_history_length(1000)

    while True:
        try:
            user_input = input(f"you [{current_llm}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        finally:
            readline.write_history_file(history_file)

        if not user_input:
            continue

        # ── slash commands ────────────────────────────────────────────────
        if user_input.startswith("/"):
            cmd = user_input.lower()

            if cmd in ("/quit", "/exit", "/q"):
                break

            elif cmd == "/new":
                new_conv_fn()
                is_first_message = True
                print(c("  Started fresh conversation.\n", DIM))

            elif cmd == "/switch":
                other = "gemini" if current_llm == "chatgpt" else "chatgpt"
                if other not in send_fns:
                    print(c(f"  {other} is not connected in this session.\n", RED))
                else:
                    current_llm = other
                    send_fn = send_fns[current_llm]
                    new_conv_fn = new_conv_fns[current_llm]
                    is_first_message = True
                    print(c(f"  Switched to {current_llm}.\n", DIM))

            elif cmd == "/help":
                print_header(current_llm)

            else:
                print(c(f"  Unknown command: {user_input}\n", RED))

            continue

        # ── normal message ────────────────────────────────────────────────
        agent_turn(send_fn, user_input, is_first_message, system_prompt, mcp_manager)
        is_first_message = False


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Browser LLM Coding Agent")
    parser.add_argument(
        "--llm",
        choices=["chatgpt", "gemini", "both", "claude", "ollama"],
        default="gemini",
        help="Which LLM to use (default: gemini). 'claude' uses the Anthropic API directly — no browser needed.",
    )
    args = parser.parse_args()

    # ── Claude path: no browser required ─────────────────────────────────────
    if args.llm == "claude":
        mcp_manager = MCPManager()
        n = mcp_manager.load_and_connect(status_cb=lambda msg: print(c(msg, DIM)))
        if n:
            print(c(f"  {n} MCP server(s) ready\n", DIM))
        try:
            claude_shell(mcp_manager)
        finally:
            mcp_manager.stop_all()
            print(c("\nBye.\n", DIM))
        return

    # ── Browser path: ChatGPT / Gemini ────────────────────────────────────────
    from playwright.sync_api import sync_playwright
    from browser_llm_agent.llm.chatgpt import open_chatgpt, send_message as chatgpt_send, new_conversation as chatgpt_new
    from browser_llm_agent.llm.gemini import open_gemini, send_message as gemini_send, new_conversation as gemini_new

    with sync_playwright() as p:
        # Expose CDP on a fixed port so chrome-devtools-mcp can connect to this browser
        browser = p.chromium.launch(headless=False, args=["--remote-debugging-port=9222"])

        pages = {}
        send_fns = {}
        new_conv_fns = {}

        if args.llm in ("chatgpt", "both"):
            page = open_chatgpt(browser)
            pages["chatgpt"] = page
            send_fns["chatgpt"] = lambda msg, pg=page: chatgpt_send(pg, msg)
            new_conv_fns["chatgpt"] = lambda pg=page: chatgpt_new(pg)

        if args.llm in ("gemini", "both"):
            page = open_gemini(browser)
            pages["gemini"] = page
            send_fns["gemini"] = lambda msg, pg=page: gemini_send(pg, msg)
            new_conv_fns["gemini"] = lambda pg=page: gemini_new(pg)

        start_llm = "chatgpt" if args.llm == "chatgpt" else "gemini"

        # Connect MCP servers after browser is up — chrome-devtools-mcp needs port 9222 live
        mcp_manager = MCPManager()
        n = mcp_manager.load_and_connect(status_cb=lambda msg: print(c(msg, DIM)))
        if n:
            print(c(f"  {n} MCP server(s) ready\n", DIM))

        try:
            interactive_shell(pages, send_fns, new_conv_fns, start_llm, mcp_manager)
        finally:
            mcp_manager.stop_all()
            browser.close()
            print(c("\nBye.\n", DIM))


if __name__ == "__main__":
    main()
