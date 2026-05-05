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
import socket
import subprocess
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


def _browser_launch_args() -> list[str]:
    """Prefer CDP port 9222 for MCP, but do not fail if another browser owns it."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind(("127.0.0.1", 9222))
        except OSError:
            print(c("  Port 9222 is in use; Chrome DevTools MCP may not attach to this browser.", YELLOW))
            return ["--remote-debugging-port=0"]
    return ["--remote-debugging-port=9222"]


def _write_history_file(path: str):
    try:
        readline.write_history_file(path)
    except OSError:
        pass


def _short_git_status(cwd: str = ".") -> str:
    try:
        result = subprocess.run(
            ["git", "status", "--short"],
            capture_output=True,
            text=True,
            timeout=3,
            cwd=cwd,
        )
    except Exception as e:
        return f"(git status unavailable: {e})"
    if result.returncode != 0:
        return f"(git status failed: {result.stderr.strip() or result.stdout.strip()})"
    status = result.stdout.strip()
    if not status:
        return "(clean)"
    lines = status.splitlines()
    if len(lines) > 40:
        return "\n".join(lines[:40]) + f"\n... ({len(lines) - 40} more files)"
    return status


def build_turn_message(user_message: str) -> str:
    """Attach small, fresh context to every turn so browser models stay grounded."""
    return (
        f"{user_message}\n\n"
        "<rawagent_turn_context>\n"
        f"cwd: {os.getcwd()}\n"
        "git status --short:\n"
        f"{_short_git_status()}\n"
        "Use this context to avoid stale assumptions. For tracebacks, inspect the named file/line before editing.\n"
        "</rawagent_turn_context>"
    )


def build_tool_result_message(results: list[dict]) -> str:
    body = "\n".join(
        f"Tool `{r['tool']}` result:\n```\n{r['result']}\n```"
        for r in results
    )
    has_error = any(
        any(kw in r["result"].lower() for kw in
            ("error", "failed", "traceback", "exception", "not found", "no such file"))
        for r in results
    )
    protocol = (
        "\n\nTool-result protocol: if any result is an Error, failed edit, missing string, "
        "or failed patch check, stop relying on the previous assumption. "
        "Form a new hypothesis grounded in this output, then test it with a focused command or edit."
    )
    if has_error:
        protocol += (
            "\n[Failure detected] Do not repeat the same approach. "
            "Treat the error as new evidence — revise your hypothesis and try a different fix."
        )
    return body + protocol


# ── Verification feedback loop ─────────────────────────────────────────────────

# Phrases that suggest the agent believes the task is complete
_COMPLETION_PHRASES = (
    "fixed", "resolved", "should work", "should now", "now works",
    "the issue is", "bug is fixed", "problem is fixed", "implemented the",
    "changes are in place", "the fix", "successfully applied", "has been fixed",
    "has been resolved", "the change has",
)

# Tool name fragments that indicate actual execution (not just file reading)
_VERIFICATION_TOOL_FRAGMENTS = ("bash", "python", "run", "execute", "test")

_VERIFICATION_NUDGE = (
    "[Evidence required] You described a fix but have not run any verification yet. "
    "Follow the hypothesis-test cycle:\n"
    "1. State your hypothesis — if the fix is correct, what should running X produce?\n"
    "2. Execute the verification: run the relevant test, build command, or reproduction script.\n"
    "3. Read the actual output. If it confirms, report it with the evidence shown.\n"
    "4. If it fails, analyse what the output reveals, form a revised hypothesis, and test again.\n"
    "Do not report success until you have seen actual passing output."
)


def _claims_completion(text: str) -> bool:
    """Return True if text suggests the agent thinks the task is done."""
    lower = text.lower()
    return any(phrase in lower for phrase in _COMPLETION_PHRASES)


def _tool_is_verification(tool_name: str) -> bool:
    """Return True if the tool name implies actual execution rather than passive reading."""
    lower = tool_name.lower()
    return any(frag in lower for frag in _VERIFICATION_TOOL_FRAGMENTS)


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
    """Send a message and handle the full tool loop until LLM stops calling tools.

    Includes a verification feedback loop: if the agent claims the task is done
    without having run any execution tool, it is nudged to actually verify before
    the turn ends.
    """
    turn_message = build_turn_message(user_message)
    if is_first_message:
        full_message = f"{system_prompt}\n\n{turn_message}"
    else:
        full_message = turn_message

    print(c("  thinking...", DIM), end="\r")
    response = send_fn(full_message)
    print("              ", end="\r")  # clear "thinking..."

    max_tool_turns = 40  # higher ceiling to allow full hypothesis-test cycles
    turn = 0
    ran_verification = False   # set True once any execution/test tool is used
    used_any_tools = False     # set True once any tool call runs

    while turn < max_tool_turns:
        tool_calls = parse_tool_calls(response)

        # print any prose in the response
        print_llm(response)

        if not tool_calls:
            # Agent stopped — if it claimed success without ever running a check, force it.
            prose = strip_tool_blocks(response)
            if used_any_tools and _claims_completion(prose) and not ran_verification:
                print(c("  [verification required — no execution tool was run]", YELLOW))
                time.sleep(1)
                print(c("  thinking...", DIM), end="\r")
                response = send_fn(_VERIFICATION_NUDGE)
                print("              ", end="\r")
                turn += 1
                continue
            break

        used_any_tools = True

        # execute each tool call
        results = []
        for call in tool_calls:
            print_tool_call(call)
            result = execute_tool(call, mcp_manager)
            print_tool_result(result)
            results.append({"tool": call["name"], "result": result})
            if _tool_is_verification(call["name"]):
                ran_verification = True

        # feed results back
        result_text = build_tool_result_message(results)
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
    """One user turn using Claude's native tool calling. Mutates `messages` in place.

    Includes a verification feedback loop: if the agent claims the task is done
    without having run any execution tool, it is nudged to actually verify before
    the turn ends.
    """
    messages.append({"role": "user", "content": user_message})

    max_turns = 40  # higher ceiling to allow full hypothesis-test cycles
    ran_verification = False   # set True once any execution/test tool is used
    used_any_tools = False     # set True once any tool call runs

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
            # Agent stopped — if it claimed success without running any check, force it.
            final_text = " ".join(b.text for b in text_blocks)
            if used_any_tools and _claims_completion(final_text) and not ran_verification:
                print(c("  [verification required — no execution tool was run]", YELLOW))
                messages.append({"role": "user", "content": _VERIFICATION_NUDGE})
                continue
            break

        used_any_tools = True

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
            if _tool_is_verification(tu.name):
                ran_verification = True

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
    except OSError:
        pass
    readline.set_history_length(1000)

    while True:
        try:
            user_input = input("you [claude]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        finally:
            _write_history_file(history_file)

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
    except OSError:
        pass
    readline.set_history_length(1000)

    while True:
        try:
            user_input = input(f"you [{current_llm}]> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        finally:
            _write_history_file(history_file)

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
    parser.add_argument(
        "--ollama-model",
        default=os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:14b"),
        help="Ollama coder model (default: OLLAMA_MODEL or qwen2.5-coder:14b).",
    )
    parser.add_argument(
        "--ollama-reasoning-model",
        default=os.environ.get("OLLAMA_REASONING_MODEL", "qwen3:14b"),
        help=(
            "Ollama reasoning model (default: qwen3:14b). The reasoning model plans and "
            "delegates code writing to --ollama-model. Set to empty string to disable."
        ),
    )
    parser.add_argument(
        "--ollama-url",
        default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        help="Ollama base URL with --llm ollama (default: http://localhost:11434).",
    )
    args = parser.parse_args()

    # ── Ollama path: no browser required ─────────────────────────────────────
    if args.llm == "ollama":
        mcp_manager = MCPManager()
        n = mcp_manager.load_and_connect(status_cb=lambda msg: print(c(msg, DIM)))
        if n:
            print(c(f"  {n} MCP server(s) ready\n", DIM))

        if args.ollama_reasoning_model:
            from browser_llm_agent.llm.ollama import create_reasoning_chat
            chat = create_reasoning_chat(
                reasoning_model=args.ollama_reasoning_model,
                coder_model=args.ollama_model,
                base_url=args.ollama_url,
            )
            print(c(f"  Ollama two-model mode:", DIM))
            print(c(f"    reasoning: {args.ollama_reasoning_model}", DIM))
            print(c(f"    coder:     {args.ollama_model}", DIM))
            print(c(f"    url:       {args.ollama_url}\n", DIM))
        else:
            from browser_llm_agent.llm.ollama import create_chat
            chat = create_chat(model=args.ollama_model, base_url=args.ollama_url)
            print(c(f"  Using Ollama model '{args.ollama_model}' at {args.ollama_url}\n", DIM))

        try:
            interactive_shell(
                pages={},
                send_fns={"ollama": chat.send_message},
                new_conv_fns={"ollama": chat.new_conversation},
                start_llm="ollama",
                mcp_manager=mcp_manager,
            )
        finally:
            mcp_manager.stop_all()
            print(c("\nBye.\n", DIM))
        return

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
        browser = p.chromium.launch(headless=False, args=_browser_launch_args())

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
