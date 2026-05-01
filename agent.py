#!/usr/bin/env python3
"""
browser-llm-agent: interactive coding shell powered by ChatGPT or Gemini.

Usage:
    python agent.py --llm chatgpt
    python agent.py --llm gemini
"""

import argparse
import json
import re
import readline  # enables arrow keys + history in input()
import sys

from playwright.sync_api import sync_playwright

from llm.chatgpt import open_chatgpt, send_message as chatgpt_send, new_conversation as chatgpt_new
from llm.gemini import open_gemini, send_message as gemini_send, new_conversation as gemini_new
from tools.file_tools import read_file, write_file, edit_file, list_dir
from tools.bash_tools import run_bash


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


# ── System prompt ─────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a coding assistant with file and shell access. You are running inside an interactive terminal shell.

When you need to use a tool, output a JSON block wrapped in triple backticks tagged as "tool":

```tool
{"name": "bash", "command": "ls -la"}
```

Available tools:
- bash:       run any shell command    {"name": "bash", "command": "...", "cwd": "optional path"}
- read_file:  read a file              {"name": "read_file", "path": "..."}
- write_file: create/overwrite file    {"name": "write_file", "path": "...", "content": "..."}
- edit_file:  replace text in file     {"name": "edit_file", "path": "...", "old": "...", "new": "..."}
- list_dir:   list directory contents  {"name": "list_dir", "path": "..."}

Debugging methodology — follow this for every bug:

  1. OBSERVE   Read the relevant files. If the bug involves runtime behaviour
               (wrong output, empty response, unexpected format), use bash to
               run the code and capture actual output before reading anything else.
               Example: run a test script, print a value, call an endpoint.

  2. HYPOTHESIZE  State a specific theory. "The bug is X because Y."
               Do not proceed without a theory grounded in observed evidence.

  3. EXPERIMENT  Use bash to verify or falsify the theory cheaply before
               changing production code. Print the suspect value. Run a
               minimal reproduction. Check a log file.

  4. FIX       Apply one targeted edit_file change. Keep it minimal.

  5. VERIFY    Use bash to confirm the fix works: build, run tests, or
               re-run the reproduction from step 3.

Rules:
- Never diagnose from static code alone when bash can show you the runtime truth.
- Always read a file before editing it.
- One tool call at a time — wait for the result before the next.
- If a step fails, return to OBSERVE before retrying.
- When all steps are done, say so in plain text. Do not use a special done tool.
- NEVER ask for permission before using a tool. Do not say "Should I run X?",
  "Do you want me to proceed?", "Shall I apply this fix?", or any similar phrase.
  Emit the tool block immediately and let the result speak for itself.
- NEVER ask the user to fetch information you can obtain yourself. Do not say
  "What does the log show?", "Check your console", "Can you run X and tell me
  the output?", or any similar phrase. If you need a file, read it. If you need
  command output, run it. You have bash — use it.
- Do not ask clarifying questions — make reasonable assumptions and proceed.
"""


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


def execute_tool(call: dict) -> str:
    name = call.get("name")
    if name == "bash":
        cmd = call.get("command")
        if not cmd:
            return "Error: bash tool requires a 'command' field"
        return run_bash(cmd, cwd=call.get("cwd"))
    elif name == "read_file":
        path = call.get("path")
        if not path:
            return "Error: read_file requires a 'path' field"
        return read_file(path)
    elif name == "write_file":
        path, content = call.get("path"), call.get("content")
        if not path or content is None:
            return "Error: write_file requires 'path' and 'content' fields"
        return write_file(path, content)
    elif name == "edit_file":
        path, old, new = call.get("path"), call.get("old"), call.get("new")
        if not path or old is None or new is None:
            return "Error: edit_file requires 'path', 'old', and 'new' fields"
        return edit_file(path, old, new)
    elif name == "list_dir":
        return list_dir(call.get("path", "."))
    else:
        return f"Error: unknown tool '{name}'"


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

def agent_turn(send_fn, user_message: str, is_first_message: bool):
    """Send a message and handle the full tool loop until LLM stops calling tools."""
    if is_first_message:
        full_message = f"{SYSTEM_PROMPT}\n\n{user_message}"
    else:
        full_message = user_message

    try:
        print(c("  thinking...", DIM), end="\r")
        response = send_fn(full_message)
        print("              ", end="\r")  # clear "thinking..."
    except Exception as e:
        print(c(f"\n  [browser error on send: {e}]\n", RED))
        return

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
            try:
                result = execute_tool(call)
            except Exception as e:
                result = f"Error executing tool '{call.get('name', '?')}': {e}"
            print_tool_result(result)
            results.append({"tool": call.get("name", "?"), "result": result})

        # feed results back
        result_text = "\n".join(
            f"Tool `{r['tool']}` result:\n```\n{r['result']}\n```"
            for r in results
        )
        try:
            print(c("  thinking...", DIM), end="\r")
            response = send_fn(result_text)
            print("              ", end="\r")
        except Exception as e:
            print(c(f"\n  [browser error on tool result: {e}]\n", RED))
            break
        turn += 1

    if turn >= max_tool_turns:
        print(c("  [reached max tool turns]", RED))


# ── Main interactive shell ────────────────────────────────────────────────────

def interactive_shell(pages: dict, send_fns: dict, new_conv_fns: dict, start_llm: str):
    current_llm = start_llm
    send_fn = send_fns[current_llm]
    new_conv_fn = new_conv_fns[current_llm]
    is_first_message = True

    print_header(current_llm)

    while True:
        try:
            prompt_label = c(f"you [{current_llm}]> ", CYAN + BOLD)
            user_input = input(prompt_label).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

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
        agent_turn(send_fn, user_input, is_first_message)
        is_first_message = False


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Browser LLM Coding Agent")
    parser.add_argument(
        "--llm",
        choices=["chatgpt", "gemini", "both"],
        default="gemini",
        help="Which LLM to use (default: gemini)",
    )
    args = parser.parse_args()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=False)

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

        try:
            interactive_shell(pages, send_fns, new_conv_fns, start_llm)
        finally:
            browser.close()
            print(c("\nBye.\n", DIM))


if __name__ == "__main__":
    main()
