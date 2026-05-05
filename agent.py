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
from llm.deepseek import open_deepseek, send_message as deepseek_send, new_conversation as deepseek_new
from llm.ollama import send_message as ollama_send
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

  1. OBSERVE   Run list_dir to understand the project layout. Then read the
               relevant files. If the bug involves runtime behaviour (wrong
               output, empty response, unexpected format), use bash to run the
               code and capture actual output before reading anything else.

  2. HYPOTHESIZE  State a specific theory. "The bug is X because Y."
               Do not proceed without a theory grounded in observed evidence.

  3. EXPERIMENT  Use bash to verify or falsify the theory cheaply before
               changing production code. Print the suspect value. Run a
               minimal reproduction. Check a log file.

  4. FIX       Apply one targeted edit_file change. Keep it minimal.

  5. VERIFY    Use bash to confirm the fix works: build, run tests, or
               re-run the reproduction from step 3.

Advanced debugging — when the basic 5 steps stall or the bug is non-obvious:

  ISOLATE (differential diagnosis)
    When code "should work" but doesn't, extract the suspect logic into a
    standalone script and run it outside the server/framework. If the
    isolated version works, the bug is in the CONTEXT (middleware, event
    lifecycle, framework version, environment), not the logic itself.
    This is the single most powerful technique for "works here, fails there" bugs.

  INSTRUMENT (when you can't see logs)
    If the failing code runs in a background server whose stdout you cannot see,
    do NOT rely on console.log/print. Instead, inject diagnostic data into the
    response itself — add state to error messages, emit debug events, write to a
    temp file, or return extra fields in the JSON response. Remove after diagnosis.

  LAYER-BISECT (binary search across the stack)
    Systematically eliminate layers. For a request that fails end-to-end:
      a. Test the external dependency directly (e.g., curl the LLM API).
      b. Test the library wrapper in isolation (e.g., run the SDK call standalone).
      c. Test the route handler with a minimal payload.
      d. Test through the proxy/gateway.
    The layer where behavior diverges from the isolated test IS the problem layer.

  VERSION-CHECK (framework/runtime surprises)
    When behavior contradicts documentation or intuition, check the exact
    version of the framework (package.json, go.mod, requirements.txt) and
    read changelogs or migration guides. Common traps:
      - Express 5 vs 4: req event lifecycle, middleware signatures, error handling
      - Node.js 20+: native fetch, --env-file, ES module resolution changes
      - React 18+: strict mode double-mounting, concurrent features
      - Python 3.12+: new typing syntax, deprecation removals
    When in doubt, write a 5-line test script that exercises the suspect API.

  STATE-SNAPSHOT (race conditions and lifecycle bugs)
    When a flag, variable, or connection state has an unexpected value, add
    logging that captures the value at EVERY mutation point — not just where
    the failure occurs. Pattern: "closed=${closed}" at assignment, at check,
    and at the point where it's consumed. The gap between expected and actual
    mutation timing reveals the root cause.

Auto-detection rules — NEVER ask; detect instead:
- Framework/language: read package.json, pyproject.toml, go.mod, Cargo.toml,
  pom.xml, requirements.txt, or look at file extensions.
- Entry point: check common names (index.ts, main.py, app.py, server.ts,
  main.go, src/main.rs) via list_dir or bash find.
- Routes/handlers: use bash grep for "app.get", "router.", "@app.route", etc.
- Any other project detail: use bash, list_dir, or read_file to discover it.

Absolute prohibitions — NEVER output any of these phrases:
- "What framework are you using?"
- "What language is this?"
- "What file is X in?"
- "Which block should I look at?"
- "Would you like me to look at X?"
- "Should I check X?"
- "Do you want me to proceed?"
- "Shall I apply this fix?"
- "Can you run X and tell me the output?"
- "Check your console" / "What does the log show?"
- "Could you share X?" / "Please provide X"
- Any question whose answer you can obtain with a tool.

Rules:
- Never diagnose from static code alone when bash can show you the runtime truth.
- Always read a file before editing it.
- One tool call at a time — wait for the result before the next.
- If a step fails, return to OBSERVE before retrying.
- When all steps are done, say so in plain text. Do not use a special done tool.
- NEVER ask for permission before using a tool. Emit the tool block immediately.
- NEVER offer to do something — just do it. Turn any "Would you like me to
  look at X?" into an immediate read_file or bash call, no question asked.
- NEVER ask the user to fetch information you can obtain yourself. If you need
  a file, read it. If you need command output, run it. You have bash — use it.
- Do not ask clarifying questions — make reasonable assumptions and proceed.
- NEVER just describe a fix in text — execute it with edit_file or write_file.
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
    seen: set[str] = set()
    calls: list[dict] = []

    def _add(obj: dict) -> None:
        key = json.dumps(obj, sort_keys=True)
        if key not in seen:
            seen.add(key)
            calls.append(obj)

    # 1. explicit ```tool ... ``` blocks
    for m in re.findall(r"```tool\s*(.*?)```", text, re.DOTALL):
        try:
            _add(json.loads(m.strip()))
        except json.JSONDecodeError:
            pass

    # 2. any fenced block containing "name": — runs regardless of step 1 results
    for m in re.findall(r"```(?:json)?\s*(.*?)```", text, re.DOTALL):
        stripped = m.strip()
        if '"name"' in stripped:
            try:
                obj = json.loads(stripped)
                if isinstance(obj, dict) and "name" in obj:
                    _add(obj)
            except json.JSONDecodeError:
                pass

    # 3. bracket-match any bare JSON object with a "name" key — catches tool calls
    #    whose opening fence was stripped by inner_text() but closing fence was not,
    #    or that were emitted without any fencing at all.
    for obj in _extract_json_objects(text):
        if "name" in obj:
            _add(obj)

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
    elif name == "node_exec":
        code = call.get("code")
        if not code:
            return "Error: node_exec requires a 'code' field"
        import tempfile, os as _os
        with tempfile.NamedTemporaryFile(suffix=".js", mode="w", delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp = f.name
        try:
            return run_bash(f"node {tmp}")
        finally:
            _os.unlink(tmp)
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

_NUDGE_MSG = (
    "You described what you intend to do but did not emit any tool calls. "
    "Do NOT describe actions — perform them. Emit the ```tool\\n{...}\\n``` "
    "blocks RIGHT NOW for each operation you just described."
)

_ORCHESTRATOR_PROMPT = (
    "You are a intent classifier. Given an AI assistant's response, decide whether "
    "it DESCRIBES actions it intends to take (like 'I will stage the files', "
    "'Let me commit', 'I'll run the command') WITHOUT actually emitting tool call "
    "JSON blocks (```tool {...} ```).\n\n"
    "Reply with ONLY one word:\n"
    "- UNFULFILLED — if the response promises/describes actions but has no tool call blocks\n"
    "- COMPLETE — if the response either contains tool calls OR is just a normal answer/explanation "
    "with no promised actions\n\n"
    "Response to classify:\n"
)


def _has_unfulfilled_intent(response: str) -> bool:
    """Use a small local LLM to detect if the response describes actions
    without actually emitting tool calls."""
    try:
        import ollama as _ollama
        result = _ollama.chat(
            model="qwen2.5-coder:1.5b",
            messages=[{"role": "user", "content": _ORCHESTRATOR_PROMPT + response}],
            options={"temperature": 0, "num_predict": 10},
        )
        verdict = result["message"]["content"].strip().upper()
        return "UNFULFILLED" in verdict
    except Exception:
        # If ollama isn't running or model not available, skip the check
        return False


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
    nudge_attempts = 0

    while turn < max_tool_turns:
        tool_calls = parse_tool_calls(response)

        # print any prose in the response
        print_llm(response)

        if not tool_calls:
            # If the LLM described actions without emitting tool calls,
            # re-prompt it to actually execute instead of just describing.
            if nudge_attempts < 2 and _has_unfulfilled_intent(response):
                nudge_attempts += 1
                print(c("  [nudging LLM to emit tool calls...]", DIM))
                try:
                    print(c("  thinking...", DIM), end="\r")
                    response = send_fn(_NUDGE_MSG)
                    print("              ", end="\r")
                except Exception as e:
                    print(c(f"\n  [browser error on nudge: {e}]\n", RED))
                    break
                turn += 1
                continue
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
                available = list(send_fns.keys())
                if len(available) > 1:
                    other = [m for m in available if m != current_llm][0]
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
        "--ollama-model",
        default="qwen2.5-coder:14b",
        help="Ollama model to use (default: qwen2.5-coder:14b)",
    )
    parser.add_argument(
        "--llm",
        choices=["chatgpt", "gemini", "deepseek", "ollama", "all"],
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

        if args.llm in ("gemini", "all"):
            page = open_gemini(browser)
            pages["gemini"] = page
            send_fns["gemini"] = lambda msg, pg=page: gemini_send(pg, msg)
            new_conv_fns["gemini"] = lambda pg=page: gemini_new(pg)

        if args.llm in ("deepseek", "all"):
            page = open_deepseek(browser)
            pages["deepseek"] = page
            send_fns["deepseek"] = lambda msg, pg=page: deepseek_send(pg, msg)
            new_conv_fns["deepseek"] = lambda pg=page: deepseek_new(pg)

        if args.llm in ("ollama", "all"):
            send_fns["ollama"] = lambda msg: ollama_send(args.ollama_model, msg)
            new_conv_fns["ollama"] = lambda: None  # Stateless API

        start_llm = "chatgpt" if args.llm == "chatgpt" else "gemini"

        try:
            interactive_shell(pages, send_fns, new_conv_fns, start_llm)
        finally:
            browser.close()
            print(c("\nBye.\n", DIM))


if __name__ == "__main__":
    main()
