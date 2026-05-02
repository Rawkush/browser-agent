#!/usr/bin/env python3
"""
claude-raw — single command launcher.

Threading model
---------------
  Playwright thread  — init browser, run process_requests() loop
  HTTP server thread — daemon, forwards requests to Playwright thread via queue
  Main thread        — runs `claude` CLI subprocess, then triggers shutdown

Usage:
    claude-raw                   # uses gemini, port 8765
    claude-raw --llm chatgpt
    claude-raw --port 9000
    claude-raw -- --no-stream    # pass extra flags to claude CLI
"""

import argparse
import os
import subprocess
import sys
import threading
import time
import urllib.request

from browser_llm_agent.cli import build_system_prompt, c, DIM, GREEN, RED, BLUE, BOLD
from browser_llm_agent.mcp_client import MCPManager
from browser_llm_agent.api_server import (
    RawAgentHandler, process_requests, _REQUEST_QUEUE,
)


def _wait_ready(host: str, port: int, timeout: int = 60) -> bool:
    url = f"http://{host}:{port}/v1/models"
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except Exception:
            time.sleep(0.3)
    return False


def main():
    parser = argparse.ArgumentParser(
        description="claude-raw: Claude Code UI backed by rawagent brain",
    )
    parser.add_argument("--llm", choices=["chatgpt", "gemini"], default="gemini")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("claude_args", nargs=argparse.REMAINDER,
                        help="Extra args forwarded to claude CLI (after --)")
    args = parser.parse_args()

    claude_extra = args.claude_args
    if claude_extra and claude_extra[0] == "--":
        claude_extra = claude_extra[1:]

    print(c("\n  claude-raw", BOLD + BLUE))
    print(c(f"  brain: rawagent [{args.llm}]  |  port: {args.port}", BLUE))
    print(c("─" * 60, BLUE))

    # Shared state set by Playwright thread, read by main thread after ready event
    _pw_ready = threading.Event()
    _pw_error: list = []
    _resources: dict = {}   # holds playwright, browser, mcp_manager for cleanup

    def playwright_thread_fn():
        """
        Runs entirely on this thread — Playwright lives here.
        1. Init browser + LLM page
        2. Signal ready
        3. Run process_requests() loop (blocking, exits on queue None)
        """
        from playwright.sync_api import sync_playwright
        from browser_llm_agent.llm.chatgpt import open_chatgpt, send_message as chatgpt_send
        from browser_llm_agent.llm.gemini import open_gemini, send_message as gemini_send, new_conversation as gemini_new_convo

        try:
            print(c("  Starting browser...", DIM), flush=True)
            pw = sync_playwright().start()
            browser = pw.chromium.launch(
                headless=False, args=["--remote-debugging-port=9222"]
            )

            if args.llm == "chatgpt":
                page = open_chatgpt(browser)
                send_fn = lambda msg: chatgpt_send(page, msg)
            else:
                page = open_gemini(browser)
                gemini_new_convo(page)   # start a fresh conversation each session
                send_fn = lambda msg: gemini_send(page, msg)

            mcp_manager = MCPManager()
            n = mcp_manager.load_and_connect(status_cb=lambda msg: print(c(msg, DIM)))
            if n:
                print(c(f"  {n} MCP server(s) ready", DIM), flush=True)

            _resources["pw"] = pw
            _resources["browser"] = browser
            _resources["mcp"] = mcp_manager

            _pw_ready.set()

            # Blocking — processes agent requests forwarded from the HTTP thread
            # Pass None so we never fall back to rawagent's system prompt.
            # In claude-raw mode the system prompt comes from Claude Code's request.
            process_requests(send_fn, None, mcp_manager)

        except Exception as exc:
            _pw_error.append(exc)
            _pw_ready.set()   # unblock main thread even on error

        finally:
            mcp = _resources.get("mcp")
            bro = _resources.get("browser")
            pw  = _resources.get("pw")
            if mcp:  mcp.stop_all()
            if bro:  bro.close()
            if pw:   pw.stop()

    # ── Start Playwright thread ───────────────────────────────────────────────
    pw_thread = threading.Thread(target=playwright_thread_fn, daemon=True)
    pw_thread.start()

    _pw_ready.wait()
    if _pw_error:
        print(c(f"  Browser init failed: {_pw_error[0]}", RED))
        sys.exit(1)

    # ── Start HTTP server in a daemon thread ──────────────────────────────────
    from http.server import HTTPServer
    http_server = HTTPServer((args.host, args.port), RawAgentHandler)
    http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    http_thread.start()

    print(c("  Waiting for server...", DIM), end="", flush=True)
    if not _wait_ready(args.host, args.port):
        print(c(" FAILED", RED))
        _REQUEST_QUEUE.put(None)
        sys.exit(1)
    print(c(" ready", DIM), flush=True)

    # ── Launch claude CLI (main thread) ──────────────────────────────────────
    env = {
        **os.environ,
        "ANTHROPIC_BASE_URL": f"http://{args.host}:{args.port}",
        "ANTHROPIC_API_KEY": "rawagent",
    }
    print(c(f"\n  Launching claude CLI → rawagent brain\n", BOLD))

    try:
        subprocess.run(["claude"] + claude_extra, env=env)
    except FileNotFoundError:
        print(c("  Error: 'claude' not found. Install: npm i -g @anthropic-ai/claude-code", RED))
    except KeyboardInterrupt:
        pass
    finally:
        print(c("\n  Shutting down...", DIM), flush=True)
        http_server.shutdown()
        _REQUEST_QUEUE.put(None)   # signal Playwright thread to exit
        pw_thread.join(timeout=5)
        print(c("  Bye.\n", DIM))


if __name__ == "__main__":
    main()
