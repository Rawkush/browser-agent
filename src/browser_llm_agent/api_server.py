#!/usr/bin/env python3
"""
rawagent API server — Anthropic-compatible HTTP API backed by rawagent.

Threading model
---------------
Playwright sync API must run on the thread it was created on.
So all agent work is dispatched via a queue to the Playwright thread.

  HTTP server thread  →  _REQUEST_QUEUE  →  Playwright thread
                      ←  concurrent.futures.Future  ←

Usage
-----
Terminal 1:
    python -m browser_llm_agent.api_server --llm gemini --port 8765

Terminal 2:
    ANTHROPIC_BASE_URL=http://localhost:8765 ANTHROPIC_API_KEY=rawagent claude
"""

import argparse
import concurrent.futures
import json
import queue
import threading
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler

from browser_llm_agent.cli import (
    SYSTEM_PROMPT,
    parse_tool_calls,
    strip_tool_blocks,
    execute_tool,
    print_tool_call,
    print_tool_result,
    c, DIM, GREEN, RED, BLUE, BOLD,
)
from browser_llm_agent.mcp_client import MCPManager


# ── Shared request queue (HTTP thread → Playwright thread) ────────────────────

_REQUEST_QUEUE: queue.Queue = queue.Queue()


# ── Agent loop (runs on Playwright thread only) ───────────────────────────────

def run_agent(send_fn, user_message: str, is_first_message: bool,
              system_prompt: str, mcp_manager: MCPManager | None = None) -> str:
    """Run full rawagent tool loop. Must be called from the Playwright thread."""
    full_message = f"{system_prompt}\n\n{user_message}" if is_first_message else user_message

    print(c("  [rawagent] thinking...", DIM), flush=True)
    response = send_fn(full_message)

    final_text = ""
    for _ in range(20):
        tool_calls = parse_tool_calls(response)
        prose = strip_tool_blocks(response).strip()
        if prose:
            print(f"\n{c(prose, GREEN)}\n", flush=True)
            final_text = prose

        if not tool_calls:
            break

        results = []
        for call in tool_calls:
            print_tool_call(call)
            result = execute_tool(call, mcp_manager)
            print_tool_result(result)
            results.append({"tool": call.get("name", "?"), "result": result})

        result_text = "\n".join(
            f"Tool `{r['tool']}` result:\n```\n{r['result']}\n```"
            for r in results
        )
        time.sleep(1)
        print(c("  [rawagent] thinking...", DIM), flush=True)
        response = send_fn(result_text)
    else:
        print(c("  [rawagent] reached max tool turns", RED), flush=True)

    return final_text or strip_tool_blocks(response)


def process_requests(send_fn, system_prompt: str,
                     mcp_manager: MCPManager | None = None):
    """
    Blocking queue loop — MUST run on the Playwright thread.
    Processes agent requests forwarded by the HTTP handler.
    Exits when None is put in the queue (shutdown signal).
    """
    is_first = True
    while True:
        item = _REQUEST_QUEUE.get()
        if item is None:
            break
        user_msg, future = item
        try:
            answer = run_agent(send_fn, user_msg, is_first, system_prompt, mcp_manager)
            is_first = False
            future.set_result(answer)
        except Exception as exc:
            is_first = False
            future.set_exception(exc)


# ── HTTP handler ──────────────────────────────────────────────────────────────

class RawAgentHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        if "HEAD" not in (fmt % args):
            print(c(f"  [http] {fmt % args}", DIM), flush=True)

    def _path(self) -> str:
        return self.path.split("?")[0].rstrip("/")

    def do_HEAD(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        if self._path() == "/v1/models":
            body = json.dumps({
                "object": "list",
                "data": [{"id": "claude-sonnet-4-6", "object": "model",
                          "created": 0, "owned_by": "rawagent"}],
            }).encode()
            self._reply(200, "application/json", body)
        else:
            self._reply(200, "text/plain", b"rawagent ok")

    def do_POST(self):
        if self._path() != "/v1/messages":
            self._reply(404, "text/plain", b"not found")
            return

        length = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(length))
        except Exception:
            self._reply(400, "text/plain", b"bad json")
            return

        user_msg = self._extract_last_user_message(body)
        if not user_msg:
            self._reply(400, "text/plain", b"no user message found in request")
            return

        model = body.get("model", "claude-sonnet-4-6")
        want_stream = body.get("stream", False)

        # Dispatch to Playwright thread and wait for result
        future: concurrent.futures.Future = concurrent.futures.Future()
        _REQUEST_QUEUE.put((user_msg, future))
        try:
            answer = future.result(timeout=300)  # 5 min max
        except Exception as exc:
            answer = f"Error: {exc}"

        if want_stream:
            self._stream_response(answer, model)
        else:
            self._json_response(answer, model)

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_last_user_message(self, body: dict) -> str:
        for msg in reversed(body.get("messages", [])):
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(
                    b["text"] for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
        return ""

    def _json_response(self, text: str, model: str = "claude-sonnet-4-6"):
        payload = {
            "id": f"msg_{uuid.uuid4().hex[:24]}",
            "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": text}],
            "model": model, "stop_reason": "end_turn", "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": len(text.split())},
        }
        self._reply(200, "application/json", json.dumps(payload).encode())

    def _stream_response(self, text: str, model: str = "claude-sonnet-4-6"):
        msg_id = f"msg_{uuid.uuid4().hex[:24]}"
        chunks = [text[i:i+20] for i in range(0, len(text), 20)] or [""]

        events = [
            ("message_start", {"type": "message_start", "message": {
                "id": msg_id, "type": "message", "role": "assistant",
                "content": [], "model": model,
                "stop_reason": None, "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            }}),
            ("content_block_start", {"type": "content_block_start", "index": 0,
                                     "content_block": {"type": "text", "text": ""}}),
            ("ping", {"type": "ping"}),
            *[("content_block_delta", {"type": "content_block_delta", "index": 0,
                                       "delta": {"type": "text_delta", "text": ch}})
              for ch in chunks],
            ("content_block_stop", {"type": "content_block_stop", "index": 0}),
            ("message_delta", {"type": "message_delta",
                               "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                               "usage": {"output_tokens": len(text.split())}}),
            ("message_stop", {"type": "message_stop"}),
        ]

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.end_headers()

        for event_type, data in events:
            try:
                self.wfile.write(f"event: {event_type}\ndata: {json.dumps(data)}\n\n".encode())
                self.wfile.flush()
            except BrokenPipeError:
                break

    def _reply(self, status: int, content_type: str, body: bytes):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


# ── Entry point (standalone server) ──────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="rawagent API server — Claude Code CLI backed by rawagent"
    )
    parser.add_argument("--llm", choices=["chatgpt", "gemini"], default="gemini")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    from playwright.sync_api import sync_playwright
    from browser_llm_agent.llm.chatgpt import open_chatgpt, send_message as chatgpt_send
    from browser_llm_agent.llm.gemini import open_gemini, send_message as gemini_send

    print(c(f"\n  rawagent API server  [{args.llm}]", BOLD + BLUE))
    print(c(f"  http://{args.host}:{args.port}", BLUE))
    print(c("─" * 60, BLUE))
    print(c("  Starting browser...", DIM), flush=True)

    playwright = sync_playwright().start()
    browser = playwright.chromium.launch(headless=False, args=["--remote-debugging-port=9222"])

    if args.llm == "chatgpt":
        page = open_chatgpt(browser)
        send_fn = lambda msg: chatgpt_send(page, msg)
    else:
        page = open_gemini(browser)
        send_fn = lambda msg: gemini_send(page, msg)

    mcp_manager = MCPManager()
    n = mcp_manager.load_and_connect(status_cb=lambda msg: print(c(msg, DIM)))
    if n:
        print(c(f"  {n} MCP server(s) ready", DIM), flush=True)

    # HTTP server runs in background thread; Playwright stays on main thread
    http_server = HTTPServer((args.host, args.port), RawAgentHandler)
    t = threading.Thread(target=http_server.serve_forever, daemon=True)
    t.start()

    print(c(f"\n  Ready. In another terminal:", BOLD))
    print(c(f"  ANTHROPIC_BASE_URL=http://{args.host}:{args.port} ANTHROPIC_API_KEY=rawagent claude", GREEN))
    print(c("─" * 60, BLUE) + "\n")

    try:
        # Main thread = Playwright thread: process agent requests from queue
        process_requests(send_fn, SYSTEM_PROMPT, mcp_manager)
    except KeyboardInterrupt:
        pass
    finally:
        http_server.shutdown()
        mcp_manager.stop_all()
        browser.close()
        playwright.stop()
        print(c("\nBye.\n", DIM))


if __name__ == "__main__":
    main()
