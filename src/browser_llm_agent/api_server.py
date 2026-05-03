#!/usr/bin/env python3
"""
rawagent API server — Anthropic-compatible HTTP API backed by rawagent.

Supports three backend modes:
  --llm gemini    Browser automation (Playwright thread)
  --llm chatgpt   Browser automation (Playwright thread)
  --llm claude    Direct Claude API (no browser, no Playwright)

Threading model (browser modes)
-------------------------------
Playwright sync API must run on the thread it was created on.
So all agent work is dispatched via a queue to the Playwright thread.

  HTTP server thread  →  _REQUEST_QUEUE  →  Playwright thread
                      ←  concurrent.futures.Future  ←

Threading model (claude mode)
-----------------------------
No Playwright. Agent runs directly on the request handler thread.
Each request creates a fresh Claude API conversation.

Usage
-----
Terminal 1:
    python -m browser_llm_agent.api_server --llm gemini --port 8765
    python -m browser_llm_agent.api_server --llm claude --port 8765

Terminal 2:
    ANTHROPIC_BASE_URL=http://localhost:8765 ANTHROPIC_API_KEY=rawagent claude
"""

import argparse
import concurrent.futures
import json
import os
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler

# Import tools package to trigger @tool decorator registration
import browser_llm_agent.tools  # noqa: F401

from browser_llm_agent.cli import (
    build_system_prompt,
    parse_tool_calls,
    strip_tool_blocks,
    print_tool_call,
    print_tool_result,
    c, DIM, GREEN, RED, BLUE, BOLD,
)
from browser_llm_agent.tools.registry import execute_tool, get_claude_tools
from browser_llm_agent.mcp_client import MCPManager


# ── Conversation state ───────────────────────────────────────────────────────

@dataclass
class ConversationState:
    is_first_message: bool = True
    last_activity: float = field(default_factory=time.time)
    messages: list = field(default_factory=list)  # For Claude backend


_CONVERSATIONS: dict[str, ConversationState] = {}
_CONV_LOCK = threading.Lock()
_CONV_TTL = 1800  # 30 minutes


def _get_conversation(conv_id: str) -> ConversationState:
    """Get or create a conversation state."""
    with _CONV_LOCK:
        # Cleanup old conversations
        now = time.time()
        expired = [k for k, v in _CONVERSATIONS.items() if now - v.last_activity > _CONV_TTL]
        for k in expired:
            del _CONVERSATIONS[k]

        if conv_id not in _CONVERSATIONS:
            _CONVERSATIONS[conv_id] = ConversationState()
        conv = _CONVERSATIONS[conv_id]
        conv.last_activity = now
        return conv


# ── Shared request queue (HTTP thread → Playwright thread) ────────────────────

_REQUEST_QUEUE: queue.Queue = queue.Queue()

# Module-level references for claude mode
_CLAUDE_CLIENT = None
_MCP_MANAGER: MCPManager | None = None


def _compose_system_prompt(default_system: str | None, request_system: str | None) -> str:
    """Keep rawagent's tool contract while preserving client-provided context."""
    default_system = (default_system or "").strip()
    request_system = (request_system or "").strip()
    if default_system and request_system:
        return (
            default_system
            + "\n\nClient system prompt/context follows. Rawagent's tool format, tool names, "
              "workspace-safety rules, and verification rules above take priority.\n\n"
            + request_system
        )
    return default_system or request_system


# ── Agent loop (browser backends — runs on Playwright thread only) ────────────

def run_agent(send_fn, user_message: str, is_first_message: bool,
              system_prompt: str, mcp_manager: MCPManager | None = None) -> str:
    """Run full rawagent tool loop. Must be called from the Playwright thread."""
    if is_first_message and system_prompt:
        full_message = f"{system_prompt}\n\n{user_message}"
    else:
        full_message = user_message

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


def process_requests(send_fn, default_system: str | None,
                     mcp_manager: MCPManager | None = None):
    """
    Blocking queue loop — MUST run on the Playwright thread.
    Queue items: (user_msg, request_system, conv_id, future)
    Exits when None is put in the queue (shutdown signal).
    """
    while True:
        item = _REQUEST_QUEUE.get()
        if item is None:
            break
        user_msg, request_system, conv_id, future = item
        conv = _get_conversation(conv_id)
        system = _compose_system_prompt(default_system, request_system)
        try:
            answer = run_agent(send_fn, user_msg, conv.is_first_message, system, mcp_manager)
            conv.is_first_message = False
            future.set_result(answer)
        except Exception as exc:
            conv.is_first_message = False
            future.set_exception(exc)


# ── Claude API agent loop (no browser) ───────────────────────────────────────

def run_claude_agent(user_message: str, conv_id: str,
                     request_system: str, mcp_manager: MCPManager | None = None) -> str:
    """Run a Claude API agent turn with native tool calling."""
    from browser_llm_agent.llm.claude import SYSTEM_PROMPT as CLAUDE_SYSTEM_PROMPT

    conv = _get_conversation(conv_id)
    conv.messages.append({"role": "user", "content": user_message})

    system = _compose_system_prompt(CLAUDE_SYSTEM_PROMPT, request_system)
    tools = get_claude_tools(mcp_manager)

    for _ in range(20):
        print(c("  [claude] thinking...", DIM), flush=True)
        response = _CLAUDE_CLIENT.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=8096,
            system=system,
            tools=tools,
            messages=conv.messages,
        )

        conv.messages.append({"role": "assistant", "content": response.content})

        text_blocks = [b for b in response.content if b.type == "text"]
        tool_uses = [b for b in response.content if b.type == "tool_use"]

        for block in text_blocks:
            if block.text.strip():
                print(f"\n{c(block.text.strip(), GREEN)}\n", flush=True)

        if not tool_uses:
            # Return last text
            texts = [b.text for b in text_blocks if b.text.strip()]
            return "\n".join(texts) if texts else "(no response)"

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

        conv.messages.append({"role": "user", "content": tool_results})
    else:
        print(c("  [claude] reached max tool turns", RED), flush=True)

    return "(max tool turns reached)"


# ── HTTP handler ──────────────────────────────────────────────────────────────

class RawAgentHandler(BaseHTTPRequestHandler):
    # Set by main() to select backend
    use_claude_backend = False

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

        # Extract system prompt from request
        raw_system = body.get("system", "")
        if isinstance(raw_system, list):
            request_system = "\n".join(
                b["text"] for b in raw_system
                if isinstance(b, dict) and b.get("type") == "text"
            )
        else:
            request_system = raw_system or ""

        # Conversation ID from header or generate one
        conv_id = self.headers.get("x-conversation-id", str(uuid.uuid4())[:8])

        if self.use_claude_backend:
            # Claude mode: run directly, no Playwright thread needed
            try:
                answer = run_claude_agent(user_msg, conv_id, request_system, _MCP_MANAGER)
            except Exception as exc:
                answer = f"Error: {exc}"
        else:
            # Browser mode: dispatch to Playwright thread
            future: concurrent.futures.Future = concurrent.futures.Future()
            _REQUEST_QUEUE.put((user_msg, request_system, conv_id, future))
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
    global _CLAUDE_CLIENT, _MCP_MANAGER

    parser = argparse.ArgumentParser(
        description="rawagent API server — Claude Code CLI backed by rawagent"
    )
    parser.add_argument("--llm", choices=["chatgpt", "gemini", "claude", "ollama"], default="gemini")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--host", default="127.0.0.1")
    args = parser.parse_args()

    print(c(f"\n  rawagent API server  [{args.llm}]", BOLD + BLUE))
    print(c(f"  http://{args.host}:{args.port}", BLUE))
    print(c("─" * 60, BLUE))

    mcp_manager = MCPManager()
    n = mcp_manager.load_and_connect(status_cb=lambda msg: print(c(msg, DIM)))
    if n:
        print(c(f"  {n} MCP server(s) ready", DIM), flush=True)
    _MCP_MANAGER = mcp_manager

    if args.llm == "claude":
        # ── Claude mode: no browser, no Playwright ──────────────────────────
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print(c("  Error: ANTHROPIC_API_KEY not set.", RED))
            return

        import anthropic
        _CLAUDE_CLIENT = anthropic.Anthropic(api_key=api_key)
        RawAgentHandler.use_claude_backend = True

        # Enable agent spawning
        from browser_llm_agent.tools.agent_tools import _configure as _configure_agents
        _configure_agents(api_key)

        http_server = HTTPServer((args.host, args.port), RawAgentHandler)

        print(c(f"\n  Ready (Claude API mode — no browser needed).", BOLD))
        print(c(f"  ANTHROPIC_BASE_URL=http://{args.host}:{args.port} ANTHROPIC_API_KEY=rawagent claude", GREEN))
        print(c("─" * 60, BLUE) + "\n")

        try:
            http_server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            http_server.shutdown()
            mcp_manager.stop_all()
            print(c("\nBye.\n", DIM))

    else:
        # ── Browser mode: Playwright thread ─────────────────────────────────
        from playwright.sync_api import sync_playwright
        from browser_llm_agent.llm.chatgpt import open_chatgpt, send_message as chatgpt_send
        from browser_llm_agent.llm.gemini import open_gemini, send_message as gemini_send

        print(c("  Starting browser...", DIM), flush=True)

        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=False, args=["--remote-debugging-port=9222"])

        if args.llm == "chatgpt":
            page = open_chatgpt(browser)
            send_fn = lambda msg: chatgpt_send(page, msg)
        else:
            page = open_gemini(browser)
            send_fn = lambda msg: gemini_send(page, msg)

        # HTTP server runs in background thread; Playwright stays on main thread
        RawAgentHandler.use_claude_backend = False
        http_server = HTTPServer((args.host, args.port), RawAgentHandler)
        t = threading.Thread(target=http_server.serve_forever, daemon=True)
        t.start()

        system_prompt = build_system_prompt(mcp_manager)

        print(c(f"\n  Ready. In another terminal:", BOLD))
        print(c(f"  ANTHROPIC_BASE_URL=http://{args.host}:{args.port} ANTHROPIC_API_KEY=rawagent claude", GREEN))
        print(c("─" * 60, BLUE) + "\n")

        try:
            process_requests(send_fn, system_prompt, mcp_manager)
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
