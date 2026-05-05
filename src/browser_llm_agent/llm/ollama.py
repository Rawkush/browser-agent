"""Ollama backend for rawagent.

Uses Ollama's local HTTP API directly. No browser and no API key required.
"""

import json
import re
import urllib.error
import urllib.request


# ── Context compaction ─────────────────────────────────────────────────────────

# Compact when the serialised conversation exceeds this many characters.
# ~4 chars/token → 80 000 chars ≈ 20 K tokens, leaving headroom for the reply.
_MAX_CONTEXT_CHARS = 80_000

# Always keep this many of the most-recent messages intact during compaction
# so the model retains immediate working context.
_KEEP_RECENT = 6

# Prompt sent to the model to produce the compaction summary
_COMPACT_PROMPT = (
    "Summarize the conversation above into a dense context block. "
    "Include: the original task, every file read or edited (with paths), "
    "every command run and its key output, the current state of the codebase, "
    "errors encountered and how they were addressed, and what remains to be done. "
    "Be specific and concise — this summary will replace the full conversation history."
)

# Keywords that indicate Ollama rejected the request because of context size
_CONTEXT_ERR_KW = ("context", "token", "length", "memory", "exceed", "kv cache")

_DIM    = "\033[2m"
_YELLOW = "\033[33m"
_RESET  = "\033[0m"


# ── Reasoning-model system prompt ──────────────────────────────────────────────

_REASONING_SYSTEM = (
    "You are a reasoning agent. Think through problems carefully before responding.\n\n"
    "When you need to write or generate code, output a code delegation block:\n\n"
    "<write_code>\n"
    "Describe precisely what code to write: language, purpose, function signatures,\n"
    "inputs/outputs, and any constraints or requirements.\n"
    "</write_code>\n\n"
    "A specialized coding model will write the code and return it to you.\n"
    "You can then incorporate it into your final response.\n"
    "For answers that do not require code, respond directly."
)


# ── Shared HTTP helper ─────────────────────────────────────────────────────────

def _http_call(base_url: str, model: str, messages: list[dict], timeout: int = 600) -> str:
    """POST to Ollama /api/chat and return the assistant content string.

    On any error, returns a string starting with "Error …".
    """
    payload = {"model": model, "messages": messages, "stream": False}
    try:
        req = urllib.request.Request(
            f"{base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        return f"Error connecting to Ollama: HTTP {e.code}: {body}"
    except urllib.error.URLError as e:
        return f"Error connecting to Ollama at {base_url}: {e}. Make sure `ollama serve` is running."
    except TimeoutError:
        return "Error connecting to Ollama: request timed out."
    except Exception as e:  # noqa: BLE001
        return f"Error connecting to Ollama: {e}"

    content = data.get("message", {}).get("content", "")
    return content if content else f"Error: Ollama returned no message content: {data}"


def _is_context_error(text: str) -> bool:
    lower = text.lower()
    return any(kw in lower for kw in _CONTEXT_ERR_KW)


def _context_chars(messages: list[dict]) -> int:
    return sum(len(m.get("content", "")) for m in messages)


# ── OllamaChat ─────────────────────────────────────────────────────────────────

class OllamaChat:
    def __init__(
        self,
        model: str = "qwen2.5-coder:14b",
        base_url: str = "http://localhost:11434",
        max_context_chars: int = _MAX_CONTEXT_CHARS,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_context_chars = max_context_chars
        self.messages: list[dict] = []

    def new_conversation(self):
        self.messages.clear()

    # ── compaction ────────────────────────────────────────────────────────────

    def _compact(self) -> None:
        """Summarise old messages into a single context block, keep recent ones."""
        n = len(self.messages)
        if n <= _KEEP_RECENT:
            # Nothing substantial to summarise — just wipe the oldest half.
            self.messages = self.messages[n // 2 :]
            print(f"{_DIM}  [context compacted: dropped oldest {n // 2} messages]{_RESET}")
            return

        to_summarise = self.messages[:-_KEEP_RECENT]
        recent       = self.messages[-_KEEP_RECENT:]

        print(f"{_DIM}  [compacting context: summarising {len(to_summarise)} messages…]{_RESET}")

        summary = _http_call(
            self.base_url,
            self.model,
            to_summarise + [{"role": "user", "content": _COMPACT_PROMPT}],
        )

        if summary.startswith("Error"):
            # Summarisation itself failed — fall back to dropping old messages
            print(f"{_YELLOW}  [compaction summary failed, dropping old messages]{_RESET}")
            self.messages = recent
        else:
            self.messages = [
                {
                    "role": "user",
                    "content": f"[Compacted conversation summary]\n{summary}",
                },
                {
                    "role": "assistant",
                    "content": "Understood. I have the full context from the summary above.",
                },
                *recent,
            ]
            print(f"{_DIM}  [context compacted: {len(to_summarise)} messages → summary]{_RESET}")

    # ── public API ────────────────────────────────────────────────────────────

    def send_message(self, message: str) -> str:
        self.messages.append({"role": "user", "content": message})

        # Proactively compact before hitting the limit
        if _context_chars(self.messages) > self.max_context_chars:
            self._compact()

        content = _http_call(self.base_url, self.model, self.messages)

        # Reactively compact if Ollama rejected the request due to context size
        if content.startswith("Error") and _is_context_error(content):
            print(f"{_YELLOW}  [context window exceeded — compacting and retrying]{_RESET}")
            self._compact()
            content = _http_call(self.base_url, self.model, self.messages)

        if content.startswith("Error"):
            return content

        self.messages.append({"role": "assistant", "content": content})
        return content


# ── OllamaReasoningChat ────────────────────────────────────────────────────────

class OllamaReasoningChat:
    """Two-model Ollama chat: a reasoning model plans and delegates code writing to a coder model.

    The reasoning model (e.g. deepseek-r1) decides what to do. When it needs code written,
    it emits a <write_code>...</write_code> block which is routed to the coder model
    (e.g. qwen2.5-coder). The resulting code is fed back to the reasoning model so it can
    produce a final, complete response.
    """

    def __init__(
        self,
        reasoning_model: str = "deepseek-r1:14b",
        coder_model: str = "qwen2.5-coder:14b",
        base_url: str = "http://localhost:11434",
        max_delegations: int = 5,
        max_context_chars: int = _MAX_CONTEXT_CHARS,
    ):
        self.reasoning_model = reasoning_model
        self.coder_model = coder_model
        self.base_url = base_url.rstrip("/")
        self.max_delegations = max_delegations
        self.max_context_chars = max_context_chars
        self.messages: list[dict] = [{"role": "system", "content": _REASONING_SYSTEM}]

    def new_conversation(self):
        self.messages = [{"role": "system", "content": _REASONING_SYSTEM}]

    # ── compaction ────────────────────────────────────────────────────────────

    def _compact(self) -> None:
        """Summarise old messages, preserving the system prompt and recent messages."""
        # Always keep the system message at index 0
        system = self.messages[:1]
        rest   = self.messages[1:]

        n = len(rest)
        if n <= _KEEP_RECENT:
            self.messages = system + rest[n // 2 :]
            print(f"{_DIM}  [context compacted: dropped oldest {n // 2} messages]{_RESET}")
            return

        to_summarise = rest[:-_KEEP_RECENT]
        recent       = rest[-_KEEP_RECENT:]

        print(f"{_DIM}  [compacting context: summarising {len(to_summarise)} messages…]{_RESET}")

        summary = _http_call(
            self.base_url,
            self.reasoning_model,
            system + to_summarise + [{"role": "user", "content": _COMPACT_PROMPT}],
        )

        if summary.startswith("Error"):
            print(f"{_YELLOW}  [compaction summary failed, dropping old messages]{_RESET}")
            self.messages = system + recent
        else:
            self.messages = system + [
                {
                    "role": "user",
                    "content": f"[Compacted conversation summary]\n{summary}",
                },
                {
                    "role": "assistant",
                    "content": "Understood. I have the full context from the summary above.",
                },
                *recent,
            ]
            print(f"{_DIM}  [context compacted: {len(to_summarise)} messages → summary]{_RESET}")

    # ── internal helpers ──────────────────────────────────────────────────────

    def _call(self, model: str, messages: list[dict]) -> str:
        return _http_call(self.base_url, model, messages)

    def _extract_code_specs(self, text: str) -> list[str]:
        return re.findall(r"<write_code>(.*?)</write_code>", text, re.DOTALL)

    def _strip_code_blocks(self, text: str) -> str:
        return re.sub(r"<write_code>.*?</write_code>", "", text, flags=re.DOTALL).strip()

    # ── public API ────────────────────────────────────────────────────────────

    def send_message(self, message: str) -> str:
        self.messages.append({"role": "user", "content": message})

        # Proactively compact before hitting the limit
        if _context_chars(self.messages) > self.max_context_chars:
            self._compact()

        for _ in range(self.max_delegations):
            response = self._call(self.reasoning_model, self.messages)

            # Reactively compact if context window was exceeded
            if response.startswith("Error") and _is_context_error(response):
                print(f"{_YELLOW}  [context window exceeded — compacting and retrying]{_RESET}")
                self._compact()
                response = self._call(self.reasoning_model, self.messages)

            if response.startswith("Error"):
                return response

            specs = self._extract_code_specs(response)
            if not specs:
                # No delegation needed — reasoning model answered directly
                self.messages.append({"role": "assistant", "content": response})
                return response

            # Print any reasoning prose before the delegation blocks
            visible = self._strip_code_blocks(response)
            if visible:
                print(f"\n\033[32m{visible}\033[0m")

            print(f"\n\033[33m  [{self.reasoning_model} → {self.coder_model}: {len(specs)} code delegation(s)]\033[0m")
            self.messages.append({"role": "assistant", "content": response})

            # Delegate each spec to the coder model
            code_results = []
            for i, spec in enumerate(specs, 1):
                spec = spec.strip()
                short = spec[:70] + ("..." if len(spec) > 70 else "")
                print(f"\033[33m  [coder({i}/{len(specs)}): {short}]\033[0m")
                code = self._call(self.coder_model, [{"role": "user", "content": spec}])
                code_results.append(f"Delegation {i} — spec: {spec}\n\nCode:\n{code}")

            feedback = (
                f"The coding model ({self.coder_model}) wrote the following:\n\n"
                + "\n\n---\n\n".join(code_results)
            )
            self.messages.append({"role": "user", "content": feedback})

        # Max delegations reached — get final response
        final = self._call(self.reasoning_model, self.messages)
        self.messages.append({"role": "assistant", "content": final})
        return final


# ── factories ──────────────────────────────────────────────────────────────────

def create_chat(model: str = "qwen2.5-coder:14b", base_url: str = "http://localhost:11434") -> OllamaChat:
    return OllamaChat(model=model, base_url=base_url)


def create_reasoning_chat(
    reasoning_model: str = "deepseek-r1:14b",
    coder_model: str = "qwen2.5-coder:14b",
    base_url: str = "http://localhost:11434",
) -> OllamaReasoningChat:
    return OllamaReasoningChat(
        reasoning_model=reasoning_model,
        coder_model=coder_model,
        base_url=base_url,
    )
