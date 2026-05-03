"""Ollama backend for rawagent.

Uses Ollama's local HTTP API directly. No browser and no API key required.
"""

import json
import urllib.error
import urllib.request


class OllamaChat:
    def __init__(self, model: str = "llama3", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.messages: list[dict[str, str]] = []

    def new_conversation(self):
        self.messages.clear()

    def send_message(self, message: str) -> str:
        self.messages.append({"role": "user", "content": message})
        payload = {
            "model": self.model,
            "messages": self.messages,
            "stream": False,
        }

        try:
            req = urllib.request.Request(
                f"{self.base_url}/api/chat",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=600) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            return f"Error connecting to Ollama: HTTP {e.code}: {body}"
        except urllib.error.URLError as e:
            return f"Error connecting to Ollama at {self.base_url}: {e}. Make sure `ollama serve` is running."
        except TimeoutError:
            return "Error connecting to Ollama: request timed out."
        except Exception as e:
            return f"Error connecting to Ollama: {e}"

        content = data.get("message", {}).get("content", "")
        if not content:
            return f"Error: Ollama returned no message content: {data}"

        self.messages.append({"role": "assistant", "content": content})
        return content


def create_chat(model: str = "llama3", base_url: str = "http://localhost:11434") -> OllamaChat:
    return OllamaChat(model=model, base_url=base_url)
