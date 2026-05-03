"""
Claude API backend for rawagent.

Uses the Anthropic Python SDK with native tool calling — no browser,
no text parsing. Requires ANTHROPIC_API_KEY in the environment.
"""

import os
import anthropic
from browser_llm_agent.prompts import CODEX_STYLE_AGENT_PROMPT

# ── System prompt (no tool-format instructions — Claude uses native calling) ──

SYSTEM_PROMPT = CODEX_STYLE_AGENT_PROMPT


# ── Client factory ─────────────────────────────────────────────────────────────

def create_client(api_key: str | None = None) -> anthropic.Anthropic:
    """Return an Anthropic client. Uses ANTHROPIC_API_KEY env var by default."""
    return anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
