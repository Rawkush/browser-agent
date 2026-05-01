# rawagent

An interactive coding shell powered by ChatGPT or Gemini via browser automation. No API keys needed — uses your existing subscriptions.

## Why I built this

I wanted a coding agent like Claude Code or GitHub Copilot — something that could read files, run commands, and fix code autonomously in the terminal.

The problem:

- **Claude Code and Codex are expensive.** API pricing at $3–15 per million tokens adds up fast with heavy daily use. The subscription plans ($100–200/month) are cheaper but still significant.
- **They have strict rate limits.** Hit the limit mid-task and you're blocked.
- **Running models locally sounds great but sucks in practice.** Local models (even 32B) are noticeably worse at coding than GPT-4o or Gemini. You need a good GPU (~$800+), and even then token speed is slow.

The insight: most people already pay for ChatGPT Plus or Gemini Advanced ($20/month each) and those subscriptions have generous limits on the web interface — but no way to use them programmatically without paying again for API access.

rawagent bridges that gap. It drives a real Chrome browser so it looks like normal web usage, and exposes the LLM as a tool-calling coding agent — giving you a Claude Code-like experience at no extra cost.

## Requirements

- Python 3.10+
- Google Chrome (installed)
- A ChatGPT or Gemini account (free or paid)

## Install

```bash
git clone https://github.com/yourname/browser-llm-agent
cd browser-llm-agent

pip install -e .
playwright install chromium
```

## Run

```bash
# use Gemini (recommended — free)
rawagent --llm gemini

# use ChatGPT
rawagent --llm chatgpt

# open both, start on Gemini
rawagent --llm both
```

On first launch a Chrome window opens. Log in to ChatGPT or Gemini manually, then press Enter in the terminal to start.

## Usage

Type naturally — just like Claude Code:

```
you [gemini]> create a fastapi app with a /health endpoint
you [gemini]> add input validation to the POST /users route
you [gemini]> run the tests and fix any failures
```

### Slash commands

| Command | Action |
|---------|--------|
| `/new` | Start a fresh conversation (clears LLM memory) |
| `/switch` | Swap between ChatGPT and Gemini |
| `/help` | Show help |
| `/quit` | Exit |

### Paste support

Paste multi-line code or text directly into the prompt — it is sent as a single message.

## Tools available

The agent can use these tools automatically:

| Category | Tools |
|----------|-------|
| Files | `read_file`, `write_file`, `edit_file`, `delete_file`, `move_file`, `list_dir`, `make_dir` |
| Search | `glob`, `grep`, `find_files` |
| Shell | `bash` |
| Web | `web_fetch` |
| Todos | `todo_add`, `todo_list`, `todo_update`, `todo_delete` |
| Memory | `memory_save`, `memory_get`, `memory_list`, `memory_search`, `memory_delete` |

### Persistent memory

The agent can remember things across sessions:

```
you [gemini]> remember that this project uses PostgreSQL and FastAPI
you [gemini]> what stack does this project use?
```

### Todo tracking

```
you [gemini]> add a todo: write tests for auth module
you [gemini]> show all todos
you [gemini]> mark todo abc123 as done
```

Todos and memory are stored in `~/.llm-agent/`.

## Data stored locally

| File | Contents |
|------|---------|
| `~/.llm-agent/memory.db` | Persistent memory (SQLite) |
| `~/.llm-agent/todos.json` | Todo list |
| `~/.llm-agent/history` | Input history (arrow keys) |

## Troubleshooting

**VS Code terminal not opening** — make sure the command name does not conflict with another tool in your shell. `rawagent` should be safe.

**Tool calls not executing** — the agent parses JSON from the LLM response. If the LLM changes its output format, restart with `/new` and try again.

**Selectors broken after a site update** — edit `src/browser_llm_agent/llm/chatgpt.py` or `gemini.py` and update the CSS selectors to match the current page structure.
