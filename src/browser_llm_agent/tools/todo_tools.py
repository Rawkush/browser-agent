import json
import os
import uuid
from datetime import datetime

from browser_llm_agent.tools.registry import tool

TODO_FILE = os.path.expanduser("~/.llm-agent/todos.json")


def _load() -> list[dict]:
    if not os.path.exists(TODO_FILE):
        return []
    with open(TODO_FILE) as f:
        return json.load(f)


def _save(todos: list[dict]):
    os.makedirs(os.path.dirname(TODO_FILE), exist_ok=True)
    with open(TODO_FILE, "w") as f:
        json.dump(todos, f, indent=2)


@tool("todo_add", "Add a todo item.", {
    "type": "object",
    "properties": {
        "content": {"type": "string"},
        "priority": {"type": "string", "enum": ["high", "medium", "low"]},
    },
    "required": ["content"],
})
def todo_add(content: str, priority: str = "medium") -> str:
    todos = _load()
    item = {
        "id": str(uuid.uuid4())[:8],
        "content": content,
        "status": "pending",
        "priority": priority,
        "created": datetime.now().isoformat(),
    }
    todos.append(item)
    _save(todos)
    return f"Added [{item['id']}]: {content}"


@tool("todo_list", "List todos, optionally filtered by status.", {
    "type": "object",
    "properties": {
        "status": {"type": "string", "enum": ["pending", "in_progress", "done"]},
    },
    "required": [],
})
def todo_list(status: str = None) -> str:
    todos = _load()
    if status:
        todos = [t for t in todos if t["status"] == status]
    if not todos:
        return "No todos found."
    icons = {"pending": "○", "in_progress": "◑", "done": "✓"}
    lines = []
    for t in todos:
        icon = icons.get(t["status"], "?")
        lines.append(f"[{t['id']}] {icon} [{t['priority']}] {t['content']}")
    return "\n".join(lines)


@tool("todo_update", "Update the status of a todo item.", {
    "type": "object",
    "properties": {
        "todo_id": {"type": "string"},
        "status": {"type": "string", "enum": ["pending", "in_progress", "done"]},
    },
    "required": ["todo_id", "status"],
})
def todo_update(todo_id: str, status: str) -> str:
    valid = {"pending", "in_progress", "done"}
    if status not in valid:
        return f"Error: status must be one of {valid}"
    todos = _load()
    for t in todos:
        if t["id"] == todo_id:
            t["status"] = status
            t["updated"] = datetime.now().isoformat()
            _save(todos)
            return f"Updated [{todo_id}] → {status}"
    return f"Error: todo '{todo_id}' not found"


@tool("todo_delete", "Delete a todo item.", {
    "type": "object",
    "properties": {"todo_id": {"type": "string"}},
    "required": ["todo_id"],
})
def todo_delete(todo_id: str) -> str:
    todos = _load()
    before = len(todos)
    todos = [t for t in todos if t["id"] != todo_id]
    if len(todos) == before:
        return f"Error: todo '{todo_id}' not found"
    _save(todos)
    return f"Deleted [{todo_id}]"


@tool("plan_create", "Create a multi-step plan. Each step becomes a todo item.", {
    "type": "object",
    "properties": {
        "steps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "List of plan steps in order",
        },
    },
    "required": ["steps"],
})
def plan_create(steps: list) -> str:
    results = []
    for i, step in enumerate(steps, 1):
        results.append(todo_add(f"Step {i}: {step}", priority="high"))
    return "\n".join(results)
