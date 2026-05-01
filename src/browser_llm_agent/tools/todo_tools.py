import json
import os
import uuid
from datetime import datetime

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


def todo_delete(todo_id: str) -> str:
    todos = _load()
    before = len(todos)
    todos = [t for t in todos if t["id"] != todo_id]
    if len(todos) == before:
        return f"Error: todo '{todo_id}' not found"
    _save(todos)
    return f"Deleted [{todo_id}]"
