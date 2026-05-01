import os


def read_file(path: str) -> str:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return f"Error: file not found: {path}"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    lines = content.splitlines()
    numbered = "\n".join(f"{i+1}: {line}" for i, line in enumerate(lines))
    return numbered


def write_file(path: str, content: str) -> str:
    path = os.path.expanduser(path)
    os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"Written: {path}"


def edit_file(path: str, old: str, new: str) -> str:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return f"Error: file not found: {path}"
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    if old not in content:
        return f"Error: string not found in {path}"
    updated = content.replace(old, new, 1)
    with open(path, "w", encoding="utf-8") as f:
        f.write(updated)
    return f"Edited: {path}"


def list_dir(path: str = ".") -> str:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return f"Error: path not found: {path}"
    entries = []
    for item in sorted(os.listdir(path)):
        full = os.path.join(path, item)
        prefix = "d" if os.path.isdir(full) else "f"
        entries.append(f"[{prefix}] {item}")
    return "\n".join(entries) if entries else "(empty)"
