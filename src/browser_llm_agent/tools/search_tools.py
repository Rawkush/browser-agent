import glob as _glob
import os
import re
import subprocess
import urllib.request
import urllib.error


def glob(pattern: str, cwd: str = ".") -> str:
    cwd = os.path.expanduser(cwd)
    matches = _glob.glob(pattern, root_dir=cwd, recursive=True)
    if not matches:
        return f"No files matched: {pattern}"
    return "\n".join(sorted(matches))


def grep(pattern: str, path: str = ".", include: str = None, ignore_case: bool = False) -> str:
    path = os.path.expanduser(path)
    # prefer ripgrep, fall back to grep
    if subprocess.run(["which", "rg"], capture_output=True).returncode == 0:
        cmd = ["rg", "--line-number", "--with-filename", "--color=never"]
        if ignore_case:
            cmd.append("-i")
        if include:
            cmd += ["-g", include]
        cmd += [pattern, path]
    else:
        cmd = ["grep", "-rn", "--color=never", "--binary-files=without-match"]
        if ignore_case:
            cmd.append("-i")
        if include:
            cmd += ["--include", include]
        # skip common noise dirs
        cmd += ["--exclude-dir=.git", "--exclude-dir=node_modules", "--exclude-dir=__pycache__"]
        cmd += [pattern, path]

    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout.strip()
    if not output:
        return "No matches found."
    lines = output.splitlines()
    if len(lines) > 50:
        return "\n".join(lines[:50]) + f"\n... ({len(lines) - 50} more matches)"
    return output


def web_fetch(url: str, max_chars: int = 8000) -> str:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
        # strip HTML tags for readability
        text = re.sub(r"<style[^>]*>.*?</style>", "", raw, flags=re.DOTALL)
        text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.DOTALL)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        if len(text) > max_chars:
            text = text[:max_chars] + f"\n\n... (truncated, {len(text)} total chars)"
        return text
    except urllib.error.URLError as e:
        return f"Error fetching {url}: {e}"


def find_files(name: str, path: str = ".", file_type: str = None) -> str:
    path = os.path.expanduser(path)
    cmd = ["find", path, "-name", name]
    if file_type == "file":
        cmd += ["-type", "f"]
    elif file_type == "dir":
        cmd += ["-type", "d"]
    # skip node_modules, .git, __pycache__
    cmd += ["!", "-path", "*/node_modules/*", "!", "-path", "*/.git/*", "!", "-path", "*/__pycache__/*"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    output = result.stdout.strip()
    return output if output else "No files found."


def delete_file(path: str) -> str:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return f"Error: not found: {path}"
    if os.path.isdir(path):
        return f"Error: {path} is a directory — use bash with rm -rf for directories"
    os.remove(path)
    return f"Deleted: {path}"


def move_file(src: str, dst: str) -> str:
    import shutil
    src = os.path.expanduser(src)
    dst = os.path.expanduser(dst)
    if not os.path.exists(src):
        return f"Error: source not found: {src}"
    os.makedirs(os.path.dirname(dst) if os.path.dirname(dst) else ".", exist_ok=True)
    shutil.move(src, dst)
    return f"Moved: {src} → {dst}"


def make_dir(path: str) -> str:
    path = os.path.expanduser(path)
    os.makedirs(path, exist_ok=True)
    return f"Created: {path}"
