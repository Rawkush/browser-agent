"""
Project context auto-detection.

Scans the working directory for config files, build tools, and project
metadata. Returns a structured summary that's injected into the system
prompt so the agent starts every conversation knowing the stack.
"""

import json
import os
import subprocess

from browser_llm_agent.tools.registry import tool

# Config files that indicate the language/framework
_STACK_FILES = {
    "package.json":     "Node.js",
    "pyproject.toml":   "Python",
    "requirements.txt": "Python",
    "setup.py":         "Python",
    "Cargo.toml":       "Rust",
    "go.mod":           "Go",
    "pom.xml":          "Java (Maven)",
    "build.gradle":     "Java (Gradle)",
    "Gemfile":          "Ruby",
    "composer.json":    "PHP",
    "Package.swift":    "Swift",
    "mix.exs":          "Elixir",
}

# Files to read fully and append to the prompt
_INSTRUCTION_FILES = [
    "CLAUDE.md", "AGENTS.md", ".cursorrules",
    ".github/copilot-instructions.md",
]

# Files whose existence is reported but not read
_INFRA_FILES = [
    "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
    ".env", "Makefile", "Justfile",
]

# Test config files
_TEST_FILES = [
    "jest.config.js", "jest.config.ts", "jest.config.mjs",
    "vitest.config.ts", "vitest.config.js",
    "pytest.ini", "setup.cfg", "tox.ini",
    ".mocharc.yml", "karma.conf.js",
]


def _read_head(path: str, max_lines: int = 30) -> str:
    """Read first N lines of a file."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = []
            for i, line in enumerate(f):
                if i >= max_lines:
                    break
                lines.append(line)
        return "".join(lines)
    except Exception:
        return ""


def _read_full(path: str, max_chars: int = 5000) -> str:
    """Read a file fully, truncating at max_chars."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read(max_chars + 1)
        if len(content) > max_chars:
            content = content[:max_chars] + "\n... (truncated)"
        return content
    except Exception:
        return ""


def _git_info(cwd: str) -> str | None:
    """Get git branch and dirty file count."""
    if not os.path.isdir(os.path.join(cwd, ".git")):
        return None
    try:
        branch = subprocess.run(
            ["git", "branch", "--show-current"],
            capture_output=True, text=True, timeout=5, cwd=cwd,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True, text=True, timeout=5, cwd=cwd,
        ).stdout.strip()
        dirty = len(status.splitlines()) if status else 0
        info = f"branch: {branch}"
        if dirty:
            info += f", {dirty} uncommitted file(s)"
        return info
    except Exception:
        return None


def _extract_package_json(cwd: str) -> dict:
    """Extract useful info from package.json."""
    path = os.path.join(cwd, "package.json")
    if not os.path.exists(path):
        return {}
    try:
        with open(path) as f:
            pkg = json.load(f)
        info = {}
        if "name" in pkg:
            info["name"] = pkg["name"]
        if "scripts" in pkg:
            info["scripts"] = list(pkg["scripts"].keys())
        deps = list(pkg.get("dependencies", {}).keys())[:10]
        if deps:
            info["deps"] = deps
        return info
    except Exception:
        return {}


def _extract_pyproject(cwd: str) -> dict:
    """Extract useful info from pyproject.toml."""
    path = os.path.join(cwd, "pyproject.toml")
    if not os.path.exists(path):
        return {}
    try:
        content = _read_head(path, 50)
        info = {}
        for line in content.splitlines():
            if line.startswith("name"):
                info["name"] = line.split("=", 1)[1].strip().strip('"')
            if "dependencies" in line and "=" in line:
                info["has_deps"] = True
        return info
    except Exception:
        return {}


@tool("project_detect", "Auto-detect project type, stack, and config from the working directory.", {
    "type": "object",
    "properties": {
        "path": {"type": "string", "description": "Project root directory (default: .)"},
    },
    "required": [],
})
def project_detect(path: str = ".") -> str:
    path = os.path.expanduser(path)
    path = os.path.abspath(path)
    sections = []

    # 1. Detect stack from config files
    stacks = []
    for filename, lang in _STACK_FILES.items():
        if os.path.exists(os.path.join(path, filename)):
            stacks.append(f"{lang} ({filename})")
    if stacks:
        sections.append("Stack: " + ", ".join(stacks))
    else:
        sections.append("Stack: unknown (no recognized config files)")

    # 2. Package info
    pkg = _extract_package_json(path)
    if pkg:
        parts = []
        if "name" in pkg:
            parts.append(f"name: {pkg['name']}")
        if "scripts" in pkg:
            parts.append(f"scripts: {', '.join(pkg['scripts'][:8])}")
        if "deps" in pkg:
            parts.append(f"deps: {', '.join(pkg['deps'])}")
        sections.append("package.json: " + " | ".join(parts))

    pyproj = _extract_pyproject(path)
    if pyproj:
        parts = []
        if "name" in pyproj:
            parts.append(f"name: {pyproj['name']}")
        sections.append("pyproject.toml: " + " | ".join(parts))

    # 3. Git info
    git = _git_info(path)
    if git:
        sections.append(f"Git: {git}")

    # 4. Infrastructure files
    infra = [f for f in _INFRA_FILES if os.path.exists(os.path.join(path, f))]
    if infra:
        sections.append(f"Infra: {', '.join(infra)}")

    # 5. Test config
    tests = [f for f in _TEST_FILES if os.path.exists(os.path.join(path, f))]
    if tests:
        sections.append(f"Testing: {', '.join(tests)}")

    # 6. Instruction files (read fully)
    for filename in _INSTRUCTION_FILES:
        filepath = os.path.join(path, filename)
        if os.path.exists(filepath):
            content = _read_full(filepath)
            if content:
                sections.append(f"\n--- {filename} ---\n{content}\n--- end {filename} ---")

    return "\n".join(sections)
