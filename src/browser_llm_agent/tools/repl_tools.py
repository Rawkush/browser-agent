"""
Code execution / REPL tools.

Distinct from bash because they provide cleaner error output
(Python tracebacks, Node stack traces) and don't require shell quoting.
"""

import os
import subprocess
import tempfile

from browser_llm_agent.tools.registry import tool


def _run_code(runtime: str, code: str, timeout: int = 60) -> str:
    """Execute code via a runtime (python3, node) and return output."""
    # For multi-line code or code with quotes, use a temp file
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py" if runtime == "python3" else ".js",
            delete=False, encoding="utf-8",
        ) as f:
            f.write(code)
            tmp_path = f.name

        result = subprocess.run(
            [runtime, tmp_path],
            capture_output=True, text=True, timeout=timeout,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        return output.strip() or "(no output)"
    except FileNotFoundError:
        return f"Error: {runtime} not found. Is it installed?"
    except subprocess.TimeoutExpired:
        return f"Error: execution timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


@tool("python_exec", "Execute Python code and return output. Cleaner than bash for Python snippets.", {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "Python code to execute"},
        "timeout": {"type": "integer", "description": "Timeout in seconds (default: 60)"},
    },
    "required": ["code"],
})
def python_exec(code: str, timeout: int = 60) -> str:
    return _run_code("python3", code, min(max(timeout, 1), 300))


@tool("node_exec", "Execute JavaScript/Node.js code and return output.", {
    "type": "object",
    "properties": {
        "code": {"type": "string", "description": "JavaScript code to execute"},
        "timeout": {"type": "integer", "description": "Timeout in seconds (default: 60)"},
    },
    "required": ["code"],
})
def node_exec(code: str, timeout: int = 60) -> str:
    return _run_code("node", code, min(max(timeout, 1), 300))
