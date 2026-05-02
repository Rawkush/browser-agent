import io
import subprocess
import threading
import time
import uuid

from browser_llm_agent.tools.registry import tool

BLOCKED = [
    "rm -rf /", "rm -rf ~", ":(){ :|:& };:", "mkfs", "dd if=/dev/zero",
    "sudo rm", "> /dev/sda", "shutdown", "reboot", "format C:",
    "curl | sh", "curl | bash", "wget | sh", "wget | bash",
]

# Also block force-push to main/master
_FORCE_PUSH_BLOCKED = ["git push --force", "git push -f"]
_PROTECTED_BRANCHES = ["main", "master"]

MAX_OUTPUT = 10_000  # chars before truncation


def _check_blocked(command: str) -> str | None:
    """Return error string if command is blocked, None otherwise."""
    for blocked in BLOCKED:
        if blocked in command:
            return f"Error: blocked command: {command}"
    # Block force-push to protected branches
    for fp in _FORCE_PUSH_BLOCKED:
        if fp in command:
            for branch in _PROTECTED_BRANCHES:
                if branch in command:
                    return f"Error: force push to {branch} is blocked"
    return None


def _truncate_output(output: str) -> str:
    """Smart truncation: first 3K + last 3K with summary in middle."""
    if len(output) <= MAX_OUTPUT:
        return output
    head = output[:3000]
    tail = output[-3000:]
    omitted = len(output) - 6000
    return f"{head}\n\n... ({omitted} chars omitted) ...\n\n{tail}"


# ── Background process tracking ─────────────────────────────────────────────

_BACKGROUND: dict[str, dict] = {}
# Each entry: {"proc": Popen, "stdout": StringIO, "stderr": StringIO,
#              "start": float, "done": bool, "returncode": int|None}


def _drain_pipe(pipe, buf: io.StringIO):
    """Reader thread: drain a pipe into a StringIO buffer."""
    try:
        for line in iter(pipe.readline, ""):
            buf.write(line)
    except (ValueError, OSError):
        pass
    finally:
        try:
            pipe.close()
        except Exception:
            pass


# ── Main bash tool ───────────────────────────────────────────────────────────

@tool("bash", "Run a shell command and return stdout + stderr. Use timeout for long commands, run_in_background for async.", {
    "type": "object",
    "properties": {
        "command": {"type": "string", "description": "Shell command to run"},
        "cwd": {"type": "string", "description": "Working directory (optional)"},
        "timeout": {"type": "integer", "description": "Timeout in seconds (default 120, max 600)"},
        "run_in_background": {"type": "boolean", "description": "If true, run async and return a process ID"},
    },
    "required": ["command"],
})
def run_bash(command: str, cwd: str = None, timeout: int = 120, run_in_background: bool = False) -> str:
    blocked = _check_blocked(command)
    if blocked:
        return blocked

    timeout = min(max(timeout, 1), 600)

    if run_in_background:
        return _run_background(command, cwd)

    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd,
        )
        output = ""
        if result.stdout:
            output += result.stdout
        if result.stderr:
            output += f"\nSTDERR:\n{result.stderr}"
        output = output.strip() or "(no output)"
        return _truncate_output(output)
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s. Use run_in_background=true for long-running commands."
    except Exception as e:
        return f"Error: {e}"


def _run_background(command: str, cwd: str = None) -> str:
    """Launch a background process and return its ID."""
    pid = str(uuid.uuid4())[:8]
    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()

    proc = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        cwd=cwd,
    )

    _BACKGROUND[pid] = {
        "proc": proc,
        "stdout": stdout_buf,
        "stderr": stderr_buf,
        "start": time.time(),
        "command": command[:100],
    }

    # Drain stdout/stderr in background threads
    threading.Thread(target=_drain_pipe, args=(proc.stdout, stdout_buf), daemon=True).start()
    threading.Thread(target=_drain_pipe, args=(proc.stderr, stderr_buf), daemon=True).start()

    return f"Started background process [{pid}]: {command[:80]}\nUse bash_status to check output, bash_kill to stop."


# ── Background process management tools ──────────────────────────────────────

@tool("bash_status", "Check output and status of a background process.", {
    "type": "object",
    "properties": {
        "process_id": {"type": "string", "description": "Process ID from bash run_in_background"},
    },
    "required": ["process_id"],
})
def bash_status(process_id: str) -> str:
    entry = _BACKGROUND.get(process_id)
    if not entry:
        # List available processes if ID not found
        if _BACKGROUND:
            ids = ", ".join(_BACKGROUND.keys())
            return f"Error: process '{process_id}' not found. Active: {ids}"
        return f"Error: process '{process_id}' not found. No background processes running."

    proc = entry["proc"]
    rc = proc.poll()
    elapsed = time.time() - entry["start"]
    status = f"running ({elapsed:.0f}s)" if rc is None else f"exited (code {rc}, {elapsed:.0f}s)"

    stdout = entry["stdout"].getvalue()
    stderr = entry["stderr"].getvalue()

    output = f"[{process_id}] {entry['command']} — {status}\n"
    if stdout:
        output += f"\nSTDOUT:\n{stdout}"
    if stderr:
        output += f"\nSTDERR:\n{stderr}"

    return _truncate_output(output.strip())


@tool("bash_kill", "Kill a background process.", {
    "type": "object",
    "properties": {
        "process_id": {"type": "string", "description": "Process ID to kill"},
    },
    "required": ["process_id"],
})
def bash_kill(process_id: str) -> str:
    entry = _BACKGROUND.get(process_id)
    if not entry:
        return f"Error: process '{process_id}' not found"

    proc = entry["proc"]
    if proc.poll() is not None:
        return f"Process [{process_id}] already exited (code {proc.returncode})"

    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()

    return f"Killed process [{process_id}]"
