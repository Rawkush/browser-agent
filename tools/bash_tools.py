import json
import os
import subprocess


BLOCKED = ["rm -rf /", "rm -rf ~", ":(){ :|:& };:", "mkfs", "dd if=/dev/zero"]


def run_bash(command: str, cwd: str = None, timeout: int = 30) -> str:
    for blocked in BLOCKED:
        if blocked in command:
            return f"Error: blocked command: {command}"

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
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
    except Exception as e:
        return f"Error: {e}"


def stream_probe(provider: str = "ollama", max_chunks: int = 3, cwd: str = None) -> str:
    """
    Run the stream diagnostic script and return the real AIMessageChunk shapes.

    This closes the feedback loop: instead of guessing what chunk.content looks
    like for a given provider, the agent gets ground-truth runtime data so it
    can diagnose token-extraction bugs without human intervention.

    Returns the captured chunk shapes as pretty-printed JSON.
    """
    cmd = f"npx tsx packages/server/scripts/test-stream.ts {provider} {max_chunks}"
    run_result = run_bash(cmd, cwd=cwd, timeout=40)

    output_path = "debug/stream-test.json"
    if cwd:
        output_path = os.path.join(cwd, output_path)

    if os.path.exists(output_path):
        with open(output_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        shapes_summary = json.dumps(data.get("shapes", data), indent=2)
        return f"Probe stdout:\n{run_result}\n\nChunk shapes:\n{shapes_summary}"

    return f"Probe ran but output file not found.\nStdout:\n{run_result}"
