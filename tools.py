from __future__ import annotations

import asyncio
import fnmatch
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from urllib.request import Request, urlopen
from urllib.error import URLError

from config import Settings
from storage import MemoryStore

log = logging.getLogger(__name__)

settings = Settings()
memory_store = MemoryStore()

# ---------------------------------------------------------------------------
# Anthropic tool definitions (JSON schema format)
# ---------------------------------------------------------------------------

VOICEBOX_HOST = "127.0.0.1"
VOICEBOX_PORT = 17493
VOICEBOX_PROFILE = "a8337146-d44f-40c8-99ae-682d1b52d151"
SPEAK_FAST_SCRIPT = Path(r"C:\Users\ldfla\.openclaw\workspace\sophia-cli\speak-fast.py")
PLAY_AUDIO_SCRIPT = Path(r"C:\Users\ldfla\.openclaw\workspace\sophia-cli\_play_audio.py")

TOOL_DEFINITIONS = [
    {
        "name": "read_file",
        "description": "Read the contents of a file. Returns the file contents with line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file (absolute or relative to workspace)",
                },
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Create or overwrite a file with the given content.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Path to the file (absolute or relative to workspace)",
                },
                "content": {
                    "type": "string",
                    "description": "The content to write to the file",
                },
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_directory",
        "description": "List files and directories in a path. Returns names with [DIR] or [FILE] prefix.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Directory path (absolute or relative to workspace). Defaults to workspace root.",
                    "default": ".",
                },
            },
            "required": [],
        },
    },
    {
        "name": "search_files",
        "description": "Search for files matching a glob pattern (e.g. '**/*.py', 'src/**/*.ts').",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Glob pattern to match files against",
                },
                "path": {
                    "type": "string",
                    "description": "Directory to search in (absolute or relative to workspace). Defaults to workspace root.",
                    "default": ".",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "grep_files",
        "description": "Search file contents for lines matching a regex pattern. Returns matching lines with file paths and line numbers.",
        "input_schema": {
            "type": "object",
            "properties": {
                "pattern": {
                    "type": "string",
                    "description": "Regex pattern to search for in file contents",
                },
                "path": {
                    "type": "string",
                    "description": "File or directory to search in (absolute or relative to workspace). Defaults to workspace root.",
                    "default": ".",
                },
                "glob": {
                    "type": "string",
                    "description": "Optional glob filter for file names (e.g. '*.py', '*.ts')",
                },
            },
            "required": ["pattern"],
        },
    },
    {
        "name": "save_memory",
        "description": (
            "Save a piece of information to persistent memory. Use this to remember "
            "user preferences, important facts, project context, or anything worth "
            "recalling in future conversations. Memory persists across sessions."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "A short descriptive key (e.g. 'user_name', 'current_project', 'preference_style')",
                },
                "value": {
                    "type": "string",
                    "description": "The information to remember",
                },
            },
            "required": ["key", "value"],
        },
    },
    {
        "name": "recall_memory",
        "description": (
            "Recall information from persistent memory. Call with no key to see all "
            "stored memories, or with a specific key to recall one item."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "Optional key to recall. Omit to see all memories.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "delete_memory",
        "description": "Delete a specific key from persistent memory.",
        "input_schema": {
            "type": "object",
            "properties": {
                "key": {
                    "type": "string",
                    "description": "The key to delete",
                },
            },
            "required": ["key"],
        },
    },
    {
        "name": "run_command",
        "description": (
            "Run a shell command and return its output (stdout + stderr). "
            "Commands run in the workspace directory by default. Use this to "
            "start servers, run scripts, install packages, run tests, git operations, etc. "
            "Commands have a timeout (default 30s, max 300s). For long-running processes "
            "like servers, use 'start' or background the command."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to execute",
                },
                "cwd": {
                    "type": "string",
                    "description": "Working directory (absolute or relative to workspace). Defaults to workspace root.",
                    "default": ".",
                },
                "timeout": {
                    "type": "integer",
                    "description": "Timeout in seconds (default 30, max 300)",
                    "default": 30,
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "speak",
        "description": (
            "Speak text aloud using Sophia's voice (TTS). Use this to read responses, "
            "greet the user, narrate results, or any time voice output is appropriate. "
            "Primary: Voicebox (custom cloned voice). Fallback: Edge-TTS (fast, generic)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "The text to speak aloud",
                },
                "fast": {
                    "type": "boolean",
                    "description": "If true, skip voicebox and use fast edge-tts directly. Useful for short/quick responses.",
                    "default": False,
                },
            },
            "required": ["text"],
        },
    },
]


# ---------------------------------------------------------------------------
# Path sandboxing
# ---------------------------------------------------------------------------

def _resolve_path(path_str: str) -> Path:
    """Resolve a path, ensuring it stays within the workspace."""
    workspace = Path(settings.workspace).resolve()
    p = Path(path_str)

    if p.is_absolute():
        resolved = p.resolve()
    else:
        resolved = (workspace / p).resolve()

    # Check that the resolved path is within the workspace
    try:
        resolved.relative_to(workspace)
    except ValueError:
        raise PermissionError(
            f"Access denied: {path_str} is outside workspace ({workspace})"
        )

    return resolved


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------

def _read_file(path: str) -> str:
    resolved = _resolve_path(path)
    if not resolved.is_file():
        return f"Error: {path} is not a file or does not exist"

    try:
        text = resolved.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        return f"Error reading file: {e}"

    lines = text.splitlines()
    # Add line numbers
    numbered = [f"{i + 1:4d}  {line}" for i, line in enumerate(lines)]

    # Truncate very large files
    if len(numbered) > 500:
        numbered = numbered[:500]
        numbered.append(f"\n... truncated ({len(lines)} total lines)")

    return "\n".join(numbered)


def _write_file(path: str, content: str) -> str:
    resolved = _resolve_path(path)

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
        return f"File written: {resolved} ({len(content)} bytes)"
    except Exception as e:
        return f"Error writing file: {e}"


def _list_directory(path: str = ".") -> str:
    resolved = _resolve_path(path)
    if not resolved.is_dir():
        return f"Error: {path} is not a directory or does not exist"

    try:
        entries = sorted(resolved.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
    except Exception as e:
        return f"Error listing directory: {e}"

    lines = []
    for entry in entries[:200]:  # Cap at 200 entries
        prefix = "[DIR] " if entry.is_dir() else "[FILE]"
        lines.append(f"{prefix} {entry.name}")

    if len(list(resolved.iterdir())) > 200:
        lines.append(f"\n... truncated (200 of {len(list(resolved.iterdir()))} entries shown)")

    return "\n".join(lines) if lines else "(empty directory)"


def _search_files(pattern: str, path: str = ".") -> str:
    resolved = _resolve_path(path)
    if not resolved.is_dir():
        return f"Error: {path} is not a directory"

    matches = []
    for match in resolved.glob(pattern):
        if match.is_file():
            try:
                rel = match.relative_to(Path(settings.workspace).resolve())
                matches.append(str(rel))
            except ValueError:
                matches.append(str(match))

        if len(matches) >= 100:
            break

    if not matches:
        return f"No files matching '{pattern}' found in {path}"

    result = "\n".join(matches)
    if len(matches) == 100:
        result += "\n... (results capped at 100)"
    return result


def _grep_files(pattern: str, path: str = ".", glob_filter: str | None = None) -> str:
    resolved = _resolve_path(path)

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return f"Invalid regex: {e}"

    results = []
    max_results = 50

    # Collect files to search
    if resolved.is_file():
        files = [resolved]
    else:
        files = []
        for root, _dirs, filenames in os.walk(resolved):
            # Skip hidden dirs and common non-text dirs
            root_path = Path(root)
            if any(part.startswith(".") or part in ("node_modules", "__pycache__", ".git")
                   for part in root_path.parts):
                continue
            for fname in filenames:
                if glob_filter and not fnmatch.fnmatch(fname, glob_filter):
                    continue
                files.append(root_path / fname)
            if len(files) > 5000:
                break

    workspace_root = Path(settings.workspace).resolve()

    for fpath in files:
        if len(results) >= max_results:
            break
        try:
            text = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        for i, line in enumerate(text.splitlines(), 1):
            if regex.search(line):
                try:
                    rel = fpath.relative_to(workspace_root)
                except ValueError:
                    rel = fpath
                results.append(f"{rel}:{i}  {line.rstrip()[:200]}")
                if len(results) >= max_results:
                    break

    if not results:
        return f"No matches for '{pattern}' in {path}"

    result = "\n".join(results)
    if len(results) == max_results:
        result += f"\n... (results capped at {max_results})"
    return result


# ---------------------------------------------------------------------------
# Shell command execution
# ---------------------------------------------------------------------------

def _run_command(command: str, cwd: str = ".", timeout: int = 30) -> str:
    """Run a shell command and return combined stdout + stderr."""
    import subprocess

    # Resolve cwd within workspace
    workspace = Path(settings.workspace).resolve()
    cwd_path = Path(cwd)
    if cwd_path.is_absolute():
        resolved_cwd = cwd_path.resolve()
    else:
        resolved_cwd = (workspace / cwd_path).resolve()

    # Validate cwd is within workspace
    try:
        resolved_cwd.relative_to(workspace)
    except ValueError:
        return f"Error: cwd '{cwd}' is outside workspace ({workspace})"

    if not resolved_cwd.is_dir():
        return f"Error: cwd '{cwd}' is not a directory"

    # Clamp timeout
    timeout = max(1, min(timeout, 300))

    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=str(resolved_cwd),
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        output_parts = []
        if result.stdout:
            output_parts.append(result.stdout)
        if result.stderr:
            output_parts.append(f"[stderr]\n{result.stderr}")

        output = "\n".join(output_parts) if output_parts else "(no output)"

        # Truncate large output
        if len(output) > 10000:
            output = output[:10000] + f"\n... truncated ({len(output)} total chars)"

        exit_info = f"[exit code: {result.returncode}]"
        return f"{output}\n{exit_info}"

    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
    except Exception as e:
        return f"Error running command: {e}"


# ---------------------------------------------------------------------------
# Voice (TTS)
# ---------------------------------------------------------------------------

def _voicebox_health() -> bool:
    """Check if voicebox server is running."""
    try:
        req = Request(f"http://{VOICEBOX_HOST}:{VOICEBOX_PORT}/health")
        urlopen(req, timeout=3)
        return True
    except Exception:
        return False


def _voicebox_generate(text: str) -> str | None:
    """Generate audio via voicebox. Returns path to temp WAV or None on failure."""
    try:
        body = json.dumps({
            "profile_id": VOICEBOX_PROFILE,
            "text": text,
            "language": "en",
            "model_size": "1.7B",
        }).encode()
        req = Request(
            f"http://{VOICEBOX_HOST}:{VOICEBOX_PORT}/generate",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        resp = urlopen(req, timeout=120)
        data = json.loads(resp.read())
        audio_id = data.get("id")
        if not audio_id:
            log.warning("Voicebox returned no id: %s", data)
            return None

        # Fetch the WAV
        tmp = os.path.join(tempfile.gettempdir(), f"sophia-{audio_id}.wav")
        audio_req = Request(f"http://{VOICEBOX_HOST}:{VOICEBOX_PORT}/audio/{audio_id}")
        audio_resp = urlopen(audio_req, timeout=30)
        with open(tmp, "wb") as f:
            f.write(audio_resp.read())
        return tmp
    except Exception as e:
        log.warning("Voicebox generate failed: %s", e)
        return None


def _play_audio(filepath: str) -> None:
    """Play a WAV file. Tries Python sounddevice script, then ffplay, then Windows start."""
    import subprocess
    # Primary: Python sounddevice via _play_audio.py
    if PLAY_AUDIO_SCRIPT.exists():
        try:
            r = subprocess.run(
                ["python", str(PLAY_AUDIO_SCRIPT), filepath],
                capture_output=True, timeout=30,
            )
            if r.returncode == 0:
                return
        except Exception:
            pass
    # Fallback 1: ffplay
    try:
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "quiet", filepath],
            capture_output=True, timeout=30,
        )
        return
    except Exception:
        pass
    # Fallback 2: Windows start
    try:
        subprocess.run(
            ["cmd", "/c", "start", "", filepath],
            capture_output=True, timeout=10, shell=True,
        )
    except Exception:
        pass


def _speak_fast(text: str) -> str:
    """Use edge-tts fallback (fast, generic voice)."""
    import subprocess
    if not SPEAK_FAST_SCRIPT.exists():
        return "Error: speak-fast.py not found"
    try:
        r = subprocess.run(
            ["python", str(SPEAK_FAST_SCRIPT), text],
            capture_output=True, timeout=15, text=True,
        )
        if r.returncode == 0:
            return "Spoke (edge-tts): " + text[:100]
        return f"Edge-TTS error: {r.stderr[:200]}"
    except subprocess.TimeoutExpired:
        return "Error: edge-tts timed out"
    except Exception as e:
        return f"Error: {e}"


def _speak(text: str, fast: bool = False) -> str:
    """Speak text aloud. Returns status message."""
    if fast:
        return _speak_fast(text)

    # Try voicebox first
    if _voicebox_health():
        wav_path = _voicebox_generate(text)
        if wav_path:
            _play_audio(wav_path)
            try:
                os.unlink(wav_path)
            except Exception:
                pass
            return "Spoke (Sophia voice): " + text[:100]

    # Fallback to edge-tts
    log.info("Voicebox unavailable, falling back to edge-tts")
    return _speak_fast(text)


# ---------------------------------------------------------------------------
# Tool dispatcher
# ---------------------------------------------------------------------------

def execute_tool(name: str, tool_input: dict) -> str:
    """Execute a tool by name and return the result as a string."""
    if name == "read_file":
        return _read_file(tool_input["path"])
    elif name == "write_file":
        return _write_file(tool_input["path"], tool_input["content"])
    elif name == "list_directory":
        return _list_directory(tool_input.get("path", "."))
    elif name == "search_files":
        return _search_files(tool_input["pattern"], tool_input.get("path", "."))
    elif name == "grep_files":
        return _grep_files(
            tool_input["pattern"],
            tool_input.get("path", "."),
            tool_input.get("glob"),
        )
    elif name == "run_command":
        return _run_command(
            tool_input["command"],
            tool_input.get("cwd", "."),
            tool_input.get("timeout", 30),
        )
    elif name == "save_memory":
        return memory_store.save(tool_input["key"], tool_input["value"])
    elif name == "recall_memory":
        return memory_store.recall(tool_input.get("key"))
    elif name == "delete_memory":
        return memory_store.delete(tool_input["key"])
    elif name == "speak":
        return _speak(tool_input["text"], tool_input.get("fast", False))
    else:
        return f"Unknown tool: {name}"
