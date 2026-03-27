"""Tool registry and built-in tool implementations."""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Callable, Coroutine
from pathlib import Path
from typing import Any

from agentforge.core.schemas import ToolsConfig

MAX_OUTPUT_LENGTH = 10_000


class ToolRegistry:
    """Registry for agent tools with definitions and execution."""

    def __init__(self) -> None:
        self._tools: dict[str, dict[str, Any]] = {}
        self._handlers: dict[str, Callable[..., Coroutine[Any, Any, str]]] = {}

    def register(
        self, name: str, description: str,
        parameters: dict[str, Any],
        handler: Callable[..., Coroutine[Any, Any, str]],
    ) -> None:
        """Register a tool with its schema and handler."""
        self._tools[name] = {
            "name": name, "description": description,
            "input_schema": {
                "type": "object", "properties": parameters,
                "required": list(parameters.keys()),
            },
        }
        self._handlers[name] = handler

    def get_definitions(self) -> list[dict[str, Any]]:
        """Return tool definitions for the Anthropic API."""
        return list(self._tools.values())

    async def execute(
        self, name: str, tool_input: dict[str, Any], timeout: float = 30.0,
    ) -> str:
        """Execute a tool by name with a timeout."""
        handler = self._handlers.get(name)
        if handler is None:
            return f"Error: unknown tool '{name}'"
        try:
            result = await asyncio.wait_for(handler(**tool_input), timeout=timeout)
            if len(result) > MAX_OUTPUT_LENGTH:
                return result[:MAX_OUTPUT_LENGTH] + "\n... (output truncated)"
            return result
        except (TimeoutError, asyncio.TimeoutError):
            return f"Error: tool '{name}' timed out after {timeout}s"
        except Exception as exc:
            return f"Error executing '{name}': {exc}"

    @classmethod
    def from_config(cls, config: ToolsConfig) -> ToolRegistry:
        """Create a registry from a ToolsConfig."""
        registry = cls()
        tool_map: dict[str, tuple[str, dict[str, Any], Any]] = {
            "bash_execute": (
                "Execute a bash command and return output.",
                {"command": {"type": "string", "description": "Bash command"}},
                bash_execute,
            ),
            "file_read": (
                "Read a file and return its contents.",
                {"path": {"type": "string", "description": "File path"}},
                file_read,
            ),
            "file_write": (
                "Write content to a file.",
                {"path": {"type": "string", "description": "File path"},
                 "content": {"type": "string", "description": "File content"}},
                file_write,
            ),
            "file_search": (
                "Search for a pattern in files using grep.",
                {"pattern": {"type": "string", "description": "Search pattern"},
                 "path": {"type": "string", "description": "Directory to search"}},
                file_search,
            ),
        }
        for name in config.enabled:
            if name in tool_map:
                desc, params, handler = tool_map[name]
                registry.register(name, desc, params, handler)
        return registry


async def bash_execute(command: str) -> str:
    """Execute a bash command and return stdout + stderr."""
    try:
        proc = await asyncio.create_subprocess_shell(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        output = stdout.decode(errors="replace")
        if stderr:
            output += "\nSTDERR:\n" + stderr.decode(errors="replace")
        if proc.returncode != 0:
            output += f"\n(exit code: {proc.returncode})"
        return output.strip() or "(no output)"
    except Exception as exc:
        return f"Error: {exc}"


async def file_read(path: str) -> str:
    """Read a file and return its contents."""
    try:
        content = Path(path).read_text(encoding="utf-8")
        return content or "(empty file)"
    except Exception as exc:
        return f"Error reading '{path}': {exc}"


async def file_write(path: str, content: str) -> str:
    """Write content to a file, creating parent directories."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"Successfully wrote {len(content)} chars to {path}"
    except Exception as exc:
        return f"Error writing '{path}': {exc}"


async def file_search(pattern: str, path: str = ".") -> str:
    """Search for a pattern in files using grep."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "grep", "-rn", "--include=*.py", pattern, path,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        output = stdout.decode(errors="replace").strip()
        return output or "(no matches found)"
    except Exception as exc:
        return f"Error searching: {exc}"
