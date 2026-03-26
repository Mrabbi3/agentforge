"""Tool registry and implementations for AgentForge.

Provides the tools an agent can use: bash execution, file I/O, and code search.
Tools are registered in a registry and dispatched by name.
"""

from __future__ import annotations

import asyncio
import os
from collections.abc import Callable, Coroutine
from typing import Any

from agentforge.core.schemas import ToolsConfig

# Type alias for async tool functions
ToolFunction = Callable[..., Coroutine[Any, Any, str]]


class ToolRegistry:
    """Manages available tools and dispatches tool calls."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolFunction] = {}
        self._definitions: list[dict[str, Any]] = []

    def register(
        self,
        name: str,
        func: ToolFunction,
        description: str,
        input_schema: dict[str, Any],
    ) -> None:
        """Register a tool with its function, description, and schema."""
        self._tools[name] = func
        self._definitions.append(
            {
                "name": name,
                "description": description,
                "input_schema": input_schema,
            }
        )

    def get_definitions(self) -> list[dict[str, Any]]:
        """Get tool definitions in Anthropic's tool format."""
        return self._definitions

    async def execute(
        self, name: str, inputs: dict[str, Any], timeout: int = 30
    ) -> str:
        """Execute a tool by name with given inputs.

        Args:
            name: Tool name.
            inputs: Tool input parameters.
            timeout: Max execution time in seconds.

        Returns:
            Tool output as a string.

        Raises:
            ValueError: If tool not found.
            TimeoutError: If execution exceeds timeout.
        """
        func = self._tools.get(name)
        if func is None:
            raise ValueError(f"Unknown tool: {name}. Available: {list(self._tools.keys())}")

        try:
            result = await asyncio.wait_for(func(**inputs), timeout=timeout)
            return result
        except TimeoutError:
            raise TimeoutError(f"Tool '{name}' timed out after {timeout}s")

    @classmethod
    def from_config(cls, config: ToolsConfig) -> ToolRegistry:
        """Create a registry with tools enabled in the config."""
        registry = cls()

        tool_map: dict[str, tuple[ToolFunction, str, dict[str, Any]]] = {
            "bash_execute": (
                bash_execute,
                "Execute a bash command and return stdout + stderr. Use for running tests, "
                "installing dependencies, checking file structure, and any shell operations.",
                {
                    "type": "object",
                    "properties": {
                        "command": {
                            "type": "string",
                            "description": "The bash command to execute",
                        }
                    },
                    "required": ["command"],
                },
            ),
            "file_read": (
                file_read,
                "Read the contents of a file. Returns the full file content as a string.",
                {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to read",
                        }
                    },
                    "required": ["path"],
                },
            ),
            "file_write": (
                file_write,
                "Write content to a file. Creates the file if it doesn't exist, "
                "overwrites if it does. Creates parent directories as needed.",
                {
                    "type": "object",
                    "properties": {
                        "path": {
                            "type": "string",
                            "description": "Path to the file to write",
                        },
                        "content": {
                            "type": "string",
                            "description": "Content to write to the file",
                        },
                    },
                    "required": ["path", "content"],
                },
            ),
            "file_search": (
                file_search,
                "Search for a pattern in files using grep. Returns matching lines "
                "with file paths and line numbers.",
                {
                    "type": "object",
                    "properties": {
                        "pattern": {
                            "type": "string",
                            "description": "Pattern to search for (regex supported)",
                        },
                        "path": {
                            "type": "string",
                            "description": "Directory or file to search in",
                            "default": ".",
                        },
                        "file_pattern": {
                            "type": "string",
                            "description": "Glob pattern for files to search (e.g., '*.py')",
                            "default": "",
                        },
                    },
                    "required": ["pattern"],
                },
            ),
        }

        for tool_name in config.enabled:
            if tool_name in tool_map:
                func, description, schema = tool_map[tool_name]
                registry.register(tool_name, func, description, schema)

        return registry


# ─── Tool Implementations ────────────────────────────────────────────────────


async def bash_execute(command: str) -> str:
    """Execute a bash command and return output."""
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=os.getcwd(),
        )
        stdout, stderr = await proc.communicate()

        output_parts = []
        if stdout:
            output_parts.append(stdout.decode("utf-8", errors="replace"))
        if stderr:
            output_parts.append(f"STDERR: {stderr.decode('utf-8', errors='replace')}")
        if proc.returncode != 0:
            output_parts.append(f"Exit code: {proc.returncode}")

        return "\n".join(output_parts) or "(no output)"

    except Exception as e:
        return f"Error executing command: {e}"


async def file_read(path: str) -> str:
    """Read a file and return its contents."""
    try:
        abs_path = os.path.abspath(path)
        with open(abs_path, encoding="utf-8", errors="replace") as f:
            content = f.read()

        if len(content) > 50_000:
            return (
                content[:50_000]
                + f"\n... (file truncated, {len(content)} total characters)"
            )
        return content

    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except Exception as e:
        return f"Error reading file: {e}"


async def file_write(path: str, content: str) -> str:
    """Write content to a file."""
    try:
        abs_path = os.path.abspath(path)
        os.makedirs(os.path.dirname(abs_path), exist_ok=True)

        with open(abs_path, "w", encoding="utf-8") as f:
            f.write(content)

        return f"Successfully wrote {len(content)} characters to {path}"

    except Exception as e:
        return f"Error writing file: {e}"


async def file_search(
    pattern: str, path: str = ".", file_pattern: str = ""
) -> str:
    """Search for a pattern in files using grep."""
    try:
        cmd_parts = ["grep", "-rn", "--color=never"]

        if file_pattern:
            cmd_parts.extend(["--include", file_pattern])

        cmd_parts.extend([pattern, path])

        proc = await asyncio.create_subprocess_exec(
            *cmd_parts,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        output = stdout.decode("utf-8", errors="replace")

        if not output.strip():
            return f"No matches found for pattern: {pattern}"

        # Limit output
        lines = output.strip().split("\n")
        if len(lines) > 50:
            return "\n".join(lines[:50]) + f"\n... ({len(lines)} total matches)"

        return output.strip()

    except Exception as e:
        return f"Error searching: {e}"
