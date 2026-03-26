"""Tests for tool registry and implementations."""

import asyncio
import os
import tempfile

import pytest

from agentforge.core.schemas import ToolsConfig
from agentforge.tools.registry import (
    ToolRegistry,
    bash_execute,
    file_read,
    file_search,
    file_write,
)


@pytest.mark.asyncio
async def test_bash_execute_simple():
    """bash_execute should run commands and return output."""
    result = await bash_execute("echo 'hello world'")
    assert "hello world" in result


@pytest.mark.asyncio
async def test_bash_execute_error():
    """bash_execute should handle errors gracefully."""
    result = await bash_execute("ls /nonexistent_directory_12345")
    assert "No such file" in result or "Exit code" in result


@pytest.mark.asyncio
async def test_file_write_and_read():
    """file_write and file_read should work together."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.txt")

        write_result = await file_write(path, "hello from agentforge")
        assert "Successfully wrote" in write_result

        read_result = await file_read(path)
        assert read_result == "hello from agentforge"


@pytest.mark.asyncio
async def test_file_read_not_found():
    """file_read should return error for missing files."""
    result = await file_read("/nonexistent/file.txt")
    assert "Error" in result or "not found" in result


@pytest.mark.asyncio
async def test_file_search():
    """file_search should find patterns in files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "sample.py")
        await file_write(path, "def hello():\n    return 'world'\n")

        result = await file_search("hello", tmpdir, "*.py")
        assert "hello" in result


def test_registry_from_config():
    """ToolRegistry should create tools from config."""
    config = ToolsConfig(enabled=["bash_execute", "file_read"])
    registry = ToolRegistry.from_config(config)

    defs = registry.get_definitions()
    assert len(defs) == 2
    names = [d["name"] for d in defs]
    assert "bash_execute" in names
    assert "file_read" in names


@pytest.mark.asyncio
async def test_registry_execute():
    """ToolRegistry should dispatch tool calls correctly."""
    config = ToolsConfig(enabled=["bash_execute"])
    registry = ToolRegistry.from_config(config)

    result = await registry.execute("bash_execute", {"command": "echo test"})
    assert "test" in result


@pytest.mark.asyncio
async def test_registry_unknown_tool():
    """ToolRegistry should raise for unknown tools."""
    config = ToolsConfig(enabled=["bash_execute"])
    registry = ToolRegistry.from_config(config)

    with pytest.raises(ValueError, match="Unknown tool"):
        await registry.execute("nonexistent_tool", {})
