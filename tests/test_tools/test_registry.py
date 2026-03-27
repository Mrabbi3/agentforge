"""Tests for agentforge.tools.registry — 8 tests."""

import asyncio

import pytest

from agentforge.core.schemas import ToolsConfig
from agentforge.tools.registry import (
    ToolRegistry,
    bash_execute,
    file_read,
    file_search,
    file_write,
)


@pytest.fixture
def registry():
    """Create a registry from default config."""
    config = ToolsConfig()
    return ToolRegistry.from_config(config)


def test_registry_from_config(registry):
    """Registry creates tools from config."""
    defs = registry.get_definitions()
    names = {d["name"] for d in defs}
    assert names == {"bash_execute", "file_read", "file_write", "file_search"}


def test_registry_get_definitions_schema(registry):
    """Tool definitions include required schema fields."""
    for defn in registry.get_definitions():
        assert "name" in defn
        assert "description" in defn
        assert "input_schema" in defn
        assert defn["input_schema"]["type"] == "object"


def test_registry_execute_unknown():
    """Executing an unknown tool returns error string."""
    reg = ToolRegistry()
    result = asyncio.run(reg.execute("nonexistent", {}))
    assert "Error" in result


def test_bash_execute_echo():
    """bash_execute runs a simple echo."""
    result = asyncio.run(bash_execute("echo hello"))
    assert "hello" in result


def test_file_write_and_read(tmp_path):
    """file_write creates a file that file_read can read."""
    path = str(tmp_path / "test.txt")
    asyncio.run(file_write(path, "test content"))
    result = asyncio.run(file_read(path))
    assert "test content" in result


def test_file_read_missing():
    """file_read returns error for nonexistent file."""
    result = asyncio.run(file_read("/tmp/nonexistent_agentforge_test.txt"))
    assert "Error" in result


def test_file_search(tmp_path):
    """file_search finds a pattern in files."""
    f = tmp_path / "search_test.py"
    f.write_text("def hello_world():\n    pass\n")
    result = asyncio.run(file_search("hello_world", str(tmp_path)))
    assert "hello_world" in result


def test_registry_execute_with_timeout():
    """Tool execution respects timeout."""
    reg = ToolRegistry()

    async def slow_tool(**kwargs):
        await asyncio.sleep(10)
        return "done"

    reg.register("slow", "A slow tool", {}, slow_tool)
    result = asyncio.run(reg.execute("slow", {}, timeout=0.1))
    assert "timed out" in result
