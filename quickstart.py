#!/usr/bin/env python3
"""AgentForge quickstart — local validation without an API key.

Runs 9 test groups to verify the installation is correct.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

# Colors for terminal output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BOLD = "\033[1m"
RESET = "\033[0m"

passed = 0
failed = 0


def check(name: str, ok: bool, detail: str = "") -> None:
    """Record and display a test result."""
    global passed, failed
    if ok:
        passed += 1
        print(f"  {GREEN}\u2713{RESET} {name}")
    else:
        failed += 1
        msg = f"  {RED}\u2717{RESET} {name}"
        if detail:
            msg += f" — {detail}"
        print(msg)


def group(title: str) -> None:
    """Print a test group header."""
    print(f"\n{BOLD}{YELLOW}[{title}]{RESET}")


# ── Group 1: Imports ───────────────────────────────────────────────

group("1. Package Imports")

try:
    import agentforge
    check("import agentforge", True)
except Exception as e:
    check("import agentforge", False, str(e))

try:
    from agentforge.core.schemas import (
        AgentConfig,
        AgentResult,
        AgentStatus,
        EvaluationConfig,
        HarnessConfig,
        JudgeConfig,
        MemoryConfig,
        MemoryStrategy,
        StepType,
        Task,
        TaskCategory,
        TaskDifficulty,
        ToolCall,
        Trajectory,
        TrajectoryStep,
    )
    check("import schemas", True)
except Exception as e:
    check("import schemas", False, str(e))

try:
    from agentforge.tools.registry import ToolRegistry
    check("import ToolRegistry", True)
except Exception as e:
    check("import ToolRegistry", False, str(e))

try:
    from agentforge.memory.factory import (
        BaseMemory,
        HybridMemory,
        MemoryFactory,
        RAGMemory,
        SlidingWindowMemory,
        SummarizationMemory,
    )
    check("import memory strategies", True)
except Exception as e:
    check("import memory strategies", False, str(e))

try:
    from agentforge.evaluation.metrics import MetricsCalculator
    check("import MetricsCalculator", True)
except Exception as e:
    check("import MetricsCalculator", False, str(e))

try:
    from agentforge.evaluation.judge import ModelJudge
    check("import ModelJudge", True)
except Exception as e:
    check("import ModelJudge", False, str(e))

try:
    from agentforge.multi_agent.orchestrator import (
        MultiAgentOrchestrator,
    )
    check("import MultiAgentOrchestrator", True)
except Exception as e:
    check("import MultiAgentOrchestrator", False, str(e))


# ── Group 2: Schema Creation ──────────────────────────────────────

group("2. Schema Creation")

try:
    t = Task(name="test_task", description="A test task")
    check("Task creation", len(t.id) == 8 and t.name == "test_task")
except Exception as e:
    check("Task creation", False, str(e))

try:
    cfg = HarnessConfig()
    check(
        "HarnessConfig defaults",
        cfg.agent.model == "claude-sonnet-4-20250514"
        and cfg.memory.strategy == MemoryStrategy.SUMMARIZATION,
    )
except Exception as e:
    check("HarnessConfig defaults", False, str(e))

try:
    traj = Trajectory(task_id="t1")
    step = TrajectoryStep(
        step_number=1,
        step_type=StepType.TOOL_USE,
        input_tokens=100,
        output_tokens=50,
    )
    traj.add_step(step)
    check(
        "Trajectory tracking",
        traj.total_steps == 1 and traj.total_input_tokens == 100,
    )
except Exception as e:
    check("Trajectory tracking", False, str(e))


# ── Group 3: Tool Execution ───────────────────────────────────────

group("3. Tool Execution")

try:
    from agentforge.tools.registry import bash_execute, file_read, file_write

    result = asyncio.run(bash_execute("echo agentforge_test"))
    check("bash_execute echo", "agentforge_test" in result)
except Exception as e:
    check("bash_execute echo", False, str(e))

try:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", delete=False
    ) as f:
        tmp = f.name
    asyncio.run(file_write(tmp, "hello agentforge"))
    content = asyncio.run(file_read(tmp))
    os.unlink(tmp)
    check("file_write + file_read", "hello agentforge" in content)
except Exception as e:
    check("file_write + file_read", False, str(e))

try:
    reg = ToolRegistry.from_config(
        __import__(
            "agentforge.core.schemas", fromlist=["ToolsConfig"]
        ).ToolsConfig()
    )
    defs = reg.get_definitions()
    names = {d["name"] for d in defs}
    check(
        "ToolRegistry from_config",
        names == {"bash_execute", "file_read", "file_write", "file_search"},
    )
except Exception as e:
    check("ToolRegistry from_config", False, str(e))


# ── Group 4: Memory Strategies ─────────────────────────────────────

group("4. Memory Strategies")

try:
    mc = MemoryConfig(strategy=MemoryStrategy.SLIDING_WINDOW)
    mem = MemoryFactory.create(mc)
    msgs = [{"role": "user", "content": f"msg {i}"} for i in range(20)]
    compacted = mem.compact(msgs)
    check("SlidingWindow compact", len(compacted) < len(msgs))
except Exception as e:
    check("SlidingWindow compact", False, str(e))

try:
    mc = MemoryConfig(strategy=MemoryStrategy.SUMMARIZATION)
    mem = MemoryFactory.create(mc)
    count = mem.count_tokens(
        [{"role": "user", "content": "hello world test"}]
    )
    check("token counting", count > 0)
except Exception as e:
    check("token counting", False, str(e))

try:
    mc = MemoryConfig(
        max_context_tokens=100,
        compact_threshold=0.5,
    )
    mem = MemoryFactory.create(mc)
    long_msgs = [
        {"role": "user", "content": "x" * 500} for _ in range(5)
    ]
    check("should_compact threshold", mem.should_compact(long_msgs))
except Exception as e:
    check("should_compact threshold", False, str(e))


# ── Group 5: Evaluation Metrics ────────────────────────────────────

group("5. Evaluation Metrics")

try:
    traj1 = Trajectory(task_id="t1")
    tc = ToolCall(
        tool_name="bash",
        tool_input={},
        tool_result="ok",
        success=True,
    )
    step1 = TrajectoryStep(
        step_number=1,
        step_type=StepType.TOOL_USE,
        input_tokens=500,
        output_tokens=200,
        tool_calls=[tc],
    )
    traj1.add_step(step1)
    traj1.finalize()
    r1 = AgentResult(
        task_id="t1",
        status=AgentStatus.SUCCESS,
        trajectory=traj1,
        tests_passed=1,
        tests_total=1,
    )
    calc = MetricsCalculator([r1])
    metrics = calc.compute(["pass_rate", "tool_efficiency", "avg_steps"])
    check(
        "metrics computation",
        len(metrics) == 3 and metrics[0].value == 100.0,
    )
except Exception as e:
    check("metrics computation", False, str(e))

try:
    calc_empty = MetricsCalculator([])
    m = calc_empty.pass_rate()
    check("empty results handling", m.value == 0.0)
except Exception as e:
    check("empty results handling", False, str(e))


# ── Group 6: Task YAML Loading ─────────────────────────────────────

group("6. Task YAML Loading")

try:
    import yaml

    tasks_dir = Path(__file__).parent / "tasks" / "coding"
    task_files = sorted(tasks_dir.glob("*.yaml"))
    check("task files found", len(task_files) == 6, f"found {len(task_files)}")
except Exception as e:
    check("task files found", False, str(e))

try:
    for tf in task_files:
        with open(tf) as f:
            data = yaml.safe_load(f)
        t = Task(**data)
        assert t.name
        assert t.test_commands
    check("all task YAMLs parse", True)
except Exception as e:
    check("all task YAMLs parse", False, str(e))


# ── Group 7: Config YAML Loading ───────────────────────────────────

group("7. Config YAML Loading")

try:
    configs_dir = Path(__file__).parent / "configs"
    config_files = sorted(configs_dir.glob("*.yaml"))
    check(
        "config files found",
        len(config_files) == 2,
        f"found {len(config_files)}",
    )
except Exception as e:
    check("config files found", False, str(e))

try:
    for cf in config_files:
        with open(cf) as f:
            data = yaml.safe_load(f)
        cfg = HarnessConfig(**data)
        assert cfg.agent.model
    check("all config YAMLs parse", True)
except Exception as e:
    check("all config YAMLs parse", False, str(e))


# ── Group 8: CLI Version ──────────────────────────────────────────

group("8. CLI Version Check")

try:
    from agentforge import __version__

    check("version string", __version__ == "0.1.0")
except Exception as e:
    check("version string", False, str(e))

try:
    from click.testing import CliRunner
    from agentforge.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    check(
        "CLI --version",
        result.exit_code == 0 and "0.1.0" in result.output,
        result.output.strip(),
    )
except Exception as e:
    check("CLI --version", False, str(e))


# ── Group 9: API Key Detection ─────────────────────────────────────

group("9. API Key Detection")

try:
    has_key = bool(os.environ.get("ANTHROPIC_API_KEY"))
    if has_key:
        check("ANTHROPIC_API_KEY set", True)
    else:
        print(
            f"  {YELLOW}\u26a0{RESET} ANTHROPIC_API_KEY not set "
            f"(needed for agent runs, not for tests)"
        )
        passed += 1  # count as pass — it's expected
except Exception as e:
    check("API key check", False, str(e))


# ── Summary ────────────────────────────────────────────────────────

total = passed + failed
print(f"\n{BOLD}{'=' * 50}{RESET}")
print(
    f"{BOLD}Results: {GREEN}{passed} passed{RESET}, "
    f"{RED if failed else BOLD}{failed} failed{RESET} "
    f"/ {total} total"
)
print(f"{BOLD}{'=' * 50}{RESET}")

if failed > 0:
    sys.exit(1)
