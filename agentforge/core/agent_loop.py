"""Core agentic loop: Plan -> Act -> Observe -> Reflect."""

from __future__ import annotations

import asyncio
import json
import subprocess
import time
from typing import Any

from rich.console import Console

from agentforge.core.schemas import (
    AgentResult,
    AgentStatus,
    HarnessConfig,
    StepType,
    Task,
    ToolCall,
    Trajectory,
    TrajectoryStep,
)
from agentforge.memory.factory import MemoryFactory
from agentforge.tools.registry import ToolRegistry

console = Console()

DEFAULT_SYSTEM_PROMPT = """\
You are an expert coding agent. Your task is to fix bugs in code.

Follow this loop:
1. PLAN: Analyze the problem and form a plan.
2. ACT: Use tools to read files, search code, and make changes.
3. OBSERVE: Check the results of your actions.
4. REFLECT: Verify your fix is correct and complete.

Guidelines:
- Read the buggy code before making changes.
- Make minimal, targeted fixes.
- Run the test commands to verify your fix.
- If tests fail, analyze the output and iterate.
"""


async def run_agent_loop(
    task: Task,
    config: HarnessConfig,
    client: Any,
) -> AgentResult:
    """Run the full agent loop on a task and return the result."""
    tool_registry = ToolRegistry.from_config(config.tools)
    memory = MemoryFactory.create(config.memory)
    trajectory = Trajectory(task_id=task.id)

    for cmd in task.setup_commands:
        proc = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=60,
        )
        if proc.returncode != 0:
            console.print(f"[yellow]Setup warning:[/yellow] {proc.stderr[:200]}")

    task_prompt = (
        f"Task: {task.name}\n\n{task.description}\n\n"
        f"Test commands to verify your fix:\n"
    )
    for cmd in task.test_commands:
        task_prompt += f"  {cmd}\n"

    messages: list[dict[str, Any]] = [{"role": "user", "content": task_prompt}]
    tools = tool_registry.get_definitions()

    for step_num in range(1, config.agent.max_steps + 1):
        console.print(f"\n[bold blue]Step {step_num}/{config.agent.max_steps}[/bold blue]")

        if memory.should_compact(messages):
            console.print("[dim]Compacting memory...[/dim]")
            messages = memory.compact(messages, client=client, model=config.agent.model)

        try:
            response = client.messages.create(
                model=config.agent.model, max_tokens=4096,
                temperature=config.agent.temperature,
                system=DEFAULT_SYSTEM_PROMPT, tools=tools, messages=messages,
            )
        except Exception as exc:
            console.print(f"[red]API error:[/red] {exc}")
            trajectory.finalize()
            return AgentResult(
                task_id=task.id, status=AgentStatus.ERROR,
                trajectory=trajectory, output=str(exc),
            )

        step = TrajectoryStep(
            step_number=step_num, step_type=StepType.TOOL_USE,
            input_tokens=getattr(getattr(response, "usage", None), "input_tokens", 0),
            output_tokens=getattr(getattr(response, "usage", None), "output_tokens", 0),
        )

        assistant_content: list[dict[str, Any]] = []
        tool_use_blocks: list[dict[str, Any]] = []

        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
                step.reasoning = block.text
                console.print(f"[dim]{block.text[:200]}[/dim]")
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use", "id": block.id,
                    "name": block.name, "input": block.input,
                })
                tool_use_blocks.append({
                    "id": block.id, "name": block.name, "input": block.input,
                })

        messages.append({"role": "assistant", "content": assistant_content})

        stop_reason = getattr(response, "stop_reason", "end_turn")
        if stop_reason == "end_turn" and not tool_use_blocks:
            trajectory.add_step(step)
            break

        tool_results: list[dict[str, Any]] = []
        for tb in tool_use_blocks:
            console.print(f"  [green]Tool:[/green] {tb['name']}({json.dumps(tb['input'])[:80]})")
            t0 = time.time()
            result_str = await tool_registry.execute(
                tb["name"], tb["input"], timeout=float(config.tools.bash_timeout),
            )
            dur = (time.time() - t0) * 1000
            console.print(f"  [dim]-> {result_str[:120]}[/dim]")
            tc = ToolCall(
                tool_name=tb["name"], tool_input=tb["input"],
                tool_result=result_str, duration_ms=dur,
                success="Error" not in result_str,
                error=result_str if "Error" in result_str else None,
            )
            step.tool_calls.append(tc)
            tool_results.append({
                "type": "tool_result", "tool_use_id": tb["id"], "content": result_str,
            })

        messages.append({"role": "user", "content": tool_results})
        trajectory.add_step(step)

    trajectory.finalize()
    tests_passed = 0
    tests_total = len(task.test_commands)

    for cmd in task.test_commands:
        try:
            proc = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
            if proc.returncode == 0 and "ALL TESTS PASSED" in proc.stdout:
                tests_passed += 1
                console.print("[bold green]Tests passed[/bold green]")
            else:
                console.print(
                    f"[bold red]Tests failed[/bold red]: "
                    f"{proc.stdout[:200]}{proc.stderr[:200]}"
                )
        except Exception as exc:
            console.print(f"[red]Test error:[/red] {exc}")

    status = (
        AgentStatus.SUCCESS
        if tests_passed == tests_total and tests_total > 0
        else AgentStatus.FAILURE
    )
    return AgentResult(
        task_id=task.id, status=status, trajectory=trajectory,
        tests_passed=tests_passed, tests_total=tests_total,
    )


def run_agent(task: Task, config: HarnessConfig, client: Any) -> AgentResult:
    """Synchronous wrapper for the agent loop."""
    return asyncio.run(run_agent_loop(task, config, client))
