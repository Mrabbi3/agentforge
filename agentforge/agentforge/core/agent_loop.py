"""Core agentic loop — the heart of AgentForge.

Implements the Plan → Act → Observe → Reflect cycle with configurable
memory strategies, tool routing, and trajectory logging.
"""

from __future__ import annotations

import time
from typing import Any

import anthropic
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

# Default system prompt for the coding agent
DEFAULT_SYSTEM_PROMPT = """You are an expert software engineer solving a coding task.

You have access to tools for executing bash commands, reading files, writing files,
and searching through code. Use them iteratively to understand the problem, explore
the codebase, implement a fix, and verify it works.

Guidelines:
- Start by understanding the problem description carefully.
- Explore the relevant code to understand the existing implementation.
- Form a plan before making changes.
- Make minimal, targeted changes to fix the issue.
- Run tests to verify your fix works.
- If tests fail, analyze the output and iterate.
- Stop when all relevant tests pass.

Think step-by-step and explain your reasoning."""


class AgentLoop:
    """Runs the core agent loop with configurable memory, tools, and logging."""

    def __init__(self, config: HarnessConfig) -> None:
        self.config = config
        self.client = anthropic.Anthropic()
        self.memory = MemoryFactory.create(config.memory)
        self.tools = ToolRegistry.from_config(config.tools)

    async def run(self, task: Task) -> AgentResult:
        """Execute the agent loop on a single task.

        Args:
            task: The task definition to solve.

        Returns:
            AgentResult with status, trajectory, and output.
        """
        trajectory = Trajectory(
            task_id=task.id,
            config_name=self.config.name,
        )

        system_prompt = self.config.agent.system_prompt or DEFAULT_SYSTEM_PROMPT

        # Initialize message history with the task
        messages: list[dict[str, Any]] = [
            {"role": "user", "content": task.description},
        ]

        console.print(f"\n[bold blue]━━━ AgentForge: Running task '{task.name}' ━━━[/bold blue]")
        console.print(f"  Model: {self.config.agent.model}")
        console.print(f"  Memory: {self.config.memory.strategy.value}")
        console.print(f"  Max steps: {self.config.agent.max_steps}\n")

        final_output: str | None = None

        for step_num in range(self.config.agent.max_steps):
            step_start = time.time()

            console.print(f"[dim]Step {step_num + 1}/{self.config.agent.max_steps}[/dim]")

            try:
                # ── LLM Call ──────────────────────────────────────────
                response = self.client.messages.create(
                    model=self.config.agent.model,
                    max_tokens=4096,
                    temperature=self.config.agent.temperature,
                    system=system_prompt,
                    messages=messages,
                    tools=self.tools.get_definitions(),
                )

                step_duration = (time.time() - step_start) * 1000

                # Log the LLM call
                step = TrajectoryStep(
                    step_number=step_num,
                    step_type=StepType.LLM_CALL,
                    input_tokens=response.usage.input_tokens,
                    output_tokens=response.usage.output_tokens,
                    model=self.config.agent.model,
                    duration_ms=step_duration,
                )

                # Extract any text reasoning from the response
                text_blocks = [
                    block.text for block in response.content if block.type == "text"
                ]
                if text_blocks:
                    step.reasoning = "\n".join(text_blocks)
                    console.print(f"  [green]Reasoning:[/green] {text_blocks[0][:120]}...")

                # ── Check for end turn (no more tool use) ─────────────
                if response.stop_reason == "end_turn":
                    trajectory.add_step(step)
                    final_output = "\n".join(text_blocks) if text_blocks else None
                    console.print("  [bold green]Agent finished (end_turn)[/bold green]")
                    break

                # ── Execute tool calls ────────────────────────────────
                tool_use_blocks = [
                    block for block in response.content if block.type == "tool_use"
                ]

                tool_results_for_message: list[dict[str, Any]] = []

                for tool_block in tool_use_blocks:
                    tool_start = time.time()
                    tool_name = tool_block.name
                    tool_input = tool_block.input

                    tool_display = _truncate(str(tool_input), 80)
                    console.print(f"  [yellow]Tool:[/yellow] {tool_name}({tool_display})")

                    try:
                        result = await self.tools.execute(
                            tool_name,
                            tool_input,
                            timeout=self.config.tools.bash_timeout,
                        )
                        success = True
                        error = None
                        # Truncate long outputs
                        if len(result) > self.config.tools.max_output_chars:
                            result = (
                                result[: self.config.tools.max_output_chars]
                                + f"\n... (truncated, {len(result)} total chars)"
                            )
                        console.print(f"  [dim]Result:[/dim] {_truncate(result, 100)}")
                    except Exception as e:
                        result = f"Error: {e}"
                        success = False
                        error = str(e)
                        console.print(f"  [red]Error:[/red] {error}")

                    tool_call = ToolCall(
                        tool_name=tool_name,
                        tool_input=tool_input,
                        tool_result=result,
                        duration_ms=(time.time() - tool_start) * 1000,
                        success=success,
                        error=error,
                    )
                    step.tool_calls.append(tool_call)

                    tool_results_for_message.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_block.id,
                            "content": result,
                        }
                    )

                trajectory.add_step(step)

                # ── Update messages ───────────────────────────────────
                # Append assistant response
                messages.append({"role": "assistant", "content": response.content})
                # Append tool results
                messages.append({"role": "user", "content": tool_results_for_message})

                # ── Memory management ─────────────────────────────────
                if self.memory.should_compact(messages, self.config.memory.max_context_tokens):
                    console.print("  [magenta]Compacting memory...[/magenta]")
                    compact_start = time.time()
                    messages = await self.memory.compact(messages, self.client)
                    compact_step = TrajectoryStep(
                        step_number=step_num,
                        step_type=StepType.MEMORY_COMPACT,
                        duration_ms=(time.time() - compact_start) * 1000,
                    )
                    trajectory.add_step(compact_step)

            except anthropic.APIError as e:
                error_step = TrajectoryStep(
                    step_number=step_num,
                    step_type=StepType.ERROR,
                    duration_ms=(time.time() - step_start) * 1000,
                )
                trajectory.add_step(error_step)
                trajectory.finalize()

                return AgentResult(
                    task_id=task.id,
                    status=AgentStatus.ERROR,
                    trajectory=trajectory,
                    error_message=str(e),
                )

        else:
            # Loop completed without break — max steps exceeded
            trajectory.finalize()
            console.print("[bold red]Max steps exceeded[/bold red]")
            return AgentResult(
                task_id=task.id,
                status=AgentStatus.MAX_STEPS_EXCEEDED,
                trajectory=trajectory,
                output=final_output,
            )

        trajectory.finalize()

        # Run tests if defined
        tests_passed = 0
        tests_total = 0
        if task.test_commands:
            tests_total = len(task.test_commands)
            for test_cmd in task.test_commands:
                try:
                    result = await self.tools.execute(
                        "bash_execute", {"command": test_cmd}, timeout=60
                    )
                    if "FAILED" not in result.upper() and "ERROR" not in result.upper():
                        tests_passed += 1
                except Exception:
                    pass

        if tests_passed == tests_total and tests_total > 0:
            status = AgentStatus.SUCCESS
        else:
            status = AgentStatus.FAILURE
        if tests_total == 0:
            status = AgentStatus.SUCCESS  # No tests = assume success if agent finished

        console.print(
            f"\n[bold]Result:[/bold] {status.value} "
            f"({tests_passed}/{tests_total} tests passed, "
            f"{trajectory.total_steps} steps, "
            f"{trajectory.total_tool_calls} tool calls, "
            f"{trajectory.total_input_tokens + trajectory.total_output_tokens} tokens)\n"
        )

        return AgentResult(
            task_id=task.id,
            status=status,
            trajectory=trajectory,
            output=final_output,
            tests_passed=tests_passed,
            tests_total=tests_total,
        )


def _truncate(text: str, max_len: int) -> str:
    """Truncate text for display."""
    text = text.replace("\n", " ")
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."
