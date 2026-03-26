"""AgentForge CLI — run tasks, benchmarks, and comparisons from the command line."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.table import Table

from agentforge.core.agent_loop import AgentLoop
from agentforge.core.schemas import (
    EvaluationResult,
    HarnessConfig,
    Task,
)
from agentforge.evaluation.metrics import MetricsCalculator

console = Console()


def _load_config(config_path: str) -> HarnessConfig:
    """Load a HarnessConfig from a YAML file."""
    with open(config_path) as f:
        data = yaml.safe_load(f)
    return HarnessConfig(**data)


def _load_task(task_path: str) -> Task:
    """Load a Task from a YAML file."""
    with open(task_path) as f:
        data = yaml.safe_load(f)
    return Task(**data)


def _load_tasks_from_dir(task_dir: str) -> list[Task]:
    """Load all tasks from a directory."""
    tasks = []
    task_path = Path(task_dir)
    for f in sorted(task_path.rglob("*.yaml")):
        tasks.append(_load_task(str(f)))
    return tasks


@click.group()
@click.version_option(version="0.1.0")
def main() -> None:
    """AgentForge — Build, benchmark, and compare AI agent architectures."""
    pass


@main.command()
@click.option("--task", required=True, help="Path to task YAML file")
@click.option("--config", default=None, help="Path to harness config YAML")
@click.option("--output", default=None, help="Path to save result JSON")
def run(task: str, config: str | None, output: str | None) -> None:
    """Run the agent on a single task."""
    harness_config = _load_config(config) if config else HarnessConfig()
    task_def = _load_task(task)

    agent = AgentLoop(harness_config)
    result = asyncio.run(agent.run(task_def))

    # Display result
    _display_result(result)

    if output:
        with open(output, "w") as f:
            json.dump(result.model_dump(), f, indent=2, default=str)
        console.print(f"\n[dim]Result saved to {output}[/dim]")


@main.command()
@click.option("--config", required=True, help="Path to harness config YAML")
@click.option("--tasks-dir", default="tasks", help="Directory containing task YAML files")
@click.option("--output", default="results", help="Directory to save results")
def benchmark(config: str, tasks_dir: str, output: str) -> None:
    """Run the full benchmark suite."""
    harness_config = _load_config(config)
    tasks = _load_tasks_from_dir(tasks_dir)

    if not tasks:
        console.print("[red]No tasks found![/red]")
        sys.exit(1)

    console.print(f"\n[bold]Running benchmark: {len(tasks)} tasks[/bold]\n")

    agent = AgentLoop(harness_config)
    results = []

    for i, task_def in enumerate(tasks, 1):
        console.print(f"[bold]━━━ Task {i}/{len(tasks)}: {task_def.name} ━━━[/bold]")
        result = asyncio.run(agent.run(task_def))
        results.append(result)

    # Compute metrics
    calculator = MetricsCalculator(harness_config.evaluation)
    metrics = calculator.compute_all(results)

    eval_result = EvaluationResult(
        config_name=harness_config.name,
        total_tasks=len(tasks),
        results=results,
        metrics=metrics,
    )

    # Display summary
    _display_benchmark_summary(eval_result)

    # Save results
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    result_file = output_path / f"{harness_config.name}_results.json"
    with open(result_file, "w") as f:
        json.dump(eval_result.model_dump(), f, indent=2, default=str)

    console.print(f"\n[dim]Results saved to {result_file}[/dim]")


@main.command()
@click.option(
    "--configs", required=True, multiple=True,
    help="Paths to config YAML files to compare",
)
@click.option("--tasks-dir", default="tasks", help="Directory containing task YAML files")
def compare(configs: tuple[str, ...], tasks_dir: str) -> None:
    """Compare multiple harness configurations on the same task suite."""
    tasks = _load_tasks_from_dir(tasks_dir)

    if not tasks:
        console.print("[red]No tasks found![/red]")
        sys.exit(1)

    all_eval_results: list[EvaluationResult] = []

    for config_path in configs:
        harness_config = _load_config(config_path)
        console.print(f"\n[bold blue]━━━ Config: {harness_config.name} ━━━[/bold blue]")

        agent = AgentLoop(harness_config)
        results = []

        for task_def in tasks:
            result = asyncio.run(agent.run(task_def))
            results.append(result)

        calculator = MetricsCalculator(harness_config.evaluation)
        metrics = calculator.compute_all(results)

        eval_result = EvaluationResult(
            config_name=harness_config.name,
            total_tasks=len(tasks),
            results=results,
            metrics=metrics,
        )
        all_eval_results.append(eval_result)

    # Display comparison table
    _display_comparison(all_eval_results)


def _display_result(result: object) -> None:
    """Display a single agent result."""
    from agentforge.core.schemas import AgentResult

    if not isinstance(result, AgentResult):
        return

    table = Table(title="Agent Result")
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Status", result.status.value)
    table.add_row("Tests", f"{result.tests_passed}/{result.tests_total}")
    table.add_row("Steps", str(result.trajectory.total_steps))
    table.add_row("Tool Calls", str(result.trajectory.total_tool_calls))
    table.add_row("Input Tokens", f"{result.trajectory.total_input_tokens:,}")
    table.add_row("Output Tokens", f"{result.trajectory.total_output_tokens:,}")
    table.add_row("Duration", f"{result.trajectory.duration_seconds:.1f}s")

    console.print(table)


def _display_benchmark_summary(eval_result: EvaluationResult) -> None:
    """Display benchmark summary with metrics."""
    table = Table(title=f"Benchmark Results: {eval_result.config_name}")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")
    table.add_column("Unit")

    for metric in eval_result.metrics:
        table.add_row(metric.name, f"{metric.value}", metric.unit)

    console.print(table)


def _display_comparison(eval_results: list[EvaluationResult]) -> None:
    """Display comparison table across multiple configs."""
    table = Table(title="Configuration Comparison")
    table.add_column("Metric", style="bold")

    for er in eval_results:
        table.add_column(er.config_name, justify="right")

    # Collect all metric names
    all_metrics: set[str] = set()
    for er in eval_results:
        for m in er.metrics:
            all_metrics.add(m.name)

    for metric_name in sorted(all_metrics):
        row = [metric_name]
        for er in eval_results:
            val = next((m for m in er.metrics if m.name == metric_name), None)
            if val:
                row.append(f"{val.value} {val.unit}")
            else:
                row.append("—")
        table.add_row(*row)

    console.print(table)


if __name__ == "__main__":
    main()
