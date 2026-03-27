"""CLI for AgentForge: run, benchmark, and compare commands."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import yaml
from rich.console import Console
from rich.table import Table

from agentforge import __version__
from agentforge.core.agent_loop import run_agent
from agentforge.core.schemas import (
    AgentResult,
    EvaluationResult,
    HarnessConfig,
    Task,
)
from agentforge.evaluation.metrics import MetricsCalculator

console = Console()


def _load_config(path: str) -> HarnessConfig:
    """Load a YAML config file into a HarnessConfig."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return HarnessConfig(**data)


def _load_task(path: str) -> Task:
    """Load a YAML task file into a Task."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return Task(**data)


def _get_client() -> Any:
    """Create an Anthropic client."""
    import anthropic

    return anthropic.Anthropic()


def _display_results(results: list[AgentResult], config: HarnessConfig) -> None:
    """Display results in a Rich table."""
    table = Table(title="Agent Results")
    table.add_column("Task", style="cyan")
    table.add_column("Status", style="bold")
    table.add_column("Tests", justify="right")
    table.add_column("Steps", justify="right")

    for r in results:
        status_style = "green" if r.passed else "red"
        table.add_row(
            r.task_id,
            f"[{status_style}]{r.status}[/{status_style}]",
            f"{r.tests_passed}/{r.tests_total}",
            str(r.trajectory.total_steps),
        )
    console.print(table)

    # Metrics
    calc = MetricsCalculator(results)
    metrics = calc.compute(config.evaluation.metrics)
    if metrics:
        mtable = Table(title="Metrics")
        mtable.add_column("Metric", style="cyan")
        mtable.add_column("Value", justify="right")
        mtable.add_column("Unit")
        for m in metrics:
            mtable.add_row(m.name, str(m.value), m.unit)
        console.print(mtable)


@click.group()
@click.version_option(version=__version__, prog_name="agentforge")
def cli() -> None:
    """AgentForge: AI agent harness with pluggable memory and evaluation."""


@cli.command()
@click.option(
    "--task",
    required=True,
    type=click.Path(exists=True),
    help="Path to task YAML file.",
)
@click.option(
    "--config",
    default="configs/default.yaml",
    type=click.Path(exists=True),
    help="Path to config YAML file.",
)
@click.option(
    "--output",
    default=None,
    type=click.Path(),
    help="Path to save result JSON.",
)
def run(task: str, config: str, output: str | None) -> None:
    """Run the agent on a single task."""
    cfg = _load_config(config)
    t = _load_task(task)
    client = _get_client()

    console.print(f"[bold]Running task:[/bold] {t.name}")
    result = run_agent(t, cfg, client)
    _display_results([result], cfg)

    if output:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(
            result.model_dump_json(indent=2), encoding="utf-8"
        )
        console.print(f"[dim]Result saved to {output}[/dim]")


@cli.command()
@click.option(
    "--config",
    required=True,
    type=click.Path(exists=True),
    help="Path to config YAML file.",
)
@click.option(
    "--tasks-dir",
    default="tasks/coding",
    type=click.Path(exists=True),
    help="Directory containing task YAML files.",
)
@click.option(
    "--output",
    default="results",
    type=click.Path(),
    help="Directory to save results.",
)
def benchmark(config: str, tasks_dir: str, output: str) -> None:
    """Run a full benchmark across all tasks."""
    cfg = _load_config(config)
    client = _get_client()

    task_files = sorted(Path(tasks_dir).glob("*.yaml"))
    if not task_files:
        console.print("[red]No task files found[/red]")
        return

    console.print(
        f"[bold]Running benchmark:[/bold] {len(task_files)} tasks "
        f"with config '{cfg.name}'"
    )

    results: list[AgentResult] = []
    for tf in task_files:
        t = _load_task(str(tf))
        console.print(f"\n[bold cyan]Task: {t.name}[/bold cyan]")
        result = run_agent(t, cfg, client)
        results.append(result)

    _display_results(results, cfg)

    # Save results
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_result = EvaluationResult(
        metrics=MetricsCalculator(results).compute(cfg.evaluation.metrics),
        results=results,
    )
    out_path = out_dir / f"{cfg.name}_results.json"
    out_path.write_text(
        eval_result.model_dump_json(indent=2), encoding="utf-8"
    )
    console.print(f"[dim]Results saved to {out_path}[/dim]")


@cli.command()
@click.option(
    "--configs",
    required=True,
    nargs=2,
    type=click.Path(exists=True),
    help="Two config files to compare.",
)
@click.option(
    "--tasks-dir",
    default="tasks/coding",
    type=click.Path(exists=True),
    help="Directory containing task YAML files.",
)
def compare(configs: tuple[str, str], tasks_dir: str) -> None:
    """Compare two configurations side by side."""
    client = _get_client()
    task_files = sorted(Path(tasks_dir).glob("*.yaml"))

    if not task_files:
        console.print("[red]No task files found[/red]")
        return

    all_results: dict[str, list[AgentResult]] = {}
    for config_path in configs:
        cfg = _load_config(config_path)
        console.print(f"\n[bold]Config: {cfg.name}[/bold]")
        results: list[AgentResult] = []
        for tf in task_files:
            t = _load_task(str(tf))
            result = run_agent(t, cfg, client)
            results.append(result)
        all_results[cfg.name] = results

    # Comparison table
    table = Table(title="Configuration Comparison")
    table.add_column("Metric", style="cyan")
    for config_path in configs:
        cfg = _load_config(config_path)
        table.add_column(cfg.name, justify="right")

    cfg1 = _load_config(configs[0])
    metric_names = cfg1.evaluation.metrics
    calcs = {
        name: MetricsCalculator(res) for name, res in all_results.items()
    }

    for metric_name in metric_names:
        row = [metric_name]
        for name, calc in calcs.items():
            results_list = calc.compute([metric_name])
            if results_list:
                row.append(f"{results_list[0].value} {results_list[0].unit}")
            else:
                row.append("-")
        table.add_row(*row)

    console.print(table)


if __name__ == "__main__":
    cli()
