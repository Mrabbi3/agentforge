"""CI Agent Runner: Runs AgentForge on detected failures and generates PR comments."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agentforge.core.agent_loop import run_agent
from agentforge.core.schemas import AgentResult, HarnessConfig, Task
from agentforge.evaluation.metrics import MetricsCalculator

import anthropic
import yaml


def load_config(path: str) -> HarnessConfig:
    """Load config from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return HarnessConfig(**data)


def load_task(path: str) -> Task:
    """Load task from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return Task(**data)


def format_duration(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    return f"{minutes}m {secs:.0f}s"


def generate_pr_comment(
    pr_number: int,
    results: list[AgentResult],
    metrics: list,
    analysis: dict,
    duration: float,
) -> str:
    """Generate a markdown comment for the PR."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    all_passed = passed == total

    # Status emoji and header
    if all_passed:
        status_icon = "white_check_mark"
        status_text = "Agent successfully identified fixes"
    elif passed > 0:
        status_icon = "warning"
        status_text = f"Agent fixed {passed}/{total} issues"
    else:
        status_icon = "x"
        status_text = "Agent could not resolve the issues"

    lines = [
        f"## AgentForge Analysis",
        "",
        f"**Status:** :{status_icon}: {status_text}",
        f"**Duration:** {format_duration(duration)} | **Tasks:** {total} | **Model:** claude-sonnet",
        "",
    ]

    # Changed files summary
    if analysis.get("changed_files"):
        files = analysis["changed_files"][:10]
        lines.append("<details>")
        lines.append(f"<summary>Changed files ({len(analysis['changed_files'])})</summary>")
        lines.append("")
        for f in files:
            lines.append(f"- `{f}`")
        if len(analysis["changed_files"]) > 10:
            lines.append(f"- ... and {len(analysis['changed_files']) - 10} more")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    # Results table
    lines.append("### Results")
    lines.append("")
    lines.append("| Task | Status | Tests | Steps | Tokens |")
    lines.append("|------|--------|-------|-------|--------|")

    for r in results:
        status = "Passed" if r.passed else "Failed"
        icon = "white_check_mark" if r.passed else "x"
        total_tokens = r.trajectory.total_input_tokens + r.trajectory.total_output_tokens
        lines.append(
            f"| {r.task_id[:8]} | :{icon}: {status} | "
            f"{r.tests_passed}/{r.tests_total} | "
            f"{len(r.trajectory.steps)} | "
            f"{total_tokens:,} |"
        )

    lines.append("")

    # Metrics
    if metrics:
        lines.append("### Metrics")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("|--------|-------|")
        for m in metrics:
            if m.unit == "$":
                val = f"${m.value:.4f}"
            elif m.unit == "s":
                val = f"{m.value:.1f}s"
            else:
                val = f"{m.value:.1f}{m.unit}"
            lines.append(f"| {m.name.replace('_', ' ').title()} | {val} |")
        lines.append("")

    # Agent reasoning summary (for each result)
    lines.append("### Agent Reasoning")
    lines.append("")
    for r in results:
        lines.append(f"**Task `{r.task_id[:8]}`:**")
        # Get the last REFLECT step's reasoning
        reflect_steps = [
            s for s in r.trajectory.steps
            if s.reasoning and "REFLECT" in s.reasoning.upper()
        ]
        if reflect_steps:
            reasoning = reflect_steps[-1].reasoning
            # Trim to first 500 chars
            if len(reasoning) > 500:
                reasoning = reasoning[:500] + "..."
            lines.append(f"> {reasoning.replace(chr(10), chr(10) + '> ')}")
        elif r.output:
            output = r.output[:500] + ("..." if len(r.output) > 500 else "")
            lines.append(f"> {output.replace(chr(10), chr(10) + '> ')}")
        else:
            lines.append("> No detailed reasoning captured.")
        lines.append("")

    # Tool usage summary
    all_tools = {}
    for r in results:
        for step in r.trajectory.steps:
            for tc in step.tool_calls:
                name = tc.tool_name
                all_tools[name] = all_tools.get(name, 0) + 1

    if all_tools:
        lines.append("<details>")
        lines.append("<summary>Tool usage breakdown</summary>")
        lines.append("")
        lines.append("| Tool | Calls |")
        lines.append("|------|-------|")
        for name, count in sorted(all_tools.items(), key=lambda x: -x[1]):
            lines.append(f"| `{name}` | {count} |")
        lines.append("")
        lines.append("</details>")
        lines.append("")

    # Footer
    lines.append("---")
    lines.append(
        f"*Generated by [AgentForge](https://github.com/Mrabbi3/agentforge) "
        f"v0.1.0 at {timestamp}*"
    )

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run AgentForge agent on CI failures")
    parser.add_argument("--pr", type=int, required=True, help="PR number")
    parser.add_argument("--config", default="configs/default.yaml", help="Config path")
    parser.add_argument("--output", default="results/ci/", help="Output directory")
    args = parser.parse_args()

    ci_dir = Path(args.output)
    ci_dir.mkdir(parents=True, exist_ok=True)

    # Load analysis
    analysis_path = ci_dir / "analysis.json"
    if not analysis_path.exists():
        print("No analysis.json found. Run ci_analyze.py first.")
        sys.exit(1)

    analysis = json.loads(analysis_path.read_text())

    if not analysis.get("needs_agent"):
        print("No agent intervention needed.")
        return

    # Load config
    cfg = load_config(args.config)
    client = anthropic.Anthropic()

    # Find generated tasks
    tasks_dir = ci_dir / "tasks"
    task_files = sorted(tasks_dir.glob("*.yaml")) if tasks_dir.exists() else []

    if not task_files:
        print("No task files generated. Nothing to run.")
        comment = generate_pr_comment(args.pr, [], [], analysis, 0)
        (ci_dir / "pr_comment.md").write_text(comment)
        return

    print(f"Running agent on {len(task_files)} task(s)...")
    start_time = time.time()
    results: list[AgentResult] = []

    for tf in task_files:
        print(f"\n  Task: {tf.name}")
        try:
            task = load_task(str(tf))
            result = run_agent(task, cfg, client)
            results.append(result)
            status = "PASSED" if result.passed else "FAILED"
            print(f"  Result: {status} ({result.tests_passed}/{result.tests_total})")
        except Exception as e:
            print(f"  Error running task: {e}")

    duration = time.time() - start_time
    print(f"\nCompleted in {format_duration(duration)}")

    # Compute metrics
    calc = MetricsCalculator(results)
    metrics = calc.compute(cfg.evaluation.metrics) if results else []

    # Save results
    results_data = {
        "pr_number": args.pr,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_seconds": duration,
        "results": [r.model_dump() for r in results],
        "metrics": [m.model_dump() for m in metrics],
    }
    (ci_dir / "agent_results.json").write_text(json.dumps(results_data, indent=2, default=str))

    # Generate PR comment
    comment = generate_pr_comment(args.pr, results, metrics, analysis, duration)
    (ci_dir / "pr_comment.md").write_text(comment)
    print(f"\nPR comment saved to {ci_dir / 'pr_comment.md'}")


if __name__ == "__main__":
    main()
