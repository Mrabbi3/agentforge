"""Evaluation metrics for AgentForge benchmark runs.

Computes quantitative metrics across task performance, efficiency,
and agent quality dimensions.
"""

from __future__ import annotations

from typing import Any

from agentforge.core.schemas import (
    AgentResult,
    AgentStatus,
    EvaluationConfig,
    MetricResult,
)


class MetricsCalculator:
    """Computes evaluation metrics from a set of agent results."""

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config

    def compute_all(self, results: list[AgentResult]) -> list[MetricResult]:
        """Compute all configured metrics."""
        metrics: list[MetricResult] = []

        metric_funcs: dict[str, Any] = {
            "pass_rate": self._pass_rate,
            "partial_credit": self._partial_credit,
            "error_recovery_rate": self._error_recovery_rate,
            "tool_efficiency": self._tool_efficiency,
            "cost_per_task": self._cost_per_task,
            "context_utilization": self._context_utilization,
            "avg_steps": self._avg_steps,
            "avg_duration": self._avg_duration,
        }

        for metric_name in self.config.metrics:
            func = metric_funcs.get(metric_name)
            if func:
                metrics.append(func(results))

        return metrics

    def _pass_rate(self, results: list[AgentResult]) -> MetricResult:
        """Fraction of tasks fully passed."""
        if not results:
            return MetricResult(name="pass_rate", value=0.0, unit="%")

        passed = sum(1 for r in results if r.passed)
        rate = (passed / len(results)) * 100

        return MetricResult(
            name="pass_rate",
            value=round(rate, 2),
            unit="%",
            details={
                "passed": passed,
                "total": len(results),
                "by_status": _count_by_status(results),
            },
        )

    def _partial_credit(self, results: list[AgentResult]) -> MetricResult:
        """Average fraction of tests passed per task."""
        if not results:
            return MetricResult(name="partial_credit", value=0.0, unit="%")

        scores = []
        for r in results:
            if r.tests_total > 0:
                scores.append(r.tests_passed / r.tests_total)
            else:
                scores.append(1.0 if r.status == AgentStatus.SUCCESS else 0.0)

        avg = (sum(scores) / len(scores)) * 100

        return MetricResult(
            name="partial_credit",
            value=round(avg, 2),
            unit="%",
            details={"per_task": [round(s * 100, 1) for s in scores]},
        )

    def _error_recovery_rate(self, results: list[AgentResult]) -> MetricResult:
        """How often does the agent recover from tool errors?"""
        total_errors = 0
        recoveries = 0

        for r in results:
            steps = r.trajectory.steps
            for i, step in enumerate(steps):
                for tc in step.tool_calls:
                    if not tc.success:
                        total_errors += 1
                        # Check if a subsequent step succeeded
                        if any(
                            any(tc2.success for tc2 in s.tool_calls)
                            for s in steps[i + 1 :]
                        ):
                            recoveries += 1

        rate = (recoveries / total_errors * 100) if total_errors > 0 else 100.0

        return MetricResult(
            name="error_recovery_rate",
            value=round(rate, 2),
            unit="%",
            details={"total_errors": total_errors, "recoveries": recoveries},
        )

    def _tool_efficiency(self, results: list[AgentResult]) -> MetricResult:
        """Average tool calls per task (lower = more efficient)."""
        if not results:
            return MetricResult(name="tool_efficiency", value=0.0, unit="calls/task")

        total_calls = sum(r.trajectory.total_tool_calls for r in results)
        avg = total_calls / len(results)

        return MetricResult(
            name="tool_efficiency",
            value=round(avg, 2),
            unit="calls/task",
            details={
                "total_calls": total_calls,
                "per_task": [r.trajectory.total_tool_calls for r in results],
            },
        )

    def _cost_per_task(self, results: list[AgentResult]) -> MetricResult:
        """Estimated cost per task based on token usage.

        Uses approximate Claude Sonnet pricing:
        - Input: $3 / 1M tokens
        - Output: $15 / 1M tokens
        """
        if not results:
            return MetricResult(name="cost_per_task", value=0.0, unit="$")

        input_cost_per_token = 3.0 / 1_000_000
        output_cost_per_token = 15.0 / 1_000_000

        costs = []
        for r in results:
            cost = (
                r.trajectory.total_input_tokens * input_cost_per_token
                + r.trajectory.total_output_tokens * output_cost_per_token
            )
            costs.append(cost)

        avg_cost = sum(costs) / len(costs)

        return MetricResult(
            name="cost_per_task",
            value=round(avg_cost, 4),
            unit="$",
            details={
                "total_cost": round(sum(costs), 4),
                "per_task": [round(c, 4) for c in costs],
                "total_input_tokens": sum(r.trajectory.total_input_tokens for r in results),
                "total_output_tokens": sum(r.trajectory.total_output_tokens for r in results),
            },
        )

    def _context_utilization(self, results: list[AgentResult]) -> MetricResult:
        """How much of the available context window was used on average."""
        if not results:
            return MetricResult(name="context_utilization", value=0.0, unit="%")

        # Approximate: use total input tokens as a proxy for context usage
        # Compared against a typical 200k token context window
        max_context = 200_000

        utilizations = []
        for r in results:
            util = (r.trajectory.total_input_tokens / max_context) * 100
            utilizations.append(min(util, 100.0))

        avg = sum(utilizations) / len(utilizations)

        return MetricResult(
            name="context_utilization",
            value=round(avg, 2),
            unit="%",
            details={"per_task": [round(u, 2) for u in utilizations]},
        )

    def _avg_steps(self, results: list[AgentResult]) -> MetricResult:
        """Average number of steps per task."""
        if not results:
            return MetricResult(name="avg_steps", value=0.0, unit="steps")

        steps = [r.trajectory.total_steps for r in results]
        avg = sum(steps) / len(steps)

        return MetricResult(
            name="avg_steps",
            value=round(avg, 2),
            unit="steps",
            details={"per_task": steps},
        )

    def _avg_duration(self, results: list[AgentResult]) -> MetricResult:
        """Average duration per task in seconds."""
        if not results:
            return MetricResult(name="avg_duration", value=0.0, unit="s")

        durations = [r.trajectory.duration_seconds for r in results]
        avg = sum(durations) / len(durations)

        return MetricResult(
            name="avg_duration",
            value=round(avg, 2),
            unit="s",
            details={"per_task": [round(d, 2) for d in durations]},
        )


def _count_by_status(results: list[AgentResult]) -> dict[str, int]:
    """Count results by status."""
    counts: dict[str, int] = {}
    for r in results:
        counts[r.status.value] = counts.get(r.status.value, 0) + 1
    return counts
