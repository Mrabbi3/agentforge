"""Quantitative metrics calculator for AgentForge evaluation."""

from __future__ import annotations

from agentforge.core.schemas import AgentResult, MetricResult

SONNET_INPUT_PRICE = 3.0
SONNET_OUTPUT_PRICE = 15.0


class MetricsCalculator:
    """Calculate evaluation metrics from agent results."""

    def __init__(self, results: list[AgentResult]) -> None:
        self.results = results

    def compute(self, metric_names: list[str]) -> list[MetricResult]:
        """Compute requested metrics and return results."""
        dispatch = {
            "pass_rate": self.pass_rate,
            "partial_credit": self.partial_credit,
            "error_recovery_rate": self.error_recovery_rate,
            "tool_efficiency": self.tool_efficiency,
            "cost_per_task": self.cost_per_task,
            "context_utilization": self.context_utilization,
            "avg_steps": self.avg_steps,
            "avg_duration": self.avg_duration,
        }
        out: list[MetricResult] = []
        for name in metric_names:
            func = dispatch.get(name)
            if func is not None:
                out.append(func())
        return out

    def pass_rate(self) -> MetricResult:
        if not self.results:
            return MetricResult(name="pass_rate", value=0.0, unit="%")
        passed = sum(1 for r in self.results if r.passed)
        rate = (passed / len(self.results)) * 100
        return MetricResult(name="pass_rate", value=round(rate, 1), unit="%")

    def partial_credit(self) -> MetricResult:
        if not self.results:
            return MetricResult(name="partial_credit", value=0.0, unit="%")
        scores: list[float] = []
        for r in self.results:
            if r.tests_total > 0:
                scores.append(r.tests_passed / r.tests_total)
            else:
                scores.append(0.0)
        avg = (sum(scores) / len(scores)) * 100
        return MetricResult(name="partial_credit", value=round(avg, 1), unit="%")

    def error_recovery_rate(self) -> MetricResult:
        tasks_with_errors: list[AgentResult] = []
        for r in self.results:
            has_error = any(not tc.success for tc in r.trajectory.tool_calls)
            if has_error:
                tasks_with_errors.append(r)
        if not tasks_with_errors:
            return MetricResult(name="error_recovery_rate", value=100.0, unit="%")
        recovered = sum(1 for r in tasks_with_errors if r.passed)
        rate = (recovered / len(tasks_with_errors)) * 100
        return MetricResult(name="error_recovery_rate", value=round(rate, 1), unit="%")

    def tool_efficiency(self) -> MetricResult:
        if not self.results:
            return MetricResult(name="tool_efficiency", value=0.0, unit="%")
        ratios: list[float] = []
        for r in self.results:
            calls = r.trajectory.tool_calls
            if calls:
                success = sum(1 for tc in calls if tc.success)
                ratios.append(success / len(calls))
            else:
                ratios.append(1.0)
        avg = (sum(ratios) / len(ratios)) * 100
        return MetricResult(name="tool_efficiency", value=round(avg, 1), unit="%")

    def cost_per_task(self) -> MetricResult:
        if not self.results:
            return MetricResult(name="cost_per_task", value=0.0, unit="$")
        total_cost = 0.0
        for r in self.results:
            t = r.trajectory
            input_cost = (t.total_input_tokens / 1_000_000) * SONNET_INPUT_PRICE
            output_cost = (t.total_output_tokens / 1_000_000) * SONNET_OUTPUT_PRICE
            total_cost += input_cost + output_cost
        avg = total_cost / len(self.results)
        return MetricResult(name="cost_per_task", value=round(avg, 4), unit="$")

    def context_utilization(self) -> MetricResult:
        if not self.results:
            return MetricResult(name="context_utilization", value=0.0, unit="%")
        ratios: list[float] = []
        max_ctx = 90_000
        for r in self.results:
            total = r.trajectory.total_input_tokens + r.trajectory.total_output_tokens
            ratios.append(min(total / max_ctx, 1.0))
        avg = (sum(ratios) / len(ratios)) * 100
        return MetricResult(name="context_utilization", value=round(avg, 1), unit="%")

    def avg_steps(self) -> MetricResult:
        if not self.results:
            return MetricResult(name="avg_steps", value=0.0, unit="steps")
        total = sum(r.trajectory.total_steps for r in self.results)
        avg = total / len(self.results)
        return MetricResult(name="avg_steps", value=round(avg, 1), unit="steps")

    def avg_duration(self) -> MetricResult:
        if not self.results:
            return MetricResult(name="avg_duration", value=0.0, unit="s")
        total = sum(r.trajectory.duration for r in self.results)
        avg = total / len(self.results)
        return MetricResult(name="avg_duration", value=round(avg, 2), unit="s")
