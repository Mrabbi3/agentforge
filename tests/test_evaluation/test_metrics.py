"""Tests for agentforge.evaluation.metrics — 5 tests."""

from agentforge.core.schemas import (
    AgentResult,
    AgentStatus,
    ToolCall,
    Trajectory,
    TrajectoryStep,
    StepType,
)
from agentforge.evaluation.metrics import MetricsCalculator


def _make_result(
    passed: bool = True,
    steps: int = 5,
    input_tokens: int = 1000,
    output_tokens: int = 500,
    tool_success: bool = True,
) -> AgentResult:
    """Helper to create an AgentResult for testing."""
    traj = Trajectory(task_id="test")
    for i in range(steps):
        tc = ToolCall(
            tool_name="bash_execute",
            tool_input={"command": "echo hi"},
            tool_result="hi",
            success=tool_success,
        )
        step = TrajectoryStep(
            step_number=i + 1,
            step_type=StepType.TOOL_USE,
            input_tokens=input_tokens // steps,
            output_tokens=output_tokens // steps,
            tool_calls=[tc],
        )
        traj.add_step(step)
    traj.finalize()
    return AgentResult(
        task_id="test",
        status=AgentStatus.SUCCESS if passed else AgentStatus.FAILURE,
        trajectory=traj,
        tests_passed=1 if passed else 0,
        tests_total=1,
    )


def test_pass_rate():
    """Pass rate is correctly computed."""
    results = [_make_result(True), _make_result(True), _make_result(False)]
    calc = MetricsCalculator(results)
    m = calc.pass_rate()
    assert m.name == "pass_rate"
    assert abs(m.value - 66.7) < 0.1


def test_tool_efficiency_all_success():
    """Tool efficiency is 100% when all calls succeed."""
    results = [_make_result(tool_success=True)]
    calc = MetricsCalculator(results)
    m = calc.tool_efficiency()
    assert m.value == 100.0


def test_cost_per_task():
    """Cost is computed using Sonnet pricing."""
    results = [_make_result(input_tokens=1_000_000, output_tokens=100_000)]
    calc = MetricsCalculator(results)
    m = calc.cost_per_task()
    assert m.name == "cost_per_task"
    assert m.value > 0


def test_avg_steps():
    """Average steps is correctly computed."""
    results = [_make_result(steps=10), _make_result(steps=6)]
    calc = MetricsCalculator(results)
    m = calc.avg_steps()
    assert m.value == 8.0


def test_empty_results():
    """Metrics handle empty results gracefully."""
    calc = MetricsCalculator([])
    assert calc.pass_rate().value == 0.0
    assert calc.avg_steps().value == 0.0
    assert calc.cost_per_task().value == 0.0
