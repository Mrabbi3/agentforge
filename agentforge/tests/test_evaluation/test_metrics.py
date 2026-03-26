"""Tests for evaluation metrics calculator."""

from agentforge.core.schemas import (
    AgentResult,
    AgentStatus,
    EvaluationConfig,
    StepType,
    ToolCall,
    Trajectory,
    TrajectoryStep,
)
from agentforge.evaluation.metrics import MetricsCalculator


def _make_result(
    passed: bool = True,
    tests_passed: int = 5,
    tests_total: int = 5,
    steps: int = 3,
    tool_calls: int = 5,
    input_tokens: int = 1000,
    output_tokens: int = 500,
) -> AgentResult:
    """Helper to create test AgentResults."""
    traj = Trajectory(task_id="test", config_name="default")

    for i in range(steps):
        step = TrajectoryStep(
            step_number=i,
            step_type=StepType.LLM_CALL,
            input_tokens=input_tokens // steps,
            output_tokens=output_tokens // steps,
            model="test",
            tool_calls=[
                ToolCall(
                    tool_name="bash_execute",
                    tool_input={"command": "test"},
                    tool_result="ok",
                    duration_ms=100,
                    success=True,
                )
            ]
            if i < tool_calls
            else [],
        )
        traj.add_step(step)

    traj.finalize()

    return AgentResult(
        task_id="test",
        status=AgentStatus.SUCCESS if passed else AgentStatus.FAILURE,
        trajectory=traj,
        tests_passed=tests_passed,
        tests_total=tests_total,
    )


def test_pass_rate():
    config = EvaluationConfig(metrics=["pass_rate"])
    calc = MetricsCalculator(config)

    results = [_make_result(passed=True), _make_result(passed=True), _make_result(passed=False)]
    metrics = calc.compute_all(results)

    assert len(metrics) == 1
    assert metrics[0].name == "pass_rate"
    assert abs(metrics[0].value - 66.67) < 0.1


def test_partial_credit():
    config = EvaluationConfig(metrics=["partial_credit"])
    calc = MetricsCalculator(config)

    results = [
        _make_result(tests_passed=5, tests_total=5),
        _make_result(tests_passed=3, tests_total=5, passed=False),
    ]
    metrics = calc.compute_all(results)

    assert metrics[0].name == "partial_credit"
    assert metrics[0].value == 80.0  # (100 + 60) / 2


def test_tool_efficiency():
    config = EvaluationConfig(metrics=["tool_efficiency"])
    calc = MetricsCalculator(config)

    results = [_make_result(tool_calls=5), _make_result(tool_calls=3)]
    metrics = calc.compute_all(results)

    assert metrics[0].name == "tool_efficiency"
    assert metrics[0].value > 0


def test_cost_per_task():
    config = EvaluationConfig(metrics=["cost_per_task"])
    calc = MetricsCalculator(config)

    results = [_make_result(input_tokens=10000, output_tokens=5000)]
    metrics = calc.compute_all(results)

    assert metrics[0].name == "cost_per_task"
    assert metrics[0].value > 0
    assert metrics[0].unit == "$"


def test_empty_results():
    config = EvaluationConfig(metrics=["pass_rate", "tool_efficiency"])
    calc = MetricsCalculator(config)

    metrics = calc.compute_all([])
    assert all(m.value == 0.0 for m in metrics)
