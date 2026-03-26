"""Tests for core schemas and data models."""

import time

from agentforge.core.schemas import (
    AgentResult,
    AgentStatus,
    HarnessConfig,
    MemoryConfig,
    MemoryStrategy,
    MetricResult,
    Task,
    TaskCategory,
    TaskDifficulty,
    ToolCall,
    Trajectory,
    TrajectoryStep,
    StepType,
)


def test_harness_config_defaults():
    """HarnessConfig should have sensible defaults."""
    config = HarnessConfig()
    assert config.name == "default"
    assert config.agent.model == "claude-sonnet-4-20250514"
    assert config.agent.max_steps == 25
    assert config.memory.strategy == MemoryStrategy.SLIDING_WINDOW
    assert "bash_execute" in config.tools.enabled


def test_harness_config_custom():
    """HarnessConfig should accept custom values."""
    config = HarnessConfig(
        name="test",
        agent={"model": "claude-haiku-4-5-20251001", "max_steps": 10},
        memory={"strategy": "summarization", "max_context_tokens": 50000},
    )
    assert config.name == "test"
    assert config.agent.model == "claude-haiku-4-5-20251001"
    assert config.memory.strategy == MemoryStrategy.SUMMARIZATION


def test_task_creation():
    """Task should be creatable with required fields."""
    task = Task(
        name="test_task",
        description="Fix a bug",
        category=TaskCategory.CODING,
        difficulty=TaskDifficulty.EASY,
    )
    assert task.name == "test_task"
    assert task.category == TaskCategory.CODING
    assert len(task.id) == 8  # UUID prefix


def test_trajectory_tracking():
    """Trajectory should track steps and tokens correctly."""
    traj = Trajectory(task_id="test", config_name="default")

    step1 = TrajectoryStep(
        step_number=0,
        step_type=StepType.LLM_CALL,
        input_tokens=100,
        output_tokens=50,
        model="claude-sonnet-4-20250514",
    )
    step2 = TrajectoryStep(
        step_number=1,
        step_type=StepType.LLM_CALL,
        input_tokens=200,
        output_tokens=75,
        model="claude-sonnet-4-20250514",
        tool_calls=[
            ToolCall(
                tool_name="bash_execute",
                tool_input={"command": "ls"},
                tool_result="file1.py\nfile2.py",
                duration_ms=100,
                success=True,
            )
        ],
    )

    traj.add_step(step1)
    traj.add_step(step2)

    assert traj.total_steps == 2
    assert traj.total_input_tokens == 300
    assert traj.total_output_tokens == 125
    assert traj.total_tool_calls == 1


def test_trajectory_finalize():
    """Trajectory finalize should set end time."""
    traj = Trajectory(task_id="test", config_name="default")
    assert traj.end_time is None

    traj.finalize()
    assert traj.end_time is not None
    assert traj.duration_seconds >= 0


def test_agent_result_passed():
    """AgentResult.passed should check status and test results."""
    traj = Trajectory(task_id="test", config_name="default")

    # Passed case
    result = AgentResult(
        task_id="test",
        status=AgentStatus.SUCCESS,
        trajectory=traj,
        tests_passed=5,
        tests_total=5,
    )
    assert result.passed is True

    # Failed case — not all tests passed
    result2 = AgentResult(
        task_id="test",
        status=AgentStatus.SUCCESS,
        trajectory=traj,
        tests_passed=3,
        tests_total=5,
    )
    assert result2.passed is False

    # Failed case — wrong status
    result3 = AgentResult(
        task_id="test",
        status=AgentStatus.FAILURE,
        trajectory=traj,
        tests_passed=5,
        tests_total=5,
    )
    assert result3.passed is False


def test_metric_result():
    """MetricResult should store name, value, and details."""
    metric = MetricResult(
        name="pass_rate",
        value=85.5,
        unit="%",
        details={"passed": 17, "total": 20},
    )
    assert metric.name == "pass_rate"
    assert metric.value == 85.5
    assert metric.details["passed"] == 17
