"""Tests for agentforge.core.schemas — 7 tests."""

from datetime import datetime, timedelta

from agentforge.core.schemas import (
    AgentResult,
    AgentStatus,
    HarnessConfig,
    MemoryStrategy,
    StepType,
    Task,
    TaskCategory,
    TaskDifficulty,
    ToolCall,
    Trajectory,
    TrajectoryStep,
)


def test_enum_values():
    """StrEnum values match expected strings."""
    assert MemoryStrategy.SLIDING_WINDOW == "sliding_window"
    assert TaskCategory.CODING == "coding"
    assert TaskDifficulty.HARD == "hard"
    assert StepType.TOOL_USE == "tool_use"
    assert AgentStatus.SUCCESS == "success"


def test_task_creation():
    """Task creates with defaults and generates an id."""
    t = Task(name="test", description="desc")
    assert len(t.id) == 8
    assert t.name == "test"
    assert t.category == TaskCategory.CODING
    assert t.difficulty == TaskDifficulty.MEDIUM
    assert t.timeout == 180


def test_harness_config_defaults():
    """HarnessConfig populates all nested defaults."""
    cfg = HarnessConfig()
    assert cfg.name == "default"
    assert cfg.agent.model == "claude-sonnet-4-20250514"
    assert cfg.memory.strategy == MemoryStrategy.SUMMARIZATION
    assert len(cfg.tools.enabled) == 4
    assert cfg.evaluation.judge.enabled is True


def test_trajectory_add_step():
    """Trajectory tracks steps and accumulates tokens."""
    traj = Trajectory(task_id="abc")
    step = TrajectoryStep(
        step_number=1,
        step_type=StepType.TOOL_USE,
        input_tokens=100,
        output_tokens=50,
    )
    traj.add_step(step)
    assert traj.total_steps == 1
    assert traj.total_input_tokens == 100
    assert traj.total_output_tokens == 50


def test_trajectory_finalize_and_duration():
    """Finalize sets end_time and duration is computed."""
    traj = Trajectory(task_id="abc")
    traj.start_time = datetime.now() - timedelta(seconds=5)
    traj.finalize()
    assert traj.end_time is not None
    assert traj.duration >= 4.0


def test_trajectory_tool_calls():
    """Tool calls are aggregated across steps."""
    traj = Trajectory(task_id="t1")
    tc1 = ToolCall(tool_name="bash_execute", tool_input={"command": "ls"})
    tc2 = ToolCall(tool_name="file_read", tool_input={"path": "/tmp/x"})
    step = TrajectoryStep(
        step_number=1,
        step_type=StepType.TOOL_USE,
        tool_calls=[tc1, tc2],
    )
    traj.add_step(step)
    assert len(traj.tool_calls) == 2
    assert traj.tool_calls[0].tool_name == "bash_execute"


def test_agent_result_passed():
    """AgentResult.passed reflects test outcomes."""
    traj = Trajectory(task_id="t1")
    r1 = AgentResult(
        task_id="t1",
        status=AgentStatus.SUCCESS,
        trajectory=traj,
        tests_passed=3,
        tests_total=3,
    )
    assert r1.passed is True
    r2 = AgentResult(
        task_id="t2",
        status=AgentStatus.FAILURE,
        trajectory=traj,
        tests_passed=1,
        tests_total=3,
    )
    assert r2.passed is False
