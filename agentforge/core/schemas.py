"""Pydantic models and enums for AgentForge."""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class StrEnum(str, Enum):  # noqa: UP042
    """String enum compatible with Python 3.10+."""


# -- Enums -----------------------------------------------------------------


class MemoryStrategy(StrEnum):
    """Available memory compaction strategies."""

    SLIDING_WINDOW = "sliding_window"
    SUMMARIZATION = "summarization"
    RAG = "rag"
    HYBRID = "hybrid"


class TaskCategory(StrEnum):
    """Categories of benchmark tasks."""

    CODING = "coding"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"


class TaskDifficulty(StrEnum):
    """Task difficulty levels."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class StepType(StrEnum):
    """Types of agent loop steps."""

    PLANNING = "planning"
    TOOL_USE = "tool_use"
    OBSERVATION = "observation"
    REFLECTION = "reflection"


class AgentStatus(StrEnum):
    """Terminal status of an agent run."""

    SUCCESS = "success"
    FAILURE = "failure"
    ERROR = "error"
    TIMEOUT = "timeout"


# -- Config Models ----------------------------------------------------------


class MemoryConfig(BaseModel):
    """Configuration for the memory subsystem."""

    strategy: MemoryStrategy = MemoryStrategy.SUMMARIZATION
    max_context_tokens: int = 90_000
    compact_threshold: float = 0.8


class ToolsConfig(BaseModel):
    """Configuration for available tools."""

    enabled: list[str] = Field(
        default_factory=lambda: [
            "bash_execute",
            "file_read",
            "file_write",
            "file_search",
        ]
    )
    bash_timeout: int = 30


class JudgeConfig(BaseModel):
    """Configuration for the LLM judge."""

    enabled: bool = True
    model: str = "claude-sonnet-4-20250514"
    dimensions: list[str] = Field(
        default_factory=lambda: [
            "reasoning_coherence",
            "plan_adherence",
            "safety",
        ]
    )


class EvaluationConfig(BaseModel):
    """Configuration for evaluation metrics and judge."""

    metrics: list[str] = Field(
        default_factory=lambda: [
            "pass_rate",
            "partial_credit",
            "tool_efficiency",
            "cost_per_task",
            "context_utilization",
            "avg_steps",
            "avg_duration",
            "error_recovery_rate",
        ]
    )
    judge: JudgeConfig = Field(default_factory=JudgeConfig)


class AgentConfig(BaseModel):
    """Configuration for the agent model."""

    model: str = "claude-sonnet-4-20250514"
    max_steps: int = 25
    temperature: float = 0.0


class HarnessConfig(BaseModel):
    """Top-level harness configuration."""

    name: str = "default"
    agent: AgentConfig = Field(default_factory=AgentConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


# -- Task Models ------------------------------------------------------------


class Task(BaseModel):
    """A single benchmark task."""

    id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    name: str
    description: str
    category: TaskCategory = TaskCategory.CODING
    difficulty: TaskDifficulty = TaskDifficulty.MEDIUM
    setup_commands: list[str] = Field(default_factory=list)
    test_commands: list[str] = Field(default_factory=list)
    gold_patch: str = ""
    timeout: int = 180
    tags: list[str] = Field(default_factory=list)


# -- Trajectory Models ------------------------------------------------------


class ToolCall(BaseModel):
    """A single tool invocation within a step."""

    tool_name: str
    tool_input: dict[str, Any] = Field(default_factory=dict)
    tool_result: str = ""
    duration_ms: float = 0.0
    success: bool = True
    error: str | None = None


class TrajectoryStep(BaseModel):
    """A single step in the agent trajectory."""

    step_number: int
    step_type: StepType
    timestamp: datetime = Field(default_factory=datetime.now)
    input_tokens: int = 0
    output_tokens: int = 0
    tool_calls: list[ToolCall] = Field(default_factory=list)
    reasoning: str = ""


class Trajectory(BaseModel):
    """Full trajectory of an agent run."""

    run_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:8])
    task_id: str
    steps: list[TrajectoryStep] = Field(default_factory=list)
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    start_time: datetime = Field(default_factory=datetime.now)
    end_time: datetime | None = None

    def add_step(self, step: TrajectoryStep) -> None:
        """Append a step and update token counters."""
        self.steps.append(step)
        self.total_input_tokens += step.input_tokens
        self.total_output_tokens += step.output_tokens

    def finalize(self) -> None:
        """Mark the trajectory as complete."""
        self.end_time = datetime.now()

    @property
    def total_steps(self) -> int:
        """Return the number of steps."""
        return len(self.steps)

    @property
    def tool_calls(self) -> list[ToolCall]:
        """Return all tool calls across steps."""
        calls: list[ToolCall] = []
        for step in self.steps:
            calls.extend(step.tool_calls)
        return calls

    @property
    def duration(self) -> float:
        """Duration in seconds, or 0 if not finalized."""
        if self.end_time is None:
            return 0.0
        return (self.end_time - self.start_time).total_seconds()


# -- Result Models ----------------------------------------------------------


class AgentResult(BaseModel):
    """Result of running an agent on a task."""

    task_id: str
    status: AgentStatus
    trajectory: Trajectory
    output: str = ""
    tests_passed: int = 0
    tests_total: int = 0

    @property
    def passed(self) -> bool:
        """Whether all tests passed."""
        return (
            self.tests_total > 0
            and self.tests_passed == self.tests_total
        )


class MetricResult(BaseModel):
    """A single metric computation result."""

    name: str
    value: float
    unit: str = ""


class JudgeScore(BaseModel):
    """Score from the LLM judge on one dimension."""

    dimension: str
    score: int = Field(ge=1, le=5)
    reasoning: str = ""


class EvaluationResult(BaseModel):
    """Full evaluation result for one or more runs."""

    metrics: list[MetricResult] = Field(default_factory=list)
    judge_scores: list[JudgeScore] = Field(default_factory=list)
    results: list[AgentResult] = Field(default_factory=list)
