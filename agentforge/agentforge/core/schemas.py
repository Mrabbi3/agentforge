"""Core data schemas for AgentForge using Pydantic."""

from __future__ import annotations

import time
import uuid
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

# ─── Enums ────────────────────────────────────────────────────────────────────


class MemoryStrategy(StrEnum):
    SLIDING_WINDOW = "sliding_window"
    SUMMARIZATION = "summarization"
    RAG = "rag"
    HYBRID = "hybrid"


class TaskCategory(StrEnum):
    CODING = "coding"
    KNOWLEDGE = "knowledge"


class TaskDifficulty(StrEnum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class StepType(StrEnum):
    LLM_CALL = "llm_call"
    TOOL_USE = "tool_use"
    TOOL_RESULT = "tool_result"
    MEMORY_COMPACT = "memory_compact"
    ERROR = "error"


class AgentStatus(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"
    MAX_STEPS_EXCEEDED = "max_steps_exceeded"
    TIMEOUT = "timeout"
    ERROR = "error"


# ─── Configuration ────────────────────────────────────────────────────────────


class MemoryConfig(BaseModel):
    strategy: MemoryStrategy = MemoryStrategy.SLIDING_WINDOW
    max_context_tokens: int = 90_000
    compact_threshold: float = 0.8  # compact when context reaches this % of max
    window_size: int = 20  # for sliding window
    summary_model: str = "claude-haiku-4-5-20251001"


class ToolsConfig(BaseModel):
    enabled: list[str] = Field(
        default_factory=lambda: ["bash_execute", "file_read", "file_write", "file_search"]
    )
    bash_timeout: int = 30
    max_output_chars: int = 10_000


class JudgeConfig(BaseModel):
    enabled: bool = True
    model: str = "claude-sonnet-4-20250514"
    dimensions: list[str] = Field(
        default_factory=lambda: ["reasoning_coherence", "plan_adherence", "safety"]
    )


class EvaluationConfig(BaseModel):
    metrics: list[str] = Field(
        default_factory=lambda: [
            "pass_rate",
            "tool_efficiency",
            "cost_per_task",
            "context_utilization",
        ]
    )
    judge: JudgeConfig = Field(default_factory=JudgeConfig)


class AgentConfig(BaseModel):
    model: str = "claude-sonnet-4-20250514"
    max_steps: int = 25
    temperature: float = 0.0
    system_prompt: str | None = None


class HarnessConfig(BaseModel):
    """Top-level configuration for an AgentForge run."""

    name: str = "default"
    agent: AgentConfig = Field(default_factory=AgentConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)


# ─── Task Definitions ─────────────────────────────────────────────────────────


class Task(BaseModel):
    """A single task for the agent to solve."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str
    description: str
    category: TaskCategory = TaskCategory.CODING
    difficulty: TaskDifficulty = TaskDifficulty.MEDIUM
    setup_commands: list[str] = Field(default_factory=list)
    test_commands: list[str] = Field(default_factory=list)
    gold_patch: str | None = None
    repo_url: str | None = None
    timeout: int = 300  # seconds
    tags: list[str] = Field(default_factory=list)


# ─── Trajectory / Logging ─────────────────────────────────────────────────────


class ToolCall(BaseModel):
    """A single tool invocation within a step."""

    tool_name: str
    tool_input: dict[str, Any]
    tool_result: str
    duration_ms: float
    success: bool
    error: str | None = None


class TrajectoryStep(BaseModel):
    """A single step in the agent's execution trajectory."""

    step_number: int
    step_type: StepType
    timestamp: float = Field(default_factory=time.time)
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    reasoning: str | None = None  # extracted reasoning from the LLM response
    duration_ms: float = 0.0


class Trajectory(BaseModel):
    """Complete execution trajectory for a single task run."""

    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    config_name: str
    steps: list[TrajectoryStep] = Field(default_factory=list)
    start_time: float = Field(default_factory=time.time)
    end_time: float | None = None
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def add_step(self, step: TrajectoryStep) -> None:
        self.steps.append(step)
        self.total_input_tokens += step.input_tokens
        self.total_output_tokens += step.output_tokens

    def finalize(self) -> None:
        self.end_time = time.time()

    @property
    def total_steps(self) -> int:
        return len(self.steps)

    @property
    def total_tool_calls(self) -> int:
        return sum(len(s.tool_calls) for s in self.steps)

    @property
    def duration_seconds(self) -> float:
        if self.end_time is None:
            return time.time() - self.start_time
        return self.end_time - self.start_time


# ─── Agent Result ──────────────────────────────────────────────────────────────


class AgentResult(BaseModel):
    """The outcome of running an agent on a single task."""

    task_id: str
    status: AgentStatus
    trajectory: Trajectory
    output: str | None = None  # the agent's final answer or patch
    tests_passed: int = 0
    tests_total: int = 0
    error_message: str | None = None

    @property
    def passed(self) -> bool:
        return self.status == AgentStatus.SUCCESS and self.tests_passed == self.tests_total


# ─── Evaluation Results ───────────────────────────────────────────────────────


class MetricResult(BaseModel):
    """A single computed metric."""

    name: str
    value: float
    unit: str = ""
    details: dict[str, Any] = Field(default_factory=dict)


class JudgeScore(BaseModel):
    """Model-based evaluation score for a single dimension."""

    dimension: str
    score: float  # 1-5
    reasoning: str


class EvaluationResult(BaseModel):
    """Complete evaluation results for a benchmark run."""

    config_name: str
    total_tasks: int
    results: list[AgentResult]
    metrics: list[MetricResult] = Field(default_factory=list)
    judge_scores: list[JudgeScore] = Field(default_factory=list)
    timestamp: float = Field(default_factory=time.time)
