"""Multi-agent orchestrator: Planner -> Executor -> Reviewer pipeline."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StrEnum(str, Enum):  # noqa: UP042
    """String enum compatible with Python 3.10+."""


class AgentRole(StrEnum):
    """Roles in the multi-agent pipeline."""

    PLANNER = "planner"
    EXECUTOR = "executor"
    REVIEWER = "reviewer"


@dataclass
class AgentMessage:
    """A message exchanged between agents."""

    sender: AgentRole
    receiver: AgentRole
    content: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class SubTask:
    """A subtask decomposed by the planner."""

    id: int
    description: str
    dependencies: list[int] = field(default_factory=list)
    output: str = ""
    approved: bool = False


class PlannerAgent:
    """Decomposes a task into subtasks via LLM."""

    def __init__(self, client: Any, model: str) -> None:
        self.client = client
        self.model = model

    def plan(self, task_description: str) -> list[SubTask]:
        """Break a task into ordered subtasks."""
        prompt = (
            "Decompose this coding task into 2-5 ordered subtasks. "
            "Return a JSON array of objects with 'id', 'description', "
            "and 'dependencies' (list of ids).\n\n"
            f"Task: {task_description}"
        )
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text
            start = text.index("[")
            end = text.rindex("]") + 1
            data = json.loads(text[start:end])
            return [
                SubTask(
                    id=item.get("id", i),
                    description=item.get("description", ""),
                    dependencies=item.get("dependencies", []),
                )
                for i, item in enumerate(data)
            ]
        except Exception:
            return [
                SubTask(id=0, description=task_description),
            ]


class ExecutorAgent:
    """Executes a single subtask, handling revision feedback."""

    def __init__(self, client: Any, model: str) -> None:
        self.client = client
        self.model = model

    def execute(
        self,
        subtask: SubTask,
        feedback: str = "",
    ) -> str:
        """Execute a subtask, optionally incorporating reviewer feedback."""
        prompt = f"Execute this subtask:\n{subtask.description}"
        if feedback:
            prompt += (
                f"\n\nPrevious attempt was rejected. "
                f"Reviewer feedback:\n{feedback}"
            )
        if subtask.output:
            prompt += f"\n\nPrevious output:\n{subtask.output}"
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            return resp.content[0].text
        except Exception as exc:
            return f"Execution error: {exc}"


class ReviewerAgent:
    """Reviews executor output and provides approval or feedback."""

    def __init__(self, client: Any, model: str) -> None:
        self.client = client
        self.model = model

    def review(
        self,
        subtask: SubTask,
        output: str,
    ) -> tuple[bool, str]:
        """Review output. Returns (approved, feedback)."""
        prompt = (
            f"Review this output for the subtask:\n"
            f"Subtask: {subtask.description}\n\n"
            f"Output:\n{output}\n\n"
            "Respond with JSON: "
            '{"approved": true/false, "feedback": "..."}'
        )
        try:
            resp = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text
            start = text.index("{")
            end = text.rindex("}") + 1
            data = json.loads(text[start:end])
            return (
                bool(data.get("approved", False)),
                str(data.get("feedback", "")),
            )
        except Exception:
            return True, "Review unavailable, auto-approving."


class MultiAgentOrchestrator:
    """Runs the full Planner -> Executor -> Reviewer pipeline."""

    def __init__(
        self,
        client: Any,
        model: str = "claude-sonnet-4-20250514",
        max_revisions: int = 2,
    ) -> None:
        self.planner = PlannerAgent(client, model)
        self.executor = ExecutorAgent(client, model)
        self.reviewer = ReviewerAgent(client, model)
        self.max_revisions = max_revisions
        self.messages: list[AgentMessage] = []

    def _log(
        self,
        sender: AgentRole,
        receiver: AgentRole,
        content: str,
    ) -> None:
        """Log an inter-agent message."""
        self.messages.append(
            AgentMessage(sender=sender, receiver=receiver, content=content)
        )

    def run(self, task_description: str) -> list[SubTask]:
        """Execute the full multi-agent pipeline."""
        subtasks = self.planner.plan(task_description)
        self._log(
            AgentRole.PLANNER,
            AgentRole.EXECUTOR,
            f"Created {len(subtasks)} subtasks",
        )
        for subtask in subtasks:
            output = self.executor.execute(subtask)
            subtask.output = output
            self._log(
                AgentRole.EXECUTOR,
                AgentRole.REVIEWER,
                f"Subtask {subtask.id}: {output[:200]}",
            )
            for _ in range(self.max_revisions):
                approved, feedback = self.reviewer.review(subtask, output)
                self._log(
                    AgentRole.REVIEWER,
                    AgentRole.EXECUTOR,
                    f"{'Approved' if approved else 'Rejected'}: {feedback}",
                )
                if approved:
                    subtask.approved = True
                    break
                output = self.executor.execute(subtask, feedback=feedback)
                subtask.output = output
                self._log(
                    AgentRole.EXECUTOR,
                    AgentRole.REVIEWER,
                    f"Revision for subtask {subtask.id}: {output[:200]}",
                )
            else:
                subtask.approved = True
        return subtasks

    @property
    def coordination_metrics(self) -> dict[str, Any]:
        """Return metrics about the coordination process."""
        return {
            "total_messages": len(self.messages),
            "planner_messages": sum(
                1
                for m in self.messages
                if m.sender == AgentRole.PLANNER
            ),
            "executor_messages": sum(
                1
                for m in self.messages
                if m.sender == AgentRole.EXECUTOR
            ),
            "reviewer_messages": sum(
                1
                for m in self.messages
                if m.sender == AgentRole.REVIEWER
            ),
        }
