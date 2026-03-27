"""LLM-as-judge for trajectory quality scoring."""

from __future__ import annotations

import json
from typing import Any

from agentforge.core.schemas import JudgeConfig, JudgeScore, Trajectory

JUDGE_SYSTEM_PROMPT = """\
You are an expert evaluator of AI coding agent trajectories.

Score the trajectory on the requested dimensions using a 1-5 scale:
  1 = Very poor
  2 = Below average
  3 = Average
  4 = Good
  5 = Excellent

Return a JSON array of objects, each with:
  {"dimension": "<name>", "score": <1-5>, "reasoning": "<brief explanation>"}

Be fair, specific, and concise.
"""


def format_trajectory(trajectory: Trajectory) -> str:
    """Convert a trajectory into readable text for the judge."""
    lines: list[str] = []
    lines.append(f"Run ID: {trajectory.run_id}")
    lines.append(f"Task ID: {trajectory.task_id}")
    lines.append(f"Total steps: {trajectory.total_steps}")
    lines.append(
        f"Tokens: {trajectory.total_input_tokens} in / "
        f"{trajectory.total_output_tokens} out"
    )
    lines.append(f"Duration: {trajectory.duration:.1f}s")
    lines.append("")
    for step in trajectory.steps:
        lines.append(f"--- Step {step.step_number} ({step.step_type}) ---")
        if step.reasoning:
            lines.append(f"Reasoning: {step.reasoning[:300]}")
        for tc in step.tool_calls:
            status = "OK" if tc.success else f"FAIL: {tc.error}"
            lines.append(
                f"  Tool: {tc.tool_name} -> {status} "
                f"({tc.duration_ms:.0f}ms)"
            )
        lines.append("")
    return "\n".join(lines)


class ModelJudge:
    """LLM-based judge for evaluating trajectory quality."""

    def __init__(self, config: JudgeConfig, client: Any) -> None:
        self.config = config
        self.client = client

    def score(
        self,
        trajectory: Trajectory,
        dimensions: list[str] | None = None,
    ) -> list[JudgeScore]:
        """Score a trajectory on the specified dimensions."""
        dims = dimensions or self.config.dimensions
        trajectory_text = format_trajectory(trajectory)
        prompt = (
            f"Evaluate this agent trajectory on these dimensions: "
            f"{', '.join(dims)}\n\n{trajectory_text}"
        )
        try:
            response = self.client.messages.create(
                model=self.config.model,
                max_tokens=1024,
                system=JUDGE_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text
            return self._parse_scores(text, dims)
        except Exception:
            return [
                JudgeScore(dimension=d, score=3, reasoning="Judge unavailable")
                for d in dims
            ]

    @staticmethod
    def _parse_scores(
        text: str,
        dimensions: list[str],
    ) -> list[JudgeScore]:
        """Parse JSON scores from the judge response."""
        try:
            start = text.index("[")
            end = text.rindex("]") + 1
            data = json.loads(text[start:end])
            scores: list[JudgeScore] = []
            for item in data:
                scores.append(
                    JudgeScore(
                        dimension=item.get("dimension", "unknown"),
                        score=max(1, min(5, int(item.get("score", 3)))),
                        reasoning=item.get("reasoning", ""),
                    )
                )
            return scores
        except (ValueError, json.JSONDecodeError):
            return [
                JudgeScore(
                    dimension=d, score=3, reasoning="Failed to parse response"
                )
                for d in dimensions
            ]
