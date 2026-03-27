"""CI Analysis Script: Detects failing tests from a PR and generates task YAMLs."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


def get_pr_diff(pr_number: int) -> str:
    """Get the diff for a PR using git."""
    try:
        result = subprocess.run(
            ["git", "diff", "origin/main...HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return result.stdout
    except Exception as e:
        print(f"Warning: Could not get diff: {e}")
        return ""


def get_changed_files() -> list[str]:
    """Get list of files changed relative to main."""
    try:
        result = subprocess.run(
            ["git", "diff", "--name-only", "origin/main...HEAD"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        return [f.strip() for f in result.stdout.strip().split("\n") if f.strip()]
    except Exception:
        return []


def run_tests() -> dict:
    """Run the test suite and capture failures."""
    result = subprocess.run(
        ["python", "-m", "pytest", "tests/", "-v", "--tb=long", "-q"],
        capture_output=True,
        text=True,
        timeout=300,
    )

    failures = []
    current_failure = None

    for line in result.stdout.split("\n"):
        # Detect FAILED lines
        if "FAILED" in line:
            match = re.search(r"FAILED\s+([\w/.:]+)", line)
            if match:
                failures.append({
                    "test": match.group(1),
                    "output": "",
                })

    # Also parse stderr for tracebacks
    if result.returncode != 0:
        error_output = result.stdout + "\n" + result.stderr

    return {
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "failures": failures,
        "passed": result.returncode == 0,
    }


def detect_lint_errors() -> list[dict]:
    """Run ruff and detect lint errors."""
    try:
        result = subprocess.run(
            ["ruff", "check", "agentforge", "--output-format=json"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.stdout.strip():
            return json.loads(result.stdout)
    except Exception:
        pass
    return []


def generate_task_yaml(
    task_name: str,
    description: str,
    changed_files: list[str],
    test_output: str,
    diff: str,
) -> str:
    """Generate a task YAML for the agent to fix."""
    # Truncate diff and test output if too long
    max_len = 2000
    if len(diff) > max_len:
        diff = diff[:max_len] + "\n... (truncated)"
    if len(test_output) > max_len:
        test_output = test_output[:max_len] + "\n... (truncated)"

    files_list = "\n".join(f"  - {f}" for f in changed_files[:20])

    yaml_content = f"""name: "{task_name}"
description: |
  {description}

  Changed files:
{files_list}

  Test output:
  {test_output.replace(chr(10), chr(10) + '  ')}
category: "bug_fix"
difficulty: "medium"
language: "python"
setup_commands:
  - "pip install -e '.[dev]'"
test_commands:
  - "python -m pytest tests/ -v --tb=short"
  - "ruff check agentforge"
gold_patch: ""
buggy_code: |
  See changed files in the repository.
  Diff:
  {diff.replace(chr(10), chr(10) + '  ')}
"""
    return yaml_content


def main():
    parser = argparse.ArgumentParser(description="Analyze PR for failing tests")
    parser.add_argument("--pr", type=int, required=True, help="PR number")
    args = parser.parse_args()

    print(f"Analyzing PR #{args.pr}...")

    # Create output directory
    ci_dir = Path("results/ci")
    ci_dir.mkdir(parents=True, exist_ok=True)
    tasks_dir = ci_dir / "tasks"
    tasks_dir.mkdir(exist_ok=True)

    # Gather information
    changed_files = get_changed_files()
    print(f"  Changed files: {len(changed_files)}")

    diff = get_pr_diff(args.pr)
    print(f"  Diff size: {len(diff)} chars")

    # Run tests
    print("  Running tests...")
    test_results = run_tests()
    print(f"  Tests passed: {test_results['passed']}")
    print(f"  Failures: {len(test_results['failures'])}")

    # Check lint
    print("  Running lint...")
    lint_errors = detect_lint_errors()
    print(f"  Lint errors: {len(lint_errors)}")

    # Generate analysis report
    analysis = {
        "pr_number": args.pr,
        "changed_files": changed_files,
        "tests_passed": test_results["passed"],
        "test_failures": test_results["failures"],
        "lint_errors_count": len(lint_errors),
        "needs_agent": not test_results["passed"] or len(lint_errors) > 0,
    }

    # Save analysis
    analysis_path = ci_dir / "analysis.json"
    analysis_path.write_text(json.dumps(analysis, indent=2))
    print(f"  Analysis saved to {analysis_path}")

    if not analysis["needs_agent"]:
        print("  All tests pass and no lint errors. No agent intervention needed.")
        # Still write a success comment
        comment = """## AgentForge Analysis

**Status:** All checks passed. No agent intervention needed.

| Check | Status |
|-------|--------|
| Tests | Passed |
| Lint  | Clean  |

The AI agent reviewed this PR and found no issues to fix.
"""
        (ci_dir / "pr_comment.md").write_text(comment)
        return

    # Generate task for the agent
    if test_results["failures"]:
        task_yaml = generate_task_yaml(
            task_name=f"Fix PR #{args.pr} test failures",
            description=f"Fix {len(test_results['failures'])} failing test(s) in PR #{args.pr}",
            changed_files=changed_files,
            test_output=test_results["stdout"][-3000:],
            diff=diff,
        )
        task_path = tasks_dir / f"pr_{args.pr}_tests.yaml"
        task_path.write_text(task_yaml)
        print(f"  Generated task: {task_path}")

    if lint_errors:
        task_yaml = generate_task_yaml(
            task_name=f"Fix PR #{args.pr} lint errors",
            description=f"Fix {len(lint_errors)} lint error(s) in PR #{args.pr}",
            changed_files=changed_files,
            test_output=json.dumps(lint_errors[:20], indent=2),
            diff=diff,
        )
        task_path = tasks_dir / f"pr_{args.pr}_lint.yaml"
        task_path.write_text(task_yaml)
        print(f"  Generated task: {task_path}")

    print("  Analysis complete. Ready for agent run.")


if __name__ == "__main__":
    main()
