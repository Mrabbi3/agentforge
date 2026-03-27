# Contributing to AgentForge

Thank you for your interest in contributing to AgentForge!

## Getting Started

1. Fork the repository
2. Clone your fork: `git clone https://github.com/<you>/agentforge.git`
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Create a branch: `git checkout -b feature/your-feature`

## Development Workflow

- Run tests: `pytest tests/ -v`
- Run linter: `ruff check agentforge`
- Run type checker: `mypy agentforge`
- Run quickstart: `python quickstart.py`

## Pull Requests

- Keep PRs focused on a single change
- Add tests for new functionality
- Ensure all tests pass before submitting
- Follow existing code style and conventions

## Code Style

- Use type hints for all function signatures
- Keep lines under 100 characters
- Use `StrEnum` for enumerations
- Sort imports with `ruff`

## Reporting Issues

Open an issue on GitHub with a clear description and reproduction steps.
