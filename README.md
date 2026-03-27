# AgentForge

AI agent harness with pluggable memory strategies and built-in evaluation for benchmarking coding agents.

## Features

- **Agentic Loop**: Plan → Act → Observe → Reflect cycle using Claude API
- **Pluggable Memory**: Sliding window, summarization, RAG, and hybrid strategies
- **Tool Suite**: Bash execution, file read/write, and code search
- **Evaluation**: Quantitative metrics and LLM-as-judge scoring
- **Multi-Agent**: Planner → Executor → Reviewer coordination pipeline
- **CLI**: Run tasks, benchmarks, and compare configurations

## Quick Start

```bash
pip install -e ".[dev]"
python quickstart.py
```

## Usage

```bash
# Run a single task
agentforge run --task tasks/coding/fix_fibonacci_bug.yaml

# Run full benchmark
agentforge benchmark --config configs/default.yaml

# Compare configurations
agentforge compare --configs configs/default.yaml configs/sliding_window.yaml
```

## Configuration

See `configs/default.yaml` for the full configuration schema.

## Testing

```bash
pytest tests/ -v
ruff check agentforge
mypy agentforge
```

## License

MIT — see [LICENSE](LICENSE) for details.
