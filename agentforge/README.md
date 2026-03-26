# 🔥 AgentForge

**An open-source agent harness with pluggable memory strategies and built-in evaluation for benchmarking AI coding agents.**

AgentForge lets you build, configure, and rigorously benchmark different agent architectures against real coding and knowledge tasks — with quantitative metrics, model-based evaluation, and multi-agent coordination.

## Why AgentForge?

Modern AI agents need more than a good model — they need a good **harness**. The harness controls how the agent manages memory, selects tools, recovers from errors, and coordinates with other agents. AgentForge makes these architectural choices **pluggable and measurable**.

### Key Features

- **Pluggable Memory Strategies** — Swap between sliding window, summarization, RAG-backed, and hybrid memory with a single config change
- **Configurable Agent Loop** — Plan → Act → Observe → Reflect cycle with customizable strategies
- **Built-in Tool System** — Bash execution, file I/O, search, and extensible tool registry
- **Docker-Sandboxed Execution** — Every task runs in an isolated container for reproducibility
- **Quantitative Evaluation** — Pass rate, tool efficiency, cost tracking, context utilization, error classification
- **Model-Based Judging** — LLM-as-judge scoring for reasoning coherence, plan adherence, and safety
- **Multi-Agent Coordination** — Planner → Executor → Reviewer pipeline with message passing
- **Comparison Dashboard** — Visualize runs across different configurations with Next.js + Recharts

## Quick Start

```bash
# Clone the repo
git clone https://github.com/Mrabbi3/agentforge.git
cd agentforge

# Install in dev mode
pip install -e ".[dev]"

# Set your API key
export ANTHROPIC_API_KEY="your-key-here"

# Run a single task
agentforge run --task tasks/coding/fix_bug_001.yaml

# Run full benchmark suite
agentforge benchmark --config configs/default.yaml

# Compare two memory strategies
agentforge compare --configs configs/sliding_window.yaml configs/summarization.yaml
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  AgentForge                       │
├─────────────────────────────────────────────────┤
│  Agent Core                                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ Agent    │ │ Tool     │ │ Memory Manager   │ │
│  │ Loop     │ │ Router   │ │ (pluggable)      │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
├─────────────────────────────────────────────────┤
│  Harness Layer                                   │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ Task     │ │Trajectory│ │ Multi-Agent      │ │
│  │ Runner   │ │ Logger   │ │ Orchestrator     │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
├─────────────────────────────────────────────────┤
│  Evaluation Engine                               │
│  ┌──────────┐ ┌──────────┐ ┌──────────────────┐ │
│  │ Metrics  │ │ Model    │ │ Comparison       │ │
│  │Calculator│ │ Judge    │ │ Reports          │ │
│  └──────────┘ └──────────┘ └──────────────────┘ │
└─────────────────────────────────────────────────┘
```

## Memory Strategies

| Strategy | How It Works | Best For |
|----------|-------------|----------|
| `sliding_window` | Keeps last N turns, drops oldest | Short tasks, low cost |
| `summarization` | Summarizes old context via LLM | Medium tasks, balanced |
| `rag` | Embeds turns, retrieves relevant ones | Long tasks, precise recall |
| `hybrid` | Recent window + RAG for older context | Complex multi-step tasks |

## Evaluation Metrics

**Task Performance**: Pass rate, partial credit, error recovery rate

**Efficiency**: Tool call efficiency, cost per task, context utilization ratio

**Agent Quality** (model-based): Reasoning coherence, plan adherence, safety score

**Multi-Agent**: Coordination overhead, sub-task completion, conflict rate

## Project Structure

```
agentforge/
├── agentforge/
│   ├── core/           # Agent loop, schemas, configuration
│   ├── memory/         # Memory strategy implementations
│   ├── tools/          # Tool definitions and execution
│   ├── evaluation/     # Metrics, model-based judge, reports
│   ├── harness/        # Task runner, trajectory logging, sandbox
│   └── multi_agent/    # Multi-agent orchestration
├── tasks/              # Task definitions (coding + knowledge)
├── configs/            # Harness configuration files
├── tests/              # Unit and integration tests
├── dashboard/          # Next.js evaluation dashboard
└── docs/               # Documentation and guides
```

## Configuration

AgentForge uses YAML configs to define agent architecture:

```yaml
# configs/default.yaml
agent:
  model: claude-sonnet-4-20250514
  max_steps: 25
  temperature: 0.0

memory:
  strategy: summarization
  max_context_tokens: 90000
  compact_threshold: 0.8
  summary_model: claude-haiku-4-5-20251001

tools:
  enabled:
    - bash_execute
    - file_read
    - file_write
    - file_search
  bash_timeout: 30
  max_output_chars: 10000

evaluation:
  metrics:
    - pass_rate
    - tool_efficiency
    - cost_per_task
    - context_utilization
  judge:
    enabled: true
    model: claude-sonnet-4-20250514
    dimensions:
      - reasoning_coherence
      - plan_adherence
      - safety
```

## Contributing

Contributions are welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

MIT License — see [LICENSE](LICENSE) for details.

## Author

**MD Rabbi** — CS @ Stockton University | AI Engineer
- GitHub: [@Mrabbi3](https://github.com/Mrabbi3)
