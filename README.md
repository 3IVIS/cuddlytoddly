# cuddlytoddly

An LLM-driven autonomous planning and execution system built around a DAG (directed acyclic graph) of tasks. Give it a goal; it breaks the goal into tasks, executes them with tools, verifies results, and fills in gaps — continuously, with live terminal and web UIs.

Why "cuddlytoddly"?

Large language models are powerful at generating text and solving well-scoped problems, but they are widely recognized as weak at long-horizon, complex planning. Left alone, they often miss dependencies, overlook edge cases, or wander off track.

cuddlytoddly takes a different approach: instead of expecting the LLM to plan everything autonomously, we provide a structured framework where humans and the system collaborate. The LLM handles decomposition, execution, and verification, while the framework ensures tasks are organized, dependencies are respected, and gaps are bridged.

Think of it as holding the model’s hand while it learns to walk through complex goals — hence the name cuddlytoddly. It’s not about blind autonomy; it’s about guided, reliable progress.

## How it works

1. A plain-English **goal** is seeded into the graph.
2. The **LLMPlanner** decomposes it into a DAG of tasks with explicit dependencies.
3. The **SimpleOrchestrator** picks up ready nodes and hands them to the **LLMExecutor**.
4. The executor runs a multi-turn LLM loop, calling tools (code execution, file I/O, custom skills) until the task is done.
5. The **QualityGate** checks the result against declared outputs; if something is missing it injects a bridging task automatically.
6. Every mutation is written to an **event log** — crash and resume with no lost work.

```
goal → LLMPlanner → TaskGraph (DAG)
                        │
              SimpleOrchestrator
              ├── LLMExecutor + tools
              └── QualityGate (verify / bridge)
                        │
                   EventLog (JSONL) → replay on restart
```

## Installation

```bash
pip install cuddlytoddly                       
```

**Requirements:** Python 3.11+, `git` on your PATH (for the DAG visualiser).

## Quick start

```bash
export ANTHROPIC_API_KEY=sk-ant-...
cuddlytoddly "Write a market analysis for electric scooters"
```

Or pass no argument to use the startup screen with multiple options.
The UI opens automatically. The run data is stored locally and can be resumed later — the event log preserves all state.

## LLM backends

| Backend | Install | `create_llm_client` call |
|---|---|---|
| Anthropic Claude | included | `create_llm_client("claude", model="claude-3-5-sonnet-20241022")` |
| OpenAI / compatible | `[openai]` | `create_llm_client("openai", model="gpt-4o")` |
| Local llama.cpp | `[local]` | `create_llm_client("llamacpp", model_path="/path/to/model.gguf")` |

## Adding skills

Drop a folder with a `SKILL.md` (and optional `tools.py`) into `cuddlytoddly/skills/`. The `SkillLoader` discovers it automatically. See [docs/skills.md](docs/skills.md) for the full format.

## Documentation

- [Architecture](docs/architecture.md) — how the components fit together
- [Configuration](docs/configuration.md) — LLM backends, run directory, environment variables
- [Skills](docs/skills.md) — built-in skills and how to add custom ones
- [API Reference](docs/api.md) — public Python API

## Project structure

```
cuddlytoddly/
├── core/           # TaskGraph, events, reducer, ID generator
├── engine/         # SimpleOrchestrator, QualityGate, ExecutionStepReporter
├── infra/          # Logging, EventQueue, EventLog, replay
├── planning/       # LLM interface, LLMPlanner, LLMExecutor, output validator
├── skills/         # SkillLoader + built-in skill packs
│   ├── code_execution/
│   └── file_ops/
└── ui/             # Curses terminal UI, Git DAG projection
docs/
pyproject.toml
LICENSE
```

## Python API

```python
from cuddlytoddly.core.task_graph import TaskGraph
from cuddlytoddly.core.events import Event, ADD_NODE
from cuddlytoddly.core.reducer import apply_event
from cuddlytoddly.infra.event_queue import EventQueue
from cuddlytoddly.infra.event_log import EventLog
from cuddlytoddly.planning.llm_interface import create_llm_client
from cuddlytoddly.planning.llm_planner import LLMPlanner
from cuddlytoddly.planning.llm_executor import LLMExecutor
from cuddlytoddly.engine.quality_gate import QualityGate
from cuddlytoddly.engine.llm_orchestrator import SimpleOrchestrator
from cuddlytoddly.skills.skill_loader import SkillLoader

# LLM client — swap "claude" for "openai" or "llamacpp"
llm = create_llm_client("claude", model="claude-3-5-sonnet-20241022")

graph    = TaskGraph()
skills   = SkillLoader()
planner  = LLMPlanner(llm_client=llm, graph=graph, skills_summary=skills.prompt_summary)
executor = LLMExecutor(llm_client=llm, tool_registry=skills.registry)
gate     = QualityGate(llm_client=llm, tool_registry=skills.registry)

orchestrator = SimpleOrchestrator(
    graph=graph, planner=planner, executor=executor,
    quality_gate=gate, event_queue=EventQueue(),
)

# Seed a goal
apply_event(graph, Event(ADD_NODE, {
    "node_id": "my_goal",
    "node_type": "goal",
    "metadata": {"description": "Summarise the key risks of AGI", "expanded": False},
}))

orchestrator.start()
# orchestrator runs in the background — block however suits your use case
```

## License

MIT — see [LICENSE](LICENSE).
