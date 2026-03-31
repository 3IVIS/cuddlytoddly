# Architecture

## Overview

cuddlytoddly is a DAG-first autonomous planning system. A goal is given as a plain-English string; the system decomposes it into a directed acyclic graph (DAG) of tasks, executes them in dependency order, and iteratively refines the plan as results come in — all driven by an LLM.

```
User goal (string)
       │
       ▼
  LLMPlanner  ──── emits ADD_NODE / ADD_DEPENDENCY events ────►  TaskGraph
       │                                                              │
       │                                                    recompute_readiness()
       │                                                              │
  SimpleOrchestrator  ◄──── polls ready nodes ─────────────────────┘
       │
       ├── LLMExecutor  (runs one node via LLM + tools)
       │        │
       │        └── ExecutionStepReporter  (child nodes in DAG)
       │
       ├── QualityGate  (verifies result; may inject bridge nodes)
       │
       └── EventLog  (persists all mutations to JSONL for replay)
```

## Design Principles

**Event-sourced state.** The `TaskGraph` is never mutated directly. All changes go through `Event` objects processed by the `reducer`. This means the full history is replayable from the event log — if the process crashes, it picks up exactly where it left off.

**Read-only snapshots for planning.** The planner and orchestrator always work from `graph.get_snapshot()` (a deep copy), so they can reason about the graph without race conditions.

**LLM backends are interchangeable.** `planning/llm_interface.py` defines one `BaseLLM` with `.ask()` and `.generate()`. Swap between Anthropic Claude, OpenAI-compatible endpoints, and local llama.cpp models by changing one argument to `create_llm_client()`.

**Prompts and schemas are separated from logic.** All LLM prompt text lives in `planning/prompts.py` and all JSON schemas in `planning/schemas.py`. Implementation files import from these modules rather than embedding strings inline, so prompt engineering and schema tuning never require touching execution logic.

**No hardcoded parameters.** Every numeric limit — character budgets, turn counts, retry thresholds, polling intervals — is read from `config.toml` at startup and passed down through constructors. Changing behaviour requires only a config edit, not a code change.

**Skills are data-driven.** Drop a folder with a `SKILL.md` and optional `tools.py` into `cuddlytoddly/skills/` — the `SkillLoader` discovers and registers them automatically at startup with no code changes required.

## Data Flow

### Planning phase

1. `LLMPlanner.propose(context)` reads the current snapshot and identifies unexpanded goal nodes.
2. It builds a prompt via `prompts.build_planner_prompt()` describing the goal and asks the LLM for a list of child tasks, constrained to the `PLAN_SCHEMA`.
3. The JSON is validated by `LLMOutputValidator` and emitted as `ADD_NODE` / `ADD_DEPENDENCY` events.
4. `apply_event()` applies each event to the graph, then calls `recompute_readiness()`.

### Execution phase

1. `SimpleOrchestrator` picks up nodes whose status is `ready` and dispatches them to the `ThreadPoolExecutor`.
2. Before launching, `QualityGate.check_dependencies()` checks whether upstream results cover the node's declared `required_input`. If a gap is found, a bridge node is injected (up to `max_gap_fill_attempts` times).
3. `LLMExecutor.execute(node)` drives a multi-turn LLM loop using `prompts.build_executor_prompt()` and `EXECUTION_TURN_SCHEMA`. The LLM calls tools via JSON responses; each tool call is tracked as a child `execution_step` node by `ExecutionStepReporter`.
4. On success, `QualityGate.verify_result()` checks the result against the node's declared outputs using `RESULT_VERIFICATION_SCHEMA`. On failure the node is retried or failed.

### Persistence and replay

All events are appended to an `events.jsonl` file via `EventLog`. On restart, `rebuild_graph_from_log()` replays the log to reconstruct the exact graph state, then ephemeral `execution_step` nodes are pruned and in-flight nodes are reset to `pending`.

## Module Map

| Package | Files | Responsibility |
|---|---|---|
| `cuddlytoddly.core` | `task_graph.py`, `events.py`, `reducer.py`, `id_generator.py` | `TaskGraph`, `Node`, `Event` types, `apply_event` reducer |
| `cuddlytoddly.engine` | `llm_orchestrator.py`, `quality_gate.py`, `execution_step_reporter.py` | Plan→execute loop, result verification, bridge-node injection |
| `cuddlytoddly.planning` | `prompts.py` ★, `schemas.py` ★, `llm_interface.py`, `llm_planner.py`, `llm_executor.py`, `llm_output_validator.py` | LLM client abstraction, prompt templates, JSON schemas, planning and execution logic |
| `cuddlytoddly.infra` | `logging.py`, `event_queue.py`, `event_log.py`, `replay.py` | Logging, `EventQueue`, `EventLog`, replay |
| `cuddlytoddly.skills` | `skill_loader.py`, `*/SKILL.md`, `*/tools.py` | `SkillLoader`, `ToolRegistry`, built-in skill packs |
| `cuddlytoddly.ui` | `curses_ui.py`, `web_server.py`, `git_projection.py` | Curses terminal UI, web UI, Git DAG projection |

★ These two files are the primary edit points for prompt engineering and schema tuning.

## Concurrency Model

The orchestrator runs in a background thread. Execution of individual nodes is dispatched to a `ThreadPoolExecutor` with `max_workers` workers (default 1 for llama.cpp, which is not thread-safe). All graph mutations are protected by `graph_lock`. The curses UI runs on the main thread and communicates with the orchestrator exclusively through the shared graph and `event_queue`.

## Configuration Flow

All tuning parameters flow from `config.toml` through a single path:

```
config.toml
    │
    ▼
config.load_config()
    │
    ├── get_orchestrator_cfg() ──► SimpleOrchestrator(max_gap_fill_attempts, idle_sleep, ...)
    ├── get_executor_cfg()     ──► LLMExecutor(max_inline_result_chars, max_tool_result_chars, ...)
    ├── get_planner_cfg()      ──► LLMPlanner(min_tasks_per_goal, max_tasks_per_goal)
    └── get_file_llm_cfg()     ──► FileBasedLLM(poll_interval, timeout, ...)
```

No component reads config directly — they receive values through their constructors, which makes them independently testable with any parameter set.
