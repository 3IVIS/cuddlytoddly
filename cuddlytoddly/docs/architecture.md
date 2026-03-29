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

**Skills are data-driven.** Drop a folder with a `SKILL.md` and optional `tools.py` into `cuddlytoddly/skills/` — the `SkillLoader` discovers and registers them automatically at startup with no code changes required.

## Data Flow

### Planning phase

1. `LLMPlanner.plan(goal_id)` reads the current snapshot and identifies unexpanded goal/task nodes.
2. It builds a prompt describing the node and asks the LLM for a list of child tasks in JSON format.
3. The JSON is validated by `LLMOutputValidator` and emitted as `ADD_NODE` / `ADD_DEPENDENCY` events.
4. `apply_event()` applies each event to the graph, then calls `recompute_readiness()`.

### Execution phase

1. `SimpleOrchestrator` polls for nodes whose `status == "ready"`.
2. For each ready node it calls `QualityGate.check_dependencies()` — if upstream results are insufficient a bridge node is injected automatically.
3. `LLMExecutor.run(node)` drives a multi-turn LLM loop: the LLM calls tools via JSON responses; each tool call is tracked as a child `execution_step` node by `ExecutionStepReporter`.
4. On success, `QualityGate.verify_result()` checks the result against the node's declared outputs. On failure the node is retried or failed.

### Persistence and replay

All events are appended to an `events.jsonl` file via `EventLog`. On restart, `rebuild_graph_from_log()` replays the log to reconstruct the exact graph state, then ephemeral `execution_step` nodes are pruned and in-flight nodes are reset to `pending`.

## Module Map

| Package | Responsibility |
|---|---|
| `cuddlytoddly.core` | `TaskGraph`, `Node`, `Event` types, `apply_event` reducer, ID generator |
| `cuddlytoddly.engine` | `SimpleOrchestrator`, `QualityGate`, `ExecutionStepReporter` |
| `cuddlytoddly.planning` | LLM client abstraction, `LLMPlanner`, `LLMExecutor`, output validator |
| `cuddlytoddly.infra` | Logging, `EventQueue`, `EventLog`, replay |
| `cuddlytoddly.skills` | `SkillLoader`, `ToolRegistry`, built-in skill packs |
| `cuddlytoddly.ui` | Curses terminal UI, Git DAG projection |

## Concurrency Model

The orchestrator runs in a background thread. Execution of individual nodes is dispatched to a `ThreadPoolExecutor` with `max_workers` workers (default 1 for llama.cpp, which is not thread-safe). All graph mutations are protected by `graph_lock`. The curses UI runs on the main thread and communicates with the orchestrator exclusively through the shared graph and `event_queue`.
