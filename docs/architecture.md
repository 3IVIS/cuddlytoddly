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
  Orchestrator  ◄──── polls ready nodes ─────────────────────┘
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

**Caching is uniform across all backends.** Every backend (`LlamaCppLLM`, `ApiLLM`, `FileBasedLLM`) accepts a `cache_path` argument and uses the same `LlamaCppCache` implementation — a SHA-256-keyed JSON file. Cache key: `prompt` (no schema) or `prompt + "\x00" + schema_json` (with schema). A hit returns immediately without touching the model or API. Enabled by default via `cache_enabled = true` in the respective config section.

**Prompts and schemas are separated from logic.** All LLM prompt text lives in `planning/prompts.py` and all JSON schemas in `planning/schemas.py`. Implementation files import from these modules rather than embedding strings inline, so prompt engineering and schema tuning never require touching execution logic.

**No hardcoded parameters.** Every numeric limit — character budgets, turn counts, retry thresholds, polling intervals — is read from `config.toml` at startup and passed down through constructors. Changing behaviour requires only a config edit, not a code change.

**Skills are data-driven.** Drop a folder with a `SKILL.md` and optional `tools.py` into `cuddlytoddly/skills/` — the `SkillLoader` discovers and registers them automatically at startup with no code changes required.

**Planning is a pipeline, not a single call.** Every goal expansion involves up to four LLM calls and three deterministic stages before any event reaches the graph: a clarification generation call (Call 1), a decomposition call (Call 2), an optional scrutiny call (Call 3), structural validation (`LLMOutputValidator`), and deterministic constraint enforcement (`PlanConstraintChecker`). Each stage has a clearly scoped responsibility so failures in one don't cascade into another.

**Clarification is grounded in the goal text.** Call 1 instructs the LLM to first extract every concrete fact already stated in the goal (budget, size, constraints, locations, roles, etc.) and pre-fill those as known fields, before identifying genuinely missing context. The available tools are described so the LLM avoids surfacing questions for information that can be fetched at runtime. This means stated user intent flows into all downstream tasks as known inputs rather than being lost.

**Tasks always execute — with a broadened goal when inputs are missing.** The executor never blocks a node waiting for user input. Instead, before the tool loop starts, it checks whether all declared required inputs are available. If inputs are missing it asks the LLM to produce a broadened version of the task goal that is achievable with what is currently known. The task runs with the broadened goal, the missing fields are added to the clarification form so the user can optionally supply them, and when the user confirms the task re-executes with the original specific goal. The broadened description is stored in node metadata for UI visibility.

## Data Flow

### Planning phase

1. `LLMPlanner.propose(context)` reads the current snapshot and identifies unexpanded goal nodes.
2. **Call 1 — Clarification generation.** A dedicated LLM call via `build_clarification_prompt(goal_text, skills_summary)` generates the clarification node fields using a two-step process:
   - **Step 1 — Extract:** the LLM reads the goal text and pulls out every concrete fact already stated by the user (budget figures, size requirements, hard constraints, locations, roles, timelines, named technologies, etc.) and pre-fills those as fields with known values. Facts present in the goal are never marked `"unknown"`.
   - **Step 2 — Gap-fill:** after the extraction pass, the LLM identifies genuinely missing context that would most change what tasks are needed or how they should be done, and marks those fields `"unknown"` for optional user input. Fields for information the available tools can retrieve autonomously (market prices, public statistics, regulatory text) are not raised.

   A `clarification` node is emitted with 3–8 fields, immediately marked done, and wired as a dependency of all root tasks. On partial replans the existing clarification node is reused so user edits are preserved.
3. **Call 2 — Planning.** `build_planner_prompt()` is called with the clarification context embedded. The LLM produces a list of tasks constrained to `PLAN_SCHEMA`. Tasks declare `required_input` in two categories: **Category A** — named outputs consumed from upstream tasks (each requires a corresponding dependency), and **Category B** — user-specific context from the clarification node (company name, personal history, etc.) that only the user can supply and that cannot be retrieved by search. Category B items have no corresponding task dependency. The planner can extend the clarification node with `additional_clarification_fields` in its output.
4. **Call 3 — Optional scrutiny pass.** If `scrutinize_plan=True`, the draft is passed to `build_plan_scrutinizer_prompt()`. The scrutinizer evaluates goal coverage, task realism, output completeness, and missing steps, then returns a corrected draft. Skipped on partial replans.
5. The JSON is validated by `LLMOutputValidator`: structural checks (self-deps, unknown deps, duplicate nodes, metadata allowlist).
6. **Constraint checking** — `PlanConstraintChecker.check_and_repair()` runs five deterministic checks in order:
   - Duplicate `ADD_DEPENDENCY` edges are silently deduplicated.
   - Cycles in the proposed subgraph are detected via DFS and all cycle-member nodes are dropped.
   - `required_input` items that have no corresponding dependency AND are not covered by any clarification field (orphaned Category A items) are stripped.
   - Dependencies with no corresponding `required_input` are logged as warnings.
   - Ghost nodes (new nodes with no dependents) are resolved by a focused LLM call that selects the best dependent from a cycle-safe candidate list.
7. The surviving events are emitted as `ADD_NODE` / `ADD_DEPENDENCY` / `SET_RESULT` events.
8. `apply_event()` applies each event to the graph, then calls `recompute_readiness()`.

### Execution phase

1. `Orchestrator` picks up nodes whose status is `ready` and dispatches them to the `ThreadPoolExecutor`.
2. Before launching, `QualityGate.check_dependencies()` checks whether upstream results cover the node's declared `required_input`. If a gap is found, a bridge node is injected (up to `max_gap_fill_attempts` times).
3. **Pre-flight check.** Before the tool loop starts, `LLMExecutor._preflight_awaiting_input()` determines whether all required inputs are currently available. It runs in two phases:
   - **Phase 1 (deterministic):** Compares the node's `required_input` list against the keys present in the upstream clarification node. Any required input not present in any clarification field is a structural gap — the user cannot supply it because the form has no field for it.
   - **Phase 2 (LLM):** Asks the LLM via `build_awaiting_input_check_prompt()` and `AWAITING_INPUT_CHECK_SCHEMA` whether the task can proceed despite missing inputs. If blocked, the LLM also produces a `broadened_description` — a self-contained rephrasing of the task goal that can be executed immediately with what is currently known.
   - **Reuse:** If the set of missing inputs is identical to the previous execution, the stored `broadened_description` from node metadata is reused without a new LLM call.
4. `LLMExecutor.execute(node)` drives a multi-turn LLM loop using `prompts.build_executor_prompt()` and `EXECUTION_TURN_SCHEMA`. If a broadened description was produced in the pre-flight check it is used as the effective task goal; otherwise the original `description` is used. The LLM calls tools via JSON responses; each tool call is tracked as a child `execution_step` node by `ExecutionStepReporter`. When a node ran with a broadened description, `ExecutionStepReporter.pending_broadening` carries the signal to the orchestrator.
5. After execution, if the node ran with a broadened description, the orchestrator writes four metadata keys to the node (`broadened_description`, `broadened_for_missing`, `broadened_reason`, `broadened_output`) and patches the clarification node with any missing fields so the user is prompted to fill them in.
6. `QualityGate.verify_result()` checks the result against the node's declared outputs using `RESULT_VERIFICATION_SCHEMA`. The verifier receives three context blocks that inform its judgment: a list of unknown clarification fields (to catch invented specifics), a tool execution summary (to catch results fabricated from failed searches), and — when the node ran with a broadened goal — a notice that the result must be general in nature and must not contain invented personal data or specific figures that could only come from a targeted search. On failure the node is retried up to `max_retries` times before being permanently failed.

### Web UI — clarification and goal switching

The web server exposes `/api/node/{id}/clarification/confirm` for committing user edits to a clarification node. On confirm: the node's result is updated, its direct children are reset (triggering re-execution with the updated context), and the parent goal is marked unexpanded so the planner can add tasks on the next cycle if needed. Scrutiny is skipped on this partial replan. Nodes that previously ran with a broadened description will re-run their pre-flight check; if the newly filled fields satisfy what was missing, the node executes with the original specific goal.

The web server also exposes `/api/switch` for mid-session goal switching (available when started without an inline goal argument). On receiving a switch request the server stops the running orchestrator cleanly, resets all state, and initialises a new orchestrator in a background thread. The client polls `/api/status` and reloads when `initialized` becomes `true` — the same flow used for initial startup. After the new orchestrator is ready, the token counter is seeded from the run's cache file so the toolbar immediately reflects the correct historical total (see Persistence section below).

The web server also exposes `POST /api/export/html`, which generates a standalone HTML snapshot of the current DAG. The snapshot embeds the full node graph, the complete event log for replay, and the current token counts (`prompt`, `completion`, `total`, `calls`) — so the token pill in the static snapshot toolbar accurately reflects the run's lifetime usage at export time.

### Persistence and replay

All events are appended to an `events.jsonl` file via `EventLog`. On restart, `rebuild_graph_from_log()` replays the log to reconstruct the exact graph state, then ephemeral `execution_step` nodes are pruned and in-flight nodes are reset to `pending`.

**Token count restoration.** The module-level `token_counter` singleton starts at zero on every process start. When loading an existing run (`mode="existing"` at startup or via the switch modal), the startup code reads `llamacpp_cache.json` from the run directory and calls `token_counter.seed()` with the cumulative totals derived from the stored prompt/response pairs — using the same `len // 4` approximation that the live backends use. This ensures the token count displayed in the web UI toolbar reflects the full lifetime of the run, not just the current session. Runs that used API backends (Claude, OpenAI), which produce `api_cache.json` rather than `llamacpp_cache.json`, are skipped silently — their counter starts at zero for the current session.

## Module Map

| Package | Files | Responsibility |
|---|---|---|
| `agent_core.core` | `task_graph.py`, `events.py`, `reducer.py`, `id_generator.py` | `TaskGraph`, `Node`, `Event` types, `apply_event` reducer |
| `agent_core.engine` | `orchestrator.py`, `quality_gate.py`, `execution_step_reporter.py` | Plan→execute loop, result verification, bridge-node injection |
| `cuddlytoddly.planning` | `prompts.py` ★, `schemas.py` ★, `llm_interface.py`, `llm_planner.py`, `llm_executor.py`, `llm_output_validator.py`, `plan_constraint_checker.py` | LLM client abstraction, prompt templates, JSON schemas, planning pipeline and execution logic |
| `toddly.infra` | `logging.py`, `event_queue.py`, `event_log.py`, `replay.py` | Logging, `EventQueue`, `EventLog`, replay |
| `cuddlytoddly.skills` | `skill_loader.py`, `*/SKILL.md`, `*/tools.py` | `SkillLoader`, `ToolRegistry`, built-in skill packs |
| `cuddlytoddly.ui` | `curses_ui.py`, `web_server.py`, `git_projection.py` | Curses terminal UI, web UI (including goal switching and static HTML export), Git DAG projection |

★ These two files are the primary edit points for prompt engineering and schema tuning.

## Planning Pipeline Detail

The full journey from raw LLM output to committed graph events:

```
Goal text
       │
       ▼
LLMPlanner._generate_clarification_node()   ← Call 1: extract stated facts + identify missing context
       │                                       tool-aware: skills_summary prevents asking for
       │                                       runtime-fetchable info
       │                                       emits clarification node (immediately done)
       ▼
LLMPlanner._build_prompt()                  ← clarification fields embedded in planning prompt
       │
       ▼
LLM → raw JSON (PLAN_SCHEMA)                ← Call 2: decompose goal into tasks
       │
       ▼
LLMPlanner._normalize_events()              ← tolerant shape normalisation
       │                                       (fixes "operation" → "type", etc.)
       ▼
[optional] LLMPlanner._scrutinize()         ← Call 3: review draft for content/realism
       │                                       skipped on partial replans
       ▼
LLMOutputValidator.validate_and_normalize() ← structural invariants:
       │                                       self-deps, unknown deps, duplicate nodes,
       │                                       metadata allowlist, goal↔task dep rules
       ▼
PlanConstraintChecker.check_and_repair()    ← plan-level invariants:
       │                                       dedup edges, cycle removal, required_input
       │                                       consistency, ghost node resolution (LLM call)
       ▼
clarification events + safe_events list
       │
       ▼
apply_event() × N  →  TaskGraph
```

## Execution Pre-Flight Detail

Before the tool loop, every node goes through this check:

```
LLMExecutor.execute(node)
       │
       ▼
_preflight_awaiting_input()
       │
       ├── Phase 1: compare required_input against clarification field keys
       │     any required input absent from the form → auto_new_fields
       │
       ├── Reuse check: same missing-key set as stored broadened_for_missing?
       │     YES → return stored broadened_description (no LLM call)
       │
       └── Phase 2: LLM call (AWAITING_INPUT_CHECK_SCHEMA)
             blocked=false → use original description
             blocked=true  → broadened_description + missing/new fields
                                │
                                ▼
                         execute with broadened_description
                         reporter.on_broadened_execution(signal)
                                │
                                ▼
                         orchestrator writes metadata:
                           broadened_description
                           broadened_for_missing
                           broadened_reason
                           broadened_output
                         patches clarification node with new_fields
```

## Concurrency Model

The orchestrator runs in a background thread. Execution of individual nodes is dispatched to a `ThreadPoolExecutor` with `max_workers` workers (default 1 for llama.cpp, which is not thread-safe). All graph mutations are protected by `graph_lock`. The curses UI runs on the main thread and communicates with the orchestrator exclusively through the shared graph and `event_queue`.

The web server runs in a separate daemon thread (uvicorn). Goal switching stops the old orchestrator thread before starting the new one, ensuring only one orchestrator is live at any time.

When a user confirms clarification edits, the orchestrator's next planning cycle detects the unexpanded goal and runs a partial replan (Calls 1 and 2 only, no scrutiny). The `PlanningContext.skip_scrutiny` flag is set automatically when the planner detects that the goal already has children.

## Configuration Flow

All tuning parameters flow from `config.toml` through a single path:

```
config.toml
    │
    ▼
config.load_config()
    │
    ├── get_orchestrator_cfg() ──► Orchestrator(max_gap_fill_attempts, idle_sleep, ...)
    ├── get_executor_cfg()     ──► LLMExecutor(max_successful_turns, max_unsuccessful_turns, max_inline_result_chars, max_tool_result_chars, ...)
    ├── get_planner_cfg()      ──► LLMPlanner(min_tasks_per_goal, max_tasks_per_goal, scrutinize_plan)
    └── get_file_llm_cfg()     ──► FileBasedLLM(poll_interval, timeout, ...)
```

No component reads config directly — they receive values through their constructors, which makes them independently testable with any parameter set.