# API Reference

This page documents the public interfaces intended for programmatic use. Internal modules (reducer, event types, ID generator) are implementation details and may change between releases.

---

## `cuddlytoddly.planning.schemas`

All JSON schemas used for structured LLM output are defined here and imported by the rest of the codebase. Edit this file to change the output contract with the LLM.

| Name | Used by |
|---|---|
| `EVENT_LIST_SCHEMA` | Planner (raw event array format) |
| `PLAN_SCHEMA` | Planner (wraps `EVENT_LIST_SCHEMA` with `a_goal_result`) |
| `GOAL_SUMMARY_SCHEMA` | Goal summary generation |
| `REFINER_OUTPUT_SCHEMA` | Refiner pass |
| `EXECUTION_TURN_SCHEMA` | Executor (single tool/result turn) |
| `RESULT_VERIFICATION_SCHEMA` | QualityGate `verify_result` |
| `DEPENDENCY_CHECK_SCHEMA` | QualityGate `check_dependencies` |
| `GHOST_NODE_RESOLUTION_SCHEMA` | `PlanConstraintChecker` ghost node resolution |
| `CLARIFICATION_GENERATION_SCHEMA` | `LLMPlanner` clarification field generation (Call 1) |
| `AWAITING_INPUT_CHECK_SCHEMA` | Executor pre-flight check (determines whether to use broadened description) |
| `BROADENED_DESCRIPTION_SCHEMA` | Executor fallback call (generates broadened description when primary call returns empty) |

All schemas from this module are also re-exported by `agent_core.planning.llm_interface` for backward compatibility.

---

## `cuddlytoddly.planning.prompts`

All LLM prompt templates are defined here as standalone functions. Edit this file to change what the LLM sees without touching any implementation code.

### Functions

```python
build_executor_prompt(
    *, node_id, description, retry_notice, extra_reminder,
    outputs_block, output_instruction, inputs_text,
    tools_text, history_text, max_inline_result_chars,
    turns_remaining=0, 
) -> str
```

Assembles the full prompt for one executor turn. Called by `LLMExecutor._build_prompt()`. When the node is running with a broadened description, `description` is the broadened text rather than the original `node.metadata["description"]`. `turns_remaining` is injected as a synthesis reminder so the LLM knows when to stop searching and consolidate results.

```python
build_planner_prompt(
    *, pruned_view_json, goals_repr_json, existing_ids_note,
    skills_block, min_tasks=3, max_tasks=8, clarification_block=""
) -> str
```

Assembles the planner decomposition prompt. Called by `LLMPlanner._build_prompt()`. When a clarification node exists for the goal, `clarification_block` carries the structured field context into the prompt so the planner can produce a tailored plan.

```python
build_plan_scrutinizer_prompt(
    *, original_planning_prompt, draft_plan_json, min_tasks=3, max_tasks=8
) -> str
```

Assembles the scrutinizer prompt used to self-review a draft plan. The full original planning prompt is embedded verbatim so every constraint (DAG snapshot, existing node IDs, task-count limits, dependency semantics, format rules) remains in context during the second call. Called by `LLMPlanner._scrutinize()` when `scrutinize_plan=True`.

```python
build_clarification_prompt(goal_text: str, skills_summary: str = "") -> str
```

Generates the prompt for Call 1 — the dedicated LLM call that produces the clarification node fields before planning begins.

The prompt instructs the LLM to follow a two-step process:

- **Step 1 — Extract:** read the goal text and pull out every concrete fact already stated by the user (budget figures, size requirements, hard constraints, locations, roles, timelines, etc.) and pre-fill those as fields with known values. Facts explicitly present in the goal are never marked `"unknown"`.
- **Step 2 — Gap-fill:** identify genuinely missing context that would most change what tasks are needed or how they should be done, and add those as `"unknown"` fields for the user to optionally supply.

When `skills_summary` is provided (passed from `SkillLoader.prompt_summary`), the LLM is also told which tools will be available at execution time and instructed not to raise clarification fields for information those tools can retrieve autonomously (e.g. current market prices, public statistics, regulatory text).

Returns 3–8 structured fields (key, label, value, rationale). The prompt string is stored in the clarification node's metadata so the planner can reference it when adding fields later.

```python
build_clarification_context_block(fields: list, clarification_prompt: str) -> str
```

Formats clarification fields and the original clarification prompt into a block injected into `build_planner_prompt()`. The original prompt is embedded so the planner understands what was asked and can add fields via `additional_clarification_fields` without repeating questions.

```python
build_ghost_node_resolution_prompt(
    *, ghost_node_id, ghost_description, new_nodes, existing_nodes,
    active_goal_id, edges, valid_candidates
) -> str
```

Assembles the prompt used to resolve a ghost node — a new plan node whose output is consumed by nothing. The LLM is shown the full plan context and a pre-filtered candidate list (ancestors excluded to prevent cycles) and asked to choose the best dependent. Called by `PlanConstraintChecker._resolve_ghost_nodes()`.

```python
build_awaiting_input_check_prompt(
    *, node_id, description, tools_text,
    known_fields_text, unknown_fields_text,
    required_input_text="  (none declared)", previous_failure=""
) -> str
```

Prompt for the executor pre-flight check. The LLM decides whether the task can proceed with currently available information and tools. When inputs are missing it also produces a `broadened_description` — a self-contained rephrasing of the task goal that can run immediately using only what is known. When `previous_failure` is set (non-empty), the prompt includes the verifier's rejection reason from the last execution so the LLM can generate a broadened description that avoids repeating the same failure. The result is parsed against `AWAITING_INPUT_CHECK_SCHEMA`.

Key schema fields returned:

| Field | Type | Meaning |
|---|---|---|
| `blocked` | `bool` | Whether specific inputs are missing |
| `reason` | `str` | One sentence explaining what is missing or confirming the task can proceed |
| `missing_fields` | `list[str]` | Keys of existing unknown clarification fields that would unblock the specific goal |
| `new_fields` | `list[dict]` | New clarification fields to add to the form |
| `broadened_description` | `str` | Self-contained task description that works without the missing inputs (required when `blocked=true`) |
| `broadened_for_missing` | `list[str]` | The missing field keys active when this broadening was generated — used to decide reuse vs regenerate |
| `broadened_output` | `list[dict]` | Revised output declarations matching the broadened description — same `{name, type, description}` shape as planner outputs (required when `blocked=true`) |

```python
build_broadened_description_prompt(
    *, node_id, original_description, missing_keys, known_fields_text
) -> str
```

Focused fallback prompt used when `build_awaiting_input_check_prompt` returns `blocked=true` but an empty `broadened_description`. Makes a second, minimal LLM call against `BROADENED_DESCRIPTION_SCHEMA` to generate only the broadened description. If this call also returns empty, execution is skipped entirely — the executor never falls back to the original description for a task flagged as missing inputs.

```python
build_verify_result_prompt(
    *, node_id, description, outputs_text, result,
    unknown_fields_context="", tool_results_context="",
    broadening_context=""
) -> str
```

Prompt for `QualityGate.verify_result()`. Three optional context blocks inform the verifier's judgment:

- `unknown_fields_context` — lists clarification fields that were unknown when the task ran; the verifier flags invented specifics for those fields.
- `tool_results_context` — factual summary of tool call outcomes (all succeeded / partial / all failed); the verifier flags results containing specifics that could only have come from a successful search when searches failed.
- `broadening_context` — present when the node ran with a broadened description instead of its original goal; the verifier flags results containing invented personal data or specific figures, since the broadened goal should produce only general content.

```python
build_check_dependencies_prompt(
    *, node_id, description, inputs_text, upstream_text
) -> str
```

Prompt for `QualityGate.check_dependencies()`.

### Constants

```python
LLM_SYSTEM_PROMPT: str      # system role message for OpenAI / Claude API calls
LLAMACPP_SYSTEM_PROMPT: str # system content for the llama.cpp chat template
```

---

## `agent_core.planning.llm_interface`

### `create_llm_client(backend, **kwargs) → BaseLLM`

Factory for LLM clients. Returns a `BaseLLM` instance.

| `backend` value | Class returned | Required kwargs |
|---|---|---|
| `"claude"` | `ApiLLM` | `model`, optional `temperature`, `max_tokens`, `system_prompt`, `cache_path` |
| `"openai"` | `ApiLLM` | `model`, optional `base_url`, `temperature`, `max_tokens`, `system_prompt`, `cache_path` |
| `"llamacpp"` | `LlamaCppLLM` | `model_path`, optional `n_gpu_layers`, `n_ctx`, `max_tokens`, `temperature`, `cache_path` |
| `"file"` | `FileBasedLLM` | optional `poll_interval`, `timeout`, `progress_log_interval`, `cache_path` |

All backends accept an optional `cache_path` kwarg. When provided, a `LlamaCppCache` instance is attached: hits are served immediately without calling the model or API; misses are stored after a successful response. Pass `None` (or omit) to disable caching.

### `BaseLLM`

All backends implement:

```python
def ask(self, prompt: str) -> str:
    """Send a prompt; return raw text. Concrete subclasses also accept
    an optional schema keyword argument for structured output enforcement."""

def stop(self) -> None:
    """Set the stop flag — subsequent ask() calls raise LLMStoppedError."""

def resume(self) -> None:
    """Clear the stop flag — ask() calls proceed normally again."""

def generate(self, prompt: str) -> str:
    """Alias for ask() — kept for backward compatibility."""
```

> **Note:** The `schema` keyword argument for structured output is supported by `ApiLLM` and `LlamaCppLLM`, but is not part of the `BaseLLM` abstract interface.

### `TokenCounter`

Module-level singleton (`token_counter`) that tracks tokens consumed across all LLM calls in the current process. Imported from `agent_core.planning.llm_interface`.

```python
from agent_core.planning.llm_interface import token_counter

token_counter.prompt_tokens     # int — cumulative prompt tokens
token_counter.completion_tokens # int — cumulative completion tokens
token_counter.total_tokens      # int — prompt + completion
token_counter.calls             # int — total LLM calls made

token_counter.add(prompt: int, completion: int)
# Increments counters; called automatically by all backends after each inference.

token_counter.seed(prompt: int, completion: int, calls: int = 0)
# Sets the counter to a specific baseline. Used at startup when loading an
# existing run to restore the historical totals from the prior session.
# Should be called before any new LLM call is made.

token_counter.reset()
# Zeroes all counters.
```

When the system starts with an existing run directory (`mode="existing"`), the startup code reads `llamacpp_cache.json` from the run directory and calls `token_counter.seed()` with the totals derived from the cached prompt/response pairs. This ensures the token count displayed in the web UI toolbar reflects the full history of the run, not just the current process session. Runs using API backends (Claude, OpenAI) whose cache file is absent are skipped silently — the counter starts at zero for those sessions.

### `FileBasedLLM(response_file, prompt_log_file, poll_interval, timeout, progress_log_interval, cache_path=None)`

Development/testing backend that reads prompts and responses from plain text files. All timing values default to the `[file_llm]` section of `config.toml`.

When `cache_path` is provided, a cache hit skips the poll loop entirely — useful for replaying runs without re-entering responses. Cache key is the raw prompt string.

### `ApiLLM(provider, model, temperature, max_tokens, system_prompt, cache_path=None)`

Remote API backend (OpenAI or Anthropic Claude). Constructed via `create_llm_client("claude", ...)` or `create_llm_client("openai", ...)`.

When `cache_path` is provided, a cache hit is served immediately without any network call. Cache key is `prompt` when `schema=None`, or `prompt + "\x00" + sorted_json(schema)` when a schema is passed — identical to how `LlamaCppLLM` keys its cache, so the approach is consistent across backends.

### `LlamaCppCache(cache_path)`

Persistent JSON cache shared by all three backends. Stores `{sha256(key): {prompt, response}}` pairs. Loaded into memory at construction; written atomically to disk on every new entry.

```python
from agent_core.planning.llm_interface import LlamaCppCache

cache = LlamaCppCache("my_cache.json")
cache.set(prompt, response)
cached = cache.get(prompt)   # None on miss
len(cache)                   # entry count
cache.clear()                # wipe all entries
```

All backend classes expose a `clear_cache()` convenience method that delegates here.

---

## `cuddlytoddly.planning.llm_planner`

### `LLMPlanner(llm_client, graph, refiner=None, skills_summary="", min_tasks_per_goal=3, max_tasks_per_goal=8, scrutinize_plan=False)`

Decomposes unexpanded goal nodes into child tasks.

```python
planner = LLMPlanner(
    llm_client=llm,
    graph=graph,
    refiner=None,               # optional refiner component
    skills_summary=skills.prompt_summary,
    min_tasks_per_goal=3,     # from config [planner]
    max_tasks_per_goal=8,     # from config [planner]
    scrutinize_plan=False,    # from config [planner]
)
events: list[dict] = planner.propose(context)
```

`min_tasks_per_goal` and `max_tasks_per_goal` are injected into the planner prompt to guide the LLM's decomposition granularity. They default to the `[planner]` values in `config.toml` when the system is started via the CLI.

**Planning involves up to four LLM calls per goal:**

**Call 1 — Clarification generation.** Before decomposing the goal, the planner fires a dedicated call using `build_clarification_prompt(goal_text, skills_summary=self.skills_summary)`. The prompt instructs the LLM to follow a two-step process: first extract every concrete fact already stated in the goal text (budget, size, hard constraints, locations, roles, etc.) and pre-fill those as known field values; then identify genuinely missing context that would most affect the plan and mark those fields `"unknown"`. The `skills_summary` is included so the LLM knows which tools are available at execution time and avoids asking the user for information those tools can retrieve autonomously (market data, public statistics, regulatory text, etc.). The result is emitted as a `clarification` node that is immediately marked done and wired as a dependency of all root tasks. On partial replans (when the goal already has children), the existing clarification node is reused so user edits are not overwritten.

**Call 2 — Planning.** `build_planner_prompt()` is called with the clarification context embedded. Tasks declare `required_input` in two categories: **Category A** items name outputs produced by upstream tasks (each requires a corresponding dependency edge); **Category B** items name user-specific context only the user can supply (company name, personal history, etc.) and do not require a corresponding dependency — they are satisfied by the clarification node. The executor's pre-flight check handles Category B items at runtime by adding missing fields to the clarification form automatically. The planner can add fields to the clarification node via `additional_clarification_fields` in its output.

**Call 3 — Scrutiny (optional).** When `scrutinize_plan=True`, the draft is reviewed by `build_plan_scrutinizer_prompt()` for goal coverage, task realism, output completeness, and missing steps. Skipped on partial replans.

After structural validation by `LLMOutputValidator`, the plan passes through `PlanConstraintChecker` before the final events are returned to the orchestrator.

---

## `cuddlytoddly.planning.plan_constraint_checker`

### `PlanConstraintChecker(graph, llm_client)`

Post-validator constraint checker that enforces plan-level invariants on the `safe_events` list produced by `LLMOutputValidator`. Constructed automatically by `LLMPlanner` and called at the end of every `propose()` call.

```python
checker = PlanConstraintChecker(graph=graph, llm_client=llm)
repaired_events = checker.check_and_repair(safe_events, active_goal_id)
```

Checks are applied in this order, as earlier repairs affect what later checks see:

| # | Check | Action on violation |
|---|---|---|
| 7 | Duplicate `ADD_DEPENDENCY` edges | Silent deduplication |
| 4 | Cycles in the proposed subgraph | Drop all cycle-member nodes and incident edges; log each dropped node |
| 6b | `required_input` items with no dependency AND no matching clarification field | Strip orphaned items in-place; warn. Note: Category B items (user context) intentionally have no dependency and are NOT stripped if covered by a clarification field key |
| 6a | Dependencies with no corresponding `required_input` | Warn only — may be a valid ordering constraint |
| Ghost | New node with no dependents (nothing depends on it) | LLM call to select the best dependent; emit `ADD_DEPENDENCY` if valid candidate returned |

**Ghost node resolution** fires one focused LLM call per ghost node using `build_ghost_node_resolution_prompt()` and `GHOST_NODE_RESOLUTION_SCHEMA`. The candidate list passed to the LLM is pre-filtered to exclude all ancestors of the ghost node, preventing the LLM from suggesting a dependent that would introduce a cycle. If the LLM returns an invalid candidate or the call fails, the ghost node is left unconnected (logged as a warning) — the plan proceeds and the orchestrator will re-plan on the next cycle.

---

## `cuddlytoddly.planning.llm_executor`

### `LLMExecutor(llm_client, tool_registry, max_successful_turns, max_unsuccessful_turns, max_inline_result_chars, max_total_input_chars, max_tool_result_chars, max_history_entries)`

Executes a single task node via multi-turn LLM + tool calls.

```python
executor = LLMExecutor(
    llm_client=llm,
    tool_registry=registry,
    max_successful_turns=10,        # from config [llamacpp] / [claude] / [openai]
    max_unsuccessful_turns=10,      # from config [llamacpp] / [claude] / [openai]
    max_inline_result_chars=3000,   # from config [llamacpp] / [claude] / [openai]
    max_total_input_chars=3000,     # from config [llamacpp] / [claude] / [openai]
    max_tool_result_chars=2000,     # from config [llamacpp] / [claude] / [openai]
    max_history_entries=3,          # from config [llamacpp] / [claude] / [openai]
)
result: str | None = executor.execute(node, snapshot, reporter)
```

All numeric parameters default to reasonable values when constructing programmatically. When started via the CLI they are read from the active backend's section of `config.toml` by `__main__._init_system()`.

`max_successful_turns` caps turns where a tool call returned a result without error; `max_unsuccessful_turns` caps turns where a tool call errored or returned no results. Both counters run independently. The final turn of their combined budget is always reserved — on that turn the model receives no tools and a `FINAL TURN` notice, ensuring it always has a chance to return its result. Old configs that set `max_turns` (deprecated) set both to the same value.

`reporter` is an `ExecutionStepReporter` instance; pass `None` to skip step tracking.

#### Two-tier execution model

Before the tool loop starts, `execute()` calls `_preflight_awaiting_input()` to check whether all required inputs are available. The result determines which description is used:

| Situation | Effective description used |
|---|---|
| All required inputs available | Original `node.metadata["description"]` |
| Inputs missing, broadened description cached from previous run with same missing set | Stored `node.metadata["broadened_description"]` (no extra LLM call) |
| Inputs missing, missing set changed or first run | Freshly generated `broadened_description` from LLM pre-flight call |
| Pre-flight LLM returns empty `broadened_description` | Second focused LLM call via `build_broadened_description_prompt()` |
| Both LLM calls return empty | `execute()` returns `None` — node is skipped, not run with original description |

When the node ran with a broadened description, `ExecutionStepReporter.pending_broadening` carries the `AwaitingInputSignal` to `_on_node_done`, which writes three metadata keys to the node and patches the clarification form.

#### Character budget parameters

| Parameter | Effect |
|---|---|
| `max_inline_result_chars` | If a result would exceed this, the executor asks the LLM to write to a file instead |
| `max_total_input_chars` | Total chars across all upstream results included in the prompt; split evenly between dependencies |
| `max_tool_result_chars` | Tool call results are truncated to this length before being added to history |
| `max_history_entries` | Only the N most-recent tool calls are kept in the prompt |

#### Broadening metadata written to nodes

When a node runs with a broadened description, these keys are written to `node.metadata` after execution:

| Key | Type | Description |
|---|---|---|
| `broadened_description` | `str` | The broadened task goal that was used |
| `broadened_for_missing` | `list[str]` | The missing field keys that were absent when this broadening was generated |
| `broadened_reason` | `str` | One-sentence reason from the pre-flight LLM |

These keys are visible in both the web UI and curses UI as a "Running as (broadened goal)" indicator on the node detail panel.

---

## `agent_core.engine.orchestrator`

### `Orchestrator(graph, planner, executor, event_log, event_queue, max_workers, quality_gate, max_gap_fill_attempts, idle_sleep, max_retries)`

The top-level plan→execute loop.

```python
orchestrator = Orchestrator(
    graph=graph,
    planner=planner,
    executor=executor,
    event_log=event_log,
    event_queue=queue,
    max_workers=1,
    quality_gate=quality_gate,
    max_gap_fill_attempts=2,   # from config [orchestrator]
    idle_sleep=0.5,            # from config [orchestrator]
    max_retries=5,             # from config [orchestrator]

)
orchestrator.start()   # runs in a background thread
orchestrator.stop()    # signals shutdown
```

All arguments after `executor` are keyword arguments. Defaults: `event_log=None`, `event_queue=None`, `max_workers=4`, `quality_gate=None`, `max_gap_fill_attempts=2`, `idle_sleep=0.5`, `max_retries=5`.

**UI-facing attributes** (read by `curses_ui`):

| Attribute | Type | Description |
|---|---|---|
| `graph` | `TaskGraph` | The live graph |
| `graph_lock` | `threading.RLock` | Must be held when reading the graph for display |
| `event_queue` | `EventQueue` | Queue for user-injected events |
| `current_activity` | `str \| None` | Human-readable status string; `None` when idle |
| `llm_stopped` | `bool` | True when the LLM is paused |
| `token_counts` | `dict` | Running totals: `prompt`, `completion`, `total`, `calls` |


---

## Web API — clarification node

### `POST /api/node/{node_id}/clarification/confirm`

Commits user edits to a clarification node and triggers a partial replan.

```
POST /api/node/clarification_my_goal/clarification/confirm
Content-Type: application/json

{
  "updated_fields": [
    {"key": "current_salary", "label": "Current salary", "value": "$95,000", "rationale": "..."},
    {"key": "years_in_role",  "label": "Years in role",  "value": "3",       "rationale": "..."}
  ]
}
```

On success: updates the clarification node's result, resets its direct children (root tasks) so they re-execute with the updated context, and marks the parent goal `expanded=False` so the orchestrator's next planning cycle can add tasks if the updated context warrants it. The cascade through the rest of the DAG follows naturally from the child resets.

Nodes that previously ran with a broadened description will re-run their pre-flight check on the next execution cycle. If the newly filled clarification fields satisfy what was missing, those nodes will execute with their original specific goal rather than the broadened fallback.

Returns `{"ok": true}`.

---

## Web API — static HTML export

### `POST /api/export/html`

Generates a standalone, self-contained HTML snapshot of the current DAG and writes it to `<run_dir>/outputs/dag_snapshot_<timestamp>.html`.

The snapshot embeds:
- The full node graph (`SNAPSHOT_DATA_PLACEHOLDER`)
- Export metadata including goal title, timestamp, and **cumulative token counts** (`EXPORT_META_PLACEHOLDER`) — `{goal, timestamp, tokens: {prompt, completion, total, calls}}`
- The complete event log for replay (`REPLAY_EVENTS_PLACEHOLDER`)

The token counts embedded at export time reflect all LLM calls made across the lifetime of the run (including calls from prior sessions, thanks to startup seeding from the cache). The static HTML toolbar displays these counts in a token pill identical to the live UI.

Returns `{"ok": true, "path": "<absolute path to written file>"}`.

---

## `agent_core.engine.quality_gate`

### `QualityGate(llm_client, tool_registry=None)`

LLM-powered result verification and dependency checking. Prompt text is sourced from `planning/prompts.py`; schemas from `planning/schemas.py`.

```python
gate = QualityGate(llm_client=llm, tool_registry=registry)

satisfied, reason = gate.verify_result(node, result_str, snapshot)
bridge = gate.check_dependencies(node, snapshot)
# bridge is None or {node_id: str, description: str, output: str}
```

`verify_result` passes three context blocks to the verifier prompt, assembled from node metadata and the snapshot:

- **Unknown fields context** — clarification fields that were unknown when the task ran. The verifier flags any specific invented values (exact figures, names) for those fields.
- **Tool results context** — whether the node's tool calls succeeded, partially failed, or all failed. The verifier flags results asserting specific data that could only have come from a successful search when all searches returned errors.
- **Broadening context** — present when `node.metadata["broadened_description"]` is non-empty, i.e. the node ran with a generalised goal instead of its original description. The verifier flags results containing invented specifics (percentages, named achievements, exact figures) since broadened execution should produce general content only.

For nodes with declared file-type outputs, `verify_result` checks disk existence before calling the LLM verifier — if a declared file output does not exist on disk, the result fails immediately without an LLM call.

---

## `cuddlytoddly.skills.skill_loader`

### `SkillLoader(skills_dir=None)`

Discovers and loads skills from `cuddlytoddly/skills/` (or a custom path).

```python
skills = SkillLoader()
registry: ToolRegistry = skills.registry       # all registered tools
summary: str = skills.prompt_summary           # text to inject into planner prompt
skills.merge_mcp(other_registry)               # merge an external ToolRegistry
```

### `ToolRegistry`

```python
registry.register(tool: Tool)
result: str = registry.execute(tool_name: str, input_data: dict)
```

---

## `agent_core.core.task_graph`

### `TaskGraph`

```python
graph = TaskGraph()
graph.add_node(node_id, node_type="task", dependencies=[], metadata={})
graph.remove_node(node_id)
graph.add_dependency(node_id, depends_on)
graph.get_snapshot() -> dict[str, Node]        # deep copy; safe for concurrent reads
graph.get_ready_nodes() -> list[Node]
graph.recompute_readiness()
```

### `TaskGraph.Node`

| Attribute | Type | Description |
|---|---|---|
| `id` | `str` | Unique node identifier |
| `status` | `str` | `pending` / `ready` / `running` / `done` / `failed` |
| `node_type` | `str` | `goal`, `task`, `reflection`, `execution_step`, `clarification` |
| `dependencies` | `set[str]` | IDs of nodes this node depends on |
| `children` | `set[str]` | IDs of nodes that depend on this node |
| `result` | `str \| None` | Output of the node once done |
| `metadata` | `dict` | Arbitrary planner/executor annotations |

> **Node type notes:** `goal` and `task` are planner-created; `clarification` is emitted once per goal before the first task is created (ID: `clarification_{goal_id}`) and holds structured context fields the user can edit; `reflection` is emitted by the refiner pass; `execution_step` is an internal type created by `ExecutionStepReporter` to track individual tool calls within a task and is pruned on restart.

**Standard metadata keys written by the executor:**

| Key | Written when | Description |
|---|---|---|
| `broadened_description` | Node ran with a broadened goal | The broadened task description used for execution |
| `broadened_for_missing` | Node ran with a broadened goal | Missing field keys active when the broadening was generated |
| `broadened_reason` | Node ran with a broadened goal | One-sentence reason from the pre-flight LLM |
| `broadened_output` | Node ran with a broadened goal | Revised output declarations matching the broadened description |
| `verification_failure` | Last verification failed | The verifier's rejection reason |
| `retry_count` | Node has been retried | Number of verification-failed retries so far |

---

## `toddly.infra`

### `EventLog(path: str)`

```python
log = EventLog("events.jsonl")
log.append(event)
log.read_all() -> list[Event]
```

### `EventQueue`

Thread-safe queue wrapping `queue.Queue`.

```python
q = EventQueue()
q.put(event)
event = q.get(timeout=1.0)
```

### `rebuild_graph_from_log(event_log) → TaskGraph`

Replays all events from an `EventLog` and returns the reconstructed graph.

### `setup_logging(log_dir=None)` / `get_logger(name) → Logger`

Call `setup_logging()` once at startup. Use `get_logger(__name__)` in every module.

---

## `cuddlytoddly.config`

### `load_config() → dict`

Loads `config.toml`, creating it with auto-detected defaults on first run.

### `get_executor_cfg(cfg) → dict`
### `get_planner_cfg(cfg) → dict`
### `get_orchestrator_cfg(cfg) → dict`
### `get_file_llm_cfg(cfg) → dict`

Convenience accessors that return the named config section as a flat dict with defaults filled in. Old `config.toml` files that predate a section will work without requiring a manual edit.

`get_planner_cfg` returns `min_tasks_per_goal`, `max_tasks_per_goal`, and `scrutinize_plan`.

`get_orchestrator_cfg` returns `max_workers`, `max_successful_turns`, `max_unsuccessful_turns`, `max_gap_fill_attempts`, `idle_sleep`, and `max_retries`. The deprecated `max_turns` key is accepted as a fallback that sets both turn budgets to the same value.

```python
from cuddlytoddly.config import load_config, get_planner_cfg

cfg = load_config()
planner_cfg = get_planner_cfg(cfg)
# {"min_tasks_per_goal": 3, "max_tasks_per_goal": 8, "scrutinize_plan": False}
```