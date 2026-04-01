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

All schemas from this module are also re-exported by `cuddlytoddly.planning.llm_interface` for backward compatibility.

---

## `cuddlytoddly.planning.prompts`

All LLM prompt templates are defined here as standalone functions. Edit this file to change what the LLM sees without touching any implementation code.

### Functions

```python
build_executor_prompt(
    *, node_id, description, retry_notice, extra_reminder,
    outputs_block, output_instruction, inputs_text,
    tools_text, history_text, max_inline_result_chars, prompt_version="v3"
) -> str
```

Assembles the full prompt for one executor turn. Called by `LLMExecutor._build_prompt()`.

```python
build_planner_prompt(
    *, pruned_view_json, goals_repr_json, existing_ids_note,
    skills_block, min_tasks=3, max_tasks=8
) -> str
```

Assembles the planner decomposition prompt. Called by `LLMPlanner._build_prompt()`.

```python
build_verify_result_prompt(
    *, node_id, description, outputs_text, result
) -> str
```

Prompt for `QualityGate.verify_result()`.

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

## `cuddlytoddly.planning.llm_interface`

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

### `FileBasedLLM(response_file, prompt_log_file, poll_interval, timeout, progress_log_interval, cache_path=None)`

Development/testing backend that reads prompts and responses from plain text files. All timing values default to the `[file_llm]` section of `config.toml`.

When `cache_path` is provided, a cache hit skips the poll loop entirely — useful for replaying runs without re-entering responses. Cache key is the raw prompt string.

### `ApiLLM(provider, model, temperature, max_tokens, system_prompt, cache_path=None)`

Remote API backend (OpenAI or Anthropic Claude). Constructed via `create_llm_client("claude", ...)` or `create_llm_client("openai", ...)`.

When `cache_path` is provided, a cache hit is served immediately without any network call. Cache key is `prompt` when `schema=None`, or `prompt + "\x00" + sorted_json(schema)` when a schema is passed — identical to how `LlamaCppLLM` keys its cache, so the approach is consistent across backends.

### `LlamaCppCache(cache_path)`

Persistent JSON cache shared by all three backends. Stores `{sha256(key): {prompt, response}}` pairs. Loaded into memory at construction; written atomically to disk on every new entry.

```python
from cuddlytoddly.planning.llm_interface import LlamaCppCache

cache = LlamaCppCache("my_cache.json")
cache.set(prompt, response)
cached = cache.get(prompt)   # None on miss
len(cache)                   # entry count
cache.clear()                # wipe all entries
```

All backend classes expose a `clear_cache()` convenience method that delegates here.

---

## `cuddlytoddly.planning.llm_planner`

### `LLMPlanner(llm_client, graph, skills_summary="", min_tasks_per_goal=3, max_tasks_per_goal=8)`

Decomposes unexpanded goal nodes into child tasks.

```python
planner = LLMPlanner(
    llm_client=llm,
    graph=graph,
    skills_summary=skills.prompt_summary,
    min_tasks_per_goal=3,   # from config [planner]
    max_tasks_per_goal=8,   # from config [planner]
)
events: list[dict] = planner.propose(context)
```

`min_tasks_per_goal` and `max_tasks_per_goal` are injected into the planner prompt to guide the LLM's decomposition granularity. They default to the `[planner]` values in `config.toml` when the system is started via the CLI.

---

## `cuddlytoddly.planning.llm_executor`

### `LLMExecutor(llm_client, tool_registry, max_turns, max_inline_result_chars, max_total_input_chars, max_tool_result_chars, max_history_entries)`

Executes a single task node via multi-turn LLM + tool calls.

```python
executor = LLMExecutor(
    llm_client=llm,
    tool_registry=registry,
    max_turns=5,                    # from config [orchestrator]
    max_inline_result_chars=3000,   # from config [executor]
    max_total_input_chars=3000,     # from config [executor]
    max_tool_result_chars=2000,     # from config [executor]
    max_history_entries=3,          # from config [executor]
)
result: str | None = executor.execute(node, snapshot, reporter)
```

All numeric parameters default to reasonable values when constructing programmatically. When started via the CLI they are read from `config.toml` by `__main__._init_system()`.

`reporter` is an `ExecutionStepReporter` instance; pass `None` to skip step tracking.

#### Character budget parameters

| Parameter | Effect |
|---|---|
| `max_inline_result_chars` | If a result would exceed this, the executor asks the LLM to write to a file instead |
| `max_total_input_chars` | Total chars across all upstream results included in the prompt; split evenly between dependencies |
| `max_tool_result_chars` | Tool call results are truncated to this length before being added to history |
| `max_history_entries` | Only the N most-recent tool calls are kept in the prompt |

---

## `cuddlytoddly.engine.llm_orchestrator`

### `SimpleOrchestrator(graph, planner, executor, event_log, event_queue, max_workers, quality_gate, max_gap_fill_attempts, idle_sleep)`

The top-level plan→execute loop.

```python
orchestrator = SimpleOrchestrator(
    graph=graph,
    planner=planner,
    executor=executor,
    event_log=event_log,
    event_queue=queue,
    max_workers=1,
    quality_gate=quality_gate,
    max_gap_fill_attempts=2,   # from config [orchestrator]
    idle_sleep=0.5,            # from config [orchestrator]
)
orchestrator.start()   # runs in a background thread
orchestrator.stop()    # signals shutdown
```

All arguments after `executor` are keyword arguments. Defaults: `event_log=None`, `event_queue=None`, `max_workers=4`, `quality_gate=None`, `max_gap_fill_attempts=2`, `idle_sleep=0.5`.

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

## `cuddlytoddly.engine.quality_gate`

### `QualityGate(llm_client, tool_registry=None)`

LLM-powered result verification and dependency checking. Prompt text is sourced from `planning/prompts.py`; schemas from `planning/schemas.py`.

```python
gate = QualityGate(llm_client=llm, tool_registry=registry)

satisfied, reason = gate.verify_result(node, result_str, snapshot)
bridge = gate.check_dependencies(node, snapshot)
# bridge is None or {node_id: str, description: str, output: str}
```

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

## `cuddlytoddly.core.task_graph`

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
| `node_type` | `str` | `goal`, `task`, `reflection`, `execution_step` |
| `dependencies` | `set[str]` | IDs of nodes this node depends on |
| `children` | `set[str]` | IDs of nodes that depend on this node |
| `result` | `str \| None` | Output of the node once done |
| `metadata` | `dict` | Arbitrary planner/executor annotations |

> **Node type notes:** `goal` and `task` are planner-created; `reflection` is emitted by the refiner pass; `execution_step` is an internal type created by `ExecutionStepReporter` to track individual tool calls within a task and is pruned on restart.

---

## `cuddlytoddly.infra`

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

```python
from cuddlytoddly.config import load_config, get_executor_cfg

cfg = load_config()
exec_cfg = get_executor_cfg(cfg)
# {"max_inline_result_chars": 3000, "max_total_input_chars": 3000, ...}
```
