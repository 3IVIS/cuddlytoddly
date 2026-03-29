# API Reference

This page documents the public interfaces intended for programmatic use. Internal modules (reducer, event types, ID generator) are implementation details and may change between releases.

---

## `cuddlytoddly.planning.llm_interface`

### `create_llm_client(backend, **kwargs) → BaseLLM`

Factory for LLM clients. Returns a `BaseLLM` instance.

| `backend` value | Class returned | Required kwargs |
|---|---|---|
| `"claude"` | `ApiLLM` | `model`, optional `temperature`, `max_tokens` |
| `"openai"` | `ApiLLM` | `model`, optional `base_url`, `temperature`, `max_tokens` |
| `"llamacpp"` | `LlamaCppLLM` | `model_path`, optional `n_gpu_layers`, `n_ctx`, `max_tokens`, `temperature`, `cache_path` |

### `BaseLLM`

All backends implement:

```python
def ask(self, prompt: str, schema: dict | None = None) -> str:
    """Send a prompt; return raw JSON string. Pass schema for structured output."""

def stop(self) -> None:
    """Signal the LLM to stop mid-generation (used by the UI pause button)."""

def resume(self) -> None:
    """Resume after a stop()."""
```

---

## `cuddlytoddly.planning.llm_planner`

### `LLMPlanner(llm_client, graph, skills_summary="")`

Decomposes unexpanded nodes into child tasks.

```python
planner = LLMPlanner(llm_client=llm, graph=graph, skills_summary=skills.prompt_summary)
planner.plan(node_id)  # expands one node; emits events to graph
```

---

## `cuddlytoddly.planning.llm_executor`

### `LLMExecutor(llm_client, tool_registry, max_turns=5)`

Executes a single task node via multi-turn LLM + tool calls.

```python
executor = LLMExecutor(llm_client=llm, tool_registry=registry, max_turns=5)
result: str = executor.run(node, snapshot, reporter)
```

`reporter` is an `ExecutionStepReporter` instance; pass `None` to skip step tracking.

---

## `cuddlytoddly.engine.llm_orchestrator`

### `SimpleOrchestrator(graph, planner, executor, quality_gate, event_log, event_queue, max_workers)`

The top-level plan→execute loop.

```python
orchestrator = SimpleOrchestrator(
    graph=graph,
    planner=planner,
    executor=executor,
    quality_gate=quality_gate,
    event_log=event_log,
    event_queue=queue,
    max_workers=1,
)
orchestrator.start()   # runs in a background thread
orchestrator.stop()    # signals shutdown
```

**UI-facing attributes** (read by `curses_ui`):

| Attribute | Type | Description |
|---|---|---|
| `graph` | `TaskGraph` | The live graph |
| `graph_lock` | `threading.Lock` | Must be held when reading graph for display |
| `event_queue` | `EventQueue` | Queue for user-injected events |
| `current_activity` | `str` | Human-readable status string |
| `llm_stopped` | `bool` | True when LLM is paused |

---

## `cuddlytoddly.engine.quality_gate`

### `QualityGate(llm_client, tool_registry=None)`

LLM-powered result verification and dependency checking.

```python
gate = QualityGate(llm_client=llm, tool_registry=registry)

satisfied, reason = gate.verify_result(node, result_str, snapshot)
bridge = gate.check_dependencies(node, snapshot)  # returns None or {node_id, description, output}
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
| `node_type` | `str` | `goal`, `task`, `execution_step` |
| `dependencies` | `set[str]` | IDs of nodes this node depends on |
| `children` | `set[str]` | IDs of nodes that depend on this node |
| `result` | `str \| None` | Output of the node once done |
| `metadata` | `dict` | Arbitrary planner/executor annotations |

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
