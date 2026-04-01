# Configuration

cuddlytoddly is configured through a single TOML file in the user data directory.
No environment variables are required for basic operation, and no values are hardcoded
inside the package — every numeric limit and behavioural parameter lives in `config.toml`
and can be changed without editing source code.

## Config file location

On first run the file is created automatically with sensible defaults.

| OS | Path |
|---|---|
| Linux | `~/.local/share/cuddlytoddly/config.toml` |
| macOS | `~/Library/Application Support/cuddlytoddly/config.toml` |
| Windows | `%LOCALAPPDATA%\3IVIS\cuddlytoddly\config.toml` |

Print the exact path on your machine:

```bash
python -c "from cuddlytoddly.config import CONFIG_PATH; print(CONFIG_PATH)"
```

---

## Full reference

```toml
# ── LLM backend ───────────────────────────────────────────────────────────────
[llm]

# Which backend to use: "llamacpp" (local), "claude", or "openai"
backend = "llamacpp"

# ── Local model (llama.cpp) ───────────────────────────────────────────────────
[llamacpp]

# GGUF model filename. Searched in this order:
#   1. CUDDLYTODDLY_MODEL_PATH env var  (full path — overrides everything)
#   2. LLAMA_CACHE / ~/.cache/llama.cpp/      (llama-cli / llama-server -hf)
#   3. HF_HOME / ~/.cache/huggingface/hub/    (huggingface-cli download)
#   4. <data_dir>/models/<model_filename>     (cuddlytoddly's own folder)
model_filename = "Llama-3.3-70B-Instruct-Q4_K_M.gguf"

# GPU layer offload: -1 = all layers on GPU, 0 = CPU only
n_gpu_layers = -1

# Context window size in tokens
n_ctx = 16384

# Maximum tokens to generate per response
max_tokens = 8192

# Sampling temperature (lower = more deterministic)
temperature = 0.1

# Cache LLM responses to disk; speeds up resumed runs
cache_enabled = true

# ── Anthropic Claude (API) ────────────────────────────────────────────────────
[claude]

# Requires the ANTHROPIC_API_KEY environment variable.
model         = "claude-opus-4-6"
temperature   = 0.1
max_tokens    = 8192

# Cache API responses to disk; avoids re-sending identical prompts.
# Cache file: <run_dir>/api_cache.json
cache_enabled = true

# ── OpenAI-compatible API ─────────────────────────────────────────────────────
[openai]

# API key can be set here or via the OPENAI_API_KEY environment variable.
model         = "gpt-4o"
temperature   = 0.1
max_tokens    = 8192

# Cache API responses to disk; avoids re-sending identical prompts.
# Cache file: <run_dir>/api_cache.json
cache_enabled = true

# Uncomment for OpenAI-compatible providers (Together, Groq, Mistral, etc.)
# base_url = "https://api.together.xyz/v1"
# api_key  = ""   # set here or via OPENAI_API_KEY

# ── Orchestrator ──────────────────────────────────────────────────────────────
[orchestrator]

# Parallel task execution threads.
# Keep at 1 when backend = "llamacpp" — llama.cpp is not thread-safe.
max_workers = 1

# Maximum LLM turns per task node before marking it failed.
max_turns = 5

# Maximum times a bridge node is injected for a single blocked task before
# giving up and executing the task anyway.
max_gap_fill_attempts = 2

# Seconds the orchestrator loop sleeps when there is no planning or execution
# work to do. Lower values are more responsive; higher values reduce CPU load.
idle_sleep = 0.5

# ── Planner ───────────────────────────────────────────────────────────────────
[planner]

# Task count range the planner is instructed to stay within when decomposing
# a goal. Increasing max_tasks_per_goal allows finer-grained plans; decreasing
# it keeps plans coarser but faster to generate.
min_tasks_per_goal = 3
max_tasks_per_goal = 8

# ── Executor ──────────────────────────────────────────────────────────────────
[executor]

# Maximum characters a task result may contain before the executor asks the
# LLM to write the content to a file instead of returning it inline.
# Increase if tasks routinely need to return large text blobs.
max_inline_result_chars = 3000

# Total character budget shared across all upstream results included in a
# single execution prompt. Budget is split evenly between dependencies.
# Increase if tasks need more context from upstream nodes.
max_total_input_chars = 3000

# Maximum characters from a single tool-call result before it is truncated.
# Tools that return large outputs (e.g. web pages) are clipped to this length.
max_tool_result_chars = 2000

# Number of most-recent tool-call entries kept in the executor's history per
# turn. Older entries are dropped to keep prompts short.
max_history_entries = 3

# ── File-based LLM (development / testing only) ───────────────────────────────
[file_llm]

# Seconds between polls when waiting for a human-written response.
poll_interval = 0.5

# Seconds before the file-based backend raises TimeoutError.
timeout = 300

# Seconds between "still waiting" log messages.
progress_log_interval = 2

# Cache responses to disk; on a cache hit the poll loop is skipped entirely.
# Cache file: <run_dir>/file_llm_cache.json
cache_enabled = true

# ── Web / terminal server ─────────────────────────────────────────────────────
[server]

host = "127.0.0.1"
port = 8765
```

---

## Sections at a glance

| Section | Tunes |
|---|---|
| `[llm]` | Which backend is active |
| `[llamacpp]` | Local model file, GPU offload, context size, temperature, caching |
| `[claude]` / `[openai]` | Remote model name, temperature, token limit, caching |
| `[orchestrator]` | Parallelism, turn limits, gap-fill retries, idle polling rate |
| `[planner]` | Task count range per goal decomposition |
| `[executor]` | Character budgets for inputs, tool results, and inline results |
| `[file_llm]` | Polling, timeout, and caching for the file-based development backend |
| `[server]` | Host and port for the web UI |

---

## Switching backends

Edit `[llm] backend` in `config.toml`. The only other step is making sure the
credentials or model file for the chosen backend are available.

### Local llama.cpp

```toml
[llm]
backend = "llamacpp"

[llamacpp]
model_filename = "Llama-3.3-70B-Instruct-Q4_K_M.gguf"
n_gpu_layers   = -1
```

Install the extra if you haven't already:

```bash
# macOS (Apple Silicon)
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
pip install outlines

# Linux / NVIDIA CUDA
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
pip install outlines
```

See [llama-cpp-python installation](https://github.com/abetlen/llama-cpp-python#installation)
for other hardware (ROCm, Vulkan, CPU-only).

### Anthropic Claude

```toml
[llm]
backend = "claude"

[claude]
model         = "claude-opus-4-6"
cache_enabled = true
```

```bash
pip install cuddlytoddly[claude]
export ANTHROPIC_API_KEY=sk-ant-...
```

### OpenAI

```toml
[llm]
backend = "openai"

[openai]
model         = "gpt-4o"
cache_enabled = true
```

```bash
pip install cuddlytoddly[openai]
export OPENAI_API_KEY=sk-...
```

### OpenAI-compatible provider (Together, Groq, Mistral, etc.)

```toml
[llm]
backend = "openai"

[openai]
model    = "mistralai/Mixtral-8x7B-Instruct-v0.1"
base_url = "https://api.together.xyz/v1"
api_key  = "..."    # or set OPENAI_API_KEY
```

---

## Response caching

All three backends support caching prompt → response pairs to a JSON file in the run directory. On a cache hit the backend returns immediately without making any API call or local inference.

| Backend | Cache file | Key |
|---|---|---|
| `llamacpp` | `<run_dir>/llamacpp_cache.json` | `prompt` or `prompt + schema` |
| `claude` / `openai` | `<run_dir>/api_cache.json` | `prompt` or `prompt + schema` |
| `file` | `<run_dir>/file_llm_cache.json` | `prompt` |

The cache is **per run** — each goal slug gets its own directory so caches from different runs don't interfere. Resuming a run reuses the existing cache, which means tasks whose prompts haven't changed (e.g. a planning call for a goal that was already expanded before the crash) return instantly on restart.

**Disabling the cache** for a single backend:

```toml
[claude]
cache_enabled = false
```

**Clearing the cache** while the server is running: use `orchestrator.planner.llm.clear_cache()` (or the equivalent for the executor's client) from a Python console, or simply delete the JSON file from the run directory.

**When to disable:** during active prompt engineering, where you want every call to reach the model so you can see the effect of your edits immediately.

## Tuning guide

### Tasks feel too coarse / too granular

Adjust `[planner] min_tasks_per_goal` and `max_tasks_per_goal`. The planner prompt instructs the LLM to stay within this range.

```toml
[planner]
min_tasks_per_goal = 4
max_tasks_per_goal = 12
```

### Tasks hit the turn limit before finishing

Raise `[orchestrator] max_turns`. This allows the executor more LLM iterations per task before it is marked failed.

```toml
[orchestrator]
max_turns = 10
```

### Upstream context is being truncated

Raise `[executor] max_total_input_chars` to give tasks more of their upstream results. Note that larger values mean larger prompts and higher token costs.

```toml
[executor]
max_total_input_chars = 6000
```

### Tool results are cut off

Raise `[executor] max_tool_result_chars` if tools that return long text (web fetches, file reads) are being truncated too aggressively.

```toml
[executor]
max_tool_result_chars = 4000
```

### Gap-fill bridge nodes keep being injected

Lower `[orchestrator] max_gap_fill_attempts` to reduce how many bridging nodes are injected before the system gives up and executes the task with what it has.

```toml
[orchestrator]
max_gap_fill_attempts = 1
```

---

## Customising prompts and schemas

All prompt templates and JSON output schemas live in dedicated files — not scattered across implementation modules:

| File | Contents |
|---|---|
| `cuddlytoddly/planning/prompts.py` | All prompt text: planner decomposition, executor task prompt, quality-gate verification and dependency-check prompts, system prompt constants |
| `cuddlytoddly/planning/schemas.py` | All JSON schemas: `PLAN_SCHEMA`, `EXECUTION_TURN_SCHEMA`, `RESULT_VERIFICATION_SCHEMA`, `DEPENDENCY_CHECK_SCHEMA`, and the shared sub-schemas |

Edit those files directly to change what the LLM sees. Prompt functions use standard Python f-strings; schema dicts are plain Python — no DSL or templating engine to learn.

---

## Model file search

When `backend = "llamacpp"`, the model is located by probing four places in
order. The first file that exists wins.

| Priority | Location | Set by |
|---|---|---|
| 1 | `CUDDLYTODDLY_MODEL_PATH` env var | You |
| 2 | `~/.cache/llama.cpp/<filename>` | `llama-cli -hf` / `llama-server -hf` |
| 3 | `~/.cache/huggingface/hub/**/<filename>` | `huggingface-cli download` |
| 4 | `<data_dir>/models/<filename>` | Your own download |

If nothing is found, cuddlytoddly prints a clear error with the exact download
command for the configured `model_filename`.

---

## Environment variables

| Variable | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | API key for the `claude` backend |
| `OPENAI_API_KEY` | API key for the `openai` backend |
| `CUDDLYTODDLY_MODEL_PATH` | Full path to any `.gguf` file — overrides model search |
| `LLAMA_CACHE` | Override llama.cpp's cache dir (default `~/.cache/llama.cpp`) |
| `HF_HOME` | Override Hugging Face home (default `~/.cache/huggingface`) |

For the `claude` backend, the API key must be provided via the `ANTHROPIC_API_KEY`
environment variable. For the `openai` backend, the key can be set either via
`OPENAI_API_KEY` or as `api_key` under `[openai]` in `config.toml` — both are
supported.

---

## Data directory layout

```
<data_dir>/
├── config.toml              ← the file described on this page
├── models/                  ← optional: place .gguf files here
└── runs/
    └── <goal_slug>/
        ├── events.jsonl         ← full event log (enables crash recovery)
        ├── llamacpp_cache.json  ← response cache for the llamacpp backend
        ├── api_cache.json       ← response cache for claude / openai backends
        ├── file_llm_cache.json  ← response cache for the file-based backend
        ├── logs/
        ├── outputs/             ← working directory for file-writing tools
        └── dag_repo/            ← Git repo mirroring the DAG
```
