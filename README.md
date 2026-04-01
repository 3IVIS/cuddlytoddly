# cuddlytoddly

An LLM-driven autonomous planning and execution system built around a DAG (directed acyclic graph) of tasks. Give it a goal; it breaks the goal into tasks, executes them with tools, verifies results, and fills in gaps — continuously, with live terminal and web UIs.

**Why "cuddlytoddly"?**

Large language models are powerful at generating text and solving well-scoped problems, but they are widely recognized as weak at long-horizon, complex planning. Left alone, they often miss dependencies, overlook edge cases, or wander off track.

cuddlytoddly takes a different approach: instead of expecting the LLM to plan everything autonomously, we provide a structured framework where humans and the system collaborate. The LLM handles decomposition, execution, and verification, while the framework ensures tasks are organized, dependencies are respected, and gaps are bridged.

Think of it as holding the model's hand while it learns to walk through complex goals — hence the name cuddlytoddly. It's not about blind autonomy; it's about guided, reliable progress.

---

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

---

## Installation

```bash
pip install cuddlytoddly
```

**Requirements:** Python 3.11+, `git` on your PATH (for the DAG visualiser).

Then install one or more LLM backend extras depending on how you want to run the model:

| Backend | Extra to install |
|---|---|
| Anthropic Claude | `pip install cuddlytoddly[claude]` |
| OpenAI / compatible | `pip install cuddlytoddly[openai]` |
| Local llama.cpp | `pip install cuddlytoddly[local]` — see [Local model setup](#local-model-setup-llamacpp) |
| Everything | `pip install cuddlytoddly[all]` |

---

## Quick start

```bash
pip install cuddlytoddly[claude]
export ANTHROPIC_API_KEY=sk-ant-...
cuddlytoddly "Write a market analysis for electric scooters"
```

On first run, a `config.toml` is written to your user data directory with all defaults pre-filled. Open it to change backends, model settings, temperature, and more — **no code editing required**.

```bash
# Print the config file location
python -c "from cuddlytoddly.config import CONFIG_PATH; print(CONFIG_PATH)"
```

Pass no argument to open the startup screen (resume a previous run, load a manual plan, etc.). The web UI opens automatically. Run data is stored locally and can be resumed — the event log preserves all state.

### Switching backends

Edit `[llm] backend` in `config.toml`. That's the only change needed.

```toml
# config.toml

[llm]
backend = "claude"    # or "openai" or "llamacpp"

[claude]
model = "claude-opus-4-6"

[openai]
model    = "gpt-4o"
# base_url = "https://api.together.xyz/v1"   # any OpenAI-compatible provider
```

Then install the matching extra and set the API key:

| Backend | Extra | Env var |
|---|---|---|
| `claude` | `pip install cuddlytoddly[claude]` | `ANTHROPIC_API_KEY` |
| `openai` | `pip install cuddlytoddly[openai]` | `OPENAI_API_KEY` |
| `llamacpp` | see [Local model setup](#local-model-setup-llamacpp) | — |

---

## Local model setup (llama.cpp)

Running a model locally gives you full privacy, no API costs, and offline operation. The local backend uses [llama-cpp-python](https://github.com/abetlen/llama-cpp-python), a Python binding for [llama.cpp](https://github.com/ggerganov/llama.cpp).

### Step 1 — Install llama-cpp-python

The right install command depends on your hardware. The plain `pip install cuddlytoddly[local]` build is CPU-only and very slow for large models. Choose the command that matches your setup:

**macOS (Apple Silicon — Metal GPU)**
```bash
CMAKE_ARGS="-DGGML_METAL=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**Linux / Windows — NVIDIA GPU (CUDA)**
```bash
CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --force-reinstall --no-cache-dir
```

**Linux — CPU only**
```bash
pip install llama-cpp-python
```

For other hardware (ROCm, Vulkan, SYCL) and detailed build options, see the official installation guide:
👉 **https://github.com/abetlen/llama-cpp-python#installation**

After installing llama-cpp-python, install the remaining local extras:

```bash
pip install "outlines>=0.0.46"
# or in one shot:
pip install cuddlytoddly[local]   # then re-run the GPU install above to override
```

### Step 2 — Download a model

Models must be in **GGUF format**. The default model is **Llama 3.3 70B Instruct Q4_K_M** — a good balance of quality and speed on 48 GB+ VRAM or unified memory.

**If you already have this model downloaded** (via `llama-cli -hf`, `llama-server -hf`, or `huggingface-cli download`), cuddlytoddly will find it automatically — no extra steps needed. It probes these locations in order:

1. `CUDDLYTODDLY_MODEL_PATH` env var — explicit override, any path
2. `~/.cache/llama.cpp/` — llama.cpp's native download cache
3. `~/.cache/huggingface/hub/` — Hugging Face hub cache
4. `<data dir>/models/` — cuddlytoddly's own models folder

If the model isn't found anywhere, you'll get a clear error message with the exact download command to run.

**To download the default model** into cuddlytoddly's own folder:

```bash
pip install huggingface-hub

# Linux / macOS
DATA_DIR=$(python -c "from platformdirs import user_data_dir; print(user_data_dir('cuddlytoddly', '3IVIS'))")
mkdir -p "$DATA_DIR/models"
huggingface-cli download bartowski/Llama-3.3-70B-Instruct-GGUF \
  Llama-3.3-70B-Instruct-Q4_K_M.gguf \
  --local-dir "$DATA_DIR/models"

# Windows PowerShell
$dataDir = python -c "from platformdirs import user_data_dir; print(user_data_dir('cuddlytoddly', '3IVIS'))"
New-Item -ItemType Directory -Force "$dataDir\models"
huggingface-cli download bartowski/Llama-3.3-70B-Instruct-GGUF Llama-3.3-70B-Instruct-Q4_K_M.gguf --local-dir "$dataDir\models"
```

**To use a different model or a custom path**, set the env var:

```bash
export CUDDLYTODDLY_MODEL_PATH=/path/to/your-model.gguf
```

### Step 3 — Configure the backend

Open your `config.toml` and set:

```toml
[llm]
backend = "llamacpp"

[llamacpp]
model_filename = "Llama-3.3-70B-Instruct-Q4_K_M.gguf"
n_gpu_layers   = -1    # -1 = all layers on GPU, 0 = CPU only
n_ctx          = 16384
max_tokens     = 8192
temperature    = 0.1
cache_enabled  = true
```

Change `model_filename` to match whatever you downloaded. Everything else can stay at defaults to start.

### Step 4 — Run

```bash
cuddlytoddly "Write a market analysis for electric scooters"
```

The first run will load the model into memory (10–30 seconds depending on hardware), then proceed normally. Subsequent runs reuse the response cache (`llamacpp_cache.json`) to skip identical prompts. The same caching applies to the `claude` and `openai` backends too (`api_cache.json`).

---

## LLM backends — full reference

See [docs/configuration.md](docs/configuration.md) for the complete config file reference and all available options per backend.

---

## Customising prompts and schemas

All LLM prompt templates and JSON output schemas are consolidated into two files — you never need to dig through the implementation to adjust them:

| File | What it contains |
|---|---|
| `cuddlytoddly/planning/prompts.py` | Every prompt template sent to the LLM: planner, executor, verify-result, check-dependencies, plus the system prompt constants |
| `cuddlytoddly/planning/schemas.py` | Every JSON schema used for structured output: `PLAN_SCHEMA`, `EXECUTION_TURN_SCHEMA`, `RESULT_VERIFICATION_SCHEMA`, etc. |

Each function in `prompts.py` is documented with its parameters so it's clear what context is injected where. Edit the text freely — the functions use standard Python f-strings with named parameters.

---

## Adding skills

Drop a folder with a `SKILL.md` (and optional `tools.py`) into `cuddlytoddly/skills/`. The `SkillLoader` discovers it automatically. See [docs/skills.md](docs/skills.md) for the full format.

---

## Documentation

- [Architecture](docs/architecture.md) — how the components fit together
- [Configuration](docs/configuration.md) — LLM backends, run directory, tuning parameters, environment variables
- [Skills](docs/skills.md) — built-in skills and how to add custom ones
- [API Reference](docs/api.md) — public Python API

---

## Where is my data?

Models and run data are stored in the OS user data directory, completely separate from the package code. This works correctly whether you run from source or install via pip.

```bash
# Print the exact path on your machine
python -c "from platformdirs import user_data_dir; print(user_data_dir('cuddlytoddly', '3IVIS'))"
```

```
~/.local/share/cuddlytoddly/     ← Linux
~/Library/Application Support/cuddlytoddly/  ← macOS
%LOCALAPPDATA%\3IVIS\cuddlytoddly\  ← Windows

├── config.toml
├── models/
│   └── Llama-3.3-70B-Instruct-Q4_K_M.gguf
└── runs/
    └── write_a_market_analysis.../
        ├── events.jsonl         # full event log — enables crash recovery
        ├── llamacpp_cache.json  # response cache (llamacpp backend)
        ├── api_cache.json       # response cache (claude / openai backends)
        ├── file_llm_cache.json  # response cache (file backend)
        ├── logs/
        ├── outputs/             # working directory for file-writing tools
        └── dag_repo/            # Git repo mirroring the DAG
```

## Project structure

```
cuddlytoddly/
├── core/           # TaskGraph, events, reducer, ID generator
├── engine/         # SimpleOrchestrator, QualityGate, ExecutionStepReporter
├── infra/          # Logging, EventQueue, EventLog, replay
├── planning/
│   ├── prompts.py      ← all LLM prompt templates (edit here)
│   ├── schemas.py      ← all JSON output schemas (edit here)
│   ├── llm_interface.py
│   ├── llm_planner.py
│   ├── llm_executor.py
│   └── llm_output_validator.py
├── skills/         # SkillLoader + built-in skill packs
│   ├── code_execution/
│   └── file_ops/
└── ui/             # Curses terminal UI, web UI, Git DAG projection
docs/
pyproject.toml
LICENSE
```

---

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

# Swap "claude" for "openai" or "llamacpp" — everything else is identical
llm = create_llm_client("claude", model="claude-opus-4-6")

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

All numeric limits (`max_turns`, `max_workers`, etc.) default to the values in `config.toml` when the system is started via the CLI. When constructing components programmatically you can pass them as keyword arguments — see [docs/api.md](docs/api.md) for the full signature of each class.

---

## License

MIT — see [LICENSE](LICENSE).
