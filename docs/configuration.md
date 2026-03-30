# Configuration

All runtime configuration is passed directly to `main()` or to individual components — there is no global config file. The values below are the defaults set in `cuddlytoddly/__main__.py` and can be overridden by editing that file or by calling the Python API directly.

## LLM backends

cuddlytoddly supports three backends selected via `create_llm_client(backend=...)`.

### Anthropic Claude (default recommended)

```python
from cuddlytoddly.planning.llm_interface import create_llm_client

llm = create_llm_client(
    "claude",
    model="claude-3-5-sonnet-20241022",  # any Anthropic model
    temperature=0.1,
    max_tokens=8192,
)
```

Requires the `ANTHROPIC_API_KEY` environment variable.

### OpenAI-compatible API

```python
llm = create_llm_client(
    "openai",
    model="gpt-4o",
    base_url="https://api.openai.com/v1",   # or any compatible endpoint
    temperature=0.1,
    max_tokens=8192,
)
```

Requires `OPENAI_API_KEY`, or set `base_url` + `api_key` for a custom endpoint.  
Install the extra: `pip install cuddlytoddly[openai]`.

### Local llama.cpp

```python
llm = create_llm_client(
    "llamacpp",
    model_path="/path/to/model.gguf",
    n_gpu_layers=-1,       # -1 = all layers on GPU
    temperature=0.1,
    n_ctx=16384,
    max_tokens=8192,
    cache_path="llamacpp_cache.json",  # optional response cache
)
```

Install the extra: `pip install cuddlytoddly[local]`.

## Orchestrator options

| Parameter | Default | Description |
|---|---|---|
| `max_workers` | `1` | Parallel node execution threads. Use `1` for llama.cpp (not thread-safe). |
| `max_turns` | `5` | Max LLM turns per node execution before giving up. |

## Run directory

Each invocation creates a `runs/<goal_slug>/` directory containing:

```
runs/how_to_go_to_mars/
├── events.jsonl        # full event log (enables crash recovery)
├── llamacpp_cache.json # LLM response cache (optional)
├── logs/               # rotating log files
├── outputs/            # working directory for file-writing tools
└── dag_repo/           # Git repo mirroring the DAG (for visualization)
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | For `claude` backend | Anthropic API key |
| `OPENAI_API_KEY` | For `openai` backend | OpenAI API key |

## Git DAG projection

The Git projection (`ui/git_projection.py`) requires `git` to be installed on the system. It is purely visual — it does not affect the DAG or execution. The path to the repo is set in `main()`:

```python
import cuddlytoddly.ui.git_projection as git_proj
git_proj.REPO_PATH = str(run_dir / "dag_repo")
```
