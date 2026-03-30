# Configuration

cuddlytoddly is configured through a single TOML file in the user data directory.
No environment variables are required for basic operation, and nothing is hardcoded
inside the package itself.

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
model       = "claude-opus-4-6"
temperature = 0.1
max_tokens  = 8192

# ── OpenAI-compatible API ─────────────────────────────────────────────────────
[openai]

# API key can be set here or via the OPENAI_API_KEY environment variable.
model       = "gpt-4o"
temperature = 0.1
max_tokens  = 8192

# Uncomment for OpenAI-compatible providers (Together, Groq, Mistral, etc.)
# base_url = "https://api.together.xyz/v1"
# api_key  = ""   # set here or via OPENAI_API_KEY

# ── Orchestrator ──────────────────────────────────────────────────────────────
[orchestrator]

# Parallel task execution threads.
# Keep at 1 when backend = "llamacpp" — llama.cpp is not thread-safe.
max_workers = 1

# Maximum LLM turns per task node before marking it failed
max_turns = 5

# ── Web / terminal server ─────────────────────────────────────────────────────
[server]

host = "127.0.0.1"
port = 8765
```

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
model = "claude-opus-4-6"
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
model = "gpt-4o"
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
        ├── events.jsonl     ← full event log (enables crash recovery)
        ├── llamacpp_cache.json
        ├── logs/
        ├── outputs/         ← working directory for file-writing tools
        └── dag_repo/        ← Git repo mirroring the DAG
```