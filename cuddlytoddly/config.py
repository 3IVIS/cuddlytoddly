"""
cuddlytoddly.config
~~~~~~~~~~~~~~~~~~~
Configuration management.

On first run a default ``config.toml`` is written to the user data directory.
The backend is auto-detected from environment variables so the file is
immediately usable without any manual editing in the common cases.

Data directory per OS:
  Linux   : ~/.local/share/cuddlytoddly/
  macOS   : ~/Library/Application Support/cuddlytoddly/
  Windows : %LOCALAPPDATA%\\3IVIS\\cuddlytoddly\\
"""

from __future__ import annotations

import os
import sys
import tomllib
from pathlib import Path

from platformdirs import user_data_dir

from cuddlytoddly.infra.logging import get_logger

logger = get_logger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_DIR = Path(user_data_dir("cuddlytoddly", "3IVIS"))
CONFIG_PATH = DATA_DIR / "config.toml"

# ── Approximate download sizes for known model size classes ───────────────────

_MODEL_SIZES: dict[str, str] = {
    "70B": "~40 GB",
    "72B": "~43 GB",
    "65B": "~35 GB",
    "34B": "~19 GB",
    "13B": "~7 GB",
    "8B": "~5 GB",
    "7B": "~4 GB",
    "3B": "~2 GB",
    "1B": "~1 GB",
}

# ── Default config template ───────────────────────────────────────────────────
# {backend} is substituted at first-run time from _detect_backend().

_DEFAULT_CONFIG_TEMPLATE = """\
# cuddlytoddly configuration
# Edit this file to customise backends, model settings, and server options.
# Docs: https://github.com/3IVIS/cuddlytoddly/blob/main/docs/configuration.md

# ── LLM backend ───────────────────────────────────────────────────────────────
[llm]

# Which backend to use: "llamacpp" (local), "claude", or "openai"
# Auto-detected from environment variables on first run.
backend = "{backend}"

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

# Cache API responses to disk; avoids re-sending identical prompts
cache_enabled = true

# ── OpenAI-compatible API ─────────────────────────────────────────────────────
[openai]

# Requires the OPENAI_API_KEY environment variable (or api_key below).
model         = "gpt-4o"
temperature   = 0.1
max_tokens    = 8192

# Cache API responses to disk; avoids re-sending identical prompts
cache_enabled = true

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

# Maximum times the orchestrator injects a bridge node for a single blocked
# task before giving up and executing it anyway.
max_gap_fill_attempts = 2

# Maximum verification failures before a node is permanently failed instead
# of being reset and retried.  Each retry waits an exponentially increasing
# backoff (1s, 2s, 4s … capped at 60s) before re-launching.
max_retries = 5

# Seconds the orchestrator loop sleeps when idle (no planning or execution work).
idle_sleep = 0.5

# ── Planner ───────────────────────────────────────────────────────────────────
[planner]

# Task count guidelines per goal decomposition.
# The planner prompt instructs the LLM to stay within this range.
min_tasks_per_goal = 3
max_tasks_per_goal = 8

# When true, every planning call is followed by a scrutinizing call where the
# LLM reviews its own plan against all original constraints and produces an
# improved version.  The improved plan is what reaches the reducer.
# Set to false to skip scrutiny and use the raw plan directly (faster, cheaper).
scrutinize_plan = true

# ── Executor ──────────────────────────────────────────────────────────────────
[executor]

# Maximum characters a task result may contain before the executor asks the
# LLM to write the content to a file instead of returning it inline.
max_inline_result_chars = 3000

# Total character budget shared across all upstream task results included in
# a single execution prompt.  Budget is split evenly between dependencies.
max_total_input_chars = 3000

# Maximum characters from a single tool-call result before it is truncated.
max_tool_result_chars = 2000

# Number of most-recent tool-call entries kept in the executor's history
# context per turn.  Older entries are dropped to keep prompts short.
max_history_entries = 3



# ── File-based LLM (development / testing only) ───────────────────────────────
[file_llm]

# Seconds between polls when waiting for a response in the file-based backend.
poll_interval = 0.5

# Seconds before the file-based backend raises TimeoutError.
timeout = 300

# Seconds between progress-log messages while waiting for a response.
progress_log_interval = 2

# Cache responses to disk; on a cache hit the poll loop is skipped entirely.
cache_enabled = true

# ── Web / terminal server ─────────────────────────────────────────────────────
[server]

host = "127.0.0.1"
port = 8765
"""

_VALID_BACKENDS = {"llamacpp", "claude", "openai"}

# ── Public API ────────────────────────────────────────────────────────────────


def load_config() -> dict:
    """
    Load ``DATA_DIR/config.toml``, creating it with smart defaults if absent.

    On first run the backend is chosen automatically:
      * ``ANTHROPIC_API_KEY`` set → ``"claude"``
      * ``OPENAI_API_KEY`` set    → ``"openai"``
      * neither                  → ``"llamacpp"``

    A notice is printed to stderr so the user immediately sees where the
    config file lives and what backend was chosen.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not CONFIG_PATH.exists():
        backend = _detect_backend()
        content = _DEFAULT_CONFIG_TEMPLATE.replace('"{backend}"', f'"{backend}"')
        CONFIG_PATH.write_text(content, encoding="utf-8")
        _print_first_run_notice(backend)
        logger.info("[CONFIG] Created default config at %s (backend=%s)", CONFIG_PATH, backend)
    else:
        logger.info("[CONFIG] Loading %s", CONFIG_PATH)

    with CONFIG_PATH.open("rb") as fh:
        cfg = tomllib.load(fh)

    _validate(cfg)
    return cfg


def resolve_model_path(cfg: dict) -> str:
    """
    Return the absolute path to the GGUF model file specified in *cfg*.

    Search order
    ------------
    1. ``CUDDLYTODDLY_MODEL_PATH`` env var
    2. ``LLAMA_CACHE`` / ``~/.cache/llama.cpp/``
    3. ``HF_HOME`` / ``~/.cache/huggingface/hub/``
    4. ``DATA_DIR/models/<model_filename>``

    Raises ``FileNotFoundError`` with a download command and size hint.
    """
    filename = cfg.get("llamacpp", {}).get("model_filename", "Llama-3.3-70B-Instruct-Q4_K_M.gguf")

    env_override = os.environ.get("CUDDLYTODDLY_MODEL_PATH")
    if env_override:
        p = Path(env_override).expanduser().resolve()
        if p.exists():
            logger.info("[MODEL] Using CUDDLYTODDLY_MODEL_PATH: %s", p)
            return str(p)
        logger.warning("[MODEL] CUDDLYTODDLY_MODEL_PATH set but not found: %s", p)

    candidates: list[Path] = []

    llama_cache = Path(os.environ.get("LLAMA_CACHE", Path.home() / ".cache" / "llama.cpp"))
    candidates.append(llama_cache / filename)

    hf_hub = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub"
    if hf_hub.exists():
        candidates.extend(sorted(hf_hub.glob(f"**/snapshots/**/{filename}"), reverse=True))

    own_models = DATA_DIR / "models"
    candidates.append(own_models / filename)

    for candidate in candidates:
        if candidate.exists():
            logger.info("[MODEL] Found model: %s", candidate)
            return str(candidate)

    size_hint = _model_size_hint(filename)
    size_note = f"  (approx. {size_hint} download)\n" if size_hint else ""

    raise FileNotFoundError(
        f"\nModel '{filename}' not found in any standard location.\n"
        f"{size_note}\n"
        f"Option 1 — download into cuddlytoddly's models folder:\n"
        f"  pip install huggingface-hub\n"
        f"  huggingface-cli download bartowski/Llama-3.3-70B-Instruct-GGUF"
        f" {filename} --local-dir {own_models}\n\n"
        f"Option 2 — point to an existing file:\n"
        f"  export CUDDLYTODDLY_MODEL_PATH=/path/to/{filename}\n\n"
        f"Option 3 — change the filename in the config:\n"
        f'  {CONFIG_PATH}  →  [llamacpp] model_filename = "your-model.gguf"\n'
    )


def preflight_check(cfg: dict) -> list[dict]:
    """
    Run lightweight pre-flight checks for the configured backend.

    Returns a list of issue dicts (empty = all good). Each dict has:
      ``level``   – ``"error"`` or ``"warning"``
      ``message`` – one-line description
      ``fix``     – short actionable hint

    Checks are fast: no model loading, no API calls.
    """
    issues: list[dict] = []
    backend = cfg.get("llm", {}).get("backend", "")
    llama_cfg = cfg.get("llamacpp", {})

    # ── Cross-backend: API key present but wrong backend ──────────────────────
    if backend == "llamacpp":
        if os.environ.get("ANTHROPIC_API_KEY"):
            issues.append(
                {
                    "level": "warning",
                    "message": "ANTHROPIC_API_KEY is set but the backend is 'llamacpp'.",
                    "fix": f'To use Claude, set backend = "claude" in {CONFIG_PATH}',
                }
            )
        elif os.environ.get("OPENAI_API_KEY"):
            issues.append(
                {
                    "level": "warning",
                    "message": "OPENAI_API_KEY is set but the backend is 'llamacpp'.",
                    "fix": f'To use OpenAI, set backend = "openai" in {CONFIG_PATH}',
                }
            )

    # ── llamacpp checks ───────────────────────────────────────────────────────
    if backend == "llamacpp":
        llama_installed = False
        try:
            import llama_cpp  # noqa: F401

            llama_installed = True
        except ImportError:
            issues.append(
                {
                    "level": "error",
                    "message": "llama-cpp-python is not installed.",
                    "fix": "pip install cuddlytoddly[local]  "
                    "(add GPU build flags for acceleration — see docs)",
                }
            )

        if llama_installed:
            n_gpu = llama_cfg.get("n_gpu_layers", -1)
            if n_gpu != 0 and not _llama_has_gpu_support():
                filename = llama_cfg.get("model_filename", "")
                size_hint = _model_size_hint(filename)
                size_note = f" ({size_hint})" if size_hint else ""
                issues.append(
                    {
                        "level": "warning",
                        "message": (
                            f"llama-cpp-python has no GPU support — "
                            f"'{filename}'{size_note} will run on CPU and be very slow."
                        ),
                        "fix": (
                            'macOS:  CMAKE_ARGS="-DGGML_METAL=on" '
                            "pip install llama-cpp-python --force-reinstall --no-cache-dir  |  "
                            'NVIDIA:  CMAKE_ARGS="-DGGML_CUDA=on" ...'
                        ),
                    }
                )

        try:
            resolve_model_path(cfg)
        except FileNotFoundError:
            filename = llama_cfg.get("model_filename", "model.gguf")
            own_dir = DATA_DIR / "models"
            size_hint = _model_size_hint(filename)
            size_note = f" ({size_hint} download)" if size_hint else ""
            issues.append(
                {
                    "level": "error",
                    "message": f"Model file '{filename}' not found{size_note}.",
                    "fix": (
                        f"huggingface-cli download bartowski/Llama-3.3-70B-Instruct-GGUF"
                        f" {filename} --local-dir {own_dir}"
                    ),
                }
            )

    # ── claude checks ─────────────────────────────────────────────────────────
    elif backend == "claude":
        try:
            import anthropic  # noqa: F401
        except ImportError:
            issues.append(
                {
                    "level": "error",
                    "message": "anthropic package is not installed.",
                    "fix": "pip install cuddlytoddly[claude]",
                }
            )

        if not os.environ.get("ANTHROPIC_API_KEY"):
            issues.append(
                {
                    "level": "error",
                    "message": "ANTHROPIC_API_KEY environment variable is not set.",
                    "fix": "export ANTHROPIC_API_KEY=sk-ant-...",
                }
            )

    # ── openai checks ─────────────────────────────────────────────────────────
    elif backend == "openai":
        try:
            import openai  # noqa: F401
        except ImportError:
            issues.append(
                {
                    "level": "error",
                    "message": "openai package is not installed.",
                    "fix": "pip install cuddlytoddly[openai]",
                }
            )

        has_key = bool(os.environ.get("OPENAI_API_KEY") or cfg.get("openai", {}).get("api_key"))
        if not has_key:
            issues.append(
                {
                    "level": "error",
                    "message": "No OpenAI API key found.",
                    "fix": (
                        "export OPENAI_API_KEY=sk-...  "
                        f"(or set api_key under [openai] in {CONFIG_PATH})"
                    ),
                }
            )

    return issues


# ── Config section accessors ──────────────────────────────────────────────────
# Convenience helpers used by __main__.py to read the new sections with
# sensible defaults (so configs written before these sections were added
# continue to work without requiring a manual edit).


def get_executor_cfg(cfg: dict) -> dict:
    """Return the [executor] section with defaults filled in."""
    c = cfg.get("executor", {})
    return {
        "max_inline_result_chars": c.get("max_inline_result_chars", 3000),
        "max_total_input_chars": c.get("max_total_input_chars", 3000),
        "max_tool_result_chars": c.get("max_tool_result_chars", 2000),
        "max_history_entries": c.get("max_history_entries", 3),
    }


def get_planner_cfg(cfg: dict) -> dict:
    """Return the [planner] section with defaults filled in."""
    c = cfg.get("planner", {})
    return {
        "min_tasks_per_goal": c.get("min_tasks_per_goal", 3),
        "max_tasks_per_goal": c.get("max_tasks_per_goal", 8),
        "scrutinize_plan": c.get("scrutinize_plan", False),
    }


def get_orchestrator_cfg(cfg: dict) -> dict:
    """Return the [orchestrator] section with defaults filled in."""
    c = cfg.get("orchestrator", {})
    return {
        "max_workers": c.get("max_workers", 1),
        "max_turns": c.get("max_turns", 5),
        "max_gap_fill_attempts": c.get("max_gap_fill_attempts", 2),
        "max_retries": c.get("max_retries", 5),
        "idle_sleep": c.get("idle_sleep", 0.5),
    }


def get_file_llm_cfg(cfg: dict) -> dict:
    """Return the [file_llm] section with defaults filled in."""
    c = cfg.get("file_llm", {})
    return {
        "poll_interval": c.get("poll_interval", 0.5),
        "timeout": c.get("timeout", 300),
        "progress_log_interval": c.get("progress_log_interval", 2),
        "cache_enabled": c.get("cache_enabled", True),
    }


# ── Internal helpers ──────────────────────────────────────────────────────────


def _detect_backend() -> str:
    """Return the best default backend based on set environment variables."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "llamacpp"


def _model_size_hint(filename: str) -> str:
    upper = filename.upper()
    for tag, size in _MODEL_SIZES.items():
        if tag in upper:
            return size
    return ""


def _llama_has_gpu_support() -> bool:
    try:
        from llama_cpp import llama_supports_gpu_offload

        return bool(llama_supports_gpu_offload())
    except (ImportError, AttributeError):
        return False


def _print_first_run_notice(backend: str) -> None:
    sep = "─" * 60
    lines = [
        "",
        sep,
        "  cuddlytoddly — first run",
        "",
        "  Config file created at:",
        f"    {CONFIG_PATH}",
        "",
    ]

    if backend == "claude":
        lines += [
            "  Detected ANTHROPIC_API_KEY → backend set to 'claude'.",
            "  You're ready to go.",
        ]
    elif backend == "openai":
        lines += [
            "  Detected OPENAI_API_KEY → backend set to 'openai'.",
            "  You're ready to go.",
        ]
    else:
        lines += [
            "  No API key detected → backend defaulted to 'llamacpp'.",
            "",
            "  Quick setup options:",
            "    • Cloud API (easier):  edit config.toml and set",
            '        backend = "claude"  then  export ANTHROPIC_API_KEY=sk-ant-...',
            "      or",
            '        backend = "openai"  then  export OPENAI_API_KEY=sk-...',
            "",
            "    • Local model:  install GPU support and download a GGUF model.",
            "        See docs/configuration.md for platform-specific instructions.",
        ]

    lines += [sep, ""]
    print("\n".join(lines), file=sys.stderr)


def _validate(cfg: dict) -> None:
    backend = cfg.get("llm", {}).get("backend", "")
    if backend not in _VALID_BACKENDS:
        raise ValueError(
            f"[CONFIG] llm.backend = {backend!r} is not valid.\n"
            f"Choose one of: {', '.join(sorted(_VALID_BACKENDS))}.\n"
            f"Edit {CONFIG_PATH} to fix this."
        )
