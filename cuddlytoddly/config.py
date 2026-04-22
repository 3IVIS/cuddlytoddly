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

from toddly.infra.logging import get_logger

# FIX #8: import model name constants so the config template uses the same
# single source of truth as the LLM backends, rather than hard-coded strings
# that go stale when default models are updated.
from toddly.planning.llm_interface import _DEFAULT_CLAUDE_MODEL, _DEFAULT_OPENAI_MODEL
from toddly.utils.config_utils import (
    detect_backend as _detect_backend,
)
from toddly.utils.config_utils import (
    llama_has_gpu_support as _llama_has_gpu_support,
)
from toddly.utils.config_utils import (
    model_size_hint as _model_size_hint,
)
from toddly.utils.config_utils import (
    resolve_model_path as _resolve_model_path_generic,
)
from toddly.utils.config_utils import (
    validate_config as _validate,
)

logger = get_logger(__name__)

# ── Paths ──────────────────────────────────────────────────────────────────────

DATA_DIR = Path(user_data_dir("cuddlytoddly", "3IVIS"))
CONFIG_PATH = DATA_DIR / "config.toml"


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

# ── Execution limits (llamacpp) ───────────────────────────────────────────────
# Conservative limits for local inference: single-threaded, tight context window.

# Parallel task execution threads.
# Must stay at 1 — llama.cpp is not thread-safe.
max_workers = 1

# Maximum LLM turns per task node before marking it failed.
max_turns = 5

# Maximum tasks the planner generates per goal.
max_tasks_per_goal = 8

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

# ── Anthropic Claude (API) ────────────────────────────────────────────────────
[claude]

# Requires the ANTHROPIC_API_KEY environment variable.
model         = "{claude_model}"
temperature   = 0.1
max_tokens    = 8192

# Cache API responses to disk; avoids re-sending identical prompts
cache_enabled = true

# ── Execution limits (claude) ─────────────────────────────────────────────────
# Higher limits for remote API: large context windows, parallelisable calls.

max_workers             = 4
max_turns               = 10
max_tasks_per_goal      = 15
max_inline_result_chars = 12000
max_total_input_chars   = 12000
max_tool_result_chars   = 8000
max_history_entries     = 10

# ── OpenAI-compatible API ─────────────────────────────────────────────────────
[openai]

# Requires the OPENAI_API_KEY environment variable (or api_key below).
model         = "{openai_model}"
temperature   = 0.1
max_tokens    = 8192

# Cache API responses to disk; avoids re-sending identical prompts
cache_enabled = true

# Uncomment for OpenAI-compatible providers (Together, Groq, Mistral, etc.)
# base_url = "https://api.together.xyz/v1"
#
# FIX #11: SECURITY WARNING — setting api_key here stores your secret key in a
# plain-text file on disk at a well-known path.  Anyone with read access to
# your home directory can read it.  Prefer setting the OPENAI_API_KEY
# environment variable instead (e.g. in your shell profile or a .env file that
# is NOT committed to version control).  Only use this option when an env var
# is not practical, and ensure the config file has restrictive permissions
# (chmod 600 on Linux/macOS).
# api_key  = ""   # prefer OPENAI_API_KEY env var — see security note above

# ── Execution limits (openai) ─────────────────────────────────────────────────
# Higher limits for remote API: large context windows, parallelisable calls.

max_workers             = 4
max_turns               = 10
max_tasks_per_goal      = 15
max_inline_result_chars = 12000
max_total_input_chars   = 12000
max_tool_result_chars   = 8000
max_history_entries     = 10

# ── Orchestrator ──────────────────────────────────────────────────────────────
# These settings are backend-agnostic and apply regardless of which LLM is used.
[orchestrator]

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
# These settings are backend-agnostic.  max_tasks_per_goal and max_turns are
# set per-backend above because their ideal values differ significantly.
[planner]

# Minimum tasks the planner must generate per goal.
min_tasks_per_goal = 3

# When true, every planning call is followed by a scrutinizing call where the
# LLM reviews its own plan against all original constraints and produces an
# improved version.  The improved plan is what reaches the reducer.
# Set to false to skip scrutiny and use the raw plan directly (faster, cheaper).
scrutinize_plan = true

# Number of clarification fields the planner generates per goal.
# These control how many context questions appear in the clarification node.
# Intentionally separate from task count limits: a goal that needs 15 tasks
# does not necessarily need 15 clarification questions.
min_clarification_fields = 2
max_clarification_fields = 6

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

# Execution limits for the file backend (dev/testing — same as llamacpp).
max_workers             = 1
max_turns               = 5
max_tasks_per_goal      = 8
max_inline_result_chars = 3000
max_total_input_chars   = 3000
max_tool_result_chars   = 2000
max_history_entries     = 3

# ── Web / terminal server ─────────────────────────────────────────────────────
[server]

host = "127.0.0.1"
port = 8765
"""

# FIX #8: include "file" so that backend = "file" in config.toml is accepted
# by _validate() instead of raising ValueError.  The "file" backend is a
# legitimate dev/testing backend fully supported by _build_llm_client().
_VALID_BACKENDS = {"llamacpp", "claude", "openai", "file"}

# Backends that use a remote API and can therefore afford higher limits.
_API_BACKENDS = {"claude", "openai"}

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
        # FIX #8: substitute all three placeholders so the written file always
        # reflects the current default model names from llm_interface.py.
        content = (
            _DEFAULT_CONFIG_TEMPLATE.replace('"{backend}"', f'"{backend}"')
            .replace('"{claude_model}"', f'"{_DEFAULT_CLAUDE_MODEL}"')
            .replace('"{openai_model}"', f'"{_DEFAULT_OPENAI_MODEL}"')
        )
        CONFIG_PATH.write_text(content, encoding="utf-8")
        _print_first_run_notice(backend)
        logger.info("[CONFIG] Created default config at %s (backend=%s)", CONFIG_PATH, backend)
    else:
        logger.info("[CONFIG] Loading %s", CONFIG_PATH)

    with CONFIG_PATH.open("rb") as fh:
        cfg = tomllib.load(fh)

    _validate(cfg)

    # FIX #11: warn at load time if a plain-text api_key is present in the
    # [openai] section so that users who set it there are reminded of the
    # security implication every time the application starts.
    if cfg.get("openai", {}).get("api_key"):
        logger.warning(
            "[CONFIG] api_key is set in plain text under [openai] in %s. "
            "Consider using the OPENAI_API_KEY environment variable instead "
            "to avoid storing secrets on disk. "
            "Ensure the config file has restrictive permissions (chmod 600).",
            CONFIG_PATH,
        )

    return cfg


def resolve_model_path(cfg: dict) -> str:
    """
    Return the absolute path to the GGUF model file specified in *cfg*.

    Thin wrapper around toddly.utils.config_utils.resolve_model_path
    that supplies the application-specific DATA_DIR and CONFIG_PATH so
    all existing callers (preflight_check, __main__) continue to work
    without any changes.
    """
    return _resolve_model_path_generic(cfg, data_dir=DATA_DIR, config_path=CONFIG_PATH)


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
            "",
            "  Execution limits are configured under [claude] in config.toml.",
        ]
    elif backend == "openai":
        lines += [
            "  Detected OPENAI_API_KEY → backend set to 'openai'.",
            "  You're ready to go.",
            "",
            "  Execution limits are configured under [openai] in config.toml.",
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
