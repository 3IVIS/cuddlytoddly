"""
toddly.utils.config_utils
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Backend detection and tiered config-resolution helpers.

These are pure dict-in / value-out functions with no dependency on any
application-specific paths (DATA_DIR, CONFIG_PATH) or UI code.  The
cuddlytoddly.config module owns those paths and calls these helpers,
passing concrete Path objects wherever they are needed.
"""

from __future__ import annotations

import os
from pathlib import Path

from toddly.infra.logging import get_logger

logger = get_logger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

# Backends that use a remote API and can therefore afford higher limits.
_API_BACKENDS = {"claude", "openai"}

_VALID_BACKENDS = {"llamacpp", "claude", "openai", "file"}

# Approximate download sizes for known model size classes.
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


# ── Backend detection ─────────────────────────────────────────────────────────


def detect_backend() -> str:
    """Return the best default backend based on set environment variables."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "claude"
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    return "llamacpp"


# ── Validation ────────────────────────────────────────────────────────────────


def validate_config(cfg: dict, config_path: Path | None = None) -> None:
    """Raise ValueError if the config contains an unrecognised backend."""
    backend = cfg.get("llm", {}).get("backend", "")
    if backend not in _VALID_BACKENDS:
        hint = f"  Edit {config_path} to fix this." if config_path else ""
        raise ValueError(
            f"[CONFIG] llm.backend = {backend!r} is not valid.\n"
            f"Choose one of: {', '.join(sorted(_VALID_BACKENDS))}.\n"
            f"{hint}".rstrip()
        )


# ── Section accessors ─────────────────────────────────────────────────────────
# Lookup order for per-backend parameters:
#   1. cfg[active_backend][key]   — explicit per-backend value (wins)
#   2. cfg[shared_section][key]   — legacy shared-section value (compat)
#   3. hardcoded default          — last resort


def get_backend(cfg: dict) -> str:
    """Return the configured backend name, lower-cased."""
    return cfg.get("llm", {}).get("backend", "llamacpp").lower()


def is_api_backend(cfg: dict) -> bool:
    """Return True when the configured backend is a remote API provider."""
    return get_backend(cfg) in _API_BACKENDS


def _get(cfg: dict, key: str, shared_section: str, api_default, local_default):
    """
    Resolve a per-backend parameter with graceful fallback.

    Checks the active backend's config section first, then the shared
    section for backward compatibility with pre-restructured configs,
    then falls back to a hardcoded default chosen by backend tier.
    """
    backend = get_backend(cfg)
    val = cfg.get(backend, {}).get(key)
    if val is not None:
        return val
    val = cfg.get(shared_section, {}).get(key)
    if val is not None:
        return val
    return api_default if is_api_backend(cfg) else local_default


def get_executor_cfg(cfg: dict) -> dict:
    """Return executor parameters, resolved from the active backend's section."""
    return {
        "max_inline_result_chars": _get(cfg, "max_inline_result_chars", "executor", 12_000, 3_000),
        "max_total_input_chars": _get(cfg, "max_total_input_chars", "executor", 12_000, 3_000),
        "max_tool_result_chars": _get(cfg, "max_tool_result_chars", "executor", 8_000, 2_000),
        "max_history_entries": _get(cfg, "max_history_entries", "executor", 10, 3),
    }


def get_planner_cfg(cfg: dict) -> dict:
    """Return planner parameters, with max_tasks_per_goal resolved per-backend."""
    c = cfg.get("planner", {})
    return {
        "min_tasks_per_goal": c.get("min_tasks_per_goal", 3),
        "max_tasks_per_goal": _get(cfg, "max_tasks_per_goal", "planner", 15, 8),
        "scrutinize_plan": c.get("scrutinize_plan", True),
        "min_clarification_fields": c.get("min_clarification_fields", 2),
        "max_clarification_fields": c.get("max_clarification_fields", 6),
    }


def get_orchestrator_cfg(cfg: dict) -> dict:
    """Return orchestrator parameters, with backend-sensitive values resolved per-backend."""
    c = cfg.get("orchestrator", {})
    return {
        "max_workers": _get(cfg, "max_workers", "orchestrator", 4, 1),
        "max_successful_turns": _get(cfg, "max_successful_turns", "orchestrator", 10, 10),
        "max_unsuccessful_turns": _get(cfg, "max_unsuccessful_turns", "orchestrator", 10, 10),
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


def get_web_research_cfg(cfg: dict) -> dict:
    """Return the [web_research] section with defaults filled in."""
    c = cfg.get("web_research", {})
    return {
        "search_engine": c.get("search_engine", "duckduckgo"),
        "google_api_key": c.get("google_api_key", ""),
        "google_cx": c.get("google_cx", ""),
        "max_results": int(c.get("max_results", 5)),
    }


# ── Model resolution ──────────────────────────────────────────────────────────


def model_size_hint(filename: str) -> str:
    """Return a human-readable download-size hint for a known model filename."""
    upper = filename.upper()
    for tag, size in _MODEL_SIZES.items():
        if tag in upper:
            return size
    return ""


def llama_has_gpu_support() -> bool:
    """Return True when the installed llama-cpp-python was built with GPU support."""
    try:
        from llama_cpp import llama_supports_gpu_offload

        return bool(llama_supports_gpu_offload())
    except (ImportError, AttributeError):
        return False


def resolve_model_path(cfg: dict, data_dir: Path, config_path: Path | None = None) -> str:
    """
    Return the absolute path to the GGUF model file specified in *cfg*.

    Search order
    ------------
    1. ``CUDDLYTODDLY_MODEL_PATH`` env var
    2. ``LLAMA_CACHE`` / ``~/.cache/llama.cpp/``
    3. ``HF_HOME`` / ``~/.cache/huggingface/hub/``
    4. ``data_dir/models/<model_filename>``

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

    own_models = data_dir / "models"
    candidates.append(own_models / filename)

    for candidate in candidates:
        if candidate.exists():
            logger.info("[MODEL] Found model: %s", candidate)
            return str(candidate)

    size_hint = model_size_hint(filename)
    size_note = f"  (approx. {size_hint} download)\n" if size_hint else ""
    config_hint = (
        f'\nOption 3 — change the filename in the config:\n  {config_path}  →  [llamacpp] model_filename = "your-model.gguf"\n'
        if config_path
        else ""
    )

    raise FileNotFoundError(
        f"\nModel '{filename}' not found in any standard location.\n"
        f"{size_note}\n"
        f"Option 1 — download into the models folder:\n"
        f"  pip install huggingface-hub\n"
        f"  huggingface-cli download bartowski/Llama-3.3-70B-Instruct-GGUF"
        f" {filename} --local-dir {own_models}\n\n"
        f"Option 2 — point to an existing file:\n"
        f"  export CUDDLYTODDLY_MODEL_PATH=/path/to/{filename}\n"
        f"{config_hint}"
    )
