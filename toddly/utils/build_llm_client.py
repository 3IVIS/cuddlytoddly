"""
toddly.utils.build_llm_client
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Construct the correct BaseLLM instance from a loaded config dict.

All app-specific paths (data_dir) are passed as explicit parameters so
this module has no dependency on any particular host application.

Usage::

    from toddly.utils.build_llm_client import build_llm_client

    llm = build_llm_client(cfg, run_dir=run_dir, data_dir=data_dir)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from toddly.infra.logging import get_logger
from toddly.planning.llm_interface import (
    _DEFAULT_CLAUDE_MODEL,
    _DEFAULT_OPENAI_MODEL,
    TokenCounter,
    create_llm_client,
)
from toddly.utils.config_utils import get_file_llm_cfg, resolve_model_path

if TYPE_CHECKING:
    from toddly.core.id_generator import StableIDGenerator


def build_llm_client(
    cfg: dict,
    run_dir: Path,
    data_dir: Path,
    id_gen: "StableIDGenerator | None" = None,
    run_token_counter: "TokenCounter | None" = None,
):
    """
    Construct and return the correct BaseLLM from the loaded config.

    Parameters
    ----------
    cfg:
        Fully loaded and validated config dict (from ``load_config()``).
    run_dir:
        Per-run directory; cache files are written here so each run's
        cache is isolated from every other run.
    data_dir:
        Application data directory; used as the fallback search root when
        locating a local GGUF model file.
    id_gen:
        Per-run StableIDGenerator.  Forwarded to the file backend so each
        run owns its own generator instead of overwriting a shared global.
    run_token_counter:
        Per-run TokenCounter instance.  Pass ``None`` to fall back to the
        module-level global counter in llm_interface.
    """
    backend = cfg["llm"]["backend"]  # already validated by load_config()
    llm_cfg = cfg.get(backend, {})

    _logger = get_logger(__name__)
    _logger.info("[LLM] Backend: %s", backend)

    if backend == "llamacpp":
        model_path = resolve_model_path(cfg, data_dir)
        cache_path = (
            str(run_dir / "llamacpp_cache.json") if llm_cfg.get("cache_enabled", True) else None
        )
        return create_llm_client(
            "llamacpp",
            model_path=model_path,
            n_gpu_layers=llm_cfg.get("n_gpu_layers", -1),
            n_ctx=llm_cfg.get("n_ctx", 16384),
            max_tokens=llm_cfg.get("max_tokens", 8192),
            temperature=llm_cfg.get("temperature", 0.1),
            cache_path=cache_path,
            token_counter_instance=run_token_counter,
        )

    if backend == "claude":
        cache_path = str(run_dir / "api_cache.json") if llm_cfg.get("cache_enabled", True) else None
        return create_llm_client(
            "claude",
            model=llm_cfg.get("model", _DEFAULT_CLAUDE_MODEL),
            temperature=llm_cfg.get("temperature", 0.1),
            max_tokens=llm_cfg.get("max_tokens", 8192),
            cache_path=cache_path,
            token_counter_instance=run_token_counter,
        )

    if backend == "openai":
        cache_path = str(run_dir / "api_cache.json") if llm_cfg.get("cache_enabled", True) else None
        kwargs: dict = dict(
            model=llm_cfg.get("model", _DEFAULT_OPENAI_MODEL),
            temperature=llm_cfg.get("temperature", 0.1),
            max_tokens=llm_cfg.get("max_tokens", 8192),
            cache_path=cache_path,
        )
        if "base_url" in llm_cfg:
            kwargs["base_url"] = llm_cfg["base_url"]
        if "api_key" in llm_cfg:
            kwargs["api_key"] = llm_cfg["api_key"]
        kwargs["token_counter_instance"] = run_token_counter
        return create_llm_client("openai", **kwargs)

    if backend == "file":
        file_cfg = get_file_llm_cfg(cfg)
        cache_path = (
            str(run_dir / "file_llm_cache.json") if file_cfg.get("cache_enabled", True) else None
        )
        # Pass the per-run id_gen so the file backend never touches the
        # module-level singleton.
        return create_llm_client(
            "file",
            poll_interval=file_cfg["poll_interval"],
            timeout=file_cfg["timeout"],
            progress_log_interval=file_cfg["progress_log_interval"],
            cache_path=cache_path,
            id_gen=id_gen,
            token_counter_instance=run_token_counter,
        )

    # Should never reach here — validate_config() in load_config() guards this.
    raise ValueError(f"Unknown backend: {backend!r}")
