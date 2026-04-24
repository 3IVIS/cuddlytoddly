from __future__ import annotations

from toddly.planning.llm_backends_api import (  # noqa: F401
    ApiLLM,
)
from toddly.planning.llm_backends_local import (  # noqa: F401
    FileBasedLLM,
    LlamaCppLLM,
)

# planning/llm_interface.py
#
# Thin re-export facade — all existing imports continue to work unchanged.
#
# The implementation has been split into three focused modules:
#
#   llm_base.py            — shared primitives (BaseLLM, TokenCounter,
#                            LLMStoppedError, PromptCache, NativeToolResponse,
#                            module-level constants and id_gen singleton)
#   llm_backends_local.py  — FileBasedLLM, LlamaCppLLM
#   llm_backends_api.py    — ApiLLM
#
# Nothing in this file contains logic.  Import from the specific modules above
# when you only need a subset, or continue importing from here for full compat.
from toddly.planning.llm_base import (  # noqa: F401
    _API_MAX_ATTEMPTS,
    _API_RATE_LIMIT_INITIAL_BACKOFF,
    _API_RATE_LIMIT_MAX_BACKOFF,
    _DEFAULT_CLAUDE_MODEL,
    _DEFAULT_OPENAI_MODEL,
    _DEFAULT_POLL_INTERVAL,
    _DEFAULT_PROGRESS_LOG_INTERVAL,
    _DEFAULT_TIMEOUT,
    PROJECT_ROOT,
    PROMPT_LOG_FILE,
    RESPONSE_FILE,
    BaseLLM,
    LlamaCppCache,
    LLMStoppedError,
    NativeToolResponse,
    PromptCache,
    TokenCounter,
    id_gen,
    logger,
    token_counter,
)

# Re-export schemas so existing callers that imported them from here continue to work.
from toddly.planning.schemas import (  # noqa: F401
    DEPENDENCY_CHECK_SCHEMA,
    EVENT_LIST_SCHEMA,
    EXECUTION_TURN_SCHEMA,
    GOAL_SUMMARY_SCHEMA,
    PLAN_SCHEMA,
    REFINER_OUTPUT_SCHEMA,
    RESULT_VERIFICATION_SCHEMA,
)

# ---------------------------------------------------------------------------
# Factory — single entry point for the rest of the codebase
# ---------------------------------------------------------------------------


def create_llm_client(backend: str = "file", **kwargs) -> BaseLLM:
    backend = backend.lower()
    logger.info("[LLM FACTORY] Creating backend=%s", backend)

    if backend == "file":
        return FileBasedLLM(**kwargs)

    elif backend == "llamacpp":
        if "model_path" not in kwargs:
            raise ValueError(
                "llamacpp backend requires a 'model_path' keyword argument "
                "pointing to a .gguf file."
            )
        return LlamaCppLLM(**kwargs)

    elif backend in ("openai", "claude"):
        return ApiLLM(provider=backend, **kwargs)

    else:
        raise ValueError(
            f"Unknown backend '{backend}'. Valid options: 'file', 'llamacpp', 'openai', 'claude'."
        )
