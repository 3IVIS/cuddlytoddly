from __future__ import annotations

# planning/llm_base.py
#
# Shared primitives used by all LLM backends:
#   - Module-level constants and paths
#   - NativeToolResponse dataclass
#   - TokenCounter
#   - LLMStoppedError
#   - BaseLLM abstract base class
#   - PromptCache (persistent disk-backed prompt → response cache)
#
# Import this module directly when you only need the types, not a backend.
import hashlib
import json
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from dataclasses import field as _dc_field
from pathlib import Path

from toddly.core.id_generator import StableIDGenerator
from toddly.infra.logging import get_logger

# ---------------------------------------------------------------------------
# Named constants for default model strings so that a single edit
# updates every reference — __main__.py, ApiLLM._DEFAULTS, and the config
# template all resolve from the same source of truth.
# ---------------------------------------------------------------------------

_DEFAULT_CLAUDE_MODEL: str = "claude-opus-4-6"
_DEFAULT_OPENAI_MODEL: str = "gpt-4o"

# ---------------------------------------------------------------------------
# Native tool-use response type
# ---------------------------------------------------------------------------


@dataclass
class NativeToolResponse:
    """
    Returned by ApiLLM.ask_with_tools() for each LLM turn.

    kind        : "text"      — model produced a final plain-text answer
                  "tool_call" — model wants to invoke a tool
    text        : final answer text (populated when kind="text")
    tool_name   : name of the tool to call (populated when kind="tool_call")
    tool_args   : dict of arguments for the tool call
    tool_use_id : provider-assigned ID that must be echoed back in the
                  tool_result message so the conversation stays valid
    """

    kind: str
    text: str = ""
    tool_name: str = ""
    tool_args: dict = _dc_field(default_factory=dict)
    tool_use_id: str = ""


# ---------------------------------------------------------------------------
# Token counter
# ---------------------------------------------------------------------------


class TokenCounter:
    """
    Thread-safe token counter.  One instance is created per LLM backend so
    that concurrent web-mode runs each track their own usage independently.
    A module-level fallback singleton (``token_counter``) is kept for
    backward compatibility with code that imports it directly.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._calls = 0

    def add(self, prompt: int, completion: int) -> None:
        with self._lock:
            self._prompt_tokens += prompt
            self._completion_tokens += completion
            self._calls += 1

    def seed(self, prompt: int, completion: int, calls: int = 0) -> None:
        """
        Set a baseline derived from a previously-persisted run so that the
        toolbar shows the correct historical total immediately after loading.

        This replaces whatever is currently in the counter; it must be called
        before any new LLM call is made (i.e. during startup, right after the
        graph is rebuilt from the event log).
        """
        with self._lock:
            self._prompt_tokens = prompt
            self._completion_tokens = completion
            self._calls = calls

    @property
    def prompt_tokens(self) -> int:
        return self._prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self._completion_tokens

    @property
    def total_tokens(self) -> int:
        return self._prompt_tokens + self._completion_tokens

    @property
    def calls(self) -> int:
        return self._calls

    def reset(self) -> None:
        with self._lock:
            self._prompt_tokens = self._completion_tokens = self._calls = 0


# Module-level singleton — import this wherever you need token counts.
token_counter = TokenCounter()
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPT_LOG_FILE = PROJECT_ROOT / "llm_prompts.txt"
RESPONSE_FILE = PROJECT_ROOT / "llm_responses.txt"

# Default values — overridden by config when passed to FileBasedLLM.__init__
_DEFAULT_POLL_INTERVAL = 0.5
_DEFAULT_TIMEOUT = 300
_DEFAULT_PROGRESS_LOG_INTERVAL = 2

# API retry / rate-limit backoff constants.
# Exposed as module-level names so they are easy to find and adjust without
# hunting for bare numbers inside ask().
_API_MAX_ATTEMPTS: int = 4  # total attempts before raising
_API_RATE_LIMIT_INITIAL_BACKOFF: int = 5  # seconds before first retry on 429
_API_RATE_LIMIT_MAX_BACKOFF: int = 60  # ceiling for exponential backoff

# Module-level id_gen is kept as an in-memory-only fallback for
# callers that don't pass an explicit id_gen.  It is NEVER replaced at
# runtime, so concurrent web-mode runs can't race on it.
id_gen: StableIDGenerator = StableIDGenerator(id_length=6)


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class LLMStoppedError(RuntimeError):
    """Raised when an LLM call is attempted while the stop flag is set."""


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class BaseLLM(ABC):
    """
    All backends implement this interface.
    Callers only ever use .ask() or .generate().

    Stop flag
    ---------
    Each instance owns a threading.Event (_stop_event).  When set, any call
    to ask() raises LLMStoppedError immediately — without touching the model
    or making any network call.

    Use stop() / resume() to set/clear the flag.
    The orchestrator calls these on all its clients via stop_llm_calls() /
    resume_llm_calls(), which the UI triggers with the 's' key.

    Native tool use
    ---------------
    supports_native_tools = True means the backend implements ask_with_tools()
    and the LLMExecutor will use the provider's structured tool-use API instead
    of the legacy JSON-in-prompt protocol.  Defaults False for all backends
    except ApiLLM (openai / claude).
    """

    supports_native_tools: bool = False
    # Set to True by backends that implement on_token / on_heartbeat streaming.
    # When False the executor does not pass those kwargs so legacy/stub LLMs
    # (e.g. FakeLLM in tests) keep working without changes.
    supports_streaming: bool = False

    def __init__(self, token_counter_instance: "TokenCounter | None" = None):
        self._stop_event = threading.Event()
        # Each backend owns its counter so concurrent web-mode runs
        # do not accumulate tokens into a shared module-level singleton.
        # Falls back to the module-level ``token_counter`` when none supplied
        # (e.g. third-party code constructing backends directly).
        self._token_counter: TokenCounter = (
            token_counter_instance if token_counter_instance is not None else token_counter
        )

    def stop(self) -> None:
        self._stop_event.set()
        logger.info("[LLM] Stop flag SET on %s", self.__class__.__name__)

    def resume(self) -> None:
        self._stop_event.clear()
        logger.info("[LLM] Stop flag CLEARED on %s", self.__class__.__name__)

    @property
    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def _check_stop(self) -> None:
        if self._stop_event.is_set():
            raise LLMStoppedError("LLM calls are paused — resume before retrying")

    @abstractmethod
    def ask(self, prompt: str, schema=None, *, on_token=None, on_heartbeat=None) -> str:
        """Send a prompt, block until a response is available, return raw text.

        Parameters
        ----------
        prompt        : The prompt text.
        schema        : Optional JSON schema; when supplied the backend must
                        return a JSON string conforming to it.
        on_token      : Optional ``(chunk: str) -> None`` callback fired for
                        each output token/chunk as it is generated.  Only
                        forwarded by the executor when ``supports_streaming``
                        is True; ignored silently by non-streaming backends.
        on_heartbeat  : Optional ``(elapsed: float) -> None`` callback fired
                        every ~2 s during inference.  Same gating as on_token.
        """

    def generate(self, prompt: str, *, on_token=None, on_heartbeat=None) -> str:
        """Alias for ask() without schema — kept for backward compatibility.

        Streaming kwargs are only forwarded when this backend sets
        supports_streaming = True, so subclasses that only define
        ask(self, prompt) continue to work without modification.
        """
        if self.supports_streaming:
            return self.ask(prompt, on_token=on_token, on_heartbeat=on_heartbeat)
        return self.ask(prompt)


# ---------------------------------------------------------------------------
# Rename LlamaCppCache → PromptCache to reflect that it is now used
# by all three backends.  A module-level alias preserves backward compat for
# any external code that already imports LlamaCppCache by name.
# ---------------------------------------------------------------------------


class PromptCache:
    """
    Persistent, disk-backed cache for prompt → response pairs.

    Originally written for LlamaCppLLM; now used by all backends (ApiLLM and
    FileBasedLLM included) so that identical prompts never hit the model or
    API twice within the same run — or across runs when the cache is reused.

    The cache is stored as a JSON file mapping SHA-256 prompt hashes to their
    responses. Both an in-memory dict and the JSON file are kept in sync so
    that:
      - Lookups within a single process are O(1) (no disk reads after load).
      - Results survive process restarts.
    """

    def __init__(self, cache_path: Path | str):
        self.cache_path = Path(cache_path)
        self._store: dict[str, str] = {}
        self._lock = threading.Lock()
        self._load()

    @staticmethod
    def _hash(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def _load(self) -> None:
        if not self.cache_path.exists():
            logger.info("[CACHE] No cache file found — starting empty")
            return
        try:
            with self.cache_path.open("r", encoding="utf-8") as f:
                self._store = json.load(f)
            if not isinstance(self._store, dict):
                raise ValueError("Cache root must be dict")
            logger.info(
                "[CACHE] Loaded %d cached entries from %s",
                len(self._store),
                self.cache_path,
            )
        except Exception as e:
            logger.error("[CACHE] Corrupted cache file detected: %s", e)
            backup = self.cache_path.with_suffix(".corrupt.json")
            try:
                self.cache_path.rename(backup)
                logger.warning("[CACHE] Corrupted cache backed up to %s", backup)
            except OSError:
                logger.warning("[CACHE] Could not backup corrupted cache")
            self._store = {}

    def _save(self, store_snapshot: "dict | None" = None) -> None:
        # Callers pass a snapshot taken while the lock was held so
        # this method can do disk I/O without holding the lock at all.
        # When called internally (e.g. _load -> corruption recovery) with no
        # snapshot, fall back to reading self._store directly; that call site
        # is always single-threaded so there is no contention risk.
        data = store_snapshot if store_snapshot is not None else self._store
        tmp = self.cache_path.with_suffix(".tmp")
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            tmp.replace(self.cache_path)
        except OSError as e:
            logger.error("[CACHE] Failed to write cache file: %s", e)
            tmp.unlink(missing_ok=True)

    def get(self, prompt: str) -> str | None:
        with self._lock:
            entry = self._store.get(self._hash(prompt))
        if entry is None:
            return None
        return entry["response"] if isinstance(entry, dict) else entry

    def set(self, prompt: str, response: str) -> None:
        key = self._hash(prompt)
        # Take a snapshot of the store under the lock, then write to
        # disk outside it.  Holding the lock during json.dump + file rename
        # blocks every concurrent get() call for the full duration of I/O.
        with self._lock:
            self._store[key] = {"prompt": prompt, "response": response}
            store_snapshot = dict(self._store)
        self._save(store_snapshot)
        logger.info("[CACHE] Stored new entry (hash=%s…)", key[:12])

    def __len__(self) -> int:
        with self._lock:
            return len(self._store)

    def clear(self) -> None:
        with self._lock:
            self._store = {}
            store_snapshot: dict = {}
        self._save(store_snapshot)
        logger.info("[CACHE] Cache cleared")


# Backward-compat alias — external code using the old name continues to work.
LlamaCppCache = PromptCache
