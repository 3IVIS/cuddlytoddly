# planning/llm_interface.py
#
# Three interchangeable LLM backends, all sharing the same .ask() / .generate() interface.
#
# ┌─────────────────────────────────────────────────────────────┐
# │  Backend        │  How to select                           │
# ├─────────────────┼──────────────────────────────────────────┤
# │  FileBasedLLM   │  create_llm_client(backend="file")       │
# │  LlamaCppLLM    │  create_llm_client(backend="llamacpp")   │
# │  ApiLLM         │  create_llm_client(backend="openai")     │
# │                 │  create_llm_client(backend="claude")     │
# └─────────────────────────────────────────────────────────────┘
#
# All backends accept a `prompt: str` and return a `str` (raw JSON text).
# Structured output (outlines grammar) is applied inside LlamaCppLLM so the
# rest of the codebase never needs to change.
#
# Schemas live in planning/schemas.py.
# Prompt text lives in planning/prompts.py.

from __future__ import annotations

import hashlib
import json
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from dataclasses import field as _dc_field
from pathlib import Path
from typing import Any

from cuddlytoddly.core.id_generator import StableIDGenerator
from cuddlytoddly.infra.logging import get_logger

# System prompt text is owned by prompts.py.
from cuddlytoddly.planning.prompts import (
    EXECUTOR_NATIVE_SYSTEM_PROMPT,
    LLAMACPP_SYSTEM_PROMPT,
    LLM_SYSTEM_PROMPT,
)

# Re-export schemas so existing callers that imported them from here continue to work.
from cuddlytoddly.planning.schemas import (  # noqa: F401  (public re-exports)
    DEPENDENCY_CHECK_SCHEMA,
    EVENT_LIST_SCHEMA,
    EXECUTION_TURN_SCHEMA,
    GOAL_SUMMARY_SCHEMA,
    PLAN_SCHEMA,
    REFINER_OUTPUT_SCHEMA,
    RESULT_VERIFICATION_SCHEMA,
)

# ---------------------------------------------------------------------------
# FIX #14: Named constants for default model strings so that a single edit
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

# FIX #5: module-level id_gen is kept as an in-memory-only fallback for
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

    def __init__(self, token_counter_instance: "TokenCounter | None" = None):
        self._stop_event = threading.Event()
        # FIX #3: Each backend owns its counter so concurrent web-mode runs
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
    def ask(self, prompt: str) -> str:
        """Send a prompt, block until a response is available, return raw text."""

    def generate(self, prompt: str) -> str:
        """Alias for ask() — kept for backward compatibility."""
        return self.ask(prompt)


# ---------------------------------------------------------------------------
# Backend 1 — FileBasedLLM  (development / testing)
# ---------------------------------------------------------------------------


class FileBasedLLM(BaseLLM):
    """
    Simulates an LLM using text files with unique IDs.

    Workflow:
      1. Prompts are appended to llm_prompts.txt  (id:<uid>\\n<prompt>\\n)
      2. A human (or external process) writes responses to llm_responses.txt
         using the same id:<uid> prefix.
      3. get_response() polls until the matching block appears.

    All timing constants come from the application config (passed via __init__)
    so they can be adjusted without editing source code.

    When cache_path is provided, a cache hit skips the file-based poll loop
    entirely — useful for replaying a run without re-entering responses.

    FIX #5: accepts an optional ``id_gen`` parameter so each run uses its own
    StableIDGenerator rather than overwriting the module-level singleton.
    """

    def __init__(
        self,
        response_file: Path | str = RESPONSE_FILE,
        prompt_log_file: Path | str = PROMPT_LOG_FILE,
        poll_interval: float = _DEFAULT_POLL_INTERVAL,
        timeout: float = _DEFAULT_TIMEOUT,
        progress_log_interval: float = _DEFAULT_PROGRESS_LOG_INTERVAL,
        cache_path: Path | str | None = None,
        id_gen: StableIDGenerator | None = None,
        token_counter_instance: "TokenCounter | None" = None,
    ):
        super().__init__(token_counter_instance=token_counter_instance)
        self.response_file = Path(response_file)
        self.prompt_log_file = Path(prompt_log_file)
        self.poll_interval = poll_interval
        self.timeout = timeout
        self.progress_log_interval = progress_log_interval
        self._cache = PromptCache(cache_path) if cache_path is not None else None
        # FIX #5: use the caller-supplied generator so each run is isolated;
        # fall back to the module-level default only when none is provided.
        import cuddlytoddly.planning.llm_interface as _self_mod

        self._id_gen: StableIDGenerator = id_gen if id_gen is not None else _self_mod.id_gen
        logger.info("[LLM] Initialized FileBasedLLM")
        logger.info("[LLM] Prompt file path: %s", self.prompt_log_file.resolve())
        logger.info("[LLM] Response file path: %s", self.response_file.resolve())
        logger.info(
            "[LLM] Cache: %s",
            f"enabled ({len(self._cache)} entries)" if self._cache else "disabled",
        )

    def send_prompt(self, prompt: str) -> str:
        logger.info("[LLM] send_prompt() called")
        # FIX #5: use the per-instance generator instead of the module global
        prompt_id = self._id_gen.get_id(prompt, "prompts")
        logger.info("[LLM] Generated prompt_id=%s", prompt_id)

        entry = f"id:{prompt_id}\n{prompt}\n"

        if self.prompt_log_file.exists():
            logger.debug("[LLM] Prompt file exists, checking for duplicate id")
            with self.prompt_log_file.open("r") as f:
                for line in f:
                    if line.startswith("id:") and line[len("id:") :].strip() == prompt_id:
                        logger.warning(
                            "[LLM] Prompt id=%s already exists — skipping write",
                            prompt_id,
                        )
                        return prompt_id
        else:
            logger.info("[LLM] Prompt file does not exist — will create new one")

        try:
            with self.prompt_log_file.open("a") as f:
                f.write(entry)
            logger.info("[LLM] Prompt written (id=%s)", prompt_id)
        except Exception as e:
            logger.error("[LLM] Failed to write prompt id=%s: %s", prompt_id, e)
            raise

        return prompt_id

    def get_response(self, prompt_id: str) -> str:
        """
        Poll the response file for a block matching ``prompt_id``.

        FIX #3: Replaced the two-open approach (one incremental seek to detect
        new bytes, a second full re-read to parse blocks) with a single
        consistent read per poll.  The old design used last_offset to guard
        against redundant full-reads, but then immediately opened the file a
        second time — making the offset tracking useless and introducing a
        TOCTOU window where content written between the two opens could be
        silently skipped or double-counted.

        The new implementation reads the entire file once per poll and skips
        the parse step when the file size has not grown since the last read,
        preserving the cost-reduction goal without the inconsistency.
        """
        logger.info("[LLM] get_response() called for id=%s", prompt_id)
        start_time = time.time()
        last_progress_time = start_time
        last_known_size: int = 0

        while True:
            # Honour stop() calls that arrive while waiting for a response.
            if self._stop_event.is_set():
                raise LLMStoppedError(f"LLM calls paused while waiting for response id={prompt_id}")

            now = time.time()

            if self.response_file.exists():
                current_size = self.response_file.stat().st_size

                if current_size > last_known_size:
                    # File has grown — read it once atomically and parse.
                    last_known_size = current_size
                    with self.response_file.open("r", encoding="utf-8", errors="replace") as f:
                        lines_in_file = f.readlines()

                    current_id = None
                    block_lines: list[str] = []
                    for line in lines_in_file:
                        line = line.rstrip("\n")
                        if line.startswith("id:"):
                            if current_id == prompt_id and block_lines:
                                response_text = "\n".join(ln for ln in block_lines if ln.strip())
                                logger.info("[LLM] Response matched id=%s", prompt_id)
                                return response_text
                            current_id = line[len("id:") :].strip()
                            block_lines = []
                        else:
                            block_lines.append(line)

                    # last block
                    if current_id == prompt_id and block_lines:
                        response_text = "\n".join(ln for ln in block_lines if ln.strip())
                        logger.info("[LLM] Response matched id=%s (last block)", prompt_id)
                        return response_text
            else:
                logger.debug("[LLM] Response file does not yet exist")

            if now - last_progress_time > self.progress_log_interval:
                elapsed = int(now - start_time)
                logger.info(
                    "[LLM] Waiting for response (id=%s)... %ds elapsed",
                    prompt_id,
                    elapsed,
                )
                last_progress_time = now

            if now - start_time > self.timeout:
                logger.error("[LLM] Timeout waiting for response id=%s", prompt_id)
                raise TimeoutError(f"LLM response for id={prompt_id} not found within timeout")

            time.sleep(self.poll_interval)

    def ask(self, prompt: str) -> str:
        self._check_stop()
        logger.info("[LLM] ask() called")

        if self._cache is not None:
            cached = self._cache.get(prompt)
            if cached is not None:
                logger.info("[LLM] Cache HIT — skipping file poll")
                self._token_counter.add(len(prompt) // 4, len(cached) // 4)
                return cached

        prompt_id = self.send_prompt(prompt)
        logger.info("[LLM] ask() obtained prompt_id=%s", prompt_id)
        response = self.get_response(prompt_id)
        logger.info("[LLM] ask() completed for id=%s", prompt_id)
        self._token_counter.add(len(prompt) // 4, len(response) // 4)

        if self._cache is not None:
            self._cache.set(prompt, response)

        return response

    def clear_cache(self) -> None:
        if self._cache is not None:
            self._cache.clear()
            logger.info("[LLM] FileBasedLLM cache cleared")
        else:
            logger.info("[LLM] FileBasedLLM cache is disabled")


# ---------------------------------------------------------------------------
# FIX #13: Rename LlamaCppCache → PromptCache to reflect that it is now used
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
        # FIX #4: callers pass a snapshot taken while the lock was held so
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
        # FIX #4: take a snapshot of the store under the lock, then write to
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


# ---------------------------------------------------------------------------
# Backend 2 — LlamaCppLLM  (local model via llama-cpp-python + outlines)
# ---------------------------------------------------------------------------


class LlamaCppLLM(BaseLLM):
    """
    Runs a local GGUF model via llama-cpp-python.

    Parameters
    ----------
    model_path    : str | Path
    n_ctx         : int
    n_gpu_layers  : int
    temperature   : float
    max_tokens    : int
    schema        : dict | None   default schema stored but NOT used for generation
                                  unless passed explicitly to ask().
    cache_path    : str | Path | None
    """

    def __init__(
        self,
        model_path,
        n_ctx: int = 4096,
        n_gpu_layers: int = 0,
        temperature: float = 0.2,
        max_tokens: int = 2048,
        schema=None,
        cache_path: Path | str | None = PROJECT_ROOT / "llamacpp_cache.json",
        token_counter_instance: "TokenCounter | None" = None,
    ):
        super().__init__(token_counter_instance=token_counter_instance)
        self.model_path = str(model_path)
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.default_schema = schema or EVENT_LIST_SCHEMA

        logger.info("[LLAMACPP] Initializing LlamaCppLLM")
        logger.info("[LLAMACPP] Model path: %s", self.model_path)
        logger.info(
            "[LLAMACPP] n_ctx=%d  n_gpu_layers=%d  temperature=%.2f  max_tokens=%d",
            n_ctx,
            n_gpu_layers,
            temperature,
            max_tokens,
        )

        if cache_path is not None:
            self._cache = PromptCache(cache_path)
            logger.info(
                "[LLAMACPP] Prompt cache enabled -- %s (%d entries loaded)",
                Path(cache_path),
                len(self._cache),
            )
        else:
            self._cache = None
            logger.info("[LLAMACPP] Prompt cache disabled")

        self._llama = None
        self._outlines_model = None
        self._generators: dict = {}
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    # ── Model loading ─────────────────────────────────────────────────────────

    def _load_model(self):
        if self._llama is not None:
            return
        with self._load_lock:
            # Second check inside the lock — another thread may have loaded
            # the model between the outer check and acquiring the lock.
            if self._llama is not None:
                return
            try:
                from llama_cpp import Llama
            except ImportError as e:
                raise ImportError(
                    "llama-cpp-python is not installed. "
                    "Run: CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python "
                    "--force-reinstall --no-cache-dir"
                ) from e

            model_path = str(Path(self.model_path).expanduser().resolve())
            logger.info("[LLAMACPP] Loading model (first call -- may take 10-30s)...")
            self._llama = Llama(
                model_path=model_path,
                n_ctx=self.n_ctx,
                n_gpu_layers=self.n_gpu_layers,
                verbose=False,
            )
            logger.info("[LLAMACPP] Model loaded")

    def _load_outlines(self):
        if self._outlines_model is not None:
            return
        with self._load_lock:
            # Second check inside the lock — another thread may have loaded
            # the outlines model between the outer check and acquiring the lock.
            if self._outlines_model is not None:
                return
            try:
                import outlines
            except ImportError as e:
                raise ImportError("outlines is not installed. Run: pip install outlines") from e
            self._outlines_model = outlines.from_llamacpp(self._llama)
            logger.info("[LLAMACPP] Outlines model wrapper ready")

    def _load(self):
        self._load_model()

    # ── Constrained generator cache ───────────────────────────────────────────

    def _get_generator(self, schema: dict):
        import outlines

        fingerprint = json.dumps(schema, sort_keys=True)
        if fingerprint not in self._generators:
            self._load_outlines()
            logger.info(
                "[LLAMACPP] Building constrained generator for schema %s...",
                fingerprint[:40],
            )
            output_type = outlines.json_schema(fingerprint)
            self._generators[fingerprint] = outlines.Generator(self._outlines_model, output_type)
            logger.info("[LLAMACPP] Constrained generator ready")
        return self._generators[fingerprint]

    # ── Chat template ─────────────────────────────────────────────────────────

    def _apply_chat_template(self, prompt: str) -> str:
        # System prompt text is defined in prompts.py
        system = LLAMACPP_SYSTEM_PROMPT
        try:
            if self._llama.metadata.get("tokenizer.chat_template"):
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ]
                result = self._llama.tokenizer_.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                logger.debug("[LLAMACPP] Chat template applied via llama.cpp tokenizer")
                return result
        except Exception as e:
            logger.debug("[LLAMACPP] Built-in chat template unavailable (%s), using fallback", e)

        # Llama 3 hardcoded fallback
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}"
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    # ── Generation ────────────────────────────────────────────────────────────

    def _run_watchdog(self):
        done = threading.Event()

        def _watch():
            start = time.time()
            while not done.wait(timeout=30):
                logger.info("[LLAMACPP] Still generating... %.0fs elapsed", time.time() - start)

        t = threading.Thread(target=_watch, daemon=True, name="llm-watchdog")
        t.start()
        return done

    def _run_unconstrained(self, prompt: str, safe_max: int) -> str:
        logger.info("[LLAMACPP] Running unconstrained inference (max_tokens=%d)...", safe_max)
        result = self._llama(
            prompt,
            max_tokens=safe_max,
            temperature=self.temperature,
            echo=False,
        )
        return result["choices"][0]["text"]

    def _run_constrained(self, prompt: str, schema: dict, safe_max: int) -> str:
        logger.info("[LLAMACPP] Running constrained inference (max_tokens=%d)...", safe_max)
        generator = self._get_generator(schema)
        raw = generator(prompt, max_tokens=safe_max)
        if isinstance(raw, str):
            return raw
        return json.dumps(raw)

    def _run_model(self, prompt: str, constrained_schema=None) -> str:
        formatted = self._apply_chat_template(prompt)
        prompt_tokens = len(self._llama.tokenize(formatted.encode("utf-8")))

        safe_max = self.n_ctx - prompt_tokens - 64
        if safe_max <= 0:
            raise ValueError(
                f"Prompt is too long: {prompt_tokens} tokens leaves no room "
                f"in context window of {self.n_ctx}"
            )
        safe_max = min(self.max_tokens, safe_max)

        with self._inference_lock:
            done = self._run_watchdog()
            t0 = time.time()
            try:
                if constrained_schema is None:
                    raw = self._run_unconstrained(formatted, safe_max)
                else:
                    raw = self._run_constrained(formatted, constrained_schema, safe_max)
            finally:
                done.set()

        completion_tokens = len(self._llama.tokenize(raw.encode("utf-8")))
        self._token_counter.add(prompt_tokens, completion_tokens)

        logger.info(
            "[LLAMACPP] Inference complete in %.1fs -- %d chars",
            time.time() - t0,
            len(raw),
        )
        return raw

    # ── Truncation repair ─────────────────────────────────────────────────────

    def _repair_truncated_json(self, text: str):
        text = text.strip()
        if not text.startswith("["):
            return None
        pos = len(text) - 1
        while pos >= 0:
            pos = text.rfind("}", 0, pos + 1)
            if pos == -1:
                break
            candidate = text[: pos + 1].rstrip().rstrip(",") + "]"
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list) and len(parsed) > 0:
                    logger.warning(
                        "[LLAMACPP] Truncated output repaired: %d event(s) recovered "
                        "(max_tokens=%d)",
                        len(parsed),
                        self.max_tokens,
                    )
                    return candidate
            except json.JSONDecodeError:
                pass
            pos -= 1
        logger.error(
            "[LLAMACPP] Could not repair truncated output. Increase max_tokens (currently %d).",
            self.max_tokens,
        )
        return None

    # ── Public interface ──────────────────────────────────────────────────────

    def ask(self, prompt: str, schema: dict | None = None) -> str:
        self._check_stop()
        logger.info("[LLAMACPP] ask() called")

        if schema is None:
            cache_key = prompt
            constrained_schema = None
        else:
            cache_key = prompt + "\x00" + json.dumps(schema, sort_keys=True)
            constrained_schema = schema

        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.info("[LLAMACPP] Cache HIT")
                return cached

        self._load_model()

        for attempt in range(2):
            response_text = self._run_model(prompt, constrained_schema)
            try:
                parsed = json.loads(response_text)
                # FIX: mirror the ApiLLM fix — only reject a genuinely null
                # response.  The old check `not parsed and parsed != 0` treated
                # {} and [] as invalid because `not {}` and `not []` are True in
                # Python, causing unnecessary retries and ultimately raising
                # "Model repeatedly returned invalid JSON" for any task where
                # the correct response is an empty object or empty array.
                if parsed is None:
                    raise ValueError("Null JSON response")
                if self._cache is not None:
                    self._cache.set(cache_key, response_text)
                return response_text
            except Exception as e:
                logger.warning("[LLAMACPP] Invalid JSON on attempt %d: %s", attempt + 1, e)
                if attempt == 0:
                    repaired = self._repair_truncated_json(response_text)
                    if repaired is not None:
                        if self._cache is not None:
                            self._cache.set(cache_key, repaired)
                        return repaired
                    logger.warning("[LLAMACPP] Repair failed -- retrying full generation")

        raise ValueError("Model repeatedly returned invalid JSON")

    def clear_cache(self):
        if self._cache is not None:
            self._cache.clear()
            logger.info("[LLAMACPP] Cache cleared")
        else:
            logger.info("[LLAMACPP] Cache is disabled -- nothing to clear")


# ---------------------------------------------------------------------------
# Backend 3 — ApiLLM  (OpenAI-compatible or Anthropic API)
# ---------------------------------------------------------------------------


class ApiLLM(BaseLLM):
    """
    Calls a remote LLM API.  Supports:
      - OpenAI  (and any OpenAI-compatible endpoint, e.g. Together, Groq)
      - Anthropic Claude

    The system prompt text is defined in planning/prompts.py (LLM_SYSTEM_PROMPT)
    and can be overridden per-instance via the system_prompt parameter.

    Native tool use
    ---------------
    ask_with_tools() uses the provider's structured tool-use API (Claude's
    `tools=` parameter / OpenAI's `tools=` + `tool_choice="auto"`).  This is
    significantly more reliable than the legacy JSON-in-prompt approach because
    the model was fine-tuned on the native format, argument validation is
    handled by the provider, and there are no JSON parse errors on tool calls.
    """

    supports_native_tools: bool = True

    # FIX #14: reference the named constants so there is one canonical source.
    _DEFAULTS = {
        "openai": _DEFAULT_OPENAI_MODEL,
        "claude": _DEFAULT_CLAUDE_MODEL,
    }

    def __init__(
        self,
        provider: str = "openai",
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        system_prompt: str | None = None,
        cache_path: Path | str | None = None,
        token_counter_instance: "TokenCounter | None" = None,
    ):
        super().__init__(token_counter_instance=token_counter_instance)
        provider = provider.lower()
        if provider not in self._DEFAULTS:
            raise ValueError(f"Unknown provider '{provider}'. Choose 'openai' or 'claude'.")

        self.provider = provider
        self.api_key = api_key
        self.model = model or self._DEFAULTS[provider]
        self.base_url = base_url
        self.temperature = temperature
        self.max_tokens = max_tokens
        # System prompt text defaults to prompts.py; callers can still override.
        self.system_prompt = system_prompt or LLM_SYSTEM_PROMPT
        self._cache = PromptCache(cache_path) if cache_path is not None else None

        logger.info("[API] Initialized ApiLLM  provider=%s  model=%s", self.provider, self.model)
        if base_url:
            logger.info("[API] Using custom base_url: %s", base_url)
        logger.info(
            "[API] Cache: %s",
            f"enabled ({len(self._cache)} entries)" if self._cache else "disabled",
        )

        self._client = None  # lazy-loaded
        # FIX #1: protect lazy client initialisation with a lock so that
        # concurrent executor threads (max_workers > 1) cannot race through
        # the `if self._client is None` check and each create their own client.
        self._load_lock = threading.Lock()

    # ── Client loading ────────────────────────────────────────────────────────

    def _load(self):
        # Fast path — client already initialised (no lock needed after first load).
        if self._client is not None:
            return

        # FIX #1: double-checked locking — identical pattern to LlamaCppLLM.
        with self._load_lock:
            if self._client is not None:
                return

            if self.provider == "openai":
                try:
                    from openai import OpenAI
                except ImportError as e:
                    raise ImportError(
                        "openai package is not installed. Run: pip install openai"
                    ) from e
                kwargs: dict[str, Any] = {}
                if self.api_key:
                    kwargs["api_key"] = self.api_key
                if self.base_url:
                    kwargs["base_url"] = self.base_url
                self._client = OpenAI(**kwargs)
                logger.info("[API] OpenAI client ready")

            elif self.provider == "claude":
                try:
                    import anthropic
                except ImportError as e:
                    raise ImportError(
                        "anthropic package is not installed. Run: pip install anthropic"
                    ) from e
                kwargs = {}
                if self.api_key:
                    kwargs["api_key"] = self.api_key
                self._client = anthropic.Anthropic(**kwargs)
                logger.info("[API] Anthropic client ready")

    # ── Schema helpers ────────────────────────────────────────────────────────

    @staticmethod
    def _schema_root_type(schema: dict) -> str:
        return schema.get("type", "object")

    @staticmethod
    def _schema_prefill(schema: dict) -> str:
        return "[" if ApiLLM._schema_root_type(schema) == "array" else "{"

    @staticmethod
    def _inject_schema_into_prompt(prompt: str, schema: dict) -> str:
        schema_str = json.dumps(schema, indent=2)
        return (
            prompt
            + "\n\nYou MUST respond with JSON that strictly conforms to this schema:\n"
            + f"```json\n{schema_str}\n```\n"
            + "Respond with valid JSON only. No explanation, no markdown fences."
        )

    # ── OpenAI call ───────────────────────────────────────────────────────────

    def _ask_openai(self, prompt: str, schema: dict | None) -> str:
        if schema is not None:
            prompt_to_send = self._inject_schema_into_prompt(prompt, schema)
        else:
            prompt_to_send = prompt

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": prompt_to_send},
        ]

        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # response_format=json_object is an OpenAI-specific extension.
        # Many compatible providers (Groq, Together, Mistral, etc.) reject
        # it with a 400 error when base_url is set to their endpoint.
        # Only include it when hitting the real OpenAI API (no base_url override).
        if self.base_url is None:
            kwargs["response_format"] = {"type": "json_object"}

        logger.debug(
            "[API] Sending OpenAI request  model=%s  schema=%s  json_mode=%s",
            self.model,
            "yes" if schema else "no",
            "yes" if self.base_url is None else "no (custom base_url)",
        )
        response = self._client.chat.completions.create(**kwargs)

        if response.usage:
            self._token_counter.add(response.usage.prompt_tokens, response.usage.completion_tokens)
        content = response.choices[0].message.content or ""
        logger.info("[API] OpenAI response received (%d chars)", len(content))
        logger.debug("[API] Raw response:\n%s", content)
        return content

    # ── Claude call ───────────────────────────────────────────────────────────

    def _ask_claude(self, prompt: str, schema: dict | None) -> str:
        if schema is not None:
            augmented_prompt = self._inject_schema_into_prompt(prompt, schema)
            prefill = self._schema_prefill(schema)
        else:
            augmented_prompt = (
                prompt + "\n\nRespond with valid JSON only. "
                "No explanation, no markdown, no code fences."
            )
            prefill = "{"

        logger.debug("[API] Sending Anthropic request  model=%s  prefill=%r", self.model, prefill)
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=[
                {"role": "user", "content": augmented_prompt},
                {"role": "assistant", "content": prefill},
            ],
            temperature=self.temperature,
        )
        self._token_counter.add(response.usage.input_tokens, response.usage.output_tokens)
        raw = response.content[0].text
        content = prefill + raw
        logger.info("[API] Claude response received (%d chars)", len(content))
        logger.debug("[API] Raw response:\n%s", content)
        return content

    # ── Public interface ──────────────────────────────────────────────────────

    def ask(self, prompt: str, schema: dict | None = None) -> str:
        self._check_stop()
        logger.info("[API] ask() called  provider=%s  model=%s", self.provider, self.model)
        self._load()
        logger.debug("[API] Prompt (first 200 chars): %.200s", prompt)

        cache_key = (
            prompt if schema is None else prompt + "\x00" + json.dumps(schema, sort_keys=True)
        )
        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.info("[API] Cache HIT")
                return cached

        for attempt in range(2):
            try:
                if self.provider == "openai":
                    raw = self._ask_openai(prompt, schema)
                else:
                    raw = self._ask_claude(prompt, schema)
            except LLMStoppedError:
                raise
            except Exception as e:
                logger.error("[API] Request failed on attempt %d: %s", attempt + 1, e)
                if attempt == 0:
                    logger.warning("[API] Retrying after request error...")
                    continue
                raise

            try:
                parsed = json.loads(raw)
                # Fix #2: only reject a genuinely absent/null response, not a
                # legitimately empty JSON object {} or array [].  The previous
                # check `not parsed and parsed != 0` treated {} and [] as
                # invalid because `not {}` is True, causing unnecessary retries.
                if parsed is None:
                    raise ValueError("Null JSON response")
                if self._cache is not None:
                    self._cache.set(cache_key, raw)
                return raw
            except Exception as e:
                logger.warning(
                    "[API] Invalid JSON on attempt %d: %s  raw=%.200s",
                    attempt + 1,
                    e,
                    raw,
                )
                if attempt == 0:
                    logger.warning("[API] Retrying due to JSON parse failure...")
                    continue

        raise ValueError(f"[API] {self.provider} returned invalid JSON after 2 attempts")

    def generate(self, prompt: str) -> str:
        """
        Free-text completion — does NOT inject JSON instructions or an assistant
        prefill.

        BaseLLM.generate() simply delegates to ask(), which on the API backend
        always injects "Respond with valid JSON only…" and (for Claude) a '{'
        prefill.  That is wrong for callers that want unstructured plain-text
        output.  This override sends the prompt verbatim so the model can reply
        in natural language.
        """
        self._check_stop()
        self._load()
        logger.info("[API] generate() called  provider=%s  model=%s", self.provider, self.model)
        logger.debug("[API] generate() prompt (first 200 chars): %.200s", prompt)

        if self.provider == "openai":
            messages = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
            kwargs: dict[str, Any] = dict(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            response = self._client.chat.completions.create(**kwargs)
            if response.usage:
                self._token_counter.add(
                    response.usage.prompt_tokens, response.usage.completion_tokens
                )
            content = response.choices[0].message.content or ""
            logger.info("[API] generate() OpenAI response received (%d chars)", len(content))
            return content

        else:  # claude
            response = self._client.messages.create(
                model=self.model,
                max_tokens=self.max_tokens,
                system=self.system_prompt,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
            )
            self._token_counter.add(response.usage.input_tokens, response.usage.output_tokens)
            text = response.content[0].text
            logger.info("[API] generate() Claude response received (%d chars)", len(text))
            return text

    # ── Native tool-use API ───────────────────────────────────────────────────

    @staticmethod
    def _normalize_input_schema(schema: dict) -> dict:
        """
        Ensure a tool's input_schema is a valid JSON Schema object.

        Local skills use a simplified flat format: {"arg": "type_string"}.
        MCP tools already arrive as proper JSON Schema ({"type": "object", ...}).
        Both Claude and OpenAI require proper JSON Schema for their tools API.

        Detection rule:
          - Empty dict → return minimal schema.
          - Top-level key "type" present → already proper JSON Schema, pass through.
          - All values are strings → simplified format, promote to JSON Schema.
          - Mixed / already dict values → treat as proper JSON Schema, pass through.
        """
        if not schema:
            return {"type": "object", "properties": {}}

        # Already proper JSON Schema
        if "type" in schema:
            return schema

        # Check whether values look like simplified type strings
        if not all(isinstance(v, str) for v in schema.values()):
            return {"type": "object", "properties": schema}

        # Simplified {"arg_name": "type_string"} format
        properties: dict = {}
        required: list = []
        for key, type_str in schema.items():
            base_type = type_str.split()[0].rstrip(".,;")
            is_optional = "optional" in type_str.lower()
            if base_type not in (
                "string",
                "integer",
                "number",
                "boolean",
                "array",
                "object",
            ):
                base_type = "string"
            properties[key] = {"type": base_type}
            if not is_optional:
                required.append(key)

        result: dict = {"type": "object", "properties": properties}
        if required:
            result["required"] = required
        return result

    @staticmethod
    def _tools_to_anthropic(tools: list) -> list[dict]:
        """Convert a list of Tool objects to Anthropic's tool definition format."""
        return [
            {
                "name": t.name,
                "description": t.description or t.name,
                "input_schema": ApiLLM._normalize_input_schema(getattr(t, "input_schema", {})),
            }
            for t in tools
        ]

    @staticmethod
    def _tools_to_openai(tools: list) -> list[dict]:
        """Convert a list of Tool objects to OpenAI's tool definition format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or t.name,
                    "parameters": ApiLLM._normalize_input_schema(getattr(t, "input_schema", {})),
                },
            }
            for t in tools
        ]

    @staticmethod
    def _build_native_messages_claude(
        task_prompt: str,
        history: list[dict],
    ) -> list[dict]:
        """
        Build a Claude-format message list from the task prompt and tool history.

        Each history entry is one of:
          - Normal tool call: {name, args, result, tool_use_id}
            Produces alternating assistant (tool_use) + user (tool_result) pairs.
          - Correction: {kind: "correction", content: str}
            Fix #10: rendered as a plain user message so the executor can nudge
            the model (e.g. "you must call write_file") without fabricating a
            tool_use block with an ID the model never issued, which providers
            reject with a 400 error.
        """
        messages: list[dict] = [{"role": "user", "content": task_prompt}]
        for entry in history:
            if entry.get("kind") == "correction":
                messages.append({"role": "user", "content": entry["content"]})
                continue
            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": entry["tool_use_id"],
                            "name": entry["name"],
                            "input": entry["args"],
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": entry["tool_use_id"],
                            "content": str(entry["result"]),
                        }
                    ],
                }
            )
        return messages

    @staticmethod
    def _build_native_messages_openai(
        task_prompt: str,
        history: list[dict],
    ) -> list[dict]:
        """
        Build an OpenAI-format message list from the task prompt and tool history.

        Each history entry is one of:
          - Normal tool call: {name, args, result, tool_use_id}
            Produces alternating assistant (tool_calls) + tool (tool_result) pairs.
          - Correction: {kind: "correction", content: str}
            Fix #10: rendered as a plain user message so the executor can nudge
            the model without fabricating a tool_calls block with an ID the model
            never issued, which providers reject with a 400 error.
        """
        messages: list[dict] = [{"role": "user", "content": task_prompt}]
        for entry in history:
            if entry.get("kind") == "correction":
                messages.append({"role": "user", "content": entry["content"]})
                continue
            messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": entry["tool_use_id"],
                            "type": "function",
                            "function": {
                                "name": entry["name"],
                                "arguments": json.dumps(entry["args"]),
                            },
                        }
                    ],
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": entry["tool_use_id"],
                    "content": str(entry["result"]),
                }
            )
        return messages

    def ask_with_tools(
        self,
        task_prompt: str,
        tools: list,
        history: list[dict],
    ) -> NativeToolResponse:
        """
        Send a single executor turn using the provider's native tool-use API.

        Parameters
        ----------
        task_prompt : str
            The full task description prompt (built once by the executor).
        tools : list[Tool]
            Tool objects from the ToolRegistry.  Converted to provider format
            internally; callers never need to know the provider schema.
        history : list[dict]
            Tool-call history accumulated so far this execution.  Each entry:
              {"name": str, "args": dict, "result": str, "tool_use_id": str}

        Returns
        -------
        NativeToolResponse
            kind="tool_call"  — model wants to invoke a tool; executor should
                                run it and append the result to history.
            kind="text"       — model produced a final plain-text answer.
        """
        self._check_stop()
        self._load()
        logger.info(
            "[API] ask_with_tools()  provider=%s  model=%s  history_len=%d",
            self.provider,
            self.model,
            len(history),
        )

        if self.provider == "claude":
            return self._ask_with_tools_claude(task_prompt, tools, history)
        else:
            return self._ask_with_tools_openai(task_prompt, tools, history)

    def _ask_with_tools_claude(
        self,
        task_prompt: str,
        tools: list,
        history: list[dict],
    ) -> NativeToolResponse:
        messages = self._build_native_messages_claude(task_prompt, history)
        native_tools = self._tools_to_anthropic(tools)

        logger.debug(
            "[API/Claude] ask_with_tools: %d message(s), %d tool(s)",
            len(messages),
            len(native_tools),
        )

        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=EXECUTOR_NATIVE_SYSTEM_PROMPT,
            tools=native_tools,
            messages=messages,
            temperature=self.temperature,
        )
        self._token_counter.add(response.usage.input_tokens, response.usage.output_tokens)
        logger.info(
            "[API/Claude] ask_with_tools response: stop_reason=%s  content_blocks=%d",
            response.stop_reason,
            len(response.content),
        )

        if response.stop_reason == "tool_use":
            tool_block = next((b for b in response.content if b.type == "tool_use"), None)
            if tool_block is None:
                text = next((b.text for b in response.content if b.type == "text"), "")
                logger.warning("[API/Claude] stop_reason=tool_use but no tool_use block found")
                return NativeToolResponse(kind="text", text=text)

            logger.info(
                "[API/Claude] Tool call: %s  args=%s",
                tool_block.name,
                str(tool_block.input)[:120],
            )
            return NativeToolResponse(
                kind="tool_call",
                tool_name=tool_block.name,
                tool_args=dict(tool_block.input),
                tool_use_id=tool_block.id,
            )

        text = next((b.text for b in response.content if b.type == "text"), "")
        logger.info("[API/Claude] Final answer (%d chars)", len(text))
        return NativeToolResponse(kind="text", text=text)

    def _ask_with_tools_openai(
        self,
        task_prompt: str,
        tools: list,
        history: list[dict],
    ) -> NativeToolResponse:
        messages = [
            {"role": "system", "content": EXECUTOR_NATIVE_SYSTEM_PROMPT},
        ] + self._build_native_messages_openai(task_prompt, history)
        native_tools = self._tools_to_openai(tools)

        logger.debug(
            "[API/OpenAI] ask_with_tools: %d message(s), %d tool(s)",
            len(messages),
            len(native_tools),
        )

        response = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=native_tools,
            tool_choice="auto",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
        if response.usage:
            self._token_counter.add(response.usage.prompt_tokens, response.usage.completion_tokens)

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason
        logger.info("[API/OpenAI] ask_with_tools response: finish_reason=%s", finish_reason)

        if finish_reason == "tool_calls" and msg.tool_calls:
            tc = msg.tool_calls[0]
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                logger.warning(
                    "[API/OpenAI] Could not parse tool arguments: %s",
                    tc.function.arguments,
                )
                args = {}
            logger.info(
                "[API/OpenAI] Tool call: %s  args=%s",
                tc.function.name,
                str(args)[:120],
            )
            return NativeToolResponse(
                kind="tool_call",
                tool_name=tc.function.name,
                tool_args=args,
                tool_use_id=tc.id,
            )

        text = msg.content or ""
        logger.info("[API/OpenAI] Final answer (%d chars)", len(text))
        return NativeToolResponse(kind="text", text=text)

    def clear_cache(self) -> None:
        if self._cache is not None:
            self._cache.clear()
            logger.info("[API] Cache cleared")
        else:
            logger.info("[API] Cache is disabled")


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
