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

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import threading

from cuddlytoddly.core.id_generator import StableIDGenerator
from cuddlytoddly.infra.logging import get_logger
import threading

class TokenCounter:
    """
    Module-level singleton tracking tokens consumed across all LLM calls
    in this process.  Thread-safe; all attributes are read-only properties.
    """
    def __init__(self):
        self._lock          = threading.Lock()
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._calls         = 0

    def add(self, prompt: int, completion: int) -> None:
        with self._lock:
            self._prompt_tokens     += prompt
            self._completion_tokens += completion
            self._calls             += 1

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


# Module-level singleton — import this wherever you need token counts
token_counter = TokenCounter()
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Shared constants (kept from original FileBasedLLM)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPT_LOG_FILE = PROJECT_ROOT / "llm_prompts.txt"
RESPONSE_FILE = PROJECT_ROOT / "llm_responses.txt"
POLL_INTERVAL = 0.5
TIMEOUT = 300
PROGRESS_LOG_INTERVAL = 2

id_gen = StableIDGenerator(id_length=6)


# ---------------------------------------------------------------------------
# JSON Schema — shared by all backends that support structured output.
# Describes the list-of-events format the planner/reflector expect.
# ---------------------------------------------------------------------------

# Add alongside EVENT_LIST_SCHEMA and REFINER_OUTPUT_SCHEMA

GOAL_SUMMARY_SCHEMA = {
    "type": "object",
    "required": ["description", "plan_summary"],
    "additionalProperties": False,
    "properties": {
        "description": {
            "type": "string",
            "description": (
                "One sentence (max 20 words) naming what this goal achieves. "
                "Used as the node label in the UI."
            ),
        },
        "plan_summary": {
            "type": "string",
            "description": (
                "2-4 sentences explaining how the planned tasks combine to "
                "achieve the goal. Cover what each task produces and how the "
                "outputs chain together into the final result."
            ),
        },
    },
}


_IO_ITEM = {
    "type": "object",
    "required": ["name", "type", "description"],
    "additionalProperties": False,
    "properties": {
        "name":        {"type": "string",
                        "description": "Short snake_case identifier, e.g. 'investment_report'"},
        "type":        {"type": "string",
                        "enum": ["file", "document", "data", "list", "url", "text", "json", "code"],
                        "description": "What kind of artifact this is"},
        "description": {"type": "string",
                        "description": "One sentence: what this artifact contains"},
    }
}

EVENT_LIST_SCHEMA = {
    "type": "array",
    "items": {
        "oneOf": [
            {
                "type": "object",
                "title": "ADD_NODE event",
                "required": ["type", "payload"],
                "additionalProperties": False,
                "properties": {
                    "type": {"type": "string", "const": "ADD_NODE"},
                    "payload": {
                        "type": "object",
                        "required": ["node_id", "node_type", "dependencies", "metadata"],
                        "additionalProperties": False,
                        "properties": {
                            "node_id":      {"type": "string"},
                            "node_type":    {"type": "string", "enum": ["task", "goal", "reflection"]},
                            "dependencies": {"type": "array", "items": {"type": "string"}},
                            "metadata": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "description":      {"type": "string"},
                                    "parallel_group":   {"type": ["string", "null"]},
                                    "required_input":   {"type": "array", "items": _IO_ITEM},
                                    "output":           {"type": "array", "items": _IO_ITEM},
                                    "reflection_notes": {"type": "array", "items": {"type": "string"}},
                                    "precedes":         {"type": "array", "items": {"type": "string"}}
                                }
                            }
                        }
                    }
                }
            },
            {
                "type": "object",
                "title": "ADD_DEPENDENCY event",
                "required": ["type", "payload"],
                "additionalProperties": False,
                "properties": {
                    "type": {"type": "string", "const": "ADD_DEPENDENCY"},
                    "payload": {
                        "type": "object",
                        "required": ["node_id", "depends_on"],
                        "additionalProperties": False,
                        "properties": {
                            "node_id":       {"type": "string"},
                            "depends_on": {"type": "string"}
                        }
                    }
                }
            }
        ]
    }
}

PLAN_SCHEMA = {
    "type": "object",
    "required": ["a_goal_result", "events"],
    "additionalProperties": False,
    "properties": {
        "a_goal_result": {
            "type": "string",
            "description": (
                "2-4 sentences explaining how these specific tasks chain together "
                "to achieve the goal. Name each task, what it produces, and why "
                "the next task depends on that output. Make the dependency "
                "reasoning explicit."
            ),
        },
        "events": {
            "type": "array",
            "items": EVENT_LIST_SCHEMA["items"],  # reuses existing item definitions
        },
    },
}


# ---------------------------------------------------------------------------
# JSON Schema — used by LLMRefiner.
# The refiner returns a single object, not an array.
# ---------------------------------------------------------------------------

REFINER_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["needs_refinement", "tasks_to_expand", "validated_atomic", "dependency_issues", "reasoning"],
    "additionalProperties": False,
    "properties": {
        "needs_refinement": {
            "type": "boolean"
        },
        "tasks_to_expand": {
            "type": "array",
            "items": {"type": "string"}
        },
        "validated_atomic": {
            "type": "array",
            "items": {"type": "string"}
        },
        "dependency_issues": {
            "type": "array",
            "items": {"type": "string"}
        },
        "reasoning": {
            "type": "string"
        }
    }
}

# -----------------------------------------------------------------------------
# 1. New exception  — add near the top of llm_interface.py, after imports
# -----------------------------------------------------------------------------

class LLMStoppedError(RuntimeError):
    """Raised when an LLM call is attempted while the stop flag is set."""


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 2. BaseLLM  — replace existing BaseLLM class with this
# -----------------------------------------------------------------------------

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
    """

    def __init__(self):
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Set the stop flag — subsequent ask() calls raise LLMStoppedError."""
        self._stop_event.set()
        logger.info("[LLM] Stop flag SET on %s", self.__class__.__name__)

    def resume(self) -> None:
        """Clear the stop flag — ask() calls proceed normally again."""
        self._stop_event.clear()
        logger.info("[LLM] Stop flag CLEARED on %s", self.__class__.__name__)

    @property
    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def _check_stop(self) -> None:
        """Call at the top of ask() in every backend."""
        if self._stop_event.is_set():
            raise LLMStoppedError("LLM calls are paused — resume before retrying")

    @abstractmethod
    def ask(self, prompt: str) -> str:
        """Send a prompt, block until a response is available, return raw text."""

    def generate(self, prompt: str) -> str:
        """Alias for ask() — kept for backward compatibility."""
        return self.ask(prompt)



# ---------------------------------------------------------------------------
# Backend 1 — FileBasedLLM  (original implementation, fully preserved)
# ---------------------------------------------------------------------------

class FileBasedLLM(BaseLLM):
    """
    Simulates an LLM using text files with unique IDs.

    Workflow:
      1. Prompts are appended to llm_prompts.txt  (id:<uid>\n<prompt>\n)
      2. A human (or external process) writes responses to llm_responses.txt
         using the same id:<uid> prefix.
      3. get_response() polls until the matching block appears.
    """

    def __init__(
        self,
        response_file: Path | str = RESPONSE_FILE,
        prompt_log_file: Path | str = PROMPT_LOG_FILE,
    ):
        super().__init__()
        self.response_file = Path(response_file)
        self.prompt_log_file = Path(prompt_log_file)
        logger.info("[LLM] Initialized FileBasedLLM")
        logger.info("[LLM] Prompt file path: %s", self.prompt_log_file.resolve())
        logger.info("[LLM] Response file path: %s", self.response_file.resolve())

    # --------------------------------------------------
    def send_prompt(self, prompt: str) -> str:
        logger.info("[LLM] send_prompt() called")
        prompt_id = id_gen.get_id(prompt, "prompts")
        logger.info("[LLM] Generated prompt_id=%s", prompt_id)

        entry = f"id:{prompt_id}\n{prompt}\n"

        if self.prompt_log_file.exists():
            logger.debug("[LLM] Prompt file exists, checking for duplicate id")
            with self.prompt_log_file.open("r") as f:
                for line in f:
                    if line.startswith("id:") and line[len("id:"):].strip() == prompt_id:
                        logger.warning(
                            "[LLM] Prompt id=%s already exists — skipping write", prompt_id
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

    # --------------------------------------------------
    def get_response(self, prompt_id: str) -> str:
        logger.info("[LLM] get_response() called for id=%s", prompt_id)
        start_time = time.time()
        last_progress_time = start_time

        while True:
            now = time.time()

            if self.response_file.exists():
                with self.response_file.open() as f:
                    lines = f.readlines()

                current_id = None
                block_lines = []
                for line in lines:
                    line = line.rstrip("\n")
                    if line.startswith("id:"):
                        if current_id == prompt_id and block_lines:
                            response_text = "\n".join([l for l in block_lines if l.strip()])
                            logger.info("[LLM] Response matched id=%s", prompt_id)
                            logger.debug("[LLM] Response content:\n%s", response_text)
                            return response_text
                        current_id = line[len("id:"):].strip()
                        block_lines = []
                    else:
                        block_lines.append(line)

                # last block
                if current_id == prompt_id and block_lines:
                    response_text = "\n".join([l for l in block_lines if l.strip()])
                    logger.info("[LLM] Response matched id=%s (last block)", prompt_id)
                    logger.debug("[LLM] Response content:\n%s", response_text)
                    return response_text
            else:
                logger.debug("[LLM] Response file does not yet exist")

            if now - last_progress_time > PROGRESS_LOG_INTERVAL:
                elapsed = int(now - start_time)
                logger.info(
                    "[LLM] Waiting for response (id=%s)... %ds elapsed", prompt_id, elapsed
                )
                last_progress_time = now

            if now - start_time > TIMEOUT:
                logger.error("[LLM] Timeout waiting for response id=%s", prompt_id)
                raise TimeoutError(
                    f"LLM response for id={prompt_id} not found within timeout"
                )

            time.sleep(POLL_INTERVAL)

    # --------------------------------------------------
    def ask(self, prompt: str) -> str:          # FileBasedLLM
        self._check_stop()                      # ← ADD THIS LINE
        logger.info("[LLM] ask() called")
        prompt_id = self.send_prompt(prompt)
        logger.info("[LLM] ask() obtained prompt_id=%s", prompt_id)
        response = self.get_response(prompt_id)
        logger.info("[LLM] ask() completed for id=%s", prompt_id)
        token_counter.add(len(prompt) // 4, len(response) // 4)

        return response


# ---------------------------------------------------------------------------
# Prompt-response cache for LlamaCppLLM
# ---------------------------------------------------------------------------

class LlamaCppCache:
    """
    Persistent, disk-backed cache for LlamaCppLLM prompt/response pairs.

    The cache is stored as a JSON file mapping SHA-256 prompt hashes to their
    responses. Both an in-memory dict and the JSON file are kept in sync so
    that:
      - Lookups within a single process are O(1) (no disk reads after load).
      - Results survive process restarts.

    Parameters
    ----------
    cache_path : Path | str
        Path to the JSON cache file. Created automatically if absent.
    """

    def __init__(self, cache_path: Path | str):
        self.cache_path = Path(cache_path)
        self._store: dict[str, str] = {}
        self._load()

    # --------------------------------------------------
    @staticmethod
    def _hash(prompt: str) -> str:
        """Return a stable SHA-256 hex digest of the prompt string."""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    # --------------------------------------------------
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
                len(self._store), self.cache_path,
            )

        except Exception as e:
            logger.error("[CACHE] Corrupted cache file detected: %s", e)

            # Backup corrupted file
            backup = self.cache_path.with_suffix(".corrupt.json")
            try:
                self.cache_path.rename(backup)
                logger.warning("[CACHE] Corrupted cache backed up to %s", backup)
            except OSError:
                logger.warning("[CACHE] Could not backup corrupted cache")

            self._store = {}

    # --------------------------------------------------
    def _save(self) -> None:
        """Persist the in-memory store to disk atomically via a temp file."""
        tmp = self.cache_path.with_suffix(".tmp")
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(self._store, f, indent=2)
            tmp.replace(self.cache_path)
        except OSError as e:
            logger.error("[CACHE] Failed to write cache file: %s", e)
            tmp.unlink(missing_ok=True)

    # --------------------------------------------------
    def get(self, prompt: str) -> str | None:
        entry = self._store.get(self._hash(prompt))
        if entry is None:
            return None
        # handle both old format (bare string) and new format (dict)
        return entry["response"] if isinstance(entry, dict) else entry

    # --------------------------------------------------
    def set(self, prompt: str, response: str) -> None:
        key = self._hash(prompt)
        self._store[key] = {"prompt": prompt, "response": response}
        self._save()
        logger.info("[CACHE] Stored new entry (hash=%s…)", key[:12])

    # --------------------------------------------------
    def __len__(self) -> int:
        return len(self._store)

    # --------------------------------------------------
    def clear(self) -> None:
        """Wipe all cached entries from memory and disk."""
        self._store = {}
        self._save()
        logger.info("[CACHE] Cache cleared")


# ---------------------------------------------------------------------------
# Backend 2 — LlamaCppLLM  (local model via llama-cpp-python + outlines)
# ---------------------------------------------------------------------------

# These must already exist in llm_interface.py — referenced here for clarity
# from cuddlytoddly.planning.llm_interface import (
#     BaseLLM, LlamaCppCache, EVENT_LIST_SCHEMA, PROJECT_ROOT, logger
# )

# planning/llm_interface_llamacpp.py
#
# Drop-in replacement for the LlamaCppLLM class in planning/llm_interface.py.
#
# Key design:
#   ask(prompt)               -> unconstrained generation (fast, ~10-30s)
#                                Used by the planner. JSON repair handles any
#                                malformed output. Matches how the 138 cached
#                                entries were originally generated.
#
#   ask(prompt, schema=X)     -> outlines-constrained generation (slower but
#                                guaranteed-valid JSON).
#                                Used by the executor (EXECUTION_TURN_SCHEMA).
#
# Cache keys:
#   planner  -> prompt string only       (backward-compatible with existing cache)
#   executor -> prompt + schema fingerprint  (no collision with planner entries)

import json
import threading
import time
from pathlib import Path

# These must already exist in llm_interface.py — referenced here for clarity
# from cuddlytoddly.planning.llm_interface import (
#     BaseLLM, LlamaCppCache, EVENT_LIST_SCHEMA, PROJECT_ROOT, logger
# )

# planning/llm_interface_llamacpp.py
#
# Drop-in replacement for the LlamaCppLLM class in planning/llm_interface.py.
#
# Key design:
#   ask(prompt)               -> unconstrained generation (fast, ~10-30s)
#                                Used by the planner. JSON repair handles any
#                                malformed output. Matches how the 138 cached
#                                entries were originally generated.
#
#   ask(prompt, schema=X)     -> outlines-constrained generation (slower but
#                                guaranteed-valid JSON).
#                                Used by the executor (EXECUTION_TURN_SCHEMA).
#
# Cache keys:
#   planner  -> prompt string only       (backward-compatible with existing cache)
#   executor -> prompt + schema fingerprint  (no collision with planner entries)

import json
import threading
import time
from pathlib import Path

# These must already exist in llm_interface.py — referenced here for clarity
# from cuddlytoddly.planning.llm_interface import (
#     BaseLLM, LlamaCppCache, EVENT_LIST_SCHEMA, PROJECT_ROOT, logger
# )


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
        n_ctx=4096,
        n_gpu_layers=0,
        temperature=0.2,
        max_tokens=2048,
        schema=None,
        cache_path=PROJECT_ROOT / "llamacpp_cache.json",
    ):
        super().__init__()
        self.model_path     = str(model_path)
        self.n_ctx          = n_ctx
        self.n_gpu_layers   = n_gpu_layers
        self.temperature    = temperature
        self.max_tokens     = max_tokens
        self.default_schema = schema or EVENT_LIST_SCHEMA

        logger.info("[LLAMACPP] Initializing LlamaCppLLM")
        logger.info("[LLAMACPP] Model path: %s", self.model_path)
        logger.info("[LLAMACPP] n_ctx=%d  n_gpu_layers=%d  temperature=%.2f  max_tokens=%d",
                    n_ctx, n_gpu_layers, temperature, max_tokens)

        if cache_path is not None:
            self._cache = LlamaCppCache(cache_path)
            logger.info("[LLAMACPP] Prompt cache enabled -- %s (%d entries loaded)",
                        Path(cache_path), len(self._cache))
        else:
            self._cache = None
            logger.info("[LLAMACPP] Prompt cache disabled")

        self._llama          = None   # llama_cpp.Llama -- loaded once
        self._outlines_model = None   # outlines wrapper -- only built if needed
        self._generators     = {}     # schema fingerprint -> outlines.Generator
        self._load_lock      = __import__("threading").Lock()  # prevents double-load
        # llama.cpp is NOT thread-safe -- all inference must be serialised
        self._inference_lock = threading.Lock()

    # -------------------------------------------------------------------------
    # Model loading
    # -------------------------------------------------------------------------

    def _load_model(self):
        """Load the Llama model. Called once on first use. Thread-safe."""
        if self._llama is not None:
            return
        with self._load_lock:
            if self._llama is not None:  # double-checked locking
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
        """Build the outlines model wrapper. Thread-safe."""
        if self._outlines_model is not None:
            return
        with self._load_lock:
            if self._outlines_model is not None:
                return

        try:
            import outlines
        except ImportError as e:
            raise ImportError(
                "outlines is not installed. Run: pip install outlines"
            ) from e

        self._outlines_model = outlines.from_llamacpp(self._llama)
        logger.info("[LLAMACPP] Outlines model wrapper ready")

    # Keep old name as alias
    def _load(self):
        self._load_model()

    # -------------------------------------------------------------------------
    # Constrained generator cache
    # -------------------------------------------------------------------------

    def _get_generator(self, schema: dict):
        """Return a cached outlines Generator for schema (build on first use)."""
        import outlines
        fingerprint = json.dumps(schema, sort_keys=True)
        if fingerprint not in self._generators:
            self._load_outlines()
            logger.info("[LLAMACPP] Building constrained generator for schema %s...",
                        fingerprint[:40])
            output_type = outlines.json_schema(fingerprint)
            self._generators[fingerprint] = outlines.Generator(
                self._outlines_model, output_type
            )
            logger.info("[LLAMACPP] Constrained generator ready")
        return self._generators[fingerprint]

    # -------------------------------------------------------------------------
    # Chat template
    # -------------------------------------------------------------------------

    def _apply_chat_template(self, prompt: str) -> str:
        system = (
            "You are a DAG planning assistant. "
            "Always respond with a valid JSON array and nothing else. "
            "No explanation, no markdown, no code fences."
        )
        try:
            if self._llama.metadata.get("tokenizer.chat_template"):
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ]
                result = self._llama.tokenizer_.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
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

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------

    def _run_watchdog(self):
        """Return a (done_event, thread) watchdog that logs every 30s."""
        done = threading.Event()
        def _watch():
            start = time.time()
            while not done.wait(timeout=30):
                logger.info("[LLAMACPP] Still generating... %.0fs elapsed",
                            time.time() - start)
        t = threading.Thread(target=_watch, daemon=True, name="llm-watchdog")
        t.start()
        return done

    def _run_unconstrained(self, prompt: str, safe_max: int) -> str:
        """
        Fast path: raw llama.cpp generation with no grammar constraint.
        Used by the planner. ~10-30x faster than outlines on large schemas.
        """
        logger.info("[LLAMACPP] Running unconstrained inference (max_tokens=%d)...", safe_max)
        result = self._llama(
            prompt,
            max_tokens=safe_max,
            temperature=self.temperature,
            echo=False,
        )
        return result["choices"][0]["text"]

    def _run_constrained(self, prompt: str, schema: dict, safe_max: int) -> str:
        """
        Constrained path: outlines grammar enforcement.
        Used by the executor for guaranteed-valid JSON.
        """
        logger.info("[LLAMACPP] Running constrained inference (max_tokens=%d)...", safe_max)
        generator = self._get_generator(schema)
        raw = generator(prompt, max_tokens=safe_max)
        if isinstance(raw, str):
            return raw
        return json.dumps(raw)

    def _run_model(self, prompt: str, constrained_schema=None) -> str:
        """
        Run inference, with or without schema constraint.
        constrained_schema=None -> fast unconstrained (planner)
        constrained_schema=dict -> outlines constrained (executor)

        Serialised via _inference_lock: llama.cpp is not thread-safe.
        Parallel executor nodes will queue here and run one at a time.
        """
        formatted     = self._apply_chat_template(prompt)
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
            t0   = time.time()
            try:
                if constrained_schema is None:
                    raw = self._run_unconstrained(formatted, safe_max)
                else:
                    raw = self._run_constrained(formatted, constrained_schema, safe_max)
            finally:
                done.set()

        completion_tokens = len(self._llama.tokenize(raw.encode("utf-8")))
        token_counter.add(prompt_tokens, completion_tokens)

        logger.info("[LLAMACPP] Inference complete in %.1fs -- %d chars",
                    time.time() - t0, len(raw))
        return raw

    # -------------------------------------------------------------------------
    # Truncation repair
    # -------------------------------------------------------------------------

    def _repair_truncated_json(self, text: str):
        text = text.strip()
        if not text.startswith("["):
            return None
        pos = len(text) - 1
        while pos >= 0:
            pos = text.rfind("}", 0, pos + 1)
            if pos == -1:
                break
            candidate = text[:pos + 1].rstrip().rstrip(",") + "]"
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list) and len(parsed) > 0:
                    logger.warning(
                        "[LLAMACPP] Truncated output repaired: %d event(s) recovered "
                        "(max_tokens=%d)", len(parsed), self.max_tokens,
                    )
                    return candidate
            except json.JSONDecodeError:
                pass
            pos -= 1
        logger.error("[LLAMACPP] Could not repair truncated output. "
                     "Increase max_tokens (currently %d).", self.max_tokens)
        return None

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def ask(self, prompt: str, schema: dict | None = None) -> str:
        """
        Generate a response.

        schema=None  -> unconstrained generation (fast, used by planner)
        schema=dict  -> constrained generation   (slower, used by executor)
        """
        self._check_stop()
        logger.info("[LLAMACPP] ask() called")

        # Cache key: prompt-only for planner (backward-compatible with 138
        # existing entries); schema-namespaced for executor.
        if schema is None:
            cache_key        = prompt
            constrained_schema = None
        else:
            cache_key        = prompt + "\x00" + json.dumps(schema, sort_keys=True)
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
                if not parsed:
                    raise ValueError("Empty JSON response")
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

# Replacement for ApiLLM in planning/llm_interface.py
# Drop this class in place of the existing ApiLLM definition.

class ApiLLM(BaseLLM):
    """
    Calls a remote LLM API.  Supports:
      - OpenAI  (and any OpenAI-compatible endpoint, e.g. Together, Groq)
      - Anthropic Claude

    Schema enforcement
    ------------------
    OpenAI:  When schema is provided, uses structured outputs
             (response_format type json_schema).  Falls back to
             json_object mode if the model does not support structured outputs
             (older checkpoints).

    Claude:  Schema is serialised into the prompt so the model knows the
             exact shape expected.  The assistant prefill character is
             chosen based on the schema root type ("{" for objects,
             "[" for arrays) so the model cannot produce the wrong container.

    Both backends validate the response JSON and retry once on failure.

    Dependencies (install the one you need):
        pip install openai
        pip install anthropic

    Parameters
    ----------
    provider : str
        "openai" or "claude".
    api_key : str
        Your API key. If None, reads from the environment variable
        OPENAI_API_KEY or ANTHROPIC_API_KEY automatically.
    model : str | None
        Model name.  Defaults per provider:
          openai -> "gpt-4o"
          claude -> "claude-opus-4-6"
    base_url : str | None
        Override API base URL for OpenAI-compatible providers
        (e.g. "https://api.together.xyz/v1").
    temperature : float
        Sampling temperature.
    max_tokens : int
        Maximum tokens to generate.
    system_prompt : str | None
        Optional system prompt prepended to every request.
    """

    _DEFAULTS = {
        "openai": "gpt-4o",
        "claude": "claude-opus-4-6",
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
    ):
        super().__init__()
        provider = provider.lower()
        if provider not in self._DEFAULTS:
            raise ValueError(
                f"Unknown provider '{provider}'. Choose 'openai' or 'claude'."
            )

        self.provider      = provider
        self.api_key       = api_key
        self.model         = model or self._DEFAULTS[provider]
        self.base_url      = base_url
        self.temperature   = temperature
        self.max_tokens    = max_tokens
        self.system_prompt = system_prompt or (
            "You are a DAG planning assistant. "
            "Always respond with valid JSON and nothing else. "
            "No explanation, no markdown, no code fences."
        )

        logger.info("[API] Initialized ApiLLM  provider=%s  model=%s",
                    self.provider, self.model)
        if base_url:
            logger.info("[API] Using custom base_url: %s", base_url)

        self._client = None  # lazy-loaded

    # ---- Client loading -------------------------------------------------------

    def _load(self):
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

    # ---- Schema helpers -------------------------------------------------------

    @staticmethod
    def _schema_root_type(schema: dict) -> str:
        """Return 'object' or 'array' based on the schema root type field."""
        return schema.get("type", "object")

    @staticmethod
    def _schema_prefill(schema: dict) -> str:
        """Return the correct opening character for a JSON prefill."""
        return "[" if ApiLLM._schema_root_type(schema) == "array" else "{"

    @staticmethod
    def _inject_schema_into_prompt(prompt: str, schema: dict) -> str:
        """
        Append the JSON schema to the prompt so the model knows the exact
        shape required.  Used for Claude where grammar enforcement is not
        available natively.
        """
        schema_str = json.dumps(schema, indent=2)
        return (
            prompt
            + f"\n\nYou MUST respond with JSON that strictly conforms to this schema:\n"
            + f"```json\n{schema_str}\n```\n"
            + "Respond with valid JSON only. No explanation, no markdown fences."
        )

    # ---- OpenAI call ---------------------------------------------------------

    def _ask_openai(self, prompt: str, schema: dict | None) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": prompt},
        ]

        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        if schema is not None:
            # Structured outputs: enforces the exact schema server-side.
            # Requires gpt-4o-2024-08-06 or later.  Older models that do not
            # support it will raise an error that we catch and fall back from.
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name":   "response",
                    "schema": schema,
                    "strict": False,  # strict=True requires no $defs / additionalProperties
                },
            }
        else:
            # json_object mode: guarantees valid JSON but not a specific shape.
            kwargs["response_format"] = {"type": "json_object"}

        logger.debug("[API] Sending OpenAI request  model=%s  schema=%s",
                     self.model, "yes" if schema else "no")
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            # Some older models / compatible endpoints don't support json_schema.
            # Fall back to json_object mode so execution continues.
            err_str = str(e).lower()
            if schema is not None and (
                "json_schema" in err_str or "response_format" in err_str
                or "unsupported" in err_str
            ):
                logger.warning(
                    "[API] Structured outputs not supported by this model/endpoint "
                    "— falling back to json_object mode"
                )
                kwargs["response_format"] = {"type": "json_object"}
                response = self._client.chat.completions.create(**kwargs)
            else:
                raise
        if response.usage:
            token_counter.add(response.usage.prompt_tokens,
                              response.usage.completion_tokens)
        content = response.choices[0].message.content or ""
        logger.info("[API] OpenAI response received (%d chars)", len(content))
        logger.debug("[API] Raw response:\n%s", content)
        return content

    # ---- Claude call ---------------------------------------------------------

    def _ask_claude(self, prompt: str, schema: dict | None) -> str:
        # Embed the schema into the prompt so the model knows the exact shape.
        if schema is not None:
            augmented_prompt = self._inject_schema_into_prompt(prompt, schema)
            prefill = self._schema_prefill(schema)
        else:
            augmented_prompt = (
                prompt
                + "\n\nRespond with valid JSON only. "
                "No explanation, no markdown, no code fences."
            )
            prefill = "{"   # default to object; planner always returns objects

        logger.debug("[API] Sending Anthropic request  model=%s  prefill=%r",
                     self.model, prefill)
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=[
                {"role": "user",      "content": augmented_prompt},
                {"role": "assistant", "content": prefill},
            ],
            temperature=self.temperature,
        )
        token_counter.add(response.usage.input_tokens,
                          response.usage.output_tokens)
        # The prefill is not included in the response text — prepend it back.
        raw     = response.content[0].text
        content = prefill + raw
        logger.info("[API] Claude response received (%d chars)", len(content))
        logger.debug("[API] Raw response:\n%s", content)
        return content

    # ---- Public interface ----------------------------------------------------

    def ask(self, prompt: str, schema: dict | None = None) -> str:
        """
        Generate a response.

        schema=None  -> JSON object/array mode only (no shape enforcement)
        schema=dict  -> structured output enforcement (OpenAI json_schema /
                        Claude schema-in-prompt + correct prefill)

        Validates the response JSON and retries once on parse failure.
        """
        self._check_stop()
        logger.info("[API] ask() called  provider=%s  model=%s", self.provider, self.model)
        self._load()

        logger.debug("[API] Prompt (first 200 chars): %.200s", prompt)

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

            # Validate the response is parseable JSON
            try:
                parsed = json.loads(raw)
                if not parsed and parsed != 0:
                    raise ValueError("Empty JSON response")
                return raw
            except Exception as e:
                logger.warning("[API] Invalid JSON on attempt %d: %s  raw=%.200s",
                               attempt + 1, e, raw)
                if attempt == 0:
                    logger.warning("[API] Retrying due to JSON parse failure...")
                    continue

        raise ValueError(
            f"[API] {self.provider} returned invalid JSON after 2 attempts"
        )
# ---------------------------------------------------------------------------
# Factory — single entry point for the rest of the codebase
# ---------------------------------------------------------------------------

def create_llm_client(backend: str = "file", **kwargs) -> BaseLLM:
    """
    Factory that returns the right LLM backend.

    Usage examples
    --------------
    # File-based (original behaviour — no extra args needed)
    llm = create_llm_client("file")

    # Local llama.cpp model with outlines schema enforcement + caching
    llm = create_llm_client(
        "llamacpp",
        model_path="/models/mistral-7b-instruct.Q4_K_M.gguf",
        n_gpu_layers=35,
        temperature=0.1,
        # cache_path defaults to <PROJECT_ROOT>/llamacpp_cache.json
        # pass cache_path=None to disable caching
    )

    # OpenAI
    llm = create_llm_client("openai", api_key="sk-...", model="gpt-4o")

    # OpenAI-compatible provider (Together, Groq, etc.)
    llm = create_llm_client(
        "openai",
        base_url="https://api.together.xyz/v1",
        api_key="...",
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    )

    # Anthropic Claude
    llm = create_llm_client("claude", api_key="sk-ant-...")
    """
    backend = backend.lower()
    logger.info("[LLM FACTORY] Creating backend=%s", backend)

    if backend == "file":
        return FileBasedLLM(**kwargs)

    elif backend == "llamacpp":
        if "model_path" not in kwargs:
            raise ValueError(
                "llamacpp backend requires a 'model_path' keyword argument pointing "
                "to a .gguf file."
            )
        return LlamaCppLLM(**kwargs)

    elif backend in ("openai", "claude"):
        return ApiLLM(provider=backend, **kwargs)

    else:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            "Valid options: 'file', 'llamacpp', 'openai', 'claude'."
        )
