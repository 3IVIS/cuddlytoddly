from __future__ import annotations

# planning/llm_backends_local.py
#
# Local LLM backends:
#   - FileBasedLLM  — file-poll dev/test stub
#   - LlamaCppLLM   — local GGUF model via llama-cpp-python + outlines
import json
import threading
import time
from pathlib import Path

from toddly.core.id_generator import StableIDGenerator
from toddly.infra.logging import get_logger
from toddly.planning.llm_base import (
    _DEFAULT_POLL_INTERVAL,
    _DEFAULT_PROGRESS_LOG_INTERVAL,
    _DEFAULT_TIMEOUT,
    PROJECT_ROOT,
    PROMPT_LOG_FILE,
    RESPONSE_FILE,
    BaseLLM,
    LLMStoppedError,
    NativeToolResponse,
    PromptCache,
    TokenCounter,
)
from toddly.planning.llm_base import (
    id_gen as _module_id_gen,
)
from toddly.planning.schemas import EXECUTION_TURN_SCHEMA

logger = get_logger(__name__)


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
        # Use the caller-supplied generator so each run is isolated;
        # fall back to the module-level default only when none is provided.
        self._id_gen: StableIDGenerator = id_gen if id_gen is not None else _module_id_gen
        logger.info("[LLM] Initialized FileBasedLLM")
        logger.info("[LLM] Prompt file path: %s", self.prompt_log_file.resolve())
        logger.info("[LLM] Response file path: %s", self.response_file.resolve())
        logger.info(
            "[LLM] Cache: %s",
            f"enabled ({len(self._cache)} entries)" if self._cache else "disabled",
        )

    def send_prompt(self, prompt: str) -> str:
        logger.info("[LLM] send_prompt() called")
        # Use the per-instance generator instead of the module global
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

    def ask(self, prompt: str, schema=None, *, on_token=None, on_heartbeat=None) -> str:
        # on_token / on_heartbeat are not applicable to the file-poll backend
        # and are accepted only to satisfy the BaseLLM interface.
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
        n_ctx: int = 16384,  # matches config default for [llamacpp]
        n_gpu_layers: int = 0,
        temperature: float = 0.1,  # matches config default for [llamacpp]
        max_tokens: int = 8192,  # matches config default for [llamacpp]
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
        self.default_schema = schema or EXECUTION_TURN_SCHEMA

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

        # Optional progress callback — set this attribute after construction,
        # before the first ask() call.  Signature: (kind: str, payload: dict) -> None
        # kind "llm_loading" fires at start and every ~2s during load.
        # kind "llm_ready"   fires once loading completes.
        self.status_callback = None

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

            # Human-readable file size for the loading message
            try:
                size_gb = Path(model_path).stat().st_size / (1024**3)
                size_str = f"{size_gb:.1f} GB"
            except OSError:
                size_str = ""

            logger.info("[LLAMACPP] Loading model (first call — may take 10-30s)...")

            # Notify UI that loading is starting
            if self.status_callback:
                base_msg = f"Loading model ({size_str})" if size_str else "Loading model"
                self.status_callback(
                    "llm_loading",
                    {
                        "message": f"{base_msg}…",
                        "elapsed": 0,
                    },
                )

            # Watchdog: fires a heartbeat every 2s while Llama() blocks so
            # the status panel keeps updating with elapsed time.
            _done = threading.Event()
            _t0 = time.time()
            _base_msg = f"Loading model ({size_str})" if size_str else "Loading model"

            def _watch():
                while not _done.wait(timeout=2.0):
                    elapsed = int(time.time() - _t0)
                    logger.info("[LLAMACPP] Still loading model… %ds elapsed", elapsed)
                    if self.status_callback:
                        self.status_callback(
                            "llm_loading",
                            {
                                "message": f"{_base_msg}… {elapsed}s",
                                "elapsed": elapsed,
                            },
                        )

            threading.Thread(target=_watch, daemon=True, name="llm-load-watchdog").start()

            try:
                self._llama = Llama(
                    model_path=model_path,
                    n_ctx=self.n_ctx,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=False,
                )
            finally:
                _done.set()

            elapsed = round(time.time() - _t0, 1)
            logger.info("[LLAMACPP] Model loaded in %.1fs", elapsed)

            if self.status_callback:
                self.status_callback(
                    "llm_ready",
                    {
                        "message": f"Model ready ({elapsed}s)",
                        "elapsed": elapsed,
                    },
                )

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
            if self.status_callback:
                self.status_callback(
                    "llm_loading",
                    {
                        "message": "Initialising constrained generator…",
                        "elapsed": 0,
                    },
                )
            self._outlines_model = outlines.from_llamacpp(self._llama)
            logger.info("[LLAMACPP] Outlines model wrapper ready")
            if self.status_callback:
                self.status_callback("llm_ready", {"message": "Constrained generator ready"})

    def _load(self):
        self._load_model()

    # ── Constrained generator cache ───────────────────────────────────────────

    def _get_generator(self, schema: dict):
        import outlines

        fingerprint = json.dumps(schema, sort_keys=True)
        if fingerprint not in self._generators:
            # _load_outlines is guarded by a lock and a double-checked flag so
            # its status events (llm_loading / llm_ready) fire at most once per
            # process lifetime — for the initial outlines model build only.
            # Subsequent schema compilations (e.g. CORRECTION_TURN_SCHEMA on
            # the first correction turn) are fast (< 1s) and emit no events so
            # the user does not see a spurious second "Initialising…" row.
            self._load_outlines()
            logger.info(
                "[LLAMACPP] Building constrained generator for schema %s...",
                fingerprint[:40],
            )
            output_type = outlines.json_schema(fingerprint)
            self._generators[fingerprint] = outlines.Generator(self._outlines_model, output_type)
            logger.info("[LLAMACPP] Constrained generator ready for schema %s", fingerprint[:40])
        return self._generators[fingerprint]

    # ── Chat template ─────────────────────────────────────────────────────────

    def _apply_chat_template(self, prompt: str) -> str:
        from toddly.planning.prompts import LLAMACPP_SYSTEM_PROMPT

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

    def _run_unconstrained(self, prompt: str, safe_max: int, *, on_token=None) -> str:
        """Unconstrained inference.  Streams token-by-token when on_token is set."""
        logger.info("[LLAMACPP] Running unconstrained inference (max_tokens=%d)...", safe_max)
        if on_token is not None:
            chunks: list[str] = []
            for chunk in self._llama(
                prompt,
                max_tokens=safe_max,
                temperature=self.temperature,
                echo=False,
                stream=True,
            ):
                if self._stop_event.is_set():
                    raise LLMStoppedError("LLM stop requested mid-stream")
                text = chunk["choices"][0]["text"]
                if text:
                    chunks.append(text)
                    on_token(text)
            return "".join(chunks)
        else:
            result = self._llama(
                prompt,
                max_tokens=safe_max,
                temperature=self.temperature,
                echo=False,
            )
            return result["choices"][0]["text"]

    def _run_constrained(self, prompt: str, schema: dict, safe_max: int, *, on_token=None) -> str:
        """Constrained inference via outlines with optional per-token streaming.

        outlines.Generator exposes a ``.stream()`` method that yields one token
        string at a time, giving us the same token-level granularity as the
        unconstrained llama.cpp path.  When ``on_token`` is provided we iterate
        the stream and fire the callback for each chunk; otherwise we call the
        generator directly for a single-shot result (unchanged behaviour).
        """
        logger.info("[LLAMACPP] Running constrained inference (max_tokens=%d)...", safe_max)
        generator = self._get_generator(schema)

        if on_token is not None:
            chunks: list[str] = []
            for token in generator.stream(prompt, max_tokens=safe_max):
                if self._stop_event.is_set():
                    raise LLMStoppedError("LLM stop requested mid-stream")
                text = token if isinstance(token, str) else json.dumps(token)
                if text:
                    chunks.append(text)
                    on_token(text)
            raw = "".join(chunks)
        else:
            raw = generator(prompt, max_tokens=safe_max)

        if isinstance(raw, str):
            return raw
        return json.dumps(raw)

    def _run_model(
        self, prompt: str, constrained_schema=None, *, on_token=None, on_heartbeat=None
    ) -> str:
        """Run inference, emitting callbacks for streaming and heartbeat.

        The heartbeat thread replaces the old 30s-interval _run_watchdog:
        it fires on_heartbeat every 2s so progress indicators stay alive
        during long constrained-inference calls where no tokens are emitted.
        """
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
            # Heartbeat watchdog — fires every 2s regardless of constrained vs
            # unconstrained path.  For constrained inference it's the only
            # signal that anything is happening; for unconstrained it runs in
            # parallel with the on_token stream.
            _done = threading.Event()
            _t0 = time.time()

            def _watch():
                while not _done.wait(timeout=2.0):
                    elapsed = time.time() - _t0
                    logger.info("[LLAMACPP] Still generating… %.0fs elapsed", elapsed)
                    if on_heartbeat is not None:
                        on_heartbeat(elapsed)

            threading.Thread(target=_watch, daemon=True, name="llm-watchdog").start()

            t0 = _t0
            try:
                if constrained_schema is None:
                    raw = self._run_unconstrained(formatted, safe_max, on_token=on_token)
                else:
                    raw = self._run_constrained(
                        formatted, constrained_schema, safe_max, on_token=on_token
                    )
            finally:
                _done.set()

        completion_tokens = len(self._llama.tokenize(raw.encode("utf-8")))
        self._token_counter.add(prompt_tokens, completion_tokens)

        logger.info(
            "[LLAMACPP] Inference complete in %.1fs — %d chars",
            time.time() - t0,
            len(raw),
        )
        return raw

    # ── JSON repair ───────────────────────────────────────────────────────────

    def _repair_json(self, text: str):
        """
        Attempt to recover valid JSON from malformed LLM output.

        Two repair strategies are tried in order:

        1. Prose-wrapper strip — the model sometimes wraps the JSON payload in
           markdown fences or preamble text (e.g. "Here is the result:\\n```json\\n
           {...}\\n```").  We scan for the first ``{`` or ``[`` and the matching
           closing bracket, extract that substring, and try to parse it.  This
           catches the most common unconstrained-inference failure mode where
           ``json.loads`` fails at position 0 because the output starts with a
           letter, not a brace.

        2. Truncation repair — when constrained generation runs out of token
           budget mid-array, the output is a valid-prefix ``[`` followed by one
           or more complete objects and then an abrupt cut.  We walk backwards
           from the last ``}`` and try to close the array, recovering however
           many complete events were emitted before the truncation.

        Returns the repaired JSON string on success, or None if both strategies
        fail.
        """
        import re as _re

        text = text.strip()

        # ── Strategy 1: strip prose wrapper / markdown fences ─────────────────
        # Strip ```json ... ``` or ``` ... ``` fences first so the bracket
        # search below finds the real payload rather than fence punctuation.
        stripped = _re.sub(r"```(?:json)?\s*", "", text).strip()

        # Find the first opening bracket and its matching closer.
        for opener, closer in (("{", "}"), ("[", "]")):
            start = stripped.find(opener)
            if start == -1:
                continue
            # Walk forward tracking nesting depth to find the matching closer.
            depth = 0
            end = -1
            in_string = False
            escape_next = False
            for i, ch in enumerate(stripped[start:], start=start):
                if escape_next:
                    escape_next = False
                    continue
                if ch == "\\" and in_string:
                    escape_next = True
                    continue
                if ch == '"' and not escape_next:
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        end = i
                        break
            if end != -1:
                candidate = stripped[start : end + 1]
                try:
                    parsed = json.loads(candidate)
                    if parsed is not None:
                        logger.warning(
                            "[LLAMACPP] Prose-wrapped output repaired: extracted %s "
                            "from %d-char response",
                            type(parsed).__name__,
                            len(text),
                        )
                        return candidate
                except json.JSONDecodeError:
                    pass

        # ── Strategy 2: truncated array repair ────────────────────────────────
        # Only applies to array output (EVENT_LIST_SCHEMA, PLAN_SCHEMA events).
        if not text.startswith("["):
            logger.error(
                "[LLAMACPP] Could not repair output. Increase max_tokens (currently %d).",
                self.max_tokens,
            )
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

    # Appended to the prompt on the unconstrained retry when constrained
    # generation produced invalid JSON.  The reminder tells the model to emit
    # raw JSON rather than prose so _repair_json has something to work with.
    _JSON_RETRY_HINT: str = (
        "\n\n[IMPORTANT: Your previous response was not valid JSON. "
        "Respond ONLY with valid JSON — no prose, no markdown, no code fences. "
        "Begin your response immediately with { or [ as required by the schema.]"
    )

    def ask(
        self, prompt: str, schema: dict | None = None, *, on_token=None, on_heartbeat=None
    ) -> str:
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
            # On the retry (attempt 1), switch to unconstrained inference with
            # an explicit JSON format reminder appended.  The grammar-constrained
            # path already failed once; giving the model a plain-language
            # instruction is more likely to produce parseable output than
            # repeating the constrained call verbatim.
            if attempt == 0:
                run_prompt = prompt
                run_schema = constrained_schema
            else:
                run_prompt = prompt + self._JSON_RETRY_HINT
                run_schema = None  # unconstrained — grammar already failed once
                logger.info("[LLAMACPP] Retrying with unconstrained inference + JSON hint")

            # on_token is forwarded on both paths: _run_unconstrained streams
            # tokens natively; _run_constrained uses generator.stream() for
            # the same effect on the outlines-constrained path.
            response_text = self._run_model(
                run_prompt, run_schema, on_token=on_token, on_heartbeat=on_heartbeat
            )

            # When no schema was requested (generate() / plain ask()), the
            # caller expects raw prose — skip JSON validation entirely and
            # return immediately.  Applying json.loads here would cause all
            # generate() calls to fail whenever the model writes a natural
            # language response, which is always the case without a schema.
            if schema is None:
                if self._cache is not None:
                    self._cache.set(cache_key, response_text)
                return response_text

            # Schema was requested — validate that the response is parseable JSON.
            try:
                parsed = json.loads(response_text)
                # Mirror the ApiLLM fix — only reject a genuinely null
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
                    repaired = self._repair_json(response_text)
                    if repaired is not None:
                        if self._cache is not None:
                            self._cache.set(cache_key, repaired)
                        return repaired
                    logger.warning("[LLAMACPP] Repair failed -- retrying with JSON hint")

        raise ValueError("Model repeatedly returned invalid JSON")

    supports_native_tools: bool = True  # ask_with_tools() implemented below
    supports_streaming: bool = (
        True  # both _run_unconstrained and _run_constrained stream via on_token
    )

    def ask_with_tools(
        self,
        task_prompt: str,
        tools: list,
        history: list[dict],
        *,
        on_token=None,
        on_heartbeat=None,
    ) -> "NativeToolResponse":
        """
        Native tool-use for llama.cpp using constrained JSON generation.

        llama.cpp has no provider-level tool-calling API, so we implement the
        same done/tool_call protocol used by the legacy path but expose it
        through the ask_with_tools() interface so the native executor loop
        (_execute_native) can drive this backend identically to ApiLLM.

        The full conversation — task prompt, tool schema, and accumulated
        history — is serialised into a single prompt string and passed to
        ask() with EXECUTION_TURN_SCHEMA so constrained generation forces
        a valid {done, result, tool_call} object every time.
        """

        # Build tool summary block
        if tools:
            tool_lines = [
                f"- {t.name}: {getattr(t, 'description', t.name)}. "
                f"Args: {json.dumps(getattr(t, 'input_schema', {}))}"
                for t in tools
            ]
            tools_block = "Available tools:\n" + "\n".join(tool_lines)
        else:
            tools_block = (
                "NO TOOLS ARE AVAILABLE FOR THIS TASK. "
                "Set done=true with a result based on your knowledge."
            )

        # Build history block
        history_parts = []
        for entry in history:
            if entry.get("kind") == "correction":
                history_parts.append(f"[Correction]: {entry['content']}")
            else:
                history_parts.append(
                    f"Tool: {entry.get('name', '?')}\n"
                    f"Args: {json.dumps(entry.get('args', {}))}\n"
                    f"Result: {entry.get('result', '')}"
                )
        history_block = (
            "Previous tool calls this turn:\n" + "\n\n".join(history_parts) if history_parts else ""
        )

        # Assemble full prompt
        parts = [task_prompt, tools_block]
        if history_block:
            parts.append(history_block)
        parts.append(
            "Respond only in JSON. "
            "Set done=true and result=<final answer> when the task is complete. "
            "Set done=false and tool_call={name, args} to invoke a tool."
        )
        combined_prompt = "\n\n".join(parts)

        logger.info(
            "[LLAMACPP] ask_with_tools()  tools=%d  history_len=%d",
            len(tools),
            len(history),
        )
        raw = self.ask(combined_prompt, schema=EXECUTION_TURN_SCHEMA, on_heartbeat=on_heartbeat)
        parsed = json.loads(raw)

        if parsed.get("done"):
            return NativeToolResponse(kind="text", text=parsed.get("result", ""))

        tool_call = parsed.get("tool_call") or {}
        return NativeToolResponse(
            kind="tool_call",
            tool_name=tool_call.get("name", ""),
            tool_args=tool_call.get("args", {}),
            tool_use_id="",  # llama.cpp does not assign provider-side IDs
        )

    def clear_cache(self):
        if self._cache is not None:
            self._cache.clear()
            logger.info("[LLAMACPP] Cache cleared")
        else:
            logger.info("[LLAMACPP] Cache is disabled -- nothing to clear")
