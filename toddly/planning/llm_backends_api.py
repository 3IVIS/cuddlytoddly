from __future__ import annotations

# planning/llm_backends_api.py
#
# Remote API backend:
#   - ApiLLM — OpenAI-compatible and Anthropic Claude
import json
from pathlib import Path
from typing import Any

from toddly.infra.logging import get_logger
from toddly.planning.llm_base import (
    _API_MAX_ATTEMPTS,
    _API_RATE_LIMIT_INITIAL_BACKOFF,
    _API_RATE_LIMIT_MAX_BACKOFF,
    _DEFAULT_CLAUDE_MODEL,
    _DEFAULT_OPENAI_MODEL,
    BaseLLM,
    LLMStoppedError,
    NativeToolResponse,
    PromptCache,
    TokenCounter,
)

logger = get_logger(__name__)


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
        temperature: float = 0.1,  # matches config default for [claude] and [openai]
        max_tokens: int = 8192,  # matches config default for [claude] and [openai]
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
        # Deferred import to avoid circular dependency at module load time.
        if system_prompt is not None:
            self.system_prompt = system_prompt
        else:
            from toddly.planning.prompts import LLM_SYSTEM_PROMPT

            self.system_prompt = LLM_SYSTEM_PROMPT
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
        self._load_lock = __import__("threading").Lock()

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

    # ── Rate-limit detection ──────────────────────────────────────────────────

    @staticmethod
    def _is_rate_limit_error(exc: Exception) -> bool:
        """
        Return True when *exc* signals an HTTP 429 / rate-limit condition.

        Both the ``openai`` and ``anthropic`` SDKs raise a ``RateLimitError``
        subclass for 429 responses, but the exact class path differs between
        SDK versions and providers.  We therefore detect by name and message
        text rather than by isinstance(), so this works across all versions
        and also catches compatible third-party providers.
        """
        name = type(exc).__name__.lower()
        msg = str(exc).lower()
        return (
            "ratelimit" in name  # openai.RateLimitError, anthropic.RateLimitError
            or "rate_limit" in name  # some SDKs use underscored variant
            or "429" in msg
            or "rate limit" in msg
            or "too many requests" in msg
        )

    # ── Public interface ──────────────────────────────────────────────────────

    def _interruptible_sleep(self, seconds: float) -> None:
        """
        Sleep for up to `seconds` while checking the stop flag every 0.5 s.

        FIX 4: replaces bare time.sleep(backoff) in rate-limit retry paths.
        A plain time.sleep(60) is completely unresponsive to the user pressing
        pause — the stop flag is set but the sleeping thread cannot observe it.
        By polling in short increments we give sub-second pause responsiveness
        even at maximum backoff duration.
        """
        import time as _time

        deadline = _time.monotonic() + seconds
        while _time.monotonic() < deadline:
            if self._stop_event.is_set():
                logger.info("[API] Stop flag set during backoff sleep — aborting sleep")
                raise LLMStoppedError("LLM paused during rate-limit backoff")
            _time.sleep(min(0.5, deadline - _time.monotonic()))

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

        # FIX: use 4 attempts (was 2) so rate-limit retries with backoff still
        # leave room for a JSON-parse retry without exhausting all attempts.
        for attempt in range(_API_MAX_ATTEMPTS):
            try:
                if self.provider == "openai":
                    raw = self._ask_openai(prompt, schema)
                else:
                    raw = self._ask_claude(prompt, schema)
            except LLMStoppedError:
                raise
            except Exception as e:
                # FIX: rate-limit errors (HTTP 429) require exponential backoff
                # before retrying.  The original code retried immediately on
                # any exception, which turned a 429 into two back-to-back
                # requests — the second also rate-limited — and then re-raised.
                # Under concurrent execution with max_workers > 1, this caused
                # a burst of fast-failing requests that worsened the situation.
                if self._is_rate_limit_error(e):
                    backoff = min(
                        _API_RATE_LIMIT_INITIAL_BACKOFF * (2**attempt),
                        _API_RATE_LIMIT_MAX_BACKOFF,
                    )  # e.g. 5s, 10s, 20s, 40s…
                    logger.warning(
                        "[API] Rate limit hit on attempt %d/%d — sleeping %.0fs before retry",
                        attempt + 1,
                        _API_MAX_ATTEMPTS,
                        backoff,
                    )
                    # FIX 4: sleep in short increments so the stop flag is
                    # checked frequently.  A bare time.sleep(backoff) blocks for
                    # up to 60 s; the user pressing pause cannot interrupt it and
                    # the orchestrator's llm_stopped flag has no effect until the
                    # sleep completes.  Polling every 0.5 s gives sub-second
                    # pause responsiveness even at maximum backoff.
                    self._interruptible_sleep(backoff)
                    continue
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
                if attempt < _API_MAX_ATTEMPTS - 1:
                    logger.warning("[API] Retrying due to JSON parse failure...")
                    continue

        raise ValueError(
            f"[API] {self.provider} returned invalid JSON after {_API_MAX_ATTEMPTS} attempts"
        )

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

        FIX 3: wraps the provider call in a retry loop matching the logic in
        ask().  Previously any transient error (rate limit, network blip)
        during a native tool-use turn propagated immediately to the executor,
        which returned None and caused the orchestrator to mark the node failed
        and start its own exponential backoff retry cycle.  The ask() path
        handles the same errors transparently with up to _API_MAX_ATTEMPTS
        retries; ask_with_tools() now does the same.
        """
        self._check_stop()
        self._load()
        logger.info(
            "[API] ask_with_tools()  provider=%s  model=%s  history_len=%d",
            self.provider,
            self.model,
            len(history),
        )

        for attempt in range(_API_MAX_ATTEMPTS):
            try:
                if self.provider == "claude":
                    return self._ask_with_tools_claude(task_prompt, tools, history)
                else:
                    return self._ask_with_tools_openai(task_prompt, tools, history)
            except LLMStoppedError:
                raise  # always propagate stop immediately — no retry
            except Exception as e:
                if self._is_rate_limit_error(e):
                    backoff = min(
                        _API_RATE_LIMIT_INITIAL_BACKOFF * (2**attempt),
                        _API_RATE_LIMIT_MAX_BACKOFF,
                    )
                    logger.warning(
                        "[API] ask_with_tools() rate limit on attempt %d/%d "
                        "— sleeping %.0fs before retry",
                        attempt + 1,
                        _API_MAX_ATTEMPTS,
                        backoff,
                    )
                    self._interruptible_sleep(backoff)
                    continue
                logger.error(
                    "[API] ask_with_tools() request failed on attempt %d: %s",
                    attempt + 1,
                    e,
                )
                if attempt == 0:
                    logger.warning("[API] ask_with_tools() retrying after transient error…")
                    continue
                raise

        # Exhausted all attempts — this path is only reached when every attempt
        # raised a non-rate-limit exception and was re-raised; the raise above
        # means we never actually reach here, but satisfy the type checker.
        raise RuntimeError(f"[API] ask_with_tools() exhausted {_API_MAX_ATTEMPTS} attempts")

    def _ask_with_tools_claude(
        self,
        task_prompt: str,
        tools: list,
        history: list[dict],
    ) -> NativeToolResponse:
        from toddly.planning.prompts import EXECUTOR_NATIVE_SYSTEM_PROMPT

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
        from toddly.planning.prompts import EXECUTOR_NATIVE_SYSTEM_PROMPT

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
