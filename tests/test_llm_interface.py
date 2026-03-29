"""Tests for cuddlytoddly.planning.llm_interface (LLM base + API layer)."""
import json
import threading
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from cuddlytoddly.planning.llm_interface import (
    LLMStoppedError, BaseLLM, TokenCounter,
    FileBasedLLM, LlamaCppCache, create_llm_client,
)


# ── TokenCounter ──────────────────────────────────────────────────────────────

class TestTokenCounter:
    def test_initial_values_zero(self):
        tc = TokenCounter()
        assert tc.prompt_tokens == 0
        assert tc.completion_tokens == 0
        assert tc.total_tokens == 0
        assert tc.calls == 0

    def test_add_increments_counts(self):
        tc = TokenCounter()
        tc.add(100, 50)
        assert tc.prompt_tokens == 100
        assert tc.completion_tokens == 50
        assert tc.total_tokens == 150
        assert tc.calls == 1

    def test_add_accumulates(self):
        tc = TokenCounter()
        tc.add(10, 5)
        tc.add(20, 10)
        assert tc.prompt_tokens == 30
        assert tc.completion_tokens == 15
        assert tc.calls == 2

    def test_reset_clears_all(self):
        tc = TokenCounter()
        tc.add(100, 50)
        tc.reset()
        assert tc.total_tokens == 0
        assert tc.calls == 0

    def test_thread_safe_concurrent_adds(self):
        tc = TokenCounter()
        errors = []

        def adder():
            try:
                for _ in range(100):
                    tc.add(1, 1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=adder) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert tc.calls == 1000
        assert tc.prompt_tokens == 1000


# ── LLMStoppedError ───────────────────────────────────────────────────────────

class TestLLMStoppedError:
    def test_is_runtime_error(self):
        assert issubclass(LLMStoppedError, RuntimeError)

    def test_can_be_raised_and_caught(self):
        with pytest.raises(LLMStoppedError):
            raise LLMStoppedError("paused")


# ── BaseLLM stop / resume ─────────────────────────────────────────────────────

class ConcreteLLM(BaseLLM):
    """Minimal concrete subclass for testing BaseLLM behaviour."""
    def ask(self, prompt: str, schema=None) -> str:
        self._check_stop()
        return "{}"


class TestBaseLLM:
    def test_not_stopped_initially(self):
        llm = ConcreteLLM()
        assert not llm.is_stopped

    def test_stop_sets_flag(self):
        llm = ConcreteLLM()
        llm.stop()
        assert llm.is_stopped

    def test_resume_clears_flag(self):
        llm = ConcreteLLM()
        llm.stop()
        llm.resume()
        assert not llm.is_stopped

    def test_ask_raises_when_stopped(self):
        llm = ConcreteLLM()
        llm.stop()
        with pytest.raises(LLMStoppedError):
            llm.ask("hello")

    def test_ask_succeeds_after_resume(self):
        llm = ConcreteLLM()
        llm.stop()
        llm.resume()
        result = llm.ask("hello")
        assert result == "{}"

    def test_generate_is_alias_for_ask(self):
        llm = ConcreteLLM()
        assert llm.generate("hello") == llm.ask("hello")


# ── LlamaCppCache ─────────────────────────────────────────────────────────────

class TestLlamaCppCache:
    def test_get_miss_returns_none(self, tmp_path):
        cache = LlamaCppCache(tmp_path / "cache.json")
        assert cache.get("unknown prompt") is None

    def test_set_and_get(self, tmp_path):
        cache = LlamaCppCache(tmp_path / "cache.json")
        cache.set("my prompt", '{"answer": 42}')
        result = cache.get("my prompt")
        assert result == '{"answer": 42}'

    def test_persists_to_disk(self, tmp_path):
        path = tmp_path / "cache.json"
        cache1 = LlamaCppCache(path)
        cache1.set("prompt", "response")

        cache2 = LlamaCppCache(path)
        assert cache2.get("prompt") == "response"

    def test_len_reflects_entries(self, tmp_path):
        cache = LlamaCppCache(tmp_path / "cache.json")
        assert len(cache) == 0
        cache.set("p1", "r1")
        cache.set("p2", "r2")
        assert len(cache) == 2

    def test_clear_empties_cache(self, tmp_path):
        cache = LlamaCppCache(tmp_path / "cache.json")
        cache.set("p", "r")
        cache.clear()
        assert len(cache) == 0
        assert cache.get("p") is None

    def test_corrupted_file_handled_gracefully(self, tmp_path):
        path = tmp_path / "cache.json"
        path.write_text("not valid json {{{")
        cache = LlamaCppCache(path)
        assert len(cache) == 0

    def test_atomic_save_via_tmp_file(self, tmp_path):
        path = tmp_path / "cache.json"
        cache = LlamaCppCache(path)
        cache.set("p", "r")
        # No .tmp file should remain
        tmp_file = path.with_suffix(".tmp")
        assert not tmp_file.exists()

    def test_different_prompts_different_keys(self, tmp_path):
        cache = LlamaCppCache(tmp_path / "cache.json")
        cache.set("prompt A", "resp A")
        cache.set("prompt B", "resp B")
        assert cache.get("prompt A") == "resp A"
        assert cache.get("prompt B") == "resp B"

    def test_backward_compat_bare_string_value(self, tmp_path):
        """Old cache format stored bare string, not a dict."""
        import hashlib
        path = tmp_path / "cache.json"
        prompt = "old prompt"
        key = hashlib.sha256(prompt.encode()).hexdigest()
        path.write_text(json.dumps({key: "bare string response"}))
        cache = LlamaCppCache(path)
        assert cache.get(prompt) == "bare string response"


# ── create_llm_client factory ─────────────────────────────────────────────────

class TestCreateLLMClient:
    def test_file_backend(self, tmp_path):
        llm = create_llm_client("file",
                                prompt_log_file=str(tmp_path / "prompts.txt"),
                                response_file=str(tmp_path / "responses.txt"))
        assert isinstance(llm, FileBasedLLM)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError, match="Unknown backend"):
            create_llm_client("magic_llm")

    def test_llamacpp_requires_model_path(self):
        with pytest.raises(ValueError, match="model_path"):
            create_llm_client("llamacpp")

    def test_openai_backend_returns_api_llm(self):
        from cuddlytoddly.planning.llm_interface import ApiLLM
        llm = create_llm_client("openai", api_key="fake-key")
        assert isinstance(llm, ApiLLM)
        assert llm.provider == "openai"

    def test_claude_backend_returns_api_llm(self):
        from cuddlytoddly.planning.llm_interface import ApiLLM
        llm = create_llm_client("claude", api_key="fake-key")
        assert isinstance(llm, ApiLLM)
        assert llm.provider == "claude"

    def test_backend_case_insensitive(self):
        llm = create_llm_client("FILE",
                                prompt_log_file="/tmp/p.txt",
                                response_file="/tmp/r.txt")
        assert isinstance(llm, FileBasedLLM)


# ── ApiLLM schema helpers ─────────────────────────────────────────────────────

class TestApiLLMHelpers:
    def setup_method(self):
        from cuddlytoddly.planning.llm_interface import ApiLLM
        self.ApiLLM = ApiLLM

    def test_schema_root_type_object(self):
        schema = {"type": "object", "properties": {}}
        assert self.ApiLLM._schema_root_type(schema) == "object"

    def test_schema_root_type_array(self):
        schema = {"type": "array", "items": {}}
        assert self.ApiLLM._schema_root_type(schema) == "array"

    def test_schema_prefill_object(self):
        schema = {"type": "object"}
        assert self.ApiLLM._schema_prefill(schema) == "{"

    def test_schema_prefill_array(self):
        schema = {"type": "array"}
        assert self.ApiLLM._schema_prefill(schema) == "["

    def test_inject_schema_into_prompt(self):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        result = self.ApiLLM._inject_schema_into_prompt("base prompt", schema)
        assert "base prompt" in result
        assert "JSON" in result

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            self.ApiLLM(provider="google")
