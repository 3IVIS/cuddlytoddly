"""Tests for cuddlytoddly.planning.llm_executor.LLMExecutor."""
import json
import pytest
from unittest.mock import MagicMock, patch
from cuddlytoddly.planning.llm_executor import LLMExecutor
from cuddlytoddly.planning.llm_interface import LLMStoppedError
from cuddlytoddly.core.task_graph import TaskGraph
from conftest import FakeLLM, add_node


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_node(node_id="task_1", description="do something", output=None, deps=None):
    g = TaskGraph()
    add_node(g, node_id, metadata={
        "description": description,
        "output": output or [],
        "required_input": [],
    })
    if deps:
        for dep_id, result in deps.items():
            add_node(g, dep_id, metadata={"description": dep_id, "output": []})
            g.nodes[dep_id].result = result
            g.nodes[dep_id].status = "done"
            g.nodes[node_id].dependencies.add(dep_id)
    return g.nodes[node_id], g.get_snapshot()


def done_response(result="all done"):
    return json.dumps({"done": True, "result": result})


def tool_response(name, args):
    return json.dumps({"done": False, "tool_call": {"name": name, "args": args}})


def make_tool_registry(tools_dict):
    """Build a minimal ToolRegistry from a dict of name → callable."""
    from cuddlytoddly.skills.skill_loader import ToolRegistry, Tool
    registry = ToolRegistry()
    for name, fn in tools_dict.items():
        registry.register(Tool(
            name=name,
            description=f"Tool {name}",
            input_schema={"input": "string"},
            fn=fn,
        ))
    return registry


# ── Basic execution ───────────────────────────────────────────────────────────

class TestExecutorBasic:
    def test_done_on_first_turn(self):
        llm = FakeLLM(done_response("result text"))
        executor = LLMExecutor(llm_client=llm, max_turns=5)
        node, snapshot = make_node()
        result = executor.execute(node, snapshot)
        assert result == "result text"

    def test_returns_none_on_llm_stopped_error(self):
        llm = FakeLLM(None)
        llm.stop()
        executor = LLMExecutor(llm_client=llm, max_turns=5)
        node, snapshot = make_node()
        result = executor.execute(node, snapshot)
        assert result is None

    def test_returns_none_after_max_turns(self):
        """LLM keeps requesting tool calls and never sets done=True."""
        call_count = [0]
        def responses(prompt, schema=None):
            call_count[0] += 1
            return tool_response("my_tool", {"input": "x"})

        llm = FakeLLM(responses)
        registry = make_tool_registry({"my_tool": lambda args: "tool result"})
        executor = LLMExecutor(llm_client=llm, tool_registry=registry, max_turns=3)
        node, snapshot = make_node()
        result = executor.execute(node, snapshot)
        assert result is None
        assert call_count[0] == 3

    def test_returns_none_on_json_parse_error(self):
        llm = FakeLLM("this is not json at all")
        executor = LLMExecutor(llm_client=llm, max_turns=3)
        node, snapshot = make_node()
        result = executor.execute(node, snapshot)
        assert result is None

    def test_returns_none_when_done_false_no_tool_call(self):
        llm = FakeLLM(json.dumps({"done": False}))
        executor = LLMExecutor(llm_client=llm, max_turns=3)
        node, snapshot = make_node()
        result = executor.execute(node, snapshot)
        assert result is None


# ── Tool calls ────────────────────────────────────────────────────────────────

class TestExecutorToolCalls:
    def test_tool_called_then_done(self):
        call_log = []
        turns = [
            tool_response("my_tool", {"input": "hello"}),
            done_response("final result"),
        ]
        idx = [0]

        def responses(prompt, schema=None):
            r = turns[idx[0]]
            idx[0] += 1
            return r

        llm = FakeLLM(responses)
        registry = make_tool_registry({
            "my_tool": lambda args: (call_log.append(args), "tool output")[1]
        })
        executor = LLMExecutor(llm_client=llm, tool_registry=registry, max_turns=5)
        node, snapshot = make_node()
        result = executor.execute(node, snapshot)
        assert result == "final result"
        assert len(call_log) == 1
        assert call_log[0]["input"] == "hello"

    def test_unknown_tool_returns_error_in_history(self):
        turns = [
            tool_response("nonexistent_tool", {"x": "1"}),
            done_response("recovered"),
        ]
        idx = [0]

        def responses(prompt, schema=None):
            r = turns[idx[0]]
            idx[0] += 1
            return r

        llm = FakeLLM(responses)
        executor = LLMExecutor(llm_client=llm, tool_registry=None, max_turns=5)
        node, snapshot = make_node()
        result = executor.execute(node, snapshot)
        # Execution continues after the error
        assert result == "recovered"

    def test_tool_exception_does_not_crash_executor(self):
        turns = [
            tool_response("bad_tool", {"x": "1"}),
            done_response("survived"),
        ]
        idx = [0]

        def responses(prompt, schema=None):
            r = turns[idx[0]]
            idx[0] += 1
            return r

        def bad_fn(args):
            raise RuntimeError("tool exploded")

        llm = FakeLLM(responses)
        registry = make_tool_registry({"bad_tool": bad_fn})
        executor = LLMExecutor(llm_client=llm, tool_registry=registry, max_turns=5)
        node, snapshot = make_node()
        result = executor.execute(node, snapshot)
        assert result == "survived"


# ── File output enforcement ───────────────────────────────────────────────────

class TestExecutorFileOutput:
    def _file_output_node(self, filename="report.md"):
        g = TaskGraph()
        add_node(g, "task_1", metadata={
            "description": "write a report",
            "output": [{"name": filename, "type": "file",
                         "description": "the report"}],
            "required_input": [],
        })
        return g.nodes["task_1"], g.get_snapshot()

    def test_done_without_write_file_triggers_correction(self):
        """If done=True without write_file being called, inject correction turn."""
        written = []
        turns = [
            done_response("report content"),              # turn 1: done without writing
            tool_response("write_file", {"path": "report.md", "content": "actual content"}),
            done_response("file_written: report.md\nsummary: content"),
        ]
        idx = [0]

        def responses(prompt, schema=None):
            r = turns[min(idx[0], len(turns) - 1)]
            idx[0] += 1
            return r

        llm = FakeLLM(responses)
        registry = make_tool_registry({
            "write_file": lambda args: (written.append(args), "written")[1]
        })
        executor = LLMExecutor(llm_client=llm, tool_registry=registry, max_turns=5)
        node, snapshot = self._file_output_node()
        result = executor.execute(node, snapshot)
        # Executor should not have accepted the first done=True without write_file
        assert result is not None

    def test_write_file_called_before_done_accepted(self):
        """If write_file is called first, done=True is accepted normally."""
        turns = [
            tool_response("write_file", {"path": "report.md", "content": "hello"}),
            done_response("file_written: report.md\nsummary: hello"),
        ]
        idx = [0]

        def responses(prompt, schema=None):
            r = turns[idx[0]]
            idx[0] += 1
            return r

        llm = FakeLLM(responses)
        registry = make_tool_registry({
            "write_file": lambda args: f"Written {len(args['content'])} chars"
        })
        executor = LLMExecutor(llm_client=llm, tool_registry=registry, max_turns=5)
        node, snapshot = self._file_output_node()
        result = executor.execute(node, snapshot)
        assert result is not None
        assert "file_written" in result


# ── Input resolution ──────────────────────────────────────────────────────────

class TestExecutorInputResolution:
    def test_upstream_result_included_in_prompt(self):
        prompts = []

        def capture(prompt, schema=None):
            prompts.append(prompt)
            return done_response("done")

        llm = FakeLLM(capture)
        executor = LLMExecutor(llm_client=llm, max_turns=5)
        node, snapshot = make_node(deps={"upstream_task": "upstream output data"})
        executor.execute(node, snapshot)
        assert any("upstream output data" in p for p in prompts)

    def test_long_upstream_result_truncated(self):
        prompts = []

        def capture(prompt, schema=None):
            prompts.append(prompt)
            return done_response("done")

        llm = FakeLLM(capture)
        executor = LLMExecutor(llm_client=llm, max_turns=5)
        long_result = "x" * 10_000
        node, snapshot = make_node(deps={"upstream": long_result})
        executor.execute(node, snapshot)
        combined = " ".join(prompts)
        assert "truncated" in combined or len(combined) < 20_000


# ── Reporter integration ──────────────────────────────────────────────────────

class TestExecutorReporter:
    def test_reporter_on_tool_start_called(self):
        turns = [
            tool_response("my_tool", {"input": "x"}),
            done_response("done"),
        ]
        idx = [0]

        def responses(prompt, schema=None):
            r = turns[idx[0]]
            idx[0] += 1
            return r

        llm = FakeLLM(responses)
        registry = make_tool_registry({"my_tool": lambda args: "result"})
        executor = LLMExecutor(llm_client=llm, tool_registry=registry, max_turns=5)

        reporter = MagicMock()
        reporter.on_tool_start.return_value = "step_id"

        node, snapshot = make_node()
        executor.execute(node, snapshot, reporter=reporter)
        reporter.on_tool_start.assert_called_once()
        reporter.on_tool_done.assert_called_once()

    def test_reporter_on_synthesis_not_called_when_no_tools(self):
        llm = FakeLLM(done_response("immediate"))
        executor = LLMExecutor(llm_client=llm, max_turns=5)
        reporter = MagicMock()
        node, snapshot = make_node()
        executor.execute(node, snapshot, reporter=reporter)
        reporter.on_tool_start.assert_not_called()

    def test_reporter_on_llm_error_called_on_json_failure(self):
        llm = FakeLLM("broken json {{{")
        executor = LLMExecutor(llm_client=llm, max_turns=3)
        reporter = MagicMock()
        node, snapshot = make_node()
        executor.execute(node, snapshot, reporter=reporter)
        reporter.on_llm_error.assert_called()
