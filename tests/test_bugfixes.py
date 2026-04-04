"""
Tests for the four bug fixes:

  Issue 3 — user_input nodes skip quality-gate verification (never loop)
  Issue 4 — _all_tool_calls_failed pre-check blocks hallucinated results
  Issue 5 — max_retries cap + exponential backoff on verification failure
  Issue 6 — tool error strings (no exception raised) are flagged as error=True
"""
import json
import time
import pytest
from concurrent.futures import Future
from unittest.mock import MagicMock, patch
from cuddlytoddly.core.task_graph import TaskGraph
from cuddlytoddly.core.events import Event, ADD_NODE, MARK_DONE
from cuddlytoddly.core.reducer import apply_event
from cuddlytoddly.engine.llm_orchestrator import Orchestrator
from cuddlytoddly.engine.quality_gate import QualityGate
from cuddlytoddly.planning.llm_executor import LLMExecutor
from cuddlytoddly.infra.event_queue import EventQueue
from conftest import FakeLLM, add_node, mark_done


# ── Shared helpers ────────────────────────────────────────────────────────────

def make_orchestrator(graph=None, executor_result="mock result",
                      gate_satisfied=True, max_retries=5):
    g = graph or TaskGraph()
    mock_planner = MagicMock()
    mock_planner.propose.return_value = []
    mock_executor = MagicMock()
    mock_executor.execute.return_value = executor_result
    mock_gate = MagicMock()
    mock_gate.verify_result.return_value = (gate_satisfied, "reason")
    mock_gate.check_dependencies.return_value = None

    orch = Orchestrator(
        graph=g,
        planner=mock_planner,
        executor=mock_executor,
        quality_gate=mock_gate,
        event_queue=EventQueue(),
        max_workers=1,
        max_retries=max_retries,
    )
    orch._llm_clients = []   # prevent llm_stopped from blocking execution
    return orch, g, mock_executor, mock_gate


def make_tool_registry(tools_dict):
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


def make_reporter(parent_id, graph):
    from cuddlytoddly.engine.execution_step_reporter import ExecutionStepReporter
    import threading
    lock = threading.RLock()
    apply_events = []

    def apply_fn(event):
        apply_event(graph, event)
        apply_events.append(event)

    return ExecutionStepReporter(
        parent_node_id=parent_id,
        apply_fn=apply_fn,
        graph_lock=lock,
        graph=graph,
    )


def done_response(result="all done"):
    return json.dumps({"done": True, "result": result})


def tool_response(name, args):
    return json.dumps({"done": False, "tool_call": {"name": name, "args": args}})


def unsatisfied():
    return json.dumps({"satisfied": False, "reason": "not good enough"})


# ── Issue 3: user_input nodes skip quality-gate verification ─────────────────

class TestUserInputSkipsVerification:

    def _make_future(self, result):
        f = Future()
        f.set_result(result)
        return f

    def test_user_input_node_marked_done_without_calling_verify_result(self):
        """verify_result must never be called for user_input nodes."""
        orch, g, _, mock_gate = make_orchestrator()
        add_node(g, "ask_user", node_type="user_input", metadata={
            "description": "List your achievements",
            "output": [{"name": "achievements_list", "type": "list",
                        "description": "Your achievements"}],
        })
        g.nodes["ask_user"].status = "running"

        template = (
            "[Template — please provide the following information]\n"
            "Task: List your achievements\n\n"
            "  achievements_list: <please fill in>"
        )
        orch._on_node_done("ask_user", self._make_future(template))

        mock_gate.verify_result.assert_not_called()
        assert g.nodes["ask_user"].status == "done"
        assert g.nodes["ask_user"].result == template

    def test_user_input_node_never_reset_on_template_result(self):
        """A user_input node must not loop back to pending after producing
        a template — that was the infinite-retry bug."""
        orch, g, _, mock_gate = make_orchestrator()
        add_node(g, "ask_user", node_type="user_input", metadata={
            "description": "Provide salary info",
            "output": [{"name": "salary", "type": "string", "description": "salary"}],
        })
        g.nodes["ask_user"].status = "running"

        template = "[Template — please provide the following information]\n  salary: <please fill in>"
        for _ in range(3):
            orch._on_node_done("ask_user", self._make_future(template))

        # Should remain done after first call; no loop
        assert g.nodes["ask_user"].status == "done"
        mock_gate.verify_result.assert_not_called()

    def test_regular_task_still_calls_verify_result(self):
        """Sanity check: ordinary task nodes still go through verification."""
        orch, g, _, mock_gate = make_orchestrator(gate_satisfied=True)
        add_node(g, "regular_task", node_type="task", metadata={
            "description": "Do research",
            "output": [{"name": "report", "type": "document",
                        "description": "the report"}],
        })
        g.nodes["regular_task"].status = "running"

        f = Future()
        f.set_result("Substantive research report content here.")
        orch._on_node_done("regular_task", f)

        mock_gate.verify_result.assert_called_once()


# ── Issue 4: _all_tool_calls_failed blocks hallucinated output ────────────────

class TestAllToolCallsFailed:

    def _make_gate(self):
        llm = FakeLLM(json.dumps({"satisfied": True, "reason": "looks fine"}))
        return QualityGate(llm_client=llm)

    def _make_node_with_steps(self, node_id, attempts_per_step):
        """Build a TaskGraph with a parent node and one step node per entry
        in `attempts_per_step`.  Each entry is a list of result strings."""
        g = TaskGraph()
        add_node(g, node_id, metadata={
            "description": "research task",
            "output": [{"name": "report", "type": "document",
                        "description": "the report"}],
        })
        for i, attempt_results in enumerate(attempts_per_step):
            step_id = f"{node_id}__step_web_search_{i}"
            add_node(g, step_id, node_type="execution_step", metadata={
                "step_type": "tool_call",
                "tool_name": "web_search",
                "attempts": [
                    {"turn": j, "args": {}, "result": r, "status": "ok"}
                    for j, r in enumerate(attempt_results)
                ],
            })
        return g.nodes[node_id], g.get_snapshot()

    # ── _all_tool_calls_failed unit tests ─────────────────────────────────────

    def test_all_error_strings_returns_true(self):
        gate = self._make_gate()
        node, snap = self._make_node_with_steps("t", [
            ["ERROR: web search failed — decode error"],
            ["No results found for: average salary"],
        ])
        assert gate._all_tool_calls_failed(node, snap) is True

    def test_mixed_results_returns_false(self):
        """At least one useful result → should NOT be treated as all-failed."""
        gate = self._make_gate()
        node, snap = self._make_node_with_steps("t", [
            ["ERROR: timeout"],
            ["Title: Software Engineer Salary\nURL: glassdoor.com\nSnippet: $120k avg"],
        ])
        assert gate._all_tool_calls_failed(node, snap) is False

    def test_no_step_nodes_returns_false(self):
        """A node with no tool calls is not subject to this check."""
        gate = self._make_gate()
        g = TaskGraph()
        add_node(g, "t", metadata={"description": "t", "output": [
            {"name": "report", "type": "document", "description": "r"}
        ]})
        assert gate._all_tool_calls_failed(g.nodes["t"], g.get_snapshot()) is False

    def test_search_skipped_prefix_counts_as_failed(self):
        gate = self._make_gate()
        node, snap = self._make_node_with_steps("t", [
            ["SEARCH SKIPPED: query contained only placeholder values"],
        ])
        assert gate._all_tool_calls_failed(node, snap) is True

    def test_all_no_results_returns_true(self):
        gate = self._make_gate()
        node, snap = self._make_node_with_steps("t", [
            ["No results found for: current salary unknown"],
            ["No results found for: average salary unknown"],
        ])
        assert gate._all_tool_calls_failed(node, snap) is True

    # ── Integration: verify_result uses the pre-check ─────────────────────────

    def test_verify_result_fails_when_all_tool_calls_errored(self):
        """Even if the LLM verifier would say 'satisfied', the deterministic
        pre-check should veto a result when all searches failed."""
        # LLM would pass it
        llm = FakeLLM(json.dumps({"satisfied": True, "reason": "looks substantive"}))
        gate = QualityGate(llm_client=llm)
        node, snap = self._make_node_with_steps("t", [
            ["ERROR: web search failed — Body collection error: error decoding response body"],
            ["No results found for: market salary rates"],
        ])
        # Update snap node to have declared outputs so verify_result proceeds
        # (make_node_with_steps already sets output on the parent node)
        ok, reason = gate.verify_result(node, "Salary is between $60k-$90k.", snap)
        assert ok is False
        assert "fabricated" in reason.lower() or "errors" in reason.lower()
        # The LLM verifier should not have been called
        assert llm.calls == []

    def test_verify_result_proceeds_to_llm_when_searches_had_data(self):
        """If at least one search returned real data, the LLM verifier runs."""
        llm = FakeLLM(json.dumps({"satisfied": True, "reason": "good data"}))
        gate = QualityGate(llm_client=llm)
        node, snap = self._make_node_with_steps("t", [
            ["Title: Salary Survey\nURL: levels.fyi\nSnippet: median $130k"],
        ])
        ok, reason = gate.verify_result(node, "Based on levels.fyi the median is $130k.", snap)
        assert ok is True
        assert len(llm.calls) == 1


# ── Issue 5: max_retries cap + exponential backoff ────────────────────────────

class TestMaxRetriesAndBackoff:

    def _make_future(self, result):
        f = Future()
        f.set_result(result)
        return f

    def _run_verification_failures(self, orch, g, node_id, count):
        """Drive _on_node_done `count` times with a failing verification."""
        for _ in range(count):
            if g.nodes[node_id].status in ("failed",):
                # Simulate the orchestrator's RESET_NODE putting it back to ready
                g.nodes[node_id].status = "running"
            orch._on_node_done(node_id, self._make_future("template output"))

    def test_node_permanently_failed_after_max_retries(self):
        orch, g, _, mock_gate = make_orchestrator(
            gate_satisfied=False, max_retries=3
        )
        add_node(g, "t", node_type="task", metadata={
            "description": "task", "output": [
                {"name": "report", "type": "document", "description": "r"}
            ]
        })
        g.nodes["t"].status = "running"

        # Drive 3 failures (0-indexed: attempts 0, 1, 2 all fail)
        for attempt in range(3):
            g.nodes["t"].status = "running"
            orch._on_node_done("t", self._make_future("bad output"))

        # After max_retries failures the node must stay failed — not reset.
        # retry_count holds the number of resets-for-retry, which is
        # max_retries - 1: the permanent-fail branch fires before incrementing.
        assert g.nodes["t"].status == "failed"
        assert g.nodes["t"].metadata.get("retry_count", 0) == 2  # (max_retries - 1)

    def test_node_resets_below_max_retries(self):
        """Before the cap is hit the node should be reset to pending/ready."""
        orch, g, _, _ = make_orchestrator(gate_satisfied=False, max_retries=5)
        add_node(g, "t", node_type="task", metadata={
            "description": "task",
            "output": [{"name": "r", "type": "document", "description": "r"}]
        })
        g.nodes["t"].status = "running"
        orch._on_node_done("t", self._make_future("bad output"))

        # After one failure (retry_count=1, max_retries=5) the node is reset
        assert g.nodes["t"].status in ("pending", "ready")

    def test_retry_after_is_set_on_failure(self):
        """Each verification failure must stamp retry_after in metadata."""
        orch, g, _, _ = make_orchestrator(gate_satisfied=False, max_retries=5)
        add_node(g, "t", node_type="task", metadata={
            "description": "task",
            "output": [{"name": "r", "type": "document", "description": "r"}]
        })
        g.nodes["t"].status = "running"
        before = time.time()
        orch._on_node_done("t", self._make_future("bad output"))
        after = time.time()

        retry_after = g.nodes["t"].metadata.get("retry_after", 0)
        assert retry_after > before, "retry_after should be in the future"
        # First retry: backoff = 2**0 = 1s (retry_count was 0 before this call)
        assert retry_after <= after + 2.0, "backoff should not exceed ~1s for first retry"

    def test_backoff_increases_with_each_failure(self):
        """retry_after should grow across successive failures."""
        orch, g, _, _ = make_orchestrator(gate_satisfied=False, max_retries=10)
        add_node(g, "t", node_type="task", metadata={
            "description": "task",
            "output": [{"name": "r", "type": "document", "description": "r"}]
        })

        backoffs = []
        for _ in range(4):
            g.nodes["t"].status = "running"
            before = time.time()
            orch._on_node_done("t", self._make_future("bad output"))
            ra = g.nodes["t"].metadata.get("retry_after", 0)
            backoffs.append(ra - before)

        # Each successive backoff should be larger than the previous
        for i in range(1, len(backoffs)):
            assert backoffs[i] > backoffs[i - 1], (
                f"backoff[{i}]={backoffs[i]:.2f} not > backoff[{i-1}]={backoffs[i-1]:.2f}"
            )

    def test_execution_pass_skips_node_in_backoff_window(self):
        """A ready node whose retry_after hasn't elapsed must not be launched."""
        orch, g, mock_executor, _ = make_orchestrator()
        add_node(g, "t", node_type="task", metadata={"description": "t", "output": []})
        g.nodes["t"].metadata["retry_after"] = time.time() + 60  # 60s from now
        g.nodes["t"].status = "ready"

        launched = orch._execution_pass()
        assert launched == 0
        mock_executor.execute.assert_not_called()

    def test_execution_pass_launches_node_after_backoff_expires(self):
        """Once retry_after is in the past the node is launched normally."""
        orch, g, mock_executor, _ = make_orchestrator()
        add_node(g, "t", node_type="task", metadata={"description": "t", "output": []})
        g.nodes["t"].metadata["retry_after"] = time.time() - 1  # already elapsed
        g.nodes["t"].status = "ready"

        launched = orch._execution_pass()
        assert launched == 1

    def test_max_retries_zero_fails_immediately(self):
        """max_retries=0 means the first failure is permanent."""
        orch, g, _, _ = make_orchestrator(gate_satisfied=False, max_retries=0)
        add_node(g, "t", node_type="task", metadata={
            "description": "task",
            "output": [{"name": "r", "type": "document", "description": "r"}]
        })
        g.nodes["t"].status = "running"
        orch._on_node_done("t", self._make_future("output"))

        assert g.nodes["t"].status == "failed"
        assert g.nodes["t"].metadata.get("retry_count", 0) == 0


# ── Issue 6: tool error strings flagged correctly ─────────────────────────────

class TestToolErrorStringDetection:

    def test_error_string_sets_error_flag_on_reporter(self):
        """When a tool returns 'ERROR: ...' without raising, the step node
        in the graph must receive status='error' (i.e. MARK_FAILED)."""
        g = TaskGraph()
        add_node(g, "parent", node_type="task", metadata={
            "description": "research", "output": []
        })
        g.nodes["parent"].status = "running"
        reporter = make_reporter("parent", g)

        def error_tool(args):
            return "ERROR: web search failed — Body collection error: error decoding response body"

        registry = make_tool_registry({"web_search": error_tool})

        turns = [
            tool_response("web_search", {"query": "salary data"}),
            done_response("done"),
        ]
        idx = [0]
        def responses(prompt, schema=None):
            r = turns[idx[0]]
            idx[0] = min(idx[0] + 1, len(turns) - 1)
            return r

        llm = FakeLLM(responses)
        executor = LLMExecutor(llm_client=llm, tool_registry=registry, max_turns=5)
        node, snapshot = g.nodes["parent"], g.get_snapshot()
        executor.execute(node, snapshot, reporter=reporter)

        step_id = "parent__step_web_search"
        assert step_id in g.nodes, "step node should be in graph"
        step_node = g.nodes[step_id]
        attempts = step_node.metadata.get("attempts", [])
        assert attempts, "step node should have recorded attempts"
        assert attempts[0]["status"] == "error", (
            f"Expected status='error', got {attempts[0]['status']!r}"
        )

    def test_non_error_string_keeps_ok_status(self):
        """A tool that returns a normal result must still get status='ok'."""
        g = TaskGraph()
        add_node(g, "parent", node_type="task", metadata={
            "description": "research", "output": []
        })
        g.nodes["parent"].status = "running"
        reporter = make_reporter("parent", g)

        def good_tool(args):
            return "Title: Glassdoor salary survey\nURL: glassdoor.com\nSnippet: $120k avg"

        registry = make_tool_registry({"web_search": good_tool})

        turns = [
            tool_response("web_search", {"query": "salary data"}),
            done_response("The median is $120k."),
        ]
        idx = [0]
        def responses(prompt, schema=None):
            r = turns[idx[0]]
            idx[0] = min(idx[0] + 1, len(turns) - 1)
            return r

        llm = FakeLLM(responses)
        executor = LLMExecutor(llm_client=llm, tool_registry=registry, max_turns=5)
        node, snapshot = g.nodes["parent"], g.get_snapshot()
        executor.execute(node, snapshot, reporter=reporter)

        step_id = "parent__step_web_search"
        assert step_id in g.nodes
        attempts = g.nodes[step_id].metadata.get("attempts", [])
        assert attempts[0]["status"] == "ok"

    def test_exception_still_sets_error_flag(self):
        """The original path (tool raises) must still mark status='error'."""
        g = TaskGraph()
        add_node(g, "parent", node_type="task", metadata={
            "description": "research", "output": []
        })
        g.nodes["parent"].status = "running"
        reporter = make_reporter("parent", g)

        def exploding_tool(args):
            raise RuntimeError("connection refused")

        registry = make_tool_registry({"web_search": exploding_tool})

        turns = [
            tool_response("web_search", {"query": "salary"}),
            done_response("fallback"),
        ]
        idx = [0]
        def responses(prompt, schema=None):
            r = turns[idx[0]]
            idx[0] = min(idx[0] + 1, len(turns) - 1)
            return r

        llm = FakeLLM(responses)
        executor = LLMExecutor(llm_client=llm, tool_registry=registry, max_turns=5)
        executor.execute(g.nodes["parent"], g.get_snapshot(), reporter=reporter)

        attempts = g.nodes["parent__step_web_search"].metadata.get("attempts", [])
        assert attempts[0]["status"] == "error"

    def test_issue6_and_issue4_integration(self):
        """End-to-end: tool returns ERROR string → error flag set on step node
        → _all_tool_calls_failed detects it → verify_result rejects the output
        without calling the LLM verifier."""
        # Build graph with a parent node and a step that recorded an error attempt
        g = TaskGraph()
        add_node(g, "salary_node", node_type="task", metadata={
            "description": "Research salary",
            "output": [{"name": "report", "type": "document", "description": "salary report"}],
        })
        step_id = "salary_node__step_web_search"
        add_node(g, step_id, node_type="execution_step", metadata={
            "step_type": "tool_call",
            "tool_name": "web_search",
            "attempts": [
                {
                    "turn": 0,
                    "args": {"query": "salary"},
                    "result": "ERROR: web search failed — Body collection error: error decoding response body",
                    "status": "error",  # correctly set by fix 6
                }
            ],
        })

        llm = FakeLLM(json.dumps({"satisfied": True, "reason": "looks great"}))
        gate = QualityGate(llm_client=llm)

        ok, reason = gate.verify_result(
            g.nodes["salary_node"],
            "Salaries range from $60k to $90k based on national averages.",
            g.get_snapshot(),
        )

        assert ok is False
        assert llm.calls == [], "LLM verifier must not have been called"