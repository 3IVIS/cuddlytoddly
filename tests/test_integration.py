"""
Integration tests: end-to-end flows through multiple components.

These tests wire together real components (graph, reducer, event log,
validator, planner stub) rather than individual units.
"""
import json
import time
import threading
import pytest
from unittest.mock import MagicMock

from cuddlytoddly.core.task_graph import TaskGraph
from cuddlytoddly.core.events import Event, ADD_NODE, MARK_DONE
from cuddlytoddly.core.reducer import apply_event
from cuddlytoddly.infra.event_log import EventLog
from cuddlytoddly.infra.event_queue import EventQueue
from cuddlytoddly.infra.replay import rebuild_graph_from_log
from cuddlytoddly.planning.llm_output_validator import LLMOutputValidator
from cuddlytoddly.engine.llm_orchestrator import Orchestrator
from conftest import FakeLLM, add_node, mark_done


# ── Crash-and-resume ──────────────────────────────────────────────────────────

class TestCrashAndResume:
    def test_graph_fully_restored_from_event_log(self, tmp_path):
        """Write events to a log, then replay them into a fresh graph."""
        log = EventLog(str(tmp_path / "events.jsonl"))
        g = TaskGraph()
        for node_id, deps in [("a", []), ("b", ["a"]), ("c", ["b"])]:
            apply_event(g, Event(ADD_NODE, {
                "node_id": node_id, "node_type": "task",
                "dependencies": deps, "metadata": {},
            }), event_log=log)
        apply_event(g, Event(MARK_DONE, {"node_id": "a", "result": "a done"}),
                    event_log=log)

        restored = rebuild_graph_from_log(log)
        assert set(restored.nodes.keys()) == {"a", "b", "c"}
        assert restored.nodes["a"].status == "done"
        assert restored.nodes["a"].result == "a done"
        assert "a" in restored.nodes["b"].dependencies

    def test_partial_completion_preserved_across_replay(self, tmp_path):
        log = EventLog(str(tmp_path / "events.jsonl"))
        g = TaskGraph()
        for node_id in ["t1", "t2", "t3"]:
            apply_event(g, Event(ADD_NODE, {
                "node_id": node_id, "node_type": "task",
                "dependencies": [], "metadata": {},
            }), event_log=log)
        apply_event(g, Event(MARK_DONE, {"node_id": "t1", "result": "done"}),
                    event_log=log)
        apply_event(g, Event(MARK_DONE, {"node_id": "t2", "result": "done"}),
                    event_log=log)

        restored = rebuild_graph_from_log(log)
        assert restored.nodes["t1"].status == "done"
        assert restored.nodes["t2"].status == "done"
        assert restored.nodes["t3"].status == "ready"

    def test_large_event_log_replays_correctly(self, tmp_path):
        log = EventLog(str(tmp_path / "events.jsonl"))
        g = TaskGraph()
        for i in range(100):
            apply_event(g, Event(ADD_NODE, {
                "node_id": f"task_{i}", "node_type": "task",
                "dependencies": [f"task_{i-1}"] if i > 0 else [],
                "metadata": {"description": f"task {i}"},
            }), event_log=log)
        for i in range(50):
            apply_event(g, Event(MARK_DONE, {
                "node_id": f"task_{i}", "result": f"result {i}",
            }), event_log=log)

        restored = rebuild_graph_from_log(log)
        assert len(restored.nodes) == 100
        done_count = sum(1 for n in restored.nodes.values() if n.status == "done")
        assert done_count == 50


# ── Validator → graph round-trip ──────────────────────────────────────────────

class TestValidatorGraphRoundTrip:
    def test_validated_events_applied_to_graph(self):
        g = TaskGraph()
        validator = LLMOutputValidator(g)
        raw_events = [
            {"type": ADD_NODE, "payload": {
                "node_id": "task_a", "node_type": "task",
                "dependencies": [], "metadata": {"description": "A"},
            }},
            {"type": ADD_NODE, "payload": {
                "node_id": "task_b", "node_type": "task",
                "dependencies": ["task_a"],
                "metadata": {"description": "B"},
            }},
        ]
        safe_events = validator.validate_and_normalize(raw_events, "planning")
        for evt in safe_events:
            apply_event(g, Event(evt["type"], evt["payload"]))
        assert "task_a" in g.nodes
        assert "task_b" in g.nodes
        assert "task_a" in g.nodes["task_b"].dependencies

    def test_bad_events_filtered_good_ones_applied(self):
        g = TaskGraph()
        validator = LLMOutputValidator(g)
        raw_events = [
            {"type": ADD_NODE, "payload": {
                "node_id": "good_task", "node_type": "task",
                "dependencies": [], "metadata": {},
            }},
            {"type": ADD_NODE, "payload": {
                # missing node_id — should be rejected
                "node_type": "task", "dependencies": [],
            }},
        ]
        safe_events = validator.validate_and_normalize(raw_events, "planning")
        for evt in safe_events:
            apply_event(g, Event(evt["type"], evt["payload"]))
        assert "good_task" in g.nodes
        assert len(g.nodes) == 1


# ── Orchestrator → execution flow ─────────────────────────────────────────────

class TestOrchestratorExecutionFlow:
    def test_single_task_runs_and_completes(self):
        g = TaskGraph()
        add_node(g, "solo_task")

        mock_executor = MagicMock()
        mock_executor.execute.return_value = "task output"
        mock_gate = MagicMock()
        mock_gate.verify_result.return_value = (True, "ok")
        mock_gate.check_dependencies.return_value = None
        mock_planner = MagicMock()
        mock_planner.propose.return_value = []

        orch = Orchestrator(
            graph=g, planner=mock_planner, executor=mock_executor,
            quality_gate=mock_gate, event_queue=EventQueue(), max_workers=1,
        )
        orch.start()

        deadline = time.time() + 5.0
        while time.time() < deadline:
            if g.nodes.get("solo_task", None) and g.nodes["solo_task"].status == "done":
                break
            time.sleep(0.05)

        orch.stop()
        assert g.nodes["solo_task"].status == "done"
        assert g.nodes["solo_task"].result == "task output"

    def test_failed_verification_retries_node(self):
        """Verify that a node retried after quality gate rejects it."""
        g = TaskGraph()
        add_node(g, "retry_task")

        call_count = [0]
        def execute_fn(node, snapshot, reporter=None):
            call_count[0] += 1
            return "output"

        verify_call_count = [0]
        def verify_fn(node, result, snapshot):
            verify_call_count[0] += 1
            if verify_call_count[0] < 2:
                return False, "not good enough"
            return True, "ok now"

        mock_executor = MagicMock()
        mock_executor.execute = execute_fn
        mock_gate = MagicMock()
        mock_gate.verify_result = verify_fn
        mock_gate.check_dependencies.return_value = None
        mock_planner = MagicMock()
        mock_planner.propose.return_value = []

        orch = Orchestrator(
            graph=g, planner=mock_planner, executor=mock_executor,
            quality_gate=mock_gate, event_queue=EventQueue(), max_workers=1,
        )
        orch.start()

        deadline = time.time() + 8.0
        while time.time() < deadline:
            if g.nodes.get("retry_task") and g.nodes["retry_task"].status == "done":
                break
            time.sleep(0.05)

        orch.stop()
        assert g.nodes["retry_task"].status == "done"
        assert call_count[0] >= 2  # was retried

    def test_chain_executes_in_order(self):
        """task_a must complete before task_b starts."""
        g = TaskGraph()
        add_node(g, "task_a")
        add_node(g, "task_b", deps=["task_a"])

        execution_order = []
        lock = threading.Lock()

        def execute_fn(node, snapshot, reporter=None):
            with lock:
                execution_order.append(node.id)
            time.sleep(0.05)
            return f"{node.id} result"

        mock_executor = MagicMock()
        mock_executor.execute = execute_fn
        mock_gate = MagicMock()
        mock_gate.verify_result.return_value = (True, "ok")
        mock_gate.check_dependencies.return_value = None
        mock_planner = MagicMock()
        mock_planner.propose.return_value = []

        orch = Orchestrator(
            graph=g, planner=mock_planner, executor=mock_executor,
            quality_gate=mock_gate, event_queue=EventQueue(), max_workers=2,
        )
        orch.start()

        deadline = time.time() + 8.0
        while time.time() < deadline:
            if (g.nodes.get("task_b") and g.nodes["task_b"].status == "done"):
                break
            time.sleep(0.05)

        orch.stop()
        assert execution_order.index("task_a") < execution_order.index("task_b")


# ── Full planning + execution stub ────────────────────────────────────────────

class TestPlannerExecutorStub:
    def test_planner_expands_goal_then_executor_runs_tasks(self):
        g = TaskGraph()
        add_node(g, "my_goal", node_type="goal",
                 metadata={"description": "achieve greatness", "expanded": False})

        def planner_propose(context):
            return [
                {
                    "type": ADD_NODE,
                    "payload": {
                        "node_id": "generated_task",
                        "node_type": "task",
                        "dependencies": [],
                        "origin": "planning",
                        "metadata": {"description": "auto task"},
                    }
                }
            ]

        mock_planner = MagicMock()
        mock_planner.propose = planner_propose

        mock_executor = MagicMock()
        mock_executor.execute.return_value = "task done"
        mock_gate = MagicMock()
        mock_gate.verify_result.return_value = (True, "ok")
        mock_gate.check_dependencies.return_value = None

        orch = Orchestrator(
            graph=g, planner=mock_planner, executor=mock_executor,
            quality_gate=mock_gate, event_queue=EventQueue(), max_workers=1,
        )
        # MagicMock attributes are truthy — clear the client list so
        # llm_stopped returns False and the orchestrator isn't immediately paused.
        orch._llm_clients = []
        orch.start()

        deadline = time.time() + 8.0
        while time.time() < deadline:
            if "generated_task" in g.nodes and g.nodes["generated_task"].status == "done":
                break
            time.sleep(0.05)

        orch.stop()
        assert "generated_task" in g.nodes
        assert g.nodes["generated_task"].status == "done"
