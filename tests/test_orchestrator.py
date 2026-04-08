"""Tests for cuddlytoddly.engine.llm_orchestrator.Orchestrator."""

from unittest.mock import MagicMock

from conftest import FakeLLM, mark_done

from cuddlytoddly.core.events import ADD_NODE, RESET_SUBTREE, Event
from cuddlytoddly.core.task_graph import TaskGraph
from cuddlytoddly.engine.llm_orchestrator import Orchestrator
from cuddlytoddly.infra.event_queue import EventQueue

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_orchestrator(
    graph=None, planner=None, executor=None, quality_gate=None, max_workers=1
):
    g = graph or TaskGraph()
    mock_planner = planner or MagicMock()
    mock_planner.propose.return_value = []
    mock_executor = executor or MagicMock()
    mock_executor.execute.return_value = "mock result"
    mock_gate = quality_gate or MagicMock()
    mock_gate.verify_result.return_value = (True, "ok")
    mock_gate.check_dependencies.return_value = None

    orch = Orchestrator(
        graph=g,
        planner=mock_planner,
        executor=mock_executor,
        quality_gate=mock_gate,
        event_queue=EventQueue(),
        max_workers=max_workers,
    )
    # MagicMock attributes are truthy, so any mock with a .llm attribute makes
    # llm_stopped return True. Clear the client list so tests start unpaused.
    orch._llm_clients = []
    return orch, g, mock_planner, mock_executor, mock_gate


# ── Lifecycle ─────────────────────────────────────────────────────────────────


class TestOrchestratorLifecycle:
    def test_start_creates_background_thread(self):
        orch, *_ = make_orchestrator()
        orch.start()
        assert orch._thread is not None
        assert orch._thread.is_alive()
        orch.stop()

    def test_stop_sets_stop_event(self):
        orch, *_ = make_orchestrator()
        orch.start()
        orch.stop()
        assert orch._stop_event.is_set()

    def test_is_running_reflects_state(self):
        orch, *_ = make_orchestrator()
        orch.start()
        assert orch.is_running
        orch.stop()
        assert not orch.is_running


# ── LLM pause / resume ────────────────────────────────────────────────────────


class TestLLMPauseResume:
    def test_llm_stopped_false_initially(self):
        orch, *_ = make_orchestrator()
        # _llm_clients is empty — any() of empty is False
        assert not orch.llm_stopped

    def test_stop_llm_calls_sets_stopped(self):
        orch, *_ = make_orchestrator()
        llm = FakeLLM("{}")
        orch._llm_clients = [llm]
        orch.stop_llm_calls()
        assert orch.llm_stopped

    def test_resume_llm_calls_clears_stopped(self):
        orch, *_ = make_orchestrator()
        llm = FakeLLM("{}")
        orch._llm_clients = [llm]
        orch.stop_llm_calls()
        orch.resume_llm_calls()
        assert not orch.llm_stopped


# ── User-facing graph mutations ───────────────────────────────────────────────


class TestOrchestratorGraphMutations:
    def test_add_goal(self):
        orch, g, *_ = make_orchestrator()
        orch.add_goal("my_goal", description="do stuff")
        assert "my_goal" in g.nodes
        assert g.nodes["my_goal"].node_type == "goal"

    def test_add_task(self):
        orch, g, *_ = make_orchestrator()
        orch.add_task("my_task", description="a task")
        assert "my_task" in g.nodes
        assert g.nodes["my_task"].node_type == "task"

    def test_add_task_with_dependencies(self):
        orch, g, *_ = make_orchestrator()
        orch.add_task("dep")
        orch.add_task("child", dependencies=["dep"])
        assert "dep" in g.nodes["child"].dependencies

    def test_remove_node(self):
        orch, g, *_ = make_orchestrator()
        orch.add_task("removeme")
        orch.remove_node("removeme")
        assert "removeme" not in g.nodes

    def test_remove_running_node_blocked(self):
        orch, g, *_ = make_orchestrator()
        orch.add_task("running_task")
        g.nodes["running_task"].status = "running"
        orch._running_futures["running_task"] = MagicMock()
        orch.remove_node("running_task")
        assert "running_task" in g.nodes

    def test_add_dependency(self):
        orch, g, *_ = make_orchestrator()
        orch.add_task("a")
        orch.add_task("b")
        orch.add_dependency("b", "a")
        assert "a" in g.nodes["b"].dependencies

    def test_remove_dependency(self):
        orch, g, *_ = make_orchestrator()
        orch.add_task("a")
        orch.add_task("b", dependencies=["a"])
        orch.remove_dependency("b", "a")
        assert "a" not in g.nodes["b"].dependencies

    def test_retry_node_with_unmet_dep_resets_to_pending(self):
        """A retried node whose dependency is not done goes back to pending."""
        orch, g, *_ = make_orchestrator()
        orch.add_task("dep")
        orch.add_task("failed_task", dependencies=["dep"])
        g.nodes["failed_task"].status = "failed"
        g.nodes["failed_task"].result = "bad"
        orch.retry_node("failed_task")
        # dep is not done → failed_task has unmet dep → pending
        assert g.nodes["failed_task"].status == "pending"
        assert g.nodes["failed_task"].result is None

    def test_retry_node_no_dep_becomes_ready(self):
        """A retried node with no dependencies becomes ready immediately."""
        orch, g, *_ = make_orchestrator()
        orch.add_task("solo_task")
        g.nodes["solo_task"].status = "failed"
        g.nodes["solo_task"].result = "bad"
        orch.retry_node("solo_task")
        assert g.nodes["solo_task"].status == "ready"
        assert g.nodes["solo_task"].result is None

    def test_update_metadata(self):
        orch, g, *_ = make_orchestrator()
        orch.add_task("a")
        orch.update_metadata("a", {"custom_key": "custom_val"})
        assert g.nodes["a"].metadata["custom_key"] == "custom_val"


# ── get_status ────────────────────────────────────────────────────────────────


class TestOrchestratorGetStatus:
    def test_get_status_counts_nodes(self):
        orch, g, *_ = make_orchestrator()
        orch.add_task("a")
        orch.add_task("b")
        status = orch.get_status()
        assert status["total"] == 2

    def test_get_status_by_status(self):
        orch, g, *_ = make_orchestrator()
        orch.add_task("a")
        orch.add_task("b")
        g.nodes["a"].status = "done"
        status = orch.get_status()
        assert status["by_status"].get("done", 0) >= 1

    def test_get_status_running_nodes(self):
        orch, g, *_ = make_orchestrator()
        orch.add_task("a")
        orch._running_futures["a"] = MagicMock()
        status = orch.get_status()
        assert "a" in status["running_nodes"]


# ── get_snapshot ──────────────────────────────────────────────────────────────


class TestOrchestratorSnapshot:
    def test_get_snapshot_returns_nodes(self):
        orch, g, *_ = make_orchestrator()
        orch.add_task("a")
        snap = orch.get_snapshot()
        assert "a" in snap

    def test_snapshot_is_deep_copy(self):
        orch, g, *_ = make_orchestrator()
        orch.add_task("a")
        snap = orch.get_snapshot()
        snap["a"].status = "running"
        assert g.nodes["a"].status != "running"


# ── Event queue drain ─────────────────────────────────────────────────────────


class TestEventQueueDrain:
    def test_queued_add_node_event_processed(self):
        orch, g, *_ = make_orchestrator()
        orch.event_queue.put(
            Event(
                ADD_NODE,
                {
                    "node_id": "queued_node",
                    "node_type": "task",
                    "dependencies": [],
                    "metadata": {"description": "queued"},
                },
            )
        )
        orch._drain_event_queue()
        assert "queued_node" in g.nodes

    def test_reset_subtree_resets_node_and_downstream(self):
        """After RESET_SUBTREE on a:
        - a (no deps) → becomes ready
        - b (dep on a, now not done) → becomes pending
        """
        orch, g, *_ = make_orchestrator()
        orch.add_task("a")
        orch.add_task("b", dependencies=["a"])
        g.nodes["a"].status = "done"
        g.nodes["b"].status = "done"
        orch.event_queue.put(Event(RESET_SUBTREE, {"node_id": "a"}))
        orch._drain_event_queue()
        assert g.nodes["a"].status == "ready"  # no deps → ready after reset
        assert g.nodes["b"].status == "pending"  # dep on non-done a → pending


# ── Goal auto-completion ──────────────────────────────────────────────────────


class TestGoalCompletion:
    def test_goal_marked_done_when_all_deps_done(self):
        orch, g, *_ = make_orchestrator()
        orch.add_task("task_a")
        orch.add_goal("goal_1", description="the goal")
        orch.add_dependency("goal_1", "task_a")
        g.nodes["goal_1"].metadata["expanded"] = True
        mark_done(g, "task_a", result="task result")
        orch._complete_finished_goals()
        assert g.nodes["goal_1"].status == "done"

    def test_goal_not_done_when_dep_still_pending(self):
        orch, g, *_ = make_orchestrator()
        orch.add_task("task_a")
        orch.add_task("task_b")
        orch.add_goal("goal_1", description="the goal")
        orch.add_dependency("goal_1", "task_a")
        orch.add_dependency("goal_1", "task_b")
        g.nodes["goal_1"].metadata["expanded"] = True
        mark_done(g, "task_a")
        orch._complete_finished_goals()
        assert g.nodes["goal_1"].status != "done"

    def test_goal_not_completed_when_not_expanded(self):
        orch, g, *_ = make_orchestrator()
        orch.add_task("task_a")
        orch.add_goal("goal_1")
        g.nodes["goal_1"].metadata["expanded"] = False
        orch.add_dependency("goal_1", "task_a")
        mark_done(g, "task_a")
        orch._complete_finished_goals()
        assert g.nodes["goal_1"].status != "done"


# ── Token counts ──────────────────────────────────────────────────────────────


class TestTokenCounts:
    def test_token_counts_property(self):
        orch, *_ = make_orchestrator()
        counts = orch.token_counts
        assert "prompt" in counts
        assert "completion" in counts
        assert "total" in counts
        assert "calls" in counts

    def test_token_counts_are_numeric(self):
        orch, *_ = make_orchestrator()
        counts = orch.token_counts
        for v in counts.values():
            assert isinstance(v, int)


# ── Integration: execution pass ───────────────────────────────────────────────


class TestExecutionPass:
    def test_ready_task_launched(self):
        orch, g, _, mock_executor, mock_gate = make_orchestrator()
        orch.add_task("ready_task")
        assert g.nodes["ready_task"].status == "ready"
        launched = orch._execution_pass()
        assert launched == 1

    def test_pending_task_not_launched(self):
        """dep is currently running; pending_task has an unmet dep — nothing new launched."""
        orch, g, _, mock_executor, _ = make_orchestrator()
        orch.add_task("dep")
        orch.add_task("pending_task", dependencies=["dep"])
        # Simulate dep already running so it's not re-launched
        g.nodes["dep"].status = "running"
        orch._running_futures["dep"] = MagicMock()
        g.recompute_readiness()
        launched = orch._execution_pass()
        assert launched == 0

    def test_already_running_task_not_launched_twice(self):
        orch, g, _, mock_executor, _ = make_orchestrator()
        orch.add_task("running_task")
        g.nodes["running_task"].status = "running"
        orch._running_futures["running_task"] = MagicMock()
        launched = orch._execution_pass()
        assert launched == 0
