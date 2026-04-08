"""Tests for cuddlytoddly.core.events and cuddlytoddly.core.reducer."""

from conftest import add_node, mark_done

from cuddlytoddly.core.events import (
    ADD_DEPENDENCY,
    ADD_NODE,
    DETACH_NODE,
    MARK_DONE,
    MARK_FAILED,
    MARK_RUNNING,
    REMOVE_DEPENDENCY,
    REMOVE_NODE,
    RESET_NODE,
    SET_NODE_TYPE,
    SET_RESULT,
    UPDATE_METADATA,
    UPDATE_STATUS,
    Event,
)
from cuddlytoddly.core.reducer import apply_event
from cuddlytoddly.core.task_graph import TaskGraph

# ── Event serialization ───────────────────────────────────────────────────────


class TestEvent:
    def test_to_dict_round_trips(self):
        e = Event("ADD_NODE", {"node_id": "x"})
        d = e.to_dict()
        e2 = Event.from_dict(d)
        assert e2.type == e.type
        assert e2.payload == e.payload

    def test_timestamp_auto_set(self):
        e = Event("ADD_NODE", {})
        assert e.timestamp is not None

    def test_explicit_timestamp(self):
        e = Event("ADD_NODE", {}, timestamp="2025-01-01T00:00:00")
        assert e.timestamp == "2025-01-01T00:00:00"

    def test_from_dict_preserves_timestamp(self):
        ts = "2025-06-15T12:00:00"
        e = Event.from_dict({"type": "X", "payload": {}, "timestamp": ts})
        assert e.timestamp == ts


# ── apply_event: structural events ───────────────────────────────────────────


class TestReducerStructural:
    def test_add_node(self):
        g = TaskGraph()
        apply_event(
            g,
            Event(
                ADD_NODE,
                {
                    "node_id": "a",
                    "node_type": "task",
                    "dependencies": [],
                    "metadata": {"description": "A"},
                },
            ),
        )
        assert "a" in g.nodes
        assert g.nodes["a"].node_type == "task"

    def test_add_node_idempotent_preserves_existing_description(self):
        g = TaskGraph()
        apply_event(
            g,
            Event(
                ADD_NODE,
                {
                    "node_id": "a",
                    "node_type": "task",
                    "dependencies": [],
                    "metadata": {"description": "original"},
                },
            ),
        )
        apply_event(
            g,
            Event(
                ADD_NODE,
                {
                    "node_id": "a",
                    "node_type": "task",
                    "dependencies": [],
                    "metadata": {"description": "overwrite attempt"},
                },
            ),
        )
        assert g.nodes["a"].metadata["description"] == "original"

    def test_remove_node(self):
        g = TaskGraph()
        add_node(g, "a")
        apply_event(g, Event(REMOVE_NODE, {"node_id": "a"}))
        assert "a" not in g.nodes

    def test_add_dependency(self):
        g = TaskGraph()
        add_node(g, "a")
        add_node(g, "b")
        apply_event(g, Event(ADD_DEPENDENCY, {"node_id": "b", "depends_on": "a"}))
        assert "a" in g.nodes["b"].dependencies

    def test_remove_dependency(self):
        g = TaskGraph()
        add_node(g, "a")
        add_node(g, "b", deps=["a"])
        apply_event(g, Event(REMOVE_DEPENDENCY, {"node_id": "b", "depends_on": "a"}))
        assert "a" not in g.nodes["b"].dependencies

    def test_set_node_type(self):
        g = TaskGraph()
        add_node(g, "a")
        apply_event(g, Event(SET_NODE_TYPE, {"node_id": "a", "node_type": "goal"}))
        assert g.nodes["a"].node_type == "goal"

    def test_structural_event_bumps_structure_version(self):
        g = TaskGraph()
        v = g.structure_version
        apply_event(
            g,
            Event(
                ADD_NODE,
                {
                    "node_id": "x",
                    "node_type": "task",
                    "dependencies": [],
                    "metadata": {},
                },
            ),
        )
        assert g.structure_version > v

    def test_insert_node_alias(self):
        """INSERT_NODE is an alias for ADD_NODE."""
        g = TaskGraph()
        apply_event(
            g,
            Event(
                "INSERT_NODE",
                {
                    "node_id": "x",
                    "node_type": "task",
                    "dependencies": [],
                    "metadata": {},
                },
            ),
        )
        assert "x" in g.nodes


# ── apply_event: execution events ────────────────────────────────────────────


class TestReducerExecution:
    def test_mark_running(self):
        g = TaskGraph()
        add_node(g, "a")
        apply_event(g, Event(MARK_RUNNING, {"node_id": "a"}))
        assert g.nodes["a"].status == "running"

    def test_mark_done(self):
        g = TaskGraph()
        add_node(g, "a")
        apply_event(g, Event(MARK_DONE, {"node_id": "a", "result": "output"}))
        assert g.nodes["a"].status == "done"
        assert g.nodes["a"].result == "output"

    def test_mark_failed(self):
        g = TaskGraph()
        add_node(g, "a")
        apply_event(g, Event(MARK_FAILED, {"node_id": "a"}))
        assert g.nodes["a"].status == "failed"

    def test_reset_node_with_unmet_dep_stays_pending(self):
        """After reset, a node whose dependency is not done stays pending."""
        g = TaskGraph()
        add_node(g, "a")
        add_node(g, "b", deps=["a"])
        g.nodes["b"].status = "done"
        g.nodes["b"].result = "old result"
        apply_event(g, Event(RESET_NODE, {"node_id": "b"}))
        # a is not done, so b cannot become ready → stays pending
        assert g.nodes["b"].status == "pending"
        assert g.nodes["b"].result is None

    def test_reset_node_no_deps_becomes_ready(self):
        """A depless node that is reset immediately becomes ready again."""
        g = TaskGraph()
        add_node(g, "a")
        mark_done(g, "a", result="old result")
        apply_event(g, Event(RESET_NODE, {"node_id": "a"}))
        assert g.nodes["a"].status == "ready"
        assert g.nodes["a"].result is None

    def test_set_result(self):
        g = TaskGraph()
        add_node(g, "a")
        apply_event(g, Event(SET_RESULT, {"node_id": "a", "result": "data"}))
        assert g.nodes["a"].result == "data"

    def test_update_status(self):
        g = TaskGraph()
        add_node(g, "a")
        apply_event(g, Event(UPDATE_STATUS, {"node_id": "a", "status": "running"}))
        assert g.nodes["a"].status == "running"

    def test_execution_event_bumps_execution_version(self):
        g = TaskGraph()
        add_node(g, "a")
        v = g.execution_version
        apply_event(g, Event(MARK_DONE, {"node_id": "a", "result": "r"}))
        assert g.execution_version > v


# ── apply_event: metadata ─────────────────────────────────────────────────────


class TestReducerMetadata:
    def test_update_metadata_merges(self):
        g = TaskGraph()
        add_node(g, "a", metadata={"description": "old", "key": "val"})
        apply_event(g, Event(UPDATE_METADATA, {"node_id": "a", "metadata": {"key": "new_val"}}))
        assert g.nodes["a"].metadata["key"] == "new_val"
        assert g.nodes["a"].metadata["description"] == "old"

    def test_update_metadata_preserves_existing_description(self):
        g = TaskGraph()
        add_node(g, "a", metadata={"description": "keep me"})
        apply_event(
            g,
            Event(
                UPDATE_METADATA,
                {
                    "node_id": "a",
                    "metadata": {"description": "overwrite", "other": "x"},
                },
            ),
        )
        assert g.nodes["a"].metadata["description"] == "keep me"

    def test_update_metadata_user_can_overwrite_description(self):
        g = TaskGraph()
        add_node(g, "a", metadata={"description": "old"})
        apply_event(
            g,
            Event(
                UPDATE_METADATA,
                {"node_id": "a", "origin": "user", "metadata": {"description": "new"}},
            ),
        )
        assert g.nodes["a"].metadata["description"] == "new"

    def test_update_metadata_missing_node_is_noop(self):
        g = TaskGraph()
        apply_event(g, Event(UPDATE_METADATA, {"node_id": "ghost", "metadata": {}}))


# ── apply_event: detach / readiness recompute ─────────────────────────────────


class TestReducerDetach:
    def test_detach_node(self):
        g = TaskGraph()
        add_node(g, "a")
        add_node(g, "b", deps=["a"])
        apply_event(g, Event(DETACH_NODE, {"node_id": "b"}))
        assert "b" not in g.nodes
        assert "a" in g.nodes

    def test_readiness_recomputed_after_each_event(self):
        g = TaskGraph()
        add_node(g, "a")
        add_node(g, "b", deps=["a"])
        assert g.nodes["b"].status == "pending"
        apply_event(g, Event(MARK_DONE, {"node_id": "a", "result": "r"}))
        assert g.nodes["b"].status == "ready"


# ── apply_event: event log integration ───────────────────────────────────────


class TestReducerEventLog:
    def test_event_appended_to_log(self, tmp_path):
        from cuddlytoddly.infra.event_log import EventLog

        log = EventLog(str(tmp_path / "events.jsonl"))
        g = TaskGraph()
        apply_event(
            g,
            Event(
                ADD_NODE,
                {
                    "node_id": "a",
                    "node_type": "task",
                    "dependencies": [],
                    "metadata": {},
                },
            ),
            event_log=log,
        )
        events = list(log.replay())
        assert len(events) == 1
        assert events[0].type == ADD_NODE

    def test_none_event_log_does_not_crash(self):
        g = TaskGraph()
        apply_event(
            g,
            Event(
                ADD_NODE,
                {
                    "node_id": "a",
                    "node_type": "task",
                    "dependencies": [],
                    "metadata": {},
                },
            ),
            event_log=None,
        )
        assert "a" in g.nodes
