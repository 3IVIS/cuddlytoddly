"""Tests for cuddlytoddly.infra: EventLog, EventQueue, replay."""
import json
import threading
import pytest
from cuddlytoddly.core.events import Event, ADD_NODE, MARK_DONE
from cuddlytoddly.core.task_graph import TaskGraph
from cuddlytoddly.core.reducer import apply_event
from cuddlytoddly.infra.event_log import EventLog
from cuddlytoddly.infra.event_queue import EventQueue
from cuddlytoddly.infra.replay import rebuild_graph_from_log
from conftest import add_node, mark_done


# ── EventLog ──────────────────────────────────────────────────────────────────

class TestEventLog:
    def test_append_and_replay(self, tmp_path):
        log = EventLog(str(tmp_path / "events.jsonl"))
        e = Event(ADD_NODE, {"node_id": "a", "node_type": "task",
                              "dependencies": [], "metadata": {}})
        log.append(e)
        events = list(log.replay())
        assert len(events) == 1
        assert events[0].type == ADD_NODE
        assert events[0].payload["node_id"] == "a"

    def test_multiple_events_in_order(self, tmp_path):
        log = EventLog(str(tmp_path / "events.jsonl"))
        for i in range(5):
            log.append(Event(ADD_NODE, {"node_id": str(i), "node_type": "task",
                                         "dependencies": [], "metadata": {}}))
        events = list(log.replay())
        assert len(events) == 5
        assert [e.payload["node_id"] for e in events] == ["0", "1", "2", "3", "4"]

    def test_replay_skips_corrupt_lines(self, tmp_path):
        path = tmp_path / "events.jsonl"
        log = EventLog(str(path))
        log.append(Event(ADD_NODE, {"node_id": "a", "node_type": "task",
                                     "dependencies": [], "metadata": {}}))
        # Inject a corrupt line
        with path.open("a") as f:
            f.write("this is not json\n")
        log.append(Event(ADD_NODE, {"node_id": "b", "node_type": "task",
                                     "dependencies": [], "metadata": {}}))
        events = list(log.replay())
        assert len(events) == 2  # corrupt line skipped

    def test_embedded_newlines_in_result_do_not_break_replay(self, tmp_path):
        log = EventLog(str(tmp_path / "events.jsonl"))
        result_with_newlines = "line1\nline2\nline3"
        log.append(Event(MARK_DONE, {"node_id": "a", "result": result_with_newlines}))
        events = list(log.replay())
        assert len(events) == 1
        assert events[0].payload["result"] == result_with_newlines

    def test_replay_empty_file(self, tmp_path):
        log = EventLog(str(tmp_path / "events.jsonl"))
        events = list(log.replay())
        assert events == []

    def test_clear_empties_file(self, tmp_path):
        log = EventLog(str(tmp_path / "events.jsonl"))
        log.append(Event(ADD_NODE, {"node_id": "a", "node_type": "task",
                                     "dependencies": [], "metadata": {}}))
        log.clear()
        events = list(log.replay())
        assert events == []

    def test_file_created_on_init(self, tmp_path):
        path = tmp_path / "new_events.jsonl"
        assert not path.exists()
        EventLog(str(path))
        assert path.exists()

    def test_append_preserves_timestamp(self, tmp_path):
        log = EventLog(str(tmp_path / "events.jsonl"))
        e = Event(ADD_NODE, {"node_id": "a"}, timestamp="2025-01-01T00:00:00")
        log.append(e)
        replayed = list(log.replay())
        assert replayed[0].timestamp == "2025-01-01T00:00:00"

    def test_concurrent_appends_all_stored(self, tmp_path):
        log = EventLog(str(tmp_path / "events.jsonl"))
        errors = []

        def writer(node_id):
            try:
                log.append(Event(ADD_NODE, {"node_id": node_id, "node_type": "task",
                                             "dependencies": [], "metadata": {}}))
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(str(i),)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        events = list(log.replay())
        assert len(events) == 20


# ── EventQueue ────────────────────────────────────────────────────────────────

class TestEventQueue:
    def test_put_and_get(self):
        q = EventQueue()
        e = Event(ADD_NODE, {"node_id": "a"})
        q.put(e)
        retrieved = q.get()
        assert retrieved is e

    def test_empty_when_nothing_enqueued(self):
        q = EventQueue()
        assert q.empty()

    def test_not_empty_after_put(self):
        q = EventQueue()
        q.put(Event(ADD_NODE, {}))
        assert not q.empty()

    def test_fifo_order(self):
        q = EventQueue()
        for i in range(5):
            q.put(Event(ADD_NODE, {"node_id": str(i)}))
        retrieved = [q.get().payload["node_id"] for _ in range(5)]
        assert retrieved == ["0", "1", "2", "3", "4"]

    def test_thread_safe_producer_consumer(self):
        q = EventQueue()
        received = []

        def producer():
            for i in range(50):
                q.put(Event(ADD_NODE, {"node_id": str(i)}))

        def consumer():
            for _ in range(50):
                received.append(q.get())

        t1 = threading.Thread(target=producer)
        t2 = threading.Thread(target=consumer)
        t1.start(); t2.start()
        t1.join(); t2.join()
        assert len(received) == 50


# ── rebuild_graph_from_log ────────────────────────────────────────────────────

class TestRebuildGraphFromLog:
    def _make_log(self, tmp_path, events):
        log = EventLog(str(tmp_path / "events.jsonl"))
        for e in events:
            log.append(e)
        return log

    def test_rebuilds_nodes(self, tmp_path):
        events = [
            Event(ADD_NODE, {"node_id": "a", "node_type": "task",
                              "dependencies": [], "metadata": {}}),
            Event(ADD_NODE, {"node_id": "b", "node_type": "task",
                              "dependencies": ["a"], "metadata": {}}),
        ]
        log = self._make_log(tmp_path, events)
        graph = rebuild_graph_from_log(log)
        assert "a" in graph.nodes
        assert "b" in graph.nodes

    def test_rebuilds_dependencies(self, tmp_path):
        events = [
            Event(ADD_NODE, {"node_id": "a", "node_type": "task",
                              "dependencies": [], "metadata": {}}),
            Event(ADD_NODE, {"node_id": "b", "node_type": "task",
                              "dependencies": [], "metadata": {}}),
            Event("ADD_DEPENDENCY", {"node_id": "b", "depends_on": "a"}),
        ]
        log = self._make_log(tmp_path, events)
        graph = rebuild_graph_from_log(log)
        assert "a" in graph.nodes["b"].dependencies

    def test_rebuilds_status(self, tmp_path):
        events = [
            Event(ADD_NODE, {"node_id": "a", "node_type": "task",
                              "dependencies": [], "metadata": {}}),
            Event(MARK_DONE, {"node_id": "a", "result": "done result"}),
        ]
        log = self._make_log(tmp_path, events)
        graph = rebuild_graph_from_log(log)
        assert graph.nodes["a"].status == "done"
        assert graph.nodes["a"].result == "done result"

    def test_duplicate_dependency_not_added_twice(self, tmp_path):
        events = [
            Event(ADD_NODE, {"node_id": "a", "node_type": "task",
                              "dependencies": [], "metadata": {}}),
            Event(ADD_NODE, {"node_id": "b", "node_type": "task",
                              "dependencies": ["a"], "metadata": {}}),
            # Replay the same dependency again
            Event("ADD_DEPENDENCY", {"node_id": "b", "depends_on": "a"}),
        ]
        log = self._make_log(tmp_path, events)
        graph = rebuild_graph_from_log(log)
        deps = list(graph.nodes["b"].dependencies)
        assert deps.count("a") == 1

    def test_empty_log_returns_empty_graph(self, tmp_path):
        log = EventLog(str(tmp_path / "events.jsonl"))
        graph = rebuild_graph_from_log(log)
        assert graph.nodes == {}

    def test_removed_node_not_in_graph(self, tmp_path):
        events = [
            Event(ADD_NODE, {"node_id": "a", "node_type": "task",
                              "dependencies": [], "metadata": {}}),
            Event("REMOVE_NODE", {"node_id": "a"}),
        ]
        log = self._make_log(tmp_path, events)
        graph = rebuild_graph_from_log(log)
        assert "a" not in graph.nodes
