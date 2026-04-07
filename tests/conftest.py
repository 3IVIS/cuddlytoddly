"""Shared pytest fixtures for the cuddlytoddly test suite."""
import json
import threading

import pytest

from cuddlytoddly.core.events import ADD_NODE, MARK_DONE, Event
from cuddlytoddly.core.reducer import apply_event
from cuddlytoddly.core.task_graph import TaskGraph
from cuddlytoddly.infra.event_queue import EventQueue

# ── Helpers ───────────────────────────────────────────────────────────────────

def add_node(graph, node_id, node_type="task", deps=None, metadata=None):
    """Convenience wrapper for adding a node via apply_event."""
    apply_event(graph, Event(ADD_NODE, {
        "node_id": node_id,
        "node_type": node_type,
        "dependencies": deps or [],
        "metadata": metadata or {"description": node_id},
    }))


def mark_done(graph, node_id, result="ok"):
    apply_event(graph, Event(MARK_DONE, {"node_id": node_id, "result": result}))


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def graph():
    return TaskGraph()


@pytest.fixture
def linear_graph():
    """goal → task_b → task_a  (task_a is the root)."""
    g = TaskGraph()
    add_node(g, "task_a")
    add_node(g, "task_b", deps=["task_a"])
    add_node(g, "goal", node_type="goal", deps=["task_b"],
             metadata={"description": "test goal", "expanded": True})
    return g


@pytest.fixture
def parallel_graph():
    """goal depends on task_a and task_b independently."""
    g = TaskGraph()
    add_node(g, "task_a")
    add_node(g, "task_b")
    add_node(g, "goal", node_type="goal", deps=["task_a", "task_b"],
             metadata={"description": "parallel goal", "expanded": True})
    return g


@pytest.fixture
def event_queue():
    return EventQueue()


class FakeLLM:
    """LLM stub that returns a fixed JSON string."""
    def __init__(self, response):
        self._response = response
        self.calls = []
        self._stop_event = threading.Event()

    @property
    def is_stopped(self):
        return self._stop_event.is_set()

    def stop(self):
        self._stop_event.set()

    def resume(self):
        self._stop_event.clear()

    def ask(self, prompt, schema=None):
        from cuddlytoddly.planning.llm_interface import LLMStoppedError
        if self._stop_event.is_set():
            raise LLMStoppedError("stopped")
        self.calls.append(prompt)
        if callable(self._response):
            return self._response(prompt, schema)
        return self._response


@pytest.fixture
def fake_llm():
    return FakeLLM(json.dumps({"satisfied": True, "reason": "looks good"}))


