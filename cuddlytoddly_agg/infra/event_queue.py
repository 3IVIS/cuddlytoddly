"""
Event Queue

Thread-safe queue for all mutations and interactions.
"""

from dataclasses import dataclass, field
from queue import Queue


class EventQueue:
    def __init__(self):
        self._queue = Queue()

    def put(self, event):
        self._queue.put(event)

    def get(self):
        return self._queue.get()

    def empty(self):
        return self._queue.empty()


@dataclass
class StatusEvent:
    """
    Lightweight status notification (not a graph-mutation Event).

    Used to surface out-of-band signals — such as LLM load failures —
    from background threads to the UI or orchestrator loop without going
    through the full event/reducer pipeline.

    Attributes
    ----------
    kind    : Short identifier, e.g. ``"llm_load_failed"``.
    payload : Arbitrary key/value context, e.g. ``{"error": "..."}``.
    """

    kind: str
    payload: dict = field(default_factory=dict)
