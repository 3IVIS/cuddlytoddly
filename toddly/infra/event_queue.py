"""
Event Queue

Thread-safe queue for all mutations and interactions.
"""

import time as _time
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
    kind           : Short identifier, e.g. ``"llm_load_failed"``.
    payload        : Arbitrary key/value context, e.g. ``{"error": "..."}``.
    created_at_ms  : Wall-clock milliseconds at creation time (epoch ms).
                     Stamped automatically so that batches of events drained
                     in the same poll tick retain their original timestamps
                     rather than all sharing the drain time.
    """

    kind: str
    payload: dict = field(default_factory=dict)
    created_at_ms: int = field(default_factory=lambda: int(_time.time() * 1000))
