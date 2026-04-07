"""
Event Queue

Thread-safe queue for all mutations and interactions.
"""
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


