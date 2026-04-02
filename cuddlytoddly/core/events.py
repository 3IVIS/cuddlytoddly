"""
Event Definitions

All mutations must go through reducer.
"""

# core/events.py

from datetime import datetime, timezone


class Event:
    def __init__(self, type, payload, timestamp=None):
        self.type = type
        self.payload = payload
        self.timestamp = timestamp or datetime.now(timezone.utc).isoformat()

    def to_dict(self):
        return {
            "type": self.type,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            type=data["type"],
            payload=data["payload"],
            timestamp=data.get("timestamp"),
        )

# Event types
ADD_NODE = "ADD_NODE"
REMOVE_NODE = "REMOVE_NODE"
ADD_DEPENDENCY = "ADD_DEPENDENCY"
REMOVE_DEPENDENCY = "REMOVE_DEPENDENCY"
MARK_RUNNING = "MARK_RUNNING"
MARK_DONE = "MARK_DONE"
MARK_FAILED = "MARK_FAILED"
RESET_NODE = "RESET_NODE"
UPDATE_METADATA = "UPDATE_METADATA"
DETACH_NODE = "DETACH_NODE"
UPDATE_STATUS = "UPDATE_STATUS"
SET_RESULT = "SET_RESULT"
SET_NODE_TYPE = "SET_NODE_TYPE"
RESET_SUBTREE = "RESET_SUBTREE"