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

# A task that cannot proceed until the user provides specific information.
# Payload: { node_id, missing_fields: list[str], awaiting_input_reason: str }
# missing_fields holds the clarification field *keys* the task is waiting on.
MARK_AWAITING_INPUT = "MARK_AWAITING_INPUT"

# Transitions a node from awaiting_input back to pending so the orchestrator
# can recompute readiness and re-launch it.
# Payload: { node_id }
RESUME_NODE = "RESUME_NODE"

# A task that has one or more execution steps that cannot be performed by the
# LLM and require the user to act in the real world before downstream tasks
# can proceed.
# Payload: { node_id, handoff_artifact: str, pending_steps: list[str] }
# handoff_artifact: human-readable instructions / drafted content for the user
# pending_steps: execution_type values of the steps awaiting the user
MARK_AWAITING_USER = "MARK_AWAITING_USER"

# Transitions a node from awaiting_user back to done once the user confirms
# the real-world steps are complete.
# Payload: { node_id }
CONFIRM_USER_DONE = "CONFIRM_USER_DONE"
