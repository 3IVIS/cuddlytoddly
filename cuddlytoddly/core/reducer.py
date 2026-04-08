from copy import deepcopy

from cuddlytoddly.core.events import (
    ADD_DEPENDENCY,
    ADD_NODE,
    DETACH_NODE,
    MARK_AWAITING_INPUT,
    MARK_DONE,
    MARK_FAILED,
    MARK_RUNNING,
    REMOVE_DEPENDENCY,
    REMOVE_NODE,
    RESET_NODE,
    RESET_SUBTREE,
    RESUME_NODE,
    SET_NODE_TYPE,
    SET_RESULT,
    UPDATE_METADATA,
    UPDATE_STATUS,
    Event,
)
from cuddlytoddly.core.task_graph import TaskGraph

STRUCTURAL_EVENTS = {
    ADD_NODE,
    REMOVE_NODE,
    ADD_DEPENDENCY,
    REMOVE_DEPENDENCY,
    SET_NODE_TYPE,
}

EXECUTION_EVENTS = {
    MARK_RUNNING,
    MARK_DONE,
    MARK_FAILED,
    RESET_NODE,
    UPDATE_METADATA,
    SET_RESULT,
    RESET_SUBTREE,
    MARK_AWAITING_INPUT,
    RESUME_NODE,
}


def apply_event(graph: TaskGraph, event: Event, event_log=None):
    t = event.type
    p = event.payload or {}

    if t == "INSERT_NODE":
        t = ADD_NODE

    # ---------------- NODE EVENTS ----------------
    if t == ADD_NODE:
        node_id = p["node_id"]
        node_type = p.get("node_type", "task")
        dependencies = p.get("dependencies", [])
        metadata = deepcopy(p.get("metadata", {}))
        origin = p.get("origin", "user")

        if node_id not in graph.nodes:
            graph.add_node(
                node_id=node_id,
                node_type=node_type,
                dependencies=dependencies,
                origin=origin,
                metadata=metadata,
            )
        else:
            node = graph.nodes[node_id]
            existing_desc = node.metadata.get("description", "")
            node.metadata.update(metadata)
            # Restore the original description if it was already populated —
            # user-provided and previously-set descriptions must not be clobbered
            if existing_desc:
                node.metadata["description"] = existing_desc
            node.node_type = node_type or node.node_type

    elif t == REMOVE_NODE:
        graph.remove_node(p["node_id"])

    # ---------------- DEPENDENCY EVENTS ----------------
    elif t == ADD_DEPENDENCY:
        graph.add_dependency(p["node_id"], p["depends_on"])

    elif t == REMOVE_DEPENDENCY:
        graph.remove_dependency(p["node_id"], p["depends_on"])

    # ---------------- EXECUTION EVENTS ----------------
    elif t == MARK_RUNNING:
        node = graph.nodes.get(p["node_id"])
        if node:
            node.status = "running"

    elif t == MARK_DONE:
        node = graph.nodes.get(p["node_id"])
        if node:
            node.status = "done"
            node.result = p.get("result")

    elif t == MARK_FAILED:
        node = graph.nodes.get(p["node_id"])
        if node:
            node.status = "failed"

    elif t == RESET_NODE:
        node = graph.nodes.get(p["node_id"])
        if node:
            node.reset()

    elif t == MARK_AWAITING_INPUT:
        node = graph.nodes.get(p["node_id"])
        if node:
            node.status = "awaiting_input"
            node.metadata["missing_fields"] = p.get("missing_fields", [])
            node.metadata["awaiting_input_reason"] = p.get("awaiting_input_reason", "")

    elif t == RESUME_NODE:
        # Transition awaiting_input → pending; recompute_readiness promotes to ready
        node = graph.nodes.get(p["node_id"])
        if node:
            node.status = "pending"
            node.result = None
            node.metadata.pop("missing_fields", None)
            node.metadata.pop("awaiting_input_reason", None)
            node.metadata.pop("retry_count", None)
            node.metadata.pop("retry_after", None)
            node.metadata.pop("verification_failure", None)

    elif t == DETACH_NODE:
        graph.detach_node(p["node_id"])

    elif t == UPDATE_STATUS:
        graph.update_status(p["node_id"], p["status"])

    elif t == SET_RESULT:
        node = graph.nodes.get(p["node_id"])
        if node:
            node.result = p.get("result")

    elif t == SET_NODE_TYPE:
        node = graph.nodes.get(p["node_id"])
        if node:
            node.node_type = p["node_type"]

    # ---------------- METADATA EVENTS ----------------

    elif t == UPDATE_METADATA:
        node = graph.nodes.get(p["node_id"])
        if node:
            existing_desc = node.metadata.get("description", "").strip()
            node.metadata.update(p.get("metadata", {}))
            if existing_desc and p.get("origin") != "user":
                node.metadata["description"] = existing_desc
            # ← remove the "if node_type in p" block entirely

    # ---------------- VERSION TRACKING ----------------
    if t in STRUCTURAL_EVENTS:
        graph.structure_version += 1
    elif t in EXECUTION_EVENTS:
        graph.execution_version += 1

    graph.recompute_readiness()

    if event_log:
        event_log.append(event)
