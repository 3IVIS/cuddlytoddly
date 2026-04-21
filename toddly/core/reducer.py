from copy import deepcopy

from toddly.core.events import (
    ADD_DEPENDENCY,
    ADD_NODE,
    CONFIRM_USER_DONE,
    DETACH_NODE,
    MARK_AWAITING_INPUT,
    MARK_AWAITING_USER,
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
from toddly.core.task_graph import TaskGraph

STRUCTURAL_EVENTS = {
    ADD_NODE,
    REMOVE_NODE,
    ADD_DEPENDENCY,
    REMOVE_DEPENDENCY,
    SET_NODE_TYPE,
    # FIX #3: DETACH_NODE mutates graph structure (removes a node's edges and
    # presence from the active node set) so it belongs in STRUCTURAL_EVENTS so
    # that structure_version is incremented and version-based invalidation in
    # the UI and any polling logic picks up the change.
    DETACH_NODE,
}

EXECUTION_EVENTS = {
    MARK_RUNNING,
    MARK_DONE,
    MARK_FAILED,
    RESET_NODE,
    UPDATE_METADATA,
    UPDATE_STATUS,
    SET_RESULT,
    RESET_SUBTREE,
    MARK_AWAITING_INPUT,
    MARK_AWAITING_USER,
    CONFIRM_USER_DONE,
    RESUME_NODE,
}


def apply_event(graph: TaskGraph, event: Event, event_log=None):
    # FIX: Write-ahead logging — persist the event to disk BEFORE mutating the
    # in-memory graph.  The previous order (mutate first, log second) meant that
    # a process crash between the two steps left the event missing from the log
    # while the mutation had already been applied.  On the next startup, replay
    # would produce a graph that is missing that event — silently incorrect state.
    #
    # With WAL, two safe outcomes are possible after a crash:
    #   • Crash before the append  → event never applied; replay is consistent.
    #   • Crash after the append   → event is in the log; replay re-applies it.
    #
    # apply_event is idempotent for ADD_NODE (already-present nodes are updated
    # rather than duplicated), and most execution events are naturally idempotent
    # (setting status to the same value again is a no-op).  The small risk of
    # double-application on replay is far less harmful than silent data loss.
    if event_log:
        event_log.append(event)

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

    elif t == MARK_AWAITING_USER:
        # A node that has produced what it can but has steps requiring user
        # action before downstream tasks can proceed.
        node = graph.nodes.get(p["node_id"])
        if node:
            node.status = "awaiting_user"
            node.metadata["handoff_artifact"] = p.get("handoff_artifact", "")
            node.metadata["pending_steps"] = p.get("pending_steps", [])

    elif t == CONFIRM_USER_DONE:
        # User has confirmed the real-world steps are complete; transition to done
        # so downstream nodes become ready.
        node = graph.nodes.get(p["node_id"])
        if node and node.status == "awaiting_user":
            node.status = "done"
            node.metadata.pop("handoff_artifact", None)
            node.metadata.pop("pending_steps", None)

    # FIX #1: RESET_SUBTREE was imported, listed in EXECUTION_EVENTS, but had
    # no handler branch — any emitted event silently did nothing to the graph
    # while still incrementing the version counter.  The handler now walks the
    # subtree rooted at node_id and resets every non-running descendant.
    elif t == RESET_SUBTREE:
        root_id = p.get("node_id")
        if root_id and root_id in graph.nodes:
            queue = list(graph.nodes[root_id].children)
            visited: set = set()
            while queue:
                child_id = queue.pop()
                if child_id in visited or child_id not in graph.nodes:
                    continue
                visited.add(child_id)
                child = graph.nodes[child_id]
                if child.status != "running":
                    child.reset()
                    queue.extend(child.children)
            # Also reset the root node itself if it is not running.
            root = graph.nodes[root_id]
            if root.status != "running":
                root.reset()

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

    # Use the incremental readiness update when only the children of the
    # completed node can become newly ready (MARK_DONE).  For all other
    # events fall back to the full graph scan which is always correct.
    if t == MARK_DONE:
        graph.recompute_readiness_for(p.get("node_id", ""))
    else:
        graph.recompute_readiness()
