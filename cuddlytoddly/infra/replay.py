# infra/replay.py

from cuddlytoddly.core.reducer import apply_event
from cuddlytoddly.core.task_graph import TaskGraph


def rebuild_graph_from_log(event_log):
    graph = TaskGraph()
    for event in event_log.replay():
        # Skip ADD_DEPENDENCY events that would create a cycle or already exist
        if event.type == "ADD_DEPENDENCY":
            node_id    = event.payload.get("node_id")
            depends_on = event.payload.get("depends_on")
            if (node_id in graph.nodes
                    and depends_on in graph.nodes
                    and depends_on in graph.nodes[node_id].dependencies):
                continue   # already wired — skip silently
        apply_event(graph, event)
    return graph


