from cuddlytoddly.core.events import ADD_DEPENDENCY, ADD_NODE
from cuddlytoddly.infra.logging import get_logger

logger = get_logger(__name__)


class LLMOutputValidator:
    """
    Validates and normalizes raw LLM output before it enters the DAG.

    - Supports ADD_NODE and ADD_DEPENDENCY
    - Accepts dependency chains transitively within the same LLM batch
    - Prevents duplicate nodes
    - Prevents self-dependencies
    - Prevents goals depending on other goals
    - Ensures all dependencies reference valid nodes
    - Validates metadata keys for parallelism, I/O, and reflection
    """

    ALLOWED_METADATA_KEYS = {
        "precedes",
        "parallel_group",
        "required_input",
        "output",
        "description",
        "reflection_notes",
        "skill",
        "tools",
        # clarification node fields
        "fields",
        "clarification_prompt",
    }

    def __init__(self, graph):
        self.graph = graph

    def validate_and_normalize(self, raw_events, forced_origin):
        if not isinstance(raw_events, list):
            logger.warning(
                "[VALIDATOR] Expected a list of events, got %s — rejecting entire output",
                type(raw_events).__name__,
            )
            return []

        existing_ids = set(self.graph.nodes.keys())
        goal_ids = {
            nid
            for nid, node in self.graph.nodes.items()
            if getattr(node, "node_type", None) == "goal"
        }
        proposed_nodes = {}
        proposed_edges = []

        # --------------------------------
        # 1️⃣ Basic schema validation
        # --------------------------------
        for event in raw_events:
            if not isinstance(event, dict):
                logger.warning("[VALIDATOR] Skipping non-dict event: %r", event)
                continue

            event_type = event.get("type")
            payload = event.get("payload")
            if not isinstance(payload, dict):
                logger.warning(
                    "[VALIDATOR] Skipping event with invalid payload (type=%s): %r",
                    event_type,
                    payload,
                )
                continue

            # -----------------------------
            # ADD_NODE
            # -----------------------------
            if event_type == ADD_NODE:
                node_id = payload.get("node_id")
                dependencies = payload.get("dependencies", [])
                node_type = payload.get("node_type", "task")
                metadata = payload.get("metadata", {})

                if not node_id or not isinstance(node_id, str):
                    logger.warning(
                        "[VALIDATOR] ADD_NODE rejected — missing or non-string node_id: %r",
                        payload,
                    )
                    continue
                if not isinstance(dependencies, list):
                    logger.warning(
                        "[VALIDATOR] ADD_NODE %s rejected — dependencies is not a list: %r",
                        node_id,
                        dependencies,
                    )
                    continue
                if node_id in dependencies:
                    logger.warning(
                        "[VALIDATOR] ADD_NODE %s rejected — self-dependency", node_id
                    )
                    continue
                if node_id in existing_ids:
                    logger.warning(
                        "[VALIDATOR] ADD_NODE %s rejected — node already exists in graph",
                        node_id,
                    )
                    # Salvage any dependency edges implied by this node's dependencies list.
                    # The node itself doesn't need re-creating, but the edges it declared
                    # may not exist yet (e.g. a goal<->task link the planner is re-asserting).
                    for dep in dependencies:
                        if isinstance(dep, str) and dep != node_id:
                            proposed_edges.append((node_id, dep))
                    continue
                if not isinstance(metadata, dict):
                    logger.warning(
                        "[VALIDATOR] ADD_NODE %s — metadata is not a dict, resetting to {}",
                        node_id,
                    )
                    metadata = {}

                filtered_metadata = {
                    k: v for k, v in metadata.items() if k in self.ALLOWED_METADATA_KEYS
                }
                stripped_keys = set(metadata.keys()) - self.ALLOWED_METADATA_KEYS
                if stripped_keys:
                    logger.debug(
                        "[VALIDATOR] ADD_NODE %s — stripped disallowed metadata keys: %s",
                        node_id,
                        stripped_keys,
                    )

                proposed_nodes[node_id] = {
                    "node_id": node_id,
                    "node_type": node_type,
                    "dependencies": dependencies,
                    "metadata": filtered_metadata,
                }

            # -----------------------------
            # ADD_DEPENDENCY
            # -----------------------------
            elif event_type == ADD_DEPENDENCY:
                node_id = payload.get("node_id")
                depends_on = payload.get("depends_on")
                if not node_id or not depends_on:
                    logger.warning(
                        "[VALIDATOR] ADD_DEPENDENCY rejected — missing node_id or depends_on: %r",
                        payload,
                    )
                    continue
                if not isinstance(node_id, str) or not isinstance(depends_on, str):
                    logger.warning(
                        "[VALIDATOR] ADD_DEPENDENCY rejected — node_id/depends_on must be strings: %r",
                        payload,
                    )
                    continue
                if node_id == depends_on:
                    logger.warning(
                        "[VALIDATOR] ADD_DEPENDENCY rejected — self-dependency on %s",
                        node_id,
                    )
                    continue

                # Block a non-goal node from depending on a goal node.
                # However, a goal depending on its final completing task is valid
                # and must be allowed through.
                dependent_is_goal = node_id in goal_ids or node_id in {
                    nid
                    for nid, nd in proposed_nodes.items()
                    if nd.get("node_type") == "goal"
                }
                if depends_on in goal_ids and not dependent_is_goal:
                    logger.warning(
                        "[VALIDATOR] ADD_DEPENDENCY rejected — non-goal %s cannot depend on goal node %s",
                        node_id,
                        depends_on,
                    )
                    continue

                proposed_edges.append((node_id, depends_on))

            else:
                logger.warning(
                    "[VALIDATOR] Unknown event type %r — skipping", event_type
                )

        # --------------------------------
        # 2️⃣ Transitive structural validation for nodes
        # --------------------------------
        accepted_nodes = {}
        available_ids = set(existing_ids)
        progress = True

        while progress:
            progress = False
            for node_id, node_data in list(proposed_nodes.items()):
                deps = node_data["dependencies"]
                if all(dep in available_ids for dep in deps):
                    accepted_nodes[node_id] = node_data
                    available_ids.add(node_id)
                    proposed_nodes.pop(node_id)
                    progress = True

        # Remaining proposed_nodes have unresolvable dependencies
        for node_id, node_data in proposed_nodes.items():
            missing = [d for d in node_data["dependencies"] if d not in available_ids]
            logger.warning(
                "[VALIDATOR] ADD_NODE %s rejected — unresolvable dependencies: %s",
                node_id,
                missing,
            )

        # --------------------------------
        # 3️⃣ Validate ADD_DEPENDENCY events
        # --------------------------------
        safe_edges = []
        for node_id, depends_on in proposed_edges:
            if node_id not in available_ids:
                logger.warning(
                    "[VALIDATOR] ADD_DEPENDENCY (%s → %s) rejected — %s not in graph or accepted batch",
                    node_id,
                    depends_on,
                    node_id,
                )
                continue
            if depends_on not in available_ids:
                logger.warning(
                    "[VALIDATOR] ADD_DEPENDENCY (%s → %s) rejected — %s not in graph or accepted batch",
                    node_id,
                    depends_on,
                    depends_on,
                )
                continue
            safe_edges.append((node_id, depends_on))

        # --------------------------------
        # 4️⃣ Normalize accepted events
        # --------------------------------
        safe_events = []

        # ADD_NODE events
        for node_id, node_data in accepted_nodes.items():
            safe_events.append(
                {
                    "type": ADD_NODE,
                    "payload": {
                        "node_id": node_id,
                        "node_type": node_data["node_type"],
                        "dependencies": node_data["dependencies"],
                        "origin": forced_origin,
                        "metadata": node_data["metadata"],
                    },
                }
            )

        # ADD_DEPENDENCY events
        for node_id, depends_on in safe_edges:
            safe_events.append(
                {
                    "type": ADD_DEPENDENCY,
                    "payload": {
                        "node_id": node_id,
                        "depends_on": depends_on,
                        "origin": forced_origin,
                    },
                }
            )

        logger.info(
            "[VALIDATOR] Result: %d raw events → %d accepted (%d nodes, %d edges)",
            len(raw_events),
            len(safe_events),
            len(accepted_nodes),
            len(safe_edges),
        )

        return safe_events
