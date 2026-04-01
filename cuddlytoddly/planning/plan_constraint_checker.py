# planning/plan_constraint_checker.py

import json

from cuddlytoddly.core.events import ADD_NODE, ADD_DEPENDENCY
from cuddlytoddly.planning.schemas import GHOST_NODE_RESOLUTION_SCHEMA
from cuddlytoddly.planning.prompts import build_ghost_node_resolution_prompt
from cuddlytoddly.infra.logging import get_logger

logger = get_logger(__name__)


class PlanConstraintChecker:
    """
    Post-validator constraint checker for planner output.

    Operates on the safe_events list produced by LLMOutputValidator and
    enforces plan-level invariants that are beyond the scope of structural
    validation.

    Checks performed (in order):
      7.  Duplicate ADD_DEPENDENCY edges  — silent deduplication
      4.  Cycle detection                 — drop cycle-member nodes + incident edges
      6b. Orphaned required_input         — strip items from nodes with no dependencies
      6a. Phantom dependencies            — warn only, no mutation
      Ghost / Goal connection             — nodes with no dependents resolved via LLM
    """

    def __init__(self, graph, llm_client):
        self.graph = graph
        self.llm   = llm_client

    def check_and_repair(self, safe_events: list, active_goal_id: str) -> list:
        """
        Run all constraint checks on safe_events and return the repaired list.
        Non-destructive: returns a new list; the input is never mutated.
        """
        events = list(safe_events)
        events = self._dedup_edges(events)
        events = self._remove_cycles(events)
        events = self._check_required_input(events)
        events = self._resolve_ghost_nodes(events, active_goal_id)
        return events

    # ── Internal helpers ──────────────────────────────────────────────────────

    @staticmethod
    def _parse_events(events):
        """
        Build working data structures from the event list.

        Returns
        -------
        new_nodes : dict[node_id → payload dict]
        edges     : set of (node_id, depends_on) tuples
                    Semantics: node_id depends ON depends_on
                    (depends_on must complete before node_id can start)
        """
        new_nodes: dict = {}
        edges: set      = set()
        for evt in events:
            t = evt.get("type")
            p = evt.get("payload", {})
            if t == ADD_NODE:
                nid = p.get("node_id")
                if nid:
                    new_nodes[nid] = p
                    for dep in p.get("dependencies", []):
                        edges.add((nid, dep))
            elif t == ADD_DEPENDENCY:
                nid = p.get("node_id")
                dep = p.get("depends_on")
                if nid and dep:
                    edges.add((nid, dep))
        return new_nodes, edges

    # ── Check 7: Deduplicate ADD_DEPENDENCY edges ─────────────────────────────

    def _dedup_edges(self, events: list) -> list:
        seen   = set()
        result = []
        for evt in events:
            if evt.get("type") == ADD_DEPENDENCY:
                key = (evt["payload"]["node_id"], evt["payload"]["depends_on"])
                if key in seen:
                    logger.debug(
                        "[CHECKER] Dropping duplicate edge %s → %s", key[0], key[1]
                    )
                    continue
                seen.add(key)
            result.append(evt)
        return result

    # ── Check 4: Cycle detection ──────────────────────────────────────────────

    def _remove_cycles(self, events: list) -> list:
        """
        Iteratively detect and remove cycles until the proposed subgraph is
        acyclic.  Only new nodes (those proposed in this batch) are ever
        dropped; existing graph nodes are treated as immutable terminals.
        """
        while True:
            new_nodes, edges = self._parse_events(events)

            # Adjacency for new nodes only (edges point toward prerequisites)
            adj: dict[str, set] = {nid: set() for nid in new_nodes}
            for (src, dep) in edges:
                if src in adj:
                    adj[src].add(dep)

            cycle_nodes = self._find_cycle_nodes(adj, set(new_nodes))
            if not cycle_nodes:
                break

            for nid in sorted(cycle_nodes):
                logger.warning("[CHECKER] Dropping cycle-member node: %s", nid)

            # Drop cycle nodes and every edge that touches them
            events = [
                evt for evt in events
                if not (
                    evt.get("type") == ADD_NODE
                    and evt["payload"]["node_id"] in cycle_nodes
                ) and not (
                    evt.get("type") == ADD_DEPENDENCY
                    and (
                        evt["payload"]["node_id"]    in cycle_nodes
                        or evt["payload"]["depends_on"] in cycle_nodes
                    )
                )
            ]

        return events

    @staticmethod
    def _find_cycle_nodes(adj: dict, new_ids: set) -> set:
        """
        DFS 3-colour cycle detection scoped to new nodes.

        Returns the minimal set of new nodes that form the first detected
        cycle, or an empty set if the graph is acyclic.

        Neighbours that are not in new_ids (i.e. existing graph nodes) are
        skipped — they are already validated and cannot form new cycles on
        their own.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color      = {nid: WHITE for nid in new_ids}
        stack_path: list = []   # current DFS path; shared via closure

        def dfs(node: str) -> set:
            color[node] = GRAY
            stack_path.append(node)
            for neighbour in adj.get(node, set()):
                if neighbour not in color:
                    continue  # existing graph node — skip
                if color[neighbour] == GRAY:
                    # Back-edge: collect cycle members from neighbour onward
                    idx = stack_path.index(neighbour)
                    return set(stack_path[idx:])
                if color[neighbour] == WHITE:
                    result = dfs(neighbour)
                    if result:
                        return result
            stack_path.pop()
            color[node] = BLACK
            return set()

        for nid in list(new_ids):
            if color.get(nid) == WHITE:
                result = dfs(nid)
                if result:
                    return result & new_ids   # intersect: only return NEW nodes
        return set()

    # ── Check 6: required_input consistency ───────────────────────────────────

    def _check_required_input(self, events: list) -> list:
        """
        6b — strip required_input from any task node that has no dependencies:
             those items are orphaned (no upstream producer can satisfy them).
        6a — warn when a task has dependencies but no required_input:
             the dependency may be a sequencing constraint with no data flow,
             which is allowed but worth flagging.
        """
        new_nodes, edges = self._parse_events(events)

        # Full incoming-dependency set per new node
        all_deps: dict[str, set] = {nid: set() for nid in new_nodes}
        for (src, dep) in edges:
            if src in all_deps:
                all_deps[src].add(dep)

        result = []
        for evt in events:
            if evt.get("type") == ADD_NODE:
                nid      = evt["payload"]["node_id"]
                metadata = evt["payload"].get("metadata", {})
                deps     = all_deps.get(nid, set())
                req_in   = metadata.get("required_input", [])
                ntype    = evt["payload"].get("node_type", "task")

                if req_in and not deps:
                    # 6b: orphaned required_input — strip in place
                    logger.warning(
                        "[CHECKER] Node %s declares required_input but has no "
                        "dependencies — stripping required_input",
                        nid,
                    )
                    evt = {
                        **evt,
                        "payload": {
                            **evt["payload"],
                            "metadata": {**metadata, "required_input": []},
                        },
                    }

                elif deps and not req_in and ntype == "task":
                    # 6a: phantom dependency — warn, do not mutate
                    logger.warning(
                        "[CHECKER] Node %s has %d dependenc%s but empty "
                        "required_input — dependency may not be data-flow justified",
                        nid,
                        len(deps),
                        "ies" if len(deps) != 1 else "y",
                    )

            result.append(evt)
        return result

    # ── Ghost node + goal connection ──────────────────────────────────────────

    def _resolve_ghost_nodes(self, events: list, active_goal_id: str) -> list:
        """
        Detect new nodes that have no dependents (nothing depends on them) and
        resolve each one with a targeted LLM call.

        The LLM is shown the full plan context and the set of valid candidate
        dependents (ancestors excluded to prevent introducing new cycles) and
        asked which node should depend on the ghost.

        The resulting ADD_DEPENDENCY events are appended to the event list.
        """
        new_nodes, edges = self._parse_events(events)
        existing_ids     = set(self.graph.nodes.keys())

        # Build dependents map for new nodes only
        dependents: dict[str, set] = {nid: set() for nid in new_nodes}
        for (src, dep) in edges:
            if dep in dependents:
                dependents[dep].add(src)

        ghost_ids = [nid for nid, deps in dependents.items() if not deps]
        if not ghost_ids:
            return events

        # Build description maps for prompt context
        new_summaries: dict[str, str] = {
            nid: p.get("metadata", {}).get("description", "")
            for nid, p in new_nodes.items()
        }
        existing_summaries: dict[str, str] = {
            nid: node.metadata.get("description", "")
            for nid, node in self.graph.nodes.items()
        }

        extra_edges = []
        for ghost_id in ghost_ids:
            logger.warning("[CHECKER] Ghost node detected: %s", ghost_id)

            # Ancestors of the ghost node must be excluded from candidates to
            # prevent introducing a cycle (if X is an ancestor of ghost, adding
            # ADD_DEPENDENCY(X, depends_on=ghost) would mean ghost→...→X→ghost).
            ancestors       = self._get_ancestors(ghost_id, edges, self.graph.nodes)
            valid_candidates = (
                (set(new_nodes.keys()) | existing_ids | {active_goal_id})
                - {ghost_id}
                - ancestors
            )

            prompt = build_ghost_node_resolution_prompt(
                ghost_node_id=ghost_id,
                ghost_description=new_summaries.get(ghost_id, ""),
                new_nodes=new_summaries,
                existing_nodes=existing_summaries,
                active_goal_id=active_goal_id,
                edges=edges,
                valid_candidates=valid_candidates,
            )

            try:
                raw          = self.llm.ask(prompt, schema=GHOST_NODE_RESOLUTION_SCHEMA)
                resp         = json.loads(raw)
                dependent_id = (resp.get("dependent_node_id") or "").strip()
                reasoning    = resp.get("reasoning", "")

                if dependent_id and dependent_id in valid_candidates:
                    logger.info(
                        "[CHECKER] Ghost %s → connected to dependent %s (%s)",
                        ghost_id, dependent_id, reasoning,
                    )
                    extra_edges.append({
                        "type": ADD_DEPENDENCY,
                        "payload": {
                            "node_id":    dependent_id,
                            "depends_on": ghost_id,
                            "origin":     "planning",
                        },
                    })
                else:
                    logger.warning(
                        "[CHECKER] LLM returned invalid dependent_id %r for ghost "
                        "node %s — no edge added",
                        dependent_id, ghost_id,
                    )

            except Exception as exc:
                logger.warning(
                    "[CHECKER] Ghost resolution LLM call failed for %s: %s",
                    ghost_id, exc,
                )

        return events + extra_edges

    @staticmethod
    def _get_ancestors(node_id: str, edges: set, graph_nodes: dict) -> set:
        """
        Return all nodes that node_id transitively depends on, spanning both
        the proposed edges and the live graph.

        Used to build the exclusion set for ghost node candidate selection so
        no cycle-creating dependent is offered to the LLM.
        """
        ancestors: set = set()
        queue          = [node_id]
        while queue:
            n = queue.pop()
            for (src, dep) in edges:
                if src == n and dep not in ancestors:
                    ancestors.add(dep)
                    queue.append(dep)
            if n in graph_nodes:
                for dep in graph_nodes[n].dependencies:
                    if dep not in ancestors:
                        ancestors.add(dep)
                        queue.append(dep)
        return ancestors