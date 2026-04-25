# planning/plan_constraint_checker.py

import json

from toddly.core.events import ADD_DEPENDENCY, ADD_NODE
from toddly.infra.logging import get_logger
from toddly.planning.prompts import build_ghost_node_resolution_prompt
from toddly.planning.schemas import GHOST_NODE_RESOLUTION_SCHEMA

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
        self.llm = llm_client

    def check_and_repair(
        self,
        safe_events: list,
        active_goal_id: str,
        known_dep_id: str | None = None,
        snapshot: dict | None = None,
    ) -> list:
        """
        Run all constraint checks on safe_events and return the repaired list.
        Non-destructive: returns a new list; the input is never mutated.

        Parameters
        ----------
        safe_events     : Validated event list from LLMOutputValidator.
        active_goal_id  : ID of the goal being planned.
        known_dep_id    : Optional node ID that will be wired as a dependency of
                          all root tasks AFTER the constraint checker runs (e.g.
                          the clarification node).  Telling the checker about it
                          prevents false-positive 6b strips (root tasks declared
                          with required_input look orphaned without this) and
                          false-positive ghost detection (root tasks with no
                          batch dependents look like ghosts without this).
        snapshot        : FIX 2: Immutable graph snapshot captured under
                          graph_lock in _planning_pass.  Passed to
                          _resolve_ghost_nodes so it reads existing node IDs and
                          descriptions from the snapshot rather than from the
                          live self.graph, eliminating the live-graph race.
        """
        events = list(safe_events)
        events = self._inject_dataflow_dependencies(events)
        events = self._dedup_edges(events)
        events = self._remove_cycles(events)
        events = self._check_required_input(events, known_dep_id=known_dep_id)
        events = self._resolve_ghost_nodes(
            events, active_goal_id, known_dep_id=known_dep_id, snapshot=snapshot
        )
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
        edges: set = set()
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

    # ── Data-flow dependency injection ───────────────────────────────────────

    @staticmethod
    def _inject_dataflow_dependencies(events: list) -> list:
        """
        Mechanically inject missing ADD_DEPENDENCY edges wherever the planner
        declared data-flow intent through output/required_input names but omitted
        the corresponding dependency edge.

        Algorithm
        ---------
        For every pair of new nodes (A, B):
          if any name in A's ``output`` list matches any name in B's
          ``required_input`` list, and B does not already declare A as a
          dependency, emit ADD_DEPENDENCY(node_id=B, depends_on=A).

        This is deterministic and requires no LLM call — the planner already
        expresses the data contract through those name fields; this pass simply
        enforces it structurally.

        Example
        -------
        A outputs ``python_code_review_tools``.
        B requires_input ``python_code_review_tools``.
        The planner placed them in the same parallel_group and emitted no edge.
        This pass adds ADD_DEPENDENCY(B, depends_on=A) so B waits for A's result
        before executing, giving it the tool names to anchor its searches on.
        """
        new_nodes, existing_edges = PlanConstraintChecker._parse_events(events)

        # Build output-name → producer node_id map.
        # If two nodes declare the same output name the last one wins; in
        # practice the validator rejects duplicate output names, so this is
        # only a safety net.
        output_to_producer: dict[str, str] = {}
        for nid, payload in new_nodes.items():
            for out in payload.get("metadata", {}).get("output", []):
                name = out.get("name") if isinstance(out, dict) else str(out)
                if name:
                    output_to_producer[name] = nid

        if not output_to_producer:
            return events

        injected: list[dict] = []
        for nid, payload in new_nodes.items():
            required_inputs = payload.get("metadata", {}).get("required_input", [])
            for req in required_inputs:
                req_name = req.get("name") if isinstance(req, dict) else str(req)
                if not req_name:
                    continue
                producer_id = output_to_producer.get(req_name)
                if not producer_id or producer_id == nid:
                    continue
                # Skip if edge already exists (declared in the ADD_NODE payload
                # or as a standalone ADD_DEPENDENCY event).
                if (nid, producer_id) in existing_edges:
                    continue
                logger.info(
                    "[CHECKER] Injecting missing data-flow edge: %s → %s "
                    "(output '%s' satisfies required_input '%s')",
                    producer_id,
                    nid,
                    req_name,
                    req_name,
                )
                injected.append(
                    {
                        "type": ADD_DEPENDENCY,
                        "payload": {
                            "node_id": nid,
                            "depends_on": producer_id,
                            "origin": "planning",
                        },
                    }
                )
                # Record so we don't emit duplicate edges if multiple
                # required_input items map to the same producer.
                existing_edges.add((nid, producer_id))

        if injected:
            logger.info("[CHECKER] Injected %d data-flow dependency edge(s)", len(injected))

        return events + injected

    # ── Check 7: Deduplicate ADD_DEPENDENCY edges ─────────────────────────────

    def _dedup_edges(self, events: list) -> list:
        seen = set()
        result = []
        for evt in events:
            if evt.get("type") == ADD_DEPENDENCY:
                key = (evt["payload"]["node_id"], evt["payload"]["depends_on"])
                if key in seen:
                    logger.debug("[CHECKER] Dropping duplicate edge %s → %s", key[0], key[1])
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
            for src, dep in edges:
                if src in adj:
                    adj[src].add(dep)

            cycle_nodes = self._find_cycle_nodes(adj, set(new_nodes))
            if not cycle_nodes:
                break

            for nid in sorted(cycle_nodes):
                logger.warning("[CHECKER] Dropping cycle-member node: %s", nid)

            # Drop cycle nodes and every edge that touches them
            events = [
                evt
                for evt in events
                if not (evt.get("type") == ADD_NODE and evt["payload"]["node_id"] in cycle_nodes)
                and not (
                    evt.get("type") == ADD_DEPENDENCY
                    and (
                        evt["payload"]["node_id"] in cycle_nodes
                        or evt["payload"]["depends_on"] in cycle_nodes
                    )
                )
            ]

        return events

    @staticmethod
    def _find_cycle_nodes(adj: dict, new_ids: set) -> set:
        """
        Iterative DFS 3-colour cycle detection scoped to new nodes.

        Returns the minimal set of new nodes that form the first detected
        cycle, or an empty set if the graph is acyclic.

        Neighbours that are not in new_ids (i.e. existing graph nodes) are
        skipped — they are already validated and cannot form new cycles on
        their own.

        The implementation is fully iterative to avoid hitting Python's default
        recursion limit on large plans.  Each stack frame is a tuple of
        (node, iterator-over-neighbours, path-snapshot) so we can reconstruct
        the DFS path when a back-edge is detected without relying on the call
        stack.
        """
        WHITE, GRAY, BLACK = 0, 1, 2
        color = {nid: WHITE for nid in new_ids}

        for start in list(new_ids):
            if color.get(start) != WHITE:
                continue

            # Each entry: (node, neighbour_iter, path_up_to_and_including_node)
            path: list[str] = []
            stack: list[tuple] = [(start, iter(adj.get(start, set())), None)]
            color[start] = GRAY
            path.append(start)

            while stack:
                node, neighbours, _ = stack[-1]
                try:
                    neighbour = next(neighbours)
                except StopIteration:
                    # All neighbours explored — colour black and backtrack
                    color[node] = BLACK
                    stack.pop()
                    if path and path[-1] == node:
                        path.pop()
                    continue

                if neighbour not in color:
                    continue  # existing graph node — skip

                if color[neighbour] == GRAY:
                    # Back-edge found — extract the cycle from the current path
                    idx = path.index(neighbour)
                    cycle = set(path[idx:])
                    return cycle & new_ids  # only return NEW nodes

                if color[neighbour] == WHITE:
                    color[neighbour] = GRAY
                    path.append(neighbour)
                    stack.append((neighbour, iter(adj.get(neighbour, set())), None))

        return set()

    # ── Check 6: required_input consistency ───────────────────────────────────

    def _check_required_input(self, events: list, known_dep_id: str | None = None) -> list:
        """
        6b — strip required_input from any task node that has no dependencies:
             those items are orphaned (no upstream producer can satisfy them).
             Exception: if known_dep_id is set, root tasks (no batch deps) are
             expected to depend on that external node, so their required_input
             is legitimate — warn but do not strip.
        6a — warn when a task has dependencies but no required_input:
             the dependency may be a sequencing constraint with no data flow,
             which is allowed but worth flagging.
        """
        new_nodes, edges = self._parse_events(events)

        # Full incoming-dependency set per new node
        all_deps: dict[str, set] = {nid: set() for nid in new_nodes}
        for src, dep in edges:
            if src in all_deps:
                all_deps[src].add(dep)

        # Pre-compute new_node_ids for root-task detection
        new_node_ids = set(new_nodes.keys())

        result = []
        for evt in events:
            if evt.get("type") == ADD_NODE:
                nid = evt["payload"]["node_id"]
                metadata = evt["payload"].get("metadata", {})
                deps = all_deps.get(nid, set())
                req_in = metadata.get("required_input", [])
                ntype = evt["payload"].get("node_type", "task")

                # A root task has no dependencies on other new nodes.
                # If known_dep_id is provided it will be wired as a dependency
                # of all root tasks after the checker runs, so required_input
                # on a root task is NOT orphaned — skip the 6b strip.
                is_root_task = not (deps & new_node_ids)

                if req_in and not deps:
                    if known_dep_id and is_root_task:
                        # Expected — root task will depend on known_dep_id
                        logger.debug(
                            "[CHECKER] Node %s has required_input and no batch deps "
                            "— will depend on %s, skipping 6b strip",
                            nid,
                            known_dep_id,
                        )
                    else:
                        # 6b: genuinely orphaned required_input — strip it
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

    def _resolve_ghost_nodes(
        self,
        events: list,
        active_goal_id: str,
        known_dep_id: str | None = None,
        snapshot: dict | None = None,
    ) -> list:
        """
        Detect new nodes that have no dependents (nothing depends on them) and
        resolve each one with a targeted LLM call.

        Root tasks — nodes with no dependencies on other new nodes — will have
        known_dep_id wired as their dependency after the checker runs.  They
        are legitimately expected to produce output for the plan and should not
        be treated as ghost nodes.  They are excluded from ghost detection when
        known_dep_id is provided.

        FIX 2: snapshot (an immutable dict captured under graph_lock in
        _planning_pass) is now used for all read-only lookups of existing node
        IDs and descriptions instead of self.graph.  The checker runs with
        graph_lock released; concurrent _on_node_done callbacks in the thread
        pool may be mutating self.graph.nodes at the same time.  The snapshot
        is safe to read without any additional locking.  Falls back to
        self.graph when no snapshot is provided (tests, legacy callers).
        """
        new_nodes, edges = self._parse_events(events)
        # FIX 2: use snapshot for existing-node lookups when available.
        graph_view: dict = snapshot if snapshot is not None else self.graph.nodes
        existing_ids = set(graph_view.keys())
        new_node_ids = set(new_nodes.keys())

        # Build dependents map for new nodes only
        dependents: dict[str, set] = {nid: set() for nid in new_nodes}
        for src, dep in edges:
            if dep in dependents:
                dependents[dep].add(src)

        # Root tasks: no deps on other new nodes.  They will depend on
        # known_dep_id once wiring runs, so they are not ghosts.
        root_task_ids: set = set()
        if known_dep_id:
            for nid, p in new_nodes.items():
                batch_deps = {d for d in p.get("dependencies", []) if d in new_node_ids}
                if not batch_deps:
                    root_task_ids.add(nid)

        ghost_ids = [
            nid for nid, deps in dependents.items() if not deps and nid not in root_task_ids
        ]
        if not ghost_ids:
            return events

        # Build description maps for prompt context
        new_summaries: dict[str, str] = {
            nid: p.get("metadata", {}).get("description", "") for nid, p in new_nodes.items()
        }
        # FIX 2: read descriptions from snapshot, not live graph.
        existing_summaries: dict[str, str] = {
            nid: node.metadata.get("description", "") for nid, node in graph_view.items()
        }

        extra_edges = []
        for ghost_id in ghost_ids:
            logger.warning("[CHECKER] Ghost node detected: %s", ghost_id)

            # Ancestors of the ghost node must be excluded from candidates to
            # prevent introducing a cycle (if X is an ancestor of ghost, adding
            # ADD_DEPENDENCY(X, depends_on=ghost) would mean ghost→...→X→ghost).
            # FIX 2: pass graph_view (snapshot) to _get_ancestors so it uses the
            # same safe data source as the rest of this method.
            ancestors = self._get_ancestors(ghost_id, edges, graph_view)
            valid_candidates = (
                (set(new_nodes.keys()) | existing_ids | {active_goal_id}) - {ghost_id} - ancestors
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
                raw = self.llm.ask(prompt, schema=GHOST_NODE_RESOLUTION_SCHEMA)
                resp = json.loads(raw)
                dependent_id = (resp.get("dependent_node_id") or "").strip()
                reasoning = resp.get("reasoning", "")

                if dependent_id and dependent_id in valid_candidates:
                    logger.info(
                        "[CHECKER] Ghost %s → connected to dependent %s (%s)",
                        ghost_id,
                        dependent_id,
                        reasoning,
                    )
                    extra_edges.append(
                        {
                            "type": ADD_DEPENDENCY,
                            "payload": {
                                "node_id": dependent_id,
                                "depends_on": ghost_id,
                                "origin": "planning",
                            },
                        }
                    )
                else:
                    logger.warning(
                        "[CHECKER] LLM returned invalid dependent_id %r for ghost "
                        "node %s — no edge added",
                        dependent_id,
                        ghost_id,
                    )

            except Exception as exc:
                logger.warning(
                    "[CHECKER] Ghost resolution LLM call failed for %s: %s",
                    ghost_id,
                    exc,
                )

        return events + extra_edges

    # FIX: removed duplicate @staticmethod decorator.  The original code had
    # two stacked @staticmethod decorators on this method.  On Python < 3.10
    # that raises TypeError at runtime (staticmethod objects are not callable
    # before 3.10).  On Python ≥ 3.10 it silently works but is still wrong.
    @staticmethod
    def _get_ancestors(node_id: str, edges: set, graph_nodes: dict) -> set:
        """
        Return all nodes that node_id transitively depends on, spanning both
        the proposed edges and the live graph.

        Used to build the exclusion set for ghost node candidate selection so
        no cycle-creating dependent is offered to the LLM.

        Fix #15: the original implementation scanned all edges on every queue
        pop — O(n×m) overall.  We now build an adjacency dict once in O(m) so
        each per-node lookup is O(out-degree) instead of O(m).
        """
        # Build adjacency dict from proposed edges once — O(m).
        proposed_adj: dict[str, list] = {}
        for src, dep in edges:
            proposed_adj.setdefault(src, []).append(dep)

        ancestors: set = set()
        queue = [node_id]
        while queue:
            n = queue.pop()
            for dep in proposed_adj.get(n, []):
                if dep not in ancestors:
                    ancestors.add(dep)
                    queue.append(dep)
            if n in graph_nodes:
                for dep in graph_nodes[n].dependencies:
                    if dep not in ancestors:
                        ancestors.add(dep)
                        queue.append(dep)
        return ancestors
