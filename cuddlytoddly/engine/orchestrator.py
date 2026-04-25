from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cuddlytoddly.engine.signals import AwaitingInputSignal

from toddly.core.events import (
    ADD_NODE,
    MARK_DONE,
    REMOVE_NODE,
    RESET_NODE,
    RESUME_NODE,
    SET_NODE_TYPE,
    SET_RESULT,
    UPDATE_METADATA,
    Event,
)
from toddly.engine.base_orchestrator import BaseOrchestrator
from toddly.infra.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# cuddlytoddly domain orchestrator
# =============================================================================


class Orchestrator(BaseOrchestrator):
    """
    cuddlytoddly-specific orchestrator.

    Extends BaseOrchestrator with the domain concepts unique to cuddlytoddly:
      - 'clarification' nodes (skipped during execution)
      - 'to_be_expanded' node promotion back to 'goal' type
      - goal auto-completion once all related tasks are done
      - awaiting_input resumption keyed on clarification-field values
      - broadening metadata write-back and propagation

    All imports and call sites that reference Orchestrator by name continue to
    work without change; only the implementation is now split between
    BaseOrchestrator (generic) and this subclass (domain-specific).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # FIX #3: track goals whose replan was deferred because children were
        # still running at the time replan_goal() was called.  The set is
        # drained by _complete_deferred_replans(), which is called from
        # _post_planning_hooks() every loop iteration.
        self._pending_replan_goals: set[str] = set()

    # ── Hook overrides ────────────────────────────────────────────────────────

    def _is_executable_node(self, node) -> bool:
        """
        Skip both 'clarification' nodes (filled by the UI, not the executor)
        and 'execution_step' nodes (managed by ExecutionStepReporter).
        """
        return node.node_type not in ("clarification", "execution_step")

    def _pre_planning_hooks(self) -> None:
        """Promote to_be_expanded nodes to goal type before planning runs."""
        self._expansion_request_pass()

    def _post_planning_hooks(self) -> None:
        """Auto-complete finished goals; resume clarification-unblocked nodes."""
        self._complete_finished_goals()
        self._resume_unblocked_pass()
        # FIX #3: complete any deferred replan resets once running children finish.
        self._complete_deferred_replans()

    def _handle_broadening(self, node_id: str, reporter) -> "list | None":
        """
        If the node ran with a broadened description, write back all broadening
        metadata and propagate the broadened output contract to downstream nodes.
        If the node ran normally but stale broadening metadata exists from a
        previous attempt, clear it.

        Returns the broadened_output list when broadening occurred (used as
        produced_output on success), or None to fall back to declared output.
        """
        if reporter is None:
            return None

        pending = getattr(reporter, "pending_broadening", None)

        if pending:
            signal = pending
            with self.graph_lock:
                if node_id in self.graph.nodes:
                    if signal.new_fields:
                        self._patch_clarification_node(
                            node_id,
                            signal.new_fields,
                            signal.clarification_node_id,
                        )
                    self._apply(
                        Event(
                            UPDATE_METADATA,
                            {
                                "node_id": node_id,
                                "metadata": {
                                    "broadened_description": signal.broadened_description,
                                    "broadened_for_missing": signal.broadened_for_missing,
                                    "broadened_reason": signal.reason,
                                    "broadened_output": signal.broadened_output,
                                    "broadened_steps": signal.broadened_steps,
                                },
                            },
                        )
                    )
                    logger.info(
                        "[EXEC] Node %s executed with broadened description (missing: %s)",
                        node_id,
                        signal.broadened_for_missing,
                    )
                    # Propagate broadened context to direct downstream dependents.
                    # Each dependent receives its own broadened_steps derived from
                    # what this node is actually producing, so it can adapt its
                    # execution if it also lacks the missing inputs.
                    self._propagate_broadened_steps_to_dependents(node_id, signal)
            return signal.broadened_output if signal.broadened_output else None
        else:
            with self.graph_lock:
                node = self.graph.nodes.get(node_id)
                if node and node.metadata.get("broadened_description"):
                    logger.info(
                        "[EXEC] Node %s: clearing stale broadened metadata "
                        "(node now runs with original description)",
                        node_id,
                    )
                    self._apply(
                        Event(
                            UPDATE_METADATA,
                            {
                                "node_id": node_id,
                                "metadata": {
                                    "broadened_description": "",
                                    "broadened_for_missing": [],
                                    "broadened_reason": "",
                                    "broadened_output": [],
                                    "broadened_steps": [],
                                },
                            },
                        )
                    )
            return None

    def _on_awaiting_user_complete(self, node_id: str, result: dict, reporter) -> None:
        """
        Write back broadening metadata in the awaiting-user path.
        Mirrors the broadening write-back in _handle_broadening but for the
        case where the node reached awaiting_user instead of done/failed.
        Called with graph_lock held.
        """
        broadening_signal = getattr(reporter, "pending_broadening", None) if reporter else None
        if not broadening_signal:
            return

        if broadening_signal.new_fields:
            self._patch_clarification_node(
                node_id,
                broadening_signal.new_fields,
                broadening_signal.clarification_node_id,
            )
        self._apply(
            Event(
                UPDATE_METADATA,
                {
                    "node_id": node_id,
                    "metadata": {
                        "broadened_description": broadening_signal.broadened_description,
                        "broadened_for_missing": broadening_signal.broadened_for_missing,
                        "broadened_reason": broadening_signal.reason,
                        "broadened_output": broadening_signal.broadened_output,
                        "broadened_steps": broadening_signal.broadened_steps,
                    },
                },
            )
        )
        logger.info(
            "[EXEC] Node %s (awaiting_user) broadened metadata written back",
            node_id,
        )

    # ── Goal auto-completion ─────────────────────────────────────────────────

    def _complete_finished_goals(self):
        with self.graph_lock:
            for node in self.graph.nodes.values():
                if node.node_type != "goal":
                    continue
                if node.status in ("done", "failed"):
                    continue
                if not node.metadata.get("expanded", False):
                    continue

                # Check direct dependencies (upstream tasks the goal relies on)
                # and direct children (nodes that depend on this goal).
                # Transitivity is implicit: an intermediate sub-goal only reaches
                # "done" after _complete_finished_goals has already processed it,
                # so checking direct relations is sufficient — no BFS needed.
                related = set(node.dependencies) | set(node.children)
                if not related:
                    continue

                all_done = all(
                    self.graph.nodes[nid].status == "done"
                    for nid in related
                    if nid in self.graph.nodes
                )
                if all_done:
                    existing_result = node.result
                    self._apply(
                        Event(
                            MARK_DONE,
                            {
                                "node_id": node.id,
                                "result": existing_result,
                            },
                        )
                    )
                    logger.info("[ORCHESTRATOR] Goal completed: %s", node.id)

    # ── Expansion request pass ────────────────────────────────────────────────

    def _expansion_request_pass(self):
        with self.graph_lock:
            to_expand = [n.id for n in self.graph.nodes.values() if n.status == "to_be_expanded"]

        for node_id in to_expand:
            logger.info("[ORCHESTRATOR] Expansion requested for node: %s", node_id)

            with self.graph_lock:
                n = self.graph.nodes.get(node_id)
                if not n:
                    continue

                self._apply(
                    Event(
                        SET_NODE_TYPE,
                        {
                            "node_id": node_id,
                            "node_type": "goal",
                        },
                    )
                )
                self._apply(
                    Event(
                        UPDATE_METADATA,
                        {
                            "node_id": node_id,
                            "metadata": {
                                "expanded": False,
                                "description": n.metadata.get("description", node_id),
                            },
                        },
                    )
                )
                self._apply(Event(RESET_NODE, {"node_id": node_id}))

                to_reset = []
                queue = list(self.graph.nodes[node_id].children)
                visited = set()
                while queue:
                    child_id = queue.pop()
                    if child_id in visited or child_id not in self.graph.nodes:
                        continue
                    visited.add(child_id)
                    child = self.graph.nodes[child_id]
                    if child.status != "running":
                        to_reset.append(child_id)
                    # FIX B: always extend the BFS queue regardless of whether
                    # this child is running.  The old code only walked children
                    # of non-running nodes, so any "done" grandchildren below a
                    # running node were silently skipped.  Those grandchildren
                    # retained stale "done" results from the old plan after the
                    # goal was re-expanded, leaving the subtree inconsistent.
                    queue.extend(child.children)

                for desc_id in to_reset:
                    self._apply(Event(RESET_NODE, {"node_id": desc_id}))
                    logger.info("[ORCHESTRATOR] Reset dependent for re-execution: %s", desc_id)

    # ── awaiting_input resumption ─────────────────────────────────────────────

    def _resume_unblocked_pass(self) -> int:
        """
        Scan all awaiting_input nodes and resume those that are now unblocked.

        Two resume paths:

        Path A — specific fields (normal case):
          The node has missing_fields populated.  Resume when every listed key
          is now non-unknown in the upstream clarification node.

        Path B — no specific fields (fallback):
          The node has missing_fields=[] (the preflight LLM couldn't identify a
          specific field, or the block is on personal data with no matching field).
          Resume when ANY previously-unknown clarification field becomes filled.
          This ensures these nodes are not permanently stuck.

        Returns the number of nodes resumed.
        """
        _PLACEHOLDERS = {
            "unknown",
            "n/a",
            "not specified",
            "not provided",
            "none",
            "unspecified",
            "tbd",
            "",
        }

        with self.graph_lock:
            snapshot = self.graph.get_snapshot()
            # FIX: build the awaiting list from the snapshot (an immutable copy)
            # rather than from self.graph.nodes (the live mutable collection).
            awaiting = [n for n in snapshot.values() if n.status == "awaiting_input"]

        resumed = 0
        for node in awaiting:
            missing_keys = node.metadata.get("missing_fields", [])

            # FIX #6: use BFS over all ancestors to find the upstream
            # clarification node, instead of scanning only direct dependencies.
            # _patch_clarification_node already documents that the clarification
            # node "may be 2+ hops away" (e.g. attached to the goal node which
            # feeds an intermediate task which feeds this awaiting node).  The
            # previous direct-dep scan silently missed those cases, leaving the
            # awaiting_input node permanently stuck even after the user filled
            # in the required fields.
            clar_node = None
            bfs_queue = list(node.dependencies)
            bfs_visited: set[str] = set(bfs_queue)
            while bfs_queue and clar_node is None:
                dep_id = bfs_queue.pop(0)
                dep = snapshot.get(dep_id)
                if dep is None:
                    continue
                if dep.node_type == "clarification" and dep.result:
                    clar_node = dep
                    break
                for ancestor_id in dep.dependencies:
                    if ancestor_id not in bfs_visited:
                        bfs_visited.add(ancestor_id)
                        bfs_queue.append(ancestor_id)

            if not clar_node:
                continue

            try:
                fields = json.loads(clar_node.result)
            except Exception:
                continue

            should_resume = False

            if missing_keys:
                # Path A: all specific missing keys must now be filled
                still_missing = []
                for key in missing_keys:
                    matched = False
                    for f in fields:
                        if f.get("key") == key:
                            matched = True
                            val = str(f.get("value", "")).strip().lower()
                            if val in _PLACEHOLDERS:
                                still_missing.append(key)
                            break
                    if not matched:
                        still_missing.append(key)
                should_resume = not still_missing
            else:
                # Path B: no specific fields — resume when any field is now filled
                any_filled = any(
                    str(f.get("value", "")).strip().lower() not in _PLACEHOLDERS for f in fields
                )
                if any_filled:
                    should_resume = True
                    logger.info(
                        "[ORCHESTRATOR] Node %s (no specific missing_fields) — "
                        "resuming because at least one clarification field is now filled",
                        node.id,
                    )

            if should_resume:
                with self.graph_lock:
                    if node.id in self.graph.nodes:
                        self._apply(Event(RESUME_NODE, {"node_id": node.id}))
                        logger.info(
                            "[ORCHESTRATOR] Node %s resumed — missing fields: %s",
                            node.id,
                            missing_keys or "(none specified)",
                        )
                        resumed += 1

        return resumed

    # ── Clarification node helpers ────────────────────────────────────────────

    def _patch_clarification_node(
        self,
        task_node_id: str,
        new_fields: list,
        hint_clar_id: str = "",
    ) -> None:
        """
        Add new_fields to the upstream clarification node for task_node_id,
        skipping any field whose key is already present.

        Must be called with self.graph_lock held.
        """
        assert self.graph_lock._is_owned(), (  # noqa: SLF001
            "_patch_clarification_node must be called with self.graph_lock held"
        )
        clar_id = hint_clar_id
        if not clar_id or clar_id not in self.graph.nodes:
            # ── FIX: BFS over all ancestors instead of direct-dep-only scan ──
            # The clarification node may be 2+ hops away (e.g. attached to the
            # goal node, which feeds an intermediate task, which feeds this one).
            # The previous direct-dep loop silently failed for those cases and
            # logged "no clarification node found upstream" even though one
            # existed; the BFS ensures we always find it.
            node = self.graph.nodes.get(task_node_id)
            if not node:
                return
            queue = list(node.dependencies)
            visited: set[str] = set(queue)
            while queue and not clar_id:
                dep_id = queue.pop(0)
                dep = self.graph.nodes.get(dep_id)
                if dep is None:
                    continue
                if dep.node_type == "clarification":
                    clar_id = dep_id
                    break
                for ancestor_id in dep.dependencies:
                    if ancestor_id not in visited:
                        visited.add(ancestor_id)
                        queue.append(ancestor_id)

        if not clar_id or clar_id not in self.graph.nodes:
            logger.warning(
                "[ORCHESTRATOR] Cannot patch clarification node for %s "
                "— no clarification node found upstream (BFS exhausted)",
                task_node_id,
            )
            return

        clar = self.graph.nodes[clar_id]
        existing_fields = clar.metadata.get("fields", [])
        existing_keys = {f.get("key") for f in existing_fields}

        fields_to_add = [f for f in new_fields if f.get("key") not in existing_keys]
        if not fields_to_add:
            return

        updated_fields = existing_fields + fields_to_add
        self._apply(
            Event(
                UPDATE_METADATA,
                {
                    "node_id": clar_id,
                    "metadata": {"fields": updated_fields},
                },
            )
        )

        current_result: str | None = clar.result

        try:
            result_fields = json.loads(current_result) if current_result else []
            result_fields.extend(fields_to_add)
            self._apply(
                Event(
                    SET_RESULT,
                    {
                        "node_id": clar_id,
                        "result": json.dumps(result_fields, ensure_ascii=False),
                    },
                )
            )
        except Exception as e:
            logger.warning(
                "[ORCHESTRATOR] Failed to patch clarification result for %s: %s",
                clar_id,
                e,
            )

        logger.info(
            "[ORCHESTRATOR] Patched clarification node %s with new field(s): %s",
            clar_id,
            [f.get("key") for f in fields_to_add],
        )

    def _propagate_broadened_steps_to_dependents(
        self, node_id: str, signal: "AwaitingInputSignal"
    ) -> None:
        """
        When a node ran with a broadened goal, notify its direct downstream
        dependents so they can also prepare broadened_steps.

        For each pending/ready dependent whose required_input includes an output
        from this node, we write the upstream node's broadened_output contract
        into the dependent's metadata under ``upstream_broadened_outputs``.
        The dependent's own _preflight_awaiting_input call will detect this on
        its next execution attempt and generate appropriate broadened_steps.

        Must be called with self.graph_lock held.
        """
        assert self.graph_lock._is_owned(), (  # noqa: SLF001
            "_propagate_broadened_steps_to_dependents must be called with graph_lock held"
        )
        node = self.graph.nodes.get(node_id)
        if not node:
            return

        # The names the upstream node will actually produce (broadened contract)
        broadened_output_names = {
            o.get("name") for o in signal.broadened_output if isinstance(o, dict) and o.get("name")
        }
        if not broadened_output_names:
            return

        for child_id in node.children:
            child = self.graph.nodes.get(child_id)
            if child is None or child.status in ("done", "running", "failed"):
                continue

            # Check whether this child declared any of the upstream outputs as required_input
            child_required = {
                r.get("name")
                for r in child.metadata.get("required_input", [])
                if isinstance(r, dict) and r.get("name")
            }
            if not child_required.intersection(broadened_output_names):
                continue

            # Store the broadened output contract from this upstream node so the
            # child's preflight can detect it and generate matching broadened_steps.
            existing = child.metadata.get("upstream_broadened_outputs", {})
            existing[node_id] = signal.broadened_output
            self._apply(
                Event(
                    UPDATE_METADATA,
                    {
                        "node_id": child_id,
                        "metadata": {"upstream_broadened_outputs": existing},
                    },
                )
            )
            logger.info(
                "[ORCHESTRATOR] Propagated broadened output contract from %s → %s",
                node_id,
                child_id,
            )

    # ── Domain public API ─────────────────────────────────────────────────────

    def add_goal(self, goal_id: str, description: str = "", dependencies: list = None):
        with self.graph_lock:
            self._apply(
                Event(
                    ADD_NODE,
                    {
                        "node_id": goal_id,
                        "node_type": "goal",
                        "dependencies": dependencies or [],
                        "origin": "user",
                        "metadata": {"description": description, "expanded": False},
                    },
                )
            )
        logger.info("[USER] Added goal: %s", goal_id)

    def replan_goal(self, goal_id: str):
        with self.graph_lock:
            goal = self.graph.nodes.get(goal_id)
            if not goal or goal.node_type != "goal":
                return

            running_children = [cid for cid in goal.children if cid in self._running_futures]
            if running_children:
                logger.warning(
                    "[ORCHESTRATOR] replan_goal(%s): %d child(ren) still running "
                    "(%s) — removing idle children and registering goal for "
                    "deferred replan once all running children complete.",
                    goal_id,
                    len(running_children),
                    running_children,
                )

            for cid in list(goal.children):
                if (
                    cid in self.graph.nodes
                    and self.graph.nodes[cid].status in ("pending", "ready")
                    and cid not in self._running_futures
                ):
                    self._apply(Event(REMOVE_NODE, {"node_id": cid}))

            if not running_children:
                # No children running — apply the reset immediately.
                self._apply(
                    Event(
                        UPDATE_METADATA,
                        {
                            "node_id": goal_id,
                            "metadata": {"expanded": False},
                        },
                    )
                )
            else:
                # FIX #3: register the goal so _complete_deferred_replans()
                # will apply expanded=False once every running child finishes.
                # Without this registration the deferred reset was never
                # completed, leaving the goal permanently stuck as expanded=True
                # with no children and never replanned.
                self._pending_replan_goals.add(goal_id)
                logger.info(
                    "[ORCHESTRATOR] replan_goal(%s): deferred reset registered — "
                    "will fire after running children %s complete",
                    goal_id,
                    running_children,
                )

    def _complete_deferred_replans(self) -> None:
        """
        Complete deferred replan resets registered by replan_goal().

        Called from _post_planning_hooks() every loop iteration.  For each
        goal in _pending_replan_goals, if no children are currently running,
        set expanded=False so _planning_pass picks it up for re-expansion.
        """
        if not self._pending_replan_goals:
            return

        with self.graph_lock:
            to_clear = []
            for goal_id in list(self._pending_replan_goals):
                goal = self.graph.nodes.get(goal_id)
                if goal is None:
                    # Goal was removed — discard silently.
                    to_clear.append(goal_id)
                    continue
                still_running = [cid for cid in goal.children if cid in self._running_futures]
                if not still_running:
                    # All previously-running children have finished; apply reset.
                    self._apply(
                        Event(
                            UPDATE_METADATA,
                            {
                                "node_id": goal_id,
                                "metadata": {"expanded": False},
                            },
                        )
                    )
                    to_clear.append(goal_id)
                    logger.info(
                        "[ORCHESTRATOR] Deferred replan for %s complete — "
                        "expanded=False applied, goal will be replanned next cycle",
                        goal_id,
                    )
            for goal_id in to_clear:
                self._pending_replan_goals.discard(goal_id)
