# engine/llm_orchestrator.py

import json
import threading
import time
import os
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

from cuddlytoddly.core.events import (
    Event,
    ADD_NODE, ADD_DEPENDENCY, REMOVE_DEPENDENCY, UPDATE_METADATA,
    MARK_RUNNING, MARK_DONE, MARK_FAILED, REMOVE_NODE, RESET_NODE,
    SET_NODE_TYPE
)
from cuddlytoddly.core.reducer import apply_event
from cuddlytoddly.infra.event_queue import EventQueue
from cuddlytoddly.infra.logging import get_logger
from cuddlytoddly.planning.llm_interface import LLMStoppedError
from cuddlytoddly.engine.execution_step_reporter import ExecutionStepReporter
from cuddlytoddly.planning.llm_interface import token_counter

logger = get_logger(__name__)

PlanningContext = namedtuple("PlanningContext", ["snapshot", "goals"])

_IDLE_SLEEP = 0.5

# Sentinel used to distinguish "never stored" from "stored value was None"
_NO_PREV = object()

# Maximum number of times the orchestrator will attempt to inject a bridging
# node for any single blocked node before giving up and just launching it.
_MAX_GAP_FILL_ATTEMPTS = 2

class SimpleOrchestrator:
    """
    Minimal orchestrator: LLM plans, executor runs, user edits via the UI.

    The curses UI expects:
        .graph, .graph_lock, .event_queue
        .current_activity, .activity_started
        .llm_stopped, .stop_llm_calls(), .resume_llm_calls()
    """
    @property
    def token_counts(self) -> dict:
        return {
            "prompt":     token_counter.prompt_tokens,
            "completion": token_counter.completion_tokens,
            "total":      token_counter.total_tokens,
            "calls":      token_counter.calls,
        }
    
    def __init__(self, graph, planner, executor,
                 event_log=None, event_queue=None, max_workers=4, quality_gate=None):
        self.graph       = graph
        self.planner     = planner
        self.executor    = executor
        self.event_log   = event_log
        self.event_queue = event_queue or EventQueue()
        self.max_workers = max_workers
        self.quality_gate = quality_gate 

        # UI contract
        self.graph_lock       = threading.RLock()
        self.current_activity: str | None   = None
        self.activity_started: float | None = None

        # Internals
        self._pool                          = ThreadPoolExecutor(max_workers=max_workers)
        self._running_futures: dict[str, object] = {}
        self._stop_event                    = threading.Event()
        self._thread: threading.Thread | None = None

        # Stores each node's result immediately before it starts executing so
        # _on_node_done can compare old vs new and decide whether to cascade.
        self._prev_results: dict[str, object] = {}

        # LLM clients for pause/resume — collected from planner and executor
        self._llm_clients = []
        self._reporters: dict[str, ExecutionStepReporter] = {}

        for component in (planner, executor, quality_gate):
            if component is not None and hasattr(component, "llm"):
                self._llm_clients.append(component.llm)

    # ── LLM pause / resume ───────────────────────────────────────────────────

    @property
    def llm_stopped(self) -> bool:
        return any(getattr(c, "is_stopped", False) for c in self._llm_clients)

    def stop_llm_calls(self) -> None:
        for c in self._llm_clients:
            if hasattr(c, "stop"):
                c.stop()
        logger.warning("[ORCHESTRATOR] LLM calls PAUSED")

    def resume_llm_calls(self) -> None:
        for c in self._llm_clients:
            if hasattr(c, "resume"):
                c.resume()
        logger.info("[ORCHESTRATOR] LLM calls RESUMED")

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        """Start in a background thread. On macOS+Metal use run_on_main_thread instead."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="simple-orchestrator"
        )
        self._thread.start()
        logger.info("[ORCHESTRATOR] Started (background thread)")

    def run_on_main_thread(self):
        """Run the loop on the calling thread (blocks).
        Required on macOS with Metal: llama.cpp GPU work must stay on the main thread."""
        self._stop_event.clear()
        self._thread = threading.current_thread()
        logger.info("[ORCHESTRATOR] Started (main thread)")
        self._loop()

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=10)
        self._pool.shutdown(wait=False)
        logger.info("[ORCHESTRATOR] Stopped")

    @property
    def is_running(self):
        return not self._stop_event.is_set()

    # ── Main loop ────────────────────────────────────────────────────────────

    def _loop(self):
        _last_idle_log = 0
        while not self._stop_event.is_set():
            try:
                self._drain_event_queue()
                self._expansion_request_pass()   # ← first, convert any expansion requests
                planned  = self._planning_pass()
                self._complete_finished_goals()
                launched = self._execution_pass()

                if planned == 0 and launched == 0:
                    now = time.time()
                    if now - _last_idle_log > 30:
                        if self._running_futures:
                            logger.debug(
                                "[ORCHESTRATOR] Waiting on %d running node(s): %s",
                                len(self._running_futures),
                                list(self._running_futures.keys()),
                            )
                        else:
                            logger.debug("[ORCHESTRATOR] Idle — nothing to do")
                        _last_idle_log = now

                    if self._is_fully_done():
                        time.sleep(_IDLE_SLEEP * 4)
                    else:
                        time.sleep(_IDLE_SLEEP)

            except Exception as e:
                logger.exception("[ORCHESTRATOR] Unhandled error in main loop: %s", e)
                time.sleep(_IDLE_SLEEP)

    # ── Event queue drain ────────────────────────────────────────────────────

    def _drain_event_queue(self):
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get()
                if event.type == "RESET_SUBTREE":
                    logger.info("[ORCHESTRATOR] RESET_SUBTREE received for: %s",
                                event.payload.get("node_id"))
                    # Handled here — needs knowledge of _running_futures
                    self._reset_subtree_impl(event.payload["node_id"])
                else:
                    with self.graph_lock:
                        self._apply(event)
            except Exception as e:
                logger.error("[ORCHESTRATOR] Error draining event: %s", e)
 
    def _reset_subtree_impl(self, root_id: str):
        """
        Reset root_id and every descendant (via children) to pending,
        clearing results and verification metadata.
        Skips any node that is currently running — those will be reset
        when their future resolves.
        Called with NO lock held; acquires it internally.
        """
        with self.graph_lock:
            if root_id not in self.graph.nodes:
                return
 
            # BFS over children (the "depends on this" direction)
            to_reset = []
            queue = [root_id]
            visited = set()
            while queue:
                nid = queue.pop(0)
                if nid in visited:
                    continue
                visited.add(nid)
                node = self.graph.nodes.get(nid)
                if not node:
                    continue
                to_reset.append(nid)
                queue.extend(node.children)
 
            for nid in to_reset:
                if nid in self._running_futures:
                    # Leave running nodes alone — the executor will
                    # mark them done/failed naturally; the result will
                    # be discarded by the quality gate on the next retry.
                    logger.debug("[RESET_SUBTREE] Skipping running node: %s", nid)
                    continue
 
                node = self.graph.nodes.get(nid)
                if not node:
                    continue
 
                node.status = "pending"
                node.result = None
                node.metadata.pop("verified",              None)
                node.metadata.pop("verification_failure",  None)
                node.metadata.pop("retry_count",           None)
                # If this is a goal that was already expanded, mark it
                # for re-expansion so the planner revisits it. 
                logger.info("[RESET_SUBTREE] Reset: %s", nid)
 
            self.graph.recompute_readiness()
 
    # ── Planning pass ────────────────────────────────────────────────────────

    def _planning_pass(self) -> int:
        if self.llm_stopped:
            return 0

        with self.graph_lock:
            unexpanded = [
                n for n in self.graph.nodes.values()
                if n.node_type == "goal"
                and not n.metadata.get("expanded", False)
            ]

        total = 0
        for goal in unexpanded:
            if self._stop_event.is_set() or self.llm_stopped:
                break

            self.current_activity = f"Planning: {goal.id}"
            self.activity_started = time.time()
            logger.info("[PLAN] Expanding goal: %s", goal.id)

            with self.graph_lock:
                branch = self.graph.get_branch(goal.id)

            context = PlanningContext(snapshot=branch, goals=[goal])
            try:
                events = self.planner.propose(context)
            except Exception as e:
                logger.error("[PLAN] Planner failed for %s: %s", goal.id, e)
                events = []
            finally:
                self.current_activity = None
                self.activity_started = None

            with self.graph_lock:
                for evt in events:
                    self._apply(Event(evt["type"], evt["payload"]))
                    total += 1
                self._apply(Event(UPDATE_METADATA, {
                    "node_id": goal.id,
                    "metadata": {"expanded": True},
                }))

            logger.info("[PLAN] Goal %s → %d events", goal.id, len(events))

        return total

    # ── Goal auto-completion ─────────────────────────────────────────────────

    def _complete_finished_goals(self):
        """
        Mark a goal as done when all its work is finished.

        The planner can wire goals two ways:
          A) goal has tasks as children     (node.children non-empty)
          B) goal depends on tasks          (node.dependencies non-empty)

        Handle both: done when every dependency AND every child is done.
        """
        with self.graph_lock:
            for node in self.graph.nodes.values():
                if node.node_type != "goal":
                    continue
                if node.status in ("done", "failed"):
                    continue
                if not node.metadata.get("expanded", False):
                    continue

                related = set(node.dependencies) | set(node.children)
                if not related:
                    continue

                all_done = all(
                    self.graph.nodes[nid].status == "done"
                    for nid in related
                    if nid in self.graph.nodes
                )
                if all_done:
                    # Preserve the plan_summary written by the planner at expansion time
                    existing_result = node.result
                    self._apply(Event(MARK_DONE, {
                        "node_id": node.id,
                        "result":  existing_result,
                    }))
                    logger.info("[ORCHESTRATOR] Goal completed: %s", node.id)

    # ── Execution pass ───────────────────────────────────────────────────────

    def _execution_pass(self) -> int:
        launched = 0

        with self.graph_lock:
            ready = [
                n for n in self.graph.nodes.values()
                if n.status == "ready"
                and n.node_type == "task"
                and n.id not in self._running_futures
            ]

        for node in ready:
            if self._stop_event.is_set():
                break

            with self.graph_lock:
                current = self.graph.nodes.get(node.id)
                if not current or current.status != "ready":
                    continue
                # Snapshot the current result so _on_node_done can detect changes
                self._prev_results[node.id] = current.result
                snapshot = self.graph.get_snapshot()

            # ── Dependency gap check (skip if no quality gate or budget exhausted) ──
            attempts = node.metadata.get("gap_fill_attempts", 0)
            if self.quality_gate and attempts < _MAX_GAP_FILL_ATTEMPTS:
                bridge = self.quality_gate.check_dependencies(node, snapshot)
                if bridge is not None:
                    self._inject_bridge_node(bridge, node.id)
                    continue

            with self.graph_lock:
                current = self.graph.nodes.get(node.id)
                if not current or current.status != "ready":
                    continue
                self._apply(Event(MARK_RUNNING, {"node_id": node.id}))
                snapshot = self.graph.get_snapshot()

            logger.info("[EXEC] Launching: %s", node.id)
            self.current_activity = f"Executing: {node.id}"
            self.activity_started = time.time()

            reporter = ExecutionStepReporter(
                parent_node_id=node.id,
                apply_fn=self._apply,
                graph_lock=self.graph_lock,
                graph=self.graph,
            )
            self._reporters[node.id] = reporter

            future = self._pool.submit(self.executor.execute, node, snapshot, reporter)
            self._running_futures[node.id] = future
            future.add_done_callback(
                lambda fut, nid=node.id: self._on_node_done(nid, fut)
            )
            launched += 1

        return launched
    
    def _expansion_request_pass(self):
        with self.graph_lock:
            to_expand = [
                n.id for n in self.graph.nodes.values()
                if n.status == "to_be_expanded"
            ]

        for node_id in to_expand:
            logger.info("[ORCHESTRATOR] Expansion requested for node: %s", node_id)

            with self.graph_lock:
                n = self.graph.nodes.get(node_id)
                if not n:
                    continue

                # 1. Convert to an unexpanded goal

                self._apply(Event(SET_NODE_TYPE, {
                    "node_id": node_id,
                    "node_type":  "goal",
                }))

                self._apply(Event(UPDATE_METADATA, {
                    "node_id": node_id,
                    "metadata": {
                        "expanded": False,
                        "description": n.metadata.get("description", node_id),
                    },
                }))

                # 2. Reset the node itself via event (clears status, result)
                self._apply(Event(RESET_NODE, {"node_id": node_id}))

                # 3. Cascade-reset all transitive dependents
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
                        queue.extend(child.children)

                for desc_id in to_reset:
                    self._apply(Event(RESET_NODE, {"node_id": desc_id}))
                    logger.info("[ORCHESTRATOR] Reset dependent for re-execution: %s", desc_id)

    def _on_node_done(self, node_id: str, future):
            self._running_futures.pop(node_id, None)

            if self.current_activity and node_id in (self.current_activity or ""):
                if self._running_futures:
                    other = next(iter(self._running_futures))
                    self.current_activity = f"Executing: {other}"
                else:
                    self.current_activity = None
                    self.activity_started = None

            try:
                result = future.result()
            except Exception as exc:
                logger.warning("[EXEC] Node %s raised: %s", node_id, exc)
                result = None

            # ── Hard failure (executor returned None) ────────────────────────────
            if result is None:
                reporter = self._reporters.pop(node_id, None)
                with self.graph_lock:
                    if node_id not in self.graph.nodes:
                        return
                    if reporter:
                        reporter.expose_all()
                    self._apply(Event(MARK_FAILED, {"node_id": node_id}))
                    logger.warning("[EXEC] Failed: %s", node_id)
                return

            # ── Pre-flight: check file outputs were actually written ─────────────
            # Done before the LLM verification call to avoid wasting an inference.
            # Initialise with safe defaults so the code below always has values.
            satisfied:      bool | None = None
            reason:         str         = ""
            expected_files: list        = []
            tool_calls_made: set        = set()

            reporter = self._reporters.get(node_id)

            if reporter and self.quality_gate:
                with self.graph_lock:
                    declared_outputs = (
                        self.graph.nodes[node_id].metadata.get("output", [])
                        if node_id in self.graph.nodes else []
                    )
                    tool_calls_made = {
                        self.graph.nodes[sid].metadata.get("tool_name")
                        for sid in reporter._all_step_ids
                        if sid in self.graph.nodes
                    }

                expected_files = [
                    o for o in declared_outputs
                    if any(str(o).endswith(ext)
                        for ext in self.quality_gate.FILE_EXTENSIONS)
                ]

                if expected_files and "write_file" not in tool_calls_made:
                    # LLM skipped write_file — try to auto-write using the result
                    file_path = expected_files[0]
                    content   = result

                    # Strip labelled prefix: "file_written: foo.md\nsummary: ..."
                    import re
                    match = re.search(
                        r'(?:summary|content)\s*:\s*(.+)',
                        result, re.DOTALL | re.IGNORECASE
                    )
                    if match:
                        content = match.group(1).strip()

                    if content and len(content) > 50:
                        try:
                            tools = getattr(self.executor, "tools", None)
                            if tools:
                                tools.execute("write_file", {
                                    "path":    file_path,
                                    "content": content,
                                })
                                logger.info(
                                    "[EXEC] Auto-wrote '%s' for node %s (%d chars)",
                                    file_path, node_id, len(content)
                                )
                                result    = (
                                    f"file_written: {file_path}\n"
                                    f"summary: {content[:200]}"
                                )
                                satisfied = True
                                reason    = (
                                    "file written by orchestrator after LLM "
                                    "omitted write_file call"
                                )
                            else:
                                satisfied = False
                                reason    = (
                                    f"declared file output {expected_files} but "
                                    f"write_file was never called and no tool "
                                    f"registry available to auto-write"
                                )
                        except Exception as e:
                            logger.warning("[EXEC] Auto-write failed for %s: %s", node_id, e)
                            satisfied = False
                            reason    = (
                                f"declared file output but write_file not called "
                                f"and auto-write failed: {e}"
                            )
                    else:
                        satisfied = False
                        reason    = (
                            f"declared file output {expected_files} but write_file "
                            f"was never called and result has insufficient content "
                            f"to auto-write ({len(content)} chars)"
                        )

            # ── LLM quality gate (skipped if pre-flight already decided) ─────────
            if satisfied is None:
                if self.quality_gate:
                    with self.graph_lock:
                        if node_id not in self.graph.nodes:
                            return
                        node     = self.graph.nodes[node_id]
                        snapshot = self.graph.get_snapshot()

                    satisfied, reason = self.quality_gate.verify_result(
                        node, result, snapshot
                    )
                else:
                    satisfied = True
                    reason    = "no quality gate configured"

            # ── Apply outcome ─────────────────────────────────────────────────────
            reporter = self._reporters.pop(node_id, None)

            with self.graph_lock:
                if node_id not in self.graph.nodes:
                    return

                live_node   = self.graph.nodes[node_id]
                retry_count = live_node.metadata.get("retry_count", 0)

                if not satisfied:
                    if reporter:
                        reporter.expose_all()

                    if retry_count >= 3:
                        # Permanently fail — clean up step nodes but keep them
                        # visible so the user can inspect what went wrong
                        logger.error(
                            "[EXEC] Node %s failed verification %d time(s) — "
                            "permanently failing. Reason: %s",
                            node_id, retry_count + 1, reason
                        )
                        if reporter:
                            for step_id in list(reporter._all_step_ids):
                                if step_id in self.graph.nodes:
                                    # Unlink from parent's dependency set so the
                                    # graph is consistent, but leave the step node
                                    # itself visible for inspection
                                    live_node.dependencies.discard(step_id)
                                    self._apply(Event(REMOVE_DEPENDENCY, {
                                        "node_id":    node_id,
                                        "depends_on": step_id,
                                    }))

                        self._apply(Event(UPDATE_METADATA, {
                            "node_id":  node_id,
                            "metadata": {"verification_failure": reason},
                        }))
                        self._apply(Event(MARK_FAILED, {"node_id": node_id}))

                    else:
                        logger.warning(
                            "[EXEC] Verification failed for %s (attempt %d/3): %s",
                            node_id, retry_count + 1, reason
                        )
                        # Detach step nodes so retry starts with a clean reporter
                        if reporter:
                            for step_id in list(reporter._all_step_ids):
                                if step_id in self.graph.nodes:
                                    self.graph.detach_node(step_id)

                        self._apply(Event(UPDATE_METADATA, {
                            "node_id":  node_id,
                            "metadata": {
                                "verification_failure": reason,
                                "retry_count":          retry_count + 1,
                            },
                        }))
                        live_node.status = "pending"
                        live_node.result = None
                        self.graph.recompute_readiness()

                else:
                    # ── Success ───────────────────────────────────────────────
                    if reporter:
                        reporter.hide_all()

                    logger.info("[EXEC] Done: %s", node_id)
                    self._apply(Event(MARK_DONE, {
                        "node_id": node_id,
                        "result":  result,
                    }))
                    self._apply(Event(UPDATE_METADATA, {
                        "node_id":  node_id,
                        "metadata": {"verified": True},
                    }))

                    # ── Lazy cascade ──────────────────────────────────────────
                    # Only reset children that have already completed ("done")
                    # and only when the result actually changed.  Each child
                    # will in turn apply the same logic when it finishes,
                    # propagating the cascade all the way to the leaf nodes
                    # without touching nodes whose upstream input is unchanged.
                    prev_result = self._prev_results.pop(node_id, _NO_PREV)
                    result_changed = (prev_result is _NO_PREV) or (result != prev_result)

                    if result_changed:
                        logger.info(
                            "[EXEC] Result changed for %s — resetting done children",
                            node_id,
                        )
                        for child_id in list(live_node.children):
                            child = self.graph.nodes.get(child_id)
                            if child and child.status == "done":
                                logger.info(
                                    "[EXEC] Cascading rerun to child: %s", child_id
                                )
                                child.status = "pending"
                                child.result = None
                                child.metadata.pop("verified",             None)
                                child.metadata.pop("verification_failure", None)
                                child.metadata.pop("retry_count",          None)
                        self.graph.recompute_readiness()
                    else:
                        logger.info(
                            "[EXEC] Result unchanged for %s — children kept as-is",
                            node_id,
                        )

    # ── Bridge node injection ────────────────────────────────────────────────

    def _inject_bridge_node(self, bridge: dict, blocked_node_id: str):
        """
        Insert a single bridging task and re-wire it as a dependency of
        the blocked node. Marks the bridge fully_refined so the refinement
        cycle never promotes it into a goal and re-decomposes it.
        """
        with self.graph_lock:
            bridge_id = bridge["node_id"]

            # Guard: don't inject the same bridge twice
            if bridge_id in self.graph.nodes:
                logger.debug(
                    "[DEPCHECK] Bridge %s already exists — skipping injection", bridge_id
                )
            else:
                self._apply(Event(ADD_NODE, {
                    "node_id":   bridge_id,
                    "node_type": "task",
                    "origin":    "orchestrator",
                    "metadata":  {
                        "description":   bridge["description"],
                        "output":        [bridge["output"]],
                        "fully_refined": True,   # never re-decomposed
                        "gap_fill":      True,   # auditing marker
                    },
                }))
                self._apply(Event(ADD_DEPENDENCY, {
                    "node_id":    blocked_node_id,
                    "depends_on": bridge_id,
                }))
                logger.info(
                    "[DEPCHECK] Injected bridge node %s → unblocks %s",
                    bridge_id, blocked_node_id
                )

            # Increment attempt counter regardless, so we converge
            current_node = self.graph.nodes.get(blocked_node_id)
            if current_node:
                attempts = current_node.metadata.get("gap_fill_attempts", 0)
                self._apply(Event(UPDATE_METADATA, {
                    "node_id":  blocked_node_id,
                    "metadata": {"gap_fill_attempts": attempts + 1},
                }))

    def verify_restored_nodes(self):
        FILE_EXTENSIONS = {
            ".md", ".txt", ".py", ".json", ".csv", ".html",
            ".yaml", ".yml", ".xml", ".pdf", ".log",
        }

        with self.graph_lock:
            done_tasks = [
                n for n in self.graph.nodes.values()
                if n.node_type == "task"
                and n.status == "done"
                and n.result is not None
            ]

        # ── Pass 1: file-existence check on ALL done nodes, even verified ones ───
        # verified=True only means the LLM was satisfied — it can't know if a file
        # was deleted or never written. This check is cheap so always run it.
        for node in done_tasks:
            if self.quality_gate and self.quality_gate._looks_like_filename(node.result):
                path = node.result.strip()
                if not self.quality_gate._file_exists(path):
                    logger.warning(
                        "[STARTUP] Node %s result is '%s' but file does not exist — resetting",
                        node.id, path
                    )
                    with self.graph_lock:
                        n = self.graph.nodes.get(node.id)
                        if n:
                            n.status = "pending"
                            n.result = None
                            n.metadata["verification_failure"] = (
                                f"file '{path}' does not exist on disk"
                            )
                            n.metadata.pop("verified", None)
                            n.metadata["retry_count"] = n.metadata.get("retry_count", 0) + 1


        # ── Pass 2: LLM verification for nodes never verified ────────────────────
        with self.graph_lock:
            candidates = [
                n for n in self.graph.nodes.values()
                if n.node_type == "task"
                and n.status == "done"
                and n.result is not None
                and not n.metadata.get("verified", False)
            ]

        if not candidates:
            logger.info("[STARTUP] All restored nodes already verified — nothing to check")
        else:
            logger.info("[STARTUP] %d restored node(s) need verification", len(candidates))

        for node in candidates:
            if self._stop_event.is_set():
                break

            logger.info("[STARTUP] Verifying restored node: %s", node.id)
            self.current_activity = f"Verifying: {node.id}"
            self.activity_started = time.time()

            with self.graph_lock:
                snapshot = self.graph.get_snapshot()

            satisfied, reason = self._verify_result(node, node.result, snapshot)

            self.current_activity = None
            self.activity_started = None

            with self.graph_lock:
                if node.id not in self.graph.nodes:
                    continue

                if not satisfied:
                    logger.warning(
                        "[STARTUP] Restored node %s failed verification: %s — resetting to pending",
                        node.id, reason
                    )
                    n = self.graph.nodes[node.id]
                    n.status  = "pending"
                    n.result  = None
                    n.metadata["verification_failure"] = reason
                    n.metadata.pop("verified", None)
                    n.metadata["retry_count"] = n.metadata.get("retry_count", 0) + 1

                else:
                    logger.info("[STARTUP] Restored node %s verified OK", node.id)
                    self._apply(Event(UPDATE_METADATA, {
                        "node_id":  node.id,
                        "metadata": {"verified": True},
                    }))

        with self.graph_lock:
            self.graph.recompute_readiness()

        ready = sum(1 for n in self.graph.nodes.values() if n.status == "ready")
        logger.info("[STARTUP] Post-verification readiness: %d node(s) ready", ready)

    # ── User-facing edit API ─────────────────────────────────────────────────

    def add_goal(self, goal_id: str, description: str = "", dependencies: list = None):
        with self.graph_lock:
            self._apply(Event(ADD_NODE, {
                "node_id":      goal_id,
                "node_type":    "goal",
                "dependencies": dependencies or [],
                "origin":       "user",
                "metadata":     {"description": description, "expanded": False},
            }))
        logger.info("[USER] Added goal: %s", goal_id)

    def add_task(self, node_id: str, dependencies: list = None,
                 description: str = "", metadata: dict = None):
        meta = {"description": description, "fully_refined": True}
        if metadata:
            meta.update(metadata)
        with self.graph_lock:
            self._apply(Event(ADD_NODE, {
                "node_id":      node_id,
                "node_type":    "task",
                "dependencies": dependencies or [],
                "origin":       "user",
                "metadata":     meta,
            }))

    def remove_node(self, node_id: str):
        with self.graph_lock:
            if node_id in self._running_futures:
                logger.warning("[USER] Cannot remove %s — currently running", node_id)
                return
            self._apply(Event(REMOVE_NODE, {"node_id": node_id}))

    def add_dependency(self, node_id: str, depends_on: str):
        with self.graph_lock:
            self._apply(Event(ADD_DEPENDENCY, {
                "node_id": node_id, "depends_on": depends_on,
            }))

    def remove_dependency(self, node_id: str, depends_on: str):
        with self.graph_lock:
            self._apply(Event(REMOVE_DEPENDENCY, {
                "node_id": node_id, "depends_on": depends_on,
            }))

    def retry_node(self, node_id: str):
        with self.graph_lock:
            node = self.graph.nodes.get(node_id)
            if not node or node_id in self._running_futures:
                return
            node.status = "pending"
            node.result = None
            self.graph.recompute_readiness()

    def replan_goal(self, goal_id: str):
        with self.graph_lock:
            goal = self.graph.nodes.get(goal_id)
            if not goal or goal.node_type != "goal":
                return
            for cid in list(goal.children):
                if (cid in self.graph.nodes
                        and self.graph.nodes[cid].status in ("pending", "ready")
                        and cid not in self._running_futures):
                    self._apply(Event(REMOVE_NODE, {"node_id": cid}))
            self._apply(Event(UPDATE_METADATA, {
                "node_id": goal_id,
                "metadata": {"expanded": False},
            }))

    def update_metadata(self, node_id: str, metadata: dict):
        with self.graph_lock:
            self._apply(Event(UPDATE_METADATA, {
                "node_id": node_id, 
                "metadata": metadata,
            }))

    # ── Read access ──────────────────────────────────────────────────────────

    def get_snapshot(self):
        with self.graph_lock:
            return self.graph.get_snapshot()

    def get_status(self) -> dict:
        with self.graph_lock:
            nodes = list(self.graph.nodes.values())
        counts: dict[str, int] = {}
        for n in nodes:
            counts[n.status] = counts.get(n.status, 0) + 1
        return {
            "total":         len(nodes),
            "by_status":     counts,
            "running_nodes": list(self._running_futures.keys()),
        }

    # ── Internals ────────────────────────────────────────────────────────────

    def _apply(self, event: Event):
        """Apply one event. Must be called with graph_lock held."""
        apply_event(self.graph, event, event_log=self.event_log)

    def _is_fully_done(self) -> bool:
        with self.graph_lock:
            return all(
                n.status in ("done", "failed")
                for n in self.graph.nodes.values()
            )