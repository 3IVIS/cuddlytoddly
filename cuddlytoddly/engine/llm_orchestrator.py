# engine/llm_orchestrator.py

from __future__ import annotations

import json
import re
import threading
import time
from collections import deque, namedtuple
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cuddlytoddly.planning.llm_executor import AwaitingInputSignal

from cuddlytoddly.core.events import (
    ADD_DEPENDENCY,
    ADD_NODE,
    CONFIRM_USER_DONE,
    MARK_AWAITING_USER,
    MARK_DONE,
    MARK_FAILED,
    MARK_RUNNING,
    REMOVE_DEPENDENCY,
    REMOVE_NODE,
    RESET_NODE,
    RESUME_NODE,
    SET_NODE_TYPE,
    SET_RESULT,
    UPDATE_METADATA,
    Event,
)
from cuddlytoddly.core.reducer import apply_event
from cuddlytoddly.engine.execution_step_reporter import ExecutionStepReporter
from cuddlytoddly.infra.event_queue import EventQueue, StatusEvent
from cuddlytoddly.infra.logging import get_logger
from cuddlytoddly.planning.llm_interface import (
    BaseLLM,
    LLMStoppedError,
    TokenCounter,
    token_counter,
)

logger = get_logger(__name__)

# Maximum seconds the orchestrator waits before retrying a failed node.
# The actual delay is min(2**retry_count, _MAX_NODE_RETRY_BACKOFF_SECS).
_MAX_NODE_RETRY_BACKOFF_SECS: int = 60

PlanningContext = namedtuple(
    "PlanningContext",
    ["snapshot", "goals", "skip_scrutiny"],
    defaults=[False],
)

# Sentinel used to distinguish "never stored" from "stored value was None"
_NO_PREV = object()


class Orchestrator:
    """
    Minimal orchestrator: LLM plans, executor runs, user edits via the UI.

    The curses UI expects:
        .graph, .graph_lock, .event_queue
        .current_activity, .activity_started
        .llm_stopped, .stop_llm_calls(), .resume_llm_calls()

    All numeric tuning values (idle_sleep, max_gap_fill_attempts) are accepted
    as constructor parameters so they can be driven from config.toml without
    editing this file.
    """

    @property
    def token_counts(self) -> dict:
        # FIX #3: read from the per-run counter (self._token_counter) so that
        # concurrent runs in web-server mode each report their own usage.
        return {
            "prompt": self._token_counter.prompt_tokens,
            "completion": self._token_counter.completion_tokens,
            "total": self._token_counter.total_tokens,
            "calls": self._token_counter.calls,
        }

    def __init__(
        self,
        graph,
        planner,
        executor,
        event_log=None,
        event_queue=None,
        max_workers: int = 4,
        quality_gate=None,
        max_gap_fill_attempts: int = 2,
        idle_sleep: float = 0.5,
        max_retries: int = 5,
        verify_timeout: float = 300.0,
        token_counter_instance: "TokenCounter | None" = None,
    ):
        # FIX #3: use the per-run counter when provided; fall back to the
        # module-level singleton for backward compatibility.
        self._token_counter: TokenCounter = (
            token_counter_instance if token_counter_instance is not None else token_counter
        )
        self.graph = graph
        self.planner = planner
        self.executor = executor
        self.event_log = event_log
        self.event_queue = event_queue or EventQueue()
        self.max_workers = max_workers
        self.quality_gate = quality_gate
        self.max_gap_fill_attempts = max_gap_fill_attempts
        self.idle_sleep = idle_sleep
        self.max_retries = max_retries
        # Maximum seconds to spend in verify_restored_nodes().  Background
        # verification runs in a daemon thread; if the LLM hangs this prevents
        # the thread from parking forever holding the LLM connection.
        self.verify_timeout = verify_timeout

        # UI contract
        self.graph_lock = threading.RLock()
        self.current_activity: str | None = None
        self.activity_started: float | None = None

        # Internals
        self._pool = ThreadPoolExecutor(max_workers=max_workers)
        self._running_futures: dict[str, object] = {}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._prev_results: dict[str, object] = {}

        # Fix #6: register LLM clients using duck typing so that _DeferredLLM
        # (which wraps a real client but is not a BaseLLM subclass) is also
        # included.  We detect real LLM clients by requiring is_stopped to be
        # a genuine @property defined on the class — MagicMock/stub objects in
        # tests auto-create attributes on demand, so their class-level
        # descriptor is a MagicMock, not a property, and they are excluded.
        self._llm_clients = []
        self._reporters: dict[str, ExecutionStepReporter] = {}

        for component in (planner, executor, quality_gate):
            client = getattr(component, "llm", None)
            if client is None:
                continue
            is_real_llm = isinstance(client, BaseLLM) or isinstance(
                getattr(type(client), "is_stopped", None), property
            )
            if is_real_llm:
                self._llm_clients.append(client)

        # Out-of-band status events emitted by background threads (e.g.
        # llm_load_failed).  The UI polls this via get_status_events().
        self._status_events: list[StatusEvent] = []
        self._status_events_lock = threading.Lock()

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
        self._stop_event.clear()
        # FIX: Recreate the thread pool if stop() was called previously.
        # ThreadPoolExecutor.shutdown() is irreversible — any subsequent
        # _pool.submit() raises RuntimeError: cannot schedule new futures after
        # shutdown.  start() is documented as restartable, so we must replace
        # the dead pool before launching the loop thread.  We use the private
        # _shutdown attribute (stable across CPython 3.9-3.12) with a safe
        # getattr fallback so this degrades gracefully on other runtimes.
        if getattr(self._pool, "_shutdown", False):
            self._pool = ThreadPoolExecutor(max_workers=self.max_workers)
            logger.info("[ORCHESTRATOR] Thread pool recreated after previous shutdown")
        self._thread = threading.Thread(target=self._loop, daemon=True, name="simple-orchestrator")
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
                self._expansion_request_pass()
                planned = self._planning_pass()
                self._complete_finished_goals()
                self._resume_unblocked_pass()
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
                        time.sleep(self.idle_sleep * 4)
                    else:
                        time.sleep(self.idle_sleep)

            except Exception as e:
                logger.exception("[ORCHESTRATOR] Unhandled error in main loop: %s", e)
                time.sleep(self.idle_sleep)

    # ── Event queue drain ────────────────────────────────────────────────────

    def _drain_event_queue(self):
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get()

                # FIX: StatusEvent is a separate dataclass (kind/payload fields,
                # no .type) used for out-of-band signals like llm_load_failed.
                # The old code fell through to `event.type` which raised
                # AttributeError, caught silently by the except clause below —
                # permanently discarding the signal and leaving the UI with no
                # feedback when LLM loading fails.
                if isinstance(event, StatusEvent):
                    self._handle_status_event(event)
                    continue

                if event.type == "RESET_SUBTREE":
                    logger.info(
                        "[ORCHESTRATOR] RESET_SUBTREE received for: %s",
                        event.payload.get("node_id"),
                    )
                    self._reset_subtree_impl(event.payload["node_id"])
                else:
                    with self.graph_lock:
                        self._apply(event)
            except Exception as e:
                logger.error("[ORCHESTRATOR] Error draining event: %s", e)

    def _handle_status_event(self, event: StatusEvent) -> None:
        """
        Dispatch out-of-band status signals that are not graph mutations.

        Currently recognised kinds:
          llm_load_failed — background LLM loader failed; payload has "error".

        Unknown kinds are logged and stored so the UI can surface them.
        """
        logger.warning(
            "[ORCHESTRATOR] StatusEvent received: kind=%s payload=%s",
            event.kind,
            event.payload,
        )
        with self._status_events_lock:
            self._status_events.append(event)

        if event.kind == "llm_load_failed":
            error_msg = event.payload.get("error", "unknown error")
            logger.error("[ORCHESTRATOR] LLM load failed: %s", error_msg)
            # Set current_activity so the UI displays the error in its
            # activity bar rather than showing nothing.
            self.current_activity = f"LLM load failed: {error_msg}"
            self.activity_started = None

    def get_status_events(self) -> list:
        """
        Return and clear all buffered StatusEvents.
        Called by the web UI on each snapshot tick so errors are surfaced.
        """
        with self._status_events_lock:
            events = list(self._status_events)
            self._status_events.clear()
        return events

    def _reset_subtree_impl(self, root_id: str):
        with self.graph_lock:
            if root_id not in self.graph.nodes:
                return

            to_reset = []
            # Use deque for O(1) popleft(); list.pop(0) is O(n) per call.
            queue: deque = deque([root_id])
            visited = set()
            while queue:
                nid = queue.popleft()
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
                    logger.debug("[RESET_SUBTREE] Skipping running node: %s", nid)
                    continue
                node = self.graph.nodes.get(nid)
                if not node:
                    continue
                node.status = "pending"
                node.result = None
                node.metadata.pop("verified", None)
                node.metadata.pop("verification_failure", None)
                node.metadata.pop("retry_count", None)
                logger.info("[RESET_SUBTREE] Reset: %s", nid)

            self.graph.recompute_readiness()

    # ── Planning pass ────────────────────────────────────────────────────────

    def _planning_pass(self) -> int:
        if self.llm_stopped:
            return 0

        with self.graph_lock:
            unexpanded = [
                n
                for n in self.graph.nodes.values()
                if n.node_type == "goal" and not n.metadata.get("expanded", False)
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

            context = PlanningContext(
                snapshot=branch,
                goals=[goal],
                skip_scrutiny=bool(goal.children),
            )
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
                self._apply(
                    Event(
                        UPDATE_METADATA,
                        {
                            "node_id": goal.id,
                            "metadata": {"expanded": True},
                        },
                    )
                )

            logger.info("[PLAN] Goal %s → %d events", goal.id, len(events))

        return total

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

    # ── Execution pass ───────────────────────────────────────────────────────

    def _execution_pass(self) -> int:
        launched = 0

        with self.graph_lock:
            ready = [
                n
                for n in self.graph.nodes.values()
                if n.status == "ready"
                and n.node_type not in ("clarification", "execution_step")
                and n.id not in self._running_futures
            ]

        for node in ready:
            if self._stop_event.is_set():
                break
            if self.llm_stopped:
                # LLM is paused — leave ready nodes as-is so they resume
                # automatically the moment the user unpauses.
                break

            with self.graph_lock:
                current = self.graph.nodes.get(node.id)
                if not current or current.status != "ready":
                    continue
                self._prev_results[node.id] = current.result
                snapshot = self.graph.get_snapshot()

            # ── Backoff window check ──────────────────────────────────────────
            retry_after = node.metadata.get("retry_after", 0)
            if retry_after and time.time() < retry_after:
                remaining = retry_after - time.time()
                logger.debug(
                    "[EXEC] Node %s in backoff — %.1fs remaining",
                    node.id,
                    remaining,
                )
                continue

            # ── Dependency gap check ──────────────────────────────────────────
            attempts = node.metadata.get("gap_fill_attempts", 0)
            if self.quality_gate and attempts < self.max_gap_fill_attempts:
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
            future.add_done_callback(lambda fut, nid=node.id: self._on_node_done(nid, fut))
            launched += 1

        return launched

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
                        queue.extend(child.children)

                for desc_id in to_reset:
                    self._apply(Event(RESET_NODE, {"node_id": desc_id}))
                    logger.info("[ORCHESTRATOR] Reset dependent for re-execution: %s", desc_id)

    def _on_node_done(self, node_id: str, future):
        # Fix #1: _on_node_done runs in a ThreadPoolExecutor done-callback
        # thread.  self._running_futures, self.current_activity, and
        # self.activity_started are also read/written by _execution_pass on the
        # orchestrator main-loop thread.  Acquire graph_lock for the shared-
        # state mutations at the top of this method so the two threads cannot
        # race.
        with self.graph_lock:
            self._running_futures.pop(node_id, None)

            # FIX #4: the original check used `node_id in self.current_activity`
            # which is a substring test.  A short node_id like "task_1" would
            # wrongly match an activity string for "task_10" or "task_1_abc",
            # causing the activity indicator to be cleared for the wrong node.
            # Use an exact string comparison instead.
            if self.current_activity == f"Executing: {node_id}":
                if self._running_futures:
                    other = next(iter(self._running_futures))
                    self.current_activity = f"Executing: {other}"
                else:
                    self.current_activity = None
                    self.activity_started = None

        try:
            result = future.result()
        except LLMStoppedError as exc:
            # FIX: _DeferredLLM raises LLMStoppedError while the real LLM is
            # still loading (before attach() is called).  _DeferredLLM is not a
            # BaseLLM subclass so it is not in _llm_clients, which means
            # self.llm_stopped would return False — causing the node to be
            # permanently MARK_FAILED rather than reset and retried once the
            # real LLM becomes available.  Handle it here explicitly so these
            # transient interruptions are always treated as a pause, not a
            # failure.
            logger.info(
                "[EXEC] Node %s interrupted by LLM stop/loading (%s) — resetting to pending",
                node_id,
                exc,
            )
            reporter = self._reporters.pop(node_id, None)
            with self.graph_lock:
                if node_id in self.graph.nodes:
                    if reporter:
                        for step_id in list(reporter._all_step_ids):
                            if step_id in self.graph.nodes:
                                self.graph.detach_node(step_id)
                    self._apply(Event(RESET_NODE, {"node_id": node_id}))
            return
        except Exception as exc:
            logger.warning("[EXEC] Node %s raised: %s", node_id, exc)
            result = None

        # ── Hard failure ──────────────────────────────────────────────────────
        if result is None:
            reporter = self._reporters.pop(node_id, None)
            with self.graph_lock:
                if node_id not in self.graph.nodes:
                    return

                # If the executor returned None because the LLM was paused
                # mid-run, reset the node to pending so it picks up cleanly
                # when the user unpauses.  Execution step nodes are detached
                # so the next run starts with a fresh reporter.
                if self.llm_stopped:
                    logger.info(
                        "[EXEC] Node %s interrupted by LLM pause — resetting to pending",
                        node_id,
                    )
                    if reporter:
                        for step_id in list(reporter._all_step_ids):
                            if step_id in self.graph.nodes:
                                self.graph.detach_node(step_id)
                    self._apply(Event(RESET_NODE, {"node_id": node_id}))
                    return

                # Genuine failure (max turns, JSON parse error, tool error, …)
                if reporter:
                    reporter.expose_all()
                self._apply(Event(MARK_FAILED, {"node_id": node_id}))
                logger.warning("[EXEC] Failed: %s", node_id)
            return

        # ── Awaiting-user: executor completed its steps but some require user action
        if isinstance(result, dict) and result.get("_awaiting_user"):
            reporter = self._reporters.pop(node_id, None)
            handoff_artifact = result.get("handoff_artifact", "")
            pending_steps = result.get("pending_steps", [])
            partial_result = result.get("partial_result", "")

            # Write back broadened metadata before the early return — the normal
            # broadening write-back block below is never reached in this path.
            broadening_signal = reporter.pending_broadening if reporter else None

            with self.graph_lock:
                if node_id not in self.graph.nodes:
                    return
                if reporter:
                    reporter.expose_all()

                # Broadened metadata write-back (mirrors the done path below)
                if broadening_signal:
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

                # Store the partial result so the user can see what was produced
                if partial_result:
                    self._apply(Event(SET_RESULT, {"node_id": node_id, "result": partial_result}))
                self._apply(
                    Event(
                        MARK_AWAITING_USER,
                        {
                            "node_id": node_id,
                            "handoff_artifact": handoff_artifact,
                            "pending_steps": pending_steps,
                        },
                    )
                )
                logger.info(
                    "[EXEC] Node %s awaiting user for %d step(s): %s",
                    node_id,
                    len(pending_steps),
                    pending_steps,
                )
            return

        # ── Broadening metadata: write back if node ran with broadened description
        # Access _reporters under graph_lock: _on_node_done runs in a
        # ThreadPoolExecutor done-callback and concurrent callbacks could
        # otherwise race with the dict writes in _execution_pass.
        with self.graph_lock:
            reporter_for_broadening = self._reporters.get(node_id)
        # Capture the broadening signal before the if/else so it remains
        # accessible in the verified-OK block further below, where we need it
        # to decide what produced_output to write.
        _broadening_signal = (
            reporter_for_broadening.pending_broadening if reporter_for_broadening else None
        )
        if reporter_for_broadening and reporter_for_broadening.pending_broadening:
            signal = reporter_for_broadening.pending_broadening
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

        # ── Pre-flight: check file outputs ────────────────────────────────────
        satisfied: bool | None = None
        reason: str = ""
        expected_files: list = []
        tool_calls_made: set = set()

        reporter = self._reporters.get(node_id)

        if reporter and self.quality_gate:
            with self.graph_lock:
                declared_outputs = (
                    self.graph.nodes[node_id].metadata.get("output", [])
                    if node_id in self.graph.nodes
                    else []
                )
                tool_calls_made = {
                    self.graph.nodes[sid].metadata.get("tool_name")
                    for sid in reporter._all_step_ids
                    if sid in self.graph.nodes
                }

            expected_files = [
                o
                for o in declared_outputs
                if any(str(o).endswith(ext) for ext in self.quality_gate.FILE_EXTENSIONS)
            ]

            if expected_files and "write_file" not in tool_calls_made:
                file_path = expected_files[0]
                content = result

                match = re.search(
                    r"(?:summary|content)\s*:\s*(.+)",
                    result,
                    re.DOTALL | re.IGNORECASE,
                )
                if match:
                    content = match.group(1).strip()

                if content and len(content) > 50:
                    try:
                        tools = getattr(self.executor, "tools", None)
                        if tools:
                            tools.execute(
                                "write_file",
                                {
                                    "path": str(file_path),
                                    "content": content,
                                },
                            )
                            logger.info("[EXEC] Auto-wrote missing file output: %s", file_path)
                    except Exception as e:
                        logger.warning("[EXEC] Auto-write failed for %s: %s", file_path, e)

        # ── LLM verification ──────────────────────────────────────────────────
        if self.quality_gate:
            with self.graph_lock:
                if node_id not in self.graph.nodes:
                    return
                node = self.graph.nodes[node_id]
                snapshot = self.graph.get_snapshot()

            satisfied, reason = self._verify_result(node, result, snapshot)
        else:
            satisfied = True
            reason = ""

        # ── Consolidate state mutation in a single lock acquisition ──────────
        with self.graph_lock:
            if node_id not in self.graph.nodes:
                return
            node = self.graph.nodes[node_id]

            if not satisfied:
                retry = node.metadata.get("retry_count", 0)
                logger.warning(
                    "[EXEC] Node %s verification FAILED (attempt %d): %s",
                    node_id,
                    retry + 1,
                    reason,
                )

                if retry + 1 >= self.max_retries:
                    logger.error(
                        "[EXEC] Node %s exhausted %d retries — marking permanently failed",
                        node_id,
                        self.max_retries,
                    )
                    if reporter:
                        reporter.expose_all()
                    self._apply(Event(MARK_FAILED, {"node_id": node_id}))
                    self._reporters.pop(node_id, None)
                    return

                # ── Exponential backoff before retry ─────────────────────────
                backoff_secs = min(2**retry, _MAX_NODE_RETRY_BACKOFF_SECS)  # 1s, 2s, 4s…
                logger.info(
                    "[EXEC] Node %s will retry in %.0fs (attempt %d/%d)",
                    node_id,
                    backoff_secs,
                    retry + 1,
                    self.max_retries,
                )
                if reporter:
                    reporter.expose_all()
                self._apply(Event(MARK_FAILED, {"node_id": node_id}))
                self._apply(Event(RESET_NODE, {"node_id": node_id}))
                # FIX: Write retry metadata AFTER RESET_NODE, not before.
                # TaskNode.reset() now clears retry_count / retry_after /
                # verification_failure so that user-triggered resets from the
                # UI don't leave stale counts.  Writing the retry state here —
                # after reset() has already run — means the orchestrator's own
                # retry cycle still accumulates correctly while external resets
                # start fresh.
                n = self.graph.nodes.get(node_id)
                if n is not None:
                    n.metadata["verification_failure"] = reason
                    n.metadata["retry_count"] = retry + 1
                    n.metadata["retry_after"] = time.time() + backoff_secs
                # Fix #3: pop the reporter on every retry, not only on final
                # failure.  The next execution cycle creates a fresh reporter
                # anyway; keeping the stale one risks accumulating dangling
                # step-node references across retries.
                self._reporters.pop(node_id, None)
            else:
                logger.info("[EXEC] Node %s verified OK. Result: %.120s", node_id, result)
                self._apply(
                    Event(
                        MARK_DONE,
                        {
                            "node_id": node_id,
                            "result": result,
                        },
                    )
                )
                # Write produced_output: reflects what this node actually
                # delivered at runtime.  Downstream nodes read this (via
                # _resolve_inputs) instead of the declared output metadata,
                # so _select_goal_mode can correctly detect that an upstream
                # node ran broadened and produced different output names than
                # the original contract — which should trigger broadening
                # downstream too.
                _produced_output = (
                    _broadening_signal.broadened_output
                    if _broadening_signal and _broadening_signal.broadened_output
                    else node.metadata.get("output", [])
                )
                self._apply(
                    Event(
                        UPDATE_METADATA,
                        {
                            "node_id": node_id,
                            "metadata": {
                                "verified": True,
                                "produced_output": _produced_output,
                            },
                        },
                    )
                )
                if reporter:
                    reporter.hide_all()
                self._reporters.pop(node_id, None)

    # ── Verification ─────────────────────────────────────────────────────────

    def _verify_result(self, node, result: str, snapshot):
        if not self.quality_gate:
            return True, ""
        return self.quality_gate.verify_result(node, result, snapshot)

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

            # Find the upstream clarification node
            clar_node = None
            for dep_id in node.dependencies:
                dep = snapshot.get(dep_id)
                if dep and dep.node_type == "clarification" and dep.result:
                    clar_node = dep
                    break

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
            node = self.graph.nodes.get(task_node_id)
            if not node:
                return
            for dep_id in node.dependencies:
                dep = self.graph.nodes.get(dep_id)
                if dep and dep.node_type == "clarification":
                    clar_id = dep_id
                    break

        if not clar_id or clar_id not in self.graph.nodes:
            logger.warning(
                "[ORCHESTRATOR] Cannot patch clarification node for %s "
                "— no clarification node found upstream",
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
        self, node_id: str, signal: AwaitingInputSignal
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

    # ── Bridge node injection ─────────────────────────────────────────────────

    def _inject_bridge_node(self, bridge: dict, blocked_node_id: str):
        bridge_id = bridge.get("node_id")
        if not bridge_id:
            return

        with self.graph_lock:
            if bridge_id in self.graph.nodes:
                return

            self._apply(
                Event(
                    ADD_NODE,
                    {
                        "node_id": bridge_id,
                        "node_type": "task",
                        "dependencies": [],
                        "origin": "quality_gate",
                        "metadata": {
                            "description": bridge.get("description", bridge_id),
                            "output": [
                                {
                                    "name": bridge.get("output", "bridge_output"),
                                    "type": "document",
                                    "description": bridge.get("output", ""),
                                }
                            ],
                            "fully_refined": True,
                        },
                    },
                )
            )

            self._apply(
                Event(
                    ADD_DEPENDENCY,
                    {
                        "node_id": blocked_node_id,
                        "depends_on": bridge_id,
                    },
                )
            )

            current_node = self.graph.nodes.get(blocked_node_id)
            if current_node:
                attempts = current_node.metadata.get("gap_fill_attempts", 0)
                self._apply(
                    Event(
                        UPDATE_METADATA,
                        {
                            "node_id": blocked_node_id,
                            "metadata": {"gap_fill_attempts": attempts + 1},
                        },
                    )
                )

        logger.info("[ORCHESTRATOR] Injected bridge node %s for %s", bridge_id, blocked_node_id)

    # ── Startup verification ─────────────────────────────────────────────────

    def verify_restored_nodes(self):
        deadline = time.time() + self.verify_timeout
        with self.graph_lock:
            done_tasks = [
                n
                for n in self.graph.nodes.values()
                if n.node_type == "task" and n.status == "done" and n.result is not None
            ]

        # Pass 1: file-existence check for nodes with declared file outputs.
        for node in done_tasks:
            if not self.quality_gate:
                continue
            declared_outputs = node.metadata.get("output", [])
            missing_files = []
            for output in declared_outputs:
                if isinstance(output, dict):
                    is_file = output.get("type") == "file" or any(
                        output.get("name", "").endswith(ext)
                        for ext in self.quality_gate.FILE_EXTENSIONS
                    )
                    path = output.get("name", "")
                else:
                    is_file = any(
                        str(output).endswith(ext) for ext in self.quality_gate.FILE_EXTENSIONS
                    )
                    path = str(output)
                if is_file and path and not self.quality_gate._file_exists(path):
                    missing_files.append(path)

            if missing_files:
                logger.warning(
                    "[STARTUP] Node %s declared file output(s) %s do not exist on disk — resetting",
                    node.id,
                    missing_files,
                )
                with self.graph_lock:
                    n = self.graph.nodes.get(node.id)
                    if n is None:
                        continue
                    # FIX (issue 2): skip if any child is already running —
                    # the orchestrator loop may have dispatched a child between
                    # the snapshot above and this lock acquisition.  Resetting
                    # a parent whose child is in-flight produces an inconsistent
                    # graph state (running node with a pending parent).
                    if any(
                        self.graph.nodes.get(cid) is not None
                        and self.graph.nodes[cid].status == "running"
                        for cid in n.children
                    ):
                        logger.info(
                            "[STARTUP] Node %s skipped — child already running",
                            node.id,
                        )
                        continue
                    retry = n.metadata.get("retry_count", 0)
                    self._apply(Event(RESET_NODE, {"node_id": node.id}))
                    # FIX (issue 4): route retry metadata through _apply so the
                    # mutations are written to the event log (WAL).  Direct dict
                    # writes bypass apply_event and are lost on the next restart.
                    self._apply(
                        Event(
                            UPDATE_METADATA,
                            {
                                "node_id": node.id,
                                "metadata": {
                                    "verification_failure": (
                                        f"declared file output(s) {missing_files} do not exist on disk"
                                    ),
                                    "retry_count": retry + 1,
                                },
                            },
                        )
                    )

        # Pass 2: LLM verification for nodes never verified
        with self.graph_lock:
            candidates = [
                n
                for n in self.graph.nodes.values()
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
            if time.time() > deadline:
                logger.warning(
                    "[STARTUP] verify_restored_nodes hit %.0fs timeout — "
                    "%d node(s) left unverified",
                    self.verify_timeout,
                    len(candidates) - candidates.index(node),
                )
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
                live = self.graph.nodes[node.id]
                # FIX (issue 2): re-check for running children under the lock.
                # Between the LLM call and this lock acquisition the orchestrator
                # loop may have dispatched a child of this node.  Resetting the
                # parent while a child is running produces an inconsistent graph.
                if not satisfied and any(
                    self.graph.nodes.get(cid) is not None
                    and self.graph.nodes[cid].status == "running"
                    for cid in live.children
                ):
                    logger.warning(
                        "[STARTUP] Node %s failed verification but a child is "
                        "already running — deferring reset to avoid race",
                        node.id,
                    )
                    continue
                if not satisfied:
                    logger.warning(
                        "[STARTUP] Restored node %s failed verification: %s — resetting",
                        node.id,
                        reason,
                    )
                    retry = live.metadata.get("retry_count", 0)
                    self._apply(Event(RESET_NODE, {"node_id": node.id}))
                    # FIX (issue 4): persist via UPDATE_METADATA so these values
                    # survive the next restart (direct dict writes are not WAL-logged).
                    self._apply(
                        Event(
                            UPDATE_METADATA,
                            {
                                "node_id": node.id,
                                "metadata": {
                                    "verification_failure": reason,
                                    "retry_count": retry + 1,
                                },
                            },
                        )
                    )
                else:
                    logger.info("[STARTUP] Restored node %s verified OK", node.id)
                    self._apply(
                        Event(
                            UPDATE_METADATA,
                            {
                                "node_id": node.id,
                                "metadata": {"verified": True},
                            },
                        )
                    )

        with self.graph_lock:
            self.graph.recompute_readiness()

        ready = sum(1 for n in self.graph.nodes.values() if n.status == "ready")
        logger.info("[STARTUP] Post-verification readiness: %d node(s) ready", ready)

    # ── User-facing edit API ─────────────────────────────────────────────────

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

    def add_task(
        self,
        node_id: str,
        dependencies: list = None,
        description: str = "",
        metadata: dict = None,
    ):
        meta = {"description": description, "fully_refined": True}
        if metadata:
            meta.update(metadata)
        with self.graph_lock:
            self._apply(
                Event(
                    ADD_NODE,
                    {
                        "node_id": node_id,
                        "node_type": "task",
                        "dependencies": dependencies or [],
                        "origin": "user",
                        "metadata": meta,
                    },
                )
            )

    def remove_node(self, node_id: str):
        with self.graph_lock:
            if node_id in self._running_futures:
                logger.warning("[USER] Cannot remove %s — currently running", node_id)
                return
            self._apply(Event(REMOVE_NODE, {"node_id": node_id}))

    def add_dependency(self, node_id: str, depends_on: str):
        with self.graph_lock:
            self._apply(
                Event(
                    ADD_DEPENDENCY,
                    {
                        "node_id": node_id,
                        "depends_on": depends_on,
                    },
                )
            )

    def remove_dependency(self, node_id: str, depends_on: str):
        with self.graph_lock:
            self._apply(
                Event(
                    REMOVE_DEPENDENCY,
                    {
                        "node_id": node_id,
                        "depends_on": depends_on,
                    },
                )
            )

    def retry_node(self, node_id: str):
        with self.graph_lock:
            node = self.graph.nodes.get(node_id)
            if not node or node_id in self._running_futures:
                return
            self._apply(Event(RESET_NODE, {"node_id": node_id}))

    def resume_node(self, node_id: str) -> bool:
        """
        Explicitly resume an awaiting_input node from the UI.

        Emits RESUME_NODE which transitions the node to pending and clears
        all awaiting_input metadata.  Returns True if the node was in
        awaiting_input status (and was resumed), False otherwise.
        """
        with self.graph_lock:
            node = self.graph.nodes.get(node_id)
            if not node:
                return False
            if node.status != "awaiting_input":
                logger.warning(
                    "[ORCHESTRATOR] resume_node called on %s "
                    "which is not awaiting_input (status=%s)",
                    node_id,
                    node.status,
                )
                return False
            self._apply(Event(RESUME_NODE, {"node_id": node_id}))
            logger.info("[ORCHESTRATOR] Node %s manually resumed by user", node_id)
            return True

    def confirm_node(self, node_id: str) -> bool:
        """
        Confirm that the user has completed the real-world steps for an
        awaiting_user node.

        Emits CONFIRM_USER_DONE which transitions the node from awaiting_user
        to done, unblocking all downstream dependents.  Returns True if the
        node was in awaiting_user status, False otherwise.
        """
        with self.graph_lock:
            node = self.graph.nodes.get(node_id)
            if not node:
                return False
            if node.status != "awaiting_user":
                logger.warning(
                    "[ORCHESTRATOR] confirm_node called on %s "
                    "which is not awaiting_user (status=%s)",
                    node_id,
                    node.status,
                )
                return False
            self._apply(Event(CONFIRM_USER_DONE, {"node_id": node_id}))
            # Recompute readiness so downstream nodes become ready immediately.
            self.graph.recompute_readiness_for(node_id)
            logger.info("[ORCHESTRATOR] Node %s confirmed done by user", node_id)
            return True

    def replan_goal(self, goal_id: str):
        with self.graph_lock:
            goal = self.graph.nodes.get(goal_id)
            if not goal or goal.node_type != "goal":
                return

            # FIX #7: check whether any children are currently running before
            # removing idle children and clearing the expanded flag.
            running_children = [cid for cid in goal.children if cid in self._running_futures]
            if running_children:
                logger.warning(
                    "[ORCHESTRATOR] replan_goal(%s): %d child(ren) still running "
                    "(%s) — removing idle children but deferring expanded=False "
                    "to avoid a mid-execution re-plan.",
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
                self._apply(
                    Event(
                        UPDATE_METADATA,
                        {
                            "node_id": goal_id,
                            "metadata": {"expanded": False},
                        },
                    )
                )

    def update_metadata(self, node_id: str, metadata: dict):
        with self.graph_lock:
            self._apply(
                Event(
                    UPDATE_METADATA,
                    {
                        "node_id": node_id,
                        "metadata": metadata,
                    },
                )
            )

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
            "total": len(nodes),
            "by_status": counts,
            "running_nodes": list(self._running_futures.keys()),
        }

    # ── Internals ────────────────────────────────────────────────────────────

    def _apply(self, event: Event):
        """Apply one event. Must be called with graph_lock held."""
        apply_event(self.graph, event, event_log=self.event_log)

    def _is_fully_done(self) -> bool:
        with self.graph_lock:
            return all(n.status in ("done", "failed") for n in self.graph.nodes.values())
