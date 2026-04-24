from __future__ import annotations

import re
import threading
import time
from collections import deque, namedtuple
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from toddly.core.events import (
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
    SET_RESULT,
    UPDATE_METADATA,
    Event,
)
from toddly.core.reducer import apply_event
from toddly.engine.execution_step_reporter import ExecutionStepReporter
from toddly.infra.event_queue import EventQueue, StatusEvent
from toddly.infra.logging import get_logger
from toddly.planning.llm_interface import (
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


class BaseOrchestrator:
    """
    Generic event-sourced orchestrator: LLM plans, executor runs.

    Owns the core loop, threading, retry/backoff, event sourcing, and LLM
    pause/resume.  All domain-specific behaviour is delegated to hook methods
    that subclasses override.

    Hook methods (override in subclasses):
        _get_plannable_nodes()        — which nodes trigger a planning call
        _is_executable_node(node)     — whether a ready node should run
        _pre_planning_hooks()         — extra passes run before _planning_pass
                                        (e.g. node-type promotion)
        _post_planning_hooks()        — extra passes run after _planning_pass
                                        (e.g. goal auto-completion, resumption)
        _handle_broadening()          — domain write-backs before verification;
                                        returns produced_output override or None
        _on_awaiting_user_complete()  — domain write-backs inside the awaiting-
                                        user branch (called with graph_lock held)

    The curses/web UI expects:
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
        # Sentinel used by _prev_results to distinguish "key absent" from
        # "key present but value is None".  node.result is None after reset(),
        # so using None as the pop() default would make every post-retry
        # completion appear unchanged and silently suppress downstream
        # invalidation.
        self._UNSET = object()
        # Tracks node IDs that have been explicitly reset at least once.
        # Used to distinguish a genuine manual-retry (node was reset then
        # completed with a different result → invalidate downstream) from a
        # brand-new first execution (node result was also None before running,
        # but no downstream nodes depended on a prior result → no invalidation).
        self._reset_node_ids: set[str] = set()

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
                self._pre_planning_hooks()
                planned = self._planning_pass()
                self._post_planning_hooks()
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
            unexpanded = self._get_plannable_nodes()

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

    # ── Execution pass ───────────────────────────────────────────────────────

    def _execution_pass(self) -> int:
        launched = 0

        with self.graph_lock:
            ready = [
                n
                for n in self.graph.nodes.values()
                if n.status == "ready"
                and self._is_executable_node(n)
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
                # Store the previous result; use _UNSET only when the
                # dict key is absent (pop default below).  Storing None here
                # is intentional — it records that the node was tracked and
                # its result was None (i.e. it was freshly reset).
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

    # ── Node done callback ────────────────────────────────────────────────────

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

            with self.graph_lock:
                if node_id not in self.graph.nodes:
                    return
                if reporter:
                    reporter.expose_all()

                # Domain-specific write-backs (e.g. broadening metadata).
                # Called with graph_lock held, after reporter.expose_all(),
                # before SET_RESULT / MARK_AWAITING_USER.
                self._on_awaiting_user_complete(node_id, result, reporter)

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

        # ── Domain-specific write-backs before verification (e.g. broadening).
        # Access _reporters under graph_lock: _on_node_done runs in a
        # ThreadPoolExecutor done-callback and concurrent callbacks could
        # otherwise race with the dict writes in _execution_pass.
        # _handle_broadening acquires graph_lock itself internally as needed.
        with self.graph_lock:
            _reporter_for_hooks = self._reporters.get(node_id)
        produced_output_override = self._handle_broadening(node_id, _reporter_for_hooks)

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

                # ── Fix 5: escalate to upstream reset when the failure reason
                # implicates input quality rather than this node's own logic.
                # Keywords like "fabricated" or "no results" indicate the node
                # was working with bad upstream data; re-running it alone won't
                # help — the suspect upstream must be reset first.
                _INPUT_QUALITY_SIGNALS = frozenset(
                    (
                        "fabricat",
                        "invent",
                        "no results",
                        "no real data",
                        "hallucin",
                        "prior knowledge",
                        "training data",
                        "made up",
                    )
                )
                reason_lower = reason.lower()
                if any(sig in reason_lower for sig in _INPUT_QUALITY_SIGNALS):
                    suspect_upstreams = [
                        dep_id
                        for dep_id in node.dependencies
                        if dep_id in self.graph.nodes
                        and self.graph.nodes[dep_id].node_type == "task"
                        and self.graph.nodes[dep_id].metadata.get("verification_failure")
                    ]
                    if suspect_upstreams:
                        for dep_id in suspect_upstreams:
                            logger.warning(
                                "[EXEC] Node %s failure implicates upstream input quality "
                                "— resetting suspect upstream %s before retry",
                                node_id,
                                dep_id,
                            )
                            self._apply(Event(RESET_NODE, {"node_id": dep_id}))

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
                # produced_output_override is set by _handle_broadening() above
                # when the node ran with a broadened description; None means
                # use the node's statically declared output.
                _produced_output = (
                    produced_output_override
                    if produced_output_override is not None
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

                # ── Downstream invalidation on result change ──────────────────
                # _prev_results is populated just before a node is launched
                # (in _execution_pass).  If the result is different from the
                # previous run — i.e. this was a retry that produced genuinely
                # new output — reset every downstream node that is already
                # "done" so it re-executes with the updated input.  Nodes that
                # are already pending/running/failed are left alone because they
                # will pick up the fresh result naturally when they run.
                prev = self._prev_results.pop(node_id, self._UNSET)
                # Consume the reset-tracking flag for this node now that it has
                # completed successfully.  Do this regardless of whether
                # invalidation fires so the set doesn't accumulate stale entries.
                was_explicitly_reset = node_id in self._reset_node_ids
                self._reset_node_ids.discard(node_id)
                # Trigger invalidation when ALL of the following are true:
                #   1. The node was tracked (prev is not _UNSET).
                #   2. The result actually changed from the last known value.
                #   3. The node either had a real previous result (prev is not
                #      None) OR was explicitly reset before this run.
                #
                # Condition 3 prevents spurious invalidation on a brand-new
                # first execution: prev=None simply means the node never ran
                # before, so no downstream node could have consumed a previous
                # result.  When prev=None AND the node was explicitly reset,
                # it means a prior result existed, was cleared by the reset,
                # and downstream nodes may have been completed on the old value
                # — invalidation is correct in that case.
                if (
                    prev is not self._UNSET
                    and (prev is not None or was_explicitly_reset)
                    and result != prev
                ):
                    logger.info(
                        "[EXEC] Node %s result changed after retry — "
                        "invalidating downstream done nodes",
                        node_id,
                    )
                    # Walk the full downstream subgraph (not just direct
                    # children) so multi-hop dependents are also invalidated.
                    to_invalidate: list[str] = list(node.children)
                    visited: set[str] = set()
                    while to_invalidate:
                        cid = to_invalidate.pop()
                        if cid in visited or cid not in self.graph.nodes:
                            continue
                        visited.add(cid)
                        child = self.graph.nodes[cid]
                        if child.status == "done":
                            logger.info(
                                "[EXEC] Resetting downstream node %s (result of %s changed)",
                                cid,
                                node_id,
                            )
                            self._apply(Event(RESET_NODE, {"node_id": cid}))
                            # Continue walking: this child's own children may
                            # also need invalidation.
                            to_invalidate.extend(child.children)

    # ── Verification ─────────────────────────────────────────────────────────

    def _verify_result(self, node, result: str, snapshot):
        if not self.quality_gate:
            return True, ""
        return self.quality_gate.verify_result(node, result, snapshot)

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
            # FIX: node.reset() clears retry_count and all retry metadata, so
            # the next execution builds a prompt identical to the original run,
            # causing the LLM client to return a cached result instead of
            # re-running the model.  We stamp a _retry_nonce timestamp that
            # reset() does NOT clear; _build_prompt() appends it to the prompt
            # as a comment, making the cache key unique for every manual retry.
            #
            # A plain counter is NOT used here because the counter resets to 0
            # when the session restarts (the nonce is not persisted via an
            # event), so retry #1 in a new session produces nonce=1 — which
            # collides with the nonce=1 cache entry written in the previous
            # session.  A wall-clock timestamp is unique per call regardless
            # of how many times the session has been restarted.
            node = self.graph.nodes.get(node_id)
            if node is not None:
                node.metadata["_retry_nonce"] = time.time()

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
        # Track every node that is explicitly reset so the downstream
        # invalidation logic in _on_node_done can distinguish a genuine
        # manual/auto retry (node was reset → ran again → different result)
        # from a first-ever execution (node.result was also None but nothing
        # downstream had yet consumed a previous result).
        if event.type == RESET_NODE:
            node_id = event.payload.get("node_id")
            if node_id:
                self._reset_node_ids.add(node_id)
        apply_event(self.graph, event, event_log=self.event_log)

    def _is_fully_done(self) -> bool:
        with self.graph_lock:
            return all(n.status in ("done", "failed") for n in self.graph.nodes.values())

    # ── Hook methods ─────────────────────────────────────────────────────────

    def _get_plannable_nodes(self) -> list:
        """
        Return the nodes that should be passed to the planner this iteration.

        Called with graph_lock held.  The default implementation selects
        'goal' nodes that have not yet been expanded.  Override to use a
        different node type or expansion flag.
        """
        return [
            n
            for n in self.graph.nodes.values()
            if n.node_type == "goal" and not n.metadata.get("expanded", False)
        ]

    def _is_executable_node(self, node) -> bool:
        """
        Return True if a ready node should be dispatched to the executor.

        The default implementation skips 'execution_step' nodes, which are
        managed internally by ExecutionStepReporter and should never be
        independently executed.  Override to exclude additional domain-specific
        node types (e.g. 'clarification').
        """
        return node.node_type not in ("execution_step",)

    def _pre_planning_hooks(self) -> None:
        """
        Called each loop iteration after _drain_event_queue and before
        _planning_pass.

        Override to run domain-specific passes that must happen before
        planning — for example, promoting nodes from a staging status into
        a plannable type so they are picked up by _get_plannable_nodes in
        the same iteration.
        """
        pass

    def _post_planning_hooks(self) -> None:
        """
        Called each loop iteration after _planning_pass and before
        _execution_pass.

        Override to run domain-specific passes that should follow planning —
        for example, auto-completing finished goal nodes or resuming nodes
        that were blocked on user input.
        """
        pass

    def _handle_broadening(self, node_id: str, reporter) -> "list | None":
        """
        Called after the awaiting-user check but before the quality gate,
        with graph_lock NOT held.

        May write domain-specific metadata to the graph (acquiring graph_lock
        internally) and return a list to override the produced_output written
        to the node on success, or None to use the node's statically declared
        output metadata.

        The default implementation is a no-op that returns None.
        """
        return None

    def _on_awaiting_user_complete(self, node_id: str, result: dict, reporter) -> None:
        """
        Called with graph_lock held, inside the awaiting-user branch, after
        reporter.expose_all() and before SET_RESULT / MARK_AWAITING_USER.

        Override to write domain-specific metadata to the graph before the
        node transitions to awaiting_user status — for example, persisting
        broadening signals that were captured during execution.

        Do NOT acquire graph_lock inside this method; it is already held by
        the caller (graph_lock is an RLock so re-entry would succeed, but it
        is unnecessary and misleading).
        """
        pass
