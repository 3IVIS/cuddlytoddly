# engine/execution_step_reporter.py

import time as _time
from datetime import datetime, timezone

from toddly.core.events import (
    ADD_DEPENDENCY,
    ADD_NODE,
    MARK_DONE,
    MARK_FAILED,
    MARK_RUNNING,
    REMOVE_DEPENDENCY,
    UPDATE_METADATA,
    Event,
)
from toddly.infra.logging import get_logger

logger = get_logger(__name__)


class ExecutionStepReporter:
    """
    Tracks every step of an LLMExecutor run as child nodes in the DAG.

    One node per unique tool call (stable across retries), plus one
    synthesis node for the final done=True turn.

    Lifecycle:
      - on_llm_turn()    called at the start of each executor loop iteration
      - on_tool_start()  called when the LLM requests a tool
      - on_tool_done()   called after the tool returns
      - on_synthesis()   called when the LLM sets done=True
      - on_llm_error()   called when the LLM or JSON parsing fails
      - hide_all()       called after parent succeeds — hides steps from main UI
      - expose_all()     called after parent fails — ensures steps are visible
    """

    def __init__(self, parent_node_id: str, apply_fn, graph_lock, graph, activity_setter=None):
        self.parent_node_id = parent_node_id
        self._apply = apply_fn
        self._graph_lock = graph_lock
        self._graph = graph

        # Optional callback used to push richer activity strings to the
        # orchestrator's current_activity field from inside executor threads.
        # Signature: (text: str) -> None.  May be None (e.g. in tests).
        self._activity_setter = activity_setter

        # tool_name -> list[node_id]  (list because the same tool may be called
        # multiple times in a single execution, e.g. write_file for two files).
        # The last entry in the list is the "active" node for retry detection.
        self._tool_nodes: dict[str, list[str]] = {}
        # ordered list of all step node ids (for hide/expose)
        self._all_step_ids: list[str] = []
        self._turn = 0
        # Set by execute() when the node ran with a broadened description.
        # Read by the orchestrator in _on_node_done to write metadata back.
        self.pending_broadening = None  # AwaitingInputSignal | None

    def on_broadened_execution(self, signal) -> None:
        """
        Called by execute() when the node is running with a broadened description
        due to missing inputs.  Stores the signal so the orchestrator can read it
        after execution completes and write broadened_description +
        broadened_for_missing into node metadata.
        """
        self.pending_broadening = signal

    # ── Turn lifecycle ────────────────────────────────────────────────────────

    def on_llm_turn(self, turn: int):
        """Called at the top of each executor loop iteration."""
        self._turn = turn
        # Immediately stamp the activity with the turn number so the status
        # panel creates a new row for every turn — including correction turns
        # that have no tool call and would otherwise be invisible.
        if self._activity_setter:
            self._activity_setter(
                f"Executing: {self.parent_node_id} · turn {turn + 1} · generating…"
            )

    def on_tool_start(self, tool_name: str, tool_args: dict) -> str:
        # A tool may legitimately be called more than once per execution
        # (e.g. write_file called for two different output files).  The old code
        # keyed the registry on tool_name alone, so the second call reused the
        # first call's step node and its description was never updated.  We now
        # key on (tool_name, call_index) by appending a numeric suffix, and only
        # reuse an existing node when the task is being *retried* from scratch
        # (fresh _tool_nodes map but the node already exists in the graph from a
        # previous attempt of the same parent task).

        # Build a candidate step_id using call-count to make it unique
        call_index = len(self._tool_nodes.get(tool_name, []))
        step_id = (
            f"{self.parent_node_id}__step_{tool_name}"
            if call_index == 0
            else f"{self.parent_node_id}__step_{tool_name}_{call_index}"
        )

        # Check if this exact step already exists in the graph (session retry)
        existing = self._graph.nodes.get(step_id)
        if existing is not None:
            logger.debug(
                "[STEPREPORTER] Reusing pre-existing step node %s (session retry)",
                step_id,
            )
            self._tool_nodes.setdefault(tool_name, []).append(step_id)
            if step_id not in self._all_step_ids:
                self._all_step_ids.append(step_id)
            with self._graph_lock:
                self._apply(Event(MARK_RUNNING, {"node_id": step_id}))
            return step_id

        # Fresh node — create as normal
        dep_list = [self._all_step_ids[-1]] if self._all_step_ids else []

        with self._graph_lock:
            self._apply(
                Event(
                    ADD_NODE,
                    {
                        "node_id": step_id,
                        "node_type": "execution_step",
                        "dependencies": dep_list,
                        "origin": "executor",
                        "metadata": {
                            "description": f"{tool_name}({self._format_args(tool_args)})",
                            "step_type": "tool_call",
                            "tool_name": tool_name,
                            "tool_args": tool_args,
                            "attempts": [],
                            "fully_refined": True,
                            "hidden": False,
                        },
                    },
                )
            )
            self._apply(Event(MARK_RUNNING, {"node_id": step_id}))

            # ── Swap parent's dependency to always point at the latest step ───
            if self._all_step_ids:
                # Remove previous frontier
                self._apply(
                    Event(
                        REMOVE_DEPENDENCY,
                        {
                            "node_id": self.parent_node_id,
                            "depends_on": self._all_step_ids[-1],
                        },
                    )
                )
            # Add new frontier
            self._apply(
                Event(
                    ADD_DEPENDENCY,
                    {
                        "node_id": self.parent_node_id,
                        "depends_on": step_id,
                    },
                )
            )

        self._tool_nodes.setdefault(tool_name, []).append(step_id)
        self._all_step_ids.append(step_id)

        # Update _live_status immediately so the status panel shows which tool
        # is currently running — not just after it returns.  Without this the
        # panel is silent for the entire duration of a tool call (potentially
        # minutes when web_search is retrying with exponential backoff).
        # We clear any stale preview/streaming from the previous turn so the
        # JS rendering path knows we're mid-call rather than showing a result.
        with self._graph_lock:
            node = self._graph.nodes.get(self.parent_node_id)
            if node and node.status == "running":
                ls = node.metadata.get("_live_status") or {}
                ls["tool"] = tool_name
                ls["args"] = tool_args
                ls.pop("preview", None)  # not yet available
                ls.pop("streaming", None)  # clear any stale LLM stream
                node.metadata["_live_status"] = ls
                self._graph.execution_version += 1

        # Update the activity header outside the lock — GIL-safe attribute write.
        if self._activity_setter:
            self._activity_setter(
                f"Executing: {self.parent_node_id} · turn {self._turn + 1} · {tool_name}…"
            )

        return step_id

    def on_synthesis(self, result: str):
        step_id = f"{self.parent_node_id}__step_synthesis"
        last_dep = self._all_step_ids[-1] if self._all_step_ids else self.parent_node_id

        with self._graph_lock:
            self._apply(
                Event(
                    ADD_NODE,
                    {
                        "node_id": step_id,
                        "node_type": "execution_step",
                        "dependencies": [last_dep],
                        "origin": "executor",
                        "metadata": {
                            "description": "synthesize result",
                            "step_type": "synthesis",
                            "fully_refined": True,
                            "hidden": False,
                        },
                    },
                )
            )
            self._apply(Event(MARK_DONE, {"node_id": step_id, "result": result}))

            # ── Swap parent to depend on synthesis as the final frontier ──────
            if self._all_step_ids:
                self._apply(
                    Event(
                        REMOVE_DEPENDENCY,
                        {
                            "node_id": self.parent_node_id,
                            "depends_on": self._all_step_ids[-1],
                        },
                    )
                )
            self._apply(
                Event(
                    ADD_DEPENDENCY,
                    {
                        "node_id": self.parent_node_id,
                        "depends_on": step_id,
                    },
                )
            )

        self._all_step_ids.append(step_id)

    def on_tool_done(
        self,
        step_id: str,
        tool_name: str,
        tool_args: dict,
        result: str,
        error: bool = False,
        *,
        duration_ms: float = 0.0,
    ):
        """
        Called after a tool returns.

        Appends this attempt to the node's history and marks done/failed.
        Truncates the result so it doesn't blow up metadata storage.

        Parameters
        ----------
        duration_ms : Wall-clock milliseconds from tool dispatch to return.
                      Stored in the attempt record so the UI can render a
                      duration badge and proportional bar in the timeline.
        """

        attempt = {
            "turn": self._turn,
            "args": tool_args,
            "result": result,
            "status": "error" if error else "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "duration_ms": round(duration_ms, 1),
        }

        with self._graph_lock:
            # Fetch current attempts and append
            node = self._get_live_node(step_id)
            existing_attempts = node.metadata.get("attempts", []) if node else []
            updated_attempts = existing_attempts + [attempt]

            self._apply(
                Event(
                    UPDATE_METADATA,
                    {
                        "node_id": step_id,
                        "metadata": {"attempts": updated_attempts},
                    },
                )
            )

            if error:
                self._apply(Event(MARK_FAILED, {"node_id": step_id}))
            else:
                self._apply(
                    Event(
                        MARK_DONE,
                        {
                            "node_id": step_id,
                            "result": result,
                        },
                    )
                )

    def on_llm_error(self, turn: int, error: str):
        step_id = f"{self.parent_node_id}__step_error_t{turn}"
        dep_list = [self._all_step_ids[-1]] if self._all_step_ids else []

        with self._graph_lock:
            self._apply(
                Event(
                    ADD_NODE,
                    {
                        "node_id": step_id,
                        "node_type": "execution_step",
                        "dependencies": dep_list,
                        "origin": "executor",
                        "metadata": {
                            "description": f"LLM error: {error[:120]}",
                            "step_type": "llm_error",
                            "fully_refined": True,
                            "hidden": False,
                        },
                    },
                )
            )
            self._apply(Event(MARK_FAILED, {"node_id": step_id}))

            if self._all_step_ids:
                self._apply(
                    Event(
                        REMOVE_DEPENDENCY,
                        {
                            "node_id": self.parent_node_id,
                            "depends_on": self._all_step_ids[-1],
                        },
                    )
                )
            self._apply(
                Event(
                    ADD_DEPENDENCY,
                    {
                        "node_id": self.parent_node_id,
                        "depends_on": step_id,
                    },
                )
            )

        self._all_step_ids.append(step_id)

    # ── Execution mode tracking ──────────────────────────────────────────────

    def on_execution_mode(self, mode: str) -> None:
        """
        Called by LLMExecutor once the execution mode is decided — either
        ``'original'`` (all required inputs present, or upstream outputs
        matched the original contract) or ``'broadened'`` (running with
        a generalised goal because some inputs are missing).

        Writes ``_active_tab`` to the parent node's metadata so the web UI
        can highlight the correct tab with a "▶ Running" badge and remove
        any ambiguity about which plan is actually being executed.
        """
        with self._graph_lock:
            self._apply(
                Event(
                    UPDATE_METADATA,
                    {
                        "node_id": self.parent_node_id,
                        "origin": "executor",
                        "metadata": {"_active_tab": mode},
                    },
                )
            )

    # ── Live progress ─────────────────────────────────────────────────────────

    def on_progress(
        self, turn: int, tool_name: str, tool_result: str, tool_args: dict | None = None
    ) -> None:
        """Write live execution status to the parent node's metadata.

        Uses direct graph mutation instead of _apply() so this transient state
        is never persisted to events.jsonl / the WAL.  On replay the node is
        reset to pending anyway, so there is no loss of correctness.

        Bumps graph.execution_version so the WebSocket pushes the update to
        the browser within its next 250 ms poll cycle.
        """
        with self._graph_lock:
            node = self._graph.nodes.get(self.parent_node_id)
            if node and node.status == "running":
                node.metadata["_live_status"] = {
                    "turn": turn + 1,
                    "tool": tool_name,
                    "preview": tool_result[:200],
                    "args": tool_args or {},
                }
                self._graph.execution_version += 1

        # Update the orchestrator's current_activity string so the status
        # panel header shows the current tool and turn number.  Called outside
        # graph_lock — simple attribute write is GIL-safe in CPython.
        if self._activity_setter:
            self._activity_setter(
                f"Executing: {self.parent_node_id} · {tool_name} (turn {turn + 1})"
            )

    # ── Post-execution visibility ─────────────────────────────────────────────

    def make_token_cb(self):
        """Return a per-call on_token callback for streaming display.

        Immediately initialises ``_live_status`` on the parent node so the
        status-panel body becomes visible as soon as the LLM call starts —
        even before any tokens arrive (e.g. for constrained local inference
        where only heartbeats fire, the panel body is still shown so the user
        sees the tool/turn context from the previous tool call).

        Incoming token chunks are accumulated in memory and throttle-flushed
        to ``_live_status["streaming"]`` at most every 200 ms (≈ 5 Hz).
        The callback is safe to call from any thread.
        """
        # Pre-initialize _live_status so the liveBlock filter
        # ``n.metadata?._live_status`` evaluates truthy in the UI immediately,
        # without waiting for the first token flush (which may never come for
        # constrained / heartbeat-only inference paths).
        with self._graph_lock:
            node = self._graph.nodes.get(self.parent_node_id)
            if node and node.status == "running":
                if not node.metadata.get("_live_status"):
                    node.metadata["_live_status"] = {}
                self._graph.execution_version += 1

        _buf: list[str] = []
        _last: list[float] = [_time.monotonic()]
        _INTERVAL = 0.2  # seconds between metadata writes

        def _on_token(chunk: str) -> None:
            _buf.append(chunk)
            now = _time.monotonic()
            if now - _last[0] < _INTERVAL:
                return
            _last[0] = now
            accumulated = "".join(_buf)
            with self._graph_lock:
                node = self._graph.nodes.get(self.parent_node_id)
                if node and node.status == "running":
                    ls = node.metadata.get("_live_status") or {}
                    ls["streaming"] = accumulated
                    node.metadata["_live_status"] = ls
                    self._graph.execution_version += 1

        return _on_token

    def make_heartbeat_cb(self):
        """Return an on_heartbeat callback that updates current_activity with elapsed time.

        Fires every 2 s from the LLM backend's watchdog thread during inference,
        giving the status panel header live elapsed-time feedback even when no
        real tokens are being emitted (e.g. constrained / outlines generation).
        Only does anything when activity_setter was wired up (suggestion A).
        """

        def _on_heartbeat(elapsed: float) -> None:
            if self._activity_setter:
                self._activity_setter(
                    f"Executing: {self.parent_node_id}"
                    f" · turn {self._turn + 1} · generating… {int(elapsed)}s"
                )

        return _on_heartbeat

    def clear_streaming(self) -> None:
        """Remove the streaming buffer from _live_status after an LLM turn.

        Called in the executor's finally block so the streaming preview is
        always cleared whether the turn succeeds, fails, or is interrupted.
        """
        with self._graph_lock:
            node = self._graph.nodes.get(self.parent_node_id)
            if node:
                ls = node.metadata.get("_live_status")
                if ls and "streaming" in ls:
                    del ls["streaming"]
                    self._graph.execution_version += 1

    def _clear_live_status(self) -> None:
        """Remove _live_status from the parent node.

        Must be called with graph_lock already held (used by hide_all / expose_all).
        """
        node = self._graph.nodes.get(self.parent_node_id)
        if node and "_live_status" in node.metadata:
            del node.metadata["_live_status"]
            self._graph.execution_version += 1

    def hide_all(self):
        with self._graph_lock:
            for step_id in self._all_step_ids:
                self._apply(
                    Event(
                        UPDATE_METADATA,
                        {
                            "node_id": step_id,
                            "metadata": {"hidden": True},
                        },
                    )
                )
            # Clear live status now that the node has completed successfully.
            self._clear_live_status()

    def expose_all(self):
        with self._graph_lock:
            for step_id in self._all_step_ids:
                self._apply(
                    Event(
                        UPDATE_METADATA,
                        {
                            "node_id": step_id,
                            "metadata": {"hidden": False},
                        },
                    )
                )
            # Clear live status when the node is being exposed after failure.
            self._clear_live_status()

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _format_args(self, args: dict) -> str:
        parts = []
        for k, v in args.items():
            v_str = str(v)[:30]
            parts.append(f"{k}={v_str}")
        return ", ".join(parts)

    def _get_live_node(self, node_id: str):
        """Must be called with graph_lock held."""
        return self._graph.nodes.get(node_id)
