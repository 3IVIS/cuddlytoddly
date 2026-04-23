# --- FILE: toddly/engine/execution_step_reporter.py ---

# engine/execution_step_reporter.py

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

    def __init__(self, parent_node_id: str, apply_fn, graph_lock, graph):
        self.parent_node_id = parent_node_id
        self._apply = apply_fn
        self._graph_lock = graph_lock
        self._graph = graph

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

    def on_tool_start(self, tool_name: str, tool_args: dict) -> str:
        # FIX #9: A tool may legitimately be called more than once per execution
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
    ):
        """
        Called after a tool returns.

        Appends this attempt to the node's history and marks done/failed.
        Truncates the result so it doesn't blow up metadata storage.
        """

        attempt = {
            "turn": self._turn,
            "args": tool_args,
            "result": result,
            "status": "error" if error else "ok",
            "timestamp": datetime.now(timezone.utc).isoformat(),
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

    # ── Post-execution visibility ─────────────────────────────────────────────

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
