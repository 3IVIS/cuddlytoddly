# ui/ui_config.py
"""
UIConfig — injectable configuration for curses_ui and web_server.

Split between a generic dataclass and a project-specific factory so that the
UI layer can be reused without any cuddlytoddly-domain imports.

Usage
-----
In __main__.py (or wherever the UI is started):

    from cuddlytoddly.ui.ui_config import make_cuddlytoddly_config
    config = make_cuddlytoddly_config()
    run_ui(orchestrator, run_dir=run_dir, config=config)
    # or
    run_web_ui(orchestrator, run_dir=run_dir, config=config)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

# ── Dataclass ─────────────────────────────────────────────────────────────────


@dataclass
class UIConfig:
    """
    Dependency-injection container for all domain-specific UI behaviour.

    Every field defaults to None / a conservative generic value so that a
    bare ``UIConfig()`` (or ``None`` passed where a config is expected) works
    without any domain knowledge.

    Shared between curses UI and web UI
    ------------------------------------
    snapshot_filter_fn
        ``(node_object) -> bool`` — True = include the node in the serialised
        snapshot sent to the client and in the git-projection graph.
        Defaults to including all nodes (no filter).

    valid_status_values
        Status values accepted by the edit-node modal (curses) and by the
        ``PUT /api/node/{id}`` status field (web).
        Defaults to the four generic lifecycle statuses.

    Curses UI
    ---------
    node_symbol_fn
        ``(node_object) -> str | None`` — override the git-graph ``*`` commit
        symbol for this node.  Return ``None`` to keep the default ``*``.

    node_detail_lines_fn
        ``(node_object) -> list[str] | None`` — return the lines that appear
        in the info-panel between the ID header and the Deps/Type/Status
        footer.  Return ``None`` to use the minimal generic renderer (just
        description).

    special_edit_fn
        ``(node, snapshot, event_queue, set_modal) -> bool`` — open a custom
        edit modal for this node.  Return ``True`` if handled, ``False`` to
        fall through to the generic edit modal.

    export_node_filter_fn
        ``(node_object) -> bool`` — True = include this node in the markdown
        export.  Defaults to including all non-hidden nodes.

    node_type_options
        Node types offered in the curses "Add node" type completion list.

    Web UI
    ------
    find_title_fn
        ``(serialised_snapshot: dict) -> str`` — extract a human-readable
        title for HTML export metadata and the static-file goal display.
        Return ``""`` if unknown.

    extra_routes_fn
        ``(app, orchestrator, run_dir) -> None`` — register domain-specific
        API routes on the FastAPI app after all generic routes have been
        added.  Called only by ``create_app`` (the already-initialised path).
        ``_create_unified_app`` is inherently project-specific and manages its
        own domain routes directly.
    """

    # ── Shared ────────────────────────────────────────────────────────────────
    snapshot_filter_fn: Callable | None = None
    valid_status_values: tuple = ("pending", "running", "done", "failed")

    # ── Curses UI ─────────────────────────────────────────────────────────────
    node_symbol_fn: Callable | None = None
    node_detail_lines_fn: Callable | None = None
    special_edit_fn: Callable | None = None
    export_node_filter_fn: Callable | None = None
    node_type_options: tuple = ("task",)

    # ── Web UI ────────────────────────────────────────────────────────────────
    find_title_fn: Callable | None = None
    extra_routes_fn: Callable | None = None


# ── cuddlytoddly factory ──────────────────────────────────────────────────────


def make_cuddlytoddly_config() -> UIConfig:
    """
    Return a UIConfig wired for the cuddlytoddly agent framework.

    All project-specific imports are deferred to call time so this module
    stays import-safe even before the rest of cuddlytoddly is fully loaded,
    and so that ui_config.py itself never creates a circular-import cycle with
    curses_ui.py.
    """
    import json as _json
    import textwrap as _tw

    # ── Shared ────────────────────────────────────────────────────────────────

    def _snapshot_filter(node) -> bool:
        """Exclude hidden execution_step nodes from serialized snapshots."""
        return not (node.node_type == "execution_step" and node.metadata.get("hidden", False))

    # ── Curses UI ─────────────────────────────────────────────────────────────

    def _node_symbol(node) -> str | None:
        """Map cuddlytoddly node types to distinct git-graph commit symbols."""
        if node.node_type == "goal":
            return "o"
        if node.node_type == "execution_step":
            hidden = node.metadata.get("hidden", False)
            failed = node.status == "failed"
            return "·" if hidden else ("✗" if failed else "◆")
        return None  # keep default "*"

    def _node_detail_lines(node) -> list[str] | None:
        """
        Return info-panel lines for the node detail block.

        Handles the two cuddlytoddly-specific cases:
          - clarification nodes: render JSON field list instead of description
          - any node with broadened_* metadata: append broadening summary
        Returns None for plain nodes that only have a description.
        """
        if node.node_type == "clarification":
            lines = [" Goal context  (press 'e' to edit, then confirm to rerun)", " "]
            try:
                fields = _json.loads(node.result or "[]")
            except Exception:
                fields = []
            for f in fields:
                label = f.get("label") or f.get("key", "?")
                value = f.get("value", "unknown")
                flag = " [?]" if value == "unknown" else ""
                lines.append(f" {label}{flag}")
                lines.append(f"   → {value}")
                lines.append(" ")
            if not fields:
                lines.append(" (no fields)")
            return lines

        # Standard node — include description plus optional broadening block.
        lines: list[str] = []
        desc = node.metadata.get("description")
        if desc:
            lines += [f" Desc:   {desc}", " "]

        broad_desc = node.metadata.get("broadened_description", "")
        broad_missing = node.metadata.get("broadened_for_missing", [])
        broad_reason = node.metadata.get("broadened_reason", "")
        if broad_desc:
            lines.append(" ⟳ Running as (broadened goal):")
            for wrapped in _tw.wrap(broad_desc, width=60):
                lines.append(f"   {wrapped}")
            if broad_missing:
                lines.append(f"   Missing: {', '.join(broad_missing)}")
            if broad_reason:
                lines.append(f"   Reason:  {broad_reason}")
            lines.append(" ")

        req_input = node.metadata.get("required_input")
        if req_input:
            lines += [f" Input:  {req_input}", " "]
        output = node.metadata.get("output")
        if output:
            lines += [f" Output: {output}", " "]

        return lines if lines else None

    def _special_edit(node, snapshot, event_queue, set_modal) -> bool:
        """Open the clarification modal for clarification nodes."""
        if node.node_type == "clarification":
            # Lazy import to avoid a circular dependency: ui_config ← curses_ui.
            from cuddlytoddly.ui.curses_ui import open_clarification_modal

            open_clarification_modal(node.id, snapshot, event_queue, set_modal)
            return True
        return False

    def _export_filter(node) -> bool:
        """Exclude all execution_step nodes and hidden nodes from markdown export."""
        return not (node.node_type == "execution_step" or node.metadata.get("hidden", False))

    # ── Web UI ────────────────────────────────────────────────────────────────

    def _find_title(snapshot: dict) -> str:
        """Return the goal node's description as the export title."""
        goal = next((n for n in snapshot.values() if n.get("node_type") == "goal"), None)
        if not goal:
            return ""
        return (goal.get("metadata") or {}).get("description") or goal.get("id", "")

    def _extra_routes(app, orchestrator, run_dir) -> None:
        """
        Register cuddlytoddly-specific API routes on the FastAPI app.

        Two routes:
          POST /api/node/{id}/clarification/confirm  — update clarification fields
          POST /api/goal/{id}/replan                 — trigger goal replanning
        """
        import json as _json2

        from cuddlytoddly.core.events import (
            RESET_NODE,
            SET_RESULT,
            UPDATE_METADATA,
            Event,
        )

        try:
            from fastapi import HTTPException
        except ImportError:
            raise ImportError(
                "Web UI requires FastAPI and uvicorn:\n  pip install fastapi 'uvicorn[standard]'"
            )

        @app.post("/api/node/{node_id:path}/clarification/confirm")
        async def confirm_clarification(node_id: str, body: dict):
            """
            Confirm user edits to a clarification node.

            Accepts updated_fields — the full list of field dicts with
            user-edited values.  Updates the node result and resets its
            direct children so the plan re-executes with the new context.

            awaiting_input children are intentionally skipped — the
            orchestrator's _resume_unblocked_pass detects filled fields and
            emits RESUME_NODE automatically on the next loop tick.
            """
            snap = orchestrator.get_snapshot()
            node = snap.get(node_id)
            if not node or node.node_type != "clarification":
                raise HTTPException(404, "clarification node not found")

            updated_fields = body.get("updated_fields")
            if not isinstance(updated_fields, list):
                raise HTTPException(400, "updated_fields must be a list")

            new_result = _json2.dumps(updated_fields, ensure_ascii=False)
            q = orchestrator.event_queue
            q.put(
                Event(
                    UPDATE_METADATA,
                    {
                        "node_id": node_id,
                        "origin": "user",
                        "metadata": {"fields": updated_fields},
                    },
                )
            )
            q.put(Event(SET_RESULT, {"node_id": node_id, "result": new_result}))

            for child_id in node.children:
                child = snap.get(child_id)
                if child and child.node_type != "clarification":
                    if child.status != "awaiting_input":
                        q.put(Event(RESET_NODE, {"node_id": child_id}))

            goal_id = node.metadata.get("parent_goal")
            if goal_id and goal_id in snap:
                q.put(
                    Event(
                        UPDATE_METADATA,
                        {
                            "node_id": goal_id,
                            "origin": "user",
                            "metadata": {"expanded": False},
                        },
                    )
                )

            return {"ok": True}

        @app.post("/api/goal/{goal_id:path}/replan")
        async def replan_goal(goal_id: str):
            orchestrator.replan_goal(goal_id)
            return {"ok": True}

    return UIConfig(
        snapshot_filter_fn=_snapshot_filter,
        valid_status_values=("pending", "ready", "running", "done", "failed", "to_be_expanded"),
        node_symbol_fn=_node_symbol,
        node_detail_lines_fn=_node_detail_lines,
        special_edit_fn=_special_edit,
        export_node_filter_fn=_export_filter,
        node_type_options=("task", "goal"),
        find_title_fn=_find_title,
        extra_routes_fn=_extra_routes,
    )
