"""
cuddlytoddly/ui/modals.py

Modal dialog components (ModalField, Modal) and their open_* factory
functions, plus export_results_to_markdown.
"""

from __future__ import annotations

import curses
import textwrap

from cuddlytoddly.ui.ui_config import UIConfig
from toddly.core.events import (
    ADD_DEPENDENCY,
    ADD_NODE,
    REMOVE_DEPENDENCY,
    REMOVE_NODE,
    RESET_NODE,
    SET_RESULT,
    UPDATE_METADATA,
    UPDATE_STATUS,
    Event,
)
from toddly.infra.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# ModalField
# ---------------------------------------------------------------------------


class ModalField:
    """A single editable field inside a modal."""

    def __init__(self, label, value="", completions=None, validator=None):
        self.label = label
        self.value = value
        self.completions = completions or []  # list of strings for autocomplete
        self.validator = validator  # callable(str) -> str|None (error msg)
        self.cursor = len(value)
        self.error = None
        self._completion_idx = -1
        self._completion_prefix = ""
        self._dd_idx = -1  # highlighted row in the visible dropdown list (-1 = none)

    def _current_token(self):
        """Return the text after the last comma (stripped), for autocomplete."""
        parts = self.value[: self.cursor].rsplit(",", 1)
        return parts[-1].strip()

    def _select_dd_item(self, completion):
        """Insert *completion*, replacing the current comma-token."""
        before_cursor = self.value[: self.cursor]
        last_comma = before_cursor.rfind(",")
        if last_comma == -1:
            self.value = completion + self.value[self.cursor :]
        else:
            prefix_part = self.value[: last_comma + 1] + " "
            self.value = prefix_part + completion + self.value[self.cursor :]
        self.cursor = len(self.value)
        self._completion_idx = -1
        self._dd_idx = -1

    def _dd_matches(self):
        """Return the current filtered completion list (up to 8)."""
        token = self._current_token()
        return [c for c in self.completions if c.startswith(token)][:8]

    def handle_key(self, k):
        if k == curses.KEY_BACKSPACE or k == 127:
            if self.cursor > 0:
                self.value = self.value[: self.cursor - 1] + self.value[self.cursor :]
                self.cursor -= 1
                self._completion_idx = -1
        elif k == curses.KEY_LEFT:
            self.cursor = max(0, self.cursor - 1)
        elif k == curses.KEY_RIGHT:
            self.cursor = min(len(self.value), self.cursor + 1)
        elif k == ord("\t") and self.completions:
            token = self._current_token()
            matches = [c for c in self.completions if c.startswith(token)]
            if matches:
                self._completion_idx = (self._completion_idx + 1) % len(matches)
                self._dd_idx = self._completion_idx  # keep dropdown highlight in sync
                completion = matches[self._completion_idx]
                # Replace only the current token, preserving everything before it
                before_cursor = self.value[: self.cursor]
                last_comma = before_cursor.rfind(",")
                if last_comma == -1:
                    # No comma — replace entire value
                    self.value = completion + self.value[self.cursor :]
                else:
                    # Replace only the token after the last comma
                    prefix_part = self.value[: last_comma + 1] + " "
                    self.value = prefix_part + completion + self.value[self.cursor :]
                self.cursor = len(self.value)
        elif 32 <= k <= 126:
            ch = chr(k)
            self.value = self.value[: self.cursor] + ch + self.value[self.cursor :]
            self.cursor += 1
            self._completion_idx = -1  # reset on typing
            self._dd_idx = -1  # reset dropdown selection on typing

    def validate(self):
        if self.validator:
            self.error = self.validator(self.value)
        return self.error is None


# ---------------------------------------------------------------------------
# Modal
# ---------------------------------------------------------------------------


class Modal:
    """
    Multi-field modal dialog rendered over the info panel area.

    fields: list of ModalField
    title: str
    on_submit: callable(dict of label->value) -> None
    on_cancel: callable() -> None
    """

    def __init__(self, title, fields, on_submit, on_cancel):
        self.title = title
        self.fields = fields
        self.on_submit = on_submit
        self.on_cancel = on_cancel
        self.active_field = 0

    def handle_key(self, k):
        if k == 27:  # Escape
            self.on_cancel()
            return

        # ── Field navigation — always available ───────────────────────────────
        if k == curses.KEY_DOWN:
            self.active_field = (self.active_field + 1) % len(self.fields)
            return

        if k == curses.KEY_UP:
            self.active_field = (self.active_field - 1) % len(self.fields)
            return

        # ── Tab: cycle completions and highlight the chosen match ─────────────
        if k == ord("\t") and self.fields[self.active_field].completions:
            self.fields[self.active_field].handle_key(k)
            return

        # ── Enter: submit ─────────────────────────────────────────────────────
        if k in (10, 13):
            all_valid = all(f.validate() for f in self.fields)
            if all_valid:
                self.on_submit({f.label: f.value for f in self.fields})
            return

        # ── All other keys → active field ─────────────────────────────────────
        self.fields[self.active_field].handle_key(k)

    def draw(self, stdscr, h, w):
        panel_x = w // 2 + 1
        panel_w = w - panel_x - 1

        # Clear the panel area before drawing
        for row in range(0, h - 2):
            try:
                stdscr.addstr(row, panel_x, " " * panel_w)
            except curses.error:
                pass

        row = 0
        # Title
        try:
            stdscr.addstr(row, panel_x, f" {self.title} ".center(panel_w, "─"), curses.A_BOLD)
        except curses.error:
            pass
        row += 2

        for i, field in enumerate(self.fields):
            is_active = i == self.active_field
            attr = curses.A_REVERSE if is_active else 0

            label_str = f" {field.label}: "
            try:
                stdscr.addstr(row, panel_x, label_str)
            except curses.error:
                pass

            val_x = panel_x + len(label_str)
            val_w = w - val_x - 1

            # Wrap the value across multiple lines
            wrapped_val = textwrap.wrap(field.value if field.value else " ", width=val_w) or [" "]
            for j, wline in enumerate(wrapped_val):
                try:
                    stdscr.addstr(row + j, val_x, wline.ljust(val_w), attr)
                except curses.error:
                    pass
            row += len(wrapped_val)

            if is_active and field.error:
                try:
                    stdscr.addstr(
                        row,
                        panel_x,
                        f" ! {field.error}",
                        curses.color_pair(curses.COLOR_RED + 1),
                    )
                except curses.error:
                    pass
                row += 1

            # Autocomplete dropdown
            if is_active and field.completions:
                matches = field._dd_matches()
                if matches:
                    for j, m in enumerate(matches):
                        is_sel = j == field._dd_idx
                        item_attr = curses.A_REVERSE if is_sel else curses.A_DIM
                        prefix = "▶ " if is_sel else "  "
                        try:
                            stdscr.addstr(row + j, val_x, (prefix + m)[:val_w], item_attr)
                        except curses.error:
                            pass
                    row += len(matches)

            row += 1
        # Footer
        try:
            stdscr.addstr(
                row + 1,
                panel_x,
                " ↑↓: switch field  Tab: complete  Enter: confirm  Esc: cancel",
                curses.A_DIM,
            )
        except curses.error:
            pass


# ---------------------------------------------------------------------------
# Markdown export
# ---------------------------------------------------------------------------


def export_results_to_markdown(snapshot, run_dir, config: UIConfig | None = None):
    """Walk the DAG in topological order and write all results to a
    Markdown file in <run_dir>/outputs/. Returns the output path."""
    from datetime import datetime

    out_dir = run_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"export_{timestamp}.md"

    # Node inclusion predicate — project-supplied or generic fallback.
    _include = (
        config.export_node_filter_fn
        if config and config.export_node_filter_fn
        else lambda n: not n.metadata.get("hidden", False)
    )

    def topo_sort(snap):
        visited, order = set(), []

        def visit(nid):
            if nid in visited:
                return
            visited.add(nid)
            node = snap.get(nid)
            if node:
                for dep in sorted(node.dependencies):
                    visit(dep)
            order.append(nid)

        for nid in sorted(snap.keys()):
            visit(nid)
        return order

    order = topo_sort(snapshot)

    # Title: prefer config's find_title_fn; fall back to first exportable
    # node whose id or description looks like a goal, then run_dir name.
    if config and config.find_title_fn:
        # find_title_fn expects the serialised (dict-of-dicts) format used by
        # the web layer; convert to a compatible form using node attributes.
        _snap_for_title = {
            nid: {"node_type": n.node_type, "id": n.id, "metadata": dict(n.metadata)}
            for nid, n in snapshot.items()
        }
        title = config.find_title_fn(_snap_for_title) or run_dir.name.replace("_", " ").title()
    else:
        title = run_dir.name.replace("_", " ").title()

    lines = [
        f"# {title}",
        "",
        f"*Exported {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Node | Type | Status |",
        "|------|------|--------|",
    ]
    for nid in order:
        node = snapshot.get(nid)
        if not node or not _include(node):
            continue
        lines.append(f"| {nid} | {node.node_type} | {node.status} |")
    lines += ["", "---", "", "## Results", ""]

    for nid in order:
        node = snapshot.get(nid)
        if not node or not _include(node):
            continue
        desc = node.metadata.get("description", "")
        lines.append(f"### {nid}")
        if desc and desc != nid:
            lines += [f"*{desc}*", ""]
        deps = ", ".join(sorted(node.dependencies)) or "none"
        lines.append(f"**Type:** {node.node_type} | **Status:** {node.status} | **Deps:** {deps}")
        lines.append("")
        req_input = node.metadata.get("required_input")
        output = node.metadata.get("output")
        if req_input:
            lines.append(f"**Input:** `{req_input}`")
        if output:
            lines.append(f"**Output:** `{output}`")
        if req_input or output:
            lines.append("")
        if node.result:
            lines += ["**Result:**", "", "```", str(node.result).strip(), "```"]
        else:
            lines.append("*No result yet.*")
        notes = node.metadata.get("reflection_notes", [])
        if notes:
            lines += ["", "**Notes:**"] + [f"- {n}" for n in notes]
        lines += ["", "---", ""]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("[EXPORT] Written to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# Modal factory functions
# ---------------------------------------------------------------------------


def open_add_modal(snapshot, event_queue, current_node, set_modal, config: UIConfig | None = None):
    node_ids = list(snapshot.keys())

    def on_submit(values):
        new_id = values["ID"].strip()
        new_desc = values["Description"].strip()
        deps_raw = values["Dependencies"].strip()
        dependents_raw = values["Dependents"].strip()
        ntype = values["Type"].strip() or "task"

        if not new_id or new_id in snapshot:
            set_modal(None)
            return

        deps = [d.strip() for d in deps_raw.split(",") if d.strip() and d.strip() in snapshot]
        dependents = [
            d.strip() for d in dependents_raw.split(",") if d.strip() and d.strip() in snapshot
        ]

        event_queue.put(
            Event(
                ADD_NODE,
                {
                    "node_id": new_id,
                    "node_type": ntype,
                    "dependencies": deps,
                    "origin": "user",
                    "metadata": {"description": new_desc},
                },
            )
        )

        for dependent_id in dependents:
            event_queue.put(
                Event(
                    ADD_DEPENDENCY,
                    {
                        "node_id": dependent_id,
                        "depends_on": new_id,
                    },
                )
            )
            # Reset the dependent so it reruns; _on_node_done cascades further if its result changes.
            event_queue.put(Event(RESET_NODE, {"node_id": dependent_id}))

        set_modal(None)

    set_modal(
        Modal(
            title="Add Node",
            fields=[
                ModalField("ID", value=""),
                ModalField("Description", value=""),
                ModalField(
                    "Type",
                    value="task",
                    completions=list(config.node_type_options if config else ("task",)),
                ),
                ModalField("Dependencies", value=current_node or "", completions=node_ids),
                ModalField("Dependents", value="", completions=node_ids),
            ],
            on_submit=on_submit,
            on_cancel=lambda: set_modal(None),
        )
    )


def open_clarification_modal(current_node, snapshot, event_queue, set_modal):
    """
    Per-field editing modal for clarification nodes.

    Fields are shown one at a time with their label, current value, and
    rationale.  Tab / arrow keys navigate between fields.  Pressing Enter
    on the "Confirm & rerun" field commits the edits and triggers a rerun
    of all direct children of this clarification node.
    """
    import json as _json

    node = snapshot[current_node]
    try:
        fields = _json.loads(node.result or "[]")
    except Exception:
        fields = []

    # Work on a mutable copy so the user can cancel without side effects.
    draft = [dict(f) for f in fields]

    def on_submit(values):
        # Merge edited values back into draft using field index labels
        for i, f in enumerate(draft):
            key = f"Field {i + 1}: {f.get('label', f['key'])}"
            if key in values:
                new_val = values[key].strip() or "unknown"
                draft[i]["value"] = new_val

        new_result = _json.dumps(draft, ensure_ascii=False)

        # Update the node metadata + result
        event_queue.put(
            Event(
                UPDATE_METADATA,
                {
                    "node_id": current_node,
                    "origin": "user",
                    "metadata": {"fields": draft},
                },
            )
        )
        event_queue.put(
            Event(
                SET_RESULT,
                {
                    "node_id": current_node,
                    "result": new_result,
                },
            )
        )

        # Reset direct children only
        for child_id in node.children:
            child = snapshot.get(child_id)
            if child and child.node_type != "clarification":
                event_queue.put(Event(RESET_NODE, {"node_id": child_id}))

        # Mark parent goal unexpanded for partial replan
        goal_id = node.metadata.get("parent_goal")
        if goal_id and goal_id in snapshot:
            event_queue.put(
                Event(
                    UPDATE_METADATA,
                    {
                        "node_id": goal_id,
                        "origin": "user",
                        "metadata": {"expanded": False},
                    },
                )
            )

        set_modal(None)

    # Build one ModalField per clarification field.
    # The label encodes the field index so on_submit can match values back.
    modal_fields = []
    for i, f in enumerate(draft):
        label = f"Field {i + 1}: {f.get('label', f['key'])}"
        cur_value = f.get("value", "unknown")
        if cur_value == "unknown":
            cur_value = ""
        modal_fields.append(
            ModalField(
                label=label,
                value=cur_value,
            )
        )

    # Final read-only info field telling the user what Confirm does
    modal_fields.append(
        ModalField(
            label="→ Press Enter on last field or submit to confirm & rerun",
            value="",
        )
    )

    set_modal(
        Modal(
            title="Edit goal context  (Tab: next field  Enter: confirm)",
            fields=modal_fields,
            on_submit=on_submit,
            on_cancel=lambda: set_modal(None),
        )
    )


def open_edit_modal(current_node, snapshot, event_queue, set_modal, config: UIConfig | None = None):
    node = snapshot[current_node]
    node_ids = [nid for nid in snapshot.keys() if nid != current_node]
    current_deps = ", ".join(node.dependencies)
    current_dependents = ", ".join(
        nid for nid, n in snapshot.items() if current_node in n.dependencies
    )
    current_result = node.result or ""

    def on_submit(values):
        new_id = values["ID"].strip()
        new_desc = values["Description"].strip()
        new_deps_raw = values["Dependencies"].strip()
        new_status = values["Status"].strip()
        new_dep_raw = values["Dependents"].strip()
        new_result = values["Result"].strip()

        new_deps = [d.strip() for d in new_deps_raw.split(",") if d.strip()]
        new_deps = [d for d in new_deps if d in snapshot]

        new_dependents = [
            d.strip()
            for d in new_dep_raw.split(",")
            if d.strip() and d.strip() in snapshot and d.strip() != current_node
        ]
        new_dependents_set = set(new_dependents)
        old_dependents = {nid for nid, n in snapshot.items() if current_node in n.dependencies}

        event_queue.put(
            Event(
                UPDATE_METADATA,
                {
                    "node_id": current_node,
                    "origin": "user",
                    "metadata": {"description": new_desc},
                },
            )
        )

        if new_status in (
            config.valid_status_values if config else ("pending", "done", "running", "failed")
        ):
            event_queue.put(
                Event(
                    UPDATE_STATUS,
                    {
                        "node_id": current_node,
                        "status": new_status,
                    },
                )
            )

        old_deps = set(node.dependencies)
        new_deps_set = set(new_deps)
        for removed in old_deps - new_deps_set:
            event_queue.put(
                Event(
                    REMOVE_DEPENDENCY,
                    {
                        "node_id": current_node,
                        "depends_on": removed,
                    },
                )
            )
        for added in new_deps_set - old_deps:
            event_queue.put(
                Event(
                    ADD_DEPENDENCY,
                    {
                        "node_id": current_node,
                        "depends_on": added,
                    },
                )
            )

        for removed in old_dependents - new_dependents_set:
            event_queue.put(
                Event(
                    REMOVE_DEPENDENCY,
                    {
                        "node_id": removed,
                        "depends_on": current_node,
                    },
                )
            )
            event_queue.put(Event(RESET_NODE, {"node_id": removed}))
        for added in new_dependents_set - old_dependents:
            event_queue.put(
                Event(
                    ADD_DEPENDENCY,
                    {
                        "node_id": added,
                        "depends_on": current_node,
                    },
                )
            )
            event_queue.put(Event(RESET_NODE, {"node_id": added}))

        # ── Cascade decision ──────────────────────────────────────────────
        # Only cascade when something that feeds into the downstream LLM
        # prompt actually changed: result, description, or dependencies.
        # Status-only changes do not affect LLM input → no cascade.
        #
        # Lazy cascade strategy:
        # • result changed  → SET_RESULT (node keeps status) + RESET_NODE on
        #                     each done direct child only.  When each child
        #                     reruns, _on_node_done compares its new result to
        #                     its previous result and cascades further only if
        #                     it changed — propagating all the way to leaf nodes.
        # • desc/deps changed → RESET_NODE on the node itself only.  After it
        #                     reruns, the same _on_node_done logic cascades to
        #                     done children if the result changes.
        # • status-only     → no cascade at all.
        result_changed = new_result != current_result
        desc_changed = new_desc != node.metadata.get("description", "")
        deps_changed = new_deps_set != old_deps
        is_rename = bool(new_id and new_id != current_node and new_id not in snapshot)

        if result_changed:
            # Preserve the node's current status.
            # Reset only done direct children — _on_node_done cascades further.
            event_queue.put(
                Event(
                    SET_RESULT,
                    {
                        "node_id": current_node,
                        "result": new_result if new_result else None,
                    },
                )
            )
            for child_id in node.children:
                child = snapshot.get(child_id)
                if child and child.status == "done":
                    event_queue.put(Event(RESET_NODE, {"node_id": child_id}))
        elif (desc_changed or deps_changed) and not is_rename:
            # Reset the node itself only; _on_node_done cascades if result changes.
            event_queue.put(Event(RESET_NODE, {"node_id": current_node}))

        if new_id and new_id != current_node and new_id not in snapshot:
            event_queue.put(
                Event(
                    ADD_NODE,
                    {
                        "node_id": new_id,
                        "node_type": node.node_type,
                        "dependencies": list(new_deps_set),
                        "origin": node.origin,
                        "metadata": {**node.metadata, "description": new_desc},
                    },
                )
            )
            for child in node.children:
                event_queue.put(Event(ADD_DEPENDENCY, {"node_id": child, "depends_on": new_id}))
                event_queue.put(
                    Event(
                        REMOVE_DEPENDENCY,
                        {"node_id": child, "depends_on": current_node},
                    )
                )
            event_queue.put(Event(REMOVE_NODE, {"node_id": current_node}))
            event_queue.put(Event(RESET_NODE, {"node_id": new_id}))

        set_modal(None)

    set_modal(
        Modal(
            title="Edit Node",
            fields=[
                ModalField("ID", value=current_node),
                ModalField("Description", value=node.metadata.get("description", "")),
                ModalField("Dependencies", value=current_deps, completions=node_ids),
                ModalField("Dependents", value=current_dependents, completions=node_ids),
                ModalField(
                    "Status",
                    value=node.status,
                    completions=list(
                        config.valid_status_values
                        if config
                        else ("pending", "running", "done", "failed")
                    ),
                ),
                ModalField("Result", value=current_result),
            ],
            on_submit=on_submit,
            on_cancel=lambda: set_modal(None),
        )
    )


def open_remove_modal(current_node, snapshot, event_queue, set_modal):
    node = snapshot[current_node]
    parents = list(node.dependencies)
    children = list(node.children)

    options = [
        ("Remove node only — rewire children to its parents", "rewire"),
        ("Remove node and all descendants", "cascade"),
        ("Remove node and disconnect everything", "disconnect"),
    ]

    def on_submit(values):
        choice = values["Action"].strip()
        mode = next((m for label, m in options if label == choice), None)

        if mode == "rewire":
            for child in children:
                event_queue.put(
                    Event(
                        REMOVE_DEPENDENCY,
                        {
                            "node_id": child,
                            "depends_on": current_node,
                        },
                    )
                )
                for parent in parents:
                    event_queue.put(
                        Event(
                            ADD_DEPENDENCY,
                            {
                                "node_id": child,
                                "depends_on": parent,
                            },
                        )
                    )
            event_queue.put(Event(REMOVE_NODE, {"node_id": current_node}))
            # Children survive — reset them and their subtrees
            for child in children:
                event_queue.put(Event(RESET_NODE, {"node_id": child}))

        elif mode == "cascade":
            # REMOVE_NODE recurses into children — nothing left to reset
            event_queue.put(Event(REMOVE_NODE, {"node_id": current_node}))

        elif mode == "disconnect":
            for child in children:
                event_queue.put(
                    Event(
                        REMOVE_DEPENDENCY,
                        {
                            "node_id": child,
                            "depends_on": current_node,
                        },
                    )
                )
            event_queue.put(Event(REMOVE_NODE, {"node_id": current_node}))
            # Children survive without this dep — reset them
            for child in children:
                event_queue.put(Event(RESET_NODE, {"node_id": child}))

        set_modal(None)

    set_modal(
        Modal(
            title=f"Remove: {current_node}  ({len(children)} children, {len(parents)} parents)",
            fields=[
                ModalField(
                    "Action",
                    value=options[0][0],
                    completions=[label for label, _ in options],
                ),
            ],
            on_submit=on_submit,
            on_cancel=lambda: set_modal(None),
        )
    )
