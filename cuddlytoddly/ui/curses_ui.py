"""
cuddlytoddly/ui/curses_ui.py

Curses UI wired to Planner Runtime

Design:
- UI never mutates TaskGraph directly
- All edits emit Events into EventQueue
- Graph state is read via snapshot
- Git repo is rebuilt from TaskGraph snapshot
- Rendering logic is minimally modified
"""

from __future__ import annotations

import curses
import hashlib
import logging
import textwrap
import time
from pathlib import Path

import cuddlytoddly.ui.git_projection as git_proj
from cuddlytoddly.ui.ansi_utils import (
    ANSI_COLOR_MAP,
    get_node_col,
    map_nodes_to_lines,
    parse_ansi,
    trace_branch_path_recursive,
)
from cuddlytoddly.ui.dag_utils import (
    build_reverse_dag,
    ensure_path_starts_at_root,
    find_path_to_node,
    find_root_node,
    get_aggregate_outputs,
    get_git_dag_text,
)
from cuddlytoddly.ui.git_projection import (
    graph_to_dag,
    rebuild_repo_from_graph,
)
from cuddlytoddly.ui.modals import (
    export_results_to_markdown,
    open_add_modal,
    open_edit_modal,
    open_remove_modal,
)
from cuddlytoddly.ui.ui_config import UIConfig
from toddly.infra.logging import get_logger

logger = get_logger(__name__)


def dag_interface(
    stdscr, orchestrator, run_dir=None, repo_path=None, config: UIConfig | None = None
):
    graph = orchestrator.graph
    graph_lock = orchestrator.graph_lock
    event_queue = orchestrator.event_queue

    active_modal = None

    def set_modal(m):
        nonlocal active_modal
        active_modal = m

    logger.debug("=== UI Debug Log Started ===")

    curses.start_color()
    curses.use_default_colors()

    for idx, color in ANSI_COLOR_MAP.items():
        curses.init_pair(color + 1, color, -1)

    stdscr.nodelay(True)
    stdscr.keypad(True)
    curses.curs_set(0)

    last_seen_version = -1
    last_exec_version = -1

    cached_git_lines = []
    cached_node_to_line = {}

    current_node = None
    parent_node = None

    parent_stack = []
    child_stack = []
    branch_mode = False
    selection_index = 0

    info_scroll = 0
    export_notice = None  # (message, expire_time) or None

    loop_count = 0
    snapshot = {}
    dag = {}
    reverse_dag = {}
    last_missing_nodes: frozenset = frozenset()
    switch_requested = False

    while True:
        loop_count += 1

        # ── Input first — always consume keypresses regardless of render state ──
        k = stdscr.getch()

        if k == ord("q"):
            break

        # ── Snapshot ──────────────────────────────────────────────────────────
        try:
            with graph_lock:
                snapshot = graph.get_snapshot()
        except Exception as e:
            logger.error("[UI LOOP %d] Failed to get snapshot: %s", loop_count, e)
            snapshot = {}

        logger.debug("[UI LOOP %d] Snapshot keys: %s", loop_count, list(snapshot.keys()))

        version_at_rebuild_start = graph.structure_version
        exec_version_at_rebuild_start = graph.execution_version

        # ── Rebuild git projection if graph changed ───────────────────────────
        skip_render = False
        if (
            graph.structure_version != last_seen_version
            or graph.execution_version != last_exec_version
        ):
            logger.debug(
                "[UI REBUILD] Version changed from %d to %d",
                last_seen_version,
                graph.structure_version,
            )

            try:
                with graph_lock:
                    snapshot = graph.get_snapshot()

                rebuild_repo_from_graph(graph)
                cached_git_lines = get_git_dag_text(repo_path)
                cached_node_to_line = map_nodes_to_lines(cached_git_lines, snapshot)

                last_seen_version = version_at_rebuild_start
                last_exec_version = exec_version_at_rebuild_start

                dag = graph_to_dag(snapshot)
                reverse_dag = build_reverse_dag(dag)

                if current_node not in snapshot:
                    current_node = find_root_node(snapshot)
                    parent_stack = []
                    child_stack = find_path_to_node(reverse_dag, current_node)
                else:
                    parent_stack = find_path_to_node(dag, current_node)
                    child_stack = find_path_to_node(reverse_dag, current_node)

                missing = set(snapshot.keys()) - set(cached_node_to_line.keys())
                if missing:
                    missing_frozen = frozenset(missing)
                    if missing_frozen != last_missing_nodes:
                        logger.warning("[UI] Nodes in snapshot but NOT in git map: %s", missing)
                        last_missing_nodes = missing_frozen
                    # force rebuild next iteration but don't skip input
                    last_seen_version = -1
                    last_exec_version = -1
                    skip_render = True
                else:
                    last_missing_nodes = frozenset()

            except Exception as e:
                logger.error("[UI REBUILD] Failed: %s", e, exc_info=True)
                skip_render = True
        else:
            try:
                with graph_lock:
                    snapshot = graph.get_snapshot()
            except Exception:
                pass
            dag = graph_to_dag(snapshot)
            reverse_dag = build_reverse_dag(dag)

        # ── Rendering ─────────────────────────────────────────────────────────
        if not skip_render:
            try:
                git_lines = cached_git_lines
                node_to_line = cached_node_to_line

                if not snapshot:
                    current_node = None
                else:
                    if current_node not in snapshot:
                        current_node = find_root_node(snapshot)

                stdscr.clear()
                h, w = stdscr.getmaxyx()

                if current_node:
                    current_line = node_to_line.get(current_node)
                    current_col = (
                        get_node_col(git_lines[current_line]) if current_line is not None else 0
                    )
                else:
                    current_line = None
                    current_col = None

                if parent_node and current_node:
                    parent_line = node_to_line.get(parent_node)
                    parent_col = (
                        get_node_col(git_lines[parent_line]) if parent_line is not None else None
                    )
                else:
                    parent_node = None
                    parent_line = None
                    parent_col = None

                # Branch path highlight
                if (
                    branch_mode
                    and parent_node
                    and current_line is not None
                    and parent_line is not None
                ):
                    path = trace_branch_path_recursive(
                        git_lines,
                        parent_line,
                        parent_col,
                        current_line,
                        current_col,
                    )
                else:
                    path = set()

                # Node-type symbol map — build a single dict from config.node_symbol_fn
                # (or keep no overrides if no config is provided, leaving all nodes as "*").
                symbol_positions: dict = {}  # line_idx -> (col, symbol_char)
                if config and config.node_symbol_fn:
                    for node_id, line_idx in node_to_line.items():
                        node = snapshot.get(node_id)
                        if not node:
                            continue
                        sym = config.node_symbol_fn(node)
                        if sym is not None:
                            symbol_positions[line_idx] = (
                                get_node_col(git_lines[line_idx]),
                                sym,
                            )

                line_to_node = {v: k for k, v in node_to_line.items()}

                # Label overrides (hash → description)
                line_label_overrides = {}
                for line_idx, node_id in line_to_node.items():
                    node = snapshot.get(node_id)
                    if not node:
                        continue
                    h6 = "#" + hashlib.sha256(node_id.encode()).hexdigest()[:6]
                    desc = node.metadata.get("description") or node_id
                    line_label_overrides[line_idx] = (h6, f"{h6} {desc}")

                if git_lines:
                    start = 0
                    if current_line is not None:
                        start = max(0, current_line - h // 2)

                    for i, line in enumerate(git_lines[start : start + h - 1]):
                        parsed = parse_ansi(line)
                        x = 0
                        current_line_idx = i + start
                        override = line_label_overrides.get(current_line_idx)

                        hash_start_x = None
                        if override:
                            h6 = override[0]
                            visible = "".join(ch for ch, _ in parsed)
                            idx = visible.find(h6)
                            if idx != -1:
                                hash_start_x = idx

                        for ch, color in parsed:
                            if x >= w // 2 - 1:
                                break
                            highlight = 0

                            if (
                                current_node
                                and current_line_idx == current_line
                                and x == current_col
                            ):
                                highlight = curses.A_REVERSE

                            if (current_line_idx, x) in path:
                                highlight = curses.A_REVERSE

                            if ch == "*" and current_line_idx in symbol_positions:
                                col, sym = symbol_positions[current_line_idx]
                                if x == col:
                                    ch = sym

                            if hash_start_x is not None and x == hash_start_x:
                                full_label = override[1]
                                available = (w // 2 - 1) - x
                                label = (
                                    (full_label[: available - 3] + "...")
                                    if len(full_label) > available
                                    else full_label
                                )
                                for lch in label:
                                    if x >= w // 2 - 1:
                                        break
                                    try:
                                        stdscr.addstr(i, x, lch, color | highlight)
                                    except curses.error:
                                        pass
                                    x += 1
                                break

                            try:
                                stdscr.addstr(i, x, ch, color | highlight)
                            except curses.error:
                                pass
                            x += 1

                # Status bars
                node_label = current_node if current_node else "<empty>"
                llm_paused = orchestrator.llm_stopped
                paused_indicator = " | [PAUSED]" if llm_paused else ""
                activity = orchestrator.current_activity
                started = orchestrator.activity_started

                if activity and started:
                    elapsed = time.time() - started
                    activity_str = f" {activity} ({elapsed:.0f}s)"
                else:
                    activity_str = ""

                status_line = (
                    "Up/Down/Left/Right/[/]: move | "
                    f"j/k </> scroll info | "
                    f"e: edit | a: add | x: remove | p: export | "
                    f"s: {'resume' if llm_paused else 'pause'} | g: switch goal | q: quit"
                    f"{paused_indicator}"
                )
                tc = orchestrator.token_counts
                token_str = f"Tokens: {tc['total']:,} ({tc['calls']} calls)"

                if export_notice and time.time() < export_notice[1]:
                    debug_line = export_notice[0]
                else:
                    export_notice = None
                    debug_line = (
                        f"Node: {node_label} | "
                        f"{'Branch' if branch_mode else 'Node'} | "
                        f"Nodes: {len(snapshot)} "
                        f"{token_str} |"
                        f"{activity_str}"
                    )

                stdscr.addstr(h - 2, 0, debug_line[: w - 1])
                stdscr.addstr(h - 1, 0, status_line[: w - 1])

                if branch_mode and parent_node:
                    selected_nodes = [parent_node, current_node]
                else:
                    selected_nodes = [current_node]

                if active_modal:
                    active_modal.draw(stdscr, h, w)
                else:
                    draw_info_panel(
                        stdscr,
                        h,
                        w,
                        current_node,
                        snapshot,
                        selected_nodes,
                        info_scroll,
                        config,
                    )

                stdscr.refresh()

            except Exception as e:
                logger.error("[UI] Render error: %s", e, exc_info=True)

        # ── Key handling — always runs ─────────────────────────────────────────
        if k == -1:
            time.sleep(0.02)
            continue

        if not current_node:
            continue

        if not parent_node:
            parents = reverse_dag.get(current_node, [])
            if parents:
                parent_node = parents[0]

        parent_stack = ensure_path_starts_at_root(dag, parent_stack + [current_node])[:-1]
        child_stack = ensure_path_starts_at_root(reverse_dag, child_stack + [current_node])[:-1]

        if active_modal:
            active_modal.handle_key(k)
            continue

        if k == curses.KEY_UP:
            info_scroll = 0
            children = dag.get(current_node, [])
            if not branch_mode and children:
                parent_stack.append(current_node)
                parent_node = current_node
                current_node = child_stack.pop() if child_stack else current_node
                branch_mode = True
            elif branch_mode:
                branch_mode = False

        elif k == curses.KEY_DOWN:
            info_scroll = 0
            parents = reverse_dag.get(current_node, [])
            if not branch_mode and parents:
                parent_node = parent_stack[-1] if parent_stack else None
                branch_mode = True
            elif branch_mode:
                if parent_stack:
                    child_stack.append(current_node)
                    current_node = parent_stack.pop()
                branch_mode = False

        elif k in (curses.KEY_LEFT, curses.KEY_RIGHT):
            info_scroll = 0
            if parent_node or parent_stack:
                if not parent_node and parent_stack:
                    parent_node = parent_stack[-1]
                siblings = dag.get(parent_node, [])
                if siblings and current_node in siblings:
                    current_index = siblings.index(current_node)
                    delta = -1 if k == curses.KEY_LEFT else 1
                    selection_index = (current_index + delta) % len(siblings)
                    current_node = siblings[selection_index]
                    child_stack = find_path_to_node(reverse_dag, current_node)

        elif k in (ord("["), ord("]")):
            info_scroll = 0
            parents = reverse_dag.get(current_node, [])
            if parent_node and parents and parent_node in parents:
                parent_index = parents.index(parent_node)
                delta = -1 if k == ord("[") else 1
                selection_index = (parent_index + delta) % len(parents)
                parent_node = parents[selection_index]
                parent_stack = find_path_to_node(dag, parent_node) + [parent_node]

        elif k == ord("s"):
            if orchestrator.llm_stopped:
                orchestrator.resume_llm_calls()
            else:
                orchestrator.stop_llm_calls()

        elif k == ord("e"):
            if current_node:
                node = snapshot.get(current_node)
                if node:
                    handled = (
                        config.special_edit_fn(node, snapshot, event_queue, set_modal)
                        if config and config.special_edit_fn
                        else False
                    )
                    if not handled:
                        open_edit_modal(current_node, snapshot, event_queue, set_modal, config)

        elif k == ord("a"):
            open_add_modal(snapshot, event_queue, current_node, set_modal, config)

        elif k == ord("x"):
            if current_node:
                open_remove_modal(current_node, snapshot, event_queue, set_modal)

        elif k == ord("p"):
            if run_dir and snapshot:
                try:
                    out_path = export_results_to_markdown(snapshot, run_dir, config)
                    export_notice = (f"Exported → {out_path.name}", time.time() + 4)
                except Exception as ex:
                    export_notice = (f"Export failed: {ex}", time.time() + 4)
                    logger.error("[EXPORT] Failed: %s", ex, exc_info=True)

        elif k in (curses.KEY_PPAGE, ord("<")):  # Page Up
            info_scroll = max(0, info_scroll - (h - 4))

        elif k in (curses.KEY_NPAGE, ord(">")):  # Page Down
            info_scroll += h - 4  # draw_info_panel clamps the max

        elif k == ord("j"):  # fine scroll down
            info_scroll += 3

        elif k == ord("k"):  # fine scroll up
            info_scroll = max(0, info_scroll - 3)

        elif k == ord("g"):
            switch_requested = True
            break

    return "switch" if switch_requested else None


def draw_info_panel(
    stdscr, h, w, node_id, snapshot, selected_nodes, scroll_offset=0, config: UIConfig | None = None
):
    if not node_id or not snapshot:
        return

    node = snapshot[node_id]
    panel_x = w // 2
    panel_w = w - panel_x

    # Border
    for row in range(0, h - 2):
        try:
            stdscr.addch(row, panel_x, curses.ACS_VLINE)
        except curses.error:
            pass

    # Node info lines
    lines = [
        f" ID:     {node.id}",
        " ",
    ]

    # ── Detail block — project-supplied or generic fallback ──────────────────
    detail_lines = (
        config.node_detail_lines_fn(node) if config and config.node_detail_lines_fn else None
    )
    if detail_lines is not None:
        lines += detail_lines
    else:
        # Generic: show description only.
        desc = node.metadata.get("description")
        if desc:
            lines += [f" Desc:   {desc}", " "]

    lines += [
        f" Deps:   {', '.join(node.dependencies) or 'none'}",
        " ",
        f" Type:   {node.node_type}",
        f" Status: {node.status}",
        f" Origin: {node.origin}",
        " ",
    ]
    notes = node.metadata.get("reflection_notes", [])
    if notes:
        lines.append(" Notes:")
        for note in notes:
            lines.append(f"   {note}")

    # --- AGGREGATE OUTPUTS ---
    results = get_aggregate_outputs(snapshot)
    if results:
        lines.append(" ")
        lines.append(" ")
        lines.append(" ")

        # --- RESULTS ---
        lines.append(" Results:")
        for nid in selected_nodes:
            node = snapshot.get(nid)
            if not node:
                continue

            if node.result is not None:
                lines.append(f"   [{nid}]")
                for wrapped_line in textwrap.wrap(str(node.result), width=panel_w - 5):
                    lines.append(f"   {wrapped_line}")
            else:
                # Node has no result (e.g. a goal) — show direct deps' results instead
                for dep_id in node.dependencies:
                    dep = snapshot.get(dep_id)
                    if dep and dep.result is not None:
                        lines.append(f"   [{dep_id}]")
                        for wrapped_line in textwrap.wrap(str(dep.result), width=panel_w - 5):
                            lines.append(f"   {wrapped_line}")

    # After showing the node's own result, show any visible execution steps.
    # Use snapshot_filter_fn to decide which nodes are "visible" — the same
    # predicate used by the web and git layers.  Falls back to excluding
    # hidden nodes when no config is provided.
    _visible = (
        config.snapshot_filter_fn
        if config and config.snapshot_filter_fn
        else lambda n: not n.metadata.get("hidden", False)
    )
    step_children = [
        n
        for n in snapshot.values()
        if n.node_type == "execution_step" and node_id in n.dependencies and _visible(n)
    ]
    if step_children:
        lines.append(" ")
        lines.append(" Execution steps:")
        for step in step_children:
            status_icon = "✓" if step.status == "done" else "✗" if step.status == "failed" else "…"
            lines.append(f"   {status_icon} {step.metadata.get('description', step.id)}")
            attempts = step.metadata.get("attempts", [])
            if len(attempts) > 1:
                lines.append(f"     ({len(attempts)} attempts)")
            if step.result:
                for wrapped in textwrap.wrap(step.result, width=panel_w - 7):
                    lines.append(f"     {wrapped}")

    # Wrap ALL lines first, then slice the visible window
    rendered = []
    for line in lines:
        if not line.strip():
            rendered.append("")  # blank spacer row
        else:
            for subline in textwrap.wrap(line, width=max(1, panel_w - 2)):
                rendered.append(subline)

    visible_rows = h - 2
    total = len(rendered)
    scroll_offset = max(0, min(scroll_offset, max(0, total - visible_rows)))

    for i, subline in enumerate(rendered[scroll_offset : scroll_offset + visible_rows]):
        if not subline:
            continue  # skip empty rows — addstr("") can error
        try:
            stdscr.addstr(i, panel_x + 1, subline[: panel_w - 2])
        except curses.error:
            pass

    # Scroll position indicator (only when content overflows)
    if total > visible_rows:
        pct = int(100 * scroll_offset / max(1, total - visible_rows))
        indicator = f" {pct}% "
        try:
            stdscr.addstr(0, w - len(indicator) - 1, indicator, curses.A_DIM)
        except curses.error:
            pass


def run_ui(
    orchestrator,
    run_dir: Path | None = None,
    repo_root: Path | None = None,
    restart_fn=None,
    git_proj_instance=None,
    config: UIConfig | None = None,
):
    """
    Run the curses DAG UI.

    If the user presses 'g', the startup screen is shown so they can pick a
    different goal or resume a previous run.  This requires ``repo_root`` and
    ``restart_fn`` to be supplied::

        run_ui(
            orchestrator,
            run_dir=run_dir,
            repo_root=REPO_ROOT,
            restart_fn=_init_system,   # callable(StartupChoice) -> (orch, run_dir)
        )

    Parameters
    ----------
    git_proj_instance:
        Optional ``GitProjection`` instance for this run.  When supplied,
        git operations use its isolated state rather than the module-level
        default, which prevents concurrent runs from clobbering each other.
        Falls back to the module-level default when ``None``.
    """
    from cuddlytoddly.ui.git_projection import GitProjection

    _git = git_proj_instance if isinstance(git_proj_instance, GitProjection) else None
    _repo_path = _git.repo_path if _git else git_proj.REPO_PATH

    root = logging.getLogger("dag")
    ch = getattr(root, "_stderr_handler", None)
    if ch:
        root.removeHandler(ch)

    import sys

    log_path = (run_dir / "logs" / "dag.log") if run_dir else Path("logs/dag.log")
    log_file = open(log_path, "a", encoding="utf-8", buffering=1)
    old_stderr = sys.stderr
    sys.stderr = log_file

    _filter = config.snapshot_filter_fn if config else None

    def _rebuild(graph):
        if _git:
            _git.rebuild_repo_from_graph(graph, snapshot_filter_fn=_filter)
        else:
            rebuild_repo_from_graph(graph, snapshot_filter_fn=_filter)

    try:
        while True:
            try:
                _rebuild(orchestrator.graph)
            except Exception as exc:
                logger.warning("[UI] Git pre-warm failed (non-fatal): %s", exc)
            result = curses.wrapper(dag_interface, orchestrator, run_dir, _repo_path, config)

            # Normal quit — nothing more to do.
            if result != "switch":
                break

            # User pressed 'g' — switch-goal is only possible when the caller
            # provided both repo_root and a restart_fn.
            if restart_fn is None or repo_root is None:
                break

            # Stop the current orchestrator gracefully before tearing down.
            try:
                orchestrator.stop_llm_calls()
            except Exception as exc:
                logger.warning("[UI] Could not stop orchestrator before switch: %s", exc)

            # Show the startup screen (runs its own curses.wrapper call).
            from cuddlytoddly.ui.curses_startup import run_startup_selection

            choice = run_startup_selection(repo_root)
            if choice is None:
                # User pressed Esc on the startup screen — just exit.
                break

            # Reopen the log file for the new run (run_dir may change).
            log_file.flush()

            # Initialise the new system.
            try:
                orchestrator, run_dir = restart_fn(choice, False)
            except Exception as exc:
                logger.error("[UI] restart_fn failed during goal switch: %s", exc, exc_info=True)
                break

            # Refresh repo_path for the new run so git_dag_text uses the right dir.
            _repo_path = _git.repo_path if _git else git_proj.REPO_PATH

            # Reopen log file pointed at the new run directory.
            log_file.close()
            new_log = (run_dir / "logs" / "dag.log") if run_dir else Path("logs/dag.log")
            log_file = open(new_log, "a", encoding="utf-8", buffering=1)
            sys.stderr = log_file

    finally:
        sys.stderr = old_stderr
        log_file.close()
        if ch:
            root.addHandler(ch)
