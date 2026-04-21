# ui/startup.py
"""
Shared startup logic — curses startup screen + shared data types.

Exports:
    StartupChoice      dataclass returned by the startup screen
    scan_runs()        list existing runs from the runs/ directory
    parse_manual_plan() freeform text -> (goal_text, events_list)
    run_startup_curses() blocking curses startup screen
"""

from __future__ import annotations

import curses
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------


@dataclass
class StartupChoice:
    mode: str  # "existing" | "new_goal" | "manual_plan"
    run_dir: Path
    goal_text: str
    plan_events: list = field(default_factory=list)
    is_fresh: bool = True


class RunInfo(NamedTuple):
    name: str
    path: str
    goal: str
    node_count: int
    mtime: float
    age: str


# ---------------------------------------------------------------------------
# Run scanner
# ---------------------------------------------------------------------------


def scan_runs(repo_root: Path) -> list[dict]:
    """Return one metadata dict per run that has a non-empty events.jsonl."""
    runs_dir = repo_root / "runs"
    if not runs_dir.exists():
        return []

    results = []
    for run_dir in sorted(runs_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not run_dir.is_dir():
            continue
        event_log = run_dir / "events.jsonl"
        if not event_log.exists() or event_log.stat().st_size == 0:
            continue

        goal_text = ""
        node_count = 0
        try:
            with event_log.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    evt = json.loads(line)
                    if evt.get("type") == "ADD_NODE":
                        node_count += 1
                        p = evt.get("payload", {})
                        if p.get("node_type") == "goal" and not goal_text:
                            goal_text = p.get("metadata", {}).get("description", "") or p.get(
                                "node_id", ""
                            )
        except Exception:
            pass

        mtime = run_dir.stat().st_mtime
        results.append(
            {
                "name": run_dir.name,
                "path": str(run_dir),
                "goal": goal_text or run_dir.name.replace("_", " "),
                "node_count": node_count,
                "mtime": mtime,
                "age": _human_age(mtime),
            }
        )

    return results


def _human_age(mtime: float) -> str:
    delta = time.time() - mtime
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta / 60)}m ago"
    if delta < 86400:
        return f"{int(delta / 3600)}h ago"
    return f"{int(delta / 86400)}d ago"


# ---------------------------------------------------------------------------
# Manual plan parser
# ---------------------------------------------------------------------------


def parse_manual_plan(text: str) -> tuple[str, list]:
    """
    Parse freeform plan text into (goal_text, events_list).

    Format:
        First non-bullet non-empty line  ->  goal description
        Lines starting with - * bullet  ->  tasks
        Dependency syntax: [depends: Task_A, Task_B]
                      or:  depends on: Task_A, Task_B  (trailing)

    Returns ("", []) on empty / unparseable input.
    """

    def to_id(s: str) -> str:
        return re.sub(r"[^a-zA-Z0-9]+", "_", s.strip()).strip("_")[:50]

    lines = text.strip().splitlines()
    goal_text = ""
    tasks: list[dict] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        is_task = line.startswith(("-", "*", "\u2022"))

        if not is_task and not goal_text:
            goal_text = re.sub(r"^[Gg]oal\s*:\s*", "", line).strip()
            continue

        if is_task:
            content = line.lstrip("-*\u2022 ").strip()

            # [depends: X, Y]  or  (depends on: X, Y)
            dep_match = re.search(
                r"[\[\(]depends?\s*(?:on)?\s*:\s*([^\]\)]+)[\]\)]",
                content,
                re.IGNORECASE,
            )
            deps_raw: list[str] = []
            if dep_match:
                deps_raw = [d.strip() for d in dep_match.group(1).split(",") if d.strip()]
                content = content[: dep_match.start()].strip()

            if not deps_raw:
                sfx = re.search(r"\s+depends?\s+on\s*:\s*(.+)$", content, re.IGNORECASE)
                if sfx:
                    deps_raw = [d.strip() for d in sfx.group(1).split(",") if d.strip()]
                    content = content[: sfx.start()].strip()

            # "Task ID: description"  or  "description"
            id_match = re.match(r"^([^:]{1,40}):\s*(.+)$", content)
            if id_match:
                task_id = to_id(id_match.group(1))
                task_desc = id_match.group(2).strip()
            else:
                task_desc = content
                task_id = to_id(content)

            tasks.append({"id": task_id, "desc": task_desc, "deps_raw": deps_raw})

    if not goal_text and tasks:
        goal_text = "Goal"
    if not goal_text:
        return "", []

    task_id_set = {t["id"] for t in tasks}
    desc_to_id = {t["desc"].lower(): t["id"] for t in tasks}

    def resolve_dep(raw: str) -> str | None:
        key = to_id(raw)
        if key in task_id_set:
            return key
        raw_lower = raw.lower().strip()
        for desc, tid in desc_to_id.items():
            if desc.startswith(raw_lower) or raw_lower.startswith(tid.lower()):
                return tid
        return None

    events: list[dict] = []
    for t in tasks:
        resolved = [r for raw in t["deps_raw"] if (r := resolve_dep(raw))]
        events.append(
            {
                "type": "ADD_NODE",
                "payload": {
                    "node_id": t["id"],
                    "node_type": "task",
                    "dependencies": resolved,
                    "metadata": {
                        "description": t["desc"],
                        "required_input": [],
                        "output": [],
                    },
                },
            }
        )

    goal_id = to_id(goal_text)
    depended_on = {r for t in tasks for raw in t["deps_raw"] if (r := resolve_dep(raw))}
    terminals = (task_id_set - depended_on) or task_id_set

    events.append(
        {
            "type": "ADD_NODE",
            "payload": {
                "node_id": goal_id,
                "node_type": "goal",
                "dependencies": [],
                "metadata": {
                    "description": goal_text,
                    "expanded": bool(tasks),
                },
            },
        }
    )
    for tid in terminals:
        events.append(
            {
                "type": "ADD_DEPENDENCY",
                "payload": {"node_id": goal_id, "depends_on": tid},
            }
        )

    return goal_text, events


# ---------------------------------------------------------------------------
# Curses startup screen
# ---------------------------------------------------------------------------


def run_startup_curses(
    repo_root: Path,
    issues: list[dict] | None = None,
) -> StartupChoice:
    """
    Show a full-screen curses startup dialog.
    Blocks until the user confirms a choice.
    Raises SystemExit(0) if the user presses q / Escape.

    *issues* is the list returned by ``config.preflight_check()`` — any
    non-empty list is rendered as a banner above the tab content so the
    user sees what needs fixing before they start a run.
    """
    result: list[StartupChoice] = []

    def _screen(stdscr):
        result.append(_startup_screen(stdscr, repo_root, issues=issues))

    curses.wrapper(_screen)
    return result[0]


def _startup_screen(
    stdscr,
    repo_root: Path,
    issues: list[dict] | None = None,
) -> StartupChoice:
    curses.start_color()
    curses.use_default_colors()
    curses.curs_set(0)
    stdscr.nodelay(False)
    stdscr.keypad(True)

    try:
        curses.init_pair(1, curses.COLOR_CYAN, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_WHITE, -1)
        curses.init_pair(5, curses.COLOR_BLACK, curses.COLOR_CYAN)
        curses.init_pair(6, curses.COLOR_RED, -1)  # errors
        curses.init_pair(7, curses.COLOR_YELLOW, -1)  # warnings (same as HI here)
    except Exception:
        pass

    ACCENT = curses.color_pair(1)
    SEL = curses.color_pair(2) | curses.A_BOLD
    HI = curses.color_pair(3)
    NORMAL = curses.color_pair(4)
    TAB_ON = curses.color_pair(5) | curses.A_BOLD
    TAB_OFF = NORMAL
    ERR = curses.color_pair(6) | curses.A_BOLD
    WARN = curses.color_pair(7) | curses.A_BOLD
    DIM = NORMAL | curses.A_DIM

    runs = scan_runs(repo_root)
    TABS = ["  Existing runs  ", "  New goal  ", "  Manual plan  "]
    tab = 0 if runs else 1

    _issues = issues or []

    # Per-tab state
    run_sel = 0
    goal_text = ""
    goal_cursor = 0
    plan_text = ""
    plan_cursor = 0
    error_msg = ""

    def _draw_banner(h, w) -> int:
        """
        Draw preflight issues below the separator (row 3).
        Returns the first row available for tab body content.
        """
        if not _issues:
            return 4
        y = 4
        for issue in _issues:
            if y >= h - 3:
                break
            is_err = issue["level"] == "error"
            prefix = "✗" if is_err else "⚠"
            attr = ERR if is_err else WARN
            line = f" {prefix} {issue['message']}"
            try:
                stdscr.addstr(y, 0, line[: w - 1], attr)
            except curses.error:
                pass
            y += 1
            fix = issue.get("fix", "")
            if fix and y < h - 3:
                try:
                    stdscr.addstr(y, 0, f"   → {fix}"[: w - 1], DIM)
                except curses.error:
                    pass
                y += 1
        # Separator after banner
        try:
            stdscr.addstr(y, 0, "─" * (w - 1), NORMAL)
        except curses.error:
            pass
        return y + 1

    def _draw():
        stdscr.erase()
        h, w = stdscr.getmaxyx()

        # Title bar
        try:
            stdscr.addstr(0, 0, " cuddlytoddly — startup ".center(w), ACCENT | curses.A_BOLD)
        except curses.error:
            pass

        # Tab bar
        x = 2
        for i, label in enumerate(TABS):
            try:
                stdscr.addstr(2, x, label, TAB_ON if i == tab else TAB_OFF)
            except curses.error:
                pass
            x += len(label) + 2

        # Separator after tabs
        try:
            stdscr.addstr(3, 0, "─" * (w - 1), NORMAL)
        except curses.error:
            pass

        # Preflight banner (may push body_top down)
        body_top = _draw_banner(h, w)
        body_h = h - body_top - 3

        # ── Tab 0: existing runs ─────────────────────────────────────────────
        if tab == 0:
            if not runs:
                try:
                    stdscr.addstr(body_top + 1, 4, "No existing runs found.", NORMAL)
                    stdscr.addstr(body_top + 2, 4, "Press Tab or → to start a new goal.", NORMAL)
                except curses.error:
                    pass
            else:
                for i, run in enumerate(runs[:body_h]):
                    y = body_top + i
                    attr = SEL if i == run_sel else NORMAL
                    ptr = "▶ " if i == run_sel else "  "
                    age = run["age"].rjust(10)
                    nc = f"({run['node_count']} nodes)"
                    line = f"{ptr}{run['goal'][: w - 30]} {nc:>14} {age}"
                    try:
                        stdscr.addstr(y, 2, line[: w - 2], attr)
                    except curses.error:
                        pass

        # ── Tab 1: new goal ──────────────────────────────────────────────────
        elif tab == 1:
            try:
                stdscr.addstr(body_top, 4, "Goal description:", ACCENT)
                stdscr.addstr(body_top + 1, 4, "─" * min(60, w - 6), NORMAL)
                stdscr.addstr(body_top + 2, 4, (goal_text or " ")[: w - 6], HI | curses.A_REVERSE)
                stdscr.addstr(body_top + 5, 4, "Type the goal then press Enter.", NORMAL)
            except curses.error:
                pass

        # ── Tab 2: manual plan ───────────────────────────────────────────────
        elif tab == 2:
            instructions = [
                "First line = goal.  Lines starting with  -  are tasks.",
                "Dependency syntax:  - Task_B: desc [depends: Task_A]",
                "Press Ctrl+G to confirm.",
            ]
            for i, ins in enumerate(instructions):
                try:
                    stdscr.addstr(body_top + i, 4, ins[: w - 6], NORMAL)
                except curses.error:
                    pass
            try:
                stdscr.addstr(body_top + len(instructions), 4, "─" * min(60, w - 6), NORMAL)
            except curses.error:
                pass

            area_top = body_top + len(instructions) + 1
            area_h = body_h - len(instructions) - 2
            plan_lines = plan_text.splitlines() or [""]
            for i, pline in enumerate(plan_lines[:area_h]):
                try:
                    stdscr.addstr(area_top + i, 4, pline[: w - 6], NORMAL)
                except curses.error:
                    pass

            # Draw cursor
            cur_line = plan_text[:plan_cursor].count("\n")
            cur_col = len(plan_text[:plan_cursor].rsplit("\n", 1)[-1])
            abs_y = area_top + cur_line
            if 0 <= abs_y < h - 1:
                split = plan_text.splitlines()
                ch = (
                    split[cur_line][cur_col]
                    if cur_line < len(split) and cur_col < len(split[cur_line])
                    else " "
                )
                try:
                    stdscr.addstr(abs_y, 4 + cur_col, ch, curses.A_REVERSE)
                except curses.error:
                    pass

        # Error line (validation errors from key handling)
        if error_msg:
            try:
                stdscr.addstr(h - 3, 2, f"! {error_msg}"[: w - 2], HI)
            except curses.error:
                pass

        # Footer
        try:
            stdscr.addstr(h - 2, 0, "─" * (w - 1), NORMAL)
            stdscr.addstr(
                h - 1,
                0,
                "Tab/←/→: switch  ↑/↓: navigate  Enter: confirm  q: quit"[: w - 1],
                NORMAL,
            )
        except curses.error:
            pass

        stdscr.refresh()

    def _make_run_dir(goal: str) -> "Path":
        from cuddlytoddly.__main__ import make_run_dir

        return make_run_dir(goal).resolve()

    while True:
        _draw()
        k = stdscr.getch()

        if k in (ord("q"), 27):
            raise SystemExit(0)

        if k in (9, curses.KEY_RIGHT):
            tab = (tab + 1) % len(TABS)
            error_msg = ""
            continue

        if k == curses.KEY_LEFT:
            tab = (tab - 1) % len(TABS)
            error_msg = ""
            continue

        if tab == 0:
            if k == curses.KEY_UP:
                run_sel = max(0, run_sel - 1)
            elif k == curses.KEY_DOWN:
                run_sel = min(len(runs) - 1, run_sel + 1)
            elif k in (10, 13) and runs:
                r = runs[run_sel]
                return StartupChoice(
                    mode="existing",
                    run_dir=Path(r["path"]),
                    goal_text=r["goal"],
                    is_fresh=False,
                )

        elif tab == 1:
            if k in (10, 13):
                if goal_text.strip():
                    return StartupChoice(
                        mode="new_goal",
                        run_dir=_make_run_dir(goal_text.strip()),
                        goal_text=goal_text.strip(),
                        is_fresh=True,
                    )
                error_msg = "Goal cannot be empty."
            elif k in (curses.KEY_BACKSPACE, 127):
                if goal_cursor > 0:
                    goal_text = goal_text[: goal_cursor - 1] + goal_text[goal_cursor:]
                    goal_cursor -= 1
            elif k == curses.KEY_LEFT:
                goal_cursor = max(0, goal_cursor - 1)
            elif k == curses.KEY_RIGHT:
                goal_cursor = min(len(goal_text), goal_cursor + 1)
            elif 32 <= k <= 126:
                goal_text = goal_text[:goal_cursor] + chr(k) + goal_text[goal_cursor:]
                goal_cursor += 1

        elif tab == 2:
            if k == 7:  # Ctrl+G — submit
                if plan_text.strip():
                    gt, evts = parse_manual_plan(plan_text)
                    if gt:
                        return StartupChoice(
                            mode="manual_plan",
                            run_dir=_make_run_dir(gt),
                            goal_text=gt,
                            plan_events=evts,
                            is_fresh=True,
                        )
                    error_msg = "Could not parse plan — add a goal line first."
                else:
                    error_msg = "Plan is empty."
            elif k in (curses.KEY_BACKSPACE, 127):
                if plan_cursor > 0:
                    plan_text = plan_text[: plan_cursor - 1] + plan_text[plan_cursor:]
                    plan_cursor -= 1
            elif k in (10, 13):
                plan_text = plan_text[:plan_cursor] + "\n" + plan_text[plan_cursor:]
                plan_cursor += 1
            elif 32 <= k <= 126:
                plan_text = plan_text[:plan_cursor] + chr(k) + plan_text[plan_cursor:]
                plan_cursor += 1


def build_manual_plan_events(goal_id: str, goal_text: str, tasks: list) -> list:
    """
    Compatibility shim for callers that pre-parse tasks themselves.
    Prefer parse_manual_plan() for new code.
    """
    events = []
    for t in tasks:
        events.append(
            {
                "type": "ADD_NODE",
                "payload": {
                    "node_id": t["node_id"],
                    "node_type": "task",
                    "dependencies": t.get("dependencies", []),
                    "metadata": {
                        "description": t.get("description", ""),
                        "required_input": [],
                        "output": [],
                    },
                },
            }
        )

    task_ids = {t["node_id"] for t in tasks}
    depended_on = {dep for t in tasks for dep in t.get("dependencies", [])}
    terminals = (task_ids - depended_on) or task_ids

    events.append(
        {
            "type": "ADD_NODE",
            "payload": {
                "node_id": goal_id,
                "node_type": "goal",
                "dependencies": [],
                "metadata": {"description": goal_text, "expanded": bool(tasks)},
            },
        }
    )
    for tid in terminals:
        events.append(
            {
                "type": "ADD_DEPENDENCY",
                "payload": {"node_id": goal_id, "depends_on": tid},
            }
        )

    return events
