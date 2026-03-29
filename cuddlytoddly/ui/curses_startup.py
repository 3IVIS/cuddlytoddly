# ui/curses_startup.py
"""
Curses startup screen shown before the main DAG UI.

Usage in __main__.py:
    from cuddlytoddly.ui.curses_startup import run_startup_selection
    choice = run_startup_selection(repo_root)
    # choice is a StartupChoice namedtuple
"""
from __future__ import annotations

import curses
import textwrap
from pathlib import Path

from cuddlytoddly.ui.startup import (
    StartupChoice, RunInfo, scan_runs, parse_manual_plan,
)


# ── Colour pairs (set up once inside curses.wrapper) ─────────────────────────
_C_TITLE   = 1
_C_SEL     = 2
_C_DIM     = 3
_C_DONE    = 4
_C_ACCENT  = 5
_C_ERR     = 6


def _init_colors():
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(_C_TITLE,  curses.COLOR_CYAN,    -1)
    curses.init_pair(_C_SEL,    curses.COLOR_BLACK,   curses.COLOR_CYAN)
    curses.init_pair(_C_DIM,    curses.COLOR_WHITE,   -1)
    curses.init_pair(_C_DONE,   curses.COLOR_GREEN,   -1)
    curses.init_pair(_C_ACCENT, curses.COLOR_YELLOW,  -1)
    curses.init_pair(_C_ERR,    curses.COLOR_RED,     -1)


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_addstr(win, y, x, text, attr=0):
    h, w = win.getmaxyx()
    if y < 0 or y >= h or x < 0 or x >= w:
        return
    max_len = w - x - 1
    if max_len <= 0:
        return
    try:
        win.addstr(y, x, text[:max_len], attr)
    except curses.error:
        pass


def _center(win, y, text, attr=0):
    _, w = win.getmaxyx()
    x = max(0, (w - len(text)) // 2)
    _safe_addstr(win, y, x, text, attr)


def _hline(win, y, char="─"):
    _, w = win.getmaxyx()
    try:
        win.addstr(y, 0, char * (w - 1))
    except curses.error:
        pass


# ── Tab 1 — Resume existing run ───────────────────────────────────────────────

def _draw_resume_tab(win, runs: list[RunInfo], sel: int, scroll: int):
    h, w = win.getmaxyx()
    y = 0
    _safe_addstr(win, y, 2, "Existing runs  (↑↓ select, Enter resume)", curses.color_pair(_C_DIM))
    y += 1
    _hline(win, y)
    y += 1

    visible = h - y - 3
    for i, run in enumerate(runs[scroll: scroll + visible]):
        idx    = i + scroll
        is_sel = idx == sel
        attr   = curses.color_pair(_C_SEL) | curses.A_BOLD if is_sel else 0

        date   = run.age
        label  = f"  {run.goal[:w - 30]}".ljust(w - 28)
        stats  = f"{run.node_count} nodes  {date}  "

        try:
            win.addstr(y + i, 0, label[:w - len(stats) - 1].ljust(w - len(stats) - 1), attr)
            win.addstr(y + i, w - len(stats) - 1, stats,
                       curses.color_pair(_C_DONE) if not is_sel else attr)
        except curses.error:
            pass

    if not runs:
        _center(win, y + 2, "No existing runs found.", curses.color_pair(_C_DIM) | curses.A_DIM)


# ── Tab 2 — New goal ──────────────────────────────────────────────────────────

def _draw_new_goal_tab(win, text: str, cursor: int, error: str):
    h, w = win.getmaxyx()
    y = 0
    _safe_addstr(win, y, 2, "New goal  (type goal, Enter to start)", curses.color_pair(_C_DIM))
    y += 1
    _hline(win, y); y += 1

    _safe_addstr(win, y, 2, "Goal:", curses.color_pair(_C_ACCENT))
    y += 1

    # Wrap the text into the available width
    box_w  = w - 6
    lines  = textwrap.wrap(text, box_w) if text else [""]
    # Find cursor position
    cur_line, cur_col = _cursor_pos(text, cursor, box_w)

    for li, line in enumerate(lines[:h - y - 4]):
        is_cur_line = li == cur_line
        attr = curses.A_REVERSE if is_cur_line else curses.color_pair(_C_DIM)
        _safe_addstr(win, y + li, 4, line.ljust(box_w), attr)
    y += max(len(lines), 1) + 1

    if error:
        _safe_addstr(win, y, 2, f"! {error}", curses.color_pair(_C_ERR))


def _cursor_pos(text: str, cursor: int, wrap_w: int) -> tuple[int, int]:
    """Return (line_index, col_index) of cursor in wrapped text."""
    before  = text[:cursor]
    wrapped = textwrap.wrap(before, wrap_w) if before else [""]
    li      = len(wrapped) - 1
    col     = len(wrapped[-1]) if wrapped else 0
    return li, col


# ── Tab 3 — Manual plan ───────────────────────────────────────────────────────

_MANUAL_PLACEHOLDER = """\
task: Task_One
desc: First thing to do

task: Task_Two
desc: Second thing to do
deps: Task_One"""

_MANUAL_HELP = [
    "Format:",
    "  First lines (before any 'task:') = goal description",
    "  task: Task_ID",
    "  desc: One sentence description",
    "  deps: Dep1, Dep2   (optional)",
]


def _draw_manual_tab(win, goal_text: str, goal_cursor: int,
                      plan_text: str, plan_cursor: int,
                      active_field: int, error: str):
    h, w = win.getmaxyx()
    y = 0
    _safe_addstr(win, y, 2, "Manual plan  (Tab: switch fields, Enter: confirm)", curses.color_pair(_C_DIM))
    y += 1; _hline(win, y); y += 1

    # Goal field
    goal_attr = curses.A_REVERSE if active_field == 0 else curses.color_pair(_C_ACCENT)
    _safe_addstr(win, y, 2, "Goal:", goal_attr)
    y += 1
    box_w = w - 6
    goal_disp = goal_text or "(enter goal description)"
    _safe_addstr(win, y, 4, goal_disp[:box_w].ljust(box_w),
                 curses.A_REVERSE if active_field == 0 else curses.color_pair(_C_DIM))
    y += 2

    # Plan textarea
    plan_attr = curses.A_REVERSE if active_field == 1 else curses.color_pair(_C_ACCENT)
    _safe_addstr(win, y, 2, "Task breakdown:", plan_attr)
    y += 1

    available = h - y - 4
    plan_lines = (plan_text or _MANUAL_PLACEHOLDER).splitlines()
    for li, pline in enumerate(plan_lines[:available]):
        is_active = active_field == 1
        attr = curses.color_pair(_C_DIM) if not is_active else 0
        _safe_addstr(win, y + li, 4, pline[:box_w].ljust(box_w), attr)
    y += min(len(plan_lines), available) + 1

    if error:
        _safe_addstr(win, min(y, h - 3), 2, f"! {error}", curses.color_pair(_C_ERR))

    for hi, hline in enumerate(_MANUAL_HELP):
        _safe_addstr(win, h - len(_MANUAL_HELP) + hi - 1, 2,
                     hline, curses.color_pair(_C_DIM) | curses.A_DIM)


# ── Tab bar ───────────────────────────────────────────────────────────────────

_TABS = ["Resume run", "New goal", "Manual plan"]


def _draw_tab_bar(win, active_tab: int):
    h, w = win.getmaxyx()
    # Title
    _center(win, 0, "── cuddlytoddly ──", curses.color_pair(_C_TITLE) | curses.A_BOLD)
    y = 2
    x = 2
    for i, name in enumerate(_TABS):
        label = f"  {name}  "
        attr  = (curses.color_pair(_C_SEL) | curses.A_BOLD) if i == active_tab \
                else curses.color_pair(_C_DIM)
        _safe_addstr(win, y, x, label, attr)
        x += len(label) + 1
    _hline(win, y + 1)

    # Footer
    _safe_addstr(win, h - 1, 2,
                 "Tab/←/→: switch tabs   ↑↓: navigate   Enter: confirm   Esc: quit",
                 curses.color_pair(_C_DIM) | curses.A_DIM)


# ── Main startup screen ───────────────────────────────────────────────────────

def _startup_screen(stdscr, repo_root: Path) -> StartupChoice | None:
    curses.curs_set(0)
    stdscr.nodelay(False)
    _init_colors()

    _raw_runs   = scan_runs(repo_root)
    runs        = [RunInfo(**r) if isinstance(r, dict) else r for r in _raw_runs]
    active_tab  = 0 if runs else 1

    # Tab 1 state
    resume_sel    = 0
    resume_scroll = 0

    # Tab 2 state
    goal_text   = ""
    goal_cursor = 0
    goal_error  = ""

    # Tab 3 state
    manual_goal_text   = ""
    manual_goal_cursor = 0
    manual_plan_text   = ""
    manual_plan_cursor = 0
    manual_active_fld  = 0   # 0=goal, 1=plan
    manual_error       = ""

    while True:
        h, w = stdscr.getmaxyx()
        stdscr.erase()

        _draw_tab_bar(stdscr, active_tab)

        # Content area starts at row 4
        content_h = h - 5
        content_w = w

        # Use a subwindow for cleaner rendering
        try:
            content = stdscr.derwin(content_h, content_w, 4, 0)
        except curses.error:
            stdscr.refresh()
            k = stdscr.getch()
            continue

        content.erase()

        if active_tab == 0:
            _draw_resume_tab(content, runs, resume_sel, resume_scroll)
        elif active_tab == 1:
            _draw_new_goal_tab(content, goal_text, goal_cursor, goal_error)
        elif active_tab == 2:
            _draw_manual_tab(content,
                             manual_goal_text, manual_goal_cursor,
                             manual_plan_text, manual_plan_cursor,
                             manual_active_fld, manual_error)

        stdscr.refresh()
        content.refresh()

        k = stdscr.getch()

        # ── Global tab switching ──────────────────────────────────────────────
        if k == 27:   # Escape → quit
            return None

        if k == curses.KEY_LEFT or (k == ord('\t') and active_tab == 0):
            active_tab = (active_tab - 1) % len(_TABS)
            continue
        if k == curses.KEY_RIGHT:
            active_tab = (active_tab + 1) % len(_TABS)
            continue
        # Tab key cycles forward through tabs when not in a text field
        if k == ord('\t') and active_tab != 2:
            active_tab = (active_tab + 1) % len(_TABS)
            continue

        # ── Tab-specific key handling ─────────────────────────────────────────
        if active_tab == 0:
            if k == curses.KEY_UP:
                resume_sel    = max(0, resume_sel - 1)
                resume_scroll = max(0, min(resume_scroll, resume_sel))
            elif k == curses.KEY_DOWN:
                resume_sel    = min(len(runs) - 1, resume_sel + 1)
                visible       = content_h - 4
                if resume_sel >= resume_scroll + visible:
                    resume_scroll = resume_sel - visible + 1
            elif k in (10, 13) and runs:
                run = runs[resume_sel]
                return StartupChoice(
                    mode      = "resume",
                    run_dir   = Path(run.path),
                    goal_text = run.goal,
                    is_fresh  = False,
                )

        elif active_tab == 1:
            goal_error = ""
            if k in (10, 13):
                gt = goal_text.strip()
                if not gt:
                    goal_error = "Goal cannot be empty."
                else:
                    return StartupChoice(
                        mode      = "new_goal",
                        run_dir   = None,
                        goal_text = gt,
                        is_fresh  = True,
                    )
            elif k == curses.KEY_BACKSPACE or k == 127:
                if goal_cursor > 0:
                    goal_text   = goal_text[:goal_cursor - 1] + goal_text[goal_cursor:]
                    goal_cursor -= 1
            elif k == curses.KEY_LEFT:
                goal_cursor = max(0, goal_cursor - 1)
            elif k == curses.KEY_RIGHT:
                goal_cursor = min(len(goal_text), goal_cursor + 1)
            elif 32 <= k <= 126:
                ch          = chr(k)
                goal_text   = goal_text[:goal_cursor] + ch + goal_text[goal_cursor:]
                goal_cursor += 1

        elif active_tab == 2:
            manual_error = ""
            if k == ord('\t'):
                # Tab switches between goal and plan fields
                manual_active_fld = 1 - manual_active_fld
            elif k in (10, 13) and manual_active_fld == 1:
                # Enter in plan field = confirm
                gt   = manual_goal_text.strip()
                plan = manual_plan_text.strip()
                if not gt:
                    manual_error = "Goal cannot be empty."
                elif not plan:
                    manual_error = "Plan cannot be empty."
                else:
                    _, tasks = parse_manual_plan(plan)
                    if not tasks:
                        manual_error = "No tasks found. Use 'task: Name' lines."
                    else:
                        return StartupChoice(
                            mode      = "manual_plan",
                            run_dir   = None,
                            goal_text = gt,
                            plan_events = tasks,
                            is_fresh    = True,
                        )
            elif manual_active_fld == 0:
                # Editing goal field
                if k == curses.KEY_BACKSPACE or k == 127:
                    if manual_goal_cursor > 0:
                        manual_goal_text   = manual_goal_text[:manual_goal_cursor - 1] + manual_goal_text[manual_goal_cursor:]
                        manual_goal_cursor -= 1
                elif k == curses.KEY_LEFT:
                    manual_goal_cursor = max(0, manual_goal_cursor - 1)
                elif k == curses.KEY_RIGHT:
                    manual_goal_cursor = min(len(manual_goal_text), manual_goal_cursor + 1)
                elif k in (10, 13):
                    manual_active_fld = 1   # move to plan
                elif 32 <= k <= 126:
                    ch                 = chr(k)
                    manual_goal_text   = manual_goal_text[:manual_goal_cursor] + ch + manual_goal_text[manual_goal_cursor:]
                    manual_goal_cursor += 1
            else:
                # Editing plan textarea — newlines allowed
                if k == curses.KEY_BACKSPACE or k == 127:
                    if manual_plan_cursor > 0:
                        manual_plan_text   = manual_plan_text[:manual_plan_cursor - 1] + manual_plan_text[manual_plan_cursor:]
                        manual_plan_cursor -= 1
                elif k == curses.KEY_LEFT:
                    manual_plan_cursor = max(0, manual_plan_cursor - 1)
                elif k == curses.KEY_RIGHT:
                    manual_plan_cursor = min(len(manual_plan_text), manual_plan_cursor + 1)
                elif k in (10, 13):
                    manual_plan_text   = manual_plan_text[:manual_plan_cursor] + "\n" + manual_plan_text[manual_plan_cursor:]
                    manual_plan_cursor += 1
                elif 32 <= k <= 126:
                    ch                 = chr(k)
                    manual_plan_text   = manual_plan_text[:manual_plan_cursor] + ch + manual_plan_text[manual_plan_cursor:]
                    manual_plan_cursor += 1


def run_startup_selection(repo_root: Path) -> StartupChoice | None:
    """
    Show the startup screen and return the user's choice.
    Returns None if the user presses Escape (quit).
    Runs in its own curses.wrapper call so it finishes cleanly before
    the main UI starts.
    """
    import sys, logging

    # Silence stderr during curses — same pattern as run_ui
    root = logging.getLogger("dag")
    ch   = getattr(root, "_stderr_handler", None)
    if ch:
        root.removeHandler(ch)

    result: list[StartupChoice | None] = [None]

    def _inner(stdscr):
        result[0] = _startup_screen(stdscr, repo_root)

    try:
        curses.wrapper(_inner)
    finally:
        if ch:
            root.addHandler(ch)

    return result[0]
