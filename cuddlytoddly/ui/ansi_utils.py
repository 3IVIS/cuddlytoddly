"""
cuddlytoddly/ui/ansi_utils.py

ANSI escape-code parsing and git-graph rendering helpers.
All functions here are pure renderers — they read curses constants but
never mutate a curses window directly.
"""

from __future__ import annotations

import curses
import hashlib
import re

from toddly.infra.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Colour map: ANSI code → curses colour constant
# ---------------------------------------------------------------------------

ANSI_COLOR_MAP = {
    30: curses.COLOR_BLACK,
    31: curses.COLOR_RED,
    32: curses.COLOR_GREEN,
    33: curses.COLOR_YELLOW,
    34: curses.COLOR_BLUE,
    35: curses.COLOR_MAGENTA,
    36: curses.COLOR_CYAN,
    37: curses.COLOR_WHITE,
    90: curses.COLOR_BLACK,
    91: curses.COLOR_RED,
    92: curses.COLOR_GREEN,
    93: curses.COLOR_YELLOW,
    94: curses.COLOR_BLUE,
    95: curses.COLOR_MAGENTA,
    96: curses.COLOR_CYAN,
    97: curses.COLOR_WHITE,
}

# ---------------------------------------------------------------------------
# ANSI parsing
# ---------------------------------------------------------------------------

ansi_regex = re.compile(r"\x1b\[[0-9;]*m")


def parse_ansi(line):
    parts = []

    current_color = curses.COLOR_WHITE
    bold = False
    attr = curses.color_pair(0)

    idx = 0
    for match in ansi_regex.finditer(line):
        # plain text before escape
        while idx < match.start():
            parts.append((line[idx], attr))
            idx += 1

        codes = match.group()[2:-1].split(";")
        if codes == [""]:
            codes = ["0"]

        for code in codes:
            code = int(code)
            if code == 0:
                current_color = curses.COLOR_WHITE
                bold = False
            elif code == 1:
                bold = True
            elif 30 <= code <= 37:
                current_color = code - 30
                bold = False
            elif 90 <= code <= 97:
                current_color = code - 90
                bold = True

        attr = curses.color_pair(current_color + 1)
        if bold:
            attr |= curses.A_BOLD

        idx = match.end()

    while idx < len(line):
        parts.append((line[idx], attr))
        idx += 1

    return parts


def strip_ansi(line):
    return ansi_regex.sub("", line)


# ---------------------------------------------------------------------------
# Mapping: git-log lines ↔ snapshot nodes
# ---------------------------------------------------------------------------


def map_nodes_to_lines(git_lines, snapshot):
    # Pre-compute the hash suffix for every node_id
    hash_to_node_id = {
        "#" + hashlib.sha256(node_id.encode()).hexdigest()[:6]: node_id for node_id in snapshot
    }

    node_map = {}
    for i, line in enumerate(git_lines):
        clean = strip_ansi(line)
        if "*" not in clean:
            continue
        star_pos = clean.index("*")
        after_star = clean[star_pos + 1 :]
        message = after_star.lstrip(" |\\/.-")
        if not message:
            continue

        m = re.search(r"(#[0-9a-f]{6})", message)
        if m:
            node_id = hash_to_node_id.get(m.group(1))
            if node_id:
                node_map[node_id] = i

    return node_map


def get_node_col(line):
    parsed = parse_ansi(line)
    for idx, (ch, _) in enumerate(parsed):
        if ch == "*":
            return idx
    return 0


# ---------------------------------------------------------------------------
# Branch-path tracer
# ---------------------------------------------------------------------------


def trace_branch_path_recursive(
    git_lines,
    row,
    col,
    child_row,
    child_col,
    step=None,
    visited=None,
    is_start=True,
    debug=False,
    _depth=0,
):
    """
    Recursively trace a visual path from (row, col) to (child_row, child_col)
    following the \\ | / characters produced by ``git log --graph``, stopping
    when another ``*`` node marker is encountered.

    Returns a set of (row, col) positions that form the highlighted path.
    When a dead-end or blocking ``*`` is detected the sentinel ``{"x"}`` is
    returned so callers can detect that this branch did not reach the target.

    ``_depth`` guards against hitting Python's recursion limit on very long git
    logs — the function returns an empty set gracefully when the limit is hit.
    """
    _MAX_DEPTH = 800  # comfortably below Python's default recursion limit of 1000

    if _depth > _MAX_DEPTH:
        logger.debug("[UI] trace_branch_path_recursive hit depth limit at (%d,%d)", row, col)
        return set()

    if visited is None:
        visited = set()
    path_positions = set()

    if step is None:
        step = 1 if child_row > row else -1

    this_line = "".join(ch for ch, _ in parse_ansi(git_lines[row]))

    # Out of bounds
    if row < 0 or row >= len(git_lines) or col < 0 or col >= len(this_line):
        return path_positions

    char = this_line[col]

    char_matrix = {}

    char_matrix[(0, +2)] = this_line[col + 2] if col + 2 < len(this_line) else ""
    char_matrix[(0, +1)] = this_line[col + 1] if col + 1 < len(this_line) else ""
    char_matrix[(0, 0)] = this_line[col] if col < len(this_line) else ""
    char_matrix[(0, -1)] = this_line[col - 1] if (0 <= col - 1 < len(this_line)) else ""
    char_matrix[(0, -2)] = this_line[col - 2] if (0 <= col - 2 < len(this_line)) else ""

    # Stop if we hit a '*' that is not the child (and not the start)
    if char == "*" and not (row == child_row and col == child_col) and not is_start:
        return path_positions | set("x")

    # Mark visited and add current cell
    if (row, col) in visited:
        return path_positions
    visited.add((row, col))
    path_positions.add((row, col))

    # Stop if reached child
    if row == child_row and col == child_col:
        return path_positions

    subpath_positions = []

    # Explore next row in step direction
    next_row = row + step
    if 0 <= next_row < len(git_lines):
        next_line = "".join(ch for ch, _ in parse_ansi(git_lines[next_row]))

        for dcol in range(-2, 3):
            char_matrix[(1, dcol)] = (
                next_line[col + dcol] if (0 <= col + dcol < len(next_line)) else ""
            )

        for dcol in range(-2, 3):
            ncol = col + dcol
            if 0 <= ncol < len(next_line):
                if (
                    (dcol == 1 and char_matrix[(1, 1)] == "/" and char_matrix[(0, 0)] == "/")
                    or (dcol == 1 and char_matrix[(1, 1)] == "/" and char_matrix[(0, 0)] == "*")
                    or (
                        dcol == 1
                        and char_matrix[(1, 1)] == "\\"
                        and char_matrix[(0, 0)] == "/"
                        and char_matrix[(1, 0)] == " "
                    )
                    or (
                        dcol == 1
                        and char_matrix[(1, 1)] == "/"
                        and char_matrix[(0, 0)] == "|"
                        and char_matrix[(1, 0)] == "|"
                        and char_matrix[(0, -1)] != "/"
                        and char_matrix[(1, -1)] != "_"
                    )
                    or (
                        dcol == 1
                        and char_matrix[(1, 1)] == "/"
                        and char_matrix[(0, 0)] == "|"
                        and char_matrix[(1, 0)] == " "
                    )
                    or (
                        dcol == 1
                        and char_matrix[(1, 1)] == "|"
                        and char_matrix[(0, 0)] == "/"
                        and char_matrix[(1, 2)] != "/"
                        and char_matrix[(1, 2)] != "_"
                    )
                    or (
                        dcol == 0
                        and char_matrix[(1, 0)] == "|"
                        and char_matrix[(0, 0)] == "*"
                        and char_matrix[(1, 1)] == "/"
                    )
                    or (dcol == 1 and char_matrix[(1, 1)] == "*" and char_matrix[(0, 0)] == "/")
                    or (dcol == 0 and char_matrix[(1, 0)] == "|" and char_matrix[(0, 0)] == "*")
                    or (dcol == 0 and char_matrix[(1, 0)] == "|" and char_matrix[(0, 0)] == "/")
                    or (dcol == 0 and char_matrix[(1, 0)] == "|" and char_matrix[(0, 0)] == "\\")
                    or (dcol == 0 and char_matrix[(1, 0)] == "|" and char_matrix[(0, 0)] == "|")
                    or (dcol == 0 and char_matrix[(1, 0)] == "\\" and char_matrix[(0, 0)] == "|")
                    or (
                        dcol == 0
                        and char_matrix[(1, 0)] == "\\"
                        and char_matrix[(0, 0)] == "/"
                        and char_matrix[(1, 1)] != "/"
                    )
                    or (dcol == 0 and char_matrix[(1, 0)] == "*" and char_matrix[(0, 0)] == "|")
                    or (
                        dcol == -1
                        and char_matrix[(1, -1)] == "|"
                        and char_matrix[(0, 0)] == "\\"
                        and char_matrix[(1, 0)] == " "
                        and char_matrix[(1, -2)] != "\\"
                    )
                    or (dcol == -1 and char_matrix[(1, -1)] == "\\" and char_matrix[(0, 0)] == "\\")
                    or (dcol == -1 and char_matrix[(1, -1)] == "*" and char_matrix[(0, 0)] == "\\")
                    or (
                        dcol == -1
                        and char_matrix[(1, -1)] == "\\"
                        and char_matrix[(0, 0)] == "|"
                        and char_matrix[(0, 1)] != "\\"
                    )
                    or (
                        dcol == -1
                        and char_matrix[(1, -1)] == "|"
                        and char_matrix[(0, 0)] == "\\"
                        and char_matrix[(1, -2)] != "\\"
                    )
                    or (dcol == -1 and char_matrix[(1, -1)] == "\\" and char_matrix[(0, 0)] == "*")
                    or (dcol == -1 and char_matrix[(1, -1)] == "*" and char_matrix[(0, 0)] == "\\")
                ):
                    subpath_positions += [
                        trace_branch_path_recursive(
                            git_lines,
                            next_row,
                            ncol,
                            child_row,
                            child_col,
                            step,
                            visited,
                            is_start=False,
                            debug=debug,
                            _depth=_depth + 1,
                        )
                    ]
        if (
            char_matrix[(1, 2)] in ["_", "/"]
            and char_matrix[(0, 0)] == "/"
            and char_matrix[(0, 1)] == "|"
            and char_matrix[(1, 1)] == "|"
        ):
            temp_set = set()
            dcol = 2
            while char_matrix[(1, dcol)] == "_" and char_matrix[(1, dcol - 1)] == "|":
                temp_set.add((next_row, col + dcol))
                dcol += 2
                char_matrix[(1, dcol)] = (
                    next_line[col + dcol] if (0 <= col + dcol < len(next_line)) else ""
                )
                char_matrix[(1, dcol - 1)] = (
                    next_line[col + dcol - 1] if (0 <= col + dcol - 1 < len(next_line)) else ""
                )

            ncol = col + dcol
            subpath_positions += [
                trace_branch_path_recursive(
                    git_lines,
                    next_row,
                    ncol,
                    child_row,
                    child_col,
                    step,
                    visited,
                    is_start=False,
                    debug=debug,
                    _depth=_depth + 1,
                )
                | temp_set
            ]

        if char_matrix[(1, -1)] in [".", "-"] and char_matrix[(0, 0)] == "\\":
            temp_set = set()
            dcol = -1
            while char_matrix[(1, dcol)] in [".", "-"]:
                temp_set.add((next_row, col + dcol))
                dcol -= 1
                char_matrix[(1, dcol)] = (
                    next_line[col + dcol] if (0 <= col + dcol < len(next_line)) else ""
                )

            ncol = col + dcol
            subpath_positions += [
                trace_branch_path_recursive(
                    git_lines,
                    next_row,
                    ncol,
                    child_row,
                    child_col,
                    step,
                    visited,
                    is_start=False,
                    debug=debug,
                    _depth=_depth + 1,
                )
                | temp_set
            ]

    if len(subpath_positions) > 1:
        nothing_added = True
        for subpath_position in subpath_positions:
            if "x" not in subpath_position:
                nothing_added = False
                path_positions |= subpath_position
        if nothing_added:
            path_positions |= set("x")
    elif len(subpath_positions) == 1:
        path_positions |= subpath_positions[0]

    return path_positions
