"""
Curses UI wired to Planner Runtime

Design:
- UI never mutates TaskGraph directly
- All edits emit Events into EventQueue
- Graph state is read via snapshot
- Git repo is rebuilt from TaskGraph snapshot
- Rendering logic is minimally modified
"""

import curses
import subprocess
import re
import time
import sys

from collections import deque, defaultdict
from cuddlytoddly.core.reducer import apply_event
from cuddlytoddly.ui.git_projection import (
    rebuild_repo_from_graph,
    graph_to_dag,
)
import textwrap
import logging
import hashlib

from collections import deque

from cuddlytoddly.core.events import (
    Event,
    ADD_NODE,
    REMOVE_NODE,
    ADD_DEPENDENCY,
    REMOVE_DEPENDENCY,
    UPDATE_METADATA,
    UPDATE_STATUS,
    RESET_SUBTREE
)

from cuddlytoddly.infra.logging import get_logger
from pathlib import Path
import cuddlytoddly.ui.git_projection as git_proj

logger = get_logger(__name__)

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
# --------------------------
# Git Repo Setup
# --------------------------



# --------------------------
# Graph Adapter
# --------------------------

# remove: ANSI + 7+ hex digits + ANSI
hash_pattern = re.compile(r'\x1b\[[0-9;]*m[0-9a-f]{7,}\x1b\[[0-9;]*m')

def remove_commit_hashes(lines):
    return [hash_pattern.sub('', line) for line in lines]


# --------------------------
# Incremental Git Layer
# --------------------------

node_to_commit = {}  # maps node_id -> latest commit hash

def get_git_dag_text():
    result = subprocess.run(
        ["git", "branch", "--list", "tip_*"],
        cwd=git_proj.REPO_PATH,
        capture_output=True,
        text=True
    )
    tip_branches = [b.strip().lstrip("* ") for b in result.stdout.splitlines() if b.strip()]

    if not tip_branches:
        tip_branches = ["master"]

    result = subprocess.run(
        ["git", "log", "--graph", "--oneline", "--color=always"] + tip_branches,
        cwd=git_proj.REPO_PATH,
        capture_output=True,
        text=True
    )
    return remove_commit_hashes(result.stdout.splitlines())

def find_root_node(snapshot):
    # root = node with no dependencies
    for node_id, node in snapshot.items():
        if not node.dependencies:
            return node_id
    # fallback: just pick the first node
    return next(iter(snapshot.keys()), None)

def find_path_to_node(dag, target_node):
    """
    Returns a list of nodes from root -> target_node (excluding target_node).
    dag: dict[node_id] -> list of child node_ids
    """
    def dfs(node, path, visited):
        if node == target_node:
            return path
        if node in visited:
            return None
        visited.add(node)
        for child in dag.get(node, []):
            res = dfs(child, path + [node], visited)
            if res is not None:
                return res
        return None

    # Assume single root (first node with no dependencies)
    all_nodes = set(dag.keys())
    all_children = {c for children in dag.values() for c in children}
    roots = list(all_nodes - all_children)
    if not roots:
        roots = list(all_nodes)
    visited = set()
    for root in roots:
        path = dfs(root, [], visited)
        if path:
            return path
    return []

def ensure_path_starts_at_root(dag, path):
    """
    Given a path (list of nodes), ensures it starts at a root node.
    If the path doesn't start at a root, extends it from the beginning
    until a root is found. The end of the path remains unchanged.

    dag: dict[node_id] -> list of child node_ids
    path: list of node_ids
    """
    if not path:
        return path

    # Build a reverse mapping: child -> list of parents
    all_nodes = set(dag.keys())
    all_children = {c for children in dag.values() for c in children}
    roots = all_nodes - all_children

    # If the path already starts at a root, return as-is
    if path[0] in roots:
        return path

    # Build parent map for reverse traversal
    parent_map = {}
    for node, children in dag.items():
        for child in children:
            parent_map.setdefault(child, []).append(node)

    # Walk backwards from path[0] using BFS until we hit a root
    def find_prefix_to_root(start_node):
        # BFS to find shortest path from any root to start_node (in reverse)
        queue = deque([[start_node]])
        visited = {start_node}

        while queue:
            current_path = queue.popleft()
            current_node = current_path[-1]

            if current_node in roots:
                # Reverse since we built it backwards
                return list(reversed(current_path))

            for parent in parent_map.get(current_node, []):
                if parent not in visited:
                    visited.add(parent)
                    queue.append(current_path + [parent])

        return None  # No root found (e.g. cyclic or disconnected)

    prefix = find_prefix_to_root(path[0])

    if prefix is None:
        return path  # Can't extend, return original

    # prefix ends with path[0], so drop the last element to avoid duplication
    return prefix[:-1] + path

def get_aggregate_outputs(snapshot):
    """
    Returns a dict of node_id -> result for all nodes that have results.
    """
    outputs = {}
    for node_id, node in snapshot.items():
        if node.result is not None:
            outputs[node_id] = node.result
    return outputs

# --------------------------
# ANSI Parsing
# --------------------------

ansi_regex = re.compile(r'\x1b\[[0-9;]*m')

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

        codes = match.group()[2:-1].split(';')
        if codes == ['']:
            codes = ['0']

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
    return ansi_regex.sub('', line)

# --------------------------
# Mapping
# --------------------------

def map_nodes_to_lines(git_lines, snapshot):
    # Pre-compute the hash suffix for every node_id
    hash_to_node_id = {
        "#" + hashlib.sha256(node_id.encode()).hexdigest()[:6]: node_id
        for node_id in snapshot
    }

    node_map = {}
    for i, line in enumerate(git_lines):
        clean = strip_ansi(line)
        if '*' not in clean:
            continue
        star_pos = clean.index('*')
        after_star = clean[star_pos + 1:]
        message = after_star.lstrip(" |\\/.-")
        if not message:
            continue

        m = re.search(r'(#[0-9a-f]{6})', message)
        if m:
            node_id = hash_to_node_id.get(m.group(1))
            if node_id:
                node_map[node_id] = i

    return node_map

def get_node_col(line):
    parsed = parse_ansi(line)
    for idx, (ch, _) in enumerate(parsed):
        if ch == '*':
            return idx
    return 0

# --------------------------
# UI
# --------------------------

def trace_branch_path_recursive(git_lines, row, col, child_row, child_col, step=None, visited=None, is_start=True, debug=False):
    """
    Recursive function to find path from (row,col) to (child_row,child_col)
    following \ | / characters, stopping at other *.
    """
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

    char_matrix[(0,+2)] = this_line[col+2] if col+2 < len(this_line) else ''
    char_matrix[(0,+1)] = this_line[col+1] if col+1 < len(this_line) else ''
    char_matrix[(0,0)] = this_line[col] if col < len(this_line) else ''
    char_matrix[(0,-1)] = this_line[col-1] if (0 <= col-1 < len(this_line)) else ''
    char_matrix[(0,-2)] = this_line[col-2] if (0 <= col-2 < len(this_line)) else ''

    # Stop if we hit a '*' that is not the child (and not the start)
    if char == '*' and not (row == child_row and col == child_col) and not is_start:
        return (path_positions|set('x'))

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

        for dcol in range(-2,3):
            char_matrix[(1,dcol)] = next_line[col+dcol] if (0 <= col+dcol < len(next_line)) else ''

        for dcol in range(-2,3):
            ncol = col + dcol
            if 0 <= ncol < len(next_line):
                if (

                    (dcol==1 and char_matrix[(1,1)] == '/' and char_matrix[(0,0)] == '/')
                    or (dcol==1 and char_matrix[(1,1)] == '/' and char_matrix[(0,0)] == '*')
                    or (dcol==1 and char_matrix[(1,1)] == '\\' and char_matrix[(0,0)] == '/' and char_matrix[(1,0)] == ' ')

                    or (dcol==1 and char_matrix[(1,1)] == '/' and char_matrix[(0,0)] == '|' and char_matrix[(1,0)] == '|' and char_matrix[(0,-1)] != '/' and char_matrix[(1,-1)] != '_')
                    or (dcol==1 and char_matrix[(1,1)] == '/' and char_matrix[(0,0)] == '|' and char_matrix[(1,0)] == ' ')
                    or (dcol==1 and char_matrix[(1,1)] == '|' and char_matrix[(0,0)] == '/' and char_matrix[(1,2)] != '/' and char_matrix[(1,2)] != '_')
                    or (dcol==0 and char_matrix[(1,0)] == '|' and char_matrix[(0,0)] == '*' and char_matrix[(1,1)] == '/')

                    or (dcol==1 and char_matrix[(1,1)] == '*' and char_matrix[(0,0)] == '/')

                    or (dcol==0 and char_matrix[(1,0)] == "|" and char_matrix[(0,0)] == '*')
                    or (dcol==0 and char_matrix[(1,0)] == "|" and char_matrix[(0,0)] == '/')
                    or (dcol==0 and char_matrix[(1,0)] == "|" and char_matrix[(0,0)] == '\\')
                    or (dcol==0 and char_matrix[(1,0)] == "|" and char_matrix[(0,0)] == '|')
                    or (dcol==0 and char_matrix[(1,0)] == "\\" and char_matrix[(0,0)] == '|')
                    or (dcol==0 and char_matrix[(1,0)] == "\\" and char_matrix[(0,0)] == '/' and char_matrix[(1,1)] != "/")

                    or (dcol==0 and char_matrix[(1,0)] == '*' and char_matrix[(0,0)] == '|')

                    or (dcol==-1 and char_matrix[(1,-1)] == '|' and char_matrix[(0,0)] == '\\' and char_matrix[(1,0)] == ' ' and char_matrix[(1,-2)] != '\\')
                    or (dcol==-1 and char_matrix[(1,-1)] == '\\' and char_matrix[(0,0)] == '\\')
                    or (dcol==-1 and char_matrix[(1,-1)] == '*' and char_matrix[(0,0)] == '\\')

                    or (dcol==-1 and char_matrix[(1,-1)] == '\\' and char_matrix[(0,0)] == '|' and char_matrix[(0,1)] != '\\')
                    or (dcol==-1 and char_matrix[(1,-1)] == '|' and char_matrix[(0,0)] == '\\' and char_matrix[(1,-2)] != '\\')
                    or (dcol==-1 and char_matrix[(1,-1)] == '\\' and char_matrix[(0,0)] == '*')

                    or (dcol==-1 and char_matrix[(1,-1)] == '*' and char_matrix[(0,0)] == '\\')

                    ):
                    subpath_positions += [trace_branch_path_recursive(
                        git_lines, next_row, ncol, child_row, child_col, step, visited, is_start=False, debug=debug
                    )]
        if (char_matrix[(1,2)] in ['_','/'] and char_matrix[(0,0)] == '/' and char_matrix[(0,1)] == '|' and char_matrix[(1,1)] == '|'):
            temp_set = set()
            dcol = 2
            while char_matrix[(1,dcol)] == '_' and char_matrix[(1,dcol-1)] == '|':
                temp_set.add((next_row, col + dcol))
                dcol += 2
                char_matrix[(1,dcol)] = next_line[col+dcol] if (0 <= col+dcol < len(next_line)) else ''
                char_matrix[(1,dcol-1)] = next_line[col+dcol-1] if (0 <= col+dcol-1 < len(next_line)) else ''

            ncol = col + dcol
            subpath_positions += [trace_branch_path_recursive(
                        git_lines, next_row, ncol, child_row, child_col, step, visited, is_start=False, debug=debug
                    )|temp_set]
            
        if (char_matrix[(1,-1)] in ['.','-'] and char_matrix[(0,0)] == '\\'):
            temp_set = set()
            dcol = -1
            while char_matrix[(1,dcol)] in ['.','-']:
                temp_set.add((next_row, col + dcol))
                dcol -= 1
                char_matrix[(1,dcol)] = next_line[col+dcol] if (0 <= col+dcol < len(next_line)) else ''

            ncol = col + dcol
            subpath_positions += [trace_branch_path_recursive(
                        git_lines, next_row, ncol, child_row, child_col, step, visited, is_start=False, debug=debug
                    )|temp_set]
        
    if len(subpath_positions)>1:
        nothing_added = True
        for subpath_position in subpath_positions:
            if 'x' not in subpath_position:
                nothing_added = False
                path_positions |= subpath_position
        if nothing_added:
            path_positions |= set('x')
    elif len(subpath_positions)==1:
        path_positions |= subpath_positions[0]


    return path_positions

def build_reverse_dag(dag):
    """Returns a dict mapping child_id -> list of parent_ids."""
    reverse = defaultdict(list)
    for parent, children in dag.items():
        for child in children:
            reverse[child].append(parent)
    return reverse

class ModalField:
    """A single editable field inside a modal."""
    def __init__(self, label, value="", completions=None, validator=None):
        self.label = label
        self.value = value
        self.completions = completions or []  # list of strings for autocomplete
        self.validator = validator             # callable(str) -> str|None (error msg)
        self.cursor = len(value)
        self.error = None
        self._completion_idx = -1
        self._completion_prefix = ""
        self._dd_idx = -1   # highlighted row in the visible dropdown list (-1 = none)

    def _current_token(self):
            """Return the text after the last comma (stripped), for autocomplete."""
            parts = self.value[:self.cursor].rsplit(",", 1)
            return parts[-1].strip()

    def _select_dd_item(self, completion):
        """Insert *completion*, replacing the current comma-token."""
        before_cursor = self.value[:self.cursor]
        last_comma = before_cursor.rfind(",")
        if last_comma == -1:
            self.value = completion + self.value[self.cursor:]
        else:
            prefix_part = self.value[:last_comma + 1] + " "
            self.value = prefix_part + completion + self.value[self.cursor:]
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
                self.value = self.value[:self.cursor-1] + self.value[self.cursor:]
                self.cursor -= 1
                self._completion_idx = -1
        elif k == curses.KEY_LEFT:
            self.cursor = max(0, self.cursor - 1)
        elif k == curses.KEY_RIGHT:
            self.cursor = min(len(self.value), self.cursor + 1)
        elif k == ord('\t') and self.completions:
            token = self._current_token()
            matches = [c for c in self.completions if c.startswith(token)]
            if matches:
                self._completion_idx = (self._completion_idx + 1) % len(matches)
                self._dd_idx = self._completion_idx   # keep dropdown highlight in sync
                completion = matches[self._completion_idx]
                # Replace only the current token, preserving everything before it
                before_cursor = self.value[:self.cursor]
                last_comma = before_cursor.rfind(",")
                if last_comma == -1:
                    # No comma — replace entire value
                    self.value = completion + self.value[self.cursor:]
                else:
                    # Replace only the token after the last comma
                    prefix_part = self.value[:last_comma + 1] + " "
                    self.value = prefix_part + completion + self.value[self.cursor:]
                self.cursor = len(self.value)
        elif 32 <= k <= 126:
            ch = chr(k)
            self.value = self.value[:self.cursor] + ch + self.value[self.cursor:]
            self.cursor += 1
            self._completion_idx = -1  # reset on typing
            self._dd_idx = -1          # reset dropdown selection on typing
            
    def validate(self):
        if self.validator:
            self.error = self.validator(self.value)
        return self.error is None

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
        if k == ord('\t') and self.fields[self.active_field].completions:
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
            is_active = (i == self.active_field)
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
                    stdscr.addstr(row, panel_x, f" ! {field.error}", curses.color_pair(curses.COLOR_RED + 1))
                except curses.error:
                    pass
                row += 1

            # Autocomplete dropdown
            if is_active and field.completions:
                token = field._current_token()
                matches = field._dd_matches()
                if matches:
                    for j, m in enumerate(matches):
                        is_sel = (j == field._dd_idx)
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
            stdscr.addstr(row + 1, panel_x,
                          " ↑↓: switch field  Tab: complete  Enter: confirm  Esc: cancel", curses.A_DIM)
        except curses.error:
            pass

def export_results_to_markdown(snapshot, run_dir):
    """Walk the DAG in topological order and write all results to a
    Markdown file in <run_dir>/outputs/. Returns the output path."""
    from datetime import datetime

    out_dir = run_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = out_dir / f"export_{timestamp}.md"

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

    goal_nodes = [snapshot[nid] for nid in order
                  if snapshot.get(nid) and snapshot[nid].node_type == "goal"]
    title = (goal_nodes[0].metadata.get("description", goal_nodes[0].id)
             if goal_nodes else run_dir.name.replace("_", " ").title())

    lines = [
        f"# {title}", "",
        f"*Exported {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*", "",
        "---", "",
        "## Summary", "",
        "| Node | Type | Status |",
        "|------|------|--------|",
    ]
    for nid in order:
        node = snapshot.get(nid)
        if not node or node.metadata.get("hidden") or node.node_type == "execution_step":
            continue
        lines.append(f"| {nid} | {node.node_type} | {node.status} |")
    lines += ["", "---", "", "## Results", ""]

    for nid in order:
        node = snapshot.get(nid)
        if not node or node.node_type == "execution_step" or node.metadata.get("hidden"):
            continue
        desc = node.metadata.get("description", "")
        lines.append(f"### {nid}")
        if desc and desc != nid:
            lines += [f"*{desc}*", ""]
        deps = ", ".join(sorted(node.dependencies)) or "none"
        lines.append(f"**Type:** {node.node_type} | **Status:** {node.status} | **Deps:** {deps}")
        lines.append("")
        req_input = node.metadata.get("required_input")
        output    = node.metadata.get("output")
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

def open_add_modal(snapshot, event_queue, current_node, set_modal):
    node_ids = list(snapshot.keys())
 
    def on_submit(values):
        new_id         = values["ID"].strip()
        new_desc       = values["Description"].strip()
        deps_raw       = values["Dependencies"].strip()
        dependents_raw = values["Dependents"].strip()
        ntype          = values["Type"].strip() or "task"
 
        if not new_id or new_id in snapshot:
            set_modal(None)
            return
 
        deps = [d.strip() for d in deps_raw.split(",")
                if d.strip() and d.strip() in snapshot]
        dependents = [d.strip() for d in dependents_raw.split(",")
                      if d.strip() and d.strip() in snapshot]
 
        event_queue.put(Event(ADD_NODE, {
            "node_id":      new_id,
            "node_type":    ntype,
            "dependencies": deps,
            "origin":       "user",
            "metadata":     {"description": new_desc},
        }))
 
        for dependent_id in dependents:
            event_queue.put(Event(ADD_DEPENDENCY, {
                "node_id":    dependent_id,
                "depends_on": new_id,
            }))
            # Reset each dependent and its subtree so they rerun
            # with the new node as a prerequisite.
            event_queue.put(Event(RESET_SUBTREE, {"node_id": dependent_id}))
 
        set_modal(None)
 
    set_modal(Modal(
        title="Add Node",
        fields=[
            ModalField("ID",           value=""),
            ModalField("Description",  value=""),
            ModalField("Type",         value="task", completions=["task", "goal"]),
            ModalField("Dependencies", value=current_node or "", completions=node_ids),
            ModalField("Dependents",   value="",                 completions=node_ids),
        ],
        on_submit=on_submit,
        on_cancel=lambda: set_modal(None),
    ))
  
def open_edit_modal(current_node, snapshot, event_queue, set_modal):
    node = snapshot[current_node]
    node_ids = [nid for nid in snapshot.keys() if nid != current_node]
    current_deps = ", ".join(node.dependencies)
    # Nodes whose dependency list includes current_node
    current_dependents = ", ".join(
        nid for nid, n in snapshot.items()
        if current_node in n.dependencies
    )
 
    def on_submit(values):
        new_id           = values["ID"].strip()
        new_desc         = values["Description"].strip()
        new_deps_raw     = values["Dependencies"].strip()
        new_status       = values["Status"].strip()
        new_dep_raw      = values["Dependents"].strip()
 
        new_deps = [d.strip() for d in new_deps_raw.split(",") if d.strip()]
        new_deps = [d for d in new_deps if d in snapshot]
 
        new_dependents = [d.strip() for d in new_dep_raw.split(",")
                          if d.strip() and d.strip() in snapshot
                          and d.strip() != current_node]
        new_dependents_set = set(new_dependents)
        old_dependents = {nid for nid, n in snapshot.items()
                          if current_node in n.dependencies}
 
        event_queue.put(Event(UPDATE_METADATA, {
            "node_id": current_node,
            "origin":  "user",
            "metadata": {"description": new_desc},
        }))
 
        if new_status in ("pending", "done", "running", "failed", "to_be_expanded"):
            event_queue.put(Event(UPDATE_STATUS, {
                "node_id": current_node,
                "status":  new_status,
            }))
 
        old_deps     = set(node.dependencies)
        new_deps_set = set(new_deps)
        for removed in old_deps - new_deps_set:
            event_queue.put(Event(REMOVE_DEPENDENCY, {
                "node_id": current_node, "depends_on": removed,
            }))
        for added in new_deps_set - old_deps:
            event_queue.put(Event(ADD_DEPENDENCY, {
                "node_id": current_node, "depends_on": added,
            }))
 
        # Apply dependent changes
        for removed in old_dependents - new_dependents_set:
            event_queue.put(Event(REMOVE_DEPENDENCY, {
                "node_id": removed, "depends_on": current_node,
            }))
            event_queue.put(Event(RESET_SUBTREE, {"node_id": removed}))
        for added in new_dependents_set - old_dependents:
            event_queue.put(Event(ADD_DEPENDENCY, {
                "node_id": added, "depends_on": current_node,
            }))
            event_queue.put(Event(RESET_SUBTREE, {"node_id": added}))
 
        if new_id and new_id != current_node and new_id not in snapshot:
            event_queue.put(Event(ADD_NODE, {
                "node_id":      new_id,
                "node_type":    node.node_type,
                "dependencies": list(new_deps_set),
                "origin":       node.origin,
                "metadata":     {**node.metadata, "description": new_desc},
            }))
            for child in node.children:
                event_queue.put(Event(ADD_DEPENDENCY,    {"node_id": child, "depends_on": new_id}))
                event_queue.put(Event(REMOVE_DEPENDENCY, {"node_id": child, "depends_on": current_node}))
            event_queue.put(Event(REMOVE_NODE, {"node_id": current_node}))
            event_queue.put(Event(RESET_SUBTREE, {"node_id": new_id}))
        else:
            event_queue.put(Event(RESET_SUBTREE, {"node_id": current_node}))
 
        set_modal(None)
 
    set_modal(Modal(
        title="Edit Node",
        fields=[
            ModalField("ID",           value=current_node),
            ModalField("Description",  value=node.metadata.get("description", "")),
            ModalField("Dependencies", value=current_deps,       completions=node_ids),
            ModalField("Dependents",   value=current_dependents, completions=node_ids),
            ModalField("Status",       value=node.status,
                       completions=["pending", "running", "done", "failed", "to_be_expanded"]),
        ],
        on_submit=on_submit,
        on_cancel=lambda: set_modal(None),
    ))
  
def open_remove_modal(current_node, snapshot, event_queue, set_modal):
    node     = snapshot[current_node]
    parents  = list(node.dependencies)
    children = list(node.children)
 
    options = [
        ("Remove node only — rewire children to its parents", "rewire"),
        ("Remove node and all descendants",                   "cascade"),
        ("Remove node and disconnect everything",             "disconnect"),
    ]
 
    def on_submit(values):
        choice = values["Action"].strip()
        mode   = next((m for label, m in options if label == choice), None)
 
        if mode == "rewire":
            for child in children:
                event_queue.put(Event(REMOVE_DEPENDENCY, {
                    "node_id": child, "depends_on": current_node,
                }))
                for parent in parents:
                    event_queue.put(Event(ADD_DEPENDENCY, {
                        "node_id": child, "depends_on": parent,
                    }))
            event_queue.put(Event(REMOVE_NODE, {"node_id": current_node}))
            # Children survive — reset them and their subtrees
            for child in children:
                event_queue.put(Event(RESET_SUBTREE, {"node_id": child}))
 
        elif mode == "cascade":
            # REMOVE_NODE recurses into children — nothing left to reset
            event_queue.put(Event(REMOVE_NODE, {"node_id": current_node}))
 
        elif mode == "disconnect":
            for child in children:
                event_queue.put(Event(REMOVE_DEPENDENCY, {
                    "node_id": child, "depends_on": current_node,
                }))
            event_queue.put(Event(REMOVE_NODE, {"node_id": current_node}))
            # Children survive without this dep — reset them
            for child in children:
                event_queue.put(Event(RESET_SUBTREE, {"node_id": child}))
 
        set_modal(None)
 
    set_modal(Modal(
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
    ))
 
def dag_interface(stdscr, orchestrator, run_dir=None):
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
    export_notice = None   # (message, expire_time) or None

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
            logger.debug("[UI REBUILD] Version changed from %d to %d",
                         last_seen_version, graph.structure_version)

            try:
                with graph_lock:
                    snapshot = graph.get_snapshot()

                rebuild_repo_from_graph(graph)
                cached_git_lines    = get_git_dag_text()
                cached_node_to_line = map_nodes_to_lines(cached_git_lines, snapshot)

                last_seen_version  = version_at_rebuild_start
                last_exec_version  = exec_version_at_rebuild_start

                dag         = graph_to_dag(snapshot)
                reverse_dag = build_reverse_dag(dag)

                if current_node not in snapshot:
                    current_node = find_root_node(snapshot)
                    parent_stack = []
                    child_stack  = find_path_to_node(reverse_dag, current_node)
                else:
                    parent_stack = find_path_to_node(dag, current_node)
                    child_stack  = find_path_to_node(reverse_dag, current_node)

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
            dag         = graph_to_dag(snapshot)
            reverse_dag = build_reverse_dag(dag)

        # ── Rendering ─────────────────────────────────────────────────────────
        if not skip_render:
            try:
                git_lines    = cached_git_lines
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
                    current_col  = get_node_col(git_lines[current_line]) \
                                   if current_line is not None else 0
                else:
                    current_line = None
                    current_col  = None

                if parent_node and current_node:
                    parent_line = node_to_line.get(parent_node)
                    parent_col  = get_node_col(git_lines[parent_line]) \
                                  if parent_line is not None else None
                else:
                    parent_node  = None
                    parent_line  = None
                    parent_col   = None

                # Branch path highlight
                if (
                    branch_mode
                    and parent_node
                    and current_line is not None
                    and parent_line is not None
                ):
                    path = trace_branch_path_recursive(
                        git_lines, parent_line, parent_col,
                        current_line, current_col,
                    )
                else:
                    path = set()

                # Node-type symbol maps
                goal_star_positions = {}
                step_star_positions = {}
                for node_id, line_idx in node_to_line.items():
                    node = snapshot.get(node_id)
                    if not node:
                        continue
                    nt = getattr(node, "node_type", None)
                    if nt == "goal":
                        goal_star_positions[line_idx] = get_node_col(git_lines[line_idx])
                    elif nt == "execution_step":
                        step_star_positions[line_idx] = (
                            get_node_col(git_lines[line_idx]),
                            node.metadata.get("hidden", False),
                            node.status == "failed",
                        )

                line_to_node = {v: k for k, v in node_to_line.items()}

                # Label overrides (hash → description)
                line_label_overrides = {}
                for line_idx, node_id in line_to_node.items():
                    node = snapshot.get(node_id)
                    if not node:
                        continue
                    h6   = "#" + hashlib.sha256(node_id.encode()).hexdigest()[:6]
                    desc = node.metadata.get("description") or node_id
                    line_label_overrides[line_idx] = (h6, f"{h6} {desc}")

                if git_lines:
                    start = 0
                    if current_line is not None:
                        start = max(0, current_line - h // 2)

                    for i, line in enumerate(git_lines[start:start + h - 1]):
                        parsed           = parse_ansi(line)
                        x                = 0
                        current_line_idx = i + start
                        override         = line_label_overrides.get(current_line_idx)

                        hash_start_x = None
                        if override:
                            h6      = override[0]
                            visible = "".join(ch for ch, _ in parsed)
                            idx     = visible.find(h6)
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

                            if ch == '*' and goal_star_positions.get(current_line_idx) == x:
                                ch = 'o'

                            if ch == '*' and current_line_idx in step_star_positions:
                                col, hidden, failed = step_star_positions[current_line_idx]
                                if x == col:
                                    ch = '·' if hidden else ('✗' if failed else '◆')

                            if hash_start_x is not None and x == hash_start_x:
                                full_label = override[1]
                                available  = (w // 2 - 1) - x
                                label = (
                                    (full_label[:available - 3] + "...")
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
                node_label      = current_node if current_node else "<empty>"
                llm_paused      = orchestrator.llm_stopped
                paused_indicator = " | [LLM PAUSED]" if llm_paused else ""
                activity        = orchestrator.current_activity
                started         = orchestrator.activity_started

                if activity and started:
                    elapsed      = time.time() - started
                    activity_str = f" {activity} ({elapsed:.0f}s)"
                else:
                    activity_str = ""

                status_line = (
                    "Up/Down/Left/Right/[/]: move | "
                    f"j/k </> scroll info | "
                    f"e: edit | a: add | x: remove | p: export | "
                    f"s: {'resume' if llm_paused else 'pause'} LLM | g: switch goal | q: quit"
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
                    draw_info_panel(stdscr, h, w, current_node, snapshot, selected_nodes, info_scroll)

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
        child_stack  = ensure_path_starts_at_root(reverse_dag, child_stack + [current_node])[:-1]

        if active_modal:
            active_modal.handle_key(k)
            continue

        if k == curses.KEY_UP:
            info_scroll = 0
            children = dag.get(current_node, [])
            if not branch_mode and children:
                parent_stack.append(current_node)
                parent_node  = current_node
                current_node = child_stack.pop() if child_stack else current_node
                branch_mode  = True
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
                    delta         = -1 if k == curses.KEY_LEFT else 1
                    selection_index = (current_index + delta) % len(siblings)
                    current_node    = siblings[selection_index]
                    child_stack     = find_path_to_node(reverse_dag, current_node)

        elif k in (ord('['), ord(']')):
            info_scroll = 0
            parents = reverse_dag.get(current_node, [])
            if parent_node and parents and parent_node in parents:
                parent_index  = parents.index(parent_node)
                delta         = -1 if k == ord('[') else 1
                selection_index = (parent_index + delta) % len(parents)
                parent_node     = parents[selection_index]
                parent_stack    = find_path_to_node(dag, parent_node) + [parent_node]

        elif k == ord("s"):
            if orchestrator.llm_stopped:
                orchestrator.resume_llm_calls()
            else:
                orchestrator.stop_llm_calls()

        elif k == ord("e"):
            if current_node:
                open_edit_modal(current_node, snapshot, event_queue, set_modal)

        elif k == ord("a"):
            open_add_modal(snapshot, event_queue, current_node, set_modal)

        elif k == ord("x"):
            if current_node:
                open_remove_modal(current_node, snapshot, event_queue, set_modal)

        elif k == ord("p"):
            if run_dir and snapshot:
                try:
                    out_path = export_results_to_markdown(snapshot, run_dir)
                    export_notice = (f"Exported → {out_path.name}", time.time() + 4)
                except Exception as ex:
                    export_notice = (f"Export failed: {ex}", time.time() + 4)
                    logger.error("[EXPORT] Failed: %s", ex, exc_info=True)

        elif k in (curses.KEY_PPAGE, ord("<")):   # Page Up
            info_scroll = max(0, info_scroll - (h - 4))

        elif k in (curses.KEY_NPAGE, ord(">")):   # Page Down
            info_scroll += (h - 4)    # draw_info_panel clamps the max

        elif k == ord("j"):            # fine scroll down
            info_scroll += 3

        elif k == ord("k"):            # fine scroll up
            info_scroll = max(0, info_scroll - 3)

        elif k == ord("g"):
            switch_requested = True
            break

    return "switch" if switch_requested else None


def draw_info_panel(stdscr, h, w, node_id, snapshot, selected_nodes, scroll_offset=0):
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

    desc = node.metadata.get("description")
    if desc:
        lines+=[
        f" Desc:   {desc}"," "
        ]

    input = node.metadata.get("required_input")
    if input:
        lines+=[
        f" Input:  {input}"," "
        ]

    output = node.metadata.get("output")
    if output:
        lines+=[
        f" Output: {output}"," "
        ]

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

    # After showing the node's own result, show any visible execution steps
    step_children = [
        n for n in snapshot.values()
        if n.node_type == "execution_step"
        and node_id in n.dependencies
        and not n.metadata.get("hidden", False)
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
            rendered.append("")          # blank spacer row
        else:
            for subline in textwrap.wrap(line, width=max(1, panel_w - 2)):
                rendered.append(subline)

    visible_rows = h - 2
    total = len(rendered)
    scroll_offset = max(0, min(scroll_offset, max(0, total - visible_rows)))

    for i, subline in enumerate(rendered[scroll_offset: scroll_offset + visible_rows]):
        if not subline:
            continue                     # skip empty rows — addstr("") can error
        try:
            stdscr.addstr(i, panel_x + 1, subline[:panel_w - 2])
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
    ):
    """
    Run the curses DAG UI.
    If the user presses 'g', the startup screen is shown so they can pick a
    different goal or resume a previous run.  This requires `repo_root` and
    `restart_fn` to be supplied:

        run_ui(
            orchestrator,
            run_dir=run_dir,
            repo_root=REPO_ROOT,
            restart_fn=_init_system,   # callable(StartupChoice) -> (orch, run_dir)
        )
    """
    import sys
    root = logging.getLogger("dag")
    ch   = getattr(root, "_stderr_handler", None)
    if ch:
        root.removeHandler(ch)

    log_path   = (run_dir / "logs" / "dag.log") if run_dir else Path("logs/dag.log")
    log_file   = open(log_path, "a", encoding="utf-8", buffering=1)
    old_stderr = sys.stderr
    sys.stderr  = log_file

    try:
        while True:
            try:
                rebuild_repo_from_graph(orchestrator.graph)
            except Exception as exc:
                logger.warning("[UI] Git pre-warm failed (non-fatal): %s", exc)
            result = curses.wrapper(dag_interface, orchestrator, run_dir)

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

            # Reopen log file pointed at the new run directory.
            log_file.close()
            new_log = (run_dir / "logs" / "dag.log") if run_dir else Path("logs/dag.log")
            log_file   = open(new_log, "a", encoding="utf-8", buffering=1)
            sys.stderr  = log_file

    finally:
        sys.stderr = old_stderr
        log_file.close()
        if ch:
            root.addHandler(ch)