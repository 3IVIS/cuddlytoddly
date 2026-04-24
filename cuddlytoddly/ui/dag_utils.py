"""
cuddlytoddly/ui/dag_utils.py

Pure DAG/graph traversal helpers — no curses dependency.
Used by curses_ui.py and potentially by other UI layers.
"""

from __future__ import annotations

import re
import subprocess
from collections import defaultdict, deque

import cuddlytoddly.ui.git_projection as git_proj
from toddly.infra.logging import get_logger

logger = get_logger(__name__)

# remove: ANSI + 7+ hex digits + ANSI
hash_pattern = re.compile(r"\x1b\[[0-9;]*m[0-9a-f]{7,}\x1b\[[0-9;]*m")

# maps node_id -> latest commit hash (incremental git layer)
node_to_commit = {}


# --------------------------
# Graph Adapter
# --------------------------


def remove_commit_hashes(lines):
    return [hash_pattern.sub("", line) for line in lines]


# --------------------------
# Incremental Git Layer
# --------------------------


def get_git_dag_text(repo_path: str | None = None):
    """Return git log lines for the current DAG.

    Parameters
    ----------
    repo_path:
        Path to the shadow git repository.  Defaults to
        ``git_proj.REPO_PATH`` (the module-level default) for backward compat.
    """
    if repo_path is None:
        repo_path = git_proj.REPO_PATH
    result = subprocess.run(
        ["git", "branch", "--list", "tip_*"],
        cwd=repo_path,
        capture_output=True,
        text=True,
    )
    tip_branches = [b.strip().lstrip("* ") for b in result.stdout.splitlines() if b.strip()]

    if not tip_branches:
        tip_branches = ["master"]

    result = subprocess.run(
        ["git", "log", "--graph", "--oneline", "--color=always"] + tip_branches,
        cwd=repo_path,
        capture_output=True,
        text=True,
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


def build_reverse_dag(dag):
    """Returns a dict mapping child_id -> list of parent_ids."""
    reverse = defaultdict(list)
    for parent, children in dag.items():
        for child in children:
            reverse[child].append(parent)
    return reverse
