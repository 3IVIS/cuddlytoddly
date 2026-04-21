# --- FILE: cuddlytoddly/ui/git_projection.py ---

"""
git_projection.py — per-run git repository state for DAG visualisation.

Each concurrent run should own a ``GitProjection`` instance so that
``repo_path``, ``repo``, and ``node_to_commit`` are never shared across runs.

The module-level shim functions (``rebuild_repo_from_graph``, ``delete_node``,
``REPO_PATH``) are kept for backward compatibility with callers that haven't
been updated to pass an instance explicitly.  New code should use
``GitProjection`` directly.
"""

import hashlib
import os
import re
import shutil
from collections import defaultdict, deque
from pathlib import Path

import git
from cuddlytoddly.infra.logging import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pure helpers (no state)
# ---------------------------------------------------------------------------


def truncate_label(label, node_id=None, max_len=20):
    if node_id is not None:
        return "#" + hashlib.sha256(node_id.encode()).hexdigest()[:6]
    return "#" + hashlib.sha256(label.encode()).hexdigest()[:6]


def graph_to_dag(snapshot):
    dag = {node_id: [] for node_id in snapshot}
    for node_id, node in snapshot.items():
        for dep in node.dependencies:
            if dep in dag:
                dag[dep].append(node_id)
    return dag


def topological_sort(dag):
    indegree = defaultdict(int)
    for node, children in dag.items():
        for child in children:
            indegree[child] += 1

    queue = deque([n for n in dag if indegree[n] == 0])
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for child in dag[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    return order


def compute_descendants(snapshot):
    """
    Return a mapping of node_id -> set of all descendant node_ids.

    Uses iterative BFS per root to avoid hitting Python's recursion limit on
    large graphs (replaces the previous recursive inner-function approach).
    """
    reverse: dict[str, set] = defaultdict(set)
    for node_id, node in snapshot.items():
        for dep in node.dependencies:
            reverse[dep].add(node_id)

    descendants: dict[str, set] = defaultdict(set)

    for root_id in snapshot:
        visited: set = set()
        queue: deque = deque(reverse[root_id])
        while queue:
            child = queue.popleft()
            if child in visited:
                continue
            visited.add(child)
            descendants[root_id].add(child)
            queue.extend(reverse[child])

    return descendants


def get_leaf_node_ids(dag):
    """Nodes with no children — tips of the DAG."""
    return {node_id for node_id, children in dag.items() if not children}


def sanitize_branch_name(node_id):
    """Replace characters invalid in Git branch names."""
    return re.sub(r"[^a-zA-Z0-9._-]", "_", node_id)


# ---------------------------------------------------------------------------
# GitProjection — per-run encapsulation of git state
# ---------------------------------------------------------------------------


class GitProjection:
    """
    Encapsulates all mutable git state for a single run.

    Keeping state per-instance (rather than in module-level globals) means
    two concurrent runs in the same process — possible when the web UI is
    used with multiple goal sessions — cannot corrupt each other's git repo.

    Parameters
    ----------
    repo_path : str | Path
        Directory where the shadow git repository will be created.
        Defaults to ``"dag_repo"`` to match the historical module default.
    """

    def __init__(self, repo_path: str | Path = "dag_repo"):
        self.repo_path = str(repo_path)
        self._repo = None
        self._node_to_commit: dict = {}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _init_repo(self, path: str):
        if os.path.exists(path):
            shutil.rmtree(path)
        os.makedirs(path)
        self._repo = git.Repo.init(path)
        return self._repo

    def _commit_nodes_from_graph(self, snapshot):
        self._node_to_commit.clear()
        repo_dir = Path(self.repo_path)
        dag = graph_to_dag(snapshot)
        order = topological_sort(dag)

        try:
            self._repo.head.reference = self._repo.commit("HEAD")
            self._repo.git.checkout("--detach", "HEAD")
        except Exception:
            pass

        for node_id in order:
            node = snapshot[node_id]
            parents = [
                self._node_to_commit[dep]
                for dep in sorted(node.dependencies)
                if dep in self._node_to_commit
            ]
            file_path = repo_dir / f"{node_id}.txt"
            file_path.write_text(f"This is node {node_id}\nStatus: {node.status}\n")
            self._repo.index.add([str(file_path.relative_to(self.repo_path))])
            label = truncate_label(node.metadata.get("description") or node_id, node_id=node_id)
            try:
                parent_commits = [self._repo.commit(p) for p in parents]
                commit_obj = self._repo.index.commit(
                    f"{label} [{node.status}]", parent_commits=parent_commits
                )
                self._node_to_commit[node_id] = commit_obj.hexsha
            except Exception as e:
                logger.exception("FAILED to commit node '%s': %s", node_id, e)

    def _commit_node_incremental(self, node_id, node, snapshot):
        """Commit a single node, updating _node_to_commit on success."""
        repo_dir = Path(self.repo_path)
        file_path = repo_dir / f"{node_id}.txt"

        file_path.write_text(
            f"This is node {node_id}\n"
            f"Dependencies: {list(node.dependencies)}\n"
            f"Status: {node.status}\n"
        )
        self._repo.index.add([str(file_path.relative_to(self.repo_path))])

        parents = [
            self._node_to_commit[dep]
            for dep in sorted(node.dependencies)
            if dep in self._node_to_commit
        ]

        try:
            parent_commits = [self._repo.commit(p) for p in parents] if parents else []
            label = truncate_label(node.metadata.get("description") or node_id, node_id=node_id)
            commit_obj = self._repo.index.commit(
                f"{label} [{node.status}]", parent_commits=parent_commits
            )
            self._node_to_commit[node_id] = commit_obj.hexsha
            node.metadata["last_commit_status"] = node.status
            node.metadata["last_commit_parents"] = sorted(parents)
            return True
        except Exception as e:
            logger.exception("Failed to commit node '%s': %s", node_id, e)
            return False

    def _update_tip_branches(self, snapshot):
        graph_to_dag(snapshot)

        for branch in list(self._repo.heads):
            if branch.name.startswith("tip_"):
                try:
                    self._repo.delete_head(branch, force=True)
                except Exception as e:
                    logger.debug("[GIT] Could not delete branch %s: %s", branch.name, e)

        for node_id in snapshot:
            if node_id in self._node_to_commit:
                branch_name = f"tip_{sanitize_branch_name(node_id)}"
                try:
                    self._repo.create_head(branch_name, self._node_to_commit[node_id], force=True)
                except Exception as e:
                    logger.debug("[GIT] Could not create branch %s: %s", branch_name, e)

        root_id = next(
            (
                nid
                for nid in snapshot
                if not snapshot[nid].dependencies and nid in self._node_to_commit
            ),
            None,
        )
        if root_id:
            try:
                self._repo.create_head("master", self._node_to_commit[root_id], force=True)
            except Exception as e:
                logger.debug("[GIT] Could not update master: %s", e)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def rebuild_repo_from_graph(self, graph, incremental=True, snapshot_filter_fn=None):
        try:
            snapshot = graph.get_snapshot()
            # Apply the supplied filter, or fall back to the safe generic
            # default which includes every node (no filtering at all).
            _include = snapshot_filter_fn or (lambda _n: True)
            snapshot = {nid: n for nid, n in snapshot.items() if _include(n)}
            dag = graph_to_dag(snapshot)

            try:
                if self._repo is None:
                    raise ValueError("no repo yet")
                self._repo.head.commit
            except (ValueError, TypeError):
                incremental = False

            if not incremental:
                self._init_repo(self.repo_path)
                self._node_to_commit.clear()
                self._commit_nodes_from_graph(snapshot)
                self._update_tip_branches(snapshot)
                return

            order = topological_sort(dag)
            dirty: set = set()

            for node_id in order:
                node = snapshot[node_id]
                last_status = node.metadata.get("last_commit_status")
                last_parents = node.metadata.get("last_commit_parents", [])
                missing_parent = False
                resolved_parents = []

                for dep in node.dependencies:
                    if dep not in self._node_to_commit:
                        missing_parent = True
                    else:
                        resolved_parents.append(self._node_to_commit[dep])

                current_parents = sorted(resolved_parents)

                if missing_parent:
                    dirty.add(node_id)
                if (
                    node_id not in self._node_to_commit
                    or node.status != last_status
                    or current_parents != last_parents
                ):
                    dirty.add(node_id)

            for node_id in order:
                if any(dep in dirty for dep in snapshot[node_id].dependencies):
                    dirty.add(node_id)

            changed = True
            while changed:
                changed = False
                for node_id in order:
                    if node_id in dirty:
                        for child_id in snapshot[node_id].children:
                            if child_id not in dirty and child_id in snapshot:
                                dirty.add(child_id)
                                changed = True

            full_order = order + [n for n in snapshot if n not in order]
            for node_id in snapshot:
                if node_id not in self._node_to_commit:
                    dirty.add(node_id)

            if self._node_to_commit:
                some_sha = next(iter(self._node_to_commit.values()))
                self._repo.git.checkout(some_sha)
            else:
                self._repo.git.checkout("--orphan", "tmp_head")
                self._repo.index.reset()

            for node_id in full_order:
                if node_id in dirty or node_id not in self._node_to_commit:
                    self._commit_node_incremental(node_id, snapshot[node_id], snapshot)

            self._repo.index.reset()
            self._update_tip_branches(snapshot)

            try:
                self._repo.git.gc(prune="now")
            except Exception as e:
                logger.warning("gc warning (non-fatal): %s", e)

        except Exception as e:
            logger.error("[GIT] rebuild_repo_from_graph failed: %s", e, exc_info=True)
            try:
                self._init_repo(self.repo_path)
                self._node_to_commit.clear()
                self._commit_nodes_from_graph(snapshot)
                self._update_tip_branches(snapshot)
                logger.info("[GIT] Full rebuild succeeded after error")
            except Exception as e2:
                logger.error("[GIT] Full rebuild also failed: %s", e2)

    def delete_node(self, node_id, graph):
        """Soft-delete a node by marking it deleted and committing."""
        if node_id in graph.nodes:
            node = graph.nodes[node_id]
            node.metadata["deleted"] = True
            self._commit_node_incremental(node_id, node, graph.get_snapshot())


# ---------------------------------------------------------------------------
# Module-level default instance — backward-compat shims
# ---------------------------------------------------------------------------
# Legacy callers (curses_ui, __main__) may still use these.  New code should
# create and pass a GitProjection instance explicitly.

_default = GitProjection("dag_repo")

# Expose repo_path as a read-only module attribute so legacy code that reads
# ``git_proj.REPO_PATH`` continues to work via the default instance.
REPO_PATH: str = _default.repo_path


def configure(repo_path: str | Path) -> GitProjection:
    """
    Re-point the module-level default instance to *repo_path* and return it.

    Replaces the old ``git_proj.REPO_PATH = ...`` assignment in ``_init_system``.
    For truly concurrent runs, prefer creating an explicit ``GitProjection``
    and passing it through the call stack instead.
    """
    global _default, REPO_PATH
    _default = GitProjection(repo_path)
    REPO_PATH = _default.repo_path
    return _default


def rebuild_repo_from_graph(graph, incremental=True, snapshot_filter_fn=None):
    """Legacy shim — delegates to the module-level default instance."""
    _default.rebuild_repo_from_graph(graph, incremental, snapshot_filter_fn)


def delete_node(node_id, graph):
    """Legacy shim — delegates to the module-level default instance."""
    _default.delete_node(node_id, graph)
