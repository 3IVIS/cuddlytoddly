"""
TaskGraph

Single source of truth for DAG planning and execution.
Nodes now include required_input/output metadata to support:
- Explicit dependency checking
- Automatic parallelism reasoning
- LLM-aware semantic planning
"""

import copy

from cuddlytoddly.infra.logging import get_logger

logger = get_logger(__name__)


class TaskGraph:
    class Node:
        # FIX #15: __slots__ reduces per-instance memory overhead for large graphs.
        # All instance attributes assigned in __init__ must appear here.
        __slots__ = (
            "id",
            "dependencies",
            "children",
            "node_type",
            "status",
            "result",
            "origin",
            "metadata",
        )

        def __init__(
            self,
            node_id,
            node_type="task",
            dependencies=None,
            origin="user",
            metadata=None,
        ):
            self.id = node_id
            self.dependencies = set(dependencies or [])
            self.children = set()
            self.node_type = node_type

            self.status = "pending"  # pending / ready / running / done / failed / awaiting_input
            self.result = None

            self.origin = origin or "user"
            self.metadata = metadata or {}

            # -----------------------------
            # New semantic fields for planning
            # -----------------------------
            # List of data/resources this node requires (produced by other tasks)
            self.metadata.setdefault("required_input", [])
            # List of data/resources this node produces for downstream tasks
            self.metadata.setdefault("output", [])
            # Optional: group for parallel execution
            self.metadata.setdefault("parallel_group", None)
            # Optional: description / notes
            self.metadata.setdefault("description", "")
            self.metadata.setdefault("reflection_notes", [])

            self.metadata["required_input"] = self._coerce_io_list(
                self.metadata.get("required_input", [])
            )
            self.metadata["output"] = self._coerce_io_list(self.metadata.get("output", []))

        @staticmethod
        def _coerce_io_list(items) -> list:
            """Upgrade legacy slug strings to typed IO objects."""
            _FILE_EXTENSIONS = {
                ".md",
                ".txt",
                ".py",
                ".json",
                ".csv",
                ".html",
                ".yaml",
                ".xml",
            }
            result = []
            for item in items:
                if isinstance(item, str):
                    t = (
                        "file"
                        if any(item.endswith(ext) for ext in _FILE_EXTENSIONS)
                        else "document"
                    )
                    result.append(
                        {
                            "name": item,
                            "type": t,
                            "description": item.replace("_", " "),
                        }
                    )
                else:
                    result.append(item)
            return result

        def reset(self):
            self.status = "pending"
            self.result = None
            # FIX: Clear retry metadata so that any RESET_NODE event — whether
            # triggered by a user edit from the UI, an LLM-pause reset, or a
            # subtree reset — does not leave stale counts behind.  Without this,
            # a node that has already retried N-1 times and is then manually
            # reset by the user would be permanently failed on its very next
            # verification failure, bypassing all configured retries.
            #
            # The orchestrator's own retry path (in _on_node_done) writes these
            # keys AFTER calling RESET_NODE, so they survive the reset and
            # correctly track progress across genuine retry cycles.
            self.metadata.pop("retry_count", None)
            self.metadata.pop("retry_after", None)
            self.metadata.pop("verification_failure", None)
            self.metadata.pop("verified", None)

        def to_dict(self):
            return {
                "id": self.id,
                "dependencies": list(self.dependencies),
                "children": list(self.children),
                "status": self.status,
                "result": self.result,
                "origin": self.origin,
                "metadata": self.metadata,
            }

    # --------------------------------------------------

    def __init__(self):
        self.nodes = {}
        self.structure_version = 0
        self.execution_version = 0

    # --------------------------------------------------
    # Node Management
    # --------------------------------------------------

    def add_node(self, node_id, node_type="task", dependencies=None, origin="user", metadata=None):
        if node_id in self.nodes:
            return

        dependencies = dependencies or []

        self.nodes[node_id] = self.Node(
            node_id=node_id,
            node_type=node_type,
            dependencies=dependencies,
            origin=origin,
            metadata=metadata,
        )

        for dep in dependencies:
            if dep in self.nodes:
                self.nodes[dep].children.add(node_id)

    # --------------------------------------------------

    def remove_node(self, node_id):
        if node_id not in self.nodes:
            return

        # Collect all nodes to remove via iterative BFS instead of recursion
        to_remove = []
        queue = [node_id]
        visited = set()

        while queue:
            current = queue.pop()
            if current in visited or current not in self.nodes:
                continue
            visited.add(current)
            to_remove.append(current)
            queue.extend(self.nodes[current].children)

        # Remove in reverse order (leaves first)
        for nid in reversed(to_remove):
            if nid not in self.nodes:
                continue
            node = self.nodes[nid]
            # Unlink from parents
            for dep in node.dependencies:
                if dep in self.nodes:
                    self.nodes[dep].children.discard(nid)
            # Unlink from children
            for child in node.children:
                if child in self.nodes:
                    self.nodes[child].dependencies.discard(nid)
            del self.nodes[nid]

    # --------------------------------------------------

    def add_dependency(self, node_id, depends_on):
        if node_id not in self.nodes or depends_on not in self.nodes:
            return

        if self._would_create_cycle(node_id, depends_on):
            logger.warning("Cycle blocked: %s -> %s", node_id, depends_on)
            return

        self.nodes[node_id].dependencies.add(depends_on)
        self.nodes[depends_on].children.add(node_id)

    # --------------------------------------------------

    def remove_dependency(self, node_id, depends_on):
        if node_id not in self.nodes:
            return

        self.nodes[node_id].dependencies.discard(depends_on)
        if depends_on in self.nodes:
            self.nodes[depends_on].children.discard(node_id)

    # --------------------------------------------------
    # Readiness / Execution
    # --------------------------------------------------

    def recompute_readiness(self):
        for node in self.nodes.values():
            if node.status in (
                "done",
                "running",
                "failed",
                "to_be_expanded",
                "awaiting_input",
            ):
                continue
            # awaiting_input deps are treated like failed — not satisfied
            if all(
                dep in self.nodes and self.nodes[dep].status == "done" for dep in node.dependencies
            ):
                node.status = "ready"
            else:
                node.status = "pending"

    def recompute_readiness_for(self, node_id: str) -> None:
        """
        Incremental readiness update for MARK_DONE events.

        Re-evaluates the direct children of *node_id* (which can become newly
        ready) AND any node currently marked "ready" whose dependencies are no
        longer all "done" (which can regress to "pending" if a sibling was
        reset or removed while this node was completing).

        This two-pass approach is faster than a full graph scan for typical
        MARK_DONE events while still catching the regression cases that the
        original children-only implementation missed.

        Falls back gracefully when the node is not present (already removed).
        """
        _SKIP = frozenset(("done", "running", "failed", "to_be_expanded", "awaiting_input"))

        node = self.nodes.get(node_id)
        if node is None:
            return

        # Pass 1 — promote children of the completed node.
        for child_id in node.children:
            child = self.nodes.get(child_id)
            if child is None or child.status in _SKIP:
                continue
            if all(
                dep in self.nodes and self.nodes[dep].status == "done" for dep in child.dependencies
            ):
                child.status = "ready"
            else:
                child.status = "pending"

        # Pass 2 — catch regressions: a node that was already "ready" may have
        # had one of its other dependencies reset or removed concurrently.
        # Scan only the currently-ready nodes (a small subset of the graph).
        for other in list(self.nodes.values()):
            if other.status != "ready":
                continue
            if not all(
                dep in self.nodes and self.nodes[dep].status == "done" for dep in other.dependencies
            ):
                other.status = "pending"

    # --------------------------------------------------
    # Snapshot
    # --------------------------------------------------

    def get_snapshot(self):
        """
        FIX #9: Return a lightweight snapshot that avoids a full deep-copy.

        Strings (``result``) are immutable — no copy needed.
        ``dependencies`` and ``children`` are copied as frozen sets so callers
        can't mutate the graph's sets.
        ``metadata`` is shallow-copied; nested lists/dicts are shared but
        callers treat snapshots as read-only so this is safe in practice.
        """
        result = {}
        for nid, node in self.nodes.items():
            snap = object.__new__(self.Node)
            snap.id = node.id
            snap.node_type = node.node_type
            snap.status = node.status
            snap.result = node.result  # str is immutable
            snap.origin = node.origin
            snap.dependencies = frozenset(node.dependencies)
            snap.children = frozenset(node.children)
            snap.metadata = copy.copy(node.metadata)  # shallow copy
            result[nid] = snap
        return result

    # --------------------------------------------------
    # Utility
    # --------------------------------------------------

    def get_ready_nodes(self):
        return [node for node in self.nodes.values() if node.status == "ready"]

    # --------------------------------------------------

    def _would_create_cycle(self, node_id, depends_on):
        """
        FIX #2: Iterative DFS to detect cycles — replaces the old recursive
        inner function that could hit Python's recursion limit on deep graphs.
        """
        if depends_on not in self.nodes or node_id not in self.nodes:
            return False

        # Walk upstream from `depends_on`; if we reach `node_id` a cycle exists.
        visited: set[str] = set()
        stack = [depends_on]
        while stack:
            current = stack.pop()
            if current == node_id:
                return True
            if current in visited:
                continue
            visited.add(current)
            node = self.nodes.get(current)
            if node:
                stack.extend(node.dependencies)
        return False

    # --------------------------------------------------
    # Branch / Descendants
    # --------------------------------------------------

    def get_branch(self, root_id):
        """
        Return all nodes reachable *upstream* from ``root_id`` (inclusive),
        by walking through each node's ``dependencies`` set.

        FIX #10: The method name "get_branch" conventionally implies a
        downstream (child-ward) walk, but this implementation walks *upstream*
        toward prerequisites — i.e. it returns the full dependency closure of
        the given node, not its descendants.  The docstring and inline comments
        have been corrected to match the actual behaviour.  Callers that need
        downstream descendants should use ``node.children`` or a BFS over it.
        """
        if root_id not in self.nodes:
            return {}

        branch_nodes = {}
        stack = [root_id]

        while stack:
            current_id = stack.pop()
            if current_id in branch_nodes:
                continue

            node = self.nodes[current_id]
            branch_nodes[current_id] = node

            # Walk upstream toward prerequisites (following dependencies, not children)
            stack.extend(node.dependencies)

        return branch_nodes

    def detach_node(self, node_id):
        """Remove a single node without touching its children or descendants."""
        if node_id not in self.nodes:
            return

        # Remove this node from its parents' children sets
        for dep in self.nodes[node_id].dependencies:
            if dep in self.nodes:
                self.nodes[dep].children.discard(node_id)

        # Remove this node from its children's dependency sets
        for child in self.nodes[node_id].children:
            if child in self.nodes:
                self.nodes[child].dependencies.discard(node_id)

        del self.nodes[node_id]

    def update_status(self, node_id, status):
        if node_id not in self.nodes:
            return
        valid = (
            "pending",
            "ready",
            "running",
            "done",
            "failed",
            "to_be_expanded",
            "awaiting_input",
        )
        if status not in valid:
            logger.warning("Invalid status '%s' for node %s", status, node_id)
            return
        self.nodes[node_id].status = status
        self.execution_version += 1
