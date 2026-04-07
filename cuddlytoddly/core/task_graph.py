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

            def _coerce_io_list(items):
                """Upgrade legacy slug strings to typed IO objects."""
                result = []
                for item in items:
                    if isinstance(item, str):
                        # Infer type from extension
                        t = "file" if any(item.endswith(ext) for ext in
                                        {".md",".txt",".py",".json",".csv",".html",".yaml",".xml"}) \
                            else "document"
                        result.append({"name": item, "type": t, "description": item.replace("_", " ")})
                    else:
                        result.append(item)
                return result

            self.metadata["required_input"] = _coerce_io_list(self.metadata.get("required_input", []))
            self.metadata["output"]         = _coerce_io_list(self.metadata.get("output", []))

        def reset(self):
            self.status = "pending"
            self.result = None

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

    def add_node(
        self, node_id, node_type="task", dependencies=None, origin="user", metadata=None
    ):
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
            if node.status in ("done", "running", "failed", "to_be_expanded", "awaiting_input"):
                continue
            # awaiting_input deps are treated like failed — not satisfied
            if all(
                dep in self.nodes
                and self.nodes[dep].status == "done"
                for dep in node.dependencies
            ):
                node.status = "ready"
            else:
                node.status = "pending"

    # --------------------------------------------------
    # Snapshot
    # --------------------------------------------------

    def get_snapshot(self):
        return copy.deepcopy(self.nodes)

    # --------------------------------------------------
    # Utility
    # --------------------------------------------------

    def get_ready_nodes(self):
        return [node for node in self.nodes.values() if node.status == "ready"]

    # --------------------------------------------------

    def _would_create_cycle(self, node_id, depends_on):
        if depends_on not in self.nodes or node_id not in self.nodes:
            return False

        visited = set()

        def dfs(n):
            if n == node_id:
                return True
            visited.add(n)
            for dep in self.nodes[n].dependencies:   # ← was: children
                if dep not in visited and dfs(dep):
                    return True
            return False

        return dfs(depends_on)

    # --------------------------------------------------
    # Branch / Descendants
    # --------------------------------------------------

    def get_branch(self, root_id):
        """
        Returns all nodes reachable from the root node (inclusive),
        walking upstream through dependencies.
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

            # Walk upstream (toward prerequisites)
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
        valid = ("pending", "ready", "running", "done", "failed", "to_be_expanded", "awaiting_input")
        if status not in valid:
            logger.warning("Invalid status '%s' for node %s", status, node_id)
            return
        self.nodes[node_id].status = status
        self.execution_version += 1




