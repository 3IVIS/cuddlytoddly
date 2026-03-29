"""Tests for cuddlytoddly.core.task_graph."""
import pytest
from cuddlytoddly.core.task_graph import TaskGraph
from conftest import add_node, mark_done


# ── add_node ──────────────────────────────────────────────────────────────────

class TestAddNode:
    def test_adds_node(self, graph):
        add_node(graph, "a")
        assert "a" in graph.nodes

    def test_default_status_is_ready_when_no_deps(self, graph):
        add_node(graph, "a")
        assert graph.nodes["a"].status == "ready"

    def test_pending_when_dep_not_done(self, graph):
        add_node(graph, "a")
        add_node(graph, "b", deps=["a"])
        assert graph.nodes["b"].status == "pending"

    def test_duplicate_node_is_ignored(self, graph):
        add_node(graph, "a")
        add_node(graph, "a")
        assert len(graph.nodes) == 1

    def test_children_wired_to_parents(self, graph):
        add_node(graph, "parent")
        add_node(graph, "child", deps=["parent"])
        assert "child" in graph.nodes["parent"].children

    def test_node_type_stored(self, graph):
        add_node(graph, "g", node_type="goal")
        assert graph.nodes["g"].node_type == "goal"

    def test_metadata_stored(self, graph):
        add_node(graph, "a", metadata={"description": "hello"})
        assert graph.nodes["a"].metadata["description"] == "hello"

    def test_missing_dep_does_not_crash(self, graph):
        graph.add_node("b", dependencies=["nonexistent"])
        assert "b" in graph.nodes
        assert graph.nodes["b"].status == "pending"


# ── remove_node ───────────────────────────────────────────────────────────────

class TestRemoveNode:
    def test_removes_single_node(self, graph):
        add_node(graph, "a")
        graph.remove_node("a")
        assert "a" not in graph.nodes

    def test_cascades_to_children(self, graph):
        add_node(graph, "a")
        add_node(graph, "b", deps=["a"])
        add_node(graph, "c", deps=["b"])
        graph.remove_node("a")
        assert "a" not in graph.nodes
        assert "b" not in graph.nodes
        assert "c" not in graph.nodes

    def test_unlinks_from_parents(self, graph):
        add_node(graph, "parent")
        add_node(graph, "child", deps=["parent"])
        graph.remove_node("child")
        assert "child" not in graph.nodes["parent"].children

    def test_remove_nonexistent_is_noop(self, graph):
        graph.remove_node("ghost")

    def test_sibling_survives_removal(self, graph):
        add_node(graph, "parent")
        add_node(graph, "child_a", deps=["parent"])
        add_node(graph, "child_b", deps=["parent"])
        graph.remove_node("child_a")
        assert "child_b" in graph.nodes


# ── add_dependency / remove_dependency ───────────────────────────────────────

class TestDependencies:
    def test_add_dependency(self, graph):
        add_node(graph, "a")
        add_node(graph, "b")
        graph.add_dependency("b", "a")
        assert "a" in graph.nodes["b"].dependencies
        assert "b" in graph.nodes["a"].children

    def test_add_dependency_blocks_cycle(self, graph):
        """Requires source fix: _would_create_cycle must follow .dependencies not .children."""
        add_node(graph, "a")
        add_node(graph, "b", deps=["a"])
        # b→a exists. Adding a→b would make a→b→a cycle.
        graph.add_dependency("a", "b")
        assert "b" not in graph.nodes["a"].dependencies

    def test_add_dependency_missing_node_is_noop(self, graph):
        add_node(graph, "a")
        graph.add_dependency("a", "ghost")

    def test_remove_dependency(self, graph):
        add_node(graph, "a")
        add_node(graph, "b", deps=["a"])
        graph.remove_dependency("b", "a")
        assert "a" not in graph.nodes["b"].dependencies
        assert "b" not in graph.nodes["a"].children

    def test_remove_nonexistent_dep_is_noop(self, graph):
        add_node(graph, "a")
        graph.remove_dependency("a", "ghost")


# ── recompute_readiness ───────────────────────────────────────────────────────

class TestReadiness:
    def test_ready_when_all_deps_done(self, graph):
        add_node(graph, "a")
        add_node(graph, "b", deps=["a"])
        assert graph.nodes["b"].status == "pending"
        mark_done(graph, "a")
        graph.recompute_readiness()
        assert graph.nodes["b"].status == "ready"

    def test_still_pending_when_one_dep_not_done(self, graph):
        add_node(graph, "a")
        add_node(graph, "b")
        add_node(graph, "c", deps=["a", "b"])
        mark_done(graph, "a")
        graph.recompute_readiness()
        assert graph.nodes["c"].status == "pending"

    def test_done_node_stays_done(self, graph):
        add_node(graph, "a")
        mark_done(graph, "a")
        graph.recompute_readiness()
        assert graph.nodes["a"].status == "done"

    def test_running_node_stays_running(self, graph):
        add_node(graph, "a")
        graph.nodes["a"].status = "running"
        graph.recompute_readiness()
        assert graph.nodes["a"].status == "running"

    def test_failed_node_stays_failed(self, graph):
        add_node(graph, "a")
        graph.nodes["a"].status = "failed"
        graph.recompute_readiness()
        assert graph.nodes["a"].status == "failed"


# ── get_snapshot ──────────────────────────────────────────────────────────────

class TestSnapshot:
    def test_returns_all_nodes(self, linear_graph):
        snap = linear_graph.get_snapshot()
        assert set(snap.keys()) == {"task_a", "task_b", "goal"}

    def test_snapshot_is_deep_copy(self, graph):
        add_node(graph, "a")
        snap = graph.get_snapshot()
        snap["a"].status = "running"
        assert graph.nodes["a"].status != "running"

    def test_empty_graph_returns_empty_dict(self, graph):
        assert graph.get_snapshot() == {}


# ── get_ready_nodes ───────────────────────────────────────────────────────────

class TestGetReadyNodes:
    def test_returns_ready_nodes(self, graph):
        add_node(graph, "a")
        add_node(graph, "b", deps=["a"])
        ready = graph.get_ready_nodes()
        assert len(ready) == 1
        assert ready[0].id == "a"

    def test_returns_multiple_ready(self, graph):
        add_node(graph, "a")
        add_node(graph, "b")
        ready_ids = {n.id for n in graph.get_ready_nodes()}
        assert ready_ids == {"a", "b"}

    def test_empty_when_nothing_ready(self, graph):
        add_node(graph, "a")
        add_node(graph, "b", deps=["a"])
        graph.nodes["a"].status = "running"
        graph.recompute_readiness()
        assert graph.get_ready_nodes() == []


# ── detach_node ───────────────────────────────────────────────────────────────

class TestDetachNode:
    def test_detach_removes_only_that_node(self, graph):
        add_node(graph, "a")
        add_node(graph, "b", deps=["a"])
        add_node(graph, "c", deps=["b"])
        graph.detach_node("b")
        assert "b" not in graph.nodes
        assert "a" in graph.nodes
        assert "c" in graph.nodes

    def test_detach_unlinks_both_directions(self, graph):
        add_node(graph, "a")
        add_node(graph, "b", deps=["a"])
        graph.detach_node("b")
        assert "b" not in graph.nodes["a"].children

    def test_detach_child_loses_dep(self, graph):
        add_node(graph, "a")
        add_node(graph, "b", deps=["a"])
        add_node(graph, "c", deps=["b"])
        graph.detach_node("b")
        assert "b" not in graph.nodes["c"].dependencies

    def test_detach_nonexistent_is_noop(self, graph):
        graph.detach_node("ghost")


# ── update_status ─────────────────────────────────────────────────────────────

class TestUpdateStatus:
    def test_valid_status_set(self, graph):
        add_node(graph, "a")
        graph.update_status("a", "running")
        assert graph.nodes["a"].status == "running"

    def test_invalid_status_rejected(self, graph):
        add_node(graph, "a")
        graph.update_status("a", "banana")
        assert graph.nodes["a"].status != "banana"

    def test_bumps_execution_version(self, graph):
        add_node(graph, "a")
        v = graph.execution_version
        graph.update_status("a", "done")
        assert graph.execution_version > v


# ── version counters ──────────────────────────────────────────────────────────

class TestVersions:
    def test_structure_version_increments_on_add(self):
        from cuddlytoddly.core.events import Event, ADD_NODE
        from cuddlytoddly.core.reducer import apply_event
        g = TaskGraph()
        v = g.structure_version
        apply_event(g, Event(ADD_NODE, {
            "node_id": "x", "node_type": "task",
            "dependencies": [], "metadata": {},
        }))
        assert g.structure_version > v

    def test_execution_version_increments_on_mark_done(self, graph):
        add_node(graph, "a")
        v = graph.execution_version
        mark_done(graph, "a")
        assert graph.execution_version > v


# ── get_branch ────────────────────────────────────────────────────────────────

class TestGetBranch:
    def test_branch_includes_root_and_ancestors(self, graph):
        add_node(graph, "a")
        add_node(graph, "b", deps=["a"])
        add_node(graph, "c", deps=["b"])
        branch = graph.get_branch("c")
        assert set(branch.keys()) == {"a", "b", "c"}

    def test_branch_excludes_unrelated_nodes(self, graph):
        add_node(graph, "a")
        add_node(graph, "b", deps=["a"])
        add_node(graph, "unrelated")
        branch = graph.get_branch("b")
        assert "unrelated" not in branch

    def test_branch_on_missing_node_returns_empty(self, graph):
        branch = graph.get_branch("ghost")
        assert branch == {}


# ── cycle detection ───────────────────────────────────────────────────────────
# These tests verify the fixed _would_create_cycle implementation.
# Requires source fix: change `children` to `dependencies` in _would_create_cycle.

class TestCycleDetection:
    def test_direct_cycle_blocked(self, graph):
        """b depends on a. Adding a depends on b would make a→b→a — must be blocked."""
        add_node(graph, "a")
        add_node(graph, "b", deps=["a"])
        graph.add_dependency("a", "b")
        assert "b" not in graph.nodes["a"].dependencies

    def test_indirect_cycle_blocked(self, graph):
        """c→b→a. Adding a→c would close the cycle — must be blocked."""
        add_node(graph, "a")
        add_node(graph, "b", deps=["a"])
        add_node(graph, "c", deps=["b"])
        graph.add_dependency("a", "c")
        assert "c" not in graph.nodes["a"].dependencies

    def test_non_cycle_dependency_allowed(self, graph):
        """a and b are independent. Adding b depends on a is fine."""
        add_node(graph, "a")
        add_node(graph, "b")
        graph.add_dependency("b", "a")
        assert "a" in graph.nodes["b"].dependencies
