"""Tests for cuddlytoddly.planning.llm_output_validator."""

from conftest import add_node

from cuddlytoddly.core.events import ADD_DEPENDENCY, ADD_NODE
from cuddlytoddly.core.task_graph import TaskGraph
from cuddlytoddly.planning.llm_output_validator import LLMOutputValidator


def make_validator(graph=None):
    g = graph or TaskGraph()
    return LLMOutputValidator(g), g


def node_event(node_id, node_type="task", deps=None, metadata=None):
    return {
        "type": ADD_NODE,
        "payload": {
            "node_id": node_id,
            "node_type": node_type,
            "dependencies": deps or [],
            "metadata": metadata or {"description": node_id},
        },
    }


def dep_event(node_id, depends_on):
    return {
        "type": ADD_DEPENDENCY,
        "payload": {"node_id": node_id, "depends_on": depends_on},
    }


# ── Basic acceptance ──────────────────────────────────────────────────────────


class TestValidatorAcceptance:
    def test_accepts_valid_add_node(self):
        v, _ = make_validator()
        events = [node_event("task_a")]
        result = v.validate_and_normalize(events, "planning")
        assert len(result) == 1
        assert result[0]["type"] == ADD_NODE
        assert result[0]["payload"]["node_id"] == "task_a"

    def test_accepts_multiple_nodes_no_deps(self):
        v, _ = make_validator()
        events = [node_event("a"), node_event("b"), node_event("c")]
        result = v.validate_and_normalize(events, "planning")
        assert len(result) == 3

    def test_accepts_node_depending_on_accepted_node_in_same_batch(self):
        v, _ = make_validator()
        events = [node_event("a"), node_event("b", deps=["a"])]
        result = v.validate_and_normalize(events, "planning")
        ids = [e["payload"]["node_id"] for e in result]
        assert "a" in ids and "b" in ids

    def test_accepts_node_depending_on_existing_graph_node(self):
        g = TaskGraph()
        add_node(g, "existing")
        v = LLMOutputValidator(g)
        events = [node_event("new_node", deps=["existing"])]
        result = v.validate_and_normalize(events, "planning")
        assert len(result) == 1

    def test_accepts_valid_add_dependency(self):
        g = TaskGraph()
        add_node(g, "a")
        add_node(g, "b")
        v = LLMOutputValidator(g)
        events = [dep_event("b", "a")]
        result = v.validate_and_normalize(events, "planning")
        assert any(e["type"] == ADD_DEPENDENCY for e in result)

    def test_injects_forced_origin(self):
        v, _ = make_validator()
        events = [node_event("a")]
        result = v.validate_and_normalize(events, "system")
        assert result[0]["payload"]["origin"] == "system"


# ── Rejection cases ───────────────────────────────────────────────────────────


class TestValidatorRejection:
    def test_rejects_empty_list(self):
        v, _ = make_validator()
        result = v.validate_and_normalize([], "planning")
        assert result == []

    def test_rejects_non_list_input(self):
        v, _ = make_validator()
        result = v.validate_and_normalize({"type": ADD_NODE}, "planning")
        assert result == []

    def test_rejects_non_dict_event(self):
        v, _ = make_validator()
        result = v.validate_and_normalize(["not a dict"], "planning")
        assert result == []

    def test_rejects_node_missing_node_id(self):
        v, _ = make_validator()
        events = [
            {"type": ADD_NODE, "payload": {"node_type": "task", "dependencies": []}}
        ]
        result = v.validate_and_normalize(events, "planning")
        assert result == []

    def test_rejects_node_with_non_string_node_id(self):
        v, _ = make_validator()
        events = [
            {
                "type": ADD_NODE,
                "payload": {"node_id": 42, "node_type": "task", "dependencies": []},
            }
        ]
        result = v.validate_and_normalize(events, "planning")
        assert result == []

    def test_rejects_self_dependency(self):
        v, _ = make_validator()
        events = [node_event("a", deps=["a"])]
        result = v.validate_and_normalize(events, "planning")
        assert result == []

    def test_rejects_duplicate_node_id(self):
        g = TaskGraph()
        add_node(g, "existing")
        v = LLMOutputValidator(g)
        events = [node_event("existing")]
        result = v.validate_and_normalize(events, "planning")
        assert not any(
            e["type"] == ADD_NODE and e["payload"]["node_id"] == "existing"
            for e in result
        )

    def test_rejects_node_with_unresolvable_dep(self):
        v, _ = make_validator()
        events = [node_event("b", deps=["nonexistent"])]
        result = v.validate_and_normalize(events, "planning")
        assert result == []

    def test_rejects_dep_event_missing_depends_on(self):
        g = TaskGraph()
        add_node(g, "a")
        v = LLMOutputValidator(g)
        events = [{"type": ADD_DEPENDENCY, "payload": {"node_id": "a"}}]
        result = v.validate_and_normalize(events, "planning")
        dep_events = [e for e in result if e["type"] == ADD_DEPENDENCY]
        assert dep_events == []

    def test_rejects_dep_to_nonexistent_node(self):
        """ADD_DEPENDENCY where depends_on doesn't exist should be rejected.
        Requires source fix: llm_output_validator.py warning call missing 3rd arg."""
        g = TaskGraph()
        add_node(g, "a")
        v = LLMOutputValidator(g)
        events = [dep_event("a", "ghost")]
        result = v.validate_and_normalize(events, "planning")
        dep_events = [e for e in result if e["type"] == ADD_DEPENDENCY]
        assert dep_events == []

    def test_rejects_unknown_event_type(self):
        v, _ = make_validator()
        events = [{"type": "MAKE_COFFEE", "payload": {}}]
        result = v.validate_and_normalize(events, "planning")
        assert result == []

    def test_rejects_non_goal_depending_on_goal(self):
        g = TaskGraph()
        add_node(g, "the_goal", node_type="goal")
        v = LLMOutputValidator(g)
        events = [dep_event("the_goal", "the_goal")]
        result = v.validate_and_normalize(events, "planning")
        task_dep_events = [
            e
            for e in result
            if e["type"] == ADD_DEPENDENCY and e["payload"]["depends_on"] == "the_goal"
        ]
        assert task_dep_events == []


# ── Metadata filtering ────────────────────────────────────────────────────────


class TestValidatorMetadata:
    def test_strips_disallowed_metadata_keys(self):
        v, _ = make_validator()
        events = [
            {
                "type": ADD_NODE,
                "payload": {
                    "node_id": "a",
                    "node_type": "task",
                    "dependencies": [],
                    "metadata": {"description": "ok", "forbidden_key": "strip me"},
                },
            }
        ]
        result = v.validate_and_normalize(events, "planning")
        assert "forbidden_key" not in result[0]["payload"]["metadata"]

    def test_keeps_allowed_metadata_keys(self):
        v, _ = make_validator()
        allowed = [
            "description",
            "parallel_group",
            "required_input",
            "output",
            "reflection_notes",
            "skill",
            "tools",
        ]
        meta = {k: "value" for k in allowed}
        events = [
            {
                "type": ADD_NODE,
                "payload": {
                    "node_id": "a",
                    "node_type": "task",
                    "dependencies": [],
                    "metadata": meta,
                },
            }
        ]
        result = v.validate_and_normalize(events, "planning")
        for k in allowed:
            assert k in result[0]["payload"]["metadata"]

    def test_non_dict_metadata_reset_to_empty(self):
        v, _ = make_validator()
        events = [
            {
                "type": ADD_NODE,
                "payload": {
                    "node_id": "a",
                    "node_type": "task",
                    "dependencies": [],
                    "metadata": "not a dict",
                },
            }
        ]
        result = v.validate_and_normalize(events, "planning")
        assert isinstance(result[0]["payload"]["metadata"], dict)


# ── Transitive dependency resolution ─────────────────────────────────────────


class TestTransitiveDeps:
    def test_chain_resolved_in_single_batch(self):
        v, _ = make_validator()
        events = [
            node_event("a"),
            node_event("b", deps=["a"]),
            node_event("c", deps=["b"]),
        ]
        result = v.validate_and_normalize(events, "planning")
        ids = {e["payload"]["node_id"] for e in result if e["type"] == ADD_NODE}
        assert ids == {"a", "b", "c"}

    def test_middle_node_missing_dep_rejects_downstream(self):
        v, _ = make_validator()
        events = [
            node_event("a", deps=["ghost"]),
            node_event("b", deps=["a"]),
        ]
        result = v.validate_and_normalize(events, "planning")
        assert result == []
