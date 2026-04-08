"""Tests for cuddlytoddly.engine.quality_gate.QualityGate."""

import json
import os
from unittest.mock import MagicMock

from conftest import FakeLLM, add_node

from cuddlytoddly.core.task_graph import TaskGraph
from cuddlytoddly.engine.quality_gate import QualityGate

# ── Helpers ───────────────────────────────────────────────────────────────────


def make_node_with_outputs(outputs, node_id="task_1"):
    g = TaskGraph()
    add_node(
        g,
        node_id,
        metadata={
            "description": "do something",
            "output": outputs,
            "required_input": [],
        },
    )
    return g.nodes[node_id], g.get_snapshot()


def satisfied_response(reason="looks good"):
    return json.dumps({"satisfied": True, "reason": reason})


def unsatisfied_response(reason="missing content"):
    return json.dumps({"satisfied": False, "reason": reason})


def ok_dep_response():
    return json.dumps({"ok": True})


def gap_dep_response(bridge_id="bridge", bridge_desc="do the thing", bridge_output="output.md"):
    return json.dumps(
        {
            "ok": False,
            "missing": "something important",
            "bridge_node": {
                "node_id": bridge_id,
                "description": bridge_desc,
                "output": bridge_output,
            },
        }
    )


# ── verify_result ─────────────────────────────────────────────────────────────


class TestVerifyResult:
    def test_satisfied_when_no_declared_outputs(self):
        llm = FakeLLM(unsatisfied_response())
        gate = QualityGate(llm_client=llm)
        node, snap = make_node_with_outputs([])
        ok, reason = gate.verify_result(node, "some result", snap)
        assert ok is True
        assert llm.calls == []

    def test_satisfied_response_from_llm(self):
        llm = FakeLLM(satisfied_response("all good"))
        gate = QualityGate(llm_client=llm)
        # Use "document" type — a "file" type triggers the disk existence check
        # which would fail in a test environment where no file is written to disk.
        node, snap = make_node_with_outputs(
            [{"name": "salary_report", "type": "document", "description": "the report"}]
        )
        ok, reason = gate.verify_result(node, "Here is the full report content...", snap)
        assert ok is True
        assert "all good" in reason

    def test_unsatisfied_response_from_llm(self):
        llm = FakeLLM(unsatisfied_response("only a label"))
        gate = QualityGate(llm_client=llm)
        node, snap = make_node_with_outputs(
            [{"name": "report.md", "type": "file", "description": "the report"}]
        )
        ok, reason = gate.verify_result(node, "report.md", snap)
        assert ok is False

    def test_bare_filename_that_does_not_exist_fails(self):
        """A bare filename result is checked for existence; missing file → fail."""
        llm = FakeLLM(satisfied_response())
        gate = QualityGate(llm_client=llm)
        node, snap = make_node_with_outputs(
            [{"name": "output.txt", "type": "file", "description": "the file"}]
        )
        ok, reason = gate.verify_result(node, "output.txt", snap)
        assert ok is False

    def test_bare_filename_matching_output_name_fails_label_check(self, tmp_path):
        """A bare result that exactly matches the declared output name is treated as
        a label stub, not actual content — even if the file exists on disk.
        The is_just_label heuristic was removed; the LLM verifier now catches this
        via the prompt instruction about label stubs."""
        real_file = tmp_path / "output.txt"
        real_file.write_text("content")
        # LLM verifier rejects the bare label — this is what a real LLM would do
        # given the prompt instruction "A result that is just a filename, a single
        # word, or a name matching the output label is NOT satisfied."
        llm = FakeLLM(unsatisfied_response("result is just a label matching the output name"))
        gate = QualityGate(llm_client=llm)
        node, snap = make_node_with_outputs(
            [{"name": "output.txt", "type": "file", "description": "the file"}]
        )
        old_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            ok, reason = gate.verify_result(node, "output.txt", snap)
        finally:
            os.chdir(old_dir)
        # File exists (disk check passes), but LLM rejects the bare label result
        assert ok is False

    def test_file_written_prefix_result_with_existing_file_passes(self, tmp_path):
        """A 'file_written: filename' result where the file exists should pass."""
        real_file = tmp_path / "report.md"
        real_file.write_text("# My report\n\nContent here.")
        llm = FakeLLM(satisfied_response())
        gate = QualityGate(llm_client=llm)
        node, snap = make_node_with_outputs(
            [{"name": "report.md", "type": "file", "description": "the report"}]
        )
        old_dir = os.getcwd()
        try:
            os.chdir(tmp_path)
            ok, _ = gate.verify_result(
                node, "file_written: report.md\nsummary: full content here", snap
            )
        finally:
            os.chdir(old_dir)
        assert ok is True

    def test_label_matching_output_name_fails(self):
        """Result that is just the output name string is rejected."""
        llm = FakeLLM(unsatisfied_response("just a label"))
        gate = QualityGate(llm_client=llm)
        node, snap = make_node_with_outputs(
            [
                {
                    "name": "investment_report",
                    "type": "document",
                    "description": "the report",
                }
            ]
        )
        ok, reason = gate.verify_result(node, "investment_report", snap)
        assert ok is False

    def test_skipped_when_llm_is_stopped(self):
        llm = FakeLLM(unsatisfied_response())
        llm.stop()
        gate = QualityGate(llm_client=llm)
        node, snap = make_node_with_outputs(
            [{"name": "report.md", "type": "file", "description": "r"}]
        )
        ok, reason = gate.verify_result(node, "something", snap)
        assert ok is True
        assert "paused" in reason.lower() or "skipped" in reason.lower()

    def test_llm_exception_returns_true(self):
        def bad_ask(prompt, schema=None):
            raise RuntimeError("network error")

        llm = MagicMock()
        llm.is_stopped = False
        llm.ask = bad_ask
        gate = QualityGate(llm_client=llm)
        # Use "document" type — a "file" type triggers the disk existence check
        # before the LLM call, which would return False rather than True on exception.
        node, snap = make_node_with_outputs(
            [{"name": "result", "type": "document", "description": "r"}]
        )
        ok, reason = gate.verify_result(node, "some substantive content here " * 10, snap)
        assert ok is True

    def test_file_label_pattern_checks_disk(self, tmp_path):
        """file_written: ghost.md → check disk → file missing → fail."""
        llm = FakeLLM(satisfied_response())
        gate = QualityGate(llm_client=llm)
        node, snap = make_node_with_outputs(
            [{"name": "foo.md", "type": "file", "description": "foo"}]
        )
        result = "file_written: ghost.md\nsummary: stuff"
        ok, reason = gate.verify_result(node, result, snap)
        assert ok is False
        assert "does not exist" in reason


# ── check_dependencies ────────────────────────────────────────────────────────


class TestCheckDependencies:
    def test_returns_none_when_ok(self):
        llm = FakeLLM(ok_dep_response())
        gate = QualityGate(llm_client=llm)
        g = TaskGraph()
        add_node(g, "task_1")
        snap = g.get_snapshot()
        bridge = gate.check_dependencies(g.nodes["task_1"], snap)
        assert bridge is None

    def test_returns_bridge_when_gap_found(self):
        llm = FakeLLM(gap_dep_response("fill_gap", "Gather missing data", "data.json"))
        gate = QualityGate(llm_client=llm)
        g = TaskGraph()
        add_node(g, "task_1")
        snap = g.get_snapshot()
        bridge = gate.check_dependencies(g.nodes["task_1"], snap)
        assert bridge is not None
        assert bridge["node_id"] == "fill_gap"
        assert bridge["description"] == "Gather missing data"

    def test_returns_none_when_llm_stopped(self):
        llm = FakeLLM(gap_dep_response())
        llm.stop()
        gate = QualityGate(llm_client=llm)
        g = TaskGraph()
        add_node(g, "task_1")
        snap = g.get_snapshot()
        bridge = gate.check_dependencies(g.nodes["task_1"], snap)
        assert bridge is None

    def test_returns_none_on_llm_exception(self):
        llm = MagicMock()
        llm.is_stopped = False
        llm.ask = MagicMock(side_effect=RuntimeError("fail"))
        gate = QualityGate(llm_client=llm)
        g = TaskGraph()
        add_node(g, "task_1")
        snap = g.get_snapshot()
        bridge = gate.check_dependencies(g.nodes["task_1"], snap)
        assert bridge is None

    def test_returns_none_when_bridge_node_missing_required_fields(self):
        incomplete_response = json.dumps(
            {"ok": False, "missing": "something", "bridge_node": {"node_id": "x"}}
        )
        llm = FakeLLM(incomplete_response)
        gate = QualityGate(llm_client=llm)
        g = TaskGraph()
        add_node(g, "task_1")
        snap = g.get_snapshot()
        bridge = gate.check_dependencies(g.nodes["task_1"], snap)
        assert bridge is None

    def test_includes_upstream_results_in_prompt(self):
        prompts = []

        def capture(prompt, schema=None):
            prompts.append(prompt)
            return ok_dep_response()

        llm = FakeLLM(capture)
        gate = QualityGate(llm_client=llm)
        g = TaskGraph()
        add_node(g, "upstream")
        add_node(g, "task_1", deps=["upstream"])
        g.nodes["upstream"].result = "upstream output data"
        g.nodes["upstream"].status = "done"
        snap = g.get_snapshot()
        gate.check_dependencies(g.nodes["task_1"], snap)
        assert any("upstream output data" in p for p in prompts)
