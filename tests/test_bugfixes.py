"""
Tests for the bug fixes and the broadened-execution feature:

  broadened execution — when required inputs are missing, the executor runs
                        with a broadened description instead of blocking. The
                        orchestrator writes broadened_description metadata back
                        to the node and patches the clarification form with any
                        new fields so the user can optionally provide them.
  Issue 4 — tool results context folded into LLM verifier prompt
  Issue 5 — max_retries cap + exponential backoff on verification failure
  Issue 6 — tool error strings (no exception raised) are flagged as error=True
"""

import json
import time
from concurrent.futures import Future
from unittest.mock import MagicMock

from conftest import FakeLLM, add_node

from cuddlytoddly.engine.orchestrator import Orchestrator
from cuddlytoddly.engine.quality_gate import QualityGate
from cuddlytoddly.planning.llm_executor import LLMExecutor
from toddly.core.reducer import apply_event
from toddly.core.task_graph import TaskGraph
from toddly.infra.event_queue import EventQueue

# ── Shared helpers ────────────────────────────────────────────────────────────


def make_orchestrator(
    graph=None, executor_result="mock result", gate_satisfied=True, max_retries=5
):
    g = graph or TaskGraph()
    mock_planner = MagicMock()
    mock_planner.propose.return_value = []
    mock_executor = MagicMock()
    mock_executor.execute.return_value = executor_result
    mock_gate = MagicMock()
    mock_gate.verify_result.return_value = (gate_satisfied, "reason")
    mock_gate.check_dependencies.return_value = None

    orch = Orchestrator(
        graph=g,
        planner=mock_planner,
        executor=mock_executor,
        quality_gate=mock_gate,
        event_queue=EventQueue(),
        max_workers=1,
        max_retries=max_retries,
    )
    orch._llm_clients = []  # prevent llm_stopped from blocking execution
    return orch, g, mock_executor, mock_gate


def make_tool_registry(tools_dict):
    from toddly.skills.skill_loader import Tool, ToolRegistry

    registry = ToolRegistry()
    for name, fn in tools_dict.items():
        registry.register(
            Tool(
                name=name,
                description=f"Tool {name}",
                input_schema={"input": "string"},
                fn=fn,
            )
        )
    return registry


def make_reporter(parent_id, graph):
    import threading

    from toddly.engine.execution_step_reporter import ExecutionStepReporter

    lock = threading.RLock()
    apply_events = []

    def apply_fn(event):
        apply_event(graph, event)
        apply_events.append(event)

    return ExecutionStepReporter(
        parent_node_id=parent_id,
        apply_fn=apply_fn,
        graph_lock=lock,
        graph=graph,
    )


def done_response(result="all done"):
    return json.dumps({"done": True, "result": result})


def tool_response(name, args):
    return json.dumps({"done": False, "tool_call": {"name": name, "args": args}})


def unsatisfied():
    return json.dumps({"satisfied": False, "reason": "not good enough"})


# ── broadened execution: preflight, broadened description, metadata write ───────


class TestAwaitingInput:
    """
    Tests for the unified awaiting_input mechanism that replaced user_input
    node type.  Covers:
      - executor preflight emitting AwaitingInputSignal for personal tasks
      - executor preflight emitting AwaitingInputSignal for org tasks
      - executor preflight passing through when context is available
      - orchestrator intercepting AwaitingInputSignal → awaiting_input status
      - orchestrator patching clarification node with new fields
      - _resume_unblocked_pass auto-resumes when fields are filled
      - resume_node public API
      - regular tasks still go through verify_result
    """

    def _make_future(self, result):
        f = Future()
        f.set_result(result)
        return f

    # ── executor preflight ────────────────────────────────────────────────────

    def _make_exec_node(self, description, unknown_keys=(), known_keys=()):
        """Build a node + resolved_inputs list simulating a clarification dep."""
        g = TaskGraph()
        add_node(
            g,
            "task_1",
            metadata={
                "description": description,
                "output": [{"name": "out", "type": "document", "description": "output"}],
            },
        )
        add_node(g, "clar_1", node_type="clarification", metadata={"description": "context"})
        g.nodes["task_1"].dependencies.add("clar_1")

        unknown = [{"key": k, "label": k} for k in unknown_keys]
        known = [{"key": k, "label": k, "value": "some value"} for k in known_keys]

        resolved = [
            {
                "node_id": "clar_1",
                "description": "context",
                "declared_output": [],
                "result": "context text",
                "_unknown_fields": unknown,
                "_known_fields": known,
            }
        ]
        return g.nodes["task_1"], resolved

    def test_preflight_skipped_when_no_unknown_fields(self):
        """When all clarification fields are filled, no LLM call is made."""
        from cuddlytoddly.planning.llm_executor import LLMExecutor

        llm = MagicMock()
        executor = LLMExecutor(llm_client=llm, max_turns=3)
        node, resolved = self._make_exec_node(
            "List personal achievements relevant to the job",
            known_keys=["key_achievements"],  # all filled
        )
        result = executor._preflight_awaiting_input(node, resolved)
        assert result is None
        llm.ask.assert_not_called()

    def test_preflight_makes_llm_call_when_unknown_fields_exist(self):
        """When unknown fields exist, the LLM is consulted."""
        from cuddlytoddly.planning.llm_executor import AwaitingInputSignal, LLMExecutor

        llm = FakeLLM(
            json.dumps(
                {
                    "blocked": True,
                    "reason": "needs personal achievements",
                    "missing_fields": ["current_salary"],
                    "new_fields": [],
                }
            )
        )
        executor = LLMExecutor(llm_client=llm, max_turns=3)
        node, resolved = self._make_exec_node(
            "List personal achievements relevant to the job",
            unknown_keys=["current_salary"],
        )
        result = executor._preflight_awaiting_input(node, resolved)
        assert isinstance(result, AwaitingInputSignal)
        assert result.reason == "needs personal achievements"
        assert result.missing_fields == ["current_salary"]
        assert len(llm.calls) == 1

    def test_preflight_returns_none_when_llm_says_not_blocked(self):
        """When the LLM says blocked=false, preflight returns None."""
        from cuddlytoddly.planning.llm_executor import LLMExecutor

        llm = FakeLLM(
            json.dumps(
                {
                    "blocked": False,
                    "reason": "can use web search",
                    "missing_fields": [],
                    "new_fields": [],
                }
            )
        )
        executor = LLMExecutor(llm_client=llm, max_turns=3)
        node, resolved = self._make_exec_node(
            "Research current market salary rates",
            unknown_keys=["job_title"],
        )
        result = executor._preflight_awaiting_input(node, resolved)
        assert result is None

    def test_preflight_fails_open_on_llm_error(self):
        """If the LLM call fails, preflight returns None (fail open)."""
        from cuddlytoddly.planning.llm_executor import LLMExecutor

        llm = MagicMock()
        llm.is_stopped = False
        llm.ask = MagicMock(side_effect=RuntimeError("LLM unavailable"))
        executor = LLMExecutor(llm_client=llm, max_turns=3)
        node, resolved = self._make_exec_node(
            "List personal achievements",
            unknown_keys=["salary"],
        )
        result = executor._preflight_awaiting_input(node, resolved)
        assert result is None  # fail open

    def test_preflight_new_fields_passed_through(self):
        """new_fields from LLM response are propagated into the signal."""
        from cuddlytoddly.planning.llm_executor import AwaitingInputSignal, LLMExecutor

        new_field = {
            "key": "company_name",
            "label": "Company name",
            "value": "unknown",
            "rationale": "needed",
        }
        llm = FakeLLM(
            json.dumps(
                {
                    "blocked": True,
                    "reason": "needs company name",
                    "missing_fields": [],
                    "new_fields": [new_field],
                }
            )
        )
        executor = LLMExecutor(llm_client=llm, max_turns=3)
        node, resolved = self._make_exec_node(
            "Gather information on the company's financial situation",
            unknown_keys=["current_salary"],
        )
        result = executor._preflight_awaiting_input(node, resolved)
        assert isinstance(result, AwaitingInputSignal)
        assert result.new_fields == [new_field]

    # ── Phase 1: deterministic required-input gap detection ───────────────────

    def _make_exec_node_with_required_input(
        self, description, required_inputs, unknown_keys=(), known_keys=()
    ):
        """Build a node with declared required_input and a clarification dep."""
        g = TaskGraph()
        add_node(
            g,
            "task_1",
            metadata={
                "description": description,
                "output": [{"name": "out", "type": "document", "description": "output"}],
                "required_input": required_inputs,
            },
        )
        add_node(g, "clar_1", node_type="clarification", metadata={"description": "ctx"})
        g.nodes["task_1"].dependencies.add("clar_1")

        unknown = [{"key": k, "label": k} for k in unknown_keys]
        known = [{"key": k, "label": k, "value": "some value"} for k in known_keys]
        resolved = [
            {
                "node_id": "clar_1",
                "description": "ctx",
                "declared_output": [],
                "result": "ctx",
                "_unknown_fields": unknown,
                "_known_fields": known,
            }
        ]
        return g.nodes["task_1"], resolved

    def test_phase1_adds_uncovered_required_input_as_new_field(self):
        """A required_input whose name is not in any clarification field is
        detected and merged into new_fields via Phase 1 gap detection.
        The LLM is still called to generate the broadened description."""
        from cuddlytoddly.planning.llm_executor import LLMExecutor

        llm = FakeLLM(
            json.dumps(
                {
                    "blocked": True,
                    "reason": "needs company culture info",
                    "missing_fields": ["current_salary"],
                    "new_fields": [],
                    "broadened_description": "Determine general negotiation strategy frameworks.",
                    "broadened_for_missing": ["company_culture", "current_salary"],
                }
            )
        )
        executor = LLMExecutor(llm_client=llm, max_turns=3)
        # company_culture is required but not in the clarification node
        node, resolved = self._make_exec_node_with_required_input(
            "Determine negotiation strategy based on company culture",
            required_inputs=[
                {
                    "name": "company_culture",
                    "type": "text",
                    "description": "Company culture info",
                }
            ],
            unknown_keys=["current_salary"],  # unrelated field
        )
        result = executor._preflight_awaiting_input(node, resolved)
        # company_culture is not in the clarification — should appear in new_fields
        assert result is not None
        new_keys = [f["key"] for f in result.new_fields]
        assert "company_culture" in new_keys, (
            f"company_culture should be in new_fields; got {new_keys}"
        )

    def test_phase1_direct_block_when_no_unknown_fields_but_gap_exists(self):
        """When all clarification fields are filled but a required_input is not
        in the form at all, Phase 1 detects the gap and the LLM is called to
        generate the broadened description."""
        from cuddlytoddly.planning.llm_executor import AwaitingInputSignal, LLMExecutor

        llm = FakeLLM(
            json.dumps(
                {
                    "blocked": True,
                    "reason": "needs company culture",
                    "missing_fields": [],
                    "new_fields": [],
                    "broadened_description": "Determine a general negotiation strategy.",
                    "broadened_for_missing": ["company_culture"],
                }
            )
        )
        executor = LLMExecutor(llm_client=llm, max_turns=3)
        node, resolved = self._make_exec_node_with_required_input(
            "Determine negotiation strategy",
            required_inputs=[
                {
                    "name": "company_culture",
                    "type": "text",
                    "description": "Culture info",
                }
            ],
            unknown_keys=[],  # nothing unknown — all fields filled
            known_keys=["salary"],  # company_culture still not present
        )
        result = executor._preflight_awaiting_input(node, resolved)
        assert isinstance(result, AwaitingInputSignal)
        assert any(f["key"] == "company_culture" for f in result.new_fields)
        assert "company_culture" in result.missing_fields

    def test_phase1_no_gap_when_required_input_already_in_clar(self):
        """When required_input is already covered by an existing clarification
        field, Phase 1 detects no gap and proceeds to Phase 2 (LLM call)."""
        from cuddlytoddly.planning.llm_executor import LLMExecutor

        llm = FakeLLM(
            json.dumps(
                {
                    "blocked": False,
                    "reason": "can proceed",
                    "missing_fields": [],
                    "new_fields": [],
                }
            )
        )
        executor = LLMExecutor(llm_client=llm, max_turns=3)
        # performance_reviews is required AND is in the clarification node as unknown
        node, resolved = self._make_exec_node_with_required_input(
            "Identify key achievements",
            required_inputs=[
                {
                    "name": "performance_reviews",
                    "type": "document",
                    "description": "Past performance reviews",
                }
            ],
            unknown_keys=["performance_reviews"],
        )
        result = executor._preflight_awaiting_input(node, resolved)
        # LLM said not blocked, so result should be None
        assert result is None
        assert len(llm.calls) == 1  # Phase 2 ran

    def test_phase1_merges_auto_new_fields_with_llm_new_fields(self):
        """When Phase 1 detects a gap AND Phase 2 also adds new_fields,
        the final signal contains both — deduplicated."""
        from cuddlytoddly.planning.llm_executor import AwaitingInputSignal, LLMExecutor

        llm_new_field = {
            "key": "job_level",
            "label": "Job level",
            "value": "unknown",
            "rationale": "needed",
        }
        llm = FakeLLM(
            json.dumps(
                {
                    "blocked": True,
                    "reason": "needs more context",
                    "missing_fields": ["current_salary"],
                    "new_fields": [llm_new_field],
                }
            )
        )
        executor = LLMExecutor(llm_client=llm, max_turns=3)
        node, resolved = self._make_exec_node_with_required_input(
            "Research strategy",
            required_inputs=[
                {
                    "name": "company_culture",
                    "type": "text",
                    "description": "Culture info",
                }
            ],
            unknown_keys=["current_salary"],
        )
        result = executor._preflight_awaiting_input(node, resolved)
        assert isinstance(result, AwaitingInputSignal)
        new_keys = [f["key"] for f in result.new_fields]
        assert "job_level" in new_keys  # from LLM
        assert "company_culture" in new_keys  # from Phase 1 auto-detection
        assert "current_salary" in result.missing_fields
        assert "company_culture" in result.missing_fields

    def test_execute_uses_broadened_description_when_inputs_missing(self):
        """execute() must never return AwaitingInputSignal — when inputs are
        missing it runs with the broadened description and returns a string."""
        import threading

        from cuddlytoddly.planning.llm_executor import AwaitingInputSignal, LLMExecutor
        from toddly.engine.execution_step_reporter import ExecutionStepReporter

        g = TaskGraph()
        add_node(
            g,
            "personal_task",
            metadata={
                "description": "List personal achievements relevant to the job",
                "output": [{"name": "out", "type": "list", "description": "achievements"}],
            },
        )
        add_node(g, "clar", node_type="clarification", metadata={"description": "ctx"})
        g.nodes["personal_task"].dependencies.add("clar")
        g.nodes["clar"].result = json.dumps(
            [
                {
                    "key": "current_salary",
                    "value": "unknown",
                    "label": "salary",
                    "rationale": "r",
                },
            ]
        )
        g.nodes["clar"].status = "done"

        # First call: preflight response (blocked, with broadened description)
        # Second call: executor turn response (done=true with result)
        call_count = [0]

        def multi_response(prompt, schema=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return json.dumps(
                    {
                        "blocked": True,
                        "reason": "needs personal history",
                        "missing_fields": ["current_salary"],
                        "new_fields": [],
                        "broadened_description": "Produce a template for articulating professional achievements.",
                        "broadened_for_missing": ["current_salary"],
                    }
                )
            return json.dumps({"done": True, "result": "Here is a template for achievements."})

        llm = FakeLLM(multi_response)
        reporter = ExecutionStepReporter(
            parent_node_id="personal_task",
            apply_fn=lambda e: apply_event(g, e),
            graph_lock=threading.RLock(),
            graph=g,
        )
        executor = LLMExecutor(llm_client=llm, max_turns=3)
        result = executor.execute(g.nodes["personal_task"], g.get_snapshot(), reporter=reporter)

        # Result must be a string, never a signal
        assert isinstance(result, str), f"expected str, got {type(result)}"
        assert result  # non-empty
        # Reporter must carry the broadening signal
        assert reporter.pending_broadening is not None
        assert isinstance(reporter.pending_broadening, AwaitingInputSignal)
        assert reporter.pending_broadening.broadened_description == (
            "Produce a template for articulating professional achievements."
        )
        assert reporter.pending_broadening.broadened_for_missing == ["current_salary"]

    # ── orchestrator broadening metadata handling ─────────────────────────────

    def test_orchestrator_writes_broadening_metadata_after_execution(self):
        """After a node executes with a broadened description, the orchestrator
        writes broadened_description and broadened_for_missing into node metadata."""
        import threading

        from cuddlytoddly.planning.llm_executor import AwaitingInputSignal
        from toddly.engine.execution_step_reporter import ExecutionStepReporter

        orch, g, _, mock_gate = make_orchestrator(gate_satisfied=True)
        add_node(
            g,
            "t",
            node_type="task",
            metadata={
                "description": "List personal achievements",
                "output": [{"name": "o", "type": "list", "description": "d"}],
            },
        )
        g.nodes["t"].status = "running"

        # Simulate: executor ran with broadened description, stored signal on reporter
        reporter = ExecutionStepReporter(
            parent_node_id="t",
            apply_fn=lambda e: apply_event(g, e),
            graph_lock=threading.RLock(),
            graph=g,
        )
        signal = AwaitingInputSignal(
            reason="needs personal history",
            missing_fields=["key_achievements"],
            new_fields=[],
            broadened_description="Produce a template for articulating achievements.",
            broadened_for_missing=["key_achievements"],
        )
        reporter.on_broadened_execution(signal)
        orch._reporters["t"] = reporter

        orch._on_node_done("t", self._make_future("Here is the template."))

        # Node should be done (not awaiting_input)
        assert g.nodes["t"].status == "done"
        # Broadening metadata must be written
        assert g.nodes["t"].metadata.get("broadened_description") == (
            "Produce a template for articulating achievements."
        )
        assert g.nodes["t"].metadata.get("broadened_for_missing") == ["key_achievements"]
        # Quality gate was still called
        mock_gate.verify_result.assert_called_once()

    def test_orchestrator_patches_clarification_node_from_reporter_signal(self):
        """new_fields in the reporter signal are added to the clarification node."""
        import threading

        from cuddlytoddly.planning.llm_executor import AwaitingInputSignal
        from toddly.engine.execution_step_reporter import ExecutionStepReporter

        g = TaskGraph()
        add_node(
            g,
            "clar",
            node_type="clarification",
            metadata={
                "description": "ctx",
                "fields": [
                    {
                        "key": "job_title",
                        "label": "Job title",
                        "value": "unknown",
                        "rationale": "r",
                    }
                ],
            },
        )
        g.nodes["clar"].result = json.dumps(
            [
                {
                    "key": "job_title",
                    "label": "Job title",
                    "value": "unknown",
                    "rationale": "r",
                },
            ]
        )
        g.nodes["clar"].status = "done"
        add_node(
            g,
            "t",
            node_type="task",
            metadata={
                "description": "Gather company financial info",
                "output": [{"name": "o", "type": "document", "description": "d"}],
            },
        )
        g.nodes["t"].dependencies.add("clar")
        g.nodes["clar"].children.add("t")
        g.nodes["t"].status = "running"

        orch, _, _, _ = make_orchestrator(graph=g)

        reporter = ExecutionStepReporter(
            parent_node_id="t",
            apply_fn=lambda e: apply_event(g, e),
            graph_lock=threading.RLock(),
            graph=g,
        )
        signal = AwaitingInputSignal(
            reason="needs company name",
            missing_fields=[],
            new_fields=[
                {
                    "key": "company_name",
                    "label": "Company name",
                    "value": "unknown",
                    "rationale": "needed for research",
                }
            ],
            clarification_node_id="clar",
            broadened_description="Research general company budget factors.",
            broadened_for_missing=["company_name"],
        )
        reporter.on_broadened_execution(signal)
        orch._reporters["t"] = reporter

        orch._on_node_done("t", self._make_future("General budget research result."))

        # clarification node should now have the new field
        clar_fields = g.nodes["clar"].metadata.get("fields", [])
        assert any(f["key"] == "company_name" for f in clar_fields)

    # ── _resume_unblocked_pass ────────────────────────────────────────────────

    def test_resume_pass_resumes_when_field_filled(self):
        """_resume_unblocked_pass resumes a node whose missing_field is now set."""
        g = TaskGraph()
        add_node(g, "clar", node_type="clarification", metadata={"description": "ctx"})
        g.nodes["clar"].result = json.dumps(
            [
                {
                    "key": "key_achievements",
                    "label": "Achievements",
                    "value": "Led team of 5, shipped product X",
                    "rationale": "r",
                },
            ]
        )
        g.nodes["clar"].status = "done"

        add_node(
            g,
            "t",
            node_type="task",
            metadata={
                "description": "list achievements",
                "output": [],
                "missing_fields": ["key_achievements"],
            },
        )
        g.nodes["t"].dependencies.add("clar")
        g.nodes["clar"].children.add("t")
        g.nodes["t"].status = "awaiting_input"

        orch, _, _, _ = make_orchestrator(graph=g)
        resumed = orch._resume_unblocked_pass()

        assert resumed == 1
        assert g.nodes["t"].status in ("pending", "ready")

    def test_resume_pass_does_not_resume_when_field_still_unknown(self):
        """_resume_unblocked_pass must not resume if the field is still unknown."""
        g = TaskGraph()
        add_node(g, "clar", node_type="clarification", metadata={"description": "ctx"})
        g.nodes["clar"].result = json.dumps(
            [
                {
                    "key": "key_achievements",
                    "label": "Achievements",
                    "value": "unknown",
                    "rationale": "r",
                },
            ]
        )
        g.nodes["clar"].status = "done"

        add_node(
            g,
            "t",
            node_type="task",
            metadata={
                "description": "list achievements",
                "output": [],
                "missing_fields": ["key_achievements"],
            },
        )
        g.nodes["t"].dependencies.add("clar")
        g.nodes["clar"].children.add("t")
        g.nodes["t"].status = "awaiting_input"

        orch, _, _, _ = make_orchestrator(graph=g)
        resumed = orch._resume_unblocked_pass()

        assert resumed == 0
        assert g.nodes["t"].status == "awaiting_input"

    def test_resume_node_public_api(self):
        """orchestrator.resume_node() transitions awaiting_input → pending."""
        g = TaskGraph()
        add_node(
            g,
            "t",
            node_type="task",
            metadata={
                "description": "t",
                "output": [],
                "missing_fields": ["key_achievements"],
            },
        )
        g.nodes["t"].status = "awaiting_input"

        orch, _, _, _ = make_orchestrator(graph=g)
        result = orch.resume_node("t")

        assert result is True
        assert g.nodes["t"].status in ("pending", "ready")

    def test_resume_node_returns_false_for_non_awaiting(self):
        """resume_node returns False when the node is not awaiting_input."""
        orch, g, _, _ = make_orchestrator()
        add_node(g, "t", node_type="task", metadata={"description": "t", "output": []})
        g.nodes["t"].status = "ready"

        result = orch.resume_node("t")
        assert result is False

    # ── Path B: empty missing_fields fallback ─────────────────────────────────

    def test_resume_pass_path_b_resumes_when_any_field_filled(self):
        """Path B: a node with missing_fields=[] resumes when any clar field is filled."""
        g = TaskGraph()
        add_node(g, "clar", node_type="clarification", metadata={"description": "ctx"})
        g.nodes["clar"].result = json.dumps(
            [
                {
                    "key": "job_title",
                    "label": "Job title",
                    "value": "Software Engineer",
                    "rationale": "r",
                },
                {
                    "key": "current_salary",
                    "label": "Current salary",
                    "value": "unknown",
                    "rationale": "r",
                },
            ]
        )
        g.nodes["clar"].status = "done"

        add_node(
            g,
            "t",
            node_type="task",
            metadata={
                "description": "Research salary ranges",
                "output": [],
                "missing_fields": [],  # empty — Path B
            },
        )
        g.nodes["t"].dependencies.add("clar")
        g.nodes["clar"].children.add("t")
        g.nodes["t"].status = "awaiting_input"

        orch, _, _, _ = make_orchestrator(graph=g)
        resumed = orch._resume_unblocked_pass()

        assert resumed == 1
        assert g.nodes["t"].status in ("pending", "ready")

    def test_resume_pass_path_b_does_not_resume_when_all_still_unknown(self):
        """Path B: a node with missing_fields=[] stays blocked when all fields unknown."""
        g = TaskGraph()
        add_node(g, "clar", node_type="clarification", metadata={"description": "ctx"})
        g.nodes["clar"].result = json.dumps(
            [
                {
                    "key": "job_title",
                    "label": "Job title",
                    "value": "unknown",
                    "rationale": "r",
                },
                {
                    "key": "current_salary",
                    "label": "Current salary",
                    "value": "unknown",
                    "rationale": "r",
                },
            ]
        )
        g.nodes["clar"].status = "done"

        add_node(
            g,
            "t",
            node_type="task",
            metadata={
                "description": "Research salary ranges",
                "output": [],
                "missing_fields": [],  # empty — Path B
            },
        )
        g.nodes["t"].dependencies.add("clar")
        g.nodes["clar"].children.add("t")
        g.nodes["t"].status = "awaiting_input"

        orch, _, _, _ = make_orchestrator(graph=g)
        resumed = orch._resume_unblocked_pass()

        assert resumed == 0
        assert g.nodes["t"].status == "awaiting_input"

    # ── regular tasks still verified ─────────────────────────────────────────

    def test_regular_task_still_calls_verify_result(self):
        """Sanity: ordinary task nodes still go through verification."""
        orch, g, _, mock_gate = make_orchestrator(gate_satisfied=True)
        add_node(
            g,
            "regular_task",
            node_type="task",
            metadata={
                "description": "Do research",
                "output": [{"name": "report", "type": "document", "description": "the report"}],
            },
        )
        g.nodes["regular_task"].status = "running"

        f = Future()
        f.set_result("Substantive research report content here.")
        orch._on_node_done("regular_task", f)

        mock_gate.verify_result.assert_called_once()


# ── Issue 4 (revised): tool results context folded into verifier prompt ─────


class TestToolResultsContext:
    """
    _all_tool_calls_failed is removed.  Tool execution outcomes are now passed
    as context into the existing LLM verifier call via _build_tool_results_context.
    These tests verify that context is computed correctly and reaches the verifier.
    """

    def _make_gate(self):
        llm = FakeLLM(json.dumps({"satisfied": True, "reason": "looks fine"}))
        return QualityGate(llm_client=llm)

    def _make_node_with_steps(self, node_id, attempts_per_step):
        """
        Build a TaskGraph with a parent node and execution step children.
        attempts_per_step: list of lists of (result_str, status_str) tuples.
        status_str is "ok" or "error".
        """
        g = TaskGraph()
        add_node(
            g,
            node_id,
            metadata={
                "description": "research task",
                "output": [{"name": "report", "type": "document", "description": "the report"}],
            },
        )
        for i, attempt_list in enumerate(attempts_per_step):
            step_id = f"{node_id}__step_web_search_{i}"
            attempts = []
            for j, item in enumerate(attempt_list):
                if isinstance(item, tuple):
                    result_str, status = item
                else:
                    result_str, status = item, "ok"
                attempts.append({"turn": j, "args": {}, "result": result_str, "status": status})
            add_node(
                g,
                step_id,
                node_type="execution_step",
                metadata={
                    "step_type": "tool_call",
                    "tool_name": "web_search",
                    "attempts": attempts,
                },
            )
        return g.nodes[node_id], g.get_snapshot()

    # ── _build_tool_results_context unit tests ────────────────────────────────

    def test_all_errors_produces_warning_context(self):
        gate = self._make_gate()
        node, snap = self._make_node_with_steps(
            "t",
            [
                [("ERROR: decode error", "error")],
                [("ERROR: no results", "error")],
            ],
        )
        summary, content = gate._build_tool_results_context(node, snap)
        assert "All" in summary and "error" in summary.lower()

    def test_mixed_results_produces_partial_context(self):
        gate = self._make_gate()
        node, snap = self._make_node_with_steps(
            "t",
            [
                [("ERROR: timeout", "error")],
                [("Title: Salary\nURL: glassdoor.com", "ok")],
            ],
        )
        summary, content = gate._build_tool_results_context(node, snap)
        assert "1 of 2" in summary
        assert "glassdoor.com" in content

    def test_no_tool_calls_returns_empty_string(self):
        gate = self._make_gate()
        g = TaskGraph()
        add_node(
            g,
            "t",
            metadata={
                "description": "t",
                "output": [{"name": "report", "type": "document", "description": "r"}],
            },
        )
        summary, content = gate._build_tool_results_context(g.nodes["t"], g.get_snapshot())
        assert summary == "" and content == ""

    def test_all_success_produces_success_context(self):
        gate = self._make_gate()
        node, snap = self._make_node_with_steps(
            "t",
            [
                [("Title: Salary Survey\nURL: glassdoor.com", "ok")],
            ],
        )
        summary, content = gate._build_tool_results_context(node, snap)
        assert "successfully" in summary.lower()
        assert "glassdoor.com" in content

    # ── Integration: verify_result passes tool context to LLM verifier ────────

    def test_verify_result_passes_tool_context_in_prompt(self):
        """Tool execution summary is included in the verifier prompt."""
        prompts_seen = []

        def capture_ask(prompt, schema=None):
            prompts_seen.append(prompt)
            return json.dumps({"satisfied": False, "reason": "fabricated"})

        llm = MagicMock()
        llm.is_stopped = False
        llm.ask = capture_ask
        gate = QualityGate(llm_client=llm)
        node, snap = self._make_node_with_steps(
            "t",
            [
                [("ERROR: decode error", "error")],
            ],
        )
        gate.verify_result(node, "Salary is $60k-$90k.", snap)
        assert prompts_seen, "LLM verifier must be called"
        assert "TOOL EXECUTION SUMMARY" in prompts_seen[0]
        assert "error" in prompts_seen[0].lower()

    def test_verify_result_llm_always_called(self):
        """The LLM verifier always runs — tool context informs its judgment."""
        llm = FakeLLM(json.dumps({"satisfied": True, "reason": "good data"}))
        gate = QualityGate(llm_client=llm)
        node, snap = self._make_node_with_steps(
            "t",
            [
                [("Title: Salary Survey\nURL: levels.fyi", "ok")],
            ],
        )
        ok, _ = gate.verify_result(node, "Based on levels.fyi the median is $130k.", snap)
        assert ok is True
        assert len(llm.calls) == 1

    def test_verify_result_no_tool_context_when_no_steps(self):
        """When a node made no tool calls, the prompt has no tool context section."""
        prompts_seen = []

        def capture_ask(prompt, schema=None):
            prompts_seen.append(prompt)
            return json.dumps({"satisfied": True, "reason": "fine"})

        llm = MagicMock()
        llm.is_stopped = False
        llm.ask = capture_ask
        gate = QualityGate(llm_client=llm)
        g = TaskGraph()
        add_node(
            g,
            "t",
            metadata={
                "description": "write a summary",
                "output": [{"name": "summary", "type": "document", "description": "s"}],
            },
        )
        gate.verify_result(g.nodes["t"], "Here is the full summary content.", g.get_snapshot())
        assert prompts_seen
        # The section header "TOOL EXECUTION SUMMARY:" only appears when tool
        # context data is present. The instruction text refers to it without the
        # colon, so checking for "TOOL EXECUTION SUMMARY:" is specific to the section.
        assert "TOOL EXECUTION SUMMARY:" not in prompts_seen[0]


# ── Fix: schema enforcement + fallback LLM call + broadening context ──────────


class TestBroadenedExecutionFixes:
    """
    Tests for all three issues from the hallucination run:

    Fix 1 — broadened_description is now required in the schema.
    Fix 2 — when broadened_description is empty, a focused fallback LLM call
             is made; execution is skipped entirely if that also fails.
    Fix 3 — the quality gate receives a broadening_context section in the
             verifier prompt when the node ran with a broadened description.
    """

    def _make_exec_node(self, description, unknown_keys=()):
        g = TaskGraph()
        add_node(
            g,
            "t",
            metadata={
                "description": description,
                "output": [{"name": "out", "type": "list", "description": "output"}],
            },
        )
        add_node(g, "clar", node_type="clarification", metadata={"description": "ctx"})
        g.nodes["t"].dependencies.add("clar")
        unknown = [{"key": k, "label": k} for k in unknown_keys]
        resolved = [
            {
                "node_id": "clar",
                "description": "ctx",
                "declared_output": [],
                "result": "ctx",
                "_unknown_fields": unknown,
                "_known_fields": [],
            }
        ]
        return g.nodes["t"], resolved, g

    # ── Fix 1: schema requires broadened_description ──────────────────────────

    def test_preflight_signal_carries_broadened_description(self):
        """When the LLM returns a broadened_description it is propagated."""
        from cuddlytoddly.planning.llm_executor import AwaitingInputSignal, LLMExecutor

        llm = FakeLLM(
            json.dumps(
                {
                    "blocked": True,
                    "reason": "needs personal history",
                    "missing_fields": ["current_salary"],
                    "new_fields": [],
                    "broadened_description": "Produce a template to help articulate achievements.",
                    "broadened_for_missing": ["current_salary"],
                }
            )
        )
        executor = LLMExecutor(llm_client=llm, max_turns=3)
        node, resolved, _ = self._make_exec_node(
            "List personal achievements", unknown_keys=["current_salary"]
        )
        signal = executor._preflight_awaiting_input(node, resolved)
        assert isinstance(signal, AwaitingInputSignal)
        assert signal.broadened_description == (
            "Produce a template to help articulate achievements."
        )
        assert signal.broadened_for_missing == ["current_salary"]

    # ── Fix 2: fallback LLM call when broadened_description is empty ──────────

    def test_execute_calls_fallback_when_broadened_description_empty(self):
        """When the preflight returns blocked=true with empty broadened_description,
        a second focused LLM call is made to generate it."""
        import threading

        from cuddlytoddly.planning.llm_executor import LLMExecutor
        from toddly.engine.execution_step_reporter import ExecutionStepReporter

        call_count = [0]

        def multi_response(prompt, schema=None):
            call_count[0] += 1
            if call_count[0] == 1:
                # Preflight: blocked but no broadened_description
                return json.dumps(
                    {
                        "blocked": True,
                        "reason": "needs personal history",
                        "missing_fields": ["current_salary"],
                        "new_fields": [],
                        "broadened_description": "",  # empty — triggers fallback
                        "broadened_for_missing": ["current_salary"],
                    }
                )
            if call_count[0] == 2:
                # Fallback broadening call
                return json.dumps(
                    {
                        "broadened_description": "Produce a guided achievement template.",
                    }
                )
            # Execution turn
            return json.dumps({"done": True, "result": "Here is the template."})

        llm = FakeLLM(multi_response)
        node, _, g = self._make_exec_node(
            "List personal achievements", unknown_keys=["current_salary"]
        )
        # Set up the clar node result so _resolve_inputs finds the unknown fields
        g.nodes["clar"].result = json.dumps(
            [
                {
                    "key": "current_salary",
                    "label": "salary",
                    "value": "unknown",
                    "rationale": "r",
                },
            ]
        )
        g.nodes["clar"].status = "done"
        reporter = ExecutionStepReporter(
            parent_node_id="t",
            apply_fn=lambda e: apply_event(g, e),
            graph_lock=threading.RLock(),
            graph=g,
        )
        executor = LLMExecutor(llm_client=llm, max_turns=3)
        result = executor.execute(node, g.get_snapshot(), reporter=reporter)

        assert isinstance(result, str), "execute() must return a string"
        assert call_count[0] >= 3, "should have made preflight + fallback + execution calls"
        assert reporter.pending_broadening is not None
        assert reporter.pending_broadening.broadened_description == (
            "Produce a guided achievement template."
        )

    def test_execute_returns_none_when_both_broadening_calls_fail(self):
        """When both the preflight and the fallback return empty descriptions,
        execution is skipped entirely (returns None) — not the original description."""
        from cuddlytoddly.planning.llm_executor import LLMExecutor

        call_count = [0]

        def multi_response(prompt, schema=None):
            call_count[0] += 1
            if call_count[0] == 1:
                return json.dumps(
                    {
                        "blocked": True,
                        "reason": "needs personal history",
                        "missing_fields": ["salary"],
                        "new_fields": [],
                        "broadened_description": "",
                        "broadened_for_missing": ["salary"],
                    }
                )
            # Fallback call also returns empty
            return json.dumps({"broadened_description": ""})

        llm = FakeLLM(multi_response)
        node, _, g = self._make_exec_node("List personal achievements", unknown_keys=["salary"])
        executor = LLMExecutor(llm_client=llm, max_turns=3)
        result = executor.execute(node, g.get_snapshot())
        assert result is None, (
            "When both broadening calls fail, execute() must return None "
            f"not the original description — got: {result!r}"
        )

    # ── Fix 3: broadening context reaches the verifier prompt ─────────────────

    def test_verifier_receives_broadening_context_when_node_ran_broadened(self):
        """When a node has broadened_description in metadata, the verifier prompt
        includes a BROADENED EXECUTION NOTICE section."""
        prompts_seen = []

        def capture_ask(prompt, schema=None):
            prompts_seen.append(prompt)
            return json.dumps({"satisfied": False, "reason": "invented specifics"})

        llm = MagicMock()
        llm.is_stopped = False
        llm.ask = capture_ask
        gate = QualityGate(llm_client=llm)

        g = TaskGraph()
        add_node(
            g,
            "t",
            metadata={
                "description": "Identify personal achievements",
                "output": [{"name": "achievements", "type": "list", "description": "a"}],
                "broadened_description": "Produce a template for articulating achievements.",
                "broadened_for_missing": ["current_salary", "performance_reviews"],
                "broadened_reason": "needs personal history",
            },
        )

        ok, reason = gate.verify_result(
            g.nodes["t"],
            '[{"personal_achievements_list": ["Increased sales by 25%"]}]',
            g.get_snapshot(),
        )
        assert ok is False
        assert prompts_seen, "verifier must be called"
        assert "BROADENED EXECUTION NOTICE" in prompts_seen[0]
        assert "current_salary" in prompts_seen[0]
        assert "Produce a template" in prompts_seen[0]

    def test_verifier_no_broadening_section_for_normal_nodes(self):
        """Nodes that ran with their original description get no broadening section."""
        prompts_seen = []

        def capture_ask(prompt, schema=None):
            prompts_seen.append(prompt)
            return json.dumps({"satisfied": True, "reason": "fine"})

        llm = MagicMock()
        llm.is_stopped = False
        llm.ask = capture_ask
        gate = QualityGate(llm_client=llm)

        g = TaskGraph()
        add_node(
            g,
            "t",
            metadata={
                "description": "Research salary ranges",
                "output": [{"name": "report", "type": "document", "description": "r"}],
                # no broadened_description in metadata
            },
        )

        gate.verify_result(g.nodes["t"], "Salaries range from $80k-$120k.", g.get_snapshot())
        assert prompts_seen
        # The section header "BROADENED EXECUTION NOTICE:" followed by a newline
        # only appears when broadening context data is present.
        assert "BROADENED EXECUTION NOTICE:\n" not in prompts_seen[0]


# ── Issue 5: max_retries cap + exponential backoff ────────────────────────────


class TestMaxRetriesAndBackoff:
    def _make_future(self, result):
        f = Future()
        f.set_result(result)
        return f

    def _run_verification_failures(self, orch, g, node_id, count):
        """Drive _on_node_done `count` times with a failing verification."""
        for _ in range(count):
            if g.nodes[node_id].status in ("failed",):
                # Simulate the orchestrator's RESET_NODE putting it back to ready
                g.nodes[node_id].status = "running"
            orch._on_node_done(node_id, self._make_future("template output"))

    def test_node_permanently_failed_after_max_retries(self):
        orch, g, _, mock_gate = make_orchestrator(gate_satisfied=False, max_retries=3)
        add_node(
            g,
            "t",
            node_type="task",
            metadata={
                "description": "task",
                "output": [{"name": "report", "type": "document", "description": "r"}],
            },
        )
        g.nodes["t"].status = "running"

        # Drive 3 failures (0-indexed: attempts 0, 1, 2 all fail)
        for attempt in range(3):
            g.nodes["t"].status = "running"
            orch._on_node_done("t", self._make_future("bad output"))

        # After max_retries failures the node must stay failed — not reset.
        # retry_count holds the number of resets-for-retry, which is
        # max_retries - 1: the permanent-fail branch fires before incrementing.
        assert g.nodes["t"].status == "failed"
        assert g.nodes["t"].metadata.get("retry_count", 0) == 2  # (max_retries - 1)

    def test_node_resets_below_max_retries(self):
        """Before the cap is hit the node should be reset to pending/ready."""
        orch, g, _, _ = make_orchestrator(gate_satisfied=False, max_retries=5)
        add_node(
            g,
            "t",
            node_type="task",
            metadata={
                "description": "task",
                "output": [{"name": "r", "type": "document", "description": "r"}],
            },
        )
        g.nodes["t"].status = "running"
        orch._on_node_done("t", self._make_future("bad output"))

        # After one failure (retry_count=1, max_retries=5) the node is reset
        assert g.nodes["t"].status in ("pending", "ready")

    def test_retry_after_is_set_on_failure(self):
        """Each verification failure must stamp retry_after in metadata."""
        orch, g, _, _ = make_orchestrator(gate_satisfied=False, max_retries=5)
        add_node(
            g,
            "t",
            node_type="task",
            metadata={
                "description": "task",
                "output": [{"name": "r", "type": "document", "description": "r"}],
            },
        )
        g.nodes["t"].status = "running"
        before = time.time()
        orch._on_node_done("t", self._make_future("bad output"))
        after = time.time()

        retry_after = g.nodes["t"].metadata.get("retry_after", 0)
        assert retry_after > before, "retry_after should be in the future"
        # First retry: backoff = 2**0 = 1s (retry_count was 0 before this call)
        assert retry_after <= after + 2.0, "backoff should not exceed ~1s for first retry"

    def test_backoff_increases_with_each_failure(self):
        """retry_after should grow across successive failures."""
        orch, g, _, _ = make_orchestrator(gate_satisfied=False, max_retries=10)
        add_node(
            g,
            "t",
            node_type="task",
            metadata={
                "description": "task",
                "output": [{"name": "r", "type": "document", "description": "r"}],
            },
        )

        backoffs = []
        for _ in range(4):
            g.nodes["t"].status = "running"
            before = time.time()
            orch._on_node_done("t", self._make_future("bad output"))
            ra = g.nodes["t"].metadata.get("retry_after", 0)
            backoffs.append(ra - before)

        # Each successive backoff should be larger than the previous
        for i in range(1, len(backoffs)):
            assert backoffs[i] > backoffs[i - 1], (
                f"backoff[{i}]={backoffs[i]:.2f} not > backoff[{i - 1}]={backoffs[i - 1]:.2f}"
            )

    def test_execution_pass_skips_node_in_backoff_window(self):
        """A ready node whose retry_after hasn't elapsed must not be launched."""
        orch, g, mock_executor, _ = make_orchestrator()
        add_node(g, "t", node_type="task", metadata={"description": "t", "output": []})
        g.nodes["t"].metadata["retry_after"] = time.time() + 60  # 60s from now
        g.nodes["t"].status = "ready"

        launched = orch._execution_pass()
        assert launched == 0
        mock_executor.execute.assert_not_called()

    def test_execution_pass_launches_node_after_backoff_expires(self):
        """Once retry_after is in the past the node is launched normally."""
        orch, g, mock_executor, _ = make_orchestrator()
        add_node(g, "t", node_type="task", metadata={"description": "t", "output": []})
        g.nodes["t"].metadata["retry_after"] = time.time() - 1  # already elapsed
        g.nodes["t"].status = "ready"

        launched = orch._execution_pass()
        assert launched == 1

    def test_max_retries_zero_fails_immediately(self):
        """max_retries=0 means the first failure is permanent."""
        orch, g, _, _ = make_orchestrator(gate_satisfied=False, max_retries=0)
        add_node(
            g,
            "t",
            node_type="task",
            metadata={
                "description": "task",
                "output": [{"name": "r", "type": "document", "description": "r"}],
            },
        )
        g.nodes["t"].status = "running"
        orch._on_node_done("t", self._make_future("output"))

        assert g.nodes["t"].status == "failed"
        assert g.nodes["t"].metadata.get("retry_count", 0) == 0


# ── Issue 6: tool error strings flagged correctly ─────────────────────────────


class TestToolErrorStringDetection:
    def test_error_string_sets_error_flag_on_reporter(self):
        """When a tool returns 'ERROR: ...' without raising, the step node
        in the graph must receive status='error' (i.e. MARK_FAILED)."""
        g = TaskGraph()
        add_node(
            g,
            "parent",
            node_type="task",
            metadata={"description": "research", "output": []},
        )
        g.nodes["parent"].status = "running"
        reporter = make_reporter("parent", g)

        def error_tool(args):
            return "ERROR: web search failed — Body collection error: error decoding response body"

        registry = make_tool_registry({"web_search": error_tool})

        turns = [
            tool_response("web_search", {"query": "salary data"}),
            done_response("done"),
        ]
        idx = [0]

        def responses(prompt, schema=None):
            r = turns[idx[0]]
            idx[0] = min(idx[0] + 1, len(turns) - 1)
            return r

        llm = FakeLLM(responses)
        executor = LLMExecutor(llm_client=llm, tool_registry=registry, max_turns=5)
        node, snapshot = g.nodes["parent"], g.get_snapshot()
        executor.execute(node, snapshot, reporter=reporter)

        step_id = "parent__step_web_search"
        assert step_id in g.nodes, "step node should be in graph"
        step_node = g.nodes[step_id]
        attempts = step_node.metadata.get("attempts", [])
        assert attempts, "step node should have recorded attempts"
        assert attempts[0]["status"] == "error", (
            f"Expected status='error', got {attempts[0]['status']!r}"
        )

    def test_non_error_string_keeps_ok_status(self):
        """A tool that returns a normal result must still get status='ok'."""
        g = TaskGraph()
        add_node(
            g,
            "parent",
            node_type="task",
            metadata={"description": "research", "output": []},
        )
        g.nodes["parent"].status = "running"
        reporter = make_reporter("parent", g)

        def good_tool(args):
            return "Title: Glassdoor salary survey\nURL: glassdoor.com\nSnippet: $120k avg"

        registry = make_tool_registry({"web_search": good_tool})

        turns = [
            tool_response("web_search", {"query": "salary data"}),
            done_response("The median is $120k."),
        ]
        idx = [0]

        def responses(prompt, schema=None):
            r = turns[idx[0]]
            idx[0] = min(idx[0] + 1, len(turns) - 1)
            return r

        llm = FakeLLM(responses)
        executor = LLMExecutor(llm_client=llm, tool_registry=registry, max_turns=5)
        node, snapshot = g.nodes["parent"], g.get_snapshot()
        executor.execute(node, snapshot, reporter=reporter)

        step_id = "parent__step_web_search"
        assert step_id in g.nodes
        attempts = g.nodes[step_id].metadata.get("attempts", [])
        assert attempts[0]["status"] == "ok"

    def test_exception_still_sets_error_flag(self):
        """The original path (tool raises) must still mark status='error'."""
        g = TaskGraph()
        add_node(
            g,
            "parent",
            node_type="task",
            metadata={"description": "research", "output": []},
        )
        g.nodes["parent"].status = "running"
        reporter = make_reporter("parent", g)

        def exploding_tool(args):
            raise RuntimeError("connection refused")

        registry = make_tool_registry({"web_search": exploding_tool})

        turns = [
            tool_response("web_search", {"query": "salary"}),
            done_response("fallback"),
        ]
        idx = [0]

        def responses(prompt, schema=None):
            r = turns[idx[0]]
            idx[0] = min(idx[0] + 1, len(turns) - 1)
            return r

        llm = FakeLLM(responses)
        executor = LLMExecutor(llm_client=llm, tool_registry=registry, max_turns=5)
        executor.execute(g.nodes["parent"], g.get_snapshot(), reporter=reporter)

        attempts = g.nodes["parent__step_web_search"].metadata.get("attempts", [])
        assert attempts[0]["status"] == "error"

    def test_issue6_and_issue4_integration(self):
        """End-to-end: tool returns ERROR string → status=error on step node
        → _build_tool_results_context sees only errors → verifier prompt includes
        the failure context → LLM verifier rejects fabricated output."""
        g = TaskGraph()
        add_node(
            g,
            "salary_node",
            node_type="task",
            metadata={
                "description": "Research salary",
                "output": [
                    {
                        "name": "report",
                        "type": "document",
                        "description": "salary report",
                    }
                ],
            },
        )
        step_id = "salary_node__step_web_search"
        add_node(
            g,
            step_id,
            node_type="execution_step",
            metadata={
                "step_type": "tool_call",
                "tool_name": "web_search",
                "attempts": [
                    {
                        "turn": 0,
                        "args": {"query": "salary"},
                        "result": "ERROR: web search failed — Body collection error",
                        "status": "error",  # set correctly by fix 6
                    }
                ],
            },
        )

        # The LLM verifier is called with tool context — it decides not satisfied
        llm = FakeLLM(json.dumps({"satisfied": False, "reason": "all searches failed"}))
        gate = QualityGate(llm_client=llm)

        ok, reason = gate.verify_result(
            g.nodes["salary_node"],
            "Salaries range from $60k to $90k based on national averages.",
            g.get_snapshot(),
        )

        # Verifier is always called now (no early-exit heuristic)
        assert ok is False
        assert len(llm.calls) == 1, "LLM verifier must be called"
        # Tool context appears in the prompt
        assert "TOOL EXECUTION SUMMARY" in llm.calls[0]
