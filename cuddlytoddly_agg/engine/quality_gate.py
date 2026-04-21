# engine/quality_gate.py

import json
import os
from pathlib import Path
from typing import Callable

from cuddlytoddly.infra.logging import get_logger
from cuddlytoddly.planning.llm_interface import LLMStoppedError

# Default prompt builders — imported here so callers that construct QualityGate
# without explicit prompt functions get the cuddlytoddly-specific behaviour
# unchanged.  When moving QualityGate to a shared agent-core library, remove
# these imports and require callers to supply both functions explicitly.
from cuddlytoddly.planning.prompts import (
    build_check_dependencies_prompt as _default_check_deps_prompt,
)
from cuddlytoddly.planning.prompts import (
    build_verify_result_prompt as _default_verify_prompt,
)
from cuddlytoddly.planning.schemas import (
    DEPENDENCY_CHECK_SCHEMA,
    RESULT_VERIFICATION_SCHEMA,
)

logger = get_logger(__name__)


class QualityGate:
    """
    LLM-based quality checks for the orchestrator.

    Kept separate from Orchestrator so the orchestration logic
    (graph mutation, scheduling) stays decoupled from the verification logic
    (prompt building, LLM calls, schema parsing).

    Mirrors the pattern of LLMPlanner / LLMExecutor — the orchestrator
    receives it as a dependency and calls its methods; it never touches
    the graph directly.

    Prompt functions
    ----------------
    verify_prompt_fn and check_deps_prompt_fn are the two domain-specific
    seams.  Pass custom callables to use different prompt text for a different
    project (e.g. code review) while reusing all the surrounding logic.

    Both functions must accept only keyword arguments and return a str.
    Their signatures must match the keyword arguments passed by verify_result()
    and check_dependencies() respectively — see the default implementations in
    planning/prompts.py for the exact parameter lists.
    """

    FILE_EXTENSIONS = frozenset(
        {
            ".md",
            ".txt",
            ".py",
            ".json",
            ".csv",
            ".html",
            ".yaml",
            ".yml",
            ".xml",
            ".pdf",
            ".log",
        }
    )

    def __init__(
        self,
        llm_client,
        tool_registry=None,
        max_total_input_chars: int = 3000,
        # FIX #5: accept the executor's working directory so that declared file
        # output names (which are relative paths like "report.md") are resolved
        # against the correct location rather than the process CWD, which may
        # differ at verification time.
        working_dir: Path | None = None,
        # Prompt functions — swap these to change verification/dependency-check
        # behaviour without subclassing.  Defaults to the cuddlytoddly-specific
        # implementations in planning/prompts.py.
        verify_prompt_fn: Callable[..., str] | None = None,
        check_deps_prompt_fn: Callable[..., str] | None = None,
    ):
        self.llm = llm_client
        self.tools = tool_registry
        self.max_total_input_chars = max_total_input_chars
        self.working_dir = working_dir
        self._verify_prompt_fn: Callable[..., str] = (
            verify_prompt_fn if verify_prompt_fn is not None else _default_verify_prompt
        )
        self._check_deps_prompt_fn: Callable[..., str] = (
            check_deps_prompt_fn if check_deps_prompt_fn is not None else _default_check_deps_prompt
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def verify_result(self, node, result: str, snapshot) -> tuple[bool, str]:
        if getattr(self.llm, "is_stopped", False):
            return True, "verification skipped — LLM paused"

        # Nodes awaiting user action are not failures — they are correctly surfaced.
        if node.status == "awaiting_user":
            return True, "node is awaiting user action — verification not applicable"

        declared_outputs = node.metadata.get("output", [])
        if not declared_outputs:
            return True, "no declared outputs to verify"

        stripped = result.strip()

        # ── Direct disk check for declared file outputs ───────────────────────
        # For any output declared as a file type (or whose name has a file
        # extension), check disk existence directly.  This is ground truth —
        # no pattern matching on the result string is needed.
        def _output_name(o):
            return o["name"] if isinstance(o, dict) else str(o)

        def _is_file_output(o):
            if isinstance(o, dict):
                return o.get("type") == "file" or any(
                    _output_name(o).endswith(ext) for ext in self.FILE_EXTENSIONS
                )
            return any(str(o).endswith(ext) for ext in self.FILE_EXTENSIONS)

        for output in declared_outputs:
            if _is_file_output(output):
                path = _output_name(output)
                # FIX #5: resolve the declared path against the executor's
                # working directory so the check is CWD-independent.
                resolved = self._resolve_output_path(path)
                if not self._file_exists(resolved):
                    return False, (f"declared file output '{path}' does not exist on disk")

        # ── Collect upstream context for the LLM verifier ────────────────────
        unknown_fields_context = self._collect_unknown_fields(node, snapshot)
        tool_results_context = self._build_tool_results_context(node, snapshot)
        broadening_context = self._build_broadening_context(node)
        upstream_results_context = self._build_upstream_results_context(node, snapshot)

        # ── LLM content check ─────────────────────────────────────────────────
        def _fmt(o):
            if isinstance(o, dict):
                return f"  - [{o['type']}] {o['name']}: {o['description']}"
            return f"  - {o}"

        outputs_text = "\n".join(_fmt(o) for o in declared_outputs)
        prompt = self._verify_prompt_fn(
            node_id=node.id,
            description=node.metadata.get("description", node.id),
            outputs_text=outputs_text,
            result=stripped,
            unknown_fields_context=unknown_fields_context,
            tool_results_context=tool_results_context,
            broadening_context=broadening_context,
            upstream_results_context=upstream_results_context,
        )

        try:
            raw = self.llm.ask(prompt, schema=RESULT_VERIFICATION_SCHEMA)
            parsed = json.loads(raw)
            return bool(parsed.get("satisfied", True)), parsed.get("reason", "")
        except LLMStoppedError:
            return True, "verification skipped — LLM stopped mid-call"
        except Exception as e:
            logger.warning("[QUALITY] verify_result error for %s: %s", node.id, e)
            return True, f"verification skipped — error: {e}"

    def _collect_unknown_fields(self, node, snapshot) -> str:
        """
        Find clarification nodes upstream of this node and return a formatted
        string listing any fields whose value was unknown.

        Returns an empty string when no upstream clarification nodes exist or
        all fields were known — the prompt helper omits the section in that case.
        """
        import json as _json

        _PLACEHOLDERS = {
            "unknown",
            "n/a",
            "not specified",
            "not provided",
            "none",
            "unspecified",
            "tbd",
            "",
        }
        unknown_labels = []

        for dep_id in node.dependencies:
            dep = snapshot.get(dep_id)
            if not dep or dep.node_type != "clarification" or not dep.result:
                continue
            try:
                fields = _json.loads(dep.result)
            except Exception:
                continue
            for f in fields:
                if not isinstance(f, dict):
                    continue
                val = str(f.get("value", "")).strip().lower()
                if val in _PLACEHOLDERS:
                    unknown_labels.append(f.get("label") or f.get("key", "?"))

        if not unknown_labels:
            return ""

        lines = [
            "The following context was unknown when this task ran "
            "(the user did not provide these values):",
        ]
        for label in unknown_labels:
            lines.append(f"  - {label}")
        lines.append(
            "If the result contains specific invented values for these fields "
            "(e.g. a specific salary figure, a named company, a precise percentage) "
            "that do not appear in the upstream task results, mark as not satisfied."
        )
        return "\n".join(lines)

    def check_dependencies(self, node, snapshot) -> dict | None:
        """
        Check whether the upstream results are sufficient to run `node`.

        Returns a bridge_node dict {node_id, description, output} if a gap
        is found, or None if everything looks fine.
        Falls back to None on any error so a broken checker never blocks execution.
        """
        if getattr(self.llm, "is_stopped", False):
            return None

        dep_lines = []
        for dep_id in node.dependencies:
            dep = snapshot.get(dep_id)
            if dep and dep.result:
                dep_lines.append(
                    f"  [{dep_id}]\n"
                    f"    Description: {dep.metadata.get('description', dep_id)}\n"
                    f"    Result:      {dep.result}"
                )
        upstream_text = "\n\n".join(dep_lines) if dep_lines else "  (none — root task)"

        required_inputs = node.metadata.get("required_input", [])

        def _fmt_input(i):
            if isinstance(i, dict):
                return f"  - [{i['type']}] {i['name']}: {i['description']}"
            return f"  - {i}"

        inputs_text = (
            "\n".join(_fmt_input(i) for i in required_inputs)
            if required_inputs
            else "  (not specified)"
        )

        prompt = self._check_deps_prompt_fn(
            node_id=node.id,
            description=node.metadata.get("description", node.id),
            inputs_text=inputs_text,
            upstream_text=upstream_text,
        )

        try:
            raw = self.llm.ask(prompt, schema=DEPENDENCY_CHECK_SCHEMA)
            parsed = json.loads(raw)
            if parsed.get("ok", True):
                return None
            bridge = parsed.get("bridge_node")
            if not bridge or not bridge.get("node_id") or not bridge.get("description"):
                return None
            logger.info(
                "[QUALITY] Gap detected for %s: %s → bridge: %s",
                node.id,
                parsed.get("missing", "?"),
                bridge["node_id"],
            )
            return bridge
        except LLMStoppedError:
            return None
        except Exception as e:
            logger.warning("[QUALITY] check_dependencies error for %s: %s", node.id, e)
            return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resolve_output_path(self, path: str) -> str:
        """
        FIX #5: Resolve a declared output path against self.working_dir if set,
        so that bare filenames like "report.md" are checked in the executor's
        working directory rather than the process CWD.

        Absolute paths are returned unchanged.
        """
        p = Path(path)
        if p.is_absolute() or self.working_dir is None:
            return str(p)
        return str(self.working_dir / p)

    def _build_tool_results_context(self, node, snapshot) -> str:
        """
        Build a factual summary of this node's tool call outcomes for inclusion
        in the verifier prompt.  Returns empty string when no tool calls were made.

        The LLM verifier uses this context to decide whether specific figures in
        the result could plausibly have come from a successful search, or whether
        they are likely fabricated from prior knowledge.
        """
        step_nodes = [
            n
            for nid, n in snapshot.items()
            if nid.startswith(node.id + "__step_") and n.metadata.get("step_type") == "tool_call"
        ]
        if not step_nodes:
            return ""

        total = 0
        failed = 0
        for sn in step_nodes:
            for attempt in sn.metadata.get("attempts", []):
                total += 1
                if attempt.get("status") == "error":
                    failed += 1

        if total == 0:
            return ""

        if failed == total:
            return (
                f"All {total} tool call attempt(s) for this task returned errors "
                "or no results. If the result contains specific figures, names, or "
                "statistics that could only come from a successful search, it is "
                "likely fabricated from the model's prior knowledge."
            )
        elif failed > 0:
            return (
                f"{failed} of {total} tool call attempt(s) returned errors or no "
                f"results; {total - failed} returned data."
            )
        return f"All {total} tool call attempt(s) returned data successfully."

    def _build_broadening_context(self, node) -> str:
        """
        If the node ran with a broadened description (because specific inputs
        were unavailable), return a warning string for the verifier prompt.

        Returns empty string when the node ran with its original description.
        """
        broadened_description = node.metadata.get("broadened_description", "")
        broadened_for_missing = node.metadata.get("broadened_for_missing", [])
        broadened_reason = node.metadata.get("broadened_reason", "")

        if not broadened_description:
            return ""

        missing_text = (
            ", ".join(broadened_for_missing) if broadened_for_missing else "unspecified fields"
        )
        reason_text = f" Reason: {broadened_reason}" if broadened_reason else ""
        return (
            f"This task ran with a broadened goal because the following inputs "
            f"were unavailable: {missing_text}.{reason_text} "
            f'The broadened goal was: "{broadened_description}". '
            f"The result must therefore be general — templates, frameworks, ranges, "
            f"or guided questions. Any specific invented values (exact percentages, "
            f"named personal achievements, specific figures) that could only come "
            f"from the user's private information or a targeted search should cause "
            f"this result to be marked as not satisfied."
        )

    def _build_upstream_results_context(self, node, snapshot) -> str:
        """
        Build a summary of the actual results produced by non-clarification
        upstream task dependencies of this node.

        The verifier uses this to distinguish between values that are legitimately
        derived from upstream task outputs versus values that are fabricated from
        the model's prior knowledge.  Without this context the verifier has no
        way to know that specifics in the result (e.g. a car model name, a price)
        came from a real upstream result rather than being hallucinated.

        The character budget mirrors the executor: max_total_input_chars is split
        evenly across all upstream deps so the verifier always sees the same data
        as the executor prompt did.

        Returns an empty string when there are no completed upstream task deps.
        """
        eligible = [
            (dep_id, dep)
            for dep_id in node.dependencies
            if (dep := snapshot.get(dep_id)) and dep.node_type != "clarification" and dep.result
        ]

        if not eligible:
            return ""

        # Mirror the executor's budget split: total chars divided evenly across
        # all upstream deps so the verifier sees the same data as the executor.
        per_dep_chars = self.max_total_input_chars // len(eligible)

        lines = []
        for dep_id, dep in eligible:
            result_snippet = dep.result
            if len(result_snippet) > per_dep_chars:
                result_snippet = result_snippet[:per_dep_chars] + "…[truncated]"
            declared = dep.metadata.get("output", [])
            output_names = (
                ", ".join(o["name"] for o in declared if isinstance(o, dict) and "name" in o)
                if declared
                else "unspecified"
            )
            lines.append(f"  [{dep_id}] (outputs: {output_names})\n    {result_snippet}")

        header = (
            "The following data was produced by upstream tasks and was available "
            "to this task as input. Specific values in the result that match or "
            "are directly derived from this data are NOT invented:"
        )
        return header + "\n" + "\n\n".join(lines)

    def _file_exists(self, path: str) -> bool:
        if self.tools and hasattr(self.tools, "execute"):
            try:
                self.tools.execute("read_file", {"path": path})
                return True
            except Exception:
                return False
        return os.path.exists(path)
