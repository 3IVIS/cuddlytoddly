# engine/quality_gate.py

import json
import os

from cuddlytoddly.infra.logging import get_logger
from cuddlytoddly.planning.llm_interface import LLMStoppedError
from cuddlytoddly.planning.prompts import (
    build_check_dependencies_prompt,
    build_verify_result_prompt,
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
    """

    FILE_EXTENSIONS = frozenset({
        ".md", ".txt", ".py", ".json", ".csv", ".html",
        ".yaml", ".yml", ".xml", ".pdf", ".log",
    })

    def __init__(self, llm_client, tool_registry=None):
        self.llm   = llm_client
        self.tools = tool_registry

    # ── Public API ────────────────────────────────────────────────────────────

    def verify_result(self, node, result: str, snapshot) -> tuple[bool, str]:
        if getattr(self.llm, "is_stopped", False):
            return True, "verification skipped — LLM paused"

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
                return (
                    o.get("type") == "file"
                    or any(_output_name(o).endswith(ext) for ext in self.FILE_EXTENSIONS)
                )
            return any(str(o).endswith(ext) for ext in self.FILE_EXTENSIONS)

        for output in declared_outputs:
            if _is_file_output(output):
                path = _output_name(output)
                if not self._file_exists(path):
                    return False, (
                        f"declared file output '{path}' does not exist on disk"
                    )

        # ── Collect upstream context for the LLM verifier ────────────────────
        unknown_fields_context = self._collect_unknown_fields(node, snapshot)
        tool_results_context   = self._build_tool_results_context(node, snapshot)
        broadening_context     = self._build_broadening_context(node)

        # ── LLM content check ─────────────────────────────────────────────────
        def _fmt(o):
            if isinstance(o, dict):
                return f"  - [{o['type']}] {o['name']}: {o['description']}"
            return f"  - {o}"

        outputs_text = "\n".join(_fmt(o) for o in declared_outputs)
        prompt = build_verify_result_prompt(
            node_id=node.id,
            description=node.metadata.get("description", node.id),
            outputs_text=outputs_text,
            result=stripped,
            unknown_fields_context=unknown_fields_context,
            tool_results_context=tool_results_context,
            broadening_context=broadening_context,
        )

        try:
            raw    = self.llm.ask(prompt, schema=RESULT_VERIFICATION_SCHEMA)
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
        _PLACEHOLDERS = {"unknown", "n/a", "not specified", "not provided",
                         "none", "unspecified", "tbd", ""}
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
        upstream_text = (
            "\n\n".join(dep_lines) if dep_lines else "  (none — root task)"
        )

        required_inputs = node.metadata.get("required_input", [])

        def _fmt_input(i):
            if isinstance(i, dict):
                return f"  - [{i['type']}] {i['name']}: {i['description']}"
            return f"  - {i}"

        inputs_text = (
            "\n".join(_fmt_input(i) for i in required_inputs)
            if required_inputs else "  (not specified)"
        )

        prompt = build_check_dependencies_prompt(
            node_id=node.id,
            description=node.metadata.get("description", node.id),
            inputs_text=inputs_text,
            upstream_text=upstream_text,
        )

        try:
            raw    = self.llm.ask(prompt, schema=DEPENDENCY_CHECK_SCHEMA)
            parsed = json.loads(raw)
            if parsed.get("ok", True):
                return None
            bridge = parsed.get("bridge_node")
            if not bridge or not bridge.get("node_id") or not bridge.get("description"):
                return None
            logger.info(
                "[QUALITY] Gap detected for %s: %s → bridge: %s",
                node.id, parsed.get("missing", "?"), bridge["node_id"],
            )
            return bridge
        except LLMStoppedError:
            return None
        except Exception as e:
            logger.warning("[QUALITY] check_dependencies error for %s: %s", node.id, e)
            return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _build_tool_results_context(self, node, snapshot) -> str:
        """
        Build a factual summary of this node's tool call outcomes for inclusion
        in the verifier prompt.  Returns empty string when no tool calls were made.

        The LLM verifier uses this context to decide whether specific figures in
        the result could plausibly have come from a successful search, or whether
        they are likely fabricated from prior knowledge.
        """
        step_nodes = [
            n for nid, n in snapshot.items()
            if nid.startswith(node.id + "__step_")
            and n.metadata.get("step_type") == "tool_call"
        ]
        if not step_nodes:
            return ""

        total  = 0
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
        broadened_reason      = node.metadata.get("broadened_reason", "")

        if not broadened_description:
            return ""

        missing_text = (
            ", ".join(broadened_for_missing)
            if broadened_for_missing else "unspecified fields"
        )
        reason_text = f" Reason: {broadened_reason}" if broadened_reason else ""
        return (
            f"This task ran with a broadened goal because the following inputs "
            f"were unavailable: {missing_text}.{reason_text} "
            f"The broadened goal was: \"{broadened_description}\". "
            f"The result must therefore be general — templates, frameworks, ranges, "
            f"or guided questions. Any specific invented values (exact percentages, "
            f"named personal achievements, specific figures) that could only come "
            f"from the user's private information or a targeted search should cause "
            f"this result to be marked as not satisfied."
        )

    def _file_exists(self, path: str) -> bool:
        if self.tools and hasattr(self.tools, "execute"):
            try:
                self.tools.execute("read_file", {"path": path})
                return True
            except Exception:
                return False
        return os.path.exists(path)



