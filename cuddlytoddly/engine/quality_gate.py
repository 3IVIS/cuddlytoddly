# engine/quality_gate.py

import json
import re

from cuddlytoddly.infra.logging import get_logger
from cuddlytoddly.planning.llm_interface import LLMStoppedError
from cuddlytoddly.planning.schemas import (
    RESULT_VERIFICATION_SCHEMA,
    DEPENDENCY_CHECK_SCHEMA,
)
from cuddlytoddly.planning.prompts import (
    build_verify_result_prompt,
    build_check_dependencies_prompt,
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

        # ── Pattern 1: bare filename ──────────────────────────────────────────
        if self._looks_like_filename(stripped):
            if not self._file_exists(stripped):
                return False, (
                    f"result is a filename ('{stripped}') "
                    f"but the file does not exist on disk"
                )

        # ── Pattern 2: labelled file confirmation ─────────────────────────────
        file_label_match = re.match(
            r'^(?:file_written|written_to|saved_to|output_file)\s*:\s*(\S+)',
            stripped, re.IGNORECASE,
        )
        if file_label_match:
            filename = file_label_match.group(1).rstrip(".,;")
            if not self._file_exists(filename):
                return False, (
                    f"result claims file was written ('{filename}') "
                    f"but the file does not exist on disk"
                )

        # ── Pattern 3: result is just a label/name ────────────────────────────
        def _output_name(o):
            return o["name"] if isinstance(o, dict) else str(o)

        is_just_label = (
            "\n" not in stripped
            and " " not in stripped
            and len(stripped) < 60
            and any(
                stripped.lower().replace("_", "") == _output_name(o).lower().replace("_", "")
                for o in declared_outputs
            )
        )
        if is_just_label:
            return False, (
                f"result '{stripped}' appears to be just a label matching the declared "
                f"output name, not actual content. The node must return the actual data."
            )

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

    def _looks_like_filename(self, result: str) -> bool:
        s = result.strip()
        if " " in s or "\n" in s or "\\n" in s:
            return False
        if any(s.startswith(c) for c in ("#", "{", "[", "-", "=", ">")):
            return False
        if len(s) > 200:
            return False
        return any(s.endswith(ext) for ext in self.FILE_EXTENSIONS)

    def _file_exists(self, path: str) -> bool:
        if self.tools and hasattr(self.tools, "execute"):
            try:
                self.tools.execute("read_file", {"path": path})
                return True
            except Exception:
                return False
        import os
        return os.path.exists(path)