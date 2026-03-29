# engine/quality_gate.py

import json
from cuddlytoddly.infra.logging import get_logger
from cuddlytoddly.planning.llm_interface import LLMStoppedError

logger = get_logger(__name__)

RESULT_VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "satisfied": {
            "type": "boolean",
            "description": (
                "True if the result fully covers every declared output. "
                "False if something is missing or clearly wrong."
            )
        },
        "reason": {
            "type": "string",
            "description": (
                "One sentence explaining why the result is satisfied or not. "
                "If satisfied=true this can be brief."
            )
        },
    },
    "required": ["satisfied", "reason"],
}

DEPENDENCY_CHECK_SCHEMA = {
    "type": "object",
    "properties": {
        "ok": {
            "type": "boolean",
            "description": (
                "True if the upstream results are sufficient to execute the node. "
                "False if there is a meaningful gap."
            )
        },
        "missing": {
            "type": "string",
            "description": "Short description of what is missing. Only required when ok=false."
        },
        "bridge_node": {
            "type": "object",
            "description": "A single task that would close the gap. Only required when ok=false.",
            "properties": {
                "node_id":     {"type": "string",
                                "description": "Snake_case identifier, no spaces."},
                "description": {"type": "string",
                                "description": "One sentence: what this task does."},
                "output":      {"type": "string",
                                "description": "The single artifact this task produces."},
            },
            "required": ["node_id", "description", "output"],
        },
    },
    "required": ["ok"],
}


class QualityGate:
    """
    LLM-based quality checks for the orchestrator.

    Kept separate from SimpleOrchestrator so the orchestration logic
    (graph mutation, scheduling) stays decoupled from the verification logic
    (prompt building, LLM calls, schema parsing).

    Mirrors the pattern of LLMPlanner / LLMExecutor — the orchestrator
    receives it as a dependency and calls its methods; it never touches
    the graph directly.
    """

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

        # ── Pattern 1: bare filename (no spaces, has extension) ──────────────────
        if self._looks_like_filename(stripped):
            if not self._file_exists(stripped):
                return False, (
                    f"result is a filename ('{stripped}') "
                    f"but the file does not exist on disk"
                )

        # ── Pattern 2: labelled file confirmation "file_written: foo.md" ─────────
        # Extract the filename from the label and check it exists
        import re
        file_label_match = re.match(
            r'^(?:file_written|written_to|saved_to|output_file)\s*:\s*(\S+)', 
            stripped, re.IGNORECASE
        )
        if file_label_match:
            filename = file_label_match.group(1).rstrip(".,;")
            if self._looks_like_filename(filename) or True:  # always check
                if not self._file_exists(filename):
                    return False, (
                        f"result claims file was written ('{filename}') "
                        f"but the file does not exist on disk"
                    )

        # ── Pattern 3: result is just a label/name with no actual content ────────
        # If the result is a single short word/phrase that exactly matches a
        # declared output name, it's a label not content — fail it
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

        # ── LLM content check ────────────────────────────────────────────────────
        def _format_output_for_verification(o):
            if isinstance(o, dict):
                return f"  - [{o['type']}] {o['name']}: {o['description']}"
            return f"  - {o}"

        outputs_text = "\n".join(_format_output_for_verification(o) for o in declared_outputs)
        prompt = f"""You are verifying whether a task result satisfies its declared outputs.

    TASK
    ID:          {node.id}
    Description: {node.metadata.get("description", node.id)}

    DECLARED OUTPUTS (what this task was supposed to produce):
    {outputs_text}

    ACTUAL RESULT:
    {stripped}

    Does the result contain actual substantive content, or is it just a label/filename/stub?
    A result that is just a filename, a single word, or a name matching the output label
    is NOT satisfied — the result must contain the actual data.

    Respond only in JSON matching the schema.
    """
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
        upstream_text = "\n\n".join(dep_lines) if dep_lines else "  (none — root task)"

        required_inputs = node.metadata.get("required_input", [])
        def _format_input_for_check(i):
            if isinstance(i, dict):
                return f"  - [{i['type']}] {i['name']}: {i['description']}"
            return f"  - {i}"

        inputs_text = (
            "\n".join(_format_input_for_check(i) for i in required_inputs)
            if required_inputs else "  (not specified)"
        )

        prompt = f"""You are checking whether a task has everything it needs to execute.

TASK TO RUN
  ID:             {node.id}
  Description:    {node.metadata.get("description", node.id)}
  Required input:
{inputs_text}

AVAILABLE UPSTREAM RESULTS:
{upstream_text}

Is there a meaningful gap — something the task clearly needs but the upstream
results do not provide?

Rules:
- Only flag a real, concrete gap. Do not invent requirements not stated in the task.
- If you flag a gap, propose ONE bridging task that closes it. Keep it coarse-grained:
  a bridging task should do substantial work, not a trivial lookup.
- If the task is a root task or the upstream results are sufficient, set ok=true.

Respond only in JSON matching the schema.
"""
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
                node.id, parsed.get("missing", "?"), bridge["node_id"]
            )
            return bridge
        except LLMStoppedError:
            return None
        except Exception as e:
            logger.warning("[QUALITY] check_dependencies error for %s: %s", node.id, e)
            return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    FILE_EXTENSIONS = {
        ".md", ".txt", ".py", ".json", ".csv", ".html",
        ".yaml", ".yml", ".xml", ".pdf", ".log",
    }

    def _looks_like_filename(self, result: str) -> bool:
        s = result.strip()

        # Must be a single token — no spaces (bare path only, not "file_written: foo.md")
        if " " in s:
            return False
        if "\n" in s or "\\n" in s:
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
