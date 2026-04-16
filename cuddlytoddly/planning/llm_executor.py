# planning/llm_executor.py

import json
import uuid
from dataclasses import dataclass
from dataclasses import field as _dc_field
from pathlib import Path

from cuddlytoddly.infra.logging import get_logger
from cuddlytoddly.planning.llm_interface import LLMStoppedError, NativeToolResponse
from cuddlytoddly.planning.prompts import (
    build_awaiting_input_check_prompt,
    build_broadened_description_prompt,
    build_executor_file_output_instruction,
    build_executor_file_reminder,
    build_executor_inline_output_instruction,
    build_executor_native_file_output_instruction,
    build_executor_native_file_reminder,
    build_executor_native_inline_output_instruction,
    build_executor_native_prompt,
    build_executor_outputs_block,
    build_executor_prompt,
    build_executor_retry_notice,
)
from cuddlytoddly.planning.schemas import (
    AWAITING_INPUT_CHECK_SCHEMA,
    BROADENED_DESCRIPTION_SCHEMA,
    EXECUTION_TURN_SCHEMA,
)

logger = get_logger(__name__)

# FIX: _CWD_LOCK removed.  The original implementation called os.chdir() +
# held a process-wide lock for the full duration of every tool call, which
# serialised all tool execution across all concurrent runs and all thread-pool
# workers.  A 30-second shell command blocked every other tool call in the
# process.
#
# The replacement approach injects the working directory via a reserved
# ``_cwd`` key in tool_args (see _run_tool below).  Each tool handles it
# directly:
#   • run_shell  — passes cwd= to subprocess.Popen; no process CWD mutation.
#   • run_python — uses a per-module lock (_PYTHON_CWD_LOCK) held only during
#                  in-process eval/exec, not for the full tool call.
#   • file tools — use absolute paths from _safe_resolve(); ignore _cwd.


@dataclass
class AwaitingInputSignal:
    """
    Produced by _preflight_awaiting_input when some required inputs are missing.

    The signal no longer blocks execution — it carries the broadened_description
    that execute() uses as the effective task goal for this run, along with
    metadata the orchestrator writes back to the node after execution completes.

    Fields
    ------
    reason               : Human-readable explanation of what is missing.
    missing_fields       : Keys of existing clarification fields that are unknown.
    new_fields           : New fields to add to the clarification form.
    clarification_node_id: Upstream clarification node to patch.
    broadened_description: Rephrased task goal that works without the missing inputs.
    broadened_for_missing: The missing field keys active when the broadened
                           description was generated — used to decide whether to
                           reuse or regenerate on the next execution.
    """

    reason: str
    missing_fields: list = _dc_field(default_factory=list)
    new_fields: list = _dc_field(default_factory=list)
    clarification_node_id: str = ""
    broadened_description: str = ""
    broadened_for_missing: list = _dc_field(default_factory=list)
    broadened_output: list = _dc_field(default_factory=list)
    broadened_steps: list = _dc_field(default_factory=list)


class LLMExecutor:
    """
    Executes a ready task node by prompting the LLM with the node's
    description, its required inputs (resolved from upstream results),
    and an optional set of tools.

    The execution loop:
      1. Build a prompt from node context + conversation history
      2. Ask the LLM → it responds with either done=True + result,
         or done=False + tool_call
      3. If tool_call, run the tool and append the result to history
      4. Repeat until done=True or max_turns reached

    All numeric limits come from the application config (passed via __init__)
    so users can tune behaviour without editing source code.

    FIX: accepts ``working_dir`` so file tools always run relative to the
    run's output sandbox.  The directory is forwarded to tools via the
    reserved ``_cwd`` key in tool_args rather than via os.chdir(), so
    concurrent executor threads never contend on a process-wide lock.
    """

    # File extensions that trigger "write_file" enforcement
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
        max_turns: int = 5,
        max_inline_result_chars: int = 3000,
        max_total_input_chars: int = 3000,
        max_tool_result_chars: int = 2000,
        max_history_entries: int = 3,
        working_dir: Path | str | None = None,
    ):
        self.llm = llm_client
        self.tools = tool_registry
        self.max_turns = max_turns
        self.max_inline_result_chars = max_inline_result_chars
        self.max_total_input_chars = max_total_input_chars
        self.max_tool_result_chars = max_tool_result_chars
        self.max_history_entries = max_history_entries
        # store working_dir as explicit state; forwarded to tools via _cwd arg key
        self.working_dir: Path | None = Path(working_dir) if working_dir else None

    # ── Tool execution with CWD sandboxing ────────────────────────────────────

    def _run_tool(self, tool_name: str, tool_args: dict):
        """Execute a registered tool, routing the working directory via args.

        Instead of calling os.chdir() under a process-wide lock (which
        serialised every tool call across all concurrent runs), we inject the
        working directory via a reserved ``_cwd`` key in a *copy* of tool_args.
        Each tool handles it without mutating the process CWD:

          • run_shell  — pops ``_cwd`` and passes it as ``cwd=`` to
                         subprocess.Popen.  No lock needed.
          • run_python — pops ``_cwd`` and uses a per-module
                         ``_PYTHON_CWD_LOCK`` held only for the duration of
                         in-process eval/exec (typically milliseconds), not
                         for the full tool-call lifetime.
          • file tools — use absolute paths via _safe_resolve() and ignore
                         ``_cwd`` entirely.

        The original caller's dict is never mutated; we always copy.
        """
        if self.working_dir is not None:
            tool_args = {**tool_args, "_cwd": str(self.working_dir)}
        return self.tools.execute(tool_name, tool_args)

    # ──────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_clarification_fields(result_json: str) -> tuple[list, list]:
        """
        Parse a clarification node result into (known_fields, unknown_fields).

        known_fields   : list of {key, label, value} where value is not a placeholder
        unknown_fields : list of {key, label} where value was "unknown" or similar

        Used by _resolve_inputs to annotate the executor prompt, and passed
        to the quality gate so it can flag fabricated values for unknown fields.
        """
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
        try:
            fields = json.loads(result_json)
        except Exception:
            return [], []

        known, unknown = [], []
        for f in fields:
            if not isinstance(f, dict):
                continue
            val = str(f.get("value", "")).strip().lower()
            if val in _PLACEHOLDERS:
                unknown.append({"key": f.get("key", ""), "label": f.get("label", f.get("key", ""))})
            else:
                known.append(
                    {
                        "key": f.get("key", ""),
                        "label": f.get("label", f.get("key", "")),
                        "value": f.get("value", ""),
                    }
                )
        return known, unknown

    def _resolve_inputs(self, node, snapshot):

        def _format_output_list(outputs):
            if not outputs:
                return []
            result = []
            for o in outputs:
                if isinstance(o, dict):
                    result.append(f"{o['name']} ({o['type']}): {o['description']}")
                else:
                    result.append(str(o))  # backward compat
            return result

        resolved = []
        for dep_id in node.dependencies:
            dep = snapshot.get(dep_id)
            if not dep or not dep.result:
                continue

            # ── Clarification node — render as structured known/unknown blocks ──
            if dep.node_type == "clarification":
                known, unknown = self._parse_clarification_fields(dep.result)
                lines = ["[Goal context from clarification node]"]
                if known:
                    lines.append("  Known — use these values directly:")
                    for f in known:
                        lines.append(f"    {f['label']}: {f['value']}")
                if unknown:
                    lines.append(
                        "  Unknown — the user has not provided these values. "
                        "Do NOT invent or assume specific values for them. "
                        "If your task cannot proceed without them, state what is "
                        "missing and produce a template or general answer instead:"
                    )
                    for f in unknown:
                        lines.append(f"    {f['label']}: not provided")
                resolved.append(
                    {
                        "node_id": dep_id,
                        "description": dep.metadata.get("description", dep_id),
                        "declared_output": [],
                        "result": "\n".join(lines),
                        "_unknown_fields": unknown,
                        "_known_fields": known,
                    }
                )
                continue

            resolved.append(
                {
                    "node_id": dep_id,
                    "description": dep.metadata.get("description", dep_id),
                    "declared_output": _format_output_list(dep.metadata.get("output", [])),
                    "result": dep.result,
                    # Use produced_output when available — it reflects what the
                    # upstream node actually delivered at runtime (which may be
                    # a broadened output contract if it ran without its full
                    # required inputs).  Fall back to the declared output
                    # metadata when produced_output is absent (e.g. the node
                    # has not yet completed, or it predates this field).
                    "_output_names": [
                        o["name"]
                        for o in (
                            dep.metadata.get("produced_output") or dep.metadata.get("output", [])
                        )
                        if isinstance(o, dict) and "name" in o
                    ],
                }
            )

        # Distribute the char budget evenly across all upstream results
        if resolved:
            budget_per_dep = self.max_total_input_chars // len(resolved)
            for entry in resolved:
                r = entry["result"]
                if len(r) > budget_per_dep:
                    entry["result"] = r[:budget_per_dep] + f"\n…[truncated, {len(r)} chars total]"

        return resolved

    def _tool_schema_summary(self):
        if not self.tools:
            return (
                "NO TOOLS ARE AVAILABLE FOR THIS TASK.\n"
                "Do NOT attempt to call any tool or function — there are none registered.\n"
                "You must complete this task using your own knowledge and reasoning.\n"
                "Set done=true with a result based on what you know."
            )
        lines = []
        for name, tool in self.tools.tools.items():
            desc = getattr(tool, "description", "no description")
            schema = getattr(tool, "input_schema", {})
            lines.append(f"- {name}: {desc}. Args: {json.dumps(schema)}")
        return "\n".join(lines)

    def _build_prompt(
        self,
        node,
        resolved_inputs,
        history,
        extra_reminder="",
        turns_remaining=0,
        description_override="",
        output_override=None,
        steps_override=None,
    ):
        def _fmt_output(o):
            if isinstance(o, dict):
                return f"  - [{o['type']}] {o['name']}: {o['description']}"
            return f"  - {o}"

        def _output_name(o):
            return o["name"] if isinstance(o, dict) else str(o)

        def _is_file(o):
            if isinstance(o, dict):
                return o.get("type") == "file" or any(
                    _output_name(o).endswith(ext) for ext in self.FILE_EXTENSIONS
                )
            return any(str(o).endswith(ext) for ext in self.FILE_EXTENSIONS)

        if resolved_inputs:
            inputs_text = "\n".join(
                f"  [{e['node_id']}]\n"
                f"    Description:      {e['description']}\n"
                f"    Declared outputs: {e['declared_output']}\n"
                f"    Actual result:    {e['result']}"
                for e in resolved_inputs
            )
        else:
            inputs_text = "  (none — this is a root task)"

        retry = node.metadata.get("retry_count", 0)
        if retry > 0:
            failure = node.metadata.get("verification_failure", "unknown")[:200]
            prev_result = node.result or "(none)"
            if len(prev_result) > 200:
                prev_result = prev_result[:200] + "…"
        else:
            failure = ""
            prev_result = ""
        retry_notice = build_executor_retry_notice(retry, failure, prev_result)

        history_text = ""
        if history:
            parts = []
            for entry in history:
                parts.append(
                    f"  Tool: {entry['name']}\n"
                    f"  Args: {json.dumps(entry['args'])}\n"
                    f"  Result: {entry['result']}"
                )
            history_text = "Previous tool calls this turn:\n" + "\n\n".join(parts)

        tools_text = self._tool_schema_summary()

        declared_outputs = (
            output_override if output_override is not None else node.metadata.get("output", [])
        )
        expected_files = [_output_name(o) for o in declared_outputs if _is_file(o)]

        if expected_files:
            output_instruction = build_executor_file_output_instruction(expected_files)
        else:
            output_instruction = build_executor_inline_output_instruction(
                self.max_inline_result_chars
            )

        outputs_text = (
            "\n".join(_fmt_output(o) for o in declared_outputs)
            if declared_outputs
            else "  (not specified)"
        )
        outputs_block = build_executor_outputs_block(outputs_text)

        # Build execution steps block
        steps = (
            steps_override
            if steps_override is not None
            else node.metadata.get("execution_steps", [])
        )
        if steps:
            steps_lines = []
            for i, s in enumerate(steps, 1):
                steps_lines.append(
                    f"  {i}. [{s.get('execution_type', '?')}] {s.get('description', '')}"
                )
                if s.get("produces"):
                    steps_lines.append(f"     → {s['produces']}")
            steps_text = "\n".join(steps_lines)
        else:
            steps_text = "  (not specified — use your best judgement)"

        logger.info(
            "[EXECUTOR] Prompt sections for %s — "
            "retry=%d chars, outputs=%d chars, inputs=%d chars, "
            "tools=%d chars, history=%d chars, steps=%d",
            node.id,
            len(retry_notice),
            len(outputs_text),
            len(inputs_text),
            len(tools_text),
            len(history_text),
            len(steps),
        )

        return build_executor_prompt(
            node_id=node.id,
            description=description_override or node.metadata.get("description", node.id),
            retry_notice=retry_notice,
            extra_reminder=extra_reminder,
            outputs_block=outputs_block,
            output_instruction=output_instruction,
            inputs_text=inputs_text,
            tools_text=tools_text,
            history_text=history_text,
            steps_text=steps_text,
            max_inline_result_chars=self.max_inline_result_chars,
            turns_remaining=turns_remaining,
        )

    def _fmt_known_fields(self, resolved_inputs: list) -> str:
        known_fields = []
        for entry in resolved_inputs:
            known_fields.extend(entry.get("_known_fields", []))
        if not known_fields:
            return ""
        return "\n".join(
            f"  - {f.get('key', '?')} ({f.get('label', '?')}): {f.get('value', '')}"
            for f in known_fields
        )

    def _generate_broadened_description(
        self, node, missing_keys: list, known_fields_text: str
    ) -> tuple[str, list]:
        """
        Fallback broadening call when the preflight returned blocked=true but
        an empty broadened_description.

        Returns (broadened_description, broadened_steps).  Both will be empty
        on failure so the caller can detect and abort.
        """
        if getattr(self.llm, "is_stopped", False):
            return "", []
        prompt = build_broadened_description_prompt(
            node_id=node.id,
            original_description=node.metadata.get("description", node.id),
            missing_keys=missing_keys,
            known_fields_text=known_fields_text,
            original_steps=node.metadata.get("execution_steps", []),
        )
        try:
            raw = self.llm.ask(prompt, schema=BROADENED_DESCRIPTION_SCHEMA)
            parsed = json.loads(raw)
            desc = parsed.get("broadened_description", "").strip()
            steps = parsed.get("broadened_steps", [])
            return desc, steps
        except Exception as e:
            logger.warning(
                "[EXECUTOR] Broadened description fallback call failed for %s: %s",
                node.id,
                e,
            )
            return "", []

    def _preflight_awaiting_input(self, node, resolved_inputs) -> "AwaitingInputSignal | None":
        """
        Ask the LLM whether this task can be executed with the currently
        available information and tools.

        Returns an AwaitingInputSignal if the task is blocked on missing user
        input, None if it can proceed.
        """
        unknown_fields = []
        known_fields = []
        clar_node_id = ""

        for entry in resolved_inputs:
            if "_unknown_fields" not in entry:
                continue
            clar_node_id = entry["node_id"]
            unknown_fields.extend(entry["_unknown_fields"])
            known_fields.extend(entry.get("_known_fields", []))

        all_clar_keys = {f.get("key") for f in known_fields + unknown_fields}
        required_inputs = node.metadata.get("required_input", [])

        upstream_output_names: set[str] = set()
        for entry in resolved_inputs:
            if "_unknown_fields" in entry:
                continue
            upstream_output_names.update(entry.get("_output_names", []))

        def _make_new_field(r: dict) -> dict:
            name = r.get("name", "")
            label = name.replace("_", " ").title()
            return {
                "key": name,
                "label": label,
                "value": "unknown",
                "rationale": (f"Required by task '{node.id}': {r.get('description', label)}"),
            }

        auto_new_fields = [
            _make_new_field(r)
            for r in required_inputs
            if r.get("name")
            and r.get("name") not in all_clar_keys
            and r.get("name") not in upstream_output_names
        ]

        if auto_new_fields:
            logger.info(
                "[EXECUTOR] Node %s has required inputs not covered by any "
                "clarification field — will add: %s",
                node.id,
                [f["key"] for f in auto_new_fields],
            )

        current_missing_keys = sorted(
            set([f.get("key") for f in unknown_fields] + [f["key"] for f in auto_new_fields])
        )

        if not current_missing_keys:
            return None

        previous_failure = node.metadata.get("verification_failure", "")
        stored_broadened = node.metadata.get("broadened_description", "")
        stored_for_missing = sorted(node.metadata.get("broadened_for_missing", []))
        if stored_broadened and stored_for_missing == current_missing_keys and not previous_failure:
            logger.info(
                "[EXECUTOR] Node %s: reusing stored broadened description "
                "(missing set unchanged: %s)",
                node.id,
                current_missing_keys,
            )
            return AwaitingInputSignal(
                reason="Reusing stored broadened description — missing fields unchanged.",
                missing_fields=[f.get("key") for f in unknown_fields],
                new_fields=auto_new_fields,
                clarification_node_id=clar_node_id,
                broadened_description=stored_broadened,
                broadened_for_missing=stored_for_missing,
                broadened_output=node.metadata.get("broadened_output", []),
                broadened_steps=node.metadata.get("broadened_steps", []),
            )

        if previous_failure:
            logger.info(
                "[EXECUTOR] Node %s: previous broadened execution failed verification "
                "('%s...') — regenerating broadened description with failure context",
                node.id,
                previous_failure[:80],
            )

        # Fix #9: when the LLM is stopped and there ARE missing fields, returning
        # None here tells execute() "no missing inputs — proceed normally", which
        # causes it to run the task with the un-broadened description and
        # potentially hallucinated field values.  The subsequent LLM call inside
        # the execution loop will raise LLMStoppedError and reset the node
        # anyway, but the cleaner approach is to raise immediately so the
        # executor's uniform LLMStoppedError handler deals with it, not a
        # silent None return that masks the real reason for the abort.
        if getattr(self.llm, "is_stopped", False):
            from cuddlytoddly.planning.llm_interface import LLMStoppedError

            raise LLMStoppedError("LLM is paused — deferring broadening call until resumed")

        def _fmt_fields(fields):
            return (
                "\n".join(
                    f"  - {f.get('key', '?')} ({f.get('label', f.get('key', '?'))}): "
                    f"{f.get('value', 'unknown')}"
                    for f in fields
                )
                if fields
                else "  (none)"
            )

        known_fields_text = _fmt_fields(known_fields)
        unknown_fields_text = _fmt_fields(unknown_fields)
        tools_text = self._tool_schema_summary()

        if required_inputs:
            required_input_text = "\n".join(
                f"  - {r.get('name', '?')} ({r.get('type', '?')}): {r.get('description', '')}"
                for r in required_inputs
            )
        else:
            required_input_text = "  (none declared)"

        prompt = build_awaiting_input_check_prompt(
            node_id=node.id,
            description=node.metadata.get("description", node.id),
            tools_text=tools_text,
            known_fields_text=known_fields_text,
            unknown_fields_text=unknown_fields_text,
            required_input_text=required_input_text,
            previous_failure=previous_failure,
        )

        try:
            raw = self.llm.ask(prompt, schema=AWAITING_INPUT_CHECK_SCHEMA)
            parsed = json.loads(raw)
        except Exception as e:
            logger.warning(
                "[EXECUTOR] Preflight LLM check failed for %s: %s — proceeding",
                node.id,
                e,
            )
            return None

        if not parsed.get("blocked", False):
            return None

        llm_new_fields = parsed.get("new_fields", [])
        llm_new_keys = {f.get("key") for f in llm_new_fields}
        merged_new = llm_new_fields + [f for f in auto_new_fields if f["key"] not in llm_new_keys]

        llm_missing = parsed.get("missing_fields", [])
        auto_missing = [f["key"] for f in auto_new_fields if f["key"] not in llm_missing]
        merged_missing = llm_missing + auto_missing

        broadened_for_missing = sorted(set(merged_missing + [f.get("key") for f in merged_new]))

        broadened_description = parsed.get("broadened_description", "")
        if not broadened_description:
            logger.warning(
                "[EXECUTOR] Node %s: preflight returned blocked=true but no "
                "broadened_description — will execute with original description",
                node.id,
            )

        broadened_steps = parsed.get("broadened_steps", [])

        # Safety net: if the LLM produced a broadened_description but returned
        # empty broadened_steps, derive a minimal step from the description so
        # the UI always has something concrete to show.
        if broadened_description and not broadened_steps:
            logger.warning(
                "[EXECUTOR] Node %s: broadened_description present but broadened_steps "
                "is empty — deriving minimal steps from original execution_steps",
                node.id,
            )
            original_steps = node.metadata.get("execution_steps", [])
            if original_steps:
                # Reuse original steps with a note that they apply to the broadened goal
                broadened_steps = [
                    {
                        "execution_type": s.get("execution_type", "write_analysis"),
                        "description": s.get("description", broadened_description),
                        "produces": s.get("produces", "Output for the broadened goal."),
                    }
                    for s in original_steps
                ]
            else:
                # No original steps to adapt — create a single generic step
                broadened_steps = [
                    {
                        "execution_type": "write_analysis",
                        "description": broadened_description,
                        "produces": "Output for the broadened goal.",
                    }
                ]

        return AwaitingInputSignal(
            reason=parsed.get("reason", "task requires user input"),
            missing_fields=merged_missing,
            new_fields=merged_new,
            clarification_node_id=clar_node_id,
            broadened_description=broadened_description,
            broadened_for_missing=broadened_for_missing,
            broadened_output=parsed.get("broadened_output", []),
            broadened_steps=broadened_steps,
        )

    def _select_goal_mode(self, node, resolved_inputs, signal):
        """
        Decide whether to run the original plan or the broadened plan.

        Returns (effective_description, effective_outputs, effective_steps).

        Logic:
        - If signal is None (all required inputs present) → original.
        - If signal is not None (broadened) → broadened.
        - Exception: if a broadened signal exists but the actual resolved input
          names all match the originally declared required_input names, the
          upstream nodes provided correctly-named outputs (they resolved back to
          their original goal), so we fall back to the original plan as well.
        """
        original_desc = node.metadata.get("description", node.id)
        original_outputs = node.metadata.get("output", [])
        original_steps = node.metadata.get("execution_steps", [])

        if signal is None:
            return original_desc, original_outputs, original_steps

        # Check if upstream inputs match the original required_input contract
        required_names = {
            r.get("name")
            for r in node.metadata.get("required_input", [])
            if r.get("name")
            and r.get("name")
            not in {
                f.get("key") for entry in resolved_inputs for f in entry.get("_unknown_fields", [])
            }
        }
        upstream_names: set[str] = set()
        for entry in resolved_inputs:
            if "_unknown_fields" in entry:
                continue
            upstream_names.update(entry.get("_output_names", []))

        inputs_match_original = required_names and required_names.issubset(upstream_names)

        if inputs_match_original:
            logger.info(
                "[EXECUTOR] Node %s: upstream inputs match original contract "
                "(%s) — using original goal despite broadened signal",
                node.id,
                sorted(required_names),
            )
            return original_desc, original_outputs, original_steps

        # Use the broadened plan
        broadened_desc = signal.broadened_description or original_desc
        broadened_outputs = signal.broadened_output if signal.broadened_output else original_outputs
        broadened_steps = signal.broadened_steps if signal.broadened_steps else original_steps
        return broadened_desc, broadened_outputs, broadened_steps

    def _llm_executable_types(self) -> frozenset:
        """
        Return the set of execution_type values the LLM can perform given the
        currently registered tools.  Always includes LLM-native types (write_*,
        analyse_*, search_web, fetch_url) when the matching tool is registered.
        """
        native = {
            "write_plan",
            "write_document",
            "write_analysis",
            "write_code",
            "analyse_data",
            "summarise",
            "synthesise",
        }
        if self.tools:
            for name in self.tools.tools:
                if name == "web_search":
                    native.add("search_web")
                elif name == "fetch_url":
                    native.add("fetch_url")
                else:
                    native.add(name)
        return frozenset(native)

    def execute(self, node, snapshot, reporter=None):
        resolved_inputs = self._resolve_inputs(node, snapshot)

        signal = self._preflight_awaiting_input(node, resolved_inputs)

        if signal is not None:
            if signal.broadened_description:
                if reporter:
                    reporter.on_broadened_execution(signal)
            else:
                logger.warning(
                    "[EXECUTOR] Node %s: preflight returned no broadened_description "
                    "— making focused fallback call",
                    node.id,
                )
                effective_description, fallback_steps = self._generate_broadened_description(
                    node=node,
                    missing_keys=signal.broadened_for_missing or signal.missing_fields,
                    known_fields_text=self._fmt_known_fields(resolved_inputs),
                )
                if not effective_description:
                    logger.error(
                        "[EXECUTOR] Node %s: fallback broadening call also returned "
                        "empty — skipping execution to avoid hallucination",
                        node.id,
                    )
                    return None
                signal = AwaitingInputSignal(
                    reason=signal.reason,
                    missing_fields=signal.missing_fields,
                    new_fields=signal.new_fields,
                    clarification_node_id=signal.clarification_node_id,
                    broadened_description=effective_description,
                    broadened_for_missing=signal.broadened_for_missing,
                    broadened_output=signal.broadened_output,
                    # Use fallback steps if available, otherwise fall back to
                    # whatever the primary preflight returned (may be empty).
                    broadened_steps=fallback_steps or signal.broadened_steps,
                )
                if reporter:
                    reporter.on_broadened_execution(signal)

        effective_description, effective_outputs, effective_steps = self._select_goal_mode(
            node, resolved_inputs, signal
        )

        use_native = (
            getattr(self.llm, "supports_native_tools", False)
            and self.tools
            and len(self.tools.tools) > 0
        )
        logger.info(
            "[EXECUTOR] Node %s: using %s execution path",
            node.id,
            "native tool-use" if use_native else "legacy JSON",
        )

        if use_native:
            return self._execute_native(
                node,
                resolved_inputs,
                effective_description,
                effective_outputs,
                effective_steps,
                reporter,
            )
        else:
            return self._execute_legacy(
                node,
                resolved_inputs,
                effective_description,
                effective_outputs,
                effective_steps,
                reporter,
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Shared tool-dispatch helpers used by both execution paths
    # ──────────────────────────────────────────────────────────────────────────

    def _dispatch_tool(
        self,
        node_id: str,
        tool_name: str,
        tool_args: dict,
        reporter,
        step_id: str | None,
    ) -> tuple[str, bool]:
        """
        Execute *tool_name* with *tool_args*, report progress, truncate the
        result to ``max_tool_result_chars``, and return ``(result_str, error)``.

        Centralising this logic removes ~40 lines of identical code that
        previously lived in both ``_execute_legacy`` and ``_execute_native``.
        """
        error = False
        try:
            tool_result = self._run_tool(tool_name, tool_args)
        except Exception as e:
            tool_result = f"ERROR: {e}"
            error = True
            logger.error("[EXECUTOR] Tool '%s' raised: %s", tool_name, e)

        tool_result_str = str(tool_result)
        if not error and tool_result_str.startswith("ERROR:"):
            error = True
            logger.warning(
                "[EXECUTOR] Tool '%s' returned an error string: %.120s",
                tool_name,
                tool_result_str,
            )

        if reporter and step_id:
            reporter.on_tool_done(step_id, tool_name, tool_args, tool_result_str, error=error)

        if len(tool_result_str) > self.max_tool_result_chars:
            tool_result_str = (
                tool_result_str[: self.max_tool_result_chars]
                + f"\n…[truncated — {len(tool_result_str)} chars total]"
            )

        return tool_result_str, error

    def _append_to_history(self, history: list, entry: dict) -> list:
        """Append *entry* to *history*, trimming to ``max_history_entries``."""
        history.append(entry)
        if len(history) > self.max_history_entries:
            history = history[-self.max_history_entries :]
        return history

    # ──────────────────────────────────────────────────────────────────────────
    # Legacy execution loop (JSON-in-prompt, done+tool_call protocol)
    # ──────────────────────────────────────────────────────────────────────────

    def _execute_legacy(
        self,
        node,
        resolved_inputs,
        effective_description: str,
        effective_outputs: list,
        effective_steps: list,
        reporter,
    ):
        def _output_name(o):
            return o["name"] if isinstance(o, dict) else str(o)

        def _is_file(o):
            if isinstance(o, dict):
                return o.get("type") == "file" or any(
                    _output_name(o).endswith(ext) for ext in self.FILE_EXTENSIONS
                )
            return any(str(o).endswith(ext) for ext in self.FILE_EXTENSIONS)

        expected_files = [_output_name(o) for o in effective_outputs if _is_file(o)]
        history: list[dict] = []
        tool_not_found_count = 0

        # ── Step capability pre-check ─────────────────────────────────────────
        # Partition the declared execution steps into those the LLM can handle
        # and those that require real-world user action.
        executable_types = self._llm_executable_types()
        awaiting_steps = []  # steps the user must perform
        llm_steps = []  # steps the LLM can perform

        for step in effective_steps:
            etype = step.get("execution_type", "")
            if etype in executable_types or not etype:
                llm_steps.append(step)
            else:
                awaiting_steps.append(step)

        if awaiting_steps:
            logger.info(
                "[EXECUTOR] Node %s: %d step(s) require user action: %s",
                node.id,
                len(awaiting_steps),
                [s.get("execution_type") for s in awaiting_steps],
            )

        # ── Early exit: all steps require user action — skip the LLM loop ────
        # If there are no LLM-executable steps at all, there is nothing for the
        # model to do. Build the handoff artifact immediately without making any
        # LLM or tool calls.
        if awaiting_steps and not llm_steps:
            logger.info(
                "[EXECUTOR] Node %s: all %d step(s) require user action — "
                "surfacing immediately without LLM calls",
                node.id,
                len(awaiting_steps),
            )
            handoff_lines = [
                "The following steps require your action before this task is complete:",
                "",
            ]
            for s in awaiting_steps:
                handoff_lines.append(f"• [{s.get('execution_type')}] {s.get('description', '')}")
                handoff_lines.append(f"  Purpose: {s.get('produces', '')}")
            return {
                "_awaiting_user": True,
                "handoff_artifact": "\n".join(handoff_lines),
                "pending_steps": [s.get("execution_type") for s in awaiting_steps],
                "partial_result": "",
            }

        for turn in range(self.max_turns):
            turns_remaining = self.max_turns - turn
            extra_reminder = ""

            if expected_files and "write_file" not in {h["name"] for h in history}:
                extra_reminder += build_executor_file_reminder(expected_files, turns_remaining)

            if reporter:
                reporter.on_llm_turn(turn)

            prompt = self._build_prompt(
                node,
                resolved_inputs,
                history,
                extra_reminder=extra_reminder,
                turns_remaining=turns_remaining,
                description_override=effective_description,
                output_override=effective_outputs,
                steps_override=llm_steps if llm_steps else effective_steps,
            )

            try:
                raw = self.llm.ask(prompt, schema=EXECUTION_TURN_SCHEMA)
            except LLMStoppedError:
                logger.warning("[EXECUTOR] LLM stopped during execution of %s", node.id)
                return None
            except Exception as e:
                logger.error("[EXECUTOR] LLM error for node %s: %s", node.id, e)
                if reporter:
                    reporter.on_llm_error(turn, str(e))
                return None

            try:
                response = json.loads(raw)
            except json.JSONDecodeError as e:
                logger.error("[EXECUTOR] JSON parse error for node %s: %s", node.id, e)
                if reporter:
                    reporter.on_llm_error(turn, f"JSON parse error: {e}")
                return None

            if response.get("done"):
                result = response.get("result", "")
                tool_names_used = {h["name"] for h in history}

                if expected_files and "write_file" not in tool_names_used:
                    logger.warning(
                        "[EXECUTOR] Node %s set done=true but write_file not in history "
                        "— injecting correction turn",
                        node.id,
                    )
                    history = self._append_to_history(
                        history,
                        {
                            "name": "write_file",
                            "args": {"path": expected_files[0], "content": ""},
                            "result": (
                                f"ERROR: write_file was called with empty content. "
                                f"You must call write_file again with the full content of "
                                f"{expected_files[0]}. Use the actual report content you "
                                f"generated — do not set done=true until the file is written "
                                f"with real content."
                            ),
                        },
                    )
                    continue

                # ── Surface awaiting_user steps ───────────────────────────────
                # If some steps required user action, build handoff artifact and
                # return a sentinel so the orchestrator emits MARK_AWAITING_USER.
                if awaiting_steps:
                    handoff_lines = [
                        "The following steps require your action before this task is complete:",
                        "",
                    ]
                    for s in awaiting_steps:
                        handoff_lines.append(
                            f"• [{s.get('execution_type')}] {s.get('description', '')}"
                        )
                        handoff_lines.append(f"  Purpose: {s.get('produces', '')}")
                    if result:
                        handoff_lines += ["", "Prepared content for your use:", "", result]
                    handoff_artifact = "\n".join(handoff_lines)

                    logger.info(
                        "[EXECUTOR] Node %s: %d step(s) awaiting user — surfacing",
                        node.id,
                        len(awaiting_steps),
                    )
                    # Return a special dict; _on_node_done detects this and emits
                    # MARK_AWAITING_USER instead of MARK_DONE.
                    return {
                        "_awaiting_user": True,
                        "handoff_artifact": handoff_artifact,
                        "pending_steps": [s.get("execution_type") for s in awaiting_steps],
                        "partial_result": result,
                    }

                logger.info("[EXECUTOR] Node %s completed. Result: %.120s", node.id, result)
                return result

            tool_call = response.get("tool_call")
            if not tool_call:
                logger.warning(
                    "[EXECUTOR] Node %s: done=false but no tool_call on turn %d",
                    node.id,
                    turn + 1,
                )
                if reporter:
                    reporter.on_llm_error(turn, "done=false but no tool_call provided")
                return None

            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})

            if not self.tools or tool_name not in self.tools.tools:
                logger.warning("[EXECUTOR] Node %s requested unknown tool '%s'", node.id, tool_name)
                if reporter:
                    step_id = reporter.on_tool_start(tool_name, tool_args)
                    reporter.on_tool_done(
                        step_id,
                        tool_name,
                        tool_args,
                        f"ERROR: tool '{tool_name}' not found",
                        error=True,
                    )
                history = self._append_to_history(
                    history,
                    {
                        "name": tool_name,
                        "args": tool_args,
                        "result": (
                            f"ERROR: tool '{tool_name}' not found. "
                            "No tools are available — you must complete this task "
                            "using your own knowledge. Set done=true with a result "
                            "based on what you know; do not call any tool."
                        ),
                    },
                )
                tool_not_found_count += 1
                if tool_not_found_count >= 2:
                    logger.error(
                        "[EXECUTOR] Node %s: %d consecutive tool-not-found errors — aborting",
                        node.id,
                        tool_not_found_count,
                    )
                    return None
                continue

            logger.info("[EXECUTOR] Node %s calling tool '%s'", node.id, tool_name)
            tool_not_found_count = 0
            step_id = reporter.on_tool_start(tool_name, tool_args) if reporter else None

            tool_result_str, _ = self._dispatch_tool(
                node.id, tool_name, tool_args, reporter, step_id
            )
            history = self._append_to_history(
                history,
                {"name": tool_name, "args": tool_args, "result": tool_result_str},
            )

        logger.error(
            "[EXECUTOR] Node %s did not complete within %d turns",
            node.id,
            self.max_turns,
        )
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Native execution loop (provider tool-use API)
    # ──────────────────────────────────────────────────────────────────────────

    def _execute_native(
        self,
        node,
        resolved_inputs,
        effective_description: str,
        effective_outputs: list,
        effective_steps: list,
        reporter,
    ):
        def _output_name(o):
            return o["name"] if isinstance(o, dict) else str(o)

        def _is_file(o):
            if isinstance(o, dict):
                return o.get("type") == "file" or any(
                    _output_name(o).endswith(ext) for ext in self.FILE_EXTENSIONS
                )
            return any(str(o).endswith(ext) for ext in self.FILE_EXTENSIONS)

        expected_files = [_output_name(o) for o in effective_outputs if _is_file(o)]
        tools = list(self.tools.tools.values())
        history: list[dict] = []
        tool_not_found_count = 0

        # ── Step capability pre-check (same logic as legacy path) ────────────
        executable_types = self._llm_executable_types()
        awaiting_steps = [
            s
            for s in effective_steps
            if s.get("execution_type") and s.get("execution_type") not in executable_types
        ]
        llm_steps = [
            s
            for s in effective_steps
            if not s.get("execution_type") or s.get("execution_type") in executable_types
        ]

        if awaiting_steps:
            logger.info(
                "[EXECUTOR] Node %s (native): %d step(s) require user action: %s",
                node.id,
                len(awaiting_steps),
                [s.get("execution_type") for s in awaiting_steps],
            )

        # ── Early exit: all steps require user action — skip the LLM loop ────
        if awaiting_steps and not llm_steps:
            logger.info(
                "[EXECUTOR] Node %s (native): all %d step(s) require user action — "
                "surfacing immediately without LLM calls",
                node.id,
                len(awaiting_steps),
            )
            handoff_lines = [
                "The following steps require your action before this task is complete:",
                "",
            ]
            for s in awaiting_steps:
                handoff_lines.append(f"• [{s.get('execution_type')}] {s.get('description', '')}")
                handoff_lines.append(f"  Purpose: {s.get('produces', '')}")
            return {
                "_awaiting_user": True,
                "handoff_artifact": "\n".join(handoff_lines),
                "pending_steps": [s.get("execution_type") for s in awaiting_steps],
                "partial_result": "",
            }

        task_prompt = self._build_native_task_prompt(
            node=node,
            resolved_inputs=resolved_inputs,
            effective_description=effective_description,
            effective_outputs=effective_outputs,
            effective_steps=llm_steps if llm_steps else effective_steps,
        )

        for turn in range(self.max_turns):
            turns_remaining = self.max_turns - turn

            extra_reminder = ""
            if expected_files and "write_file" not in {h["name"] for h in history}:
                extra_reminder = build_executor_native_file_reminder(
                    expected_files, turns_remaining
                )

            current_prompt = task_prompt + "\n" + extra_reminder if extra_reminder else task_prompt

            if reporter:
                reporter.on_llm_turn(turn)

            try:
                response: NativeToolResponse = self.llm.ask_with_tools(
                    current_prompt, tools, history
                )
            except LLMStoppedError:
                logger.warning("[EXECUTOR] LLM stopped during native execution of %s", node.id)
                return None
            except Exception as e:
                logger.error("[EXECUTOR] LLM error during native execution of %s: %s", node.id, e)
                if reporter:
                    reporter.on_llm_error(turn, str(e))
                return None

            if response.kind == "text":
                result = response.text

                if expected_files and "write_file" not in {h["name"] for h in history}:
                    logger.warning(
                        "[EXECUTOR] Node %s gave final answer without calling write_file "
                        "— injecting correction turn",
                        node.id,
                    )
                    history = self._append_to_history(
                        history,
                        {
                            "kind": "correction",
                            "content": (
                                f"Your previous response was a text answer, but this task "
                                f"requires you to call write_file to produce "
                                f"{expected_files[0]}. "
                                f"Do not give a final text answer — call write_file with the "
                                f"actual file content now."
                            ),
                        },
                    )
                    continue

                # ── Surface awaiting_user steps ───────────────────────────────
                if awaiting_steps:
                    handoff_lines = [
                        "The following steps require your action before this task is complete:",
                        "",
                    ]
                    for s in awaiting_steps:
                        handoff_lines.append(
                            f"• [{s.get('execution_type')}] {s.get('description', '')}"
                        )
                        handoff_lines.append(f"  Purpose: {s.get('produces', '')}")
                    if result:
                        handoff_lines += ["", "Prepared content for your use:", "", result]
                    handoff_artifact = "\n".join(handoff_lines)
                    logger.info(
                        "[EXECUTOR] Node %s (native): %d step(s) awaiting user — surfacing",
                        node.id,
                        len(awaiting_steps),
                    )
                    return {
                        "_awaiting_user": True,
                        "handoff_artifact": handoff_artifact,
                        "pending_steps": [s.get("execution_type") for s in awaiting_steps],
                        "partial_result": result,
                    }

                logger.info(
                    "[EXECUTOR] Node %s completed (native). Result: %.120s",
                    node.id,
                    result,
                )
                return result

            tool_name = response.tool_name
            tool_args = response.tool_args
            tool_use_id = response.tool_use_id or f"toolu_{uuid.uuid4().hex[:12]}"

            if tool_name not in self.tools.tools:
                logger.warning("[EXECUTOR] Node %s requested unknown tool '%s'", node.id, tool_name)
                if reporter:
                    step_id = reporter.on_tool_start(tool_name, tool_args)
                    reporter.on_tool_done(
                        step_id,
                        tool_name,
                        tool_args,
                        f"ERROR: tool '{tool_name}' not found",
                        error=True,
                    )
                history = self._append_to_history(
                    history,
                    {
                        "name": tool_name,
                        "args": tool_args,
                        "result": (
                            f"ERROR: tool '{tool_name}' is not available. "
                            "Use only the tools listed in the system prompt."
                        ),
                        "tool_use_id": tool_use_id,
                    },
                )
                tool_not_found_count += 1
                if tool_not_found_count >= 2:
                    logger.error(
                        "[EXECUTOR] Node %s: %d consecutive tool-not-found errors — aborting",
                        node.id,
                        tool_not_found_count,
                    )
                    return None
                continue

            logger.info("[EXECUTOR] Node %s calling tool '%s' (native)", node.id, tool_name)
            tool_not_found_count = 0
            step_id = reporter.on_tool_start(tool_name, tool_args) if reporter else None

            tool_result_str, _ = self._dispatch_tool(
                node.id, tool_name, tool_args, reporter, step_id
            )
            history = self._append_to_history(
                history,
                {
                    "name": tool_name,
                    "args": tool_args,
                    "result": tool_result_str,
                    "tool_use_id": tool_use_id,
                },
            )

        logger.error(
            "[EXECUTOR] Node %s did not complete within %d turns (native)",
            node.id,
            self.max_turns,
        )
        return None

        for turn in range(self.max_turns):
            turns_remaining = self.max_turns - turn

            extra_reminder = ""
            if expected_files and "write_file" not in {h["name"] for h in history}:
                extra_reminder = build_executor_native_file_reminder(
                    expected_files, turns_remaining
                )

            current_prompt = task_prompt + "\n" + extra_reminder if extra_reminder else task_prompt

            if reporter:
                reporter.on_llm_turn(turn)

            try:
                response: NativeToolResponse = self.llm.ask_with_tools(
                    current_prompt, tools, history
                )
            except LLMStoppedError:
                logger.warning("[EXECUTOR] LLM stopped during native execution of %s", node.id)
                return None
            except Exception as e:
                logger.error("[EXECUTOR] LLM error during native execution of %s: %s", node.id, e)
                if reporter:
                    reporter.on_llm_error(turn, str(e))
                return None

            if response.kind == "text":
                result = response.text

                if expected_files and "write_file" not in {h["name"] for h in history}:
                    logger.warning(
                        "[EXECUTOR] Node %s gave final answer without calling write_file "
                        "— injecting correction turn",
                        node.id,
                    )
                    # Fix #10: the old code injected a fake tool_use history
                    # entry (pretending the model called write_file with an ID
                    # it never issued).  Both Anthropic and OpenAI validate
                    # tool_use_id consistency and reject such conversations with
                    # a 400 error.  Use a plain "correction" entry instead,
                    # which _build_native_messages_* renders as a user message —
                    # a safe, provider-agnostic way to redirect the model.
                    history = self._append_to_history(
                        history,
                        {
                            "kind": "correction",
                            "content": (
                                f"Your previous response was a text answer, but this task "
                                f"requires you to call write_file to produce "
                                f"{expected_files[0]}. "
                                f"Do not give a final text answer — call write_file with the "
                                f"actual file content now."
                            ),
                        },
                    )
                    continue

                logger.info(
                    "[EXECUTOR] Node %s completed (native). Result: %.120s",
                    node.id,
                    result,
                )
                return result

            tool_name = response.tool_name
            tool_args = response.tool_args
            tool_use_id = response.tool_use_id or f"toolu_{uuid.uuid4().hex[:12]}"

            if tool_name not in self.tools.tools:
                logger.warning("[EXECUTOR] Node %s requested unknown tool '%s'", node.id, tool_name)
                if reporter:
                    step_id = reporter.on_tool_start(tool_name, tool_args)
                    reporter.on_tool_done(
                        step_id,
                        tool_name,
                        tool_args,
                        f"ERROR: tool '{tool_name}' not found",
                        error=True,
                    )
                history = self._append_to_history(
                    history,
                    {
                        "name": tool_name,
                        "args": tool_args,
                        "result": (
                            f"ERROR: tool '{tool_name}' is not available. "
                            "Use only the tools listed in the system prompt."
                        ),
                        "tool_use_id": tool_use_id,
                    },
                )
                tool_not_found_count += 1
                if tool_not_found_count >= 2:
                    logger.error(
                        "[EXECUTOR] Node %s: %d consecutive tool-not-found errors — aborting",
                        node.id,
                        tool_not_found_count,
                    )
                    return None
                continue

            logger.info("[EXECUTOR] Node %s calling tool '%s' (native)", node.id, tool_name)
            tool_not_found_count = 0
            step_id = reporter.on_tool_start(tool_name, tool_args) if reporter else None

            tool_result_str, _ = self._dispatch_tool(
                node.id, tool_name, tool_args, reporter, step_id
            )
            history = self._append_to_history(
                history,
                {
                    "name": tool_name,
                    "args": tool_args,
                    "result": tool_result_str,
                    "tool_use_id": tool_use_id,
                },
            )

        logger.error(
            "[EXECUTOR] Node %s did not complete within %d turns (native)",
            node.id,
            self.max_turns,
        )
        return None

    def _build_native_task_prompt(
        self,
        node,
        resolved_inputs: list,
        effective_description: str,
        effective_outputs: list,
        effective_steps: list | None = None,
    ) -> str:
        def _fmt_output(o):
            if isinstance(o, dict):
                return f"  - [{o['type']}] {o['name']}: {o['description']}"
            return f"  - {o}"

        def _output_name(o):
            return o["name"] if isinstance(o, dict) else str(o)

        def _is_file(o):
            if isinstance(o, dict):
                return o.get("type") == "file" or any(
                    _output_name(o).endswith(ext) for ext in self.FILE_EXTENSIONS
                )
            return any(str(o).endswith(ext) for ext in self.FILE_EXTENSIONS)

        if resolved_inputs:
            inputs_text = "\n".join(
                f"  [{e['node_id']}]\n"
                f"    Description:      {e['description']}\n"
                f"    Declared outputs: {e['declared_output']}\n"
                f"    Actual result:    {e['result']}"
                for e in resolved_inputs
            )
        else:
            inputs_text = "  (none — this is a root task)"

        retry = node.metadata.get("retry_count", 0)
        if retry > 0:
            failure = node.metadata.get("verification_failure", "unknown")[:200]
            prev_result = node.result or "(none)"
            if len(prev_result) > 200:
                prev_result = prev_result[:200] + "…"
        else:
            failure = ""
            prev_result = ""
        retry_notice = build_executor_retry_notice(retry, failure, prev_result)

        outputs_text = (
            "\n".join(_fmt_output(o) for o in effective_outputs)
            if effective_outputs
            else "  (not specified)"
        )
        outputs_block = build_executor_outputs_block(outputs_text)

        expected_files = [_output_name(o) for o in effective_outputs if _is_file(o)]
        if expected_files:
            output_instruction = build_executor_native_file_output_instruction(expected_files)
        else:
            output_instruction = build_executor_native_inline_output_instruction(
                self.max_inline_result_chars
            )

        steps = (
            effective_steps
            if effective_steps is not None
            else node.metadata.get("execution_steps", [])
        )
        if steps:
            steps_lines = []
            for i, s in enumerate(steps, 1):
                steps_lines.append(
                    f"  {i}. [{s.get('execution_type', '?')}] {s.get('description', '')}"
                )
                if s.get("produces"):
                    steps_lines.append(f"     → {s['produces']}")
            steps_text = "\n".join(steps_lines)
        else:
            steps_text = "  (not specified — use your best judgement)"

        return build_executor_native_prompt(
            node_id=node.id,
            description=effective_description,
            retry_notice=retry_notice,
            extra_reminder="",
            outputs_block=outputs_block,
            output_instruction=output_instruction,
            inputs_text=inputs_text,
            steps_text=steps_text,
            turns_remaining=self.max_turns,
        )
