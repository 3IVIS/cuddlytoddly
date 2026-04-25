# planning/llm_executor.py

import json
import uuid
from pathlib import Path

from cuddlytoddly.engine.signals import AwaitingInputSignal
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
)
from toddly.core.events import UPDATE_METADATA, Event
from toddly.infra.logging import get_logger
from toddly.planning.llm_interface import LLMStoppedError, NativeToolResponse
from toddly.planning.schemas import (
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
      4. Repeat until done=True or a turn budget is exhausted

    Two independent turn budgets control the loop:
      max_successful_turns   — turns where a tool call returned without error.
      max_unsuccessful_turns — turns where a tool call errored, no tool call
                               was made, or a correction was injected.
    The final turn of the combined budget is always reserved for the model
    to synthesise and return its result without calling any more tools.

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
        max_successful_turns: int = 10,
        max_unsuccessful_turns: int = 10,
        max_turns: int | None = None,  # deprecated: sets both budgets to the same value
        max_inline_result_chars: int = 3000,
        max_total_input_chars: int = 3000,
        max_tool_result_chars: int = 2000,
        max_history_entries: int = 3,
        working_dir: Path | str | None = None,
        tool_call_log=None,
        quality_gate=None,
    ):
        self.llm = llm_client
        self.tools = tool_registry
        if max_turns is not None:
            # Deprecated single-budget path: treat both limits identically.
            self.max_successful_turns = max_turns
            self.max_unsuccessful_turns = max_turns
        else:
            self.max_successful_turns = max_successful_turns
            self.max_unsuccessful_turns = max_unsuccessful_turns
        self.max_inline_result_chars = max_inline_result_chars
        self.max_total_input_chars = max_total_input_chars
        self.max_tool_result_chars = max_tool_result_chars
        self.max_history_entries = max_history_entries
        # store working_dir as explicit state; forwarded to tools via _cwd arg key
        self.working_dir: Path | None = Path(working_dir) if working_dir else None
        # optional structured tool-call log (ToolCallLog or NullToolCallLog)
        self.tool_call_log = tool_call_log
        # optional quality gate for inline result verification (Issue 6)
        self.quality_gate = quality_gate

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

        known_fields   : list of {key, label, value, hint?} where value is not a placeholder
        unknown_fields : list of {key, label, hint?} where value was "unknown" or similar

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
            hint = f.get("hint", "").strip()
            if val in _PLACEHOLDERS:
                entry = {"key": f.get("key", ""), "label": f.get("label", f.get("key", ""))}
                if hint:
                    entry["hint"] = hint
                unknown.append(entry)
            else:
                entry = {
                    "key": f.get("key", ""),
                    "label": f.get("label", f.get("key", "")),
                    "value": f.get("value", ""),
                }
                if hint:
                    entry["hint"] = hint
                known.append(entry)
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

        def _render_clarification_entry(dep_id, dep):
            """Build the resolved-input dict for a clarification node."""
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
                    hint = f.get("hint", "")
                    hint_suffix = f" ({hint})" if hint else ""
                    lines.append(f"    {f['label']}{hint_suffix}: not provided")
            return {
                "node_id": dep_id,
                "description": dep.metadata.get("description", dep_id),
                "declared_output": [],
                "result": "\n".join(lines),
                "_unknown_fields": unknown,
                "_known_fields": known,
            }

        resolved = []
        included_ids: set[str] = set()

        for dep_id in node.dependencies:
            included_ids.add(dep_id)
            dep = snapshot.get(dep_id)
            if not dep or not dep.result:
                continue

            # ── Clarification node — render as structured known/unknown blocks ──
            if dep.node_type == "clarification":
                resolved.append(_render_clarification_entry(dep_id, dep))
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

        # ── FIX: BFS over full ancestor chain for transitive clarification nodes ──
        # A clarification node may be 2+ hops away (e.g. attached to the goal,
        # which feeds Identify_Search_Terms, which feeds Filter_Repos_By_Review_Type).
        # The direct-dep loop above only captures clarification nodes that are
        # immediate parents; this BFS ensures every ancestor clarification node
        # contributes its known/unknown fields to all downstream tasks.
        queue = list(node.dependencies)
        visited = set(queue)
        ancestor_clar_entries = []

        while queue:
            dep_id = queue.pop(0)
            dep = snapshot.get(dep_id)
            if dep is None:
                continue
            if dep.node_type == "clarification" and dep.result and dep_id not in included_ids:
                ancestor_clar_entries.append(_render_clarification_entry(dep_id, dep))
                included_ids.add(dep_id)
            # Keep walking regardless — there may be a clarification node further up.
            for ancestor_id in getattr(dep, "dependencies", []):
                if ancestor_id not in visited:
                    visited.add(ancestor_id)
                    queue.append(ancestor_id)

        if ancestor_clar_entries:
            logger.debug(
                "[EXECUTOR] Node %s: found %d ancestor clarification node(s) via BFS: %s",
                node.id,
                len(ancestor_clar_entries),
                [e["node_id"] for e in ancestor_clar_entries],
            )
            # Prepend so clarification context precedes task results in the prompt
            resolved = ancestor_clar_entries + resolved

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

        # FIX: append the manual-retry nonce (if set) to extra_reminder so it
        # becomes part of the prompt string and therefore the LLM cache key.
        # The nonce is invisible to the model's reasoning but ensures a manual
        # retry never collides with a prior cached response for the same prompt.
        nonce = node.metadata.get("_retry_nonce", 0)
        if nonce:
            nonce_suffix = f"\n# retry_nonce={nonce}"
            extra_reminder = (extra_reminder or "") + nonce_suffix

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

    def _preflight_awaiting_input(
        self, node, resolved_inputs, snapshot=None
    ) -> "AwaitingInputSignal | None":
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

        # Fix 6: collect every output name declared by ANY task node in the DAG.
        # A required_input whose name matches a DAG-task output is a data-flow
        # dependency — it should NEVER be added as a clarification field
        # (which the user would be asked to fill in).  It belongs to upstream
        # task execution; the node must stay pending until that upstream result
        # is actually available.
        dag_output_names: set[str] = set()
        if snapshot:
            for snap_node in snapshot.values():
                if snap_node.node_type == "task":
                    for o in snap_node.metadata.get("output", []):
                        if isinstance(o, dict) and o.get("name"):
                            dag_output_names.add(o["name"])

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
            # Fix 6: never promote a data-flow output name to a clarification
            # field — that would ask the user to supply something that should
            # come from an upstream task result.
            and r.get("name") not in dag_output_names
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
            from toddly.planning.llm_interface import LLMStoppedError

            raise LLMStoppedError("LLM is paused — deferring broadening call until resumed")

        def _fmt_fields(fields):
            lines = []
            for f in fields:
                hint = f.get("hint", "")
                hint_suffix = f" ({hint})" if hint else ""
                lines.append(
                    f"  - {f.get('key', '?')} ({f.get('label', f.get('key', '?'))})"
                    f"{hint_suffix}: {f.get('value', 'unknown')}"
                )
            return "\n".join(lines) if lines else "  (none)"

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

        Returns (effective_description, effective_outputs, effective_steps, mode)
        where *mode* is ``'original'`` or ``'broadened'``.

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
            return original_desc, original_outputs, original_steps, "original"

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
            return original_desc, original_outputs, original_steps, "original"

        # Use the broadened plan
        broadened_desc = signal.broadened_description or original_desc
        broadened_outputs = signal.broadened_output if signal.broadened_output else original_outputs
        broadened_steps = signal.broadened_steps if signal.broadened_steps else original_steps
        return broadened_desc, broadened_outputs, broadened_steps, "broadened"

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

        # Fix 6: pass snapshot so _preflight_awaiting_input can distinguish
        # data-flow outputs (produced by task nodes) from genuine user-context
        # fields, preventing task outputs from leaking into the clarification form.
        signal = self._preflight_awaiting_input(node, resolved_inputs, snapshot)

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

        effective_description, effective_outputs, effective_steps, _exec_mode = (
            self._select_goal_mode(node, resolved_inputs, signal)
        )
        # Tell the reporter (and via it the UI) which tab is actually running.
        if reporter is not None:
            reporter.on_execution_mode(_exec_mode)

        # ── FIX: Explicit hallucination guard for broadened execution ──────────
        # When running with a broadened description (some inputs are missing),
        # the model can still fabricate values for the unknown fields even when
        # told to stay general.  We append a hard, named-field prohibition
        # directly to effective_description so it is present in every prompt
        # variant (legacy and native) and is harder to ignore than a reminder
        # buried elsewhere in the prompt.
        if signal is not None and signal.broadened_for_missing:
            missing_names = ", ".join(sorted(signal.broadened_for_missing))
            effective_description = (
                effective_description
                + f"\n\nCRITICAL CONSTRAINT: The following inputs are unavailable: "
                f"{missing_names}. "
                "Do NOT invent, assume, or assign any specific value for these fields. "
                "Any result that contains a fabricated value for an unknown field will be "
                "rejected. Produce a general answer or template that is useful without them."
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
                snapshot,
                resolved_inputs,
                effective_description,
                effective_outputs,
                effective_steps,
                reporter,
            )
        else:
            return self._execute_legacy(
                node,
                snapshot,
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
        import time as _time

        error = False
        t0 = _time.time()
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

        # Write to the structured tool-call log BEFORE truncation so the
        # record always contains the complete tool output.  _cwd is stripped
        # from args since it's an internal routing detail, not part of the
        # logical call.
        if self.tool_call_log is not None:
            clean_args = {k: v for k, v in tool_args.items() if k != "_cwd"}
            self.tool_call_log.record(
                node_id=node_id,
                tool_name=tool_name,
                args=clean_args,
                result=tool_result_str,
                duration_ms=(_time.time() - t0) * 1000,
                error=error,
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
        """
        Append *entry* to *history*, trimming to ``max_history_entries``.

        When trimming is needed, a compact summary entry is prepended to the
        retained window so the model retains awareness of earlier tool calls
        and their outcomes.  Without this, the model has no memory of what
        happened in turns that fell outside the window — leading to repeated
        near-duplicate queries and wasted turns (context amnesia).

        The summary occupies one history slot (``n_keep = max_history_entries - 1``
        real entries + 1 summary), so the total list length never exceeds
        ``max_history_entries``.

        Entry format for the summary uses ``kind="correction"`` so both
        execution paths handle it correctly without changes:
          - Native path (ApiLLM._build_native_messages_*): the existing
            ``kind="correction"`` branch renders it as a plain user message.
          - Legacy path (history_text rendering loop): falls back to the
            ``name`` / ``args`` / ``result`` fields which are always set.
        """
        history.append(entry)
        if len(history) <= self.max_history_entries:
            return history

        # Reserve one slot for the summary; keep the most recent real entries.
        n_keep = self.max_history_entries - 1
        to_drop = history[: len(history) - n_keep]
        retained = history[-n_keep:]

        # Build a one-line summary per dropped entry so the model knows what
        # happened without needing to see the full result text.
        summary_lines = []
        for e in to_drop:
            kind = e.get("kind")
            name = e.get("name", "")
            result = e.get("result", "")
            ok = not result.startswith("ERROR:") and "no results" not in result.lower()

            if kind == "correction":
                # Correction messages are injected by the executor itself;
                # the model doesn't need the full text, just that one occurred.
                summary_lines.append("[correction injected]")
            elif name == "web_search":
                q = e.get("args", {}).get("query", "?")
                status = "OK" if ok else "FAILED"
                summary_lines.append(f"[web_search query={q!r} → {status}]")
            elif name == "fetch_url":
                url = e.get("args", {}).get("url", "?")
                status = "OK" if ok else "FAILED"
                summary_lines.append(f"[fetch_url url={url!r} → {status}]")
            elif name:
                status = "OK" if ok else "FAILED"
                summary_lines.append(f"[{name} → {status}]")
            # Entries with neither kind nor name (e.g. malformed) are skipped.

        if not summary_lines:
            return retained

        summary_text = "Context from earlier turns (trimmed from window): " + "; ".join(
            summary_lines
        )
        summary_entry = {
            # kind="correction" → native path renders this as a plain user message.
            "kind": "correction",
            "content": summary_text,
            # Legacy path fallback — the history_text loop uses these fields.
            "name": "[history-summary]",
            "args": {},
            "result": summary_text,
        }
        return [summary_entry] + retained

    # ──────────────────────────────────────────────────────────────────────────
    # Legacy execution loop (JSON-in-prompt, done+tool_call protocol)
    # ──────────────────────────────────────────────────────────────────────────

    def _execute_legacy(
        self,
        node,
        snapshot,
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
        # ── Fix 7: seed failed_queries from persisted metadata so previously
        # failed queries survive session restarts (legacy path accumulator).
        # tried_queries is a superset: all queries attempted, regardless of
        # outcome.  Both are maintained independently of history trimming.
        failed_queries: set[str] = set(node.metadata.get("_failed_queries", []))
        tried_queries: set[str] = set(failed_queries)
        # ── Issue 2: track fetched URLs to prevent re-fetching the same page ──
        fetched_urls: set[str] = set()

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

        # ── FIX 3A: Per-turn live-search directive ────────────────────────────
        # On the legacy path the model decides whether to call a tool by reading
        # the prompt.  For steps typed search_web / fetch_url it can answer from
        # training data instead of calling the tool; this reminder is appended to
        # extra_reminder every turn so it is never silently dropped.
        _LIVE_SEARCH_TYPES = {"search_web", "fetch_url"}
        _needs_live_search = any(s.get("execution_type") in _LIVE_SEARCH_TYPES for s in llm_steps)
        _registered_tool_names = set(self.tools.tools.keys()) if self.tools else set()
        _live_tool_available = bool(_registered_tool_names & {"web_search", "fetch_url"})
        _live_search_reminder = ""
        if _needs_live_search and _live_tool_available:
            _live_search_reminder = (
                "\nIMPORTANT: One or more steps require a live web search. "
                "You MUST call the web_search (or fetch_url) tool before setting "
                "done=true. Do NOT answer from training data or prior knowledge — "
                "only use information retrieved from actual tool calls this session."
            )

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

        successful_turn_count = 0
        unsuccessful_turn_count = 0

        for turn in range(self.max_successful_turns + self.max_unsuccessful_turns):
            remaining_successful = max(0, self.max_successful_turns - successful_turn_count)
            remaining_unsuccessful = max(0, self.max_unsuccessful_turns - unsuccessful_turn_count)
            turns_left = remaining_successful + remaining_unsuccessful

            # Stop as soon as either individual budget is exhausted.
            if remaining_successful <= 0 or remaining_unsuccessful <= 0:
                break

            # Final turn when either budget is one away from exhaustion.
            is_final_turn = remaining_successful == 1 or remaining_unsuccessful == 1
            turns_remaining = turns_left
            extra_reminder = ""

            if expected_files and "write_file" not in {h["name"] for h in history}:
                extra_reminder += build_executor_file_reminder(expected_files, turns_remaining)

            # ── Per-turn search guidance (two-tier) ──────────────────────────
            # Mirrors the native path.  Accumulated sets survive history
            # trimming; injection happens every turn so the model always has
            # the full picture regardless of which history entries remain.
            if _live_search_reminder:
                if not any(h.get("name") in _registered_tool_names for h in history):
                    extra_reminder += _live_search_reminder

                # Tier 1 — failed queries: do not repeat.
                if failed_queries:
                    failed_str = ", ".join(f'"{q}"' for q in sorted(failed_queries))
                    extra_reminder += (
                        f"\nWARNING: The following web_search queries returned no "
                        f"results or an error: {failed_str}. "
                        "Do NOT repeat these queries. "
                        "Use completely different keywords or synonyms."
                    )

                # Tier 2 — all tried queries: avoid near-duplicates.
                non_failed = tried_queries - failed_queries
                if non_failed:
                    tried_str = ", ".join(f'"{q}"' for q in sorted(non_failed))
                    extra_reminder += (
                        f"\nNOTE: You have already searched for: {tried_str}. "
                        "Avoid repeating the same or very similar queries. "
                        "If you need more information, try a meaningfully different angle."
                    )

                # Fetch-url nudge — prefer reading a page over searching again.
                fetch_called = any(h.get("name") == "fetch_url" for h in history)
                if not fetch_called:
                    urls_available = any(
                        h.get("name") == "web_search"
                        and not h.get("result", "").startswith("ERROR:")
                        and "URL:" in h.get("result", "")
                        for h in history
                    )
                    if urls_available:
                        extra_reminder += (
                            "\nHINT: Your search results contain URLs. "
                            "Rather than searching again, consider calling "
                            "fetch_url on a promising URL to get the full "
                            "page content."
                        )

                # Issue 2: warn the model away from already-fetched URLs.
                if fetched_urls:
                    fetched_str = ", ".join(f'"{u}"' for u in sorted(fetched_urls))
                    extra_reminder += (
                        f"\nNOTE: You have already fetched these URLs this session: "
                        f"{fetched_str}. Do NOT fetch them again — they will return "
                        "identical content. Choose a different URL."
                    )

            if is_final_turn:
                extra_reminder += (
                    "\nFINAL TURN: You must not call any more tools. "
                    "Set done=true now and return your best result based "
                    "on everything you have gathered so far."
                )

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

                # ── Fix 1: reject done=true when every search tool call failed ──
                # The model satisfies the "must call web_search before finishing"
                # prompt guard by calling the tool, but then fabricates results
                # when every call errors.  Catch this here — before the result
                # ever reaches the verifier — and force a fresh attempt.
                if _needs_live_search and _live_tool_available:
                    successful_searches = [
                        h
                        for h in history
                        if h.get("name") in ("web_search", "fetch_url")
                        and not h.get("result", "").startswith("ERROR:")
                        and "no results" not in h.get("result", "").lower()
                    ]
                    if not successful_searches:
                        logger.warning(
                            "[EXECUTOR] Node %s set done=true but all search tool "
                            "calls failed — injecting correction turn",
                            node.id,
                        )
                        history = self._append_to_history(
                            history,
                            {
                                "name": "web_search",
                                "args": {},
                                "result": (
                                    "ERROR: You cannot set done=true because every "
                                    "web_search call returned an error or no results. "
                                    "You MUST retrieve real data from the web before "
                                    "completing this task. Try a completely different "
                                    "search query — use different keywords, broaden or "
                                    "narrow the terms, or break the search into smaller "
                                    "parts. Do NOT fabricate results."
                                ),
                            },
                        )
                        unsuccessful_turn_count += 1
                        continue

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
                    unsuccessful_turn_count += 1
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

                # ── Issue 6 / FIX E: inline quality gate verification ─────────
                # Verify the result within the executor's own turn budget.
                # On failure we inject a correction turn so the model can
                # revise without a full orchestrator retry cycle (backoff etc).
                # On success we tag the return value with _pre_verified=True so
                # the orchestrator skips its own redundant verification call for
                # this result — preventing the double-LLM-call problem (FIX E).
                # The tag is a wrapper dict; _on_node_done unwraps it before
                # any downstream processing.
                _pre_verified = False
                if self.quality_gate and not is_final_turn:
                    _ok, _reason = self.quality_gate.verify_result(node, result, snapshot)
                    if not _ok:
                        logger.warning(
                            "[EXECUTOR] Node %s inline verification failed: %s "
                            "— injecting correction turn (legacy)",
                            node.id,
                            _reason,
                        )
                        history = self._append_to_history(
                            history,
                            {
                                "name": "web_search",
                                "args": {},
                                "result": (
                                    f"ERROR: Your result was rejected by the quality gate: "
                                    f"{_reason}. You must revise it. Base your answer only "
                                    "on information retrieved from tool calls this session."
                                ),
                            },
                        )
                        unsuccessful_turn_count += 1
                        continue
                    # Inline verification passed — signal the orchestrator so it
                    # does not run a second redundant verification call.
                    _pre_verified = True

                logger.info("[EXECUTOR] Node %s completed. Result: %.120s", node.id, result)
                # ── Fix 7: persist failed queries before returning ────────────
                if failed_queries and reporter:
                    reporter._apply(
                        Event(
                            UPDATE_METADATA,
                            {
                                "node_id": node.id,
                                "metadata": {"_failed_queries": sorted(failed_queries)},
                            },
                        )
                    )
                # FIX E: wrap result with pre-verification flag when the inline
                # quality gate ran and passed so _on_node_done skips its own call.
                if _pre_verified:
                    return {"result": result, "_pre_verified": True}
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
                unsuccessful_turn_count += 1
                tool_not_found_count += 1
                if tool_not_found_count >= 2:
                    logger.error(
                        "[EXECUTOR] Node %s: %d consecutive tool-not-found errors — aborting",
                        node.id,
                        tool_not_found_count,
                    )
                    return None
                continue

            if is_final_turn:
                # Model ignored the FINAL TURN instruction and called a tool.
                # Skip execution — there are no remaining turns to process a result.
                logger.warning(
                    "[EXECUTOR] Node %s ignored FINAL TURN instruction (tool=%s) — aborting",
                    node.id,
                    tool_name,
                )
                break

            # ── Issue 2: URL deduplication for fetch_url ──────────────────────
            # If the model requests a URL it already fetched this session, inject
            # a correction instead of making a redundant network call.
            if tool_name == "fetch_url":
                url = tool_args.get("url", "").strip()
                if url and url in fetched_urls:
                    logger.warning(
                        "[EXECUTOR] Node %s: fetch_url skipped — URL already fetched: %s",
                        node.id,
                        url,
                    )
                    history = self._append_to_history(
                        history,
                        {
                            "name": tool_name,
                            "args": tool_args,
                            "result": (
                                f"ERROR: URL already fetched this session: {url!r}. "
                                "Fetching it again will return identical content. "
                                "Pick a different URL from your search results, or "
                                "run a new web_search with different keywords."
                            ),
                        },
                    )
                    unsuccessful_turn_count += 1
                    continue

            logger.info("[EXECUTOR] Node %s calling tool '%s'", node.id, tool_name)
            tool_not_found_count = 0
            step_id = reporter.on_tool_start(tool_name, tool_args) if reporter else None

            tool_result_str, tool_call_error = self._dispatch_tool(
                node.id, tool_name, tool_args, reporter, step_id
            )
            # ── Track queries in both tiers (mirrors native path) ────────────
            if tool_name == "web_search":
                q = tool_args.get("query", "").strip()
                if q:
                    tried_queries.add(q)
                    if (
                        tool_result_str.startswith("ERROR:")
                        or "no results" in tool_result_str.lower()
                    ):
                        failed_queries.add(q)
            # ── Issue 2: record successfully fetched URLs ─────────────────────
            elif tool_name == "fetch_url" and not tool_call_error:
                url = tool_args.get("url", "").strip()
                if url:
                    fetched_urls.add(url)

            if tool_call_error or tool_result_str.startswith("ERROR:"):
                unsuccessful_turn_count += 1
            else:
                successful_turn_count += 1

            history = self._append_to_history(
                history,
                {"name": tool_name, "args": tool_args, "result": tool_result_str},
            )

        logger.error(
            "[EXECUTOR] Node %s did not complete: %d/%d successful turns, %d/%d unsuccessful turns used",
            node.id,
            successful_turn_count,
            self.max_successful_turns,
            unsuccessful_turn_count,
            self.max_unsuccessful_turns,
        )
        # ── Fix 7: persist failed queries even on turn exhaustion ────────────
        if failed_queries and reporter:
            reporter._apply(
                Event(
                    UPDATE_METADATA,
                    {
                        "node_id": node.id,
                        "metadata": {"_failed_queries": sorted(failed_queries)},
                    },
                )
            )
        return None

    # ──────────────────────────────────────────────────────────────────────────
    # Native execution loop (provider tool-use API)
    # ──────────────────────────────────────────────────────────────────────────

    def _execute_native(
        self,
        node,
        snapshot,
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
        # ── Issue 2: track fetched URLs to prevent re-fetching the same page ──
        fetched_urls: set[str] = set()

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

        # ── Live-search reminder (mirrors legacy path) ────────────────────────
        # Build once outside the loop; injected every turn until a tool fires.
        _LIVE_SEARCH_TYPES = {"search_web", "fetch_url"}
        _needs_live_search = any(s.get("execution_type") in _LIVE_SEARCH_TYPES for s in llm_steps)
        _registered_tool_names = set(self.tools.tools.keys()) if self.tools else set()
        _live_tool_available = bool(_registered_tool_names & {"web_search", "fetch_url"})
        _live_search_reminder = ""
        if _needs_live_search and _live_tool_available:
            _live_search_reminder = (
                "\nIMPORTANT: One or more steps require a live web search. "
                "You MUST call the web_search (or fetch_url) tool before finishing. "
                "Do NOT answer from training data or prior knowledge — "
                "only use information retrieved from actual tool calls this session."
            )

        # ── FIX: tried_queries accumulator ────────────────────────────────────
        # Two tiers of tracking, both maintained independently of history
        # trimming (max_history_entries=3 on llamacpp means early entries are
        # evicted before the run ends):
        #
        #   failed_queries  — queries that returned ERROR or no results.
        #                     Seeded from persisted metadata (_failed_queries)
        #                     so they survive session restarts (Fix 7).
        #   tried_queries   — ALL queries attempted this session, regardless of
        #                     outcome.  Prevents near-duplicate queries (e.g.
        #                     "python code review tools github" vs "python code
        #                     review tools on github") that waste turns even
        #                     when the first attempt technically succeeded.
        failed_queries: set[str] = set(node.metadata.get("_failed_queries", []))
        tried_queries: set[str] = set(failed_queries)  # superset; grows each turn

        successful_turn_count = 0
        unsuccessful_turn_count = 0

        for turn in range(self.max_successful_turns + self.max_unsuccessful_turns):
            remaining_successful = max(0, self.max_successful_turns - successful_turn_count)
            remaining_unsuccessful = max(0, self.max_unsuccessful_turns - unsuccessful_turn_count)
            turns_left = remaining_successful + remaining_unsuccessful

            # Stop as soon as either individual budget is exhausted.
            if remaining_successful <= 0 or remaining_unsuccessful <= 0:
                break

            # Final turn when either budget is one away from exhaustion.
            is_final_turn = remaining_successful == 1 or remaining_unsuccessful == 1
            turns_remaining = turns_left

            extra_reminder = ""
            if expected_files and "write_file" not in {h["name"] for h in history}:
                extra_reminder = build_executor_native_file_reminder(
                    expected_files, turns_remaining
                )

            # ── FIX: per-turn search guidance ──────────────────────────────
            # Injected every turn so the guidance survives history trimming.
            if _live_search_reminder:
                extra_reminder += _live_search_reminder

                # Tier 1 — failed queries: explicitly warn the model not to
                # repeat these (they returned errors or no results).
                if failed_queries:
                    failed_str = ", ".join(f'"{q}"' for q in sorted(failed_queries))
                    extra_reminder += (
                        f"\nWARNING: The following web_search queries returned no "
                        f"results or an error: {failed_str}. "
                        "Do NOT repeat these queries. "
                        "Use completely different keywords or synonyms."
                    )

                # Tier 2 — all tried queries: nudge the model away from
                # near-duplicates of queries that technically returned results
                # but were semantically equivalent to an earlier call.
                non_failed = tried_queries - failed_queries
                if non_failed:
                    tried_str = ", ".join(f'"{q}"' for q in sorted(non_failed))
                    extra_reminder += (
                        f"\nNOTE: You have already searched for: {tried_str}. "
                        "Avoid repeating the same or very similar queries. "
                        "If you need more information, try a meaningfully different angle."
                    )

                # Fetch-url nudge — if at least one search succeeded and
                # produced URLs but fetch_url has not been called yet, remind
                # the model to read a page rather than doing another search.
                fetch_called = any(h.get("name") == "fetch_url" for h in history)
                if not fetch_called:
                    urls_available = any(
                        h.get("name") == "web_search"
                        and not h.get("result", "").startswith("ERROR:")
                        and "URL:" in h.get("result", "")
                        for h in history
                    )
                    if urls_available:
                        extra_reminder += (
                            "\nHINT: Your search results contain URLs. "
                            "Rather than searching again, consider calling "
                            "fetch_url on a promising URL to get the full "
                            "page content."
                        )

                # Issue 2: warn the model away from already-fetched URLs.
                if fetched_urls:
                    fetched_str = ", ".join(f'"{u}"' for u in sorted(fetched_urls))
                    extra_reminder += (
                        f"\nNOTE: You have already fetched these URLs this session: "
                        f"{fetched_str}. Do NOT fetch them again — they will return "
                        "identical content. Choose a different URL."
                    )

            if is_final_turn:
                extra_reminder += (
                    "\nFINAL TURN: You must not call any more tools. "
                    "Return your best result now based on everything you have gathered so far."
                )

            current_prompt = task_prompt + "\n" + extra_reminder if extra_reminder else task_prompt

            if reporter:
                reporter.on_llm_turn(turn)

            # On the final turn pass an empty tools list so the provider API
            # physically cannot emit a tool call — the model must respond with text.
            tools_for_turn = [] if is_final_turn else tools

            try:
                response: NativeToolResponse = self.llm.ask_with_tools(
                    current_prompt, tools_for_turn, history
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
                    unsuccessful_turn_count += 1
                    continue

                # ── Fix 1: reject final answer when all search calls failed ───
                if _needs_live_search and _live_tool_available:
                    successful_searches = [
                        h
                        for h in history
                        if h.get("name") in ("web_search", "fetch_url")
                        and not h.get("result", "").startswith("ERROR:")
                        and "no results" not in h.get("result", "").lower()
                    ]
                    if not successful_searches:
                        logger.warning(
                            "[EXECUTOR] Node %s gave final answer but all search tool "
                            "calls failed — injecting correction turn (native)",
                            node.id,
                        )
                        history = self._append_to_history(
                            history,
                            {
                                "kind": "correction",
                                "content": (
                                    "Your answer cannot be accepted because every "
                                    "web_search call returned an error or no results. "
                                    "You MUST successfully retrieve real data from the "
                                    "web before finishing. Try a completely different "
                                    "search query — use different keywords, broaden or "
                                    "narrow the terms, or break the search into smaller "
                                    "parts. Do NOT fabricate results."
                                ),
                            },
                        )
                        unsuccessful_turn_count += 1
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

                # ── Issue 6 / FIX E: inline quality gate verification ─────────
                # Same logic as the legacy path above: inject a correction on
                # failure, or tag a successful result with _pre_verified=True so
                # _on_node_done skips the redundant orchestrator-level call.
                _pre_verified = False
                if self.quality_gate and not is_final_turn:
                    _ok, _reason = self.quality_gate.verify_result(node, result, snapshot)
                    if not _ok:
                        logger.warning(
                            "[EXECUTOR] Node %s inline verification failed: %s "
                            "— injecting correction turn (native)",
                            node.id,
                            _reason,
                        )
                        history = self._append_to_history(
                            history,
                            {
                                "kind": "correction",
                                "content": (
                                    f"Your result was rejected by the quality gate: "
                                    f"{_reason}. You must revise it. Base your answer "
                                    "only on information retrieved from tool calls this session."
                                ),
                            },
                        )
                        unsuccessful_turn_count += 1
                        continue
                    # Inline verification passed.
                    _pre_verified = True

                logger.info(
                    "[EXECUTOR] Node %s completed (native). Result: %.120s",
                    node.id,
                    result,
                )
                # ── Fix 7: persist failed queries before returning ────────────
                if failed_queries and reporter:
                    reporter._apply(
                        Event(
                            UPDATE_METADATA,
                            {
                                "node_id": node.id,
                                "metadata": {"_failed_queries": sorted(failed_queries)},
                            },
                        )
                    )
                # FIX E: wrap with pre-verification tag when inline check passed.
                if _pre_verified:
                    return {"result": result, "_pre_verified": True}
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
                unsuccessful_turn_count += 1
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

            # ── Issue 2: URL deduplication for fetch_url ──────────────────────
            # If the model requests a URL it already fetched this session, inject
            # a correction instead of making a redundant network call.
            if tool_name == "fetch_url":
                url = tool_args.get("url", "").strip()
                if url and url in fetched_urls:
                    logger.warning(
                        "[EXECUTOR] Node %s: fetch_url skipped — URL already fetched: %s",
                        node.id,
                        url,
                    )
                    history = self._append_to_history(
                        history,
                        {
                            "kind": "correction",
                            "content": (
                                f"URL already fetched this session: {url!r}. "
                                "Fetching it again will return identical content. "
                                "Pick a different URL from your search results, or "
                                "run a new web_search with different keywords."
                            ),
                            "name": "fetch_url",
                            "args": tool_args,
                            "result": f"ERROR: duplicate fetch skipped for {url!r}",
                            "tool_use_id": tool_use_id,
                        },
                    )
                    unsuccessful_turn_count += 1
                    continue

            step_id = reporter.on_tool_start(tool_name, tool_args) if reporter else None

            tool_result_str, tool_error = self._dispatch_tool(
                node.id, tool_name, tool_args, reporter, step_id
            )

            # ── FIX: track queries in both tiers ─────────────────────────────
            # tried_queries: every web_search query regardless of outcome.
            # failed_queries: only queries that errored or returned no results.
            # Both survive history trimming; failed_queries is persisted so it
            # also survives session restarts (Fix 7).
            if tool_name == "web_search":
                q = tool_args.get("query", "").strip()
                if q:
                    tried_queries.add(q)
                    if tool_error or "no results" in tool_result_str.lower():
                        failed_queries.add(q)
            # ── Issue 2: record successfully fetched URLs ─────────────────────
            elif tool_name == "fetch_url" and not tool_error:
                url = tool_args.get("url", "").strip()
                if url:
                    fetched_urls.add(url)

            if tool_error or tool_result_str.startswith("ERROR:"):
                unsuccessful_turn_count += 1
            else:
                successful_turn_count += 1

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
            "[EXECUTOR] Node %s did not complete (native): %d/%d successful turns, %d/%d unsuccessful turns used",
            node.id,
            successful_turn_count,
            self.max_successful_turns,
            unsuccessful_turn_count,
            self.max_unsuccessful_turns,
        )
        # ── Fix 7: persist failed queries even on turn exhaustion ────────────
        if failed_queries and reporter:
            reporter._apply(
                Event(
                    UPDATE_METADATA,
                    {
                        "node_id": node.id,
                        "metadata": {"_failed_queries": sorted(failed_queries)},
                    },
                )
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
            turns_remaining=self.max_successful_turns + self.max_unsuccessful_turns,
        )
