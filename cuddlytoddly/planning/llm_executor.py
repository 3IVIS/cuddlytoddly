# planning/llm_executor.py

import json
from dataclasses import dataclass
from dataclasses import field as _dc_field

from cuddlytoddly.infra.logging import get_logger
from cuddlytoddly.planning.llm_interface import LLMStoppedError
from cuddlytoddly.planning.prompts import (
    build_awaiting_input_check_prompt,
    build_broadened_description_prompt,
    build_executor_file_output_instruction,
    build_executor_file_reminder,
    build_executor_inline_output_instruction,
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


# Version tag bumped when prompt semantics change significantly.
# Kept as a code constant (not config) because it tracks internal compatibility.
PROMPT_VERSION = "v3"


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
    """

    # File extensions that trigger "write_file" enforcement
    FILE_EXTENSIONS = frozenset({
        ".md", ".txt", ".py", ".json", ".csv", ".html",
        ".yaml", ".yml", ".xml", ".pdf", ".log",
    })

    def __init__(
        self,
        llm_client,
        tool_registry=None,
        max_turns: int = 5,
        max_inline_result_chars: int = 3000,
        max_total_input_chars: int = 3000,
        max_tool_result_chars: int = 2000,
        max_history_entries: int = 3,
    ):
        self.llm                    = llm_client
        self.tools                  = tool_registry
        self.max_turns              = max_turns
        self.max_inline_result_chars = max_inline_result_chars
        self.max_total_input_chars  = max_total_input_chars
        self.max_tool_result_chars  = max_tool_result_chars
        self.max_history_entries    = max_history_entries

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
        _PLACEHOLDERS = {"unknown", "n/a", "not specified", "not provided",
                         "none", "unspecified", "tbd", ""}
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
                known.append({"key": f.get("key", ""), "label": f.get("label", f.get("key", "")), "value": f.get("value", "")})
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
                    result.append(str(o))   # backward compat
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
                resolved.append({
                    "node_id":         dep_id,
                    "description":     dep.metadata.get("description", dep_id),
                    "declared_output": [],
                    "result":          "\n".join(lines),
                    "_unknown_fields": unknown,  # used by quality gate and preflight
                    "_known_fields":   known,    # used by preflight check
                })
                continue

            resolved.append({
                "node_id":         dep_id,
                "description":     dep.metadata.get("description", dep_id),
                "declared_output": _format_output_list(dep.metadata.get("output", [])),
                "result":          dep.result,
            })

        # Distribute the char budget evenly across all upstream results
        if resolved:
            budget_per_dep = self.max_total_input_chars // len(resolved)
            for entry in resolved:
                r = entry["result"]
                if len(r) > budget_per_dep:
                    entry["result"] = (
                        r[:budget_per_dep]
                        + f"\n…[truncated, {len(r)} chars total]"
                    )

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
            desc   = getattr(tool, "description", "no description")
            schema = getattr(tool, "input_schema", {})
            lines.append(f"- {name}: {desc}. Args: {json.dumps(schema)}")
        return "\n".join(lines)

    def _build_prompt(self, node, resolved_inputs, history,
                      extra_reminder="", turns_remaining=0,
                      description_override="", output_override=None):

        def _fmt_output(o):
            if isinstance(o, dict):
                return f"  - [{o['type']}] {o['name']}: {o['description']}"
            return f"  - {o}"

        def _output_name(o):
            return o["name"] if isinstance(o, dict) else str(o)

        def _is_file(o):
            if isinstance(o, dict):
                return (
                    o.get("type") == "file"
                    or any(_output_name(o).endswith(ext) for ext in self.FILE_EXTENSIONS)
                )
            return any(str(o).endswith(ext) for ext in self.FILE_EXTENSIONS)

        # ── Upstream results ──────────────────────────────────────────────────
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

        # ── Retry notice ──────────────────────────────────────────────────────
        retry = node.metadata.get("retry_count", 0)
        if retry > 0:
            failure    = node.metadata.get("verification_failure", "unknown")[:200]
            prev_result = node.result or "(none)"
            if len(prev_result) > 200:
                prev_result = prev_result[:200] + "…"
        else:
            failure    = ""
            prev_result = ""
        retry_notice = build_executor_retry_notice(retry, failure, prev_result)

        # ── Tool call history ─────────────────────────────────────────────────
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

        # ── Declared outputs ──────────────────────────────────────────────────
        # When running with a broadened description, output_override contains
        # the broadened output declarations that are consistent with the broadened
        # goal — use these instead of the original node metadata outputs so the
        # LLM isn't working under two contradictory output contracts.
        declared_outputs = (
            output_override
            if output_override is not None
            else node.metadata.get("output", [])
        )
        expected_files   = [_output_name(o) for o in declared_outputs if _is_file(o)]

        if expected_files:
            output_instruction = build_executor_file_output_instruction(expected_files)
        else:
            output_instruction = build_executor_inline_output_instruction(
                self.max_inline_result_chars
            )

        outputs_text = (
            "\n".join(_fmt_output(o) for o in declared_outputs)
            if declared_outputs else "  (not specified)"
        )
        outputs_block = build_executor_outputs_block(outputs_text)

        logger.info(
            "[EXECUTOR] Prompt sections for %s — "
            "retry=%d chars, outputs=%d chars, inputs=%d chars, "
            "tools=%d chars, history=%d chars",
            node.id,
            len(retry_notice), len(outputs_text),
            len(inputs_text), len(tools_text), len(history_text),
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
            max_inline_result_chars=self.max_inline_result_chars,
            turns_remaining=turns_remaining,
            prompt_version=PROMPT_VERSION,
        )

    # ──────────────────────────────────────────────────────────────────────────

    def _fmt_known_fields(self, resolved_inputs: list) -> str:
        """
        Return a formatted string of known clarification fields from the
        resolved inputs, for use in the broadened-description fallback prompt.
        """
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
    ) -> str:
        """
        Focused fallback LLM call used when the primary preflight call returned
        blocked=true but an empty broadened_description.

        Returns the broadened description string, or empty string on failure.
        """
        if getattr(self.llm, "is_stopped", False):
            return ""
        prompt = build_broadened_description_prompt(
            node_id=node.id,
            original_description=node.metadata.get("description", node.id),
            missing_keys=missing_keys,
            known_fields_text=known_fields_text,
        )
        try:
            raw    = self.llm.ask(prompt, schema=BROADENED_DESCRIPTION_SCHEMA)
            parsed = json.loads(raw)
            return parsed.get("broadened_description", "").strip()
        except Exception as e:
            logger.warning(
                "[EXECUTOR] Broadened description fallback call failed for %s: %s",
                node.id, e,
            )
            return ""

    # ──────────────────────────────────────────────────────────────────────────

    def _preflight_awaiting_input(
        self, node, resolved_inputs
    ) -> "AwaitingInputSignal | None":
        """
        Ask the LLM whether this task can be executed with the currently
        available information and tools.

        Returns an AwaitingInputSignal if the task is blocked on missing user
        input, None if it can proceed.

        Two-phase logic:

        Phase 1 — Deterministic gap detection (no LLM call):
          Compare the task's declared required_input list against all
          existing clarification field keys.  Any required input whose name
          does not appear in any clarification field is a structural gap:
          the user cannot supply it because the form has no field for it.
          These are collected as auto_new_fields.

          If gaps exist but no unknown fields do (all clar fields are filled
          and none of them covers this input), return a signal immediately so
          the orchestrator adds the missing field to the clarification form
          before attempting execution.

        Phase 2 — LLM judgment (runs when unknown fields exist):
          Ask the LLM whether the task can proceed despite the unknowns.
          If the LLM decides to block, merge any auto_new_fields from Phase 1
          into the signal's new_fields so the clarification form is always
          patched with every field the task structurally requires.
        """
        # Collect clarification context from resolved inputs
        unknown_fields = []   # list of {key, label}
        known_fields   = []   # list of {key, label, value}
        clar_node_id   = ""

        for entry in resolved_inputs:
            if "_unknown_fields" not in entry:
                continue
            clar_node_id = entry["node_id"]
            unknown_fields.extend(entry["_unknown_fields"])
            known_fields.extend(entry.get("_known_fields", []))

        # ── Phase 1: deterministic required-input gap detection ───────────────
        all_clar_keys   = {f.get("key") for f in known_fields + unknown_fields}
        required_inputs = node.metadata.get("required_input", [])

        def _make_new_field(r: dict) -> dict:
            name  = r.get("name", "")
            label = name.replace("_", " ").title()
            return {
                "key":      name,
                "label":    label,
                "value":    "unknown",
                "rationale": (
                    f"Required by task '{node.id}': {r.get('description', label)}"
                ),
            }

        auto_new_fields = [
            _make_new_field(r) for r in required_inputs
            if r.get("name") and r.get("name") not in all_clar_keys
        ]

        if auto_new_fields:
            logger.info(
                "[EXECUTOR] Node %s has required inputs not covered by any "
                "clarification field — will add: %s",
                node.id, [f["key"] for f in auto_new_fields],
            )

        # Compute the full current missing-key set (unknown fields + structural gaps)
        current_missing_keys = sorted(set(
            [f.get("key") for f in unknown_fields]
            + [f["key"] for f in auto_new_fields]
        ))

        # If nothing is missing at all, proceed with the original description
        if not current_missing_keys:
            return None

        # ── Reuse check: same missing set as last broadened execution? ─────────
        # Skip reuse if the previous execution failed verification — the stored
        # broadened description produced an output the verifier rejected, so
        # regenerating with the failure reason as context has a better chance of
        # succeeding than rerunning the exact same description again.
        previous_failure  = node.metadata.get("verification_failure", "")
        stored_broadened  = node.metadata.get("broadened_description", "")
        stored_for_missing = sorted(node.metadata.get("broadened_for_missing", []))
        if (stored_broadened
                and stored_for_missing == current_missing_keys
                and not previous_failure):
            logger.info(
                "[EXECUTOR] Node %s: reusing stored broadened description "
                "(missing set unchanged: %s)",
                node.id, current_missing_keys,
            )
            return AwaitingInputSignal(
                reason="Reusing stored broadened description — missing fields unchanged.",
                missing_fields=[f.get("key") for f in unknown_fields],
                new_fields=auto_new_fields,
                clarification_node_id=clar_node_id,
                broadened_description=stored_broadened,
                broadened_for_missing=stored_for_missing,
                broadened_output=node.metadata.get("broadened_output", []),
            )

        if previous_failure:
            logger.info(
                "[EXECUTOR] Node %s: previous broadened execution failed verification "
                "('%s...') — regenerating broadened description with failure context",
                node.id, previous_failure[:80],
            )

        # ── Phase 2: LLM judgment + broadened description generation ─────────
        if getattr(self.llm, "is_stopped", False):
            return None  # paused — let the main loop handle it

        def _fmt_fields(fields):
            # Format as "key (label): value" so the LLM can see the exact key
            # string it must use in missing_fields — not just the human label.
            return "\n".join(
                f"  - {f.get('key', '?')} ({f.get('label', f.get('key', '?'))}): "
                f"{f.get('value', 'unknown')}"
                for f in fields
            ) if fields else "  (none)"

        known_fields_text   = _fmt_fields(known_fields)
        unknown_fields_text = _fmt_fields(unknown_fields)
        tools_text          = self._tool_schema_summary()

        if required_inputs:
            required_input_text = "\n".join(
                f"  - {r.get('name', '?')} ({r.get('type', '?')}): "
                f"{r.get('description', '')}"
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
            raw    = self.llm.ask(prompt, schema=AWAITING_INPUT_CHECK_SCHEMA)
            parsed = json.loads(raw)
        except Exception as e:
            logger.warning(
                "[EXECUTOR] Preflight LLM check failed for %s: %s — proceeding",
                node.id, e,
            )
            return None  # fail open: attempt execution anyway

        if not parsed.get("blocked", False):
            return None  # all inputs available — use original description

        # Merge auto_new_fields into whatever the LLM returned, deduplicating
        llm_new_fields  = parsed.get("new_fields", [])
        llm_new_keys    = {f.get("key") for f in llm_new_fields}
        merged_new      = llm_new_fields + [
            f for f in auto_new_fields if f["key"] not in llm_new_keys
        ]

        llm_missing    = parsed.get("missing_fields", [])
        auto_missing   = [f["key"] for f in auto_new_fields
                          if f["key"] not in llm_missing]
        merged_missing = llm_missing + auto_missing

        # broadened_for_missing is the full set of absent keys — used on the
        # next execution to decide whether to reuse or regenerate.
        broadened_for_missing = sorted(set(
            merged_missing + [f.get("key") for f in merged_new]
        ))

        broadened_description = parsed.get("broadened_description", "")
        if not broadened_description:
            logger.warning(
                "[EXECUTOR] Node %s: preflight returned blocked=true but no "
                "broadened_description — will execute with original description",
                node.id,
            )

        return AwaitingInputSignal(
            reason=parsed.get("reason", "task requires user input"),
            missing_fields=merged_missing,
            new_fields=merged_new,
            clarification_node_id=clar_node_id,
            broadened_description=broadened_description,
            broadened_for_missing=broadened_for_missing,
            broadened_output=parsed.get("broadened_output", []),
        )

    # ──────────────────────────────────────────────────────────────────────────

    def execute(self, node, snapshot, reporter=None):
        resolved_inputs = self._resolve_inputs(node, snapshot)

        # ── Pre-flight: check for missing inputs and get broadened description ─
        # The preflight never blocks execution — instead it returns a signal
        # carrying either the stored or freshly-generated broadened_description.
        # If all inputs are available the signal is None and the original
        # description is used.  If inputs are missing the broadened description
        # is used as the effective task goal for this run.
        #
        # After execution completes the orchestrator reads the signal from the
        # reporter and writes broadened_description + broadened_for_missing into
        # node metadata, and patches the clarification node with any new_fields.
        signal = self._preflight_awaiting_input(node, resolved_inputs)

        if signal is not None:
            if signal.broadened_description:
                effective_description = signal.broadened_description
                logger.info(
                    "[EXECUTOR] Node %s: running with broadened description "
                    "(missing: %s)",
                    node.id, signal.broadened_for_missing,
                )
            else:
                # The primary preflight call returned blocked=true but an empty
                # broadened_description (schema enforcement may have been bypassed
                # by constrained-inference producing an empty string).
                # Make a second, focused call solely to generate the broadened
                # description.  If that also fails, skip execution entirely —
                # we must never fall back to the original description for a task
                # the preflight identified as needing unavailable inputs.
                logger.warning(
                    "[EXECUTOR] Node %s: preflight returned no broadened_description "
                    "— making focused fallback call",
                    node.id,
                )
                effective_description = self._generate_broadened_description(
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
                )
            if reporter:
                reporter.on_broadened_execution(signal)
        else:
            effective_description = node.metadata.get("description", node.id)

        # Use broadened_output when running broadened — it is consistent with
        # the broadened description so the LLM isn't working under two
        # contradictory output contracts simultaneously.
        effective_outputs = (
            signal.broadened_output
            if signal is not None and signal.broadened_output
            else node.metadata.get("output", [])
        )

        history = []

        def _output_name(o):
            return o["name"] if isinstance(o, dict) else str(o)

        def _is_file(o):
            if isinstance(o, dict):
                return (
                    o.get("type") == "file"
                    or any(_output_name(o).endswith(ext) for ext in self.FILE_EXTENSIONS)
                )
            return any(str(o).endswith(ext) for ext in self.FILE_EXTENSIONS)

        expected_files = [_output_name(o) for o in effective_outputs if _is_file(o)]

        tool_not_found_count = 0  # replaces string-match loop guard

        for turn in range(self.max_turns):
            turns_remaining = self.max_turns - turn
            extra_reminder  = ""

            # ── File-write reminder ───────────────────────────────────────────
            if expected_files and "write_file" not in {h["name"] for h in history}:
                extra_reminder += build_executor_file_reminder(expected_files, turns_remaining)

            if reporter:
                reporter.on_llm_turn(turn)

            prompt = self._build_prompt(node, resolved_inputs, history,
                                        extra_reminder=extra_reminder,
                                        turns_remaining=turns_remaining,
                                        description_override=effective_description,
                                        output_override=effective_outputs)

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
                result           = response.get("result", "")
                tool_names_used  = {h["name"] for h in history}

                if expected_files and "write_file" not in tool_names_used:
                    logger.warning(
                        "[EXECUTOR] Node %s set done=true but write_file not in history "
                        "— injecting correction turn", node.id
                    )
                    history.append({
                        "name":   "write_file",
                        "args":   {"path": expected_files[0], "content": ""},
                        "result": (
                            f"ERROR: write_file was called with empty content. "
                            f"You must call write_file again with the full content of "
                            f"{expected_files[0]}. Use the actual report content you "
                            f"generated — do not set done=true until the file is written "
                            f"with real content."
                        ),
                    })
                    continue

                logger.info("[EXECUTOR] Node %s completed. Result: %.120s", node.id, result)
                return result

            tool_call = response.get("tool_call")
            if not tool_call:
                logger.warning(
                    "[EXECUTOR] Node %s: done=false but no tool_call on turn %d",
                    node.id, turn + 1,
                )
                if reporter:
                    reporter.on_llm_error(turn, "done=false but no tool_call provided")
                return None

            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})

            if not self.tools or tool_name not in self.tools.tools:
                logger.warning("[EXECUTOR] Node %s requested unknown tool '%s'",
                               node.id, tool_name)
                if reporter:
                    step_id = reporter.on_tool_start(tool_name, tool_args)
                    reporter.on_tool_done(step_id, tool_name, tool_args,
                                         f"ERROR: tool '{tool_name}' not found",
                                         error=True)
                history.append({
                    "name":   tool_name,
                    "args":   tool_args,
                    "result": (
                        f"ERROR: tool '{tool_name}' not found. "
                        "No tools are available — you must complete this task "
                        "using your own knowledge. Set done=true with a result "
                        "based on what you know; do not call any tool."
                    ),
                })
                # Exit if the LLM is stuck calling unavailable tools.
                # A counter is used rather than string-matching the error
                # message — more robust and not tied to error text format.
                tool_not_found_count += 1
                if tool_not_found_count >= 2:
                    logger.error(
                        "[EXECUTOR] Node %s: %d consecutive tool-not-found errors — "
                        "aborting early, no tools registered for this task",
                        node.id, tool_not_found_count,
                    )
                    return None
                continue

            logger.info("[EXECUTOR] Node %s calling tool '%s'", node.id, tool_name)
            tool_not_found_count = 0  # reset on any real tool call
            step_id = reporter.on_tool_start(tool_name, tool_args) if reporter else None

            error = False
            try:
                tool_result = self.tools.execute(tool_name, tool_args)
            except Exception as e:
                tool_result = f"ERROR: {e}"
                error = True
                logger.error("[EXECUTOR] Tool '%s' raised: %s", tool_name, e)

            # Some tools (e.g. web_search) swallow exceptions and return an
            # error string instead of raising.  Detect these so the step node
            # gets status="error" in the UI and the quality gate can see that
            # all tool calls failed.
            tool_result_str = str(tool_result)
            if not error and tool_result_str.startswith("ERROR:"):
                error = True
                logger.warning(
                    "[EXECUTOR] Tool '%s' returned an error string: %.120s",
                    tool_name, tool_result_str,
                )

            if reporter and step_id:
                reporter.on_tool_done(step_id, tool_name, tool_args,
                                      tool_result_str, error=error)

            if len(tool_result_str) > self.max_tool_result_chars:
                tool_result_str = (
                    tool_result_str[:self.max_tool_result_chars]
                    + f"\n…[truncated — {len(tool_result_str)} chars total]"
                )

            history.append({
                "name":   tool_name,
                "args":   tool_args,
                "result": tool_result_str,
            })

            if len(history) > self.max_history_entries:
                history = history[-self.max_history_entries:]

        logger.error("[EXECUTOR] Node %s did not complete within %d turns",
                     node.id, self.max_turns)
        return None



