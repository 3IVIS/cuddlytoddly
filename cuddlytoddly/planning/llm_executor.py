# planning/llm_executor.py

import json
from cuddlytoddly.infra.logging import get_logger
from cuddlytoddly.planning.llm_interface import LLMStoppedError
from cuddlytoddly.planning.schemas import EXECUTION_TURN_SCHEMA
from cuddlytoddly.planning.prompts import (
    build_executor_prompt,
    build_executor_outputs_block,
    build_executor_file_output_instruction,
    build_executor_inline_output_instruction,
    build_executor_retry_notice,
    build_executor_file_reminder,
)

logger = get_logger(__name__)

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

    def _build_prompt(self, node, resolved_inputs, history, extra_reminder=""):

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
        declared_outputs = node.metadata.get("output", [])
        expected_files   = [_output_name(o) for o in declared_outputs if _is_file(o)]

        description_lower = node.metadata.get("description", "").lower()
        is_file_edit = any(
            word in description_lower
            for word in ("edit", "modify", "update", "append", "patch", "overwrite")
        )

        if expected_files or is_file_edit:
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
            description=node.metadata.get("description", node.id),
            retry_notice=retry_notice,
            extra_reminder=extra_reminder,
            outputs_block=outputs_block,
            output_instruction=output_instruction,
            inputs_text=inputs_text,
            tools_text=tools_text,
            history_text=history_text,
            max_inline_result_chars=self.max_inline_result_chars,
            prompt_version=PROMPT_VERSION,
        )

    # ──────────────────────────────────────────────────────────────────────────

    def execute(self, node, snapshot, reporter=None):
        resolved_inputs = self._resolve_inputs(node, snapshot)
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

        declared_outputs = node.metadata.get("output", [])
        expected_files   = [_output_name(o) for o in declared_outputs if _is_file(o)]

        for turn in range(self.max_turns):
            turns_remaining = self.max_turns - turn
            file_reminder   = ""
            if expected_files and "write_file" not in {h["name"] for h in history}:
                file_reminder = build_executor_file_reminder(expected_files, turns_remaining)

            if reporter:
                reporter.on_llm_turn(turn)

            prompt = self._build_prompt(node, resolved_inputs, history,
                                        extra_reminder=file_reminder)

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
                # Early exit if every turn so far has been a tool-not-found
                # error — the LLM is stuck in a loop and won't self-correct.
                if all(
                    h.get("result", "").startswith("ERROR: tool") and "not found" in h.get("result", "")
                    for h in history
                ):
                    logger.error(
                        "[EXECUTOR] Node %s: all %d turn(s) hit tool-not-found — "
                        "aborting early, no tools registered for this task",
                        node.id, len(history),
                    )
                    return None
                continue

            logger.info("[EXECUTOR] Node %s calling tool '%s'", node.id, tool_name)
            step_id = reporter.on_tool_start(tool_name, tool_args) if reporter else None

            error = False
            try:
                tool_result = self.tools.execute(tool_name, tool_args)
            except Exception as e:
                tool_result = f"ERROR: {e}"
                error = True
                logger.error("[EXECUTOR] Tool '%s' raised: %s", tool_name, e)

            if reporter and step_id:
                reporter.on_tool_done(step_id, tool_name, tool_args,
                                      str(tool_result), error=error)

            tool_result_str = str(tool_result)
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