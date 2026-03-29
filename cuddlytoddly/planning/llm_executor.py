# planning/llm_executor.py

import json
from cuddlytoddly.infra.logging import get_logger
from cuddlytoddly.planning.llm_interface import LLMStoppedError

MAX_INLINE_RESULT_CHARS = 3000

logger = get_logger(__name__)

# Schema for a single execution turn.
# The model either returns a final result or requests a tool call.
EXECUTION_TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "done": {
            "type": "boolean",
            "description": "True if this is the final answer, False if a tool call is needed."
        },
        "result": {
            "type": "string",
            "description": (
                "The final result text. Required when done=true. "
                "Must be a self-contained, detailed description of what was produced — "
                "it will be passed verbatim to downstream tasks as their input."
            )
        },
        "tool_call": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "args": {"type": "object", "additionalProperties": {"type": "string"}}
            },
            "required": ["name", "args"],
            "description": "Tool to call. Required when done=false."
        }
    },
    "required": ["done"]
}


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
    """

    def __init__(self, llm_client, tool_registry=None, max_turns=5):
        self.llm   = llm_client
        self.tools = tool_registry
        self.max_turns = max_turns

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
                    result.append(str(o))  # backward compat
            return result

        MAX_TOTAL_INPUT_CHARS = 3000   # ~750 tokens total for all upstream results
        
        resolved = []
        for dep_id in node.dependencies:
            dep = snapshot.get(dep_id)
            if not dep or not dep.result:
                continue
            resolved.append({
                "node_id":         dep_id,
                "description":     dep.metadata.get("description", dep_id),
                "declared_output": _format_output_list(dep.metadata.get("output", [])),
                "result":          dep.result,   # full for now, truncated below
            })

        # Distribute the budget evenly across all upstream results
        if resolved:
            budget_per_dep = MAX_TOTAL_INPUT_CHARS // len(resolved)
            for entry in resolved:
                r = entry["result"]
                if len(r) > budget_per_dep:
                    entry["result"] = r[:budget_per_dep] + f"\n…[truncated, {len(r)} chars total]"

        return resolved

    def _tool_schema_summary(self):
        if not self.tools:
            return "No tools available."
        lines = []
        for name, tool in self.tools.tools.items():
            desc   = getattr(tool, "description", "no description")
            schema = getattr(tool, "input_schema", {})
            lines.append(f"- {name}: {desc}. Args: {json.dumps(schema)}")
        return "\n".join(lines)

    def _build_prompt(self, node, resolved_inputs, history, extra_reminder=""):

        def _format_output_for_prompt(o):
            if isinstance(o, dict):
                return f"  - [{o['type']}] {o['name']}: {o['description']}"
            return f"  - {o}"  # backward compat

        # ── Upstream results ──────────────────────────────────────────────────
        if resolved_inputs:
            inputs_text = "\n".join(
                f"  [{entry['node_id']}]\n"
                f"    Description:      {entry['description']}\n"
                f"    Declared outputs: {entry['declared_output']}\n"
                f"    Actual result:    {entry['result']}"
                for entry in resolved_inputs
            )
        else:
            inputs_text = "  (none — this is a root task)"

        # In LLMExecutor._build_prompt:
        retry = node.metadata.get("retry_count", 0)
        if retry > 0:
            failure  = node.metadata.get("verification_failure", "unknown")[:200]
            prev_result = node.result or "(none)"
            if len(prev_result) > 200:
                prev_result = prev_result[:200] + "…"
            retry_notice = f"""
        ⚠️  RETRY ATTEMPT {retry} — PREVIOUS ATTEMPT FAILED
        Failure reason: {failure}
        Your previous result was: {prev_result}

        You MUST return different, substantive content this time.
        Do NOT return a label, filename, or the output name — return the actual data.
        """
        else:
            retry_notice = ""

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

        # ── Declared outputs this node should produce ─────────────────────────
        # Determine if this node is expected to produce a file output
        declared_outputs = node.metadata.get("output", [])
        file_extensions  = {".md", ".txt", ".py", ".json", ".csv", ".html",
                            ".yaml", ".yml", ".xml", ".pdf", ".log"}

        def _output_name(o):
            return o["name"] if isinstance(o, dict) else str(o)

        def o_type_is_file(o):
            if isinstance(o, dict):
                return (o.get("type") == "file" or
                        any(_output_name(o).endswith(ext) for ext in file_extensions))
            return any(str(o).endswith(ext) for ext in file_extensions)

        expects_file_output = any(
            o_type_is_file(o) for o in declared_outputs
        )

        description = node.metadata.get("description", "").lower()
        is_file_edit = any(word in description for word in
                        ("edit", "modify", "update", "append", "patch", "overwrite"))

        if expects_file_output or is_file_edit:
            output_instruction = f"""- This task is expected to produce a file on disk.
        Call write_file (or append_file if editing an existing file) before setting done=true.
        Your result string should confirm what was written:
            file_written: <filename>
            summary: <brief description of contents>"""
        else:
            output_instruction = f"""- Return your result as a self-contained text string.
        Do NOT write files unless explicitly required by this task's description.
        Do NOT read from or write to disk — pass results inline as text.
        If your result would exceed {MAX_INLINE_RESULT_CHARS} characters, write it to a file
        using write_file and return the filename + a summary instead."""

        outputs_text = ("\n".join(_format_output_for_prompt(o) for o in declared_outputs) 
                                 if declared_outputs else "  (not specified)"
        )

        outputs_block = f"""Expected outputs (produce the CONTENT of these, not their names):
        {outputs_text}

        IMPORTANT: Do not return the output name as your result. Return the actual content.
        For example, if the output is 'research_report', your result must contain
        the actual research findings, not the string 'research_report'."""

        logger.info(
            "[EXECUTOR] Prompt sections for %s — "
            "retry=%d chars, outputs=%d chars, inputs=%d chars, "
            "tools=%d chars, history=%d chars",
            node.id,
            len(retry_notice), len(outputs_text),
            len(inputs_text), len(tools_text), len(history_text),
        )
        PROMPT_VERSION = "v3"   # bump this when prompt semantics change significantly

        return f"""[prompt_version={PROMPT_VERSION}]
You are executing one task inside a larger automated plan.
Your result will be stored and passed directly to downstream tasks as their input,
so it must be self-contained, specific, and directly usable — not a summary or stub.

════════════════════════════════════════
TASK 
{retry_notice}
{extra_reminder}
════════════════════════════════════════
ID:          {node.id}
Description: {node.metadata.get("description", node.id)}

{outputs_block}

════════════════════════════════════════
INPUTS FROM UPSTREAM TASKS
════════════════════════════════════════
{inputs_text}

════════════════════════════════════════
AVAILABLE TOOLS
════════════════════════════════════════
{tools_text}

{history_text}

════════════════════════════════════════
INSTRUCTIONS
════════════════════════════════════════
- Use upstream results provided in this prompt directly. They are text strings,
  not files on disk. Do not attempt to read them from the filesystem.
{output_instruction}
- Your result must be detailed enough that a downstream task can use it
  without any other context. Label each output clearly:
    investment_analysis: <full content>
    risk_assessment: <full content>
- If you need to call a tool first, set done=false and provide tool_call.
- Only set done=true when you have a complete, usable result.
- Use \\n for line breaks and 4 spaces for indentation in Python code.
  Do NOT compress multi-line code onto one line with semicolons.
"""

    # ──────────────────────────────────────────────────────────────────────────

    def execute(self, node, snapshot, reporter=None):
        resolved_inputs = self._resolve_inputs(node, snapshot)
        history = []

        declared_outputs = node.metadata.get("output", [])
        file_extensions  = {".md", ".txt", ".py", ".json", ".csv", ".html",
                            ".yaml", ".yml", ".xml", ".pdf", ".log"}

        def _output_name(o):
            return o["name"] if isinstance(o, dict) else str(o)

        def o_type_is_file(o):
            if isinstance(o, dict):
                return (o.get("type") == "file" or
                        any(_output_name(o).endswith(ext) for ext in file_extensions))
            return any(str(o).endswith(ext) for ext in file_extensions)


        expected_files = [
            _output_name(o) for o in declared_outputs
            if o_type_is_file(o)
        ]

        for turn in range(self.max_turns):
            # If we have turns remaining and file outputs are declared,
            # remind the model upfront in the prompt
            turns_remaining = self.max_turns - turn
            file_reminder = ""
            if expected_files and "write_file" not in {h["name"] for h in history}:
                file_reminder = (
                    f"\nREMINDER: You must call write_file to create "
                    f"{expected_files} before setting done=true. "
                    f"You have {turns_remaining} turn(s) remaining."
                )

            prompt = self._build_prompt(node, resolved_inputs, history, 
                                        extra_reminder=file_reminder)

            if reporter:
                reporter.on_llm_turn(turn)

            prompt = self._build_prompt(node, resolved_inputs, history)

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

            # In LLMExecutor.execute(), after parsing response, before checking done:
            if response.get("done"):
                result = response.get("result", "")

                # If this node declares file outputs, verify a write_file was called
                # this session before accepting done=true

                tool_names_used  = {h["name"] for h in history}

                # In LLMExecutor.execute(), replace the correction turn injection:
                if expected_files and "write_file" not in tool_names_used:
                    logger.warning(
                        "[EXECUTOR] Node %s set done=true but write_file not in history "
                        "— injecting correction turn", node.id
                    )
                    # Inject as a complete tool exchange (call + result) so the model
                    # understands it's still in the middle of execution
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
                    # Also cap done=false by continuing — the next turn must use tool_call
                    continue

                logger.info("[EXECUTOR] Node %s completed. Result: %.120s", node.id, result)
                return result

            tool_call = response.get("tool_call")
            if not tool_call:
                logger.warning("[EXECUTOR] Node %s: done=false but no tool_call on turn %d",
                            node.id, turn + 1)
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
                    "name": tool_name, "args": tool_args,
                    "result": f"ERROR: tool '{tool_name}' not found",
                })
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

            MAX_TOOL_RESULT_CHARS = 2000
            MAX_HISTORY_ENTRIES = 3

            tool_result_str = str(tool_result)
            if len(tool_result_str) > MAX_TOOL_RESULT_CHARS:
                tool_result_str = (
                    tool_result_str[:MAX_TOOL_RESULT_CHARS]
                    + f"\n…[truncated — {len(tool_result_str)} chars total]"
                )

            history.append({
                "name":   tool_name,
                "args":   tool_args,
                "result": tool_result_str,
            })

            if len(history) > MAX_HISTORY_ENTRIES:
                history = history[-MAX_HISTORY_ENTRIES:]

        logger.error("[EXECUTOR] Node %s did not complete within %d turns",
                    node.id, self.max_turns)
        return None
