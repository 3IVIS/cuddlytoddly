# planning/prompts.py
#
# Single source of truth for every prompt template used by the LLM.
#
# Each function takes pre-computed context strings and returns the final
# prompt that is sent to the LLM.  Callers in llm_executor.py,
# llm_planner.py, and quality_gate.py handle all the data extraction and
# formatting; only the template text lives here.
#
# ── How to edit prompts ───────────────────────────────────────────────────────
# Change the text in the functions below.  Variable substitution uses
# standard Python f-strings — everything inside {curly_braces} is filled
# in by the caller.  Do not remove or rename parameters without updating
# the matching call-site.
# ─────────────────────────────────────────────────────────────────────────────

# ---------------------------------------------------------------------------
# System prompts (used by API backends and the llama.cpp chat template)
# ---------------------------------------------------------------------------

# Sent as the "system" role for every OpenAI / Claude API call.
LLM_SYSTEM_PROMPT = (
    "You are a DAG planning assistant. "
    "Always respond with valid JSON and nothing else. "
    "No explanation, no markdown, no code fences."
)

# Injected into the llama.cpp chat template when the model supports it.
# Used verbatim as the <system> turn of the Llama-3 template as a fallback.
LLAMACPP_SYSTEM_PROMPT = (
    "You are a DAG planning assistant. "
    "Always respond with a valid JSON array and nothing else. "
    "No explanation, no markdown, no code fences."
)

# ---------------------------------------------------------------------------
# Executor prompt
# ---------------------------------------------------------------------------

def build_executor_prompt(
    *,
    node_id: str,
    description: str,
    retry_notice: str,
    extra_reminder: str,
    outputs_block: str,
    output_instruction: str,
    inputs_text: str,
    tools_text: str,
    history_text: str,
    max_inline_result_chars: int,
    prompt_version: str = "v3",
) -> str:
    """
    Build the prompt for a single executor turn.

    Parameters are pre-computed by LLMExecutor._build_prompt() and injected
    here so this file stays focused on text, not data wrangling.
    """
    return f"""\
[prompt_version={prompt_version}]
You are executing one task inside a larger automated plan.
Your result will be stored and passed directly to downstream tasks as their input,
so it must be self-contained, specific, and directly usable — not a summary or stub.

════════════════════════════════════════
TASK
{retry_notice}{extra_reminder}
════════════════════════════════════════
ID:          {node_id}
Description: {description}

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


def build_executor_outputs_block(outputs_text: str) -> str:
    """
    Renders the "Expected outputs" block embedded inside the executor prompt.
    Kept separate so callers can toggle it without touching the main template.
    """
    return (
        f"Expected outputs (produce the CONTENT of these, not their names):\n"
        f"        {outputs_text}\n\n"
        f"        IMPORTANT: Do not return the output name as your result. "
        f"Return the actual content.\n"
        f"        For example, if the output is 'research_report', your result must contain\n"
        f"        the actual research findings, not the string 'research_report'."
    )


def build_executor_file_output_instruction(expected_files: list[str]) -> str:
    """Instruction block when the task is expected to write file(s) to disk."""
    return (
        "- This task is expected to produce a file on disk.\n"
        "        Call write_file (or append_file if editing an existing file) "
        "before setting done=true.\n"
        "        Your result string should confirm what was written:\n"
        "            file_written: <filename>\n"
        "            summary: <brief description of contents>"
    )


def build_executor_inline_output_instruction(max_inline_result_chars: int) -> str:
    """Instruction block when the task should return its result inline."""
    return (
        f"- Return your result as a self-contained text string.\n"
        f"        Do NOT write files unless explicitly required by this task's description.\n"
        f"        Do NOT read from or write to disk — pass results inline as text.\n"
        f"        If your result would exceed {max_inline_result_chars} characters, "
        f"write it to a file\n"
        f"        using write_file and return the filename + a summary instead."
    )


def build_executor_retry_notice(retry: int, failure: str, prev_result: str) -> str:
    """Warning block prepended to the prompt on retry attempts."""
    if retry <= 0:
        return ""
    return (
        f"\n        ⚠️  RETRY ATTEMPT {retry} — PREVIOUS ATTEMPT FAILED\n"
        f"        Failure reason: {failure}\n"
        f"        Your previous result was: {prev_result}\n\n"
        f"        You MUST return different, substantive content this time.\n"
        f"        Do NOT return a label, filename, or the output name — "
        f"return the actual data.\n"
        f"        "
    )


def build_executor_file_reminder(expected_files: list[str], turns_remaining: int) -> str:
    """Inline reminder injected when file outputs are declared but not yet written."""
    return (
        f"\nREMINDER: You must call write_file to create "
        f"{expected_files} before setting done=true. "
        f"You have {turns_remaining} turn(s) remaining."
    )


# ---------------------------------------------------------------------------
# Planner prompt
# ---------------------------------------------------------------------------

def build_planner_prompt(
    *,
    pruned_view_json: str,
    goals_repr_json: str,
    existing_ids_note: str,
    skills_block: str,
    min_tasks: int = 3,
    max_tasks: int = 8,
) -> str:
    """
    Build the prompt sent to the LLM planner when decomposing a goal into tasks.

    Parameters
    ----------
    pruned_view_json   : JSON string of the relevant DAG nodes (already serialised).
    goals_repr_json    : JSON string of the goal(s) to expand.
    existing_ids_note  : Pre-formatted note listing IDs already in the DAG.
    skills_block       : Skills summary block (empty string if no skills loaded).
    min_tasks          : Minimum tasks to generate per goal (from config).
    max_tasks          : Maximum tasks to generate per goal (from config).
    """
    return f"""
You are a DAG planning assistant.

Current DAG snapshot:
{pruned_view_json}

Goals to expand:
{goals_repr_json}
{existing_ids_note}{skills_block}
Your task is to decompose each goal into prerequisite tasks.

Guidelines:
- Produce between {min_tasks} and {max_tasks} tasks per goal. Do not exceed {max_tasks} tasks.
- Break goals into tasks at the appropriate level of granularity.
- Avoid vague or abstract tasks.
- Do NOT use verbs like "ensure", "verify", "collect all", "check completeness".
- Every task must produce at least one concrete output.
- Tasks must be actionable and executable.
- If possible, identify tasks that can run in parallel.
- Use the `parallel_group` metadata to indicate tasks that can execute concurrently.
- For each task, specify:
    - `required_input`: list of typed objects {{name, type, description}} describing what this task consumes
    - `output`: list of typed objects {{name, type, description}} describing what this task produces
      - type must be one of: file, document, data, list, url, text, json, code
      - description must be one full sentence explaining the content (not just restating the name)
    - `skill`: which skill to use (if any of the above skills apply)
    - `tools`: which specific tools from that skill are needed
- required_input and dependencies must be fully consistent:
    - Every item in a task's required_input MUST correspond to a dependency on the task
      whose output produces it.
    - Every entry in a task's dependencies must justify at least one item in that
      task's required_input.
    - Never list something in required_input without a producing task in dependencies.
    - Never add a dependency that is not justified by a required_input entry.
    - Tasks with no shared data dependency must run in parallel — do NOT impose
      sequential ordering unless the downstream task actually consumes an upstream output.

Dependency semantics:
- If node A depends on node B, then B must be completed before A.
- Dependencies always point from prerequisite → dependent.
- Goals must depend on the final task that completes them — use ADD_DEPENDENCY for this.
- Tasks must NOT depend on goals.

Response format:
Your response must be a JSON object with exactly two keys:
- "a_goal_result": write this FIRST. 2-4 sentences explaining how these specific tasks
  chain together to achieve the goal. For each dependency edge, name the upstream task,
  the output it produces, and why the downstream task requires that output before it can
  start. For tasks that run in parallel, explain what each independently produces and how
  those outputs are later combined. Be concrete — do not describe tasks generically.
  Use this as a self-check: if you cannot justify a dependency edge here, remove it
  from "events".
- "events": the array of ADD_NODE and ADD_DEPENDENCY events. Only finalise these after
  "a_goal_result" has confirmed every dependency is data-flow justified.

Example of valid output:
{{
  "a_goal_result": "Research_Investment_Options and Analyse_Risk_Profile run in parallel:
    the first produces a ranked list of options, the second a personalised risk score.
    Write_Investment_Report depends on both because it needs the options list to populate
    the recommendations table and the risk score to calibrate which options to highlight.",
  "events": [
    {{
      "type": "ADD_NODE",
      "payload": {{
        "node_id": "Research_Investment_Options",
        "node_type": "task",
        "dependencies": [],
        "metadata": {{
          "description": "Search for high-return investment options.",
          "required_input": [],
          "output": [
            {{
              "name": "investment_options_report",
              "type": "document",
              "description": "Markdown report listing 5-10 high-return investment options with risk level and expected return for each"
            }}
          ],
          "parallel_group": "Research",
          "skill": "web_research",
          "tools": ["web_search", "fetch_url"]
        }}
      }}
    }},
    {{
      "type": "ADD_NODE",
      "payload": {{
        "node_id": "Write_Investment_Report",
        "node_type": "task",
        "dependencies": ["Research_Investment_Options"],
        "metadata": {{
          "description": "Write the final investment report to a file.",
          "required_input": [
            {{
              "name": "investment_options_report",
              "type": "document",
              "description": "Markdown report listing 5-10 high-return investment options with risk level and expected return for each"
            }}
          ],
          "output": [
            {{
              "name": "investment_report.md",
              "type": "file",
              "description": "Final formatted markdown file containing the complete investment analysis, saved to disk"
            }}
          ],
          "skill": "file_ops",
          "tools": ["write_file"]
        }}
      }}
    }},
    {{
      "type": "ADD_DEPENDENCY",
      "payload": {{
        "node_id": "Goal_1",
        "depends_on": "Write_Investment_Report"
      }}
    }}
  ]
}}

Allowed operations: ADD_NODE, ADD_DEPENDENCY

IMPORTANT — response format rules:
- The top-level key must be "type", NOT "operation".
- For ADD_NODE, the body key must be "payload", NOT "node".
- For ADD_DEPENDENCY, put node_id and depends_on inside "payload", NOT at the top level.
- Do NOT include "status" inside node payloads — the system assigns this.
- Do NOT include origin. The system will assign it automatically.
- Do NOT emit ADD_NODE for any node that already exists in the DAG snapshot.
"""


def build_planner_skills_block(skills_summary: str) -> str:
    """
    Renders the skills section injected into the planner prompt.
    Returns an empty string when no skills are loaded.
    """
    if not skills_summary:
        return ""
    return (
        f"\n{skills_summary}\n\n"
        "When decomposing goals into tasks:\n"
        "- Assign the most relevant skill to each task via metadata.skill "
        '(e.g. "web_research")\n'
        "- Tasks assigned a skill should specify metadata.tools listing the specific "
        "tools they need\n"
        "- A task with no matching skill can still be completed by the LLM directly\n"
    )


# ---------------------------------------------------------------------------
# Quality-gate prompts
# ---------------------------------------------------------------------------

def build_verify_result_prompt(
    *,
    node_id: str,
    description: str,
    outputs_text: str,
    result: str,
) -> str:
    """
    Prompt asking the LLM to verify whether a task result satisfies its declared outputs.
    """
    return f"""You are verifying whether a task result satisfies its declared outputs.

    TASK
    ID:          {node_id}
    Description: {description}

    DECLARED OUTPUTS (what this task was supposed to produce):
    {outputs_text}

    ACTUAL RESULT:
    {result}

    Does the result contain actual substantive content, or is it just a label/filename/stub?
    A result that is just a filename, a single word, or a name matching the output label
    is NOT satisfied — the result must contain the actual data.

    Respond only in JSON matching the schema.
    """


def build_check_dependencies_prompt(
    *,
    node_id: str,
    description: str,
    inputs_text: str,
    upstream_text: str,
) -> str:
    """
    Prompt asking the LLM to check whether a task's upstream results are sufficient.
    """
    return f"""You are checking whether a task has everything it needs to execute.

TASK TO RUN
  ID:             {node_id}
  Description:    {description}
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