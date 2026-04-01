# planning/prompts.py
#
# Single source of truth for every prompt template used by the LLM.
#
# Each function takes pre-computed context strings and returns the final
# prompt that is sent to the LLM.  Callers in llm_executor.py,
# llm_planner.py, quality_gate.py, and plan_constraint_checker.py handle
# all the data extraction and formatting; only the template text lives here.
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
# Used verbatim as the <s> turn of the Llama-3 template as a fallback.
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
# Plan scrutinizer prompt
# ---------------------------------------------------------------------------

def build_plan_scrutinizer_prompt(
    *,
    original_planning_prompt: str,
    draft_plan_json: str,
    min_tasks: int = 3,
    max_tasks: int = 8,
) -> str:
    """
    Build the prompt used to scrutinize and improve a draft plan.

    The LLM is shown its own draft plan alongside the complete set of
    constraints from the original planning call.  It is asked to evaluate
    the plan's content and realism first, structural correctness second,
    then produce an improved plan in the same JSON format.

    Parameters
    ----------
    original_planning_prompt : The full prompt that produced the draft plan.
                               Embedding it here means every constraint
                               (DAG snapshot, goals, existing IDs, skills,
                               task-count limits, dependency semantics, format
                               rules) is visible during scrutiny — no context
                               is lost between the two LLM calls.
    draft_plan_json          : The raw JSON string returned by the first call.
    min_tasks                : Minimum tasks per goal (from config, for the
                               explicit reminder in the scrutiny instructions).
    max_tasks                : Maximum tasks per goal (same reason).
    """
    return f"""\
You are a DAG planning assistant performing a critical self-review of a draft plan.
Your primary job here is to evaluate whether the plan will actually work — whether
the tasks are realistic, sufficient, and well-described — not just whether the JSON
is correctly shaped.

══════════════════════════════════════════════════════════════════════════════
ORIGINAL PLANNING REQUEST (all constraints apply unchanged)
══════════════════════════════════════════════════════════════════════════════
{original_planning_prompt}

══════════════════════════════════════════════════════════════════════════════
DRAFT PLAN TO SCRUTINIZE
══════════════════════════════════════════════════════════════════════════════
{draft_plan_json}

══════════════════════════════════════════════════════════════════════════════
SCRUTINY INSTRUCTIONS
══════════════════════════════════════════════════════════════════════════════
Work through the following checks in order. For every problem you find,
fix it in the output — do not just note it.

── CONTENT AND REALISM (most important) ──────────────────────────────────────

1. GOAL COVERAGE — Imagine the tasks completing one by one. When every task
   is done, does the goal stated in the original request actually get achieved?
   Ask yourself: what would be missing? If the goal would not be fully met,
   add the tasks needed to close the gap. Be specific about what is missing
   and why, not just that something feels incomplete.

2. TASK REALISM — For each task, ask: could an autonomous agent actually
   execute this given only its description and the declared inputs?
   Flag and rewrite any task where:
   - The description is too vague for an executor to know what to do
     (e.g. "process the data" without saying what processing means).
   - The task implicitly assumes knowledge or context that is not provided
     in its required_input or available skills.
   - The expected effort is wildly disproportionate — a single task that
     would realistically take many separate steps should be split; a chain
     of trivial tasks that could obviously be one should be merged.

3. OUTPUT COMPLETENESS — For each task's declared outputs, ask: is this
   output specific enough that a downstream task could use it without
   guessing? A description like "report" or "data" is not sufficient.
   Each output description must state what the content actually contains.
   Rewrite any output description that is generic or circular (restating
   the output name rather than its contents).

4. MISSING IMPLICIT STEPS — Look for gaps the plan skips over. Common
   examples: a task that writes a report but no prior task gathered the
   raw information; a task that merges results but no task produced one
   of those results; a task that calls an API but no task obtained the
   credentials or endpoint. Add any missing bridging tasks.

5. REDUNDANCY — Identify tasks that duplicate each other's work or whose
   outputs are never consumed by any downstream task or by the goal.
   Remove or merge redundant tasks and update dependencies accordingly.

── CONSISTENCY (important) ───────────────────────────────────────────────────

6. INPUT / OUTPUT ALIGNMENT — For every dependency edge A → B, confirm
   that task A's outputs contain something task B's required_input needs.
   If a required_input item has no producing task in dependencies, either
   add the missing producer or remove the input. If a dependency exists
   but nothing flows across it, remove the dependency.

7. TASK DESCRIPTIONS VS METADATA — Each task's description must match
   what its output list says it produces, and what its required_input says
   it consumes. Fix any description that contradicts the metadata.

── STRUCTURAL CONSTRAINTS (required, but secondary) ──────────────────────────

8. Apply these mechanical rules to the final set of tasks and edges:
   - Task count: between {min_tasks} and {max_tasks} tasks per goal.
   - No ADD_NODE for IDs already listed in the "Nodes already in the DAG"
     section of the original request.
   - Tasks with no shared data dependency must share a parallel_group.
   - Event format: key "type" (not "operation"), body key "payload" (not
     "node"), ADD_DEPENDENCY fields inside "payload", no "status" or
     "origin" in node payloads.

── FINAL NARRATIVE ───────────────────────────────────────────────────────────

9. Rewrite a_goal_result to reflect the improved plan. It must explain,
   in concrete terms, how the final set of tasks chains together to
   achieve the goal: what each task produces, who consumes it, and why
   that ordering is necessary. If you cannot write this narrative without
   hand-waving, that is a signal the plan still has gaps — fix them first.

══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
══════════════════════════════════════════════════════════════════════════════
Return the improved plan as a JSON object with exactly two keys:
  "a_goal_result" — the updated narrative
  "events"        — the corrected ADD_NODE / ADD_DEPENDENCY array

No commentary, apologies, or text outside the JSON object.
If the draft plan is already fully correct on all checks above, reproduce it
unchanged.
"""


# ---------------------------------------------------------------------------
# Ghost node resolution prompt
# ---------------------------------------------------------------------------

def build_ghost_node_resolution_prompt(
    *,
    ghost_node_id: str,
    ghost_description: str,
    new_nodes: dict,
    existing_nodes: dict,
    active_goal_id: str,
    edges: set,
    valid_candidates: set,
) -> str:
    """
    Build the prompt used to resolve a ghost node — a new plan node that has
    no dependents (nothing in the plan depends on it, so its output goes unused).

    The LLM is shown the full plan context and a pre-filtered list of valid
    candidate dependents (ancestors excluded to prevent introducing cycles) and
    asked to choose the node that would most naturally consume the ghost's output.

    Parameters
    ----------
    ghost_node_id     : ID of the node with no dependents.
    ghost_description : Description of the ghost node.
    new_nodes         : dict[node_id → description] for all proposed new nodes.
    existing_nodes    : dict[node_id → description] for all live graph nodes.
    active_goal_id    : ID of the goal being planned for.
    edges             : set of (node_id, depends_on) tuples from the current plan.
    valid_candidates  : Pre-computed set of node IDs that may legally depend on
                        the ghost (ancestors excluded).
    """
    # ── Format plan nodes ─────────────────────────────────────────────────────
    nodes_lines = []
    for nid, desc in sorted(new_nodes.items()):
        marker = "  ◀ NO DEPENDENT" if nid == ghost_node_id else ""
        nodes_lines.append(f"  [{nid}]{marker}\n    {desc}")
    nodes_text = "\n".join(nodes_lines) or "  (none)"

    # ── Format current dependency edges ───────────────────────────────────────
    # Edge (A, B) means "A depends on B" — B must complete before A.
    edges_lines = [
        f"  {src} waits for {dep}"
        for src, dep in sorted(edges)
    ]
    edges_text = "\n".join(edges_lines) or "  (none)"

    # ── Format valid candidates ───────────────────────────────────────────────
    candidate_lines = []
    for nid in sorted(valid_candidates):
        if nid == active_goal_id:
            desc = "(the overall goal — signals that ghost node completion is required)"
        elif nid in new_nodes:
            desc = new_nodes[nid]
        else:
            desc = existing_nodes.get(nid, "")
        candidate_lines.append(f"  [{nid}]: {desc}")
    candidates_text = "\n".join(candidate_lines) or "  (none available)"

    return f"""\
You are reviewing a DAG execution plan where one node has no dependents.

A node with no dependents means its output is never consumed — it is a "ghost node"
that produces results no other task or goal uses.  Your job is to decide which node
should depend on it, i.e. which node should run AFTER it and use its output.

"A depends on B" means B must complete before A can start.
You are choosing a new A for the ghost node (which will be B).

══════════════════════════════════════════════════════════════════════════════
GHOST NODE (has no dependents)
══════════════════════════════════════════════════════════════════════════════
ID:          {ghost_node_id}
Description: {ghost_description}

══════════════════════════════════════════════════════════════════════════════
ALL NODES IN THE CURRENT PLAN
══════════════════════════════════════════════════════════════════════════════
{nodes_text}

══════════════════════════════════════════════════════════════════════════════
CURRENT DEPENDENCY EDGES  (format: A waits for B)
══════════════════════════════════════════════════════════════════════════════
{edges_text}

══════════════════════════════════════════════════════════════════════════════
VALID CANDIDATES  (nodes that may legally depend on the ghost node)
══════════════════════════════════════════════════════════════════════════════
{candidates_text}

══════════════════════════════════════════════════════════════════════════════
YOUR TASK
══════════════════════════════════════════════════════════════════════════════
Choose the single best candidate from the VALID CANDIDATES list above.
Pick the node whose work would most naturally consume the output of [{ghost_node_id}].

If no specific task clearly needs this output, choose [{active_goal_id}] — this
signals that the ghost node's completion is a required prerequisite for the goal
even if its output is not consumed by another task directly.

You MUST choose from the VALID CANDIDATES list.  Do not invent new node IDs.
Respond only in JSON matching the schema.
"""


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