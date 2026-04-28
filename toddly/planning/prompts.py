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
    "You are the planning engine of an autonomous task-execution system. "
    "Your job is to decompose goals into directed acyclic graphs (DAGs) of "
    "concrete, executable tasks and to verify that completed work meets its "
    "declared outputs. The system is domain-agnostic: goals may cover software "
    "development, document creation, web research, data analysis, content "
    "production, or any other knowledge work. "
    "\n\n"
    "Behavioral rules:\n"
    "- Always respond with valid JSON and nothing else — no explanation, no "
    "markdown, no code fences.\n"
    "- When a goal or task description is ambiguous, resolve the ambiguity "
    "conservatively: choose the narrower, safer interpretation and flag the "
    "assumption in your output where the schema allows it.\n"
    "- Never invent node IDs, field keys, or schema properties not defined in "
    "the request.\n"
    "- If asked to plan something that involves irreversible real-world actions "
    "(sending messages, publishing content, deleting data, making purchases), "
    "include that action as an explicitly labelled step so the orchestrator can "
    "surface it for human review."
)

# Injected into the llama.cpp chat template when the model supports it.
# Used verbatim as the <s> turn of the Llama-3 template as a fallback.
LLAMACPP_SYSTEM_PROMPT = (
    "You are the planning engine of an autonomous task-execution system. "
    "Your job is to decompose goals into directed acyclic graphs (DAGs) of "
    "concrete, executable tasks and to verify that completed work meets its "
    "declared outputs. The system is domain-agnostic: goals may cover software "
    "development, document creation, web research, data analysis, content "
    "production, or any other knowledge work. "
    "\n\n"
    "Behavioral rules:\n"
    "- Always respond with valid JSON and nothing else — no explanation, no "
    "markdown, no code fences.\n"
    "- When a goal or task description is ambiguous, resolve the ambiguity "
    "conservatively: choose the narrower, safer interpretation.\n"
    "- Never invent node IDs, field keys, or schema properties not defined in "
    "the request.\n"
    "- If asked to plan something that involves irreversible real-world actions "
    "(sending messages, publishing content, deleting data, making purchases), "
    "include that action as an explicitly labelled step."
)

# Used by ApiLLM.ask_with_tools() — replaces the JSON-only system prompt for
# native tool-use calls where the model responds with plain text, not JSON.
EXECUTOR_NATIVE_SYSTEM_PROMPT = (
    "You are the execution engine of an autonomous task-planning system. "
    "Each conversation represents a single task node within a larger DAG plan. "
    "The task you receive is one step in a multi-step goal; your output will be "
    "stored and passed directly to downstream tasks, so it must be self-contained "
    "and immediately usable without additional context. "
    "\n\n"
    "The system is domain-agnostic. Tasks may involve web research, writing, "
    "data analysis, code generation, document handling, or any other knowledge "
    "work. Approach each task on its own terms. "
    "\n\n"
    "Operational rules:\n"
    "- Use the tools provided to gather information. Prefer tool-retrieved facts "
    "over your training knowledge wherever the task requires current, specific, "
    "or verifiable data.\n"
    "- If a tool call fails or returns no useful result, try an alternative "
    "approach (different query, different tool) before concluding that information "
    "is unavailable. If all attempts fail, clearly state what could not be "
    "retrieved rather than substituting fabricated content.\n"
    "- Do not take irreversible real-world actions (sending emails, posting "
    "publicly, deleting files, making purchases) unless the task description "
    "explicitly instructs you to and the action is listed in your available tools.\n"
    "- When you have everything required, respond with your final answer as "
    "plain text — detailed, self-contained, and ready to be passed downstream.\n"
    "- Express uncertainty explicitly. If your result is based on incomplete "
    "information, state which parts are confident and which are best-effort."
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
    steps_text: str = "  (not specified)",
    max_inline_result_chars: int,
    turns_remaining: int = 0,
) -> str:
    if turns_remaining == 1:
        turns_line = (
            "- FINAL TURN: Do not call any more tools. Return your result now using done=true.\n"
        )
    elif turns_remaining > 1:
        turns_line = (
            f"- Turns remaining (including this one): {turns_remaining}. "
            "If you have already made multiple searches, synthesise what you have "
            "now rather than searching again.\n"
        )
    else:
        turns_line = ""
    return f"""\
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
EXECUTION STEPS
════════════════════════════════════════
Work through each of these steps in order. Only call tools that appear in the
AVAILABLE TOOLS list below. Do not attempt steps that require external actions
outside your tool capabilities — skip them and note they were skipped.
{steps_text}

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
- GROUNDING GUIDANCE: Prefer facts retrieved via tool calls in this session over
  your training knowledge for any claim that is specific, current, or verifiable
  (names, figures, URLs, statistics, code outputs). For general reasoning,
  synthesis, and widely-stable knowledge, you may draw on training knowledge
  without a tool call — but mark any such content as inference rather than
  retrieved fact when doing so would affect how a downstream task uses it.
  If tool calls returned insufficient information, state clearly what is missing
  rather than substituting invented content.
- Tool failure protocol: if a tool call fails or returns an empty or unhelpful
  result, attempt at least one alternative approach (different query, different
  tool) before treating that information as unavailable. Briefly note any failed
  attempts in your result so the quality gate has context.
- Your result must be detailed enough that a downstream task can use it
  without any other context. Label each output clearly using the output names
  declared above, for example:
    <output_name>: <full content>
    <second_output_name>: <full content>
- If you need to call a tool first, set done=false and provide tool_call.
- Only set done=true when you have a complete, usable result.
{turns_line}- Use \\n for line breaks and 4 spaces for indentation in code.
  Do NOT compress multi-line code onto one line with semicolons.
- Uncertainty: if your result is partially complete or based on incomplete
  information, prefix the affected section with [BEST EFFORT] and explain
  what could not be confirmed. A labelled partial result is more useful to
  downstream tasks than a confident fabrication.
"""


def build_executor_outputs_block(outputs_text: str) -> str:
    """
    Renders the "Expected outputs" block embedded inside the executor prompt.
    Kept separate so callers can toggle it without touching the main template.
    """
    return (
        f"Expected outputs (produce the CONTENT of these, not their names or labels):\n"
        f"        {outputs_text}\n\n"
        f"        IMPORTANT: Do not return the output name as your result. "
        f"Return the actual content.\n"
        f"        Examples across task types:\n"
        f"          If the output is 'research_report'  → return the actual research findings.\n"
        f"          If the output is 'python_script'    → return the actual code.\n"
        f"          If the output is 'summary_document' → return the actual written text.\n"
        f"          If the output is 'data_table'       → return the actual structured data."
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
# Native tool-use executor prompt (no JSON schema / done+tool_call protocol)
# ---------------------------------------------------------------------------


def build_executor_native_prompt(
    *,
    node_id: str,
    description: str,
    retry_notice: str,
    extra_reminder: str,
    outputs_block: str,
    output_instruction: str,
    inputs_text: str,
    steps_text: str = "  (not specified)",
    turns_remaining: int = 0,
) -> str:
    """
    Prompt for a single native tool-use executor turn.

    Unlike build_executor_prompt, this version:
    - Omits the AVAILABLE TOOLS text block (tools are passed via the provider API).
    - Omits the previous-tool-call history block (history lives in the message list).
    - Omits all references to done=true / tool_call JSON fields.
    - Instructs the model to respond with plain text when the task is complete.
    """
    if turns_remaining == 1:
        turns_line = (
            "- FINAL TURN: Do not call any more tools. Respond now with your complete "
            "plain-text result.\n"
        )
    elif turns_remaining > 1:
        turns_line = (
            f"- Turns remaining (including this one): {turns_remaining}. "
            "If you have already made multiple searches, synthesise what you have "
            "now rather than searching again.\n"
        )
    else:
        turns_line = ""
    return f"""\
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
EXECUTION STEPS
════════════════════════════════════════
Work through each of these steps in order using the tools available to you.
{steps_text}

════════════════════════════════════════
INPUTS FROM UPSTREAM TASKS
════════════════════════════════════════
{inputs_text}

════════════════════════════════════════
INSTRUCTIONS
════════════════════════════════════════
- Use upstream results provided in this prompt directly. They are text strings,
  not files on disk. Do not attempt to read them from the filesystem.
{output_instruction}
- GROUNDING GUIDANCE: Prefer facts retrieved via tool calls in this session over
  your training knowledge for any claim that is specific, current, or verifiable
  (names, figures, URLs, statistics, code outputs). For general reasoning,
  synthesis, and widely-stable knowledge, you may draw on training knowledge
  without a tool call — but mark any such content as inference rather than
  retrieved fact when doing so would affect how a downstream task uses it.
  If tool calls returned insufficient information, state clearly what is missing
  rather than substituting invented content.
- Tool failure protocol: if a tool call fails or returns an empty or unhelpful
  result, attempt at least one alternative approach (different query, different
  tool) before treating that information as unavailable. Briefly note any failed
  attempts in your result so the quality gate has context.
- Your result must be detailed enough that a downstream task can use it
  without any other context. Label each output clearly using the output names
  declared above, for example:
    <output_name>: <full content>
    <second_output_name>: <full content>
- Call tools as needed to gather information. When you have everything required,
  respond with your final answer as plain text — do not produce JSON unless the
  task description explicitly requires it.
{turns_line}- Use \\n for line breaks and 4 spaces for indentation in code.
  Do NOT compress multi-line code onto one line with semicolons.
- Uncertainty: if your result is partially complete or based on incomplete
  information, prefix the affected section with [BEST EFFORT] and explain
  what could not be confirmed. A labelled partial result is more useful to
  downstream tasks than a confident fabrication.
"""


def build_executor_native_file_output_instruction(expected_files: list[str]) -> str:
    """Native-path instruction when the task is expected to write file(s) to disk."""
    return (
        "- This task is expected to produce a file on disk.\n"
        "        Call write_file (or append_file if editing an existing file) "
        "before giving your final answer.\n"
        "        Your final text response should confirm what was written:\n"
        "            file_written: <filename>\n"
        "            summary: <brief description of contents>"
    )


def build_executor_native_inline_output_instruction(max_inline_result_chars: int) -> str:
    """Native-path instruction when the task should return its result inline."""
    return (
        f"- Return your result as a self-contained text string.\n"
        f"        Do NOT write files unless explicitly required by this task's description.\n"
        f"        Do NOT read from or write to disk — pass results inline as text.\n"
        f"        If your result would exceed {max_inline_result_chars} characters, "
        f"write it to a file\n"
        f"        using write_file and return the filename + a summary instead."
    )


def build_executor_native_file_reminder(expected_files: list[str], turns_remaining: int) -> str:
    """Native-path reminder injected when file outputs are declared but not yet written."""
    return (
        f"\nREMINDER: You must call write_file to create "
        f"{expected_files} before giving your final answer. "
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
    clarification_block: str = "",
    root_goal_text: str = "",
    is_replan: bool = False,
) -> str:
    """
    Build the prompt sent to the LLM planner when decomposing a goal into tasks.

    Parameters
    ----------
    pruned_view_json    : JSON string of the relevant DAG nodes (already serialised).
    goals_repr_json     : JSON string of the goal(s) to expand.
    existing_ids_note   : Pre-formatted note listing IDs already in the DAG.
    skills_block        : Skills summary block (empty string if no skills loaded).
    min_tasks           : Minimum tasks to generate per goal (from config).
    max_tasks           : Maximum tasks to generate per goal (from config).
    clarification_block : Pre-formatted block of goal context fields (empty string
                          when no clarification node exists for this goal).
    root_goal_text      : The verbatim root goal description — injected as a
                          goal-alignment anchor so the planner never drifts to
                          a surface-level keyword interpretation of the goal.
    is_replan           : When True, injects a replan idempotency notice that
                          forbids modifying or re-emitting nodes that already
                          exist in the DAG snapshot.
    """
    # Build a goal-alignment anchor block when the root goal is known.
    goal_alignment_block = ""
    if root_goal_text:
        goal_alignment_block = f"""
CRITICAL — ROOT GOAL ALIGNMENT:
The root goal you are decomposing is:
  "{root_goal_text}"

Every task you generate must directly and specifically advance this goal.
Do NOT create tasks that address a keyword from the goal text in isolation if
that keyword is not what the goal is actually asking for.
Before finalising each task, ask yourself: "If this task completes successfully,
does its output bring the plan meaningfully closer to achieving the root goal?"
If not, redesign the task.

Example of misalignment to avoid:
  Goal: "find repos that help review a Python repo"
  WRONG task: "Research popular Python repositories (stars, forks)"
    — this targets the word "repositories" but not the goal's intent.
  RIGHT task: "Find Python code-review and static-analysis tool repositories"
    — this directly serves what the goal is asking for.
"""

    # Inject a replan idempotency notice when replanning an existing goal.
    replan_block = ""
    if is_replan:
        replan_block = """
REPLAN NOTICE — this is a partial replan of an existing goal.
You MUST NOT modify, re-emit, or re-describe any node whose ID appears in the
"Nodes already in the DAG" list above, regardless of whether you think it could
be improved. Those nodes are either completed or in-progress. Only add new nodes
or new dependency edges.
"""

    return f"""\
You are a DAG planning assistant.

Current DAG snapshot:
{pruned_view_json}

Goals to expand:
{goals_repr_json}
{existing_ids_note}{replan_block}{clarification_block}{goal_alignment_block}{skills_block}
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

Scope guidance:
- Task granularity should match the goal's natural complexity. A goal that is
  genuinely simple (write a short document, look something up) should use fewer
  tasks; a goal that is genuinely complex (build a multi-step pipeline, produce a
  research report with several distinct workstreams) may use more. The task count
  range is a guardrail, not a target — aim for the minimum number of tasks that
  would actually produce a high-quality result for this specific goal.
- A well-scoped task takes a single autonomous agent session to complete. If a
  task description implies days of continuous work by a human, split it. If two
  tasks could obviously be done in the same session without loss of quality,
  merge them.

Node types:
- `task`: the only type you should emit for executable work. Every task must
  be actionable and produce at least one concrete output through research,
  analysis, writing, or real-world action — NOT by asking the user for
  information.

  CRITICAL — never create a task whose sole purpose is to collect information
  from the user. If a piece of information can only come from the user (e.g.
  their repo name, their budget, their personal goals), it belongs as a
  Category B required_input field on the task that actually needs it — not as
  a separate task node. The clarification node handles collecting that input
  automatically. Creating a task like "Determine_GitHub_Repository" or
  "Identify_User_Goal" that just asks the user for a value is always wrong.

  If a task requires personal or company-specific information that may not be
  available, still emit it as a `task` — declare the missing information as a
  Category B required_input and let the executor handle the missing context at
  runtime. Do not attempt to pre-classify tasks as needing user input by
  creating wrapper tasks around that input collection.

- For each task, specify:
    - `required_input`: list of typed objects {{name, type, description}} describing what this task consumes.
      required_input has TWO categories — declare BOTH where applicable:

      Category A — outputs from upstream tasks:
        List every named output this task needs from another task in the plan.
        Every Category A item MUST have a corresponding dependency on the task
        that produces it, and every such dependency MUST justify at least one
        Category A item.

      Category B — user context from the clarification node:
        List any specific information this task needs that only the user can
        provide and that cannot be retrieved by web search or general reasoning.
        Examples:
          - A task that researches a specific company MUST declare
            {{name: "company_name", type: "text", description: "..."}}
          - A task that tailors advice to the user's personal style MUST declare
            {{name: "personal_negotiation_style", type: "text", description: "..."}}
          - A task that analyses the user's company culture MUST declare
            {{name: "company_culture", type: "text", description: "..."}}
        Category B items do NOT need a corresponding task dependency — they are
        satisfied by the clarification node.  If a Category B item is not already
        present in the clarification fields, the executor will add it to the form
        automatically at runtime.
        Do NOT add Category B items for information that web search can provide
        (e.g. industry salary benchmarks, general negotiation frameworks, public
        company profiles when the company name is known).

    - `output`: list of typed objects {{name, type, description}} describing what this task produces
      - type must be one of: file, document, data, list, url, text, json, code
      - description must be one full sentence explaining the content (not just restating the name)
    - `execution_steps`: ordered list of concrete steps the executor must work through.
      Each step has:
        - `execution_type`: a specific verb-noun identifier. Use tool names for steps the
          LLM can perform (search_web, fetch_url, write_file, write_plan, write_document,
          write_analysis, analyse_data, run_code). Use real-world action names for steps
          that require the user to act outside the LLM (post_to_reddit, post_to_linkedin,
          send_email, send_newsletter, publish_blog_post, run_terminal_command,
          deploy_service, open_pull_request, schedule_meeting).
          Be specific — not "share" but "post_to_reddit".
          NEVER use "user_input" as an execution_type. If a step is blocked on user-
          supplied information, declare that information as a Category B required_input
          instead and let the executor handle it. The clarification node collects user
          input — task execution steps must not duplicate this.
        - `description`: one sentence — exactly what this step does in this task context.
        - `produces`: one sentence — what this step contributes toward the node's output.
      Rules for execution_steps:
        - Steps must be in the order they should be performed.
        - Every task must have at least one execution step.
        - The planner does NOT decide whether the LLM can perform a step — that is the
          executor's job at runtime. Declare all intended actions, including real-world ones.
        - Steps must be concrete enough that a human or tool can execute them without
          further clarification.
        - NEVER produce a task where ALL steps are real-world user actions with no LLM
          steps at all. If the only work is collecting user input, use Category B
          required_input instead of creating a task.
    - `skill`: which skill to use (if any of the above skills apply)
    - `tools`: which specific tools from that skill are needed
- Category A required_input and dependencies must be fully consistent:
    - Every Category A item in a task's required_input MUST correspond to a dependency
      on the task whose output produces it.
    - Every task dependency MUST justify at least one Category A required_input entry.
    - Never list a Category A item without a producing task in dependencies.
    - Never add a dependency that is not justified by a Category A required_input entry.
    - Tasks with no shared data dependency must run in parallel — do NOT impose
      sequential ordering unless the downstream task actually consumes an upstream output.
    - If a task has no required inputs at all, emit ``required_input: []`` (an empty
      array). Never use placeholder names such as ``"None"``, ``"none"``, ``"null"``,
      or ``"N/A"`` — these are not valid input names and will cause executor errors.

Uncertainty handling:
- If the goal is ambiguous in a way that the clarification fields do not resolve,
  note the assumption you are making in the task description using the prefix
  [ASSUMED: ...]. For example: "[ASSUMED: target audience is technical]
  Write a developer-focused overview of the architecture."
  This signals to the executor and quality gate that the task is operating on
  an assumption that the user may want to correct.

Safety note:
- If any task in your plan involves an irreversible real-world action — sending a
  message or email, posting content publicly, deleting or overwriting files,
  making a payment, deploying to a live environment — ensure that action is named
  explicitly in the task's execution_steps using a real-world execution_type
  (e.g. send_email, post_to_linkedin, delete_file, deploy_service). This allows
  the orchestrator to surface those steps for human confirmation before execution.
  Do not obscure irreversible actions inside vague step descriptions.

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
          "execution_steps": [
            {{
              "execution_type": "search_web",
              "description": "Search for high-return investment options with risk profiles.",
              "produces": "Raw search results listing investment vehicles and their expected returns."
            }},
            {{
              "execution_type": "fetch_url",
              "description": "Fetch the top financial reference pages for detailed data.",
              "produces": "Detailed investment option data including historical performance."
            }},
            {{
              "execution_type": "write_analysis",
              "description": "Synthesise search and page data into a ranked investment options report.",
              "produces": "The investment_options_report document passed to the downstream task."
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
          "execution_steps": [
            {{
              "execution_type": "write_file",
              "description": "Write the formatted investment report markdown file to disk.",
              "produces": "The investment_report.md file saved to the outputs directory."
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


def build_clarification_context_block(fields: list, clarification_prompt: str = "") -> str:
    """
    Render clarification fields into a prompt block for the planner.

    Known and unknown fields are presented in separate sections with explicit
    instructions so the LLM uses known values directly and works around unknowns
    rather than creating tasks to gather the missing information.

    The ``clarification_prompt`` parameter is accepted for backward compatibility
    but is no longer embedded in the output — repeating the full clarification
    prompt inside the planning prompt wastes context without adding information
    the planner does not already have from the field values themselves.

    Parameters
    ----------
    fields               : List of field dicts (key, label, value, rationale).
    clarification_prompt : Unused. Kept for call-site compatibility.
    """
    if not fields:
        return ""

    known = [f for f in fields if f.get("value") and f.get("value") != "unknown"]
    unknown = [f for f in fields if not f.get("value") or f.get("value") == "unknown"]

    lines = ["\nGoal context — treat this as given input, not as open questions to research."]
    lines.append(
        "CRITICAL: Do NOT create tasks to gather, research, or look up any of the "
        "information listed here, whether known or unknown.\n"
    )

    if known:
        lines.append("Known — use these values directly when building the plan:")
        for f in known:
            lines.append(f"  {f.get('label', f.get('key', ''))}: {f['value']}")
        lines.append("")

    if unknown:
        lines.append(
            "Unknown — the user has not provided these values. "
            "Work around their absence using broadened or generalised task descriptions. "
            "Do NOT create tasks whose sole purpose is to find, estimate, or collect "
            "these values — the clarification node handles collection automatically. "
            "Where a value is unknown but a reasonable default exists, you may assume "
            "that default and note it with [ASSUMED: ...] in the task description."
        )
        for f in unknown:
            lines.append(f"  {f.get('label', f.get('key', ''))}: unknown")
        lines.append("")

    lines.append(
        "If you identify additional context that would significantly improve the plan "
        "and is not covered above, add it to additional_clarification_fields using the "
        "same field schema (key, label, value, rationale, hint)."
    )
    return "\n".join(lines) + "\n"


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

1b. GOAL ALIGNMENT — For each task individually, ask: is this task specifically
   working toward the stated goal, or is it targeting a keyword from the goal
   text without serving the actual intent?
   Before accepting a task, ask: "If this task completes successfully, does its
   output bring the plan meaningfully closer to achieving the root goal?"
   Common failure pattern: the goal mentions a concept (e.g. "Python repo") and
   a task researches that concept generically (e.g. "list popular Python repos by
   stars") rather than in the context the goal actually requires (e.g. "find
   repos that are tools for reviewing Python code").
   Rewrite or replace any task whose description or search steps address a
   surface-level keyword rather than the goal's underlying objective.

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

5b. OVER-DECOMPOSITION — check whether the plan has been split into more tasks
   than the goal actually requires. Ask: "Could any two or three adjacent tasks
   be merged into one without losing quality or parallelism?" If yes, merge them
   and update dependencies accordingly. A plan with {max_tasks} tasks for a goal
   that a competent agent could complete in {min_tasks} sessions is worse, not
   better — it adds coordination overhead and increases the chance of context
   loss between tasks.

5c. CIRCULAR DEPENDENCY CHECK — verify that no dependency cycle exists in the
   proposed events. A cycle means task A waits for task B which waits for task A,
   directly or through a chain. If a cycle is present, break it by removing the
   weakest data-flow justification (the dependency with the least concrete output
   flowing across it).

5d. SAFETY CHECK — identify any task whose execution_steps contain an irreversible
   real-world action (send_email, post_to_linkedin, delete_file, deploy_service,
   make_payment, publish_blog_post, etc.). Verify that each such action is labelled
   with an unambiguous real-world execution_type so the orchestrator can surface it
   for human review. If an irreversible action is buried inside a vague step
   description (e.g. "finalise and submit"), rewrite the step to name the action
   explicitly.

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

   Additionally: re-read the original goal text one more time. Ask whether
   the plan as a whole addresses what the user actually wants — not just
   whether the tasks are internally consistent. If the plan solves a well-
   formed problem that is adjacent to but not the same as the stated goal,
   revise the tasks closest to the goal to close that gap.

══════════════════════════════════════════════════════════════════════════════
OUTPUT FORMAT
══════════════════════════════════════════════════════════════════════════════
Return a JSON object with exactly two keys:
  "a_goal_result" — the updated narrative reflecting the final plan
  "events"        — a DELTA array containing ONLY the events you are changing

Delta rules (read carefully before writing output):

ADD_NODE
  • Emit for tasks you are ADDING or REPLACING.
  • Do NOT re-emit tasks that need no changes — they are kept automatically.
  • To fix a task's description, required_input, output, or execution_steps:
    re-emit ADD_NODE with the corrected payload.
  • To change or remove a dependency edge on a node (e.g. task_A should no
    longer depend on task_B): re-emit ADD_NODE for task_A with the corrected
    "dependencies" array.  The merge will automatically discard ALL draft
    ADD_DEPENDENCY events that touch task_A and replace them only with what
    your delta supplies.  You do not need a REMOVE_DEPENDENCY event — it
    does not exist.  Re-emitting the node is the removal mechanism.

ADD_DEPENDENCY
  • Emit only for NEW edges that are not in the draft, or for edges that
    involve a node you are re-emitting.
  • Do NOT re-emit edges that are already correct.

Merge semantics (what happens behind the scenes):
  1. Draft ADD_NODE events are kept unless you re-emit the same node_id.
  2. When you re-emit a node, ALL draft ADD_DEPENDENCY events where that
     node appears as either endpoint are dropped.  Only the ADD_DEPENDENCY
     events in your delta for that node are applied.
  3. Draft ADD_DEPENDENCY edges between nodes you did NOT touch are kept.
  4. Your a_goal_result replaces the draft's.

  • If the draft plan is fully correct on all checks above, return
    "events": []  (an empty array).  Do not reproduce anything.
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
    edges_lines = [f"  {src} waits for {dep}" for src, dep in sorted(edges)]
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
    unknown_fields_context: str = "",
    tool_results_context: str = "",
    broadening_context: str = "",
    upstream_results_context: str = "",
    tool_call_content: str = "",
) -> str:
    """
    Prompt asking the LLM to verify whether a task result satisfies its declared outputs.

    unknown_fields_context   : optional block listing clarification fields that were
                               unknown when the task ran.
    tool_results_context     : optional factual summary of how the task's tool calls
                               fared (all failed / partial / all succeeded).
    broadening_context       : optional block indicating the task ran with a broadened
                               description because specific inputs were unavailable.
                               Tells the verifier to flag any invented specifics.
    upstream_results_context : optional block containing the actual results produced
                               by upstream task dependencies.  Tells the verifier which
                               specific values were legitimately available as inputs so
                               they are not mistakenly flagged as invented.
    tool_call_content        : optional block containing truncated content actually
                               returned by successful tool calls this session.  Lets
                               the verifier cross-check result items against what the
                               tools actually retrieved, catching fabricated values that
                               would otherwise slip past the summary-only tool_results_context.
    """
    unknown_section = ""
    if unknown_fields_context:
        unknown_section = f"""
    MISSING CONTEXT (fields the user did not provide):
    {unknown_fields_context}
"""

    tool_section = ""
    if tool_results_context:
        tool_section = f"""
    TOOL EXECUTION SUMMARY:
    {tool_results_context}
"""

    broadening_section = ""
    if broadening_context:
        broadening_section = f"""
    BROADENED EXECUTION NOTICE:
    {broadening_context}
"""

    upstream_section = ""
    if upstream_results_context:
        upstream_section = f"""
    UPSTREAM TASK RESULTS (data that was available as input to this task):
    {upstream_results_context}
"""

    tool_content_section = ""
    if tool_call_content:
        tool_content_section = f"""
    TOOL CALL CONTENT (actual data returned by successful tool calls this session):
    {tool_call_content}
"""

    return f"""You are verifying whether a task result satisfies its declared outputs.

    TASK
    ID:          {node_id}
    Description: {description}

    DECLARED OUTPUTS (what this task was supposed to produce):
    {outputs_text}
    {unknown_section}{tool_section}{broadening_section}{upstream_section}{tool_content_section}
    ACTUAL RESULT:
    {result}

    Does the result contain actual substantive content, or is it just a label/filename/stub?
    A result that is just a filename, a single word, or a name matching the output label
    is NOT satisfied — the result must contain the actual data.

    If MISSING CONTEXT is listed above: check whether the result invents specific values
    for those fields (e.g. exact figures, names, or percentages not present in any upstream
    result). Invented specifics for unknown fields should be marked as not satisfied.

    If TOOL EXECUTION SUMMARY indicates that all searches failed or returned no results:
    check whether the result asserts specific data (figures, names, statistics) that could
    only have come from a successful search. If so, mark as not satisfied — the result
    is likely fabricated from prior knowledge rather than retrieved data.

    If BROADENED EXECUTION NOTICE is present: this task ran without its specific required
    inputs, using a generalised goal instead. The result must therefore be general in
    nature — templates, frameworks, guides, or ranges. If the result contains specific
    invented values (exact percentages, named achievements, specific figures) that could
    only come from the user's private information or a successful targeted search, mark
    as not satisfied. Generic guidance without invented specifics is acceptable.

    If UPSTREAM TASK RESULTS is present: specific values in the result that match or are
    directly derived from that data are legitimate — they are not invented. Only flag
    specifics as invented if they cannot be traced to the upstream data, clarification
    fields, or a successful tool call.

    If TOOL CALL CONTENT is present: cross-check the result's specific items against what
    actually appeared in those tool call outputs. If the result contains specific values
    (names, identifiers, statistics) that do not appear anywhere in the tool call content
    and cannot be traced to upstream results or clarification fields, mark as not satisfied —
    those values are likely fabricated from the model's prior knowledge.

    PARTIAL RESULTS:
    If the result is labelled [BEST EFFORT] by the executor (indicating incomplete
    information), evaluate it by a lower bar: check that the partial result is genuinely
    useful rather than empty or misleading, and that the executor has clearly stated what
    is missing. A well-labelled partial result that honestly describes its gaps should be
    marked satisfied — do not force a retry on a result that has already accurately
    described its own limits. An unlabelled result that appears to be partially fabricated
    should still be marked not satisfied.

    FORMAT CORRECTNESS:
    If the task's declared output type is 'json' or 'code', check that the result contains
    valid content of that type (parseable JSON, syntactically plausible code). A result
    that claims to be JSON but is wrapped in prose explanation, or code that is truncated
    mid-function, should be marked not satisfied. For other output types (document, text,
    list, data), format is not strictly checked — focus on content completeness.

    USER GOAL ALIGNMENT:
    Beyond checking declared outputs, ask whether the result would actually help the user
    make progress on the overall plan. A result that is technically complete (all declared
    outputs present, no fabrication) but clearly off-topic or misaligned with what a
    downstream task needs should be marked not satisfied, with a reason that names the
    alignment gap. Apply this check conservatively — only when the result clearly serves
    a different purpose than what the task description intended.

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
- First determine the gap type:
    STRUCTURAL GAP: no upstream task was ever planned to produce this input.
      → Propose a bridging task that produces it.
    QUALITY GAP: an upstream task ran but its result is empty, failed, or
      insufficient for this task's needs.
      → Propose a repair or retry of the upstream task rather than a new task.
- If you flag a gap, propose ONE bridging or repair task. Keep it substantial:
  a bridging task should do meaningful work (research, synthesis, writing, analysis),
  not a trivial lookup or a pass-through.
- The bridging task description must be specific enough for an executor to run
  without further clarification. Vague descriptions like "gather more information"
  are not acceptable — name what information, from what source, for what purpose.
- If the task is a root task or the upstream results are sufficient, set ok=true.

Respond only in JSON matching the schema.
"""


# ---------------------------------------------------------------------------
# Executor pre-flight: awaiting-input check prompt
# ---------------------------------------------------------------------------


def build_awaiting_input_check_prompt(
    *,
    node_id: str,
    description: str,
    tools_text: str,
    known_fields_text: str,
    unknown_fields_text: str,
    required_input_text: str = "  (none declared)",
    previous_failure: str = "",
) -> str:
    """
    Prompt asking the LLM whether a task can be executed with the currently
    available information and tools.

    The executor calls this before starting the main tool loop.  The model
    must decide whether the task can produce a useful result using:
      a) the available tools (web search, file ops, etc.), and/or
      b) general LLM reasoning and public knowledge.

    A task should be marked blocked ONLY when it genuinely requires specific
    private information that:
      - cannot be retrieved by any available tool, AND
      - has not been provided in the known clarification fields.

    Parameters
    ----------
    node_id             : Task node identifier.
    description         : What this task is supposed to do and produce.
    tools_text          : Summary of available tools (empty string if none).
    known_fields_text   : Clarification fields with user-provided values,
                          formatted as  key (label): value
    unknown_fields_text : Clarification fields still marked unknown,
                          formatted as  key (label): unknown
    required_input_text : The task's declared required_input items — inputs
                          it explicitly needs from upstream or the user.
    previous_failure    : Verification failure reason from the last execution,
                          if this task was retried after a failed result.
                          When present, the broadened_description must be
                          redesigned to avoid the same failure.
    """
    known_section = (
        f"Known context (provided by user):\n{known_fields_text}"
        if known_fields_text
        else "  (no clarification context was provided)"
    )
    unknown_section = (
        f"Unknown context (user has not yet filled in):\n{unknown_fields_text}"
        if unknown_fields_text
        else "  (all clarification fields are filled)"
    )
    tools_section = (
        f"Available tools:\n{tools_text}"
        if tools_text
        else "  NO TOOLS — task must be completed from general knowledge only."
    )
    failure_section = (
        f"\nPREVIOUS ATTEMPT FAILED:\n"
        f"The last time this task ran with a broadened description, the quality "
        f"gate rejected the result with this reason:\n"
        f'  "{previous_failure}"\n'
        f"Your new broadened_description MUST be designed to avoid this failure. "
        f"Think carefully about what type of output the quality gate would accept "
        f"given the above reason, and write the broadened description accordingly.\n"
        if previous_failure
        else ""
    )

    return f"""You are deciding whether a task can be executed right now.

TASK
  ID:          {node_id}
  Description: {description}

REQUIRED INPUTS (what this task explicitly needs to produce a useful result):
{required_input_text}

{tools_section}

CLARIFICATION CONTEXT
{known_section}

{unknown_section}
{failure_section}
════════════════════════════════════════
PRE-CHECK — ENTITY IDENTITY
════════════════════════════════════════
Before applying the decision rules below, ask:

"Does this task need to research or gather information about a SPECIFIC named
entity — a particular company, organisation, person, product, or location —
rather than producing a general answer from public knowledge?"

If YES:
  a) Entity is in Known context        → proceed; the task can run.
  b) Entity identity field is Unknown  → block; list that field in missing_fields.
     (Identity fields: company_name, employer_name, person_name, organisation_name,
      location, product_name. Attribute fields such as company_budget or company_
      culture do NOT identify the entity — knowing a budget does not tell you which
      company to search for.)
  c) No identity field exists at all   → block; add an appropriate identity field to
     new_fields (e.g. company_name) so the user is prompted.

This pre-check applies regardless of what the REQUIRED INPUTS list says.

Exception: if the subject of the task is a named public entity (a public company,
a public figure, a published work, an open-source project, a government body) AND
the required information is in verifiable public records, the task can proceed —
public information is retrievable by tool even without a user-supplied value.
Apply this exception conservatively; when in doubt, treat the entity as private.

Examples that trigger this check:
  "Gather information about the company's budget and culture" — needs company_name.
  "Research [employer]'s salary bands"                       — needs employer_name.

Examples that do NOT trigger this check:
  "Research average salary ranges for software engineers" — general public data.
  "Write a negotiation framework based on company culture" — general advice.

If YES and the entity is identified → proceed to DECISION RULES.
If YES and the entity is missing    → block immediately; skip DECISION RULES.
If NO                               → proceed to DECISION RULES.

════════════════════════════════════════
DECISION RULES
════════════════════════════════════════

1. Mark blocked=false (can proceed) when the task can produce a useful result using:
   - The available tools (e.g. web search can find salary benchmarks, company news,
     market data, best practices, or any public information), OR
   - General LLM reasoning (e.g. writing frameworks, strategy advice, templates,
     explanations of concepts).
   A task that uses a personal value (like job title) as a search parameter for
   public data is NOT blocked — it can search with a general or approximate term.

2. Mark blocked=true ONLY when the task's PRIMARY OUTPUT is private personal data
   that cannot exist anywhere except in the user's own knowledge or records.
   This rule applies to tasks whose core job is to surface information FROM the
   user themselves — not tasks that happen to accept personal input to refine a
   public-knowledge result.

   Blocked by this rule (the output IS private personal data that only the user has):
     - "Identify/list the user's key achievements or contributions"
     - "Summarise the user's performance review history"
     - "Describe the user's employment or career history"
     - "List the user's personal goals, values, or preferences"
     - "Record the user's salary history or financial situation"

   NOT blocked by this rule (output is public knowledge; personal input just refines it):
     - "Research market salary ranges for the position" — salary benchmarks are
       public; job title is a search parameter, not the output itself.
     - "Determine negotiation strategy based on company culture and personal style"
       — negotiation frameworks are public knowledge; produce a general strategy.
     - "Gather company budget and culture information" — researchable publicly, or
       note that the company name is needed if it is unknown.
     - "Draft a negotiation script" — can be drafted from general templates.

   When applying this rule, give the reason as: "This task's output is private
   personal information that only the user can provide — no tool can retrieve it."

3. Mark blocked=true when a REQUIRED INPUT listed above corresponds to an unknown
   clarification field AND genuinely cannot be substituted by web search or general
   knowledge — meaning the task output would be meaningless or misleading without it.
   Do NOT apply this rule to tasks where a general or approximate answer is still
   useful even without the specific value.

4. Do NOT block tasks that deal with general topics where a useful answer is
   possible without user-specific details:
   - "Determine negotiation strategy based on company culture and personal style"
     → not blocked; produce general negotiation advice from public knowledge.
   - "Research average salary ranges for a role" → not blocked; search for
     general benchmarks (but DO apply the entity pre-check if a specific company
     is needed).
   - "Draft a negotiation script" → not blocked; draft from general templates.

5. MANDATORY: When blocked=true you MUST populate EITHER missing_fields OR
   new_fields (or both) — never leave both empty.
   - missing_fields: EXACT KEYS from the Unknown context above that would unblock
     this task if the user filled them in. The key appears before the parentheses
     (e.g. for "job_title (Job title): unknown" the key is "job_title"). Never use
     the label — always use the key exactly as written.
   - new_fields: new clarification fields to add when no existing field covers the
     required information. Use snake_case keys.
   If you cannot identify any specific field that would unblock the task, set
   blocked=false instead — a block with no actionable fields is never useful.

6. Your reason string must be plain English with no surrounding quotes, no trailing
   apostrophe-comma sequences, and no string-delimiter characters. End with a period.

7. PARTIAL CONFIDENCE — if the task can proceed (blocked=false) but will produce a
   lower-quality result because one or more unknown fields would have refined it,
   note this in your reason string:
     "Can proceed with general approach; result quality would improve if
      [field_name] were provided."
   This allows the quality gate to apply appropriate leniency when verifying the
   result against its declared outputs.

8. BROADENED DESCRIPTION — required whenever blocked=true:
   Produce a broadened_description: a complete, standalone rephrasing of the task
   goal that can be executed immediately using only what IS currently known, without
   depending on any missing field.

   Rules for the broadened description:
   - It must be a full task description the executor can use verbatim, not a note
     about what was generalised (e.g. NOT "generalised version of: ..." but rather
     a direct instruction like "Research general salary benchmarks for mid-level
     technology roles and produce a summary of typical ranges by experience level.")
   - It must not mention any missing field by name or reference the user's private
     information.
   - It should be as specific as possible given the known context — if some fields
     ARE known, incorporate them. Only broaden away from what is missing.
   - It should still produce genuinely useful output that helps the user's goal,
     not a trivial placeholder.

   Also produce broadened_output: a revised list of output declarations that match
   the broadened_description. The output names, types, and descriptions must be
   consistent with what the broadened goal actually produces — not what the original
   goal would have produced. Use the same {{name, type, description}} shape as the
   planner, and only use these allowed types:
     file, document, data, list, url, text, json, code

   The broadened_output must be coherent with broadened_description:
   - If broadened_description produces a template or guide → use type "document"
   - If broadened_description produces a structured list of questions → use type "list"
   - If broadened_description produces research findings → use type "document"
   - If broadened_description produces general advice text → use type "text"

   Also produce broadened_steps: a revised execution_steps list that matches the
   broadened_description. This is REQUIRED when blocked=true — never leave it empty.
   Each step must use the same {{execution_type, description, produces}} shape.

   Rules for broadened_steps:
   - Adapt each original step so it works without the missing context. Keep the same
     execution_type values where possible; only change description and produces text
     to reflect the generalised goal.
   - If the original task had a search_web step for a specific entity that is now
     unknown, change it to search for general information on that topic instead.
   - The steps must be coherent with broadened_description — they are the concrete
     actions needed to produce the broadened_output.
   - Must have at least one step. A broadened_description with no steps is invalid.
   - execution_type MUST be one of the canonical values listed below. Never use a
     tool name (e.g. "web_search", "fetch_url") as an execution_type — those are
     tool identifiers, not step types. Using a tool name will cause the step to be
     misrouted and the node will stall waiting for user input.

     Valid execution_type values:
       "search_web"      — retrieve information from the web
       "fetch_url"       — fetch the contents of a specific URL
       "write_plan"      — produce a structured plan or outline
       "write_document"  — produce a written document or report
       "write_analysis"  — produce an analytical write-up
       "write_code"      — produce executable code
       "analyse_data"    — analyse structured data
       "summarise"       — summarise existing content
       "synthesise"      — combine multiple inputs into a unified output

   Examples of matching broadened_description + broadened_steps:
   - Original desc:   "Research Acme Corp's GitHub repositories."
     Original steps:  [{{execution_type: "search_web", description: "Search GitHub for Acme Corp repos"}}]
     Missing: company_name
     Broadened desc:  "Research general best practices for evaluating open-source GitHub repositories."
     Broadened steps: [{{execution_type: "search_web",
                         description: "Search for criteria used to evaluate open-source GitHub repositories.",
                         produces: "List of evaluation criteria and metrics for assessing repo quality."}}]

   - Original desc:   "Identify the user's key achievements."
     Original steps:  [{{execution_type: "write_plan", description: "Extract achievements from user history"}}]
     Missing: personal history
     Broadened desc:  "Produce a structured template to help a user identify their achievements."
     Broadened steps: [{{execution_type: "write_document",
                         description: "Write a structured achievements template with guided prompts.",
                         produces: "Template document the user can fill in with their own achievements."}}]

   Examples of matching broadened_description + broadened_output:
   - Original desc:   "Identify the user's key achievements."
     Original output: [{{name: "personal_achievements_list", type: "list", ...}}]
     Missing: personal history
     Broadened desc:  "Produce a structured template and guided questions that help
                       a user identify and articulate their professional achievements."
     Broadened output: [{{name: "achievements_template", type: "document",
                         description: "Template with guided questions and placeholder
                         sections for a user to fill in their own achievements."}}]

   - Original desc:   "Gather information about Acme Corp's budget and culture."
     Original output: [{{name: "company_report", type: "document", ...}}]
     Missing: company_name
     Broadened desc:  "Research general factors that affect company budget constraints
                       and workplace culture in salary negotiation contexts."
     Broadened output: [{{name: "company_factors_guide", type: "document",
                         description: "Guide covering typical company budget cycles,
                         raise feasibility signals, and culture indicators relevant
                         to salary negotiation."}}]

   When blocked=false, set broadened_description, broadened_for_missing,
   broadened_output, and broadened_steps to empty strings/arrays.

Respond only in JSON matching the schema.
"""


# ---------------------------------------------------------------------------
# Broadened description fallback prompt
# ---------------------------------------------------------------------------


def build_broadened_description_prompt(
    *,
    node_id: str,
    original_description: str,
    missing_keys: list,
    known_fields_text: str,
    original_steps: list | None = None,
) -> str:
    """
    Focused prompt used as a fallback when the preflight LLM call returned
    blocked=true but an empty broadened_description.

    Asks the model for a self-contained rephrasing of the task goal and
    adapted execution steps that work with only the currently known context.

    Parameters
    ----------
    node_id              : Task node identifier.
    original_description : The task's original description.
    missing_keys         : List of field keys that are unavailable.
    known_fields_text    : Clarification fields with user-provided values,
                           formatted as  key (label): value
    original_steps       : The task's original execution_steps list, used
                           as the basis for generating broadened_steps.
    """
    missing_text = ", ".join(missing_keys) if missing_keys else "(none listed)"
    known_section = (
        f"Currently known context:\n{known_fields_text}"
        if known_fields_text
        else "  (no context is currently available)"
    )
    steps = original_steps or []
    if steps:
        steps_lines = "\n".join(
            f"  {i + 1}. [{s.get('execution_type', '?')}] {s.get('description', '')} → {s.get('produces', '')}"
            for i, s in enumerate(steps)
        )
        steps_section = f"ORIGINAL EXECUTION STEPS:\n{steps_lines}"
    else:
        steps_section = "ORIGINAL EXECUTION STEPS: (none declared)"

    return f"""A task needs to be rephrased so it can execute without certain unavailable inputs.

ORIGINAL TASK
  ID:          {node_id}
  Description: {original_description}

{steps_section}

UNAVAILABLE INPUTS: {missing_text}

{known_section}

MINIMUM QUALITY FLOOR:
Before writing the broadened description, ask: "Is there any version of this task
that produces genuinely useful output given only the known context above?" If the
answer is no — if the task is so dependent on the missing inputs that any broadened
version would be empty or trivially unhelpful — set broadened_description to an
empty string and broadened_steps to an empty array. The system will then skip this
task until the user provides the missing information rather than executing a
placeholder task.

Produce two things:

1. broadened_description: a complete, standalone rephrasing of this task
   that produces genuinely useful output using ONLY the known context above.
   Rules:
   - Write a direct task instruction, not a description of what you are doing.
   - Do not mention the unavailable inputs by name.
   - Incorporate any known context to make the goal as specific as possible.
   - If the task requires personal information the user must supply, produce a
     template or structured guide instead.
   - The result must be useful to downstream tasks even without the missing data.

2. broadened_steps: a revised execution_steps list that matches the
   broadened_description. Adapt each original step so it works without
   the missing context. Keep the same execution_type values where possible;
   only change the description and produces text.
   Each step: {{execution_type, description, produces}}

Respond only in JSON matching the schema.
"""


# ---------------------------------------------------------------------------
# Clarification generation prompt
# ---------------------------------------------------------------------------


def build_clarification_prompt(
    goal_text: str,
    skills_summary: str = "",
    min_fields: int = 3,
    max_fields: int = 8,
) -> str:
    """
    Build the prompt for the first LLM call that generates the clarification
    node fields — structured context that would most improve the plan.

    The prompt is also stored in the clarification node's metadata so the
    planner can see exactly what was asked when it adds extra fields.

    Parameters
    ----------
    goal_text      : The raw goal string provided by the user.
    skills_summary : The same skills/tools summary injected into the planner
                     prompt (from ``SkillLoader.prompt_summary``).  When
                     provided, the LLM is instructed not to ask the user for
                     information that those tools can retrieve autonomously.
    min_fields     : Minimum number of fields to return (from config).
    max_fields     : Maximum number of fields to return (from config).
    """
    tool_awareness_block = ""
    if skills_summary.strip():
        tool_awareness_block = f"""
The plan will have access to the following tools at execution time:

{skills_summary}

Do NOT raise a clarification field for information these tools can retrieve
autonomously (e.g. current market prices, publicly available statistics,
exchange rates, regulatory text).  Only ask for information that is private
to the user or structurally ambiguous — facts no tool can infer without the
user's direct input (e.g. personal residency status, budget constraints,
language preferences, private document contents).
"""

    return f"""\
You are preparing to plan how to achieve the following goal:

  {goal_text}
{tool_awareness_block}
Before planning begins, identify the structured context fields that would most
improve the quality and specificity of the plan.

Follow these two steps in order:

STEP 1 — Extract facts already stated in the goal text.
  Read the goal carefully and pull out every concrete fact the user has already
  provided: numbers, constraints, locations, roles, preferences, deadlines, etc.
  Each extracted fact becomes a field with its value set to what the user stated.
  Examples of what to extract: budget figures, size requirements, hard constraints
  (e.g. "no mortgage"), stated locations, stated roles or residency, named
  technologies, explicit timelines.
  Do NOT mark a fact as "unknown" if it is explicitly stated in the goal text.

STEP 2 — Identify genuinely missing information.
  After capturing the stated facts, determine what additional information —
  if known — would most significantly change what tasks are needed, how they
  should be done, or what the outputs should contain.  Only add fields here
  for information that is truly absent from the goal text and cannot be
  retrieved by the available tools.  Do not ask for information that any
  competent plan could work around or that is irrelevant to the goal.

For every field (from both steps):
  - Set value to the extracted or assumed value when known.
  - If a field's value can be reasonably inferred from the nature and intent
    of the goal — even if not stated word-for-word — set it to that inferred
    value rather than "unknown". For example, if the goal is about reviewing
    a Python repository, the package_type is clearly "code quality and static
    analysis tools", not "unknown". Use your judgment to supply a concrete
    inferred value whenever the goal makes the answer apparent.
  - Set value to "unknown" ONLY when the information is private to the user
    (e.g. personal details, private credentials, internal company specifics)
    or is completely unresolvable from the goal text and the available tools.
  - Never leave value blank.
  - For any field whose label alone might leave a user unsure what to type,
    add a hint with 2–3 concrete examples that show the expected format and
    scope. Phrase it as "e.g. …" and keep it under 15 words. Focus on
    examples, not on repeating the rationale. Fields with obvious single-word
    answers (like "Programming Language: Python") do not need a hint.
    Examples of good hints:
      package_type  → "e.g. static analysis tools, test frameworks, linters"
      budget        → "e.g. £50,000–£60,000, or leave blank if unsure"
      target_audience → "e.g. senior engineers, junior devs, non-technical stakeholders"

Field prioritisation:
  - Fields whose absence would make the entire plan significantly less useful or
    force the broadened-execution fallback should be marked with a (*) at the
    start of their label. Limit (*) fields to at most 3. All other fields are
    optional — the user may leave them blank without preventing execution.
  - Order fields so the most impactful (*) fields appear first.
  - Aim for the minimum number of fields that would genuinely improve the plan.
    Do not ask for information the plan could reasonably assume or that the
    available tools can retrieve. Fewer, higher-quality fields are better than
    many vague ones.

Always include a final field with:
  key:      "additional_context"
  label:    "Anything else I should know?"
  value:    "unknown"
  rationale: "Free-form context that does not fit the fields above."
  hint:     "e.g. deadlines, hard constraints, preferences, or background the plan should know about"

Return between {min_fields} and {max_fields} fields total (including the
additional_context field). Fields marked (*) significantly improve plan quality
when filled in. All other fields are optional — leaving them blank will cause
the executor to proceed with a generalised approach for that task, which is
acceptable.
Respond only in JSON matching the schema.
"""
