# planning/schemas.py
#
# Single source of truth for every JSON schema used by the LLM backends.
#
# Schemas are imported by:
#   llm_interface.py         — EVENT_LIST_SCHEMA, PLAN_SCHEMA, GOAL_SUMMARY_SCHEMA,
#                               REFINER_OUTPUT_SCHEMA
#   llm_executor.py          — EXECUTION_TURN_SCHEMA
#   quality_gate.py          — RESULT_VERIFICATION_SCHEMA, DEPENDENCY_CHECK_SCHEMA
#   plan_constraint_checker  — GHOST_NODE_RESOLUTION_SCHEMA
#
# Edit the schemas here to change the structured-output contract with the LLM.

# ---------------------------------------------------------------------------
# Shared sub-schema: typed I/O item
# Used inside EVENT_LIST_SCHEMA (required_input / output arrays).
# ---------------------------------------------------------------------------

_EXECUTION_STEP_ITEM = {
    "type": "object",
    "required": ["execution_type", "description", "produces"],
    "additionalProperties": False,
    "properties": {
        "execution_type": {
            "type": "string",
            "description": (
                "Concrete verb-noun identifier for what this step does. "
                "Use tool names for LLM-executable steps (search_web, fetch_url, "
                "write_file, run_code, analyse_data, write_plan, write_document). "
                "Use real-world action names for steps requiring user execution "
                "(post_to_reddit, post_to_linkedin, send_email, send_newsletter, "
                "publish_blog_post, run_terminal_command, deploy_service, "
                "open_pull_request, schedule_meeting). "
                "Be specific — not 'share' but 'post_to_reddit'."
            ),
        },
        "description": {
            "type": "string",
            "description": "One sentence: exactly what this step does in this task context.",
        },
        "produces": {
            "type": "string",
            "description": (
                "One sentence: what this step contributes toward the node's declared output. "
                "E.g. 'Provides the list of target subreddits for the posting step.'"
            ),
        },
    },
}

_IO_ITEM = {
    "type": "object",
    "required": ["name", "type", "description"],
    "additionalProperties": False,
    "properties": {
        "name": {
            "type": "string",
            "description": "Short snake_case identifier, e.g. 'investment_report'",
        },
        "type": {
            "type": "string",
            "enum": ["file", "document", "data", "list", "url", "text", "json", "code"],
            "description": "What kind of artifact this is",
        },
        "description": {
            "type": "string",
            "description": "One sentence: what this artifact contains",
        },
    },
}

# ---------------------------------------------------------------------------
# Planning schemas
# ---------------------------------------------------------------------------

EVENT_LIST_SCHEMA = {
    "type": "array",
    "items": {
        "oneOf": [
            {
                "type": "object",
                "title": "ADD_NODE event",
                "required": ["type", "payload"],
                "additionalProperties": False,
                "properties": {
                    "type": {"type": "string", "const": "ADD_NODE"},
                    "payload": {
                        "type": "object",
                        "required": [
                            "node_id",
                            "node_type",
                            "dependencies",
                            "metadata",
                        ],
                        "additionalProperties": False,
                        "properties": {
                            "node_id": {"type": "string"},
                            "node_type": {
                                "type": "string",
                                "enum": ["task", "goal", "reflection", "clarification"],
                            },
                            "dependencies": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "metadata": {
                                "type": "object",
                                "required": ["description", "output", "execution_steps"],
                                "additionalProperties": False,
                                "properties": {
                                    "description": {"type": "string"},
                                    "parallel_group": {"type": ["string", "null"]},
                                    "required_input": {
                                        "type": "array",
                                        "items": _IO_ITEM,
                                    },
                                    "output": {
                                        "type": "array",
                                        "items": _IO_ITEM,
                                    },
                                    "execution_steps": {
                                        "type": "array",
                                        "description": (
                                            "Ordered list of concrete steps the executor "
                                            "must work through to complete this task. "
                                            "Each step has a specific execution_type that "
                                            "determines whether the LLM can perform it "
                                            "or whether the user must act."
                                        ),
                                        "items": _EXECUTION_STEP_ITEM,
                                    },
                                    "reflection_notes": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                    "precedes": {
                                        "type": "array",
                                        "items": {"type": "string"},
                                    },
                                },
                            },
                        },
                    },
                },
            },
            {
                "type": "object",
                "title": "ADD_DEPENDENCY event",
                "required": ["type", "payload"],
                "additionalProperties": False,
                "properties": {
                    "type": {"type": "string", "const": "ADD_DEPENDENCY"},
                    "payload": {
                        "type": "object",
                        "required": ["node_id", "depends_on"],
                        "additionalProperties": False,
                        "properties": {
                            "node_id": {"type": "string"},
                            "depends_on": {"type": "string"},
                        },
                    },
                },
            },
        ]
    },
}

PLAN_SCHEMA = {
    "type": "object",
    "required": ["a_goal_result", "events"],
    "additionalProperties": False,
    "properties": {
        "a_goal_result": {
            "type": "string",
            "description": (
                "2-4 sentences explaining how these specific tasks chain together "
                "to achieve the goal. Name each task, what it produces, and why "
                "the next task depends on that output. Make the dependency "
                "reasoning explicit."
            ),
        },
        "events": {
            "type": "array",
            "items": EVENT_LIST_SCHEMA["items"],  # reuses item definitions above
        },
        "additional_clarification_fields": {
            "type": "array",
            "description": (
                "Optional extra fields the planner wants to add to the clarification node. "
                "Same schema as clarification fields: key, label, value, rationale."
            ),
            "items": {
                "type": "object",
                "required": ["key", "label", "value", "rationale"],
                "additionalProperties": False,
                "properties": {
                    "key": {"type": "string", "description": "snake_case identifier"},
                    "label": {
                        "type": "string",
                        "description": "Human-readable question",
                    },
                    "value": {
                        "type": "string",
                        "description": "Best-guess answer or 'unknown'",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "One sentence: why this matters",
                    },
                },
            },
        },
    },
}

GOAL_SUMMARY_SCHEMA = {
    "type": "object",
    "required": ["description", "plan_summary"],
    "additionalProperties": False,
    "properties": {
        "description": {
            "type": "string",
            "description": (
                "One sentence (max 20 words) naming what this goal achieves. "
                "Used as the node label in the UI."
            ),
        },
        "plan_summary": {
            "type": "string",
            "description": (
                "2-4 sentences explaining how the planned tasks combine to "
                "achieve the goal. Cover what each task produces and how the "
                "outputs chain together into the final result."
            ),
        },
    },
}

REFINER_OUTPUT_SCHEMA = {
    "type": "object",
    "required": [
        "needs_refinement",
        "tasks_to_expand",
        "validated_atomic",
        "dependency_issues",
        "reasoning",
    ],
    "additionalProperties": False,
    "properties": {
        "needs_refinement": {"type": "boolean"},
        "tasks_to_expand": {"type": "array", "items": {"type": "string"}},
        "validated_atomic": {"type": "array", "items": {"type": "string"}},
        "dependency_issues": {"type": "array", "items": {"type": "string"}},
        "reasoning": {"type": "string"},
    },
}

# ---------------------------------------------------------------------------
# Execution schema
# ---------------------------------------------------------------------------

EXECUTION_TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "done": {
            "type": "boolean",
            "description": ("True if this is the final answer, False if a tool call is needed."),
        },
        "result": {
            "type": "string",
            "description": (
                "The final result text. Required when done=true. "
                "Must be a self-contained, detailed description of what was produced — "
                "it will be passed verbatim to downstream tasks as their input."
            ),
        },
        "tool_call": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "args": {
                    "type": "object",
                    # Allow any JSON value type (string, number, boolean, array,
                    # object, null).  The previous {"type": "string"} constraint
                    # silently prevented tools from receiving integer, boolean, or
                    # array arguments.
                    "additionalProperties": True,
                },
            },
            "required": ["name", "args"],
            "description": "Tool to call. Required when done=false.",
        },
    },
    "required": ["done"],
}

# ---------------------------------------------------------------------------
# Quality-gate schemas
# ---------------------------------------------------------------------------

RESULT_VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "satisfied": {
            "type": "boolean",
            "description": (
                "True if the result fully covers every declared output. "
                "False if something is missing or clearly wrong."
            ),
        },
        "reason": {
            "type": "string",
            "description": (
                "One sentence explaining why the result is satisfied or not. "
                "If satisfied=true this can be brief."
            ),
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
            ),
        },
        "missing": {
            "type": "string",
            "description": ("Short description of what is missing. Only required when ok=false."),
        },
        "bridge_node": {
            "type": "object",
            "description": ("A single task that would close the gap. Only required when ok=false."),
            "properties": {
                "node_id": {
                    "type": "string",
                    "description": "Snake_case identifier, no spaces.",
                },
                "description": {
                    "type": "string",
                    "description": "One sentence: what this task does.",
                },
                "output": {
                    "type": "string",
                    "description": "The single artifact this task produces.",
                },
            },
            "required": ["node_id", "description", "output"],
        },
    },
    "required": ["ok"],
}


# ---------------------------------------------------------------------------
# Clarification generation schema
# ---------------------------------------------------------------------------

CLARIFICATION_GENERATION_SCHEMA = {
    "type": "object",
    "required": ["fields"],
    "additionalProperties": False,
    "properties": {
        "fields": {
            "type": "array",
            "description": "Structured context fields that would most improve the plan.",
            "items": {
                "type": "object",
                "required": ["key", "label", "value", "rationale"],
                "additionalProperties": False,
                "properties": {
                    "key": {
                        "type": "string",
                        "description": "snake_case machine-readable identifier, e.g. 'current_salary'",
                    },
                    "label": {
                        "type": "string",
                        "description": "Human-readable question, e.g. 'What is your current salary?'",
                    },
                    "value": {
                        "type": "string",
                        "description": (
                            "Best-guess answer as a string, or 'unknown' if no reasonable "
                            "guess can be made. Never leave blank — always use 'unknown'."
                        ),
                    },
                    "rationale": {
                        "type": "string",
                        "description": "One sentence explaining why this information would significantly improve the plan.",
                    },
                },
            },
        },
    },
}

# ---------------------------------------------------------------------------
# Executor pre-flight: awaiting-input check schema
# ---------------------------------------------------------------------------

AWAITING_INPUT_CHECK_SCHEMA = {
    "type": "object",
    "required": [
        "blocked",
        "reason",
        "broadened_description",
        "broadened_for_missing",
        "broadened_output",
        "broadened_steps",
    ],
    "additionalProperties": False,
    "properties": {
        "blocked": {
            "type": "boolean",
            "description": (
                "True if this task is missing required inputs. "
                "Even when blocked=true the task will still execute — using the "
                "broadened_description instead of the original goal. "
                "False if all required inputs are available and the original "
                "task description can be used directly."
            ),
        },
        "reason": {
            "type": "string",
            "maxLength": 200,
            "description": (
                "One sentence explaining what inputs are missing, or confirming "
                "the task can proceed as specified."
            ),
        },
        "missing_fields": {
            "type": "array",
            "description": (
                "Keys of existing clarification fields that are currently "
                "unknown but — if filled in by the user — would allow the "
                "original specific task to run. Only include fields that are "
                "directly and specifically required."
            ),
            "items": {"type": "string"},
        },
        "new_fields": {
            "type": "array",
            "description": (
                "New clarification fields to add to the form when no existing "
                "field captures the required information. Leave empty when "
                "an existing field already covers what is needed."
            ),
            "items": {
                "type": "object",
                "required": ["key", "label", "value", "rationale"],
                "additionalProperties": False,
                "properties": {
                    "key": {"type": "string", "description": "snake_case identifier"},
                    "label": {
                        "type": "string",
                        "description": "Human-readable field name shown in the UI",
                    },
                    "value": {
                        "type": "string",
                        "description": "Always 'unknown' for new fields",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "One sentence: why this field is needed",
                    },
                },
            },
        },
        "broadened_description": {
            "type": "string",
            "maxLength": 500,
            "description": (
                "A rephrased version of the task goal that produces genuinely "
                "useful output using ONLY what is currently known, without "
                "depending on any missing fields. "
                "Required whenever blocked=true. "
                "Must be a complete, standalone task description — not a note "
                "about what was generalised. "
                "Must not mention any missing field by name or reference the "
                "user's specific private information. "
                "Should be as specific as possible given the known context "
                "(e.g. if job title is known but company is not, incorporate "
                "the job title into the broadened description). "
                "Leave empty string when blocked=false."
            ),
        },
        "broadened_for_missing": {
            "type": "array",
            "description": (
                "The keys of the fields that were missing when this broadened "
                "description was generated. Used to decide whether to reuse or "
                "regenerate the broadened description on the next execution. "
                "Should match the union of missing_fields keys and new_fields keys. "
                "Leave empty when blocked=false."
            ),
            "items": {"type": "string"},
        },
        "broadened_output": {
            "type": "array",
            "description": (
                "Revised output declarations that match the broadened_description. "
                "When blocked=true, redefine the outputs so they describe what the "
                "broadened goal actually produces (e.g. a template or framework) "
                "rather than the specific personal or entity-specific data the "
                "original goal would produce. "
                "Use the same {name, type, description} shape as the planner output "
                "declarations. Type must be one of the allowed values. "
                "Leave empty when blocked=false."
            ),
            "items": _IO_ITEM,
        },
        "broadened_steps": {
            "type": "array",
            "description": (
                "Revised execution_steps that match the broadened_description. "
                "When blocked=true, produce a step list that works without the "
                "missing context — e.g. replace 'post_to_reddit: post to r/{repo_name}' "
                "with 'write_plan: draft a generic posting template for LLM repos'. "
                "Preserve the same execution_type values where possible; only change "
                "descriptions and produces text to reflect the generalised goal. "
                "Leave empty when blocked=false."
            ),
            "items": _EXECUTION_STEP_ITEM,
        },
    },
}


# ---------------------------------------------------------------------------
# Step execution report schema — structured result emitted by the executor
# ---------------------------------------------------------------------------

STEP_EXECUTION_REPORT_SCHEMA = {
    "type": "object",
    "required": ["step_reports", "overall_result"],
    "additionalProperties": False,
    "properties": {
        "step_reports": {
            "type": "array",
            "description": "One entry per execution step, in the order they were attempted.",
            "items": {
                "type": "object",
                "required": ["execution_type", "status", "evidence"],
                "additionalProperties": False,
                "properties": {
                    "execution_type": {
                        "type": "string",
                        "description": "The execution_type value from the step definition.",
                    },
                    "status": {
                        "type": "string",
                        "enum": ["completed", "awaiting_user", "worked_around", "failed"],
                        "description": (
                            "completed: the step was performed and evidence was recorded. "
                            "awaiting_user: the LLM cannot perform this step; user must act. "
                            "worked_around: step was approximated without the required tool. "
                            "failed: step was attempted but produced an error."
                        ),
                    },
                    "evidence": {
                        "type": "string",
                        "description": (
                            "For completed: brief note of what tool was called or what was produced. "
                            "For awaiting_user: the handoff artifact (drafted text, instructions). "
                            "For worked_around: explanation of the approximation used. "
                            "For failed: the error message."
                        ),
                    },
                },
            },
        },
        "overall_result": {
            "type": "string",
            "description": (
                "The combined substantive result of all completed steps, "
                "formatted so downstream tasks can use it directly. "
                "If some steps are awaiting_user, this contains only the "
                "output of the completed steps so far."
            ),
        },
    },
}


# ---------------------------------------------------------------------------
# Broadened description fallback schema
# ---------------------------------------------------------------------------

BROADENED_DESCRIPTION_SCHEMA = {
    "type": "object",
    "required": ["broadened_description", "broadened_steps"],
    "additionalProperties": False,
    "properties": {
        "broadened_description": {
            "type": "string",
            "description": (
                "A complete, standalone rephrasing of the task goal that "
                "produces useful output using only the currently known context, "
                "without depending on any unavailable inputs. Must be a direct "
                "task instruction, not a meta-description of what was generalised."
            ),
        },
        "broadened_steps": {
            "type": "array",
            "description": (
                "Revised execution_steps that match the broadened_description. "
                "Adapt each step so it works without the missing context. "
                "Use the same {execution_type, description, produces} shape."
            ),
            "items": _EXECUTION_STEP_ITEM,
        },
    },
}


# ---------------------------------------------------------------------------
# Plan constraint checker schema
# ---------------------------------------------------------------------------

GHOST_NODE_RESOLUTION_SCHEMA = {
    "type": "object",
    "required": ["dependent_node_id", "reasoning"],
    "additionalProperties": False,
    "properties": {
        "dependent_node_id": {
            "type": "string",
            "description": (
                "The ID of the node that should depend on the ghost node — "
                "i.e. the node whose work would most naturally consume the "
                "ghost node's output. Must be one of the valid candidates "
                "listed in the prompt."
            ),
        },
        "reasoning": {
            "type": "string",
            "description": (
                "One sentence explaining why this node is the best dependent "
                "for the ghost node's output."
            ),
        },
    },
}
