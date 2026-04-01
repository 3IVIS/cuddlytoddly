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
                        "required": ["node_id", "node_type", "dependencies", "metadata"],
                        "additionalProperties": False,
                        "properties": {
                            "node_id":      {"type": "string"},
                            "node_type":    {
                                "type": "string",
                                "enum": ["task", "goal", "reflection"],
                            },
                            "dependencies": {
                                "type": "array",
                                "items": {"type": "string"},
                            },
                            "metadata": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "description":      {"type": "string"},
                                    "parallel_group":   {"type": ["string", "null"]},
                                    "required_input":   {
                                        "type": "array",
                                        "items": _IO_ITEM,
                                    },
                                    "output":           {
                                        "type": "array",
                                        "items": _IO_ITEM,
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
                            "node_id":    {"type": "string"},
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
            "items": EVENT_LIST_SCHEMA["items"],   # reuses item definitions above
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
        "needs_refinement":  {"type": "boolean"},
        "tasks_to_expand":   {"type": "array", "items": {"type": "string"}},
        "validated_atomic":  {"type": "array", "items": {"type": "string"}},
        "dependency_issues": {"type": "array", "items": {"type": "string"}},
        "reasoning":         {"type": "string"},
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
            "description": (
                "True if this is the final answer, False if a tool call is needed."
            ),
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
                    "additionalProperties": {"type": "string"},
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
            "description": (
                "Short description of what is missing. Only required when ok=false."
            ),
        },
        "bridge_node": {
            "type": "object",
            "description": (
                "A single task that would close the gap. Only required when ok=false."
            ),
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