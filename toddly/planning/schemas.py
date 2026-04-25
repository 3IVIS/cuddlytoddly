# toddly/planning/schemas.py
#
# Contains only EXECUTION_TURN_SCHEMA, which is used by the generic LLM
# backends (FileBasedLLM, LlamaCppLLM) as their default structured-output
# schema for JSON-mode responses.
#
# All other schemas (PLAN_SCHEMA, RESULT_VERIFICATION_SCHEMA, etc.) live in
# cuddlytoddly/planning/schemas.py as they are specific to the cuddlytoddly
# domain (clarification nodes, broadening, quality gate, etc.).

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
