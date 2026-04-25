"""
cuddly.engine.signals
~~~~~~~~~~~~~~~~~~~~~~~~~
Lightweight signal objects passed between the orchestrator and the executor.

Keeping them here (rather than in planning/llm_executor.py) breaks the
upward dependency that would otherwise force cuddly to import from the
cuddlytoddly application package.
"""

from dataclasses import dataclass
from dataclasses import field as _dc_field


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
    broadened_steps: list = _dc_field(default_factory=list)
