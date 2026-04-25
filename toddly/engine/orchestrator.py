# toddly/engine/orchestrator.py
#
# Simple concrete orchestrator for standalone toddly use.
# All domain-specific logic (clarification nodes, broadening, goal
# auto-completion, awaiting_input resumption) lives in
# cuddlytoddly/engine/orchestrator.py, which subclasses BaseOrchestrator.

from toddly.engine.base_orchestrator import BaseOrchestrator

Orchestrator = BaseOrchestrator
