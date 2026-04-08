# planning/llm_planner.py

import json

from cuddlytoddly.core.events import (
    ADD_DEPENDENCY,
    ADD_NODE,
    SET_RESULT,
)
from cuddlytoddly.infra.logging import get_logger
from cuddlytoddly.planning.llm_output_validator import LLMOutputValidator
from cuddlytoddly.planning.plan_constraint_checker import PlanConstraintChecker
from cuddlytoddly.planning.prompts import (
    build_clarification_context_block,
    build_clarification_prompt,
    build_plan_scrutinizer_prompt,
    build_planner_prompt,
    build_planner_skills_block,
)
from cuddlytoddly.planning.schemas import (
    CLARIFICATION_GENERATION_SCHEMA,
    PLAN_SCHEMA,
)

logger = get_logger(__name__)

_VOLATILE_METADATA_KEYS = {
    "expanded",
    "fully_refined",
    "dependency_reflected",
    "last_commit_status",
    "last_commit_parents",
    "parent_goal",
    "missing_inputs",
    "reflection_notes",
    "coverage_checked",
}


def _clarification_node_id(goal_id: str) -> str:
    return f"clarification_{goal_id}"


class LLMPlanner:
    def __init__(
        self,
        llm_client,
        graph,
        refiner=None,
        skills_summary: str = "",
        min_tasks_per_goal: int = 3,
        max_tasks_per_goal: int = 8,
        scrutinize_plan: bool = False,
    ):
        self.llm = llm_client
        self.graph = graph
        self.refiner = refiner
        self.skills_summary = skills_summary
        self.min_tasks_per_goal = min_tasks_per_goal
        self.max_tasks_per_goal = max_tasks_per_goal
        self.scrutinize_plan = scrutinize_plan
        self.constraint_checker = PlanConstraintChecker(graph, llm_client)

    def propose(self, context):
        snapshot = context.snapshot
        goals = context.goals
        # skip_scrutiny is set to True by the orchestrator on partial replans
        # (goal already had children) so the expensive scrutiny pass is not
        # repeated for what is effectively a plan extension.
        skip_scrutiny = getattr(context, "skip_scrutiny", False)

        if not goals:
            return []

        active_goal = goals[0]
        goal_text = active_goal.metadata.get("description", active_goal.id)
        clarif_id = _clarification_node_id(active_goal.id)

        # ── Clarification node ────────────────────────────────────────────────
        # Call 1: generate clarification fields on first plan only.
        # On partial replans (goal already has children) the clarification node
        # already exists — reuse it so user edits are not overwritten.
        clarif_events: list = []
        clarif_fields: list = []
        clarif_prompt: str = ""

        existing_clarif = self.graph.nodes.get(clarif_id)
        if existing_clarif is not None:
            try:
                clarif_fields = json.loads(existing_clarif.result or "[]")
            except Exception:
                clarif_fields = []
            clarif_prompt = existing_clarif.metadata.get("clarification_prompt", "")
            logger.info("[PLANNER] Reusing existing clarification node %s", clarif_id)
        else:
            clarif_prompt, clarif_fields, clarif_events = self._generate_clarification_node(
                goal_text, active_goal.id, clarif_id
            )

        # ── Call 2: planning ──────────────────────────────────────────────────
        graph_view = self._serialize_snapshot(snapshot)
        prompt = self._build_prompt(
            graph_view,
            goals,
            clarif_fields=clarif_fields,
            clarif_prompt=clarif_prompt,
        )
        llm_output = self.llm.ask(prompt, schema=PLAN_SCHEMA)

        # ── Call 3: scrutiny (skipped on partial replans) ─────────────────────
        if self.scrutinize_plan and not skip_scrutiny:
            llm_output = self._scrutinize(prompt, llm_output)

        try:
            parsed = json.loads(llm_output)
        except Exception as e:
            logger.error("[PLANNER] JSON parse error: %s", e)
            return clarif_events

        goal_result = parsed.get("a_goal_result", "").strip()
        raw_events = parsed.get("events", [])

        # ── Merge extra clarification fields the planner identified ───────────
        extra_fields = parsed.get("additional_clarification_fields", [])
        if extra_fields and isinstance(extra_fields, list):
            added = [
                f
                for f in extra_fields
                if isinstance(f, dict)
                and f.get("key")
                and not any(cf["key"] == f["key"] for cf in clarif_fields)
            ]
            if added:
                clarif_fields = clarif_fields + added
                clarif_events.append(
                    {
                        "type": SET_RESULT,
                        "payload": {
                            "node_id": clarif_id,
                            "result": json.dumps(clarif_fields, ensure_ascii=False),
                        },
                    }
                )
                logger.info(
                    "[PLANNER] Planner added %d extra clarification field(s)",
                    len(added),
                )

        # ── Validate and constrain plan events ────────────────────────────────
        raw_events = self._normalize_events(raw_events)
        validator = LLMOutputValidator(self.graph)
        safe_events = validator.validate_and_normalize(raw_events, forced_origin="planning")
        # Pass clarif_id so the checker knows root tasks will get a dependency
        # on the clarification node even though it is not in this event batch.
        # Without this: root tasks look dependency-free → 6b strips their
        # required_input; if nothing else depends on them they look like ghosts.
        safe_events = self.constraint_checker.check_and_repair(
            safe_events,
            active_goal.id,
            known_dep_id=clarif_id,
        )

        # ── Wire clarification node as dependency of all root task nodes ───────
        # Root tasks: new nodes whose deps don't reference any other new node.
        # IMPORTANT: these events must go AFTER safe_events so task ADD_NODE
        # events have already been applied when the reducer processes these edges.
        # add_dependency() silently no-ops if either node doesn't exist yet.
        new_node_ids = {evt["payload"]["node_id"] for evt in safe_events if evt["type"] == ADD_NODE}
        wiring_events = []
        for evt in safe_events:
            if evt["type"] == ADD_NODE:
                node_id = evt["payload"]["node_id"]
                deps = set(evt["payload"].get("dependencies", []))
                if not (deps & new_node_ids):
                    wiring_events.append(
                        {
                            "type": ADD_DEPENDENCY,
                            "payload": {
                                "node_id": node_id,
                                "depends_on": clarif_id,
                                "origin": "planning",
                            },
                        }
                    )

        # ── Annotate nodes with parent_goal ───────────────────────────────────
        for evt in safe_events:
            if evt["type"] == ADD_NODE:
                metadata = evt["payload"].setdefault("metadata", {})
                metadata["parent_goal"] = active_goal.id

        if goal_result:
            safe_events.append(
                {
                    "type": SET_RESULT,
                    "payload": {
                        "node_id": active_goal.id,
                        "result": goal_result,
                    },
                }
            )

        # clarif_events first: clarification node must exist before safe_events
        # reference it.  wiring_events last: task ADD_NODE events must have
        # been applied before these ADD_DEPENDENCY edges can be wired.
        return clarif_events + safe_events + wiring_events

    # ── Clarification node generation (Call 1) ────────────────────────────────

    def _generate_clarification_node(
        self,
        goal_text: str,
        goal_id: str,
        clarif_id: str,
    ) -> tuple:
        """
        Generate clarification fields via a dedicated LLM call.

        Returns (clarification_prompt, fields, events) where events contains
        ADD_NODE + SET_RESULT for the clarification node, ready to be
        prepended to safe_events so the node exists before tasks reference it.
        """
        # Pass skills_summary so the LLM knows which information tools can
        # fetch at runtime and avoids surfacing those as user-facing questions.
        prompt = build_clarification_prompt(
            goal_text,
            skills_summary=self.skills_summary,
            min_fields=self.min_tasks_per_goal,
            max_fields=self.max_tasks_per_goal,
        )
        logger.info("[PLANNER] Generating clarification node for goal %s", goal_id)

        try:
            raw = self.llm.ask(prompt, schema=CLARIFICATION_GENERATION_SCHEMA)
            parsed = json.loads(raw)
            fields = parsed.get("fields", [])
        except Exception as exc:
            logger.warning(
                "[PLANNER] Clarification generation failed (%s) — using empty fields",
                exc,
            )
            fields = []

        events = [
            {
                "type": ADD_NODE,
                "payload": {
                    "node_id": clarif_id,
                    "node_type": "clarification",
                    "dependencies": [],
                    "origin": "planning",
                    "metadata": {
                        "description": (
                            "Goal context — review any unknowns and fill them in, "
                            "then click Confirm to update the plan."
                        ),
                        "fields": fields,
                        "clarification_prompt": prompt,
                        "parent_goal": goal_id,
                    },
                },
            },
            # MARK_DONE (not SET_RESULT) — sets both status="done" and result so
            # recompute_readiness() will promote dependent root tasks to "ready".
            # SET_RESULT only updates node.result without touching status, which
            # would leave the node at "ready" and block all dependent tasks.
            {
                "type": "MARK_DONE",
                "payload": {
                    "node_id": clarif_id,
                    "result": json.dumps(fields, ensure_ascii=False),
                },
            },
        ]

        logger.info(
            "[PLANNER] Clarification node %s generated with %d field(s)",
            clarif_id,
            len(fields),
        )
        return prompt, fields, events

    # ── Scrutinizer ───────────────────────────────────────────────────────────

    def _scrutinize(self, original_prompt: str, draft_json: str) -> str:
        """
        Feed the draft plan back to the LLM for self-review.

        Clarification context reaches the scrutinizer automatically because it
        is embedded verbatim inside original_prompt via build_planner_prompt().
        Returns the improved JSON string, or the original draft on failure.
        """
        scrutinizer_prompt = build_plan_scrutinizer_prompt(
            original_planning_prompt=original_prompt,
            draft_plan_json=draft_json,
            min_tasks=self.min_tasks_per_goal,
            max_tasks=self.max_tasks_per_goal,
        )

        logger.info("[PLANNER] Running plan scrutiny pass")
        try:
            improved = self.llm.ask(scrutinizer_prompt, schema=PLAN_SCHEMA)
        except Exception as exc:
            logger.warning("[PLANNER] Scrutiny LLM call failed (%s) — using draft", exc)
            return draft_json

        try:
            json.loads(improved)
        except Exception:
            logger.warning("[PLANNER] Scrutiny output is not valid JSON — using draft")
            return draft_json

        logger.info("[PLANNER] Scrutiny pass complete — using improved plan")
        return improved

    # ── Snapshot serialization ────────────────────────────────────────────────

    def _serialize_snapshot(self, snapshot):
        return [
            {
                "node_id": n.id,
                "status": n.status,
                "dependencies": sorted(n.dependencies),
                "node_type": getattr(n, "node_type", "task"),
                "metadata": {
                    k: (
                        v
                        if k == "description"
                        else (v[:120] + "…" if isinstance(v, str) and len(v) > 120 else v)
                    )
                    for k, v in n.metadata.items()
                    if k not in _VOLATILE_METADATA_KEYS
                },
            }
            for n in sorted(snapshot.values(), key=lambda n: n.id)
        ]

    # ── Prompt builder ────────────────────────────────────────────────────────

    def _build_prompt(self, graph_view, goals, clarif_fields=None, clarif_prompt=""):
        node_map = {n["node_id"]: n for n in graph_view}

        relevant_ids = set()
        for g in goals:
            relevant_ids.add(g.id)
            relevant_ids.update(g.dependencies)
            relevant_ids.update(g.children)
            for dep_id in g.dependencies:
                dep_node = node_map.get(dep_id)
                if dep_node:
                    for n in graph_view:
                        if dep_id in n.get("dependencies", []):
                            relevant_ids.add(n["node_id"])

        pruned_view = [n for n in graph_view if n["node_id"] in relevant_ids]
        existing_ids = {n["node_id"] for n in graph_view}

        goals_repr = [
            {
                "node_id": g.id,
                "node_type": g.node_type,
                "status": g.status,
                "metadata": {
                    k: v for k, v in g.metadata.items() if k not in _VOLATILE_METADATA_KEYS
                },
            }
            for g in goals
        ]

        existing_ids_note = (
            "\nNodes already in the DAG — do NOT emit ADD_NODE for any of these:\n"
            + json.dumps(sorted(existing_ids), indent=2)
            + "\n"
        )

        skills_block = build_planner_skills_block(self.skills_summary)
        clarification_block = build_clarification_context_block(clarif_fields or [], clarif_prompt)

        return build_planner_prompt(
            pruned_view_json=json.dumps(pruned_view, indent=2),
            goals_repr_json=json.dumps(goals_repr, indent=2),
            existing_ids_note=existing_ids_note,
            skills_block=skills_block,
            min_tasks=self.min_tasks_per_goal,
            max_tasks=self.max_tasks_per_goal,
            clarification_block=clarification_block,
        )

    # ── Event normalizer ──────────────────────────────────────────────────────

    def _normalize_events(self, raw_events):
        if not isinstance(raw_events, list):
            return raw_events

        normalized = []
        for item in raw_events:
            if not isinstance(item, dict):
                normalized.append(item)
                continue

            item = dict(item)
            if "operation" in item and "type" not in item:
                item["type"] = item.pop("operation")
            if "node" in item and "payload" not in item:
                item["payload"] = item.pop("node")

            event_type = item.get("type", "")

            if "type" in item and "payload" in item:
                normalized.append(item)
                continue

            if event_type == ADD_DEPENDENCY and "from" in item and "to" in item:
                normalized.append(
                    {
                        "type": ADD_DEPENDENCY,
                        "payload": {"node_id": item["to"], "depends_on": item["from"]},
                    }
                )
                continue

            if (
                "node_id" in item
                and "depends_on" not in item
                and "to" not in item
                and "type" not in item
            ):
                node_id = item["node_id"]
                metadata = {}
                for key in (
                    "description",
                    "parallel_group",
                    "required_input",
                    "output",
                    "reflection_notes",
                    "skill",
                    "tools",
                ):
                    if key in item:
                        metadata[key] = item[key]
                normalized.append(
                    {
                        "type": ADD_NODE,
                        "payload": {
                            "node_id": node_id,
                            "node_type": item.get("node_type", "task"),
                            "dependencies": item.get("dependencies", []),
                            "metadata": metadata,
                        },
                    }
                )
                continue

            if "from" in item and "to" in item and "type" not in item:
                normalized.append(
                    {
                        "type": ADD_DEPENDENCY,
                        "payload": {"node_id": item["to"], "depends_on": item["from"]},
                    }
                )
                continue

            if "node_id" in item and "depends_on" in item and "type" not in item:
                normalized.append(
                    {
                        "type": ADD_DEPENDENCY,
                        "payload": {
                            "node_id": item["node_id"],
                            "depends_on": item["depends_on"],
                        },
                    }
                )
                continue

            logger.warning("[PLANNER] Unrecognized event shape: %r", item)
            normalized.append(item)

        return normalized
