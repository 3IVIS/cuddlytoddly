# planning/llm_planner.py

import json

from cuddlytoddly.core.events import ADD_NODE, ADD_DEPENDENCY, SET_RESULT
from cuddlytoddly.planning.schemas import PLAN_SCHEMA
from cuddlytoddly.planning.prompts import (
    build_planner_prompt,
    build_planner_skills_block,
    build_plan_scrutinizer_prompt,
)
from cuddlytoddly.planning.llm_output_validator import LLMOutputValidator
from cuddlytoddly.infra.logging import get_logger

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
        self.llm                 = llm_client
        self.graph               = graph
        self.refiner             = refiner
        self.skills_summary      = skills_summary
        self.min_tasks_per_goal  = min_tasks_per_goal
        self.max_tasks_per_goal  = max_tasks_per_goal
        self.scrutinize_plan     = scrutinize_plan

    def propose(self, context):
        snapshot = context.snapshot
        goals    = context.goals

        if not goals:
            return []

        active_goal = goals[0]

        graph_view = self._serialize_snapshot(snapshot)
        prompt     = self._build_prompt(graph_view, goals)

        llm_output = self.llm.ask(prompt, schema=PLAN_SCHEMA)

        if self.scrutinize_plan:
            llm_output = self._scrutinize(prompt, llm_output)

        try:
            parsed = json.loads(llm_output)
        except Exception as e:
            logger.error("[PLANNER] JSON parse error: %s", e)
            return []

        # Field is named a_goal_result so it sorts before "events" in the schema,
        # forcing constrained decoding to generate the reasoning first.
        goal_result = parsed.get("a_goal_result", "").strip()
        raw_events  = parsed.get("events", [])

        raw_events  = self._normalize_events(raw_events)
        validator   = LLMOutputValidator(self.graph)
        safe_events = validator.validate_and_normalize(
            raw_events, forced_origin="planning"
        )

        for evt in safe_events:
            if evt["type"] == ADD_NODE:
                metadata = evt["payload"].setdefault("metadata", {})
                metadata["parent_goal"] = active_goal.id

        if goal_result:
            safe_events.append({
                "type": SET_RESULT,
                "payload": {
                    "node_id": active_goal.id,
                    "result":  goal_result,
                },
            })

        return safe_events

    # ── Scrutinizer ───────────────────────────────────────────────────────────

    def _scrutinize(self, original_prompt: str, draft_json: str) -> str:
        """
        Feed the draft plan back to the LLM for self-review.

        The scrutinizer prompt embeds the complete original planning prompt so
        no constraint (DAG snapshot, existing IDs, task-count limits, dependency
        semantics, format rules) is lost between the two calls.

        Returns the improved JSON string, or the original draft if the
        scrutiny call fails to parse.
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

        # Sanity-check: the improved output must be valid JSON before we
        # replace the draft.  If not, fall back silently.
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
                "node_id":      n.id,
                "status":       n.status,
                "dependencies": sorted(n.dependencies),
                "node_type":    getattr(n, "node_type", "task"),
                "metadata": {
                    k: (
                        v
                        if k == "description"
                        else (
                            v[:120] + "…"
                            if isinstance(v, str) and len(v) > 120
                            else v
                        )
                    )
                    for k, v in n.metadata.items()
                    if k not in _VOLATILE_METADATA_KEYS
                },
            }
            for n in sorted(snapshot.values(), key=lambda n: n.id)
        ]

    # ── Prompt builder ────────────────────────────────────────────────────────

    def _build_prompt(self, graph_view, goals):
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

        pruned_view  = [n for n in graph_view if n["node_id"] in relevant_ids]
        existing_ids = {n["node_id"] for n in graph_view}

        goals_repr = [
            {
                "node_id":   g.id,
                "node_type": g.node_type,
                "status":    g.status,
                "metadata":  {
                    k: v for k, v in g.metadata.items()
                    if k not in _VOLATILE_METADATA_KEYS
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

        return build_planner_prompt(
            pruned_view_json=json.dumps(pruned_view, indent=2),
            goals_repr_json=json.dumps(goals_repr, indent=2),
            existing_ids_note=existing_ids_note,
            skills_block=skills_block,
            min_tasks=self.min_tasks_per_goal,
            max_tasks=self.max_tasks_per_goal,
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
                normalized.append({
                    "type": ADD_DEPENDENCY,
                    "payload": {"node_id": item["to"], "depends_on": item["from"]},
                })
                continue

            if (
                "node_id" in item
                and "depends_on" not in item
                and "to" not in item
                and "type" not in item
            ):
                node_id  = item["node_id"]
                metadata = {}
                for key in (
                    "description", "parallel_group", "required_input",
                    "output", "reflection_notes", "skill", "tools"
                ):
                    if key in item:
                        metadata[key] = item[key]
                normalized.append({
                    "type": ADD_NODE,
                    "payload": {
                        "node_id":      node_id,
                        "node_type":    item.get("node_type", "task"),
                        "dependencies": item.get("dependencies", []),
                        "metadata":     metadata,
                    },
                })
                continue

            if "from" in item and "to" in item and "type" not in item:
                normalized.append({
                    "type": ADD_DEPENDENCY,
                    "payload": {"node_id": item["to"], "depends_on": item["from"]},
                })
                continue

            if "node_id" in item and "depends_on" in item and "type" not in item:
                normalized.append({
                    "type": ADD_DEPENDENCY,
                    "payload": {
                        "node_id":    item["node_id"],
                        "depends_on": item["depends_on"],
                    },
                })
                continue

            logger.warning("[PLANNER] Unrecognized event shape: %r", item)
            normalized.append(item)

        return normalized
