# planning/llm_planner.py

from cuddlytoddly.core.events import ADD_NODE, ADD_DEPENDENCY, SET_RESULT
import json
from cuddlytoddly.planning.llm_interface import PLAN_SCHEMA
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
    "coverage_checked"
}


class LLMPlanner:
    def __init__(self, llm_client, graph, refiner=None, skills_summary: str = ""):
        self.llm = llm_client
        self.graph = graph
        self.refiner = refiner
        self.skills_summary = skills_summary

    def propose(self, context):
        snapshot = context.snapshot
        goals    = context.goals

        if not goals:
            return []

        active_goal = goals[0]

        graph_view = self._serialize_snapshot(snapshot)
        prompt     = self._build_prompt(graph_view, goals)

        llm_output = self.llm.ask(prompt, schema=PLAN_SCHEMA)

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

    # ── Snapshot serialization ────────────────────────────────────────────────

    def _serialize_snapshot(self, snapshot):
        return [
            {
                "node_id":      n.id,
                "status":       n.status,
                "dependencies": sorted(n.dependencies),
                "node_type":    getattr(n, "node_type", "task"),
                "metadata": {
                    k: (v if k == "description" else (v[:120] + "…" if isinstance(v, str) and len(v) > 120 else v))
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

        skills_block = ""
        if self.skills_summary:
            skills_block = f"""
{self.skills_summary}

When decomposing goals into tasks:
- Assign the most relevant skill to each task via metadata.skill (e.g. "web_research")
- Tasks assigned a skill should specify metadata.tools listing the specific tools they need
- A task with no matching skill can still be completed by the LLM directly
"""

        existing_ids_note = (
            "\nNodes already in the DAG — do NOT emit ADD_NODE for any of these:\n"
            + json.dumps(sorted(existing_ids), indent=2)
            + "\n"
        )

        return f"""
You are a DAG planning assistant.

Current DAG snapshot:
{json.dumps(pruned_view, indent=2)}

Goals to expand:
{json.dumps(goals_repr, indent=2)}
{existing_ids_note}{skills_block}
Your task is to decompose each goal into prerequisite tasks.

Guidelines:
- Produce between 3 and 8 tasks per goal. Do not exceed 8 tasks.
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

            if "node_id" in item and "depends_on" not in item and "to" not in item and "type" not in item:
                node_id  = item["node_id"]
                metadata = {}
                for key in ("description", "parallel_group", "required_input",
                            "output", "reflection_notes", "skill", "tools"):
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
                    "payload": {"node_id": item["node_id"], "depends_on": item["depends_on"]},
                })
                continue

            logger.warning("[PLANNER] Unrecognized event shape: %r", item)
            normalized.append(item)

        return normalized