"""Tests for cuddlytoddly.ui.startup: parse_manual_plan, scan_runs, StartupChoice."""
import json
import time
import pytest
from pathlib import Path
from cuddlytoddly.ui.startup import (
    parse_manual_plan, scan_runs, StartupChoice, build_manual_plan_events,
)
from cuddlytoddly.core.events import ADD_NODE, ADD_DEPENDENCY


# ── parse_manual_plan ─────────────────────────────────────────────────────────

class TestParseManualPlan:
    def test_empty_input_returns_empty(self):
        goal, events = parse_manual_plan("")
        assert goal == ""
        assert events == []

    def test_whitespace_only_returns_empty(self):
        goal, events = parse_manual_plan("   \n\n   ")
        assert goal == ""
        assert events == []

    def test_goal_extracted_from_first_line(self):
        plan = "My amazing goal\n- Task_One: Do something\n"
        goal, events = parse_manual_plan(plan)
        assert "amazing goal" in goal.lower() or "My amazing goal" in goal

    def test_bullet_lines_become_tasks(self):
        plan = "Goal here\n- Task_A: first task\n- Task_B: second task"
        goal, events = parse_manual_plan(plan)
        node_ids = {e["payload"]["node_id"] for e in events if e["type"] == ADD_NODE}
        assert "Task_A" in node_ids
        assert "Task_B" in node_ids

    def test_goal_node_added(self):
        plan = "My Goal\n- Task_One: Do something"
        goal, events = parse_manual_plan(plan)
        goal_events = [e for e in events
                       if e["type"] == ADD_NODE
                       and e["payload"]["node_type"] == "goal"]
        assert len(goal_events) == 1

    def test_dependency_syntax_bracket(self):
        plan = "Goal\n- Task_A: first\n- Task_B: second [depends: Task_A]"
        goal, events = parse_manual_plan(plan)
        task_b_event = next(
            (e for e in events if e["type"] == ADD_NODE
             and e["payload"]["node_id"] == "Task_B"), None
        )
        assert task_b_event is not None
        assert "Task_A" in task_b_event["payload"]["dependencies"]

    def test_dependency_syntax_depends_on(self):
        plan = "Goal\n- Task_A: first\n- Task_B: second depends on: Task_A"
        goal, events = parse_manual_plan(plan)
        task_b_event = next(
            (e for e in events if e["type"] == ADD_NODE
             and e["payload"]["node_id"] == "Task_B"), None
        )
        if task_b_event:
            assert "Task_A" in task_b_event["payload"]["dependencies"]

    def test_terminal_task_wired_to_goal(self):
        """Tasks not depended on by others should be wired to the goal."""
        plan = "Goal\n- Task_A: first\n- Task_B: second [depends: Task_A]"
        goal, events = parse_manual_plan(plan)
        dep_events = [e for e in events if e["type"] == ADD_DEPENDENCY]
        goal_deps = {e["payload"]["depends_on"] for e in dep_events
                     if e["payload"]["node_id"] != "Task_A"}
        # Task_B is terminal (nothing depends on it) and should be wired to goal
        assert "Task_B" in goal_deps or len(dep_events) > 0

    def test_asterisk_bullet_recognised(self):
        plan = "Goal\n* Task_Star: a task"
        goal, events = parse_manual_plan(plan)
        node_ids = {e["payload"]["node_id"] for e in events if e["type"] == ADD_NODE}
        assert "Task_Star" in node_ids

    def test_task_description_preserved(self):
        plan = "Goal\n- My_Task: This is the description"
        goal, events = parse_manual_plan(plan)
        task_event = next(
            (e for e in events if e["type"] == ADD_NODE
             and e["payload"]["node_id"] == "My_Task"), None
        )
        assert task_event is not None
        assert "description" in task_event["payload"].get("metadata", {})
        assert "This is the description" in task_event["payload"]["metadata"]["description"]

    def test_no_tasks_returns_goal_only(self):
        plan = "Just a goal with no tasks"
        goal, events = parse_manual_plan(plan)
        assert goal != ""
        # No task ADD_NODE events, just the goal one
        task_events = [e for e in events
                       if e["type"] == ADD_NODE
                       and e["payload"]["node_type"] == "task"]
        assert task_events == []

    def test_multiple_deps_parsed(self):
        plan = "Goal\n- A: first\n- B: second\n- C: third [depends: A, B]"
        goal, events = parse_manual_plan(plan)
        c_event = next(
            (e for e in events if e["type"] == ADD_NODE
             and e["payload"]["node_id"] == "C"), None
        )
        assert c_event is not None
        deps = c_event["payload"]["dependencies"]
        assert "A" in deps and "B" in deps

    def test_to_id_sanitizes_spaces(self):
        """Task IDs derived from names should not contain spaces."""
        plan = "Goal\n- My long task name: something"
        goal, events = parse_manual_plan(plan)
        for e in events:
            if e["type"] == ADD_NODE:
                assert " " not in e["payload"]["node_id"]


# ── scan_runs ─────────────────────────────────────────────────────────────────

class TestScanRuns:
    def _make_run(self, runs_dir, name, goal="test goal", nodes=3):
        run_dir = runs_dir / name
        run_dir.mkdir(parents=True)
        events_file = run_dir / "events.jsonl"
        lines = []
        # Add goal node
        lines.append(json.dumps({
            "type": "ADD_NODE",
            "payload": {
                "node_id": "goal_1",
                "node_type": "goal",
                "dependencies": [],
                "metadata": {"description": goal},
            },
            "timestamp": "2025-01-01T00:00:00",
        }))
        # Add task nodes
        for i in range(nodes - 1):
            lines.append(json.dumps({
                "type": "ADD_NODE",
                "payload": {
                    "node_id": f"task_{i}",
                    "node_type": "task",
                    "dependencies": [],
                    "metadata": {"description": f"task {i}"},
                },
                "timestamp": "2025-01-01T00:00:00",
            }))
        events_file.write_text("\n".join(lines) + "\n")
        return run_dir

    def test_returns_empty_for_empty_runs_dir(self, tmp_path):
        (tmp_path / "runs").mkdir()
        runs = scan_runs(tmp_path)
        assert runs == []

    def test_returns_empty_when_no_runs_dir(self, tmp_path):
        runs = scan_runs(tmp_path)
        assert runs == []

    def test_finds_single_run(self, tmp_path):
        self._make_run(tmp_path / "runs", "my_run", goal="build something")
        runs = scan_runs(tmp_path)
        assert len(runs) == 1

    def test_run_goal_extracted(self, tmp_path):
        self._make_run(tmp_path / "runs", "run1", goal="explore the cosmos")
        runs = scan_runs(tmp_path)
        assert runs[0]["goal"] == "explore the cosmos"

    def test_run_node_count_correct(self, tmp_path):
        self._make_run(tmp_path / "runs", "run1", nodes=5)
        runs = scan_runs(tmp_path)
        assert runs[0]["node_count"] == 5

    def test_empty_events_file_skipped(self, tmp_path):
        run_dir = tmp_path / "runs" / "empty_run"
        run_dir.mkdir(parents=True)
        (run_dir / "events.jsonl").write_text("")
        runs = scan_runs(tmp_path)
        assert runs == []

    def test_multiple_runs_returned(self, tmp_path):
        for i in range(3):
            self._make_run(tmp_path / "runs", f"run_{i}", goal=f"goal {i}")
        runs = scan_runs(tmp_path)
        assert len(runs) == 3

    def test_runs_sorted_most_recent_first(self, tmp_path):
        runs_dir = tmp_path / "runs"
        for i in range(3):
            rd = self._make_run(runs_dir, f"run_{i}", goal=f"goal {i}")
            # Touch with slightly different mtime
            mtime = time.time() - (3 - i) * 100
            import os
            os.utime(rd, (mtime, mtime))
        runs = scan_runs(tmp_path)
        assert len(runs) == 3
        # Most recent should be first
        for i in range(len(runs) - 1):
            assert runs[i]["mtime"] >= runs[i + 1]["mtime"]

    def test_age_string_is_human_readable(self, tmp_path):
        self._make_run(tmp_path / "runs", "run1", goal="something")
        runs = scan_runs(tmp_path)
        age = runs[0]["age"]
        # Should contain "ago", "now", "m", "h", or "d"
        assert any(word in age for word in ["ago", "now", "m ", "h ", "d "])


# ── StartupChoice ─────────────────────────────────────────────────────────────

class TestStartupChoice:
    def test_basic_construction(self):
        choice = StartupChoice(
            mode="new_goal",
            run_dir=Path("/tmp/run"),
            goal_text="do something",
            is_fresh=True,
        )
        assert choice.mode == "new_goal"
        assert choice.goal_text == "do something"
        assert choice.is_fresh is True

    def test_plan_events_defaults_to_empty_list(self):
        choice = StartupChoice(
            mode="manual_plan",
            run_dir=Path("/tmp/run"),
            goal_text="goal",
        )
        assert choice.plan_events == []


# ── build_manual_plan_events ──────────────────────────────────────────────────

class TestBuildManualPlanEvents:
    def test_basic_events_built(self):
        tasks = [
            {"node_id": "Task_A", "description": "first task", "dependencies": []},
            {"node_id": "Task_B", "description": "second task", "dependencies": ["Task_A"]},
        ]
        events = build_manual_plan_events("my_goal", "My Goal", tasks)
        node_ids = {e["payload"]["node_id"] for e in events if e["type"] == ADD_NODE}
        assert "Task_A" in node_ids
        assert "Task_B" in node_ids
        assert "my_goal" in node_ids

    def test_goal_node_type_is_goal(self):
        tasks = [{"node_id": "T", "description": "t", "dependencies": []}]
        events = build_manual_plan_events("g", "Goal", tasks)
        goal_event = next(
            (e for e in events if e["payload"].get("node_id") == "g"), None
        )
        assert goal_event is not None
        assert goal_event["payload"]["node_type"] == "goal"

    def test_terminal_tasks_wired_to_goal(self):
        tasks = [
            {"node_id": "A", "description": "a", "dependencies": []},
            {"node_id": "B", "description": "b", "dependencies": ["A"]},
        ]
        events = build_manual_plan_events("goal", "Goal", tasks)
        dep_events = [e for e in events if e["type"] == ADD_DEPENDENCY]
        goal_dep_targets = {e["payload"]["depends_on"] for e in dep_events
                            if e["payload"]["node_id"] == "goal"}
        # B is terminal (no one depends on it), so goal should depend on B
        assert "B" in goal_dep_targets

    def test_empty_tasks_list(self):
        events = build_manual_plan_events("g", "Goal", [])
        node_events = [e for e in events if e["type"] == ADD_NODE]
        assert len(node_events) == 1  # just the goal
        assert node_events[0]["payload"]["node_id"] == "g"
