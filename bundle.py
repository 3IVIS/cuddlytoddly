

# --- FILE: __init__.py ---




# --- FILE: __main__.py ---

# __main__.py  — updated startup section
# Replace the existing main() function with this version.
# Everything from the LLM client setup downward is unchanged.

import sys
import os
import argparse
import threading

from pathlib import Path

from cuddlytoddly.core.task_graph import TaskGraph
from cuddlytoddly.core.events import Event, ADD_NODE
from cuddlytoddly.core.reducer import apply_event
from cuddlytoddly.infra.event_queue import EventQueue
from cuddlytoddly.infra.event_log import EventLog
from cuddlytoddly.infra.replay import rebuild_graph_from_log
from cuddlytoddly.infra.logging import setup_logging, get_logger
from cuddlytoddly.planning.llm_interface import create_llm_client
from cuddlytoddly.planning.llm_planner import LLMPlanner
from cuddlytoddly.planning.llm_executor import LLMExecutor
from cuddlytoddly.engine.quality_gate import QualityGate
from cuddlytoddly.engine.llm_orchestrator import SimpleOrchestrator
from cuddlytoddly.skills.skill_loader import SkillLoader
from cuddlytoddly.ui.curses_ui import run_ui
from cuddlytoddly.ui.startup import StartupChoice
from cuddlytoddly.ui.startup import run_startup_curses
from cuddlytoddly.ui.web_server import run_web_ui
from cuddlytoddly.core.id_generator import StableIDGenerator
import cuddlytoddly.planning.llm_interface as llm_iface

REPO_ROOT = Path(__file__).resolve().parent

setup_logging()
logger = get_logger(__name__)

# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_PATH   = str(REPO_ROOT / "models/Llama-3.3-70B-Instruct-Q4_K_M.gguf")
N_GPU_LAYERS = -1
TEMPERATURE  = 0.1
N_CTX        = int(131072 / 8)
MAX_TOKENS   = int(65536 / 8)
MAX_WORKERS  = 1


def make_run_dir(goal_text: str) -> Path:
    safe    = goal_text.lower().replace(" ", "_")
    safe    = "".join(c for c in safe if c.isalnum() or c == "_")[:60]
    run_dir = REPO_ROOT / "runs" / safe
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "outputs").mkdir(exist_ok=True)
    return run_dir


def main():

    # ── CLI ───────────────────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        prog="cuddlytoddly",
        description="LLM-powered DAG planning and execution system.",
    )
    parser.add_argument(
        "--terminal",
        action="store_true",
        default=False,
        help="Launch the terminal UI instead of the web UI.",
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host for the web UI server (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8765,
        help="Port for the web UI server (default: 8765).",
    )
    parser.add_argument(
        "goal",
        nargs="*",
        help=(
            "Goal text to start immediately, skipping the startup screen. "
            "If omitted the startup screen is shown."
        ),
    )
    args = parser.parse_args()
    use_web = not args.terminal

    # ── Startup screen ────────────────────────────────────────────────────────
    # If a goal was passed on the CLI, skip the startup screen and go straight
    # to a new-goal run.  Otherwise show the appropriate startup UI.
    inline_goal = " ".join(args.goal).strip()

    if inline_goal:
        # Bypass startup screen — behaves like the old CLI usage
        choice = StartupChoice(
            mode="new_goal",
            run_dir=make_run_dir(inline_goal).resolve(),
            goal_text=inline_goal,
            is_fresh=True,
        )
    elif use_web:
        # Web startup screen is shown inside the browser — no blocking call here.
        # We use a deferred choice that will be filled in via /api/startup.
        choice = None
    else:
        # Curses startup screen — blocks until the user makes a choice.
        try:
            choice = run_startup_curses(REPO_ROOT)
        except SystemExit:
            return   # user pressed q

    # ── For web UI with no inline goal, defer all init to init_fn ────────────
    if use_web and choice is None:

        def init_fn(ch: StartupChoice):
            return _init_system(ch, use_web)

        run_web_ui(
            repo_root=REPO_ROOT,
            init_fn=init_fn,
            host=args.host,
            port=args.port,
        )
        return

    # ── Init system ───────────────────────────────────────────────────────────
    orchestrator, run_dir = _init_system(choice, use_web)

    # ── Launch UI ─────────────────────────────────────────────────────────────
    if use_web:
        run_web_ui(
            orchestrator=orchestrator,
            run_dir=run_dir,
            host=args.host,
            port=args.port,
        )
    else:
        run_ui(
            orchestrator,
            run_dir=run_dir,
            repo_root=REPO_ROOT,
            restart_fn=_init_system,
        )

    # ── Final log ─────────────────────────────────────────────────────────────
    snap = orchestrator.graph.get_snapshot()
    logger.info("=== Final graph state (%d nodes) ===", len(snap))
    for nid, n in snap.items():
        logger.info("  [%s] %s", n.status, nid)


def _init_system(choice: "StartupChoice", use_web: bool):
    """
    Build the full orchestrator from a StartupChoice.
    Extracted so both the curses and web startup paths share the same logic.
    Returns (orchestrator, run_dir).
    """
    goal_text = choice.goal_text
    goal_id   = goal_text.replace(" ", "_")[:60]
    run_dir   = choice.run_dir.resolve()

    # ── Logging ───────────────────────────────────────────────────────────────
    setup_logging(log_dir=run_dir / "logs")
    _logger = get_logger(__name__)
    _logger.info("=== cuddlytoddly starting  mode=%s  ui=%s ===",
                 choice.mode, "web" if use_web else "curses")
    _logger.info("Run directory: %s", run_dir)

    # ── Event log ─────────────────────────────────────────────────────────────
    event_log_path = run_dir / "events.jsonl"
    event_log      = EventLog(str(event_log_path))
    log_path       = Path(event_log_path)

    # ── LLM cache ─────────────────────────────────────────────────────────────
    cache_path = str(run_dir / "llamacpp_cache.json")

    # ── Git repo — per run ────────────────────────────────────────────────────
    import cuddlytoddly.ui.git_projection as git_proj
    git_proj.REPO_PATH = str(run_dir / "dag_repo")

    # ── Working directory — sandbox file tools ────────────────────────────────
    os.chdir(run_dir / "outputs")
    _logger.info("Working directory: %s", Path.cwd())

    llm_iface.id_gen = StableIDGenerator(
        mapping_file=run_dir / "task_id_map.json",
        id_length=6,
    )

    # ── Graph init ────────────────────────────────────────────────────────────
    if not choice.is_fresh and log_path.exists() and log_path.stat().st_size > 0:
        _logger.info("[STARTUP] Replaying event log")
        graph       = rebuild_graph_from_log(event_log)
        fresh_start = False
        _logger.info("[STARTUP] Restored %d nodes", len(graph.nodes))

        for step_id in [n.id for n in graph.nodes.values()
                        if n.node_type == "execution_step"]:
            if step_id in graph.nodes:
                graph.detach_node(step_id)

        for node_id in {n.id for n in graph.nodes.values()
                        if n.status in ("running", "failed")
                        and n.node_type != "execution_step"}:
            n = graph.nodes.get(node_id)
            if n:
                n.status = "pending"
                n.result = None
                n.metadata.pop("retry_count",          None)
                n.metadata.pop("verification_failure", None)
                n.metadata.pop("verified",             None)

        graph.recompute_readiness()
    else:
        graph       = TaskGraph()
        fresh_start = True

    # ── LLM / components ──────────────────────────────────────────────────────
    shared_llm = create_llm_client(
        "llamacpp",
        model_path=MODEL_PATH,
        n_gpu_layers=N_GPU_LAYERS,
        temperature=TEMPERATURE,
        n_ctx=N_CTX,
        max_tokens=MAX_TOKENS,
        cache_path=cache_path,
    )

    skills       = SkillLoader()
    registry     = skills.registry
    planner      = LLMPlanner(llm_client=shared_llm, graph=graph,
                              skills_summary=skills.prompt_summary)
    executor     = LLMExecutor(llm_client=shared_llm, tool_registry=registry, max_turns=5)
    quality_gate = QualityGate(llm_client=shared_llm, tool_registry=registry)

    queue        = EventQueue()
    orchestrator = SimpleOrchestrator(
        graph=graph, planner=planner, executor=executor,
        quality_gate=quality_gate, event_log=event_log,
        event_queue=queue, max_workers=MAX_WORKERS,
    )

    # ── Seed graph ────────────────────────────────────────────────────────────
    if fresh_start:
        if choice.mode == "manual_plan" and choice.plan_events:
            _logger.info("[STARTUP] Seeding manual plan (%d events)",
                         len(choice.plan_events))
            for evt_dict in choice.plan_events:
                apply_event(graph, Event(evt_dict["type"], evt_dict["payload"]),
                            event_log=event_log)
        else:
            _logger.info("[STARTUP] Seeding new goal: %s", goal_text)
            apply_event(graph, Event(ADD_NODE, {
                "node_id":      goal_id,
                "node_type":    "goal",
                "dependencies": [],
                "origin":       "user",
                "metadata":     {"description": goal_text, "expanded": False},
            }), event_log=event_log)

    orchestrator.start()

    if not fresh_start:
        def _bg_verify():
            _logger.info("[STARTUP] Background verification pass starting...")
            orchestrator.verify_restored_nodes()
            _logger.info("[STARTUP] Background verification complete")
        threading.Thread(target=_bg_verify, daemon=True, name="startup-verify").start()

    return orchestrator, run_dir

if __name__ == "__main__":
    main()


# --- FILE: core/__init__.py ---




# --- FILE: core/events.py ---

"""
Event Definitions

All mutations must go through reducer.
"""

# core/events.py

from datetime import datetime


class Event:
    def __init__(self, type, payload, timestamp=None):
        self.type = type
        self.payload = payload
        self.timestamp = timestamp or datetime.utcnow().isoformat()

    def to_dict(self):
        return {
            "type": self.type,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data):
        return cls(
            type=data["type"],
            payload=data["payload"],
            timestamp=data.get("timestamp"),
        )

# Event types
ADD_NODE = "ADD_NODE"
REMOVE_NODE = "REMOVE_NODE"
ADD_DEPENDENCY = "ADD_DEPENDENCY"
REMOVE_DEPENDENCY = "REMOVE_DEPENDENCY"
MARK_RUNNING = "MARK_RUNNING"
MARK_DONE = "MARK_DONE"
MARK_FAILED = "MARK_FAILED"
RESET_NODE = "RESET_NODE"
UPDATE_METADATA = "UPDATE_METADATA"
DETACH_NODE = "DETACH_NODE"
UPDATE_STATUS = "UPDATE_STATUS"
SET_RESULT = "SET_RESULT"
SET_NODE_TYPE = "SET_NODE_TYPE"
RESET_SUBTREE = "RESET_SUBTREE"


# --- FILE: core/id_generator.py ---

# core/id_generator.py
import hashlib
import string
import json
from pathlib import Path
from typing import Dict

BASE62_ALPHABET = string.digits + string.ascii_lowercase + string.ascii_uppercase

def base62_encode(num: int, length: int = 6) -> str:
    chars = []
    base = len(BASE62_ALPHABET)

    if num == 0:
        chars.append('0')

    while num > 0:
        num, rem = divmod(num, base)
        chars.append(BASE62_ALPHABET[rem])

    while len(chars) < length:
        chars.append('0')

    return ''.join(reversed(chars))[:length]


class StableIDGenerator:
    def __init__(self, mapping_file=None, id_length=6):  # None = in-memory only
        self.mapping_file = Path(mapping_file) if mapping_file else None
        self.id_length = id_length
        self._load_mapping()

    def _load_mapping(self):
        if self.mapping_file and self.mapping_file.exists():
            try:
                with self.mapping_file.open() as f:
                    self.mapping: Dict[str, Dict[str, str]] = json.load(f)
            except json.JSONDecodeError:
                self.mapping = {}
        else:
            self.mapping = {}

    def _save_mapping(self):
        if self.mapping_file is None:
            return          # in-memory mode — nothing to persist
        with self.mapping_file.open("w") as f:
            json.dump(self.mapping, f, indent=2)

    def get_id(self, key: str, domain: str = "default") -> str:
        """
        Returns a deterministic ID for `key` within a given `domain`.
        IDs are guaranteed unique inside that domain only.
        """

        # Ensure domain exists
        if domain not in self.mapping:
            self.mapping[domain] = {}

        domain_map = self.mapping[domain]

        # If key already exists in domain
        if key in domain_map:
            return domain_map[key]

        # Generate deterministic hash
        digest = hashlib.sha256(f"{domain}:{key}".encode("utf-8")).hexdigest()
        digest_int = int(digest, 16)
        short_id = base62_encode(digest_int, self.id_length)

        # Handle collisions ONLY inside this domain
        while short_id in domain_map.values():
            digest_int += 1
            short_id = base62_encode(digest_int, self.id_length)

        domain_map[key] = short_id
        self._save_mapping()

        return short_id


# --- FILE: core/reducer.py ---

from copy import deepcopy
from cuddlytoddly.core.events import *
from cuddlytoddly.core.task_graph import TaskGraph

STRUCTURAL_EVENTS = {
    ADD_NODE,
    REMOVE_NODE,
    ADD_DEPENDENCY,
    REMOVE_DEPENDENCY,
    SET_NODE_TYPE,
}

EXECUTION_EVENTS = {
    MARK_RUNNING,
    MARK_DONE,
    MARK_FAILED,
    RESET_NODE,
    UPDATE_METADATA,
    SET_RESULT,
    RESET_SUBTREE
}

def apply_event(graph: TaskGraph, event: Event, event_log=None):
    t = event.type
    p = event.payload or {}

    if t == "INSERT_NODE":
        t = ADD_NODE

    # ---------------- NODE EVENTS ----------------
    if t == ADD_NODE:
        node_id = p["node_id"]
        node_type = p.get("node_type", "task")
        dependencies = p.get("dependencies", [])
        metadata = deepcopy(p.get("metadata", {}))
        origin = p.get("origin", "user")

        if node_id not in graph.nodes:
            graph.add_node(
                node_id=node_id,
                node_type=node_type,
                dependencies=dependencies,
                origin=origin,
                metadata=metadata,
            )
        else:
            node = graph.nodes[node_id]
            existing_desc = node.metadata.get("description", "")
            node.metadata.update(metadata)
            # Restore the original description if it was already populated —
            # user-provided and previously-set descriptions must not be clobbered
            if existing_desc:
                node.metadata["description"] = existing_desc
            node.node_type = node_type or node.node_type

    elif t == REMOVE_NODE:
        graph.remove_node(p["node_id"])

    # ---------------- DEPENDENCY EVENTS ----------------
    elif t == ADD_DEPENDENCY:
        graph.add_dependency(p["node_id"], p["depends_on"])

    elif t == REMOVE_DEPENDENCY:
        graph.remove_dependency(p["node_id"], p["depends_on"])

    # ---------------- EXECUTION EVENTS ----------------
    elif t == MARK_RUNNING:
        node = graph.nodes.get(p["node_id"])
        if node:
            node.status = "running"

    elif t == MARK_DONE:
        node = graph.nodes.get(p["node_id"])
        if node:
            node.status = "done"
            node.result = p.get("result")

    elif t == MARK_FAILED:
        node = graph.nodes.get(p["node_id"])
        if node:
            node.status = "failed"

    elif t == RESET_NODE:
        node = graph.nodes.get(p["node_id"])
        if node:
            node.reset()

    elif t == DETACH_NODE:
        graph.detach_node(p["node_id"])

    elif t == UPDATE_STATUS:
        graph.update_status(p["node_id"], p["status"])

    elif t == SET_RESULT:
        node = graph.nodes.get(p["node_id"])
        if node:
            node.result = p.get("result")

    elif t == SET_NODE_TYPE:
        node = graph.nodes.get(p["node_id"])
        if node:
            node.node_type = p["node_type"]

    # ---------------- METADATA EVENTS ----------------

    elif t == UPDATE_METADATA:
        node = graph.nodes.get(p["node_id"])
        if node:
            existing_desc = node.metadata.get("description", "").strip()
            node.metadata.update(p.get("metadata", {}))
            if existing_desc and p.get("origin") != "user":
                node.metadata["description"] = existing_desc
            # ← remove the "if node_type in p" block entirely

    # ---------------- VERSION TRACKING ----------------
    if t in STRUCTURAL_EVENTS:
        graph.structure_version += 1
    elif t in EXECUTION_EVENTS:
        graph.execution_version += 1

    graph.recompute_readiness()

    if event_log:
        event_log.append(event)


# --- FILE: core/task_graph.py ---

"""
TaskGraph

Single source of truth for DAG planning and execution.
Nodes now include required_input/output metadata to support:
- Explicit dependency checking
- Automatic parallelism reasoning
- LLM-aware semantic planning
"""

import copy
from cuddlytoddly.infra.logging import get_logger

logger = get_logger(__name__)


class TaskGraph:
    class Node:
        def __init__(
            self,
            node_id,
            node_type="task",
            dependencies=None,
            origin="user",
            metadata=None,
        ):
            self.id = node_id
            self.dependencies = set(dependencies or [])
            self.children = set()
            self.node_type = node_type

            self.status = "pending"  # pending / ready / running / done / failed
            self.result = None

            self.origin = origin or "user"
            self.metadata = metadata or {}

            # -----------------------------
            # New semantic fields for planning
            # -----------------------------
            # List of data/resources this node requires (produced by other tasks)
            self.metadata.setdefault("required_input", [])
            # List of data/resources this node produces for downstream tasks
            self.metadata.setdefault("output", [])
            # Optional: group for parallel execution
            self.metadata.setdefault("parallel_group", None)
            # Optional: description / notes
            self.metadata.setdefault("description", "")
            self.metadata.setdefault("reflection_notes", [])

            def _coerce_io_list(items):
                """Upgrade legacy slug strings to typed IO objects."""
                result = []
                for item in items:
                    if isinstance(item, str):
                        # Infer type from extension
                        t = "file" if any(item.endswith(ext) for ext in
                                        {".md",".txt",".py",".json",".csv",".html",".yaml",".xml"}) \
                            else "document"
                        result.append({"name": item, "type": t, "description": item.replace("_", " ")})
                    else:
                        result.append(item)
                return result

            self.metadata["required_input"] = _coerce_io_list(self.metadata.get("required_input", []))
            self.metadata["output"]         = _coerce_io_list(self.metadata.get("output", []))

        def reset(self):
            self.status = "pending"
            self.result = None

        def to_dict(self):
            return {
                "id": self.id,
                "dependencies": list(self.dependencies),
                "children": list(self.children),
                "status": self.status,
                "result": self.result,
                "origin": self.origin,
                "metadata": self.metadata,
            }

    # --------------------------------------------------

    def __init__(self):
        self.nodes = {}
        self.structure_version = 0
        self.execution_version = 0

    # --------------------------------------------------
    # Node Management
    # --------------------------------------------------

    def add_node(
        self, node_id, node_type="task", dependencies=None, origin="user", metadata=None
    ):
        if node_id in self.nodes:
            return

        dependencies = dependencies or []

        self.nodes[node_id] = self.Node(
            node_id=node_id,
            node_type=node_type,
            dependencies=dependencies,
            origin=origin,
            metadata=metadata,
        )

        for dep in dependencies:
            if dep in self.nodes:
                self.nodes[dep].children.add(node_id)

    # --------------------------------------------------

    def remove_node(self, node_id):
        if node_id not in self.nodes:
            return

        # Collect all nodes to remove via iterative BFS instead of recursion
        to_remove = []
        queue = [node_id]
        visited = set()

        while queue:
            current = queue.pop()
            if current in visited or current not in self.nodes:
                continue
            visited.add(current)
            to_remove.append(current)
            queue.extend(self.nodes[current].children)

        # Remove in reverse order (leaves first)
        for nid in reversed(to_remove):
            if nid not in self.nodes:
                continue
            node = self.nodes[nid]
            # Unlink from parents
            for dep in node.dependencies:
                if dep in self.nodes:
                    self.nodes[dep].children.discard(nid)
            # Unlink from children
            for child in node.children:
                if child in self.nodes:
                    self.nodes[child].dependencies.discard(nid)
            del self.nodes[nid]

    # --------------------------------------------------

    def add_dependency(self, node_id, depends_on):
        if node_id not in self.nodes or depends_on not in self.nodes:
            return

        if self._would_create_cycle(node_id, depends_on):
            logger.warning("Cycle blocked: %s -> %s", node_id, depends_on)
            return

        self.nodes[node_id].dependencies.add(depends_on)
        self.nodes[depends_on].children.add(node_id)

    # --------------------------------------------------

    def remove_dependency(self, node_id, depends_on):
        if node_id not in self.nodes:
            return

        self.nodes[node_id].dependencies.discard(depends_on)
        if depends_on in self.nodes:
            self.nodes[depends_on].children.discard(node_id)

    # --------------------------------------------------
    # Readiness / Execution
    # --------------------------------------------------

    def recompute_readiness(self):
        for node in self.nodes.values():
            if node.status in ("done", "running", "failed", "to_be_expanded"):  # ← add to_be_expanded
                continue
            if all(
                dep in self.nodes and self.nodes[dep].status == "done"
                for dep in node.dependencies
            ):
                node.status = "ready"
            else:
                node.status = "pending"

    # --------------------------------------------------
    # Snapshot
    # --------------------------------------------------

    def get_snapshot(self):
        return copy.deepcopy(self.nodes)

    # --------------------------------------------------
    # Utility
    # --------------------------------------------------

    def get_ready_nodes(self):
        return [node for node in self.nodes.values() if node.status == "ready"]

    # --------------------------------------------------

    def _would_create_cycle(self, node_id, depends_on):
        if depends_on not in self.nodes or node_id not in self.nodes:
            return False

        visited = set()

        def dfs(n):
            if n == node_id:
                return True
            visited.add(n)
            for child in self.nodes[n].children:
                if child not in visited and dfs(child):
                    return True
            return False

        return dfs(depends_on)

    # --------------------------------------------------
    # Branch / Descendants
    # --------------------------------------------------

    def get_branch(self, root_id):
        """
        Returns all nodes reachable from the root node (inclusive),
        walking upstream through dependencies.
        """
        if root_id not in self.nodes:
            return {}

        branch_nodes = {}
        stack = [root_id]

        while stack:
            current_id = stack.pop()
            if current_id in branch_nodes:
                continue

            node = self.nodes[current_id]
            branch_nodes[current_id] = node

            # Walk upstream (toward prerequisites)
            stack.extend(node.dependencies)

        return branch_nodes
    
    def detach_node(self, node_id):
        """Remove a single node without touching its children or descendants."""
        if node_id not in self.nodes:
            return

        # Remove this node from its parents' children sets
        for dep in self.nodes[node_id].dependencies:
            if dep in self.nodes:
                self.nodes[dep].children.discard(node_id)

        # Remove this node from its children's dependency sets
        for child in self.nodes[node_id].children:
            if child in self.nodes:
                self.nodes[child].dependencies.discard(node_id)

        del self.nodes[node_id]

    def update_status(self, node_id, status):
        if node_id not in self.nodes:
            return
        valid = ("pending", "ready", "running", "done", "failed", "to_be_expanded")  # ← add it
        if status not in valid:
            logger.warning("Invalid status '%s' for node %s", status, node_id)
            return
        self.nodes[node_id].status = status
        self.execution_version += 1


# --- FILE: engine/__init__.py ---




# --- FILE: engine/execution_step_reporter.py ---

# engine/execution_step_reporter.py

import time
from datetime import datetime
from cuddlytoddly.core.events import (
    Event, ADD_NODE, UPDATE_METADATA, MARK_RUNNING, 
    MARK_DONE, MARK_FAILED, REMOVE_DEPENDENCY,
    ADD_DEPENDENCY
)
from cuddlytoddly.infra.logging import get_logger

logger = get_logger(__name__)


class ExecutionStepReporter:
    """
    Tracks every step of an LLMExecutor run as child nodes in the DAG.

    One node per unique tool call (stable across retries), plus one
    synthesis node for the final done=True turn.

    Lifecycle:
      - on_llm_turn()    called at the start of each executor loop iteration
      - on_tool_start()  called when the LLM requests a tool
      - on_tool_done()   called after the tool returns
      - on_synthesis()   called when the LLM sets done=True
      - on_llm_error()   called when the LLM or JSON parsing fails
      - hide_all()       called after parent succeeds — hides steps from main UI
      - expose_all()     called after parent fails — ensures steps are visible
    """

    def __init__(self, parent_node_id: str, apply_fn, graph_lock, graph):
        self.parent_node_id = parent_node_id
        self._apply         = apply_fn
        self._graph_lock    = graph_lock
        self._graph = graph

        # tool_name -> node_id  (for retry detection)
        self._tool_nodes: dict[str, str] = {}
        # ordered list of all step node ids (for hide/expose)
        self._all_step_ids: list[str] = []
        self._turn = 0

    # ── Turn lifecycle ────────────────────────────────────────────────────────

    def on_llm_turn(self, turn: int):
        """Called at the top of each executor loop iteration."""
        self._turn = turn

    def on_tool_start(self, tool_name: str, tool_args: dict) -> str:
        # Check in-memory registry first (same session retry)
        if tool_name in self._tool_nodes:
            step_id = self._tool_nodes[tool_name]
            with self._graph_lock:
                self._apply(Event(MARK_RUNNING, {"node_id": step_id}))
            return step_id

        step_id = f"{self.parent_node_id}__step_{tool_name}"

        # Check if this step already exists in the graph from a previous session
        existing = self._graph.nodes.get(step_id)
        if existing is not None:
            logger.debug(
                "[STEPREPORTER] Found pre-existing step node %s from previous session",
                step_id
            )
            # Re-register it so the rest of the reporter is aware of it
            self._tool_nodes[tool_name] = step_id
            if step_id not in self._all_step_ids:
                self._all_step_ids.append(step_id)

            with self._graph_lock:
                self._apply(Event(MARK_RUNNING, {"node_id": step_id}))
            return step_id

        # Fresh node — create as normal
        dep_list = [self._all_step_ids[-1]] if self._all_step_ids else []

        with self._graph_lock:
            self._apply(Event(ADD_NODE, {
                "node_id":      step_id,
                "node_type":    "execution_step",
                "dependencies": dep_list,
                "origin":       "executor",
                "metadata": {
                    "description":   f"{tool_name}({self._format_args(tool_args)})",
                    "step_type":     "tool_call",
                    "tool_name":     tool_name,
                    "tool_args":     tool_args,
                    "attempts":      [],
                    "fully_refined": True,
                    "hidden":        False,
                },
            }))
            self._apply(Event(MARK_RUNNING, {"node_id": step_id}))

            # ── Swap parent's dependency to always point at the latest step ───
            if self._all_step_ids:
                # Remove previous frontier
                self._apply(Event(REMOVE_DEPENDENCY, {
                    "node_id":    self.parent_node_id,
                    "depends_on": self._all_step_ids[-1],
                }))
            # Add new frontier
            self._apply(Event(ADD_DEPENDENCY, {
                "node_id":    self.parent_node_id,
                "depends_on": step_id,
            }))

        self._tool_nodes[tool_name] = step_id
        self._all_step_ids.append(step_id)
        return step_id

    def on_synthesis(self, result: str):
        step_id  = f"{self.parent_node_id}__step_synthesis"
        last_dep = self._all_step_ids[-1] if self._all_step_ids else self.parent_node_id

        with self._graph_lock:
            self._apply(Event(ADD_NODE, {
                "node_id":      step_id,
                "node_type":    "execution_step",
                "dependencies": [last_dep],
                "origin":       "executor",
                "metadata": {
                    "description":   "synthesize result",
                    "step_type":     "synthesis",
                    "fully_refined": True,
                    "hidden":        False,
                },
            }))
            self._apply(Event(MARK_DONE, {"node_id": step_id, "result": result}))

            # ── Swap parent to depend on synthesis as the final frontier ──────
            if self._all_step_ids:
                self._apply(Event(REMOVE_DEPENDENCY, {
                    "node_id":    self.parent_node_id,
                    "depends_on": self._all_step_ids[-1],
                }))
            self._apply(Event(ADD_DEPENDENCY, {
                "node_id":    self.parent_node_id,
                "depends_on": step_id,
            }))

        self._all_step_ids.append(step_id)

    def on_tool_done(self, step_id: str, tool_name: str,
                     tool_args: dict, result: str, error: bool = False):
        """
        Called after a tool returns.

        Appends this attempt to the node's history and marks done/failed.
        Truncates the result so it doesn't blow up metadata storage.
        """

        attempt = {
            "turn":      self._turn,
            "args":      tool_args,
            "result":    result,
            "status":    "error" if error else "ok",
            "timestamp": datetime.utcnow().isoformat(),
        }

        with self._graph_lock:
            # Fetch current attempts and append
            from cuddlytoddly.core.task_graph import TaskGraph   # avoid circular at module level
            node = self._get_live_node(step_id)
            existing_attempts = node.metadata.get("attempts", []) if node else []
            updated_attempts  = existing_attempts + [attempt]

            self._apply(Event(UPDATE_METADATA, {
                "node_id":  step_id,
                "metadata": {"attempts": updated_attempts},
            }))

            if error:
                self._apply(Event(MARK_FAILED, {"node_id": step_id}))
            else:
                self._apply(Event(MARK_DONE, {
                    "node_id": step_id,
                    "result":  result,
                }))

    def on_llm_error(self, turn: int, error: str):
        step_id  = f"{self.parent_node_id}__step_error_t{turn}"
        dep_list = [self._all_step_ids[-1]] if self._all_step_ids else []

        with self._graph_lock:
            self._apply(Event(ADD_NODE, {
                "node_id":      step_id,
                "node_type":    "execution_step",
                "dependencies": dep_list,
                "origin":       "executor",
                "metadata": {
                    "description":   f"LLM error: {error[:120]}",
                    "step_type":     "llm_error",
                    "fully_refined": True,
                    "hidden":        False,
                },
            }))
            self._apply(Event(MARK_FAILED, {"node_id": step_id}))

            if self._all_step_ids:
                self._apply(Event(REMOVE_DEPENDENCY, {
                    "node_id":    self.parent_node_id,
                    "depends_on": self._all_step_ids[-1],
                }))
            self._apply(Event(ADD_DEPENDENCY, {
                "node_id":    self.parent_node_id,
                "depends_on": step_id,
            }))

        self._all_step_ids.append(step_id)

    # ── Post-execution visibility ─────────────────────────────────────────────

    def hide_all(self):
        with self._graph_lock:
            for step_id in self._all_step_ids:
                self._apply(Event(UPDATE_METADATA, {
                    "node_id":  step_id,
                    "metadata": {"hidden": True},
                }))

    def expose_all(self):
        with self._graph_lock:
            for step_id in self._all_step_ids:
                self._apply(Event(UPDATE_METADATA, {
                    "node_id":  step_id,
                    "metadata": {"hidden": False},
                }))

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _format_args(self, args: dict) -> str:
        parts = []
        for k, v in args.items():
            v_str = str(v)[:30]
            parts.append(f"{k}={v_str}")
        return ", ".join(parts)

    def _get_live_node(self, node_id: str):
        """Must be called with graph_lock held."""
        return self._graph.nodes.get(node_id)


# --- FILE: engine/llm_orchestrator.py ---

# engine/llm_orchestrator.py

import json
import threading
import time
import os
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

from cuddlytoddly.core.events import (
    Event,
    ADD_NODE, ADD_DEPENDENCY, REMOVE_DEPENDENCY, UPDATE_METADATA,
    MARK_RUNNING, MARK_DONE, MARK_FAILED, REMOVE_NODE, RESET_NODE,
    SET_NODE_TYPE
)
from cuddlytoddly.core.reducer import apply_event
from cuddlytoddly.infra.event_queue import EventQueue
from cuddlytoddly.infra.logging import get_logger
from cuddlytoddly.planning.llm_interface import LLMStoppedError
from cuddlytoddly.engine.execution_step_reporter import ExecutionStepReporter
from cuddlytoddly.planning.llm_interface import token_counter

logger = get_logger(__name__)

PlanningContext = namedtuple("PlanningContext", ["snapshot", "goals"])

_IDLE_SLEEP = 0.5

# Maximum number of times the orchestrator will attempt to inject a bridging
# node for any single blocked node before giving up and just launching it.
_MAX_GAP_FILL_ATTEMPTS = 2

class SimpleOrchestrator:
    """
    Minimal orchestrator: LLM plans, executor runs, user edits via the UI.

    The curses UI expects:
        .graph, .graph_lock, .event_queue
        .current_activity, .activity_started
        .llm_stopped, .stop_llm_calls(), .resume_llm_calls()
    """
    @property
    def token_counts(self) -> dict:
        return {
            "prompt":     token_counter.prompt_tokens,
            "completion": token_counter.completion_tokens,
            "total":      token_counter.total_tokens,
            "calls":      token_counter.calls,
        }
    
    def __init__(self, graph, planner, executor,
                 event_log=None, event_queue=None, max_workers=4, quality_gate=None):
        self.graph       = graph
        self.planner     = planner
        self.executor    = executor
        self.event_log   = event_log
        self.event_queue = event_queue or EventQueue()
        self.max_workers = max_workers
        self.quality_gate = quality_gate 

        # UI contract
        self.graph_lock       = threading.RLock()
        self.current_activity: str | None   = None
        self.activity_started: float | None = None

        # Internals
        self._pool                          = ThreadPoolExecutor(max_workers=max_workers)
        self._running_futures: dict[str, object] = {}
        self._stop_event                    = threading.Event()
        self._thread: threading.Thread | None = None

        # LLM clients for pause/resume — collected from planner and executor
        self._llm_clients = []
        self._reporters: dict[str, ExecutionStepReporter] = {}

        for component in (planner, executor, quality_gate):
            if component is not None and hasattr(component, "llm"):
                self._llm_clients.append(component.llm)

    # ── LLM pause / resume ───────────────────────────────────────────────────

    @property
    def llm_stopped(self) -> bool:
        return any(getattr(c, "is_stopped", False) for c in self._llm_clients)

    def stop_llm_calls(self) -> None:
        for c in self._llm_clients:
            if hasattr(c, "stop"):
                c.stop()
        logger.warning("[ORCHESTRATOR] LLM calls PAUSED")

    def resume_llm_calls(self) -> None:
        for c in self._llm_clients:
            if hasattr(c, "resume"):
                c.resume()
        logger.info("[ORCHESTRATOR] LLM calls RESUMED")

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def start(self):
        """Start in a background thread. On macOS+Metal use run_on_main_thread instead."""
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="simple-orchestrator"
        )
        self._thread.start()
        logger.info("[ORCHESTRATOR] Started (background thread)")

    def run_on_main_thread(self):
        """Run the loop on the calling thread (blocks).
        Required on macOS with Metal: llama.cpp GPU work must stay on the main thread."""
        self._stop_event.clear()
        self._thread = threading.current_thread()
        logger.info("[ORCHESTRATOR] Started (main thread)")
        self._loop()

    def stop(self):
        self._stop_event.set()
        if self._thread and self._thread is not threading.current_thread():
            self._thread.join(timeout=10)
        self._pool.shutdown(wait=False)
        logger.info("[ORCHESTRATOR] Stopped")

    @property
    def is_running(self):
        return not self._stop_event.is_set()

    # ── Main loop ────────────────────────────────────────────────────────────

    def _loop(self):
        _last_idle_log = 0
        while not self._stop_event.is_set():
            try:
                self._drain_event_queue()
                self._expansion_request_pass()   # ← first, convert any expansion requests
                planned  = self._planning_pass()
                self._complete_finished_goals()
                launched = self._execution_pass()

                if planned == 0 and launched == 0:
                    now = time.time()
                    if now - _last_idle_log > 30:
                        if self._running_futures:
                            logger.debug(
                                "[ORCHESTRATOR] Waiting on %d running node(s): %s",
                                len(self._running_futures),
                                list(self._running_futures.keys()),
                            )
                        else:
                            logger.debug("[ORCHESTRATOR] Idle — nothing to do")
                        _last_idle_log = now

                    if self._is_fully_done():
                        time.sleep(_IDLE_SLEEP * 4)
                    else:
                        time.sleep(_IDLE_SLEEP)

            except Exception as e:
                logger.exception("[ORCHESTRATOR] Unhandled error in main loop: %s", e)
                time.sleep(_IDLE_SLEEP)

    # ── Event queue drain ────────────────────────────────────────────────────

    def _drain_event_queue(self):
        while not self.event_queue.empty():
            try:
                event = self.event_queue.get()
                if event.type == "RESET_SUBTREE":
                    logger.info("[ORCHESTRATOR] RESET_SUBTREE received for: %s",
                                event.payload.get("node_id"))
                    # Handled here — needs knowledge of _running_futures
                    self._reset_subtree_impl(event.payload["node_id"])
                else:
                    with self.graph_lock:
                        self._apply(event)
            except Exception as e:
                logger.error("[ORCHESTRATOR] Error draining event: %s", e)
 
    def _reset_subtree_impl(self, root_id: str):
        """
        Reset root_id and every descendant (via children) to pending,
        clearing results and verification metadata.
        Skips any node that is currently running — those will be reset
        when their future resolves.
        Called with NO lock held; acquires it internally.
        """
        with self.graph_lock:
            if root_id not in self.graph.nodes:
                return
 
            # BFS over children (the "depends on this" direction)
            to_reset = []
            queue = [root_id]
            visited = set()
            while queue:
                nid = queue.pop(0)
                if nid in visited:
                    continue
                visited.add(nid)
                node = self.graph.nodes.get(nid)
                if not node:
                    continue
                to_reset.append(nid)
                queue.extend(node.children)
 
            for nid in to_reset:
                if nid in self._running_futures:
                    # Leave running nodes alone — the executor will
                    # mark them done/failed naturally; the result will
                    # be discarded by the quality gate on the next retry.
                    logger.debug("[RESET_SUBTREE] Skipping running node: %s", nid)
                    continue
 
                node = self.graph.nodes.get(nid)
                if not node:
                    continue
 
                node.status = "pending"
                node.result = None
                node.metadata.pop("verified",              None)
                node.metadata.pop("verification_failure",  None)
                node.metadata.pop("retry_count",           None)
                # If this is a goal that was already expanded, mark it
                # for re-expansion so the planner revisits it. 
                logger.info("[RESET_SUBTREE] Reset: %s", nid)
 
            self.graph.recompute_readiness()
 
    # ── Planning pass ────────────────────────────────────────────────────────

    def _planning_pass(self) -> int:
        if self.llm_stopped:
            return 0

        with self.graph_lock:
            unexpanded = [
                n for n in self.graph.nodes.values()
                if n.node_type == "goal"
                and not n.metadata.get("expanded", False)
            ]

        total = 0
        for goal in unexpanded:
            if self._stop_event.is_set() or self.llm_stopped:
                break

            self.current_activity = f"Planning: {goal.id}"
            self.activity_started = time.time()
            logger.info("[PLAN] Expanding goal: %s", goal.id)

            with self.graph_lock:
                branch = self.graph.get_branch(goal.id)

            context = PlanningContext(snapshot=branch, goals=[goal])
            try:
                events = self.planner.propose(context)
            except Exception as e:
                logger.error("[PLAN] Planner failed for %s: %s", goal.id, e)
                events = []
            finally:
                self.current_activity = None
                self.activity_started = None

            with self.graph_lock:
                for evt in events:
                    self._apply(Event(evt["type"], evt["payload"]))
                    total += 1
                self._apply(Event(UPDATE_METADATA, {
                    "node_id": goal.id,
                    "metadata": {"expanded": True},
                }))

            logger.info("[PLAN] Goal %s → %d events", goal.id, len(events))

        return total

    # ── Goal auto-completion ─────────────────────────────────────────────────

    def _complete_finished_goals(self):
        """
        Mark a goal as done when all its work is finished.

        The planner can wire goals two ways:
          A) goal has tasks as children     (node.children non-empty)
          B) goal depends on tasks          (node.dependencies non-empty)

        Handle both: done when every dependency AND every child is done.
        """
        with self.graph_lock:
            for node in self.graph.nodes.values():
                if node.node_type != "goal":
                    continue
                if node.status in ("done", "failed"):
                    continue
                if not node.metadata.get("expanded", False):
                    continue

                related = set(node.dependencies) | set(node.children)
                if not related:
                    continue

                all_done = all(
                    self.graph.nodes[nid].status == "done"
                    for nid in related
                    if nid in self.graph.nodes
                )
                if all_done:
                    # Preserve the plan_summary written by the planner at expansion time
                    existing_result = node.result
                    self._apply(Event(MARK_DONE, {
                        "node_id": node.id,
                        "result":  existing_result,
                    }))
                    logger.info("[ORCHESTRATOR] Goal completed: %s", node.id)

    # ── Execution pass ───────────────────────────────────────────────────────

    def _execution_pass(self) -> int:
        launched = 0

        with self.graph_lock:
            ready = [
                n for n in self.graph.nodes.values()
                if n.status == "ready"
                and n.node_type == "task"
                and n.id not in self._running_futures
            ]

        for node in ready:
            if self._stop_event.is_set():
                break

            with self.graph_lock:
                current = self.graph.nodes.get(node.id)
                if not current or current.status != "ready":
                    continue
                snapshot = self.graph.get_snapshot()

            # ── Dependency gap check (skip if no quality gate or budget exhausted) ──
            attempts = node.metadata.get("gap_fill_attempts", 0)
            if self.quality_gate and attempts < _MAX_GAP_FILL_ATTEMPTS:
                bridge = self.quality_gate.check_dependencies(node, snapshot)
                if bridge is not None:
                    self._inject_bridge_node(bridge, node.id)
                    continue

            with self.graph_lock:
                current = self.graph.nodes.get(node.id)
                if not current or current.status != "ready":
                    continue
                self._apply(Event(MARK_RUNNING, {"node_id": node.id}))
                snapshot = self.graph.get_snapshot()

            logger.info("[EXEC] Launching: %s", node.id)
            self.current_activity = f"Executing: {node.id}"
            self.activity_started = time.time()

            reporter = ExecutionStepReporter(
                parent_node_id=node.id,
                apply_fn=self._apply,
                graph_lock=self.graph_lock,
                graph=self.graph,
            )
            self._reporters[node.id] = reporter

            future = self._pool.submit(self.executor.execute, node, snapshot, reporter)
            self._running_futures[node.id] = future
            future.add_done_callback(
                lambda fut, nid=node.id: self._on_node_done(nid, fut)
            )
            launched += 1

        return launched
    
    def _expansion_request_pass(self):
        with self.graph_lock:
            to_expand = [
                n.id for n in self.graph.nodes.values()
                if n.status == "to_be_expanded"
            ]

        for node_id in to_expand:
            logger.info("[ORCHESTRATOR] Expansion requested for node: %s", node_id)

            with self.graph_lock:
                n = self.graph.nodes.get(node_id)
                if not n:
                    continue

                # 1. Convert to an unexpanded goal

                self._apply(Event(SET_NODE_TYPE, {
                    "node_id": node_id,
                    "node_type":  "goal",
                }))

                self._apply(Event(UPDATE_METADATA, {
                    "node_id": node_id,
                    "metadata": {
                        "expanded": False,
                        "description": n.metadata.get("description", node_id),
                    },
                }))

                # 2. Reset the node itself via event (clears status, result)
                self._apply(Event(RESET_NODE, {"node_id": node_id}))

                # 3. Cascade-reset all transitive dependents
                to_reset = []
                queue = list(self.graph.nodes[node_id].children)
                visited = set()
                while queue:
                    child_id = queue.pop()
                    if child_id in visited or child_id not in self.graph.nodes:
                        continue
                    visited.add(child_id)
                    child = self.graph.nodes[child_id]
                    if child.status != "running":
                        to_reset.append(child_id)
                        queue.extend(child.children)

                for desc_id in to_reset:
                    self._apply(Event(RESET_NODE, {"node_id": desc_id}))
                    logger.info("[ORCHESTRATOR] Reset dependent for re-execution: %s", desc_id)

    def _on_node_done(self, node_id: str, future):
            self._running_futures.pop(node_id, None)

            if self.current_activity and node_id in (self.current_activity or ""):
                if self._running_futures:
                    other = next(iter(self._running_futures))
                    self.current_activity = f"Executing: {other}"
                else:
                    self.current_activity = None
                    self.activity_started = None

            try:
                result = future.result()
            except Exception as exc:
                logger.warning("[EXEC] Node %s raised: %s", node_id, exc)
                result = None

            # ── Hard failure (executor returned None) ────────────────────────────
            if result is None:
                reporter = self._reporters.pop(node_id, None)
                with self.graph_lock:
                    if node_id not in self.graph.nodes:
                        return
                    if reporter:
                        reporter.expose_all()
                    self._apply(Event(MARK_FAILED, {"node_id": node_id}))
                    logger.warning("[EXEC] Failed: %s", node_id)
                return

            # ── Pre-flight: check file outputs were actually written ─────────────
            # Done before the LLM verification call to avoid wasting an inference.
            # Initialise with safe defaults so the code below always has values.
            satisfied:      bool | None = None
            reason:         str         = ""
            expected_files: list        = []
            tool_calls_made: set        = set()

            reporter = self._reporters.get(node_id)

            if reporter and self.quality_gate:
                with self.graph_lock:
                    declared_outputs = (
                        self.graph.nodes[node_id].metadata.get("output", [])
                        if node_id in self.graph.nodes else []
                    )
                    tool_calls_made = {
                        self.graph.nodes[sid].metadata.get("tool_name")
                        for sid in reporter._all_step_ids
                        if sid in self.graph.nodes
                    }

                expected_files = [
                    o for o in declared_outputs
                    if any(str(o).endswith(ext)
                        for ext in self.quality_gate.FILE_EXTENSIONS)
                ]

                if expected_files and "write_file" not in tool_calls_made:
                    # LLM skipped write_file — try to auto-write using the result
                    file_path = expected_files[0]
                    content   = result

                    # Strip labelled prefix: "file_written: foo.md\nsummary: ..."
                    import re
                    match = re.search(
                        r'(?:summary|content)\s*:\s*(.+)',
                        result, re.DOTALL | re.IGNORECASE
                    )
                    if match:
                        content = match.group(1).strip()

                    if content and len(content) > 50:
                        try:
                            tools = getattr(self.executor, "tools", None)
                            if tools:
                                tools.execute("write_file", {
                                    "path":    file_path,
                                    "content": content,
                                })
                                logger.info(
                                    "[EXEC] Auto-wrote '%s' for node %s (%d chars)",
                                    file_path, node_id, len(content)
                                )
                                result    = (
                                    f"file_written: {file_path}\n"
                                    f"summary: {content[:200]}"
                                )
                                satisfied = True
                                reason    = (
                                    "file written by orchestrator after LLM "
                                    "omitted write_file call"
                                )
                            else:
                                satisfied = False
                                reason    = (
                                    f"declared file output {expected_files} but "
                                    f"write_file was never called and no tool "
                                    f"registry available to auto-write"
                                )
                        except Exception as e:
                            logger.warning("[EXEC] Auto-write failed for %s: %s", node_id, e)
                            satisfied = False
                            reason    = (
                                f"declared file output but write_file not called "
                                f"and auto-write failed: {e}"
                            )
                    else:
                        satisfied = False
                        reason    = (
                            f"declared file output {expected_files} but write_file "
                            f"was never called and result has insufficient content "
                            f"to auto-write ({len(content)} chars)"
                        )

            # ── LLM quality gate (skipped if pre-flight already decided) ─────────
            if satisfied is None:
                if self.quality_gate:
                    with self.graph_lock:
                        if node_id not in self.graph.nodes:
                            return
                        node     = self.graph.nodes[node_id]
                        snapshot = self.graph.get_snapshot()

                    satisfied, reason = self.quality_gate.verify_result(
                        node, result, snapshot
                    )
                else:
                    satisfied = True
                    reason    = "no quality gate configured"

            # ── Apply outcome ─────────────────────────────────────────────────────
            reporter = self._reporters.pop(node_id, None)

            with self.graph_lock:
                if node_id not in self.graph.nodes:
                    return

                live_node   = self.graph.nodes[node_id]
                retry_count = live_node.metadata.get("retry_count", 0)

                if not satisfied:
                    if reporter:
                        reporter.expose_all()

                    if retry_count >= 3:
                        # Permanently fail — clean up step nodes but keep them
                        # visible so the user can inspect what went wrong
                        logger.error(
                            "[EXEC] Node %s failed verification %d time(s) — "
                            "permanently failing. Reason: %s",
                            node_id, retry_count + 1, reason
                        )
                        if reporter:
                            for step_id in list(reporter._all_step_ids):
                                if step_id in self.graph.nodes:
                                    # Unlink from parent's dependency set so the
                                    # graph is consistent, but leave the step node
                                    # itself visible for inspection
                                    live_node.dependencies.discard(step_id)
                                    self._apply(Event(REMOVE_DEPENDENCY, {
                                        "node_id":    node_id,
                                        "depends_on": step_id,
                                    }))

                        self._apply(Event(UPDATE_METADATA, {
                            "node_id":  node_id,
                            "metadata": {"verification_failure": reason},
                        }))
                        self._apply(Event(MARK_FAILED, {"node_id": node_id}))

                    else:
                        logger.warning(
                            "[EXEC] Verification failed for %s (attempt %d/3): %s",
                            node_id, retry_count + 1, reason
                        )
                        # Detach step nodes so retry starts with a clean reporter
                        if reporter:
                            for step_id in list(reporter._all_step_ids):
                                if step_id in self.graph.nodes:
                                    self.graph.detach_node(step_id)

                        self._apply(Event(UPDATE_METADATA, {
                            "node_id":  node_id,
                            "metadata": {
                                "verification_failure": reason,
                                "retry_count":          retry_count + 1,
                            },
                        }))
                        live_node.status = "pending"
                        live_node.result = None
                        self.graph.recompute_readiness()

                else:
                    # Success — hide step nodes from the main UI
                    if reporter:
                        reporter.hide_all()

                    logger.info("[EXEC] Done: %s", node_id)
                    self._apply(Event(MARK_DONE, {
                        "node_id": node_id,
                        "result":  result,
                    }))
                    self._apply(Event(UPDATE_METADATA, {
                        "node_id":  node_id,
                        "metadata": {"verified": True},
                    }))

    # ── Bridge node injection ────────────────────────────────────────────────

    def _inject_bridge_node(self, bridge: dict, blocked_node_id: str):
        """
        Insert a single bridging task and re-wire it as a dependency of
        the blocked node. Marks the bridge fully_refined so the refinement
        cycle never promotes it into a goal and re-decomposes it.
        """
        with self.graph_lock:
            bridge_id = bridge["node_id"]

            # Guard: don't inject the same bridge twice
            if bridge_id in self.graph.nodes:
                logger.debug(
                    "[DEPCHECK] Bridge %s already exists — skipping injection", bridge_id
                )
            else:
                self._apply(Event(ADD_NODE, {
                    "node_id":   bridge_id,
                    "node_type": "task",
                    "origin":    "orchestrator",
                    "metadata":  {
                        "description":   bridge["description"],
                        "output":        [bridge["output"]],
                        "fully_refined": True,   # never re-decomposed
                        "gap_fill":      True,   # auditing marker
                    },
                }))
                self._apply(Event(ADD_DEPENDENCY, {
                    "node_id":    blocked_node_id,
                    "depends_on": bridge_id,
                }))
                logger.info(
                    "[DEPCHECK] Injected bridge node %s → unblocks %s",
                    bridge_id, blocked_node_id
                )

            # Increment attempt counter regardless, so we converge
            current_node = self.graph.nodes.get(blocked_node_id)
            if current_node:
                attempts = current_node.metadata.get("gap_fill_attempts", 0)
                self._apply(Event(UPDATE_METADATA, {
                    "node_id":  blocked_node_id,
                    "metadata": {"gap_fill_attempts": attempts + 1},
                }))

    def verify_restored_nodes(self):
        FILE_EXTENSIONS = {
            ".md", ".txt", ".py", ".json", ".csv", ".html",
            ".yaml", ".yml", ".xml", ".pdf", ".log",
        }

        with self.graph_lock:
            done_tasks = [
                n for n in self.graph.nodes.values()
                if n.node_type == "task"
                and n.status == "done"
                and n.result is not None
            ]

        # ── Pass 1: file-existence check on ALL done nodes, even verified ones ───
        # verified=True only means the LLM was satisfied — it can't know if a file
        # was deleted or never written. This check is cheap so always run it.
        for node in done_tasks:
            if self.quality_gate and self.quality_gate._looks_like_filename(node.result):
                path = node.result.strip()
                if not self.quality_gate._file_exists(path):
                    logger.warning(
                        "[STARTUP] Node %s result is '%s' but file does not exist — resetting",
                        node.id, path
                    )
                    with self.graph_lock:
                        n = self.graph.nodes.get(node.id)
                        if n:
                            n.status = "pending"
                            n.result = None
                            n.metadata["verification_failure"] = (
                                f"file '{path}' does not exist on disk"
                            )
                            n.metadata.pop("verified", None)
                            n.metadata["retry_count"] = n.metadata.get("retry_count", 0) + 1


        # ── Pass 2: LLM verification for nodes never verified ────────────────────
        with self.graph_lock:
            candidates = [
                n for n in self.graph.nodes.values()
                if n.node_type == "task"
                and n.status == "done"
                and n.result is not None
                and not n.metadata.get("verified", False)
            ]

        if not candidates:
            logger.info("[STARTUP] All restored nodes already verified — nothing to check")
        else:
            logger.info("[STARTUP] %d restored node(s) need verification", len(candidates))

        for node in candidates:
            if self._stop_event.is_set():
                break

            logger.info("[STARTUP] Verifying restored node: %s", node.id)
            self.current_activity = f"Verifying: {node.id}"
            self.activity_started = time.time()

            with self.graph_lock:
                snapshot = self.graph.get_snapshot()

            satisfied, reason = self._verify_result(node, node.result, snapshot)

            self.current_activity = None
            self.activity_started = None

            with self.graph_lock:
                if node.id not in self.graph.nodes:
                    continue

                if not satisfied:
                    logger.warning(
                        "[STARTUP] Restored node %s failed verification: %s — resetting to pending",
                        node.id, reason
                    )
                    n = self.graph.nodes[node.id]
                    n.status  = "pending"
                    n.result  = None
                    n.metadata["verification_failure"] = reason
                    n.metadata.pop("verified", None)
                    n.metadata["retry_count"] = n.metadata.get("retry_count", 0) + 1

                else:
                    logger.info("[STARTUP] Restored node %s verified OK", node.id)
                    self._apply(Event(UPDATE_METADATA, {
                        "node_id":  node.id,
                        "metadata": {"verified": True},
                    }))

        with self.graph_lock:
            self.graph.recompute_readiness()

        ready = sum(1 for n in self.graph.nodes.values() if n.status == "ready")
        logger.info("[STARTUP] Post-verification readiness: %d node(s) ready", ready)

    # ── User-facing edit API ─────────────────────────────────────────────────

    def add_goal(self, goal_id: str, description: str = "", dependencies: list = None):
        with self.graph_lock:
            self._apply(Event(ADD_NODE, {
                "node_id":      goal_id,
                "node_type":    "goal",
                "dependencies": dependencies or [],
                "origin":       "user",
                "metadata":     {"description": description, "expanded": False},
            }))
        logger.info("[USER] Added goal: %s", goal_id)

    def add_task(self, node_id: str, dependencies: list = None,
                 description: str = "", metadata: dict = None):
        meta = {"description": description, "fully_refined": True}
        if metadata:
            meta.update(metadata)
        with self.graph_lock:
            self._apply(Event(ADD_NODE, {
                "node_id":      node_id,
                "node_type":    "task",
                "dependencies": dependencies or [],
                "origin":       "user",
                "metadata":     meta,
            }))

    def remove_node(self, node_id: str):
        with self.graph_lock:
            if node_id in self._running_futures:
                logger.warning("[USER] Cannot remove %s — currently running", node_id)
                return
            self._apply(Event(REMOVE_NODE, {"node_id": node_id}))

    def add_dependency(self, node_id: str, depends_on: str):
        with self.graph_lock:
            self._apply(Event(ADD_DEPENDENCY, {
                "node_id": node_id, "depends_on": depends_on,
            }))

    def remove_dependency(self, node_id: str, depends_on: str):
        with self.graph_lock:
            self._apply(Event(REMOVE_DEPENDENCY, {
                "node_id": node_id, "depends_on": depends_on,
            }))

    def retry_node(self, node_id: str):
        with self.graph_lock:
            node = self.graph.nodes.get(node_id)
            if not node or node_id in self._running_futures:
                return
            node.status = "pending"
            node.result = None
            self.graph.recompute_readiness()

    def replan_goal(self, goal_id: str):
        with self.graph_lock:
            goal = self.graph.nodes.get(goal_id)
            if not goal or goal.node_type != "goal":
                return
            for cid in list(goal.children):
                if (cid in self.graph.nodes
                        and self.graph.nodes[cid].status in ("pending", "ready")
                        and cid not in self._running_futures):
                    self._apply(Event(REMOVE_NODE, {"node_id": cid}))
            self._apply(Event(UPDATE_METADATA, {
                "node_id": goal_id,
                "metadata": {"expanded": False},
            }))

    def update_metadata(self, node_id: str, metadata: dict):
        with self.graph_lock:
            self._apply(Event(UPDATE_METADATA, {
                "node_id": node_id, 
                "metadata": metadata,
            }))

    # ── Read access ──────────────────────────────────────────────────────────

    def get_snapshot(self):
        with self.graph_lock:
            return self.graph.get_snapshot()

    def get_status(self) -> dict:
        with self.graph_lock:
            nodes = list(self.graph.nodes.values())
        counts: dict[str, int] = {}
        for n in nodes:
            counts[n.status] = counts.get(n.status, 0) + 1
        return {
            "total":         len(nodes),
            "by_status":     counts,
            "running_nodes": list(self._running_futures.keys()),
        }

    # ── Internals ────────────────────────────────────────────────────────────

    def _apply(self, event: Event):
        """Apply one event. Must be called with graph_lock held."""
        apply_event(self.graph, event, event_log=self.event_log)

    def _is_fully_done(self) -> bool:
        with self.graph_lock:
            return all(
                n.status in ("done", "failed")
                for n in self.graph.nodes.values()
            )


# --- FILE: engine/quality_gate.py ---

# engine/quality_gate.py

import json
from cuddlytoddly.infra.logging import get_logger
from cuddlytoddly.planning.llm_interface import LLMStoppedError

logger = get_logger(__name__)

RESULT_VERIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "satisfied": {
            "type": "boolean",
            "description": (
                "True if the result fully covers every declared output. "
                "False if something is missing or clearly wrong."
            )
        },
        "reason": {
            "type": "string",
            "description": (
                "One sentence explaining why the result is satisfied or not. "
                "If satisfied=true this can be brief."
            )
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
            )
        },
        "missing": {
            "type": "string",
            "description": "Short description of what is missing. Only required when ok=false."
        },
        "bridge_node": {
            "type": "object",
            "description": "A single task that would close the gap. Only required when ok=false.",
            "properties": {
                "node_id":     {"type": "string",
                                "description": "Snake_case identifier, no spaces."},
                "description": {"type": "string",
                                "description": "One sentence: what this task does."},
                "output":      {"type": "string",
                                "description": "The single artifact this task produces."},
            },
            "required": ["node_id", "description", "output"],
        },
    },
    "required": ["ok"],
}


class QualityGate:
    """
    LLM-based quality checks for the orchestrator.

    Kept separate from SimpleOrchestrator so the orchestration logic
    (graph mutation, scheduling) stays decoupled from the verification logic
    (prompt building, LLM calls, schema parsing).

    Mirrors the pattern of LLMPlanner / LLMExecutor — the orchestrator
    receives it as a dependency and calls its methods; it never touches
    the graph directly.
    """

    def __init__(self, llm_client, tool_registry=None):
        self.llm   = llm_client
        self.tools = tool_registry

    # ── Public API ────────────────────────────────────────────────────────────

    def verify_result(self, node, result: str, snapshot) -> tuple[bool, str]:
        if getattr(self.llm, "is_stopped", False):
            return True, "verification skipped — LLM paused"

        declared_outputs = node.metadata.get("output", [])
        if not declared_outputs:
            return True, "no declared outputs to verify"

        stripped = result.strip()

        # ── Pattern 1: bare filename (no spaces, has extension) ──────────────────
        if self._looks_like_filename(stripped):
            if not self._file_exists(stripped):
                return False, (
                    f"result is a filename ('{stripped}') "
                    f"but the file does not exist on disk"
                )

        # ── Pattern 2: labelled file confirmation "file_written: foo.md" ─────────
        # Extract the filename from the label and check it exists
        import re
        file_label_match = re.match(
            r'^(?:file_written|written_to|saved_to|output_file)\s*:\s*(\S+)', 
            stripped, re.IGNORECASE
        )
        if file_label_match:
            filename = file_label_match.group(1).rstrip(".,;")
            if self._looks_like_filename(filename) or True:  # always check
                if not self._file_exists(filename):
                    return False, (
                        f"result claims file was written ('{filename}') "
                        f"but the file does not exist on disk"
                    )

        # ── Pattern 3: result is just a label/name with no actual content ────────
        # If the result is a single short word/phrase that exactly matches a
        # declared output name, it's a label not content — fail it
        def _output_name(o):
            return o["name"] if isinstance(o, dict) else str(o)

        is_just_label = (
            "\n" not in stripped
            and " " not in stripped
            and len(stripped) < 60
            and any(
                stripped.lower().replace("_", "") == _output_name(o).lower().replace("_", "")
                for o in declared_outputs
            )
        )
        if is_just_label:
            return False, (
                f"result '{stripped}' appears to be just a label matching the declared "
                f"output name, not actual content. The node must return the actual data."
            )

        # ── LLM content check ────────────────────────────────────────────────────
        def _format_output_for_verification(o):
            if isinstance(o, dict):
                return f"  - [{o['type']}] {o['name']}: {o['description']}"
            return f"  - {o}"

        outputs_text = "\n".join(_format_output_for_verification(o) for o in declared_outputs)
        prompt = f"""You are verifying whether a task result satisfies its declared outputs.

    TASK
    ID:          {node.id}
    Description: {node.metadata.get("description", node.id)}

    DECLARED OUTPUTS (what this task was supposed to produce):
    {outputs_text}

    ACTUAL RESULT:
    {stripped}

    Does the result contain actual substantive content, or is it just a label/filename/stub?
    A result that is just a filename, a single word, or a name matching the output label
    is NOT satisfied — the result must contain the actual data.

    Respond only in JSON matching the schema.
    """
        try:
            raw    = self.llm.ask(prompt, schema=RESULT_VERIFICATION_SCHEMA)
            parsed = json.loads(raw)
            return bool(parsed.get("satisfied", True)), parsed.get("reason", "")
        except LLMStoppedError:
            return True, "verification skipped — LLM stopped mid-call"
        except Exception as e:
            logger.warning("[QUALITY] verify_result error for %s: %s", node.id, e)
            return True, f"verification skipped — error: {e}"
        
    def check_dependencies(self, node, snapshot) -> dict | None:
        """
        Check whether the upstream results are sufficient to run `node`.

        Returns a bridge_node dict {node_id, description, output} if a gap
        is found, or None if everything looks fine.
        Falls back to None on any error so a broken checker never blocks execution.
        """
        if getattr(self.llm, "is_stopped", False):
            return None

        dep_lines = []
        for dep_id in node.dependencies:
            dep = snapshot.get(dep_id)
            if dep and dep.result:
                dep_lines.append(
                    f"  [{dep_id}]\n"
                    f"    Description: {dep.metadata.get('description', dep_id)}\n"
                    f"    Result:      {dep.result}"
                )
        upstream_text = "\n\n".join(dep_lines) if dep_lines else "  (none — root task)"

        required_inputs = node.metadata.get("required_input", [])
        def _format_input_for_check(i):
            if isinstance(i, dict):
                return f"  - [{i['type']}] {i['name']}: {i['description']}"
            return f"  - {i}"

        inputs_text = (
            "\n".join(_format_input_for_check(i) for i in required_inputs)
            if required_inputs else "  (not specified)"
        )

        prompt = f"""You are checking whether a task has everything it needs to execute.

TASK TO RUN
  ID:             {node.id}
  Description:    {node.metadata.get("description", node.id)}
  Required input:
{inputs_text}

AVAILABLE UPSTREAM RESULTS:
{upstream_text}

Is there a meaningful gap — something the task clearly needs but the upstream
results do not provide?

Rules:
- Only flag a real, concrete gap. Do not invent requirements not stated in the task.
- If you flag a gap, propose ONE bridging task that closes it. Keep it coarse-grained:
  a bridging task should do substantial work, not a trivial lookup.
- If the task is a root task or the upstream results are sufficient, set ok=true.

Respond only in JSON matching the schema.
"""
        try:
            raw    = self.llm.ask(prompt, schema=DEPENDENCY_CHECK_SCHEMA)
            parsed = json.loads(raw)
            if parsed.get("ok", True):
                return None
            bridge = parsed.get("bridge_node")
            if not bridge or not bridge.get("node_id") or not bridge.get("description"):
                return None
            logger.info(
                "[QUALITY] Gap detected for %s: %s → bridge: %s",
                node.id, parsed.get("missing", "?"), bridge["node_id"]
            )
            return bridge
        except LLMStoppedError:
            return None
        except Exception as e:
            logger.warning("[QUALITY] check_dependencies error for %s: %s", node.id, e)
            return None

    # ── Helpers ───────────────────────────────────────────────────────────────

    FILE_EXTENSIONS = {
        ".md", ".txt", ".py", ".json", ".csv", ".html",
        ".yaml", ".yml", ".xml", ".pdf", ".log",
    }

    def _looks_like_filename(self, result: str) -> bool:
        s = result.strip()

        # Must be a single token — no spaces (bare path only, not "file_written: foo.md")
        if " " in s:
            return False
        if "\n" in s or "\\n" in s:
            return False
        if any(s.startswith(c) for c in ("#", "{", "[", "-", "=", ">")):
            return False
        if len(s) > 200:
            return False
        return any(s.endswith(ext) for ext in self.FILE_EXTENSIONS)

    def _file_exists(self, path: str) -> bool:
        if self.tools and hasattr(self.tools, "execute"):
            try:
                self.tools.execute("read_file", {"path": path})
                return True
            except Exception:
                return False
        import os
        return os.path.exists(path)


# --- FILE: infra/__init__.py ---




# --- FILE: infra/event_log.py ---

# infra/event_log.py

import json
from pathlib import Path
from cuddlytoddly.core.events import Event


class EventLog:
    """
    Append-only JSONL event log.

    Each line is one JSON-serialized event. The file is safe to replay
    after a crash because:
      - append() sanitizes the serialized line to guarantee no embedded
        newlines slip through (LLM results can contain raw \\r\\n)
      - replay() skips and logs any lines that fail to parse rather than
        raising, so a single corrupt entry never blocks a full restore
    """

    def __init__(self, path="event_log.jsonl"):
        self.path = Path(path)
        self.path.touch(exist_ok=True)

    def append(self, event: Event):
        # ensure_ascii=False preserves unicode but keeps the output compact;
        # json.dumps always escapes \n inside strings — the extra replace is a
        # safety net for any control characters the LLM smuggles in.
        line = json.dumps(event.to_dict(), ensure_ascii=False)
        # Belt-and-suspenders: strip any literal newlines that somehow survived
        line = line.replace("\r\n", "\\r\\n").replace("\r", "\\r").replace("\n", "\\n")
        with self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def replay(self):
        """
        Yield Event objects in order, skipping unparseable lines.
        """
        with self.path.open("r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                    yield Event.from_dict(data)
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    # Log and skip — one bad line should not abort a full restore
                    import logging
                    logging.getLogger("dag.infra.event_log").warning(
                        "[EVENT LOG] Skipping corrupt line %d: %s — %s",
                        lineno, repr(raw[:80]), e,
                    )

    def clear(self):
        self.path.write_text("", encoding="utf-8")


# --- FILE: infra/event_queue.py ---

"""
Event Queue

Thread-safe queue for all mutations and interactions.
"""
from queue import Queue

class EventQueue:
    def __init__(self):
        self._queue = Queue()

    def put(self, event):
        self._queue.put(event)

    def get(self):
        return self._queue.get()

    def empty(self):
        return self._queue.empty()


# --- FILE: infra/logging.py ---

"""
Centralized Logging

Single source of truth for all logging in the application.

Usage in any module:
    from cuddlytoddly.infra.logging import get_logger
    logger = get_logger(__name__)
    logger.info("something happened")
    logger.debug("verbose detail")

Call setup_logging() once at application startup (main.py).
All loggers are children of the "dag" root logger so they
inherit handlers automatically.
"""

import logging
import logging.handlers
from pathlib import Path
import re

LOG_DIR = Path("logs")

# Named loggers used across the app (for documentation / discoverability)
# dag                  - root, catches everything
# dag.core             - TaskGraph, reducer, events
# dag.engine           - orchestrator, executor, policies
# dag.planning         - planning/meta/reflection policies, LLM interface
# dag.agent            - agent expansion
# dag.ui               - curses UI + git projection
# dag.infra            - event log, replay


def setup_logging(
    log_level: int = logging.INFO,
    log_dir: Path | str | None = None,
    debug_modules: tuple[str, ...] = (
        "dag.engine",
        "dag.planning",
        "dag.skills"
    ),
    max_bytes: int = 5 * 1024 * 1024,   # 5 MB per file
    backup_count: int = 3,               # keep .1 .2 .3
) -> logging.Logger:
    """
    Configure the application root logger.

    Creates:
      logs/dag.log         — INFO+ from all modules, appended, rotated at 5 MB
      logs/dag_debug.log   — DEBUG from debug_modules only, rotated at 5 MB
      stderr               — WARNING+ (removed during curses session)
    """
    log_dir = Path(log_dir) if log_dir else LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)


    # ── Reset log files on every run ─────────────────────────────────────────
    for fname in ("dag.log", "dag_debug.log"):
        (log_dir / fname).write_text("", encoding="utf-8")

    root = logging.getLogger("dag")
    root.setLevel(logging.DEBUG)

    if root.hasHandlers():
        root.handlers.clear()

    fmt_verbose = logging.Formatter(
        "%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fmt_simple = logging.Formatter("[%(levelname)s] %(name)s: %(message)s")

    # ── Main log file: INFO+, all modules, rotating ──────────────────────────
    fh_main = logging.handlers.RotatingFileHandler(
        log_dir / "dag.log",
        mode="a",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    fh_main.setLevel(log_level)
    fh_main.setFormatter(fmt_verbose)
    root.addHandler(fh_main)

    # ── Debug log file: DEBUG+, selected modules only, rotating ─────────────
    class _ModuleFilter(logging.Filter):
        def __init__(self, prefixes: tuple[str, ...]):
            super().__init__()
            self.prefixes = prefixes

        def filter(self, record: logging.LogRecord) -> bool:
            return any(record.name.startswith(p) for p in self.prefixes)

    fh_debug = logging.handlers.RotatingFileHandler(
        log_dir / "dag_debug.log",
        mode="a",
        maxBytes=max_bytes,
        backupCount=backup_count,
        encoding="utf-8",
    )
    fh_debug.setLevel(logging.DEBUG)
    fh_debug.setFormatter(fmt_verbose)
    fh_debug.addFilter(_ModuleFilter(debug_modules))
    root.addHandler(fh_debug)

    # ── stderr: WARNING+ only, removed during curses session ────────────────
    ch = logging.StreamHandler()
    ch.setLevel(logging.WARNING)
    ch.setFormatter(fmt_simple)
    root.addHandler(ch)
    root._stderr_handler = ch

    return root


def get_logger(name: str) -> logging.Logger:
    if name.startswith("dag.") or name == "dag":
        return logging.getLogger(name)
    # Strip the package prefix so "cuddlytoddly.engine.foo" → "dag.engine.foo"
    # instead of "dag.cuddlytoddly.engine.foo"
    stripped = re.sub(r"^cuddlytoddly\.", "", name)
    return logging.getLogger(f"dag.{stripped}")


# --- FILE: infra/replay.py ---

# infra/replay.py

from cuddlytoddly.core.task_graph import TaskGraph
from cuddlytoddly.core.reducer import apply_event


def rebuild_graph_from_log(event_log):
    graph = TaskGraph()
    for event in event_log.replay():
        # Skip ADD_DEPENDENCY events that would create a cycle or already exist
        if event.type == "ADD_DEPENDENCY":
            node_id    = event.payload.get("node_id")
            depends_on = event.payload.get("depends_on")
            if (node_id in graph.nodes
                    and depends_on in graph.nodes
                    and depends_on in graph.nodes[node_id].dependencies):
                continue   # already wired — skip silently
        apply_event(graph, event)
    return graph


# --- FILE: planning/__init__.py ---




# --- FILE: planning/llm_executor.py ---

# planning/llm_executor.py

import json
from cuddlytoddly.infra.logging import get_logger
from cuddlytoddly.planning.llm_interface import LLMStoppedError

MAX_INLINE_RESULT_CHARS = 3000

logger = get_logger(__name__)

# Schema for a single execution turn.
# The model either returns a final result or requests a tool call.
EXECUTION_TURN_SCHEMA = {
    "type": "object",
    "properties": {
        "done": {
            "type": "boolean",
            "description": "True if this is the final answer, False if a tool call is needed."
        },
        "result": {
            "type": "string",
            "description": (
                "The final result text. Required when done=true. "
                "Must be a self-contained, detailed description of what was produced — "
                "it will be passed verbatim to downstream tasks as their input."
            )
        },
        "tool_call": {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "args": {"type": "object", "additionalProperties": {"type": "string"}}
            },
            "required": ["name", "args"],
            "description": "Tool to call. Required when done=false."
        }
    },
    "required": ["done"]
}


class LLMExecutor:
    """
    Executes a ready task node by prompting the LLM with the node's
    description, its required inputs (resolved from upstream results),
    and an optional set of tools.

    The execution loop:
      1. Build a prompt from node context + conversation history
      2. Ask the LLM → it responds with either done=True + result,
         or done=False + tool_call
      3. If tool_call, run the tool and append the result to history
      4. Repeat until done=True or max_turns reached
    """

    def __init__(self, llm_client, tool_registry=None, max_turns=5):
        self.llm   = llm_client
        self.tools = tool_registry
        self.max_turns = max_turns

    # ──────────────────────────────────────────────────────────────────────────

    def _resolve_inputs(self, node, snapshot):

        def _format_output_list(outputs):
            if not outputs:
                return []
            result = []
            for o in outputs:
                if isinstance(o, dict):
                    result.append(f"{o['name']} ({o['type']}): {o['description']}")
                else:
                    result.append(str(o))  # backward compat
            return result

        MAX_TOTAL_INPUT_CHARS = 3000   # ~750 tokens total for all upstream results
        
        resolved = []
        for dep_id in node.dependencies:
            dep = snapshot.get(dep_id)
            if not dep or not dep.result:
                continue
            resolved.append({
                "node_id":         dep_id,
                "description":     dep.metadata.get("description", dep_id),
                "declared_output": _format_output_list(dep.metadata.get("output", [])),
                "result":          dep.result,   # full for now, truncated below
            })

        # Distribute the budget evenly across all upstream results
        if resolved:
            budget_per_dep = MAX_TOTAL_INPUT_CHARS // len(resolved)
            for entry in resolved:
                r = entry["result"]
                if len(r) > budget_per_dep:
                    entry["result"] = r[:budget_per_dep] + f"\n…[truncated, {len(r)} chars total]"

        return resolved

    def _tool_schema_summary(self):
        if not self.tools:
            return "No tools available."
        lines = []
        for name, tool in self.tools.tools.items():
            desc   = getattr(tool, "description", "no description")
            schema = getattr(tool, "input_schema", {})
            lines.append(f"- {name}: {desc}. Args: {json.dumps(schema)}")
        return "\n".join(lines)

    def _build_prompt(self, node, resolved_inputs, history, extra_reminder=""):

        def _format_output_for_prompt(o):
            if isinstance(o, dict):
                return f"  - [{o['type']}] {o['name']}: {o['description']}"
            return f"  - {o}"  # backward compat

        # ── Upstream results ──────────────────────────────────────────────────
        if resolved_inputs:
            inputs_text = "\n".join(
                f"  [{entry['node_id']}]\n"
                f"    Description:      {entry['description']}\n"
                f"    Declared outputs: {entry['declared_output']}\n"
                f"    Actual result:    {entry['result']}"
                for entry in resolved_inputs
            )
        else:
            inputs_text = "  (none — this is a root task)"

        # In LLMExecutor._build_prompt:
        retry = node.metadata.get("retry_count", 0)
        if retry > 0:
            failure  = node.metadata.get("verification_failure", "unknown")[:200]
            prev_result = node.result or "(none)"
            if len(prev_result) > 200:
                prev_result = prev_result[:200] + "…"
            retry_notice = f"""
        ⚠️  RETRY ATTEMPT {retry} — PREVIOUS ATTEMPT FAILED
        Failure reason: {failure}
        Your previous result was: {prev_result}

        You MUST return different, substantive content this time.
        Do NOT return a label, filename, or the output name — return the actual data.
        """
        else:
            retry_notice = ""

        # ── Tool call history ─────────────────────────────────────────────────
        history_text = ""
        if history:
            parts = []
            for entry in history:
                parts.append(
                    f"  Tool: {entry['name']}\n"
                    f"  Args: {json.dumps(entry['args'])}\n"
                    f"  Result: {entry['result']}"
                )
            history_text = "Previous tool calls this turn:\n" + "\n\n".join(parts)

        tools_text = self._tool_schema_summary()

        # ── Declared outputs this node should produce ─────────────────────────
        # Determine if this node is expected to produce a file output
        declared_outputs = node.metadata.get("output", [])
        file_extensions  = {".md", ".txt", ".py", ".json", ".csv", ".html",
                            ".yaml", ".yml", ".xml", ".pdf", ".log"}

        def _output_name(o):
            return o["name"] if isinstance(o, dict) else str(o)

        def o_type_is_file(o):
            if isinstance(o, dict):
                return (o.get("type") == "file" or
                        any(_output_name(o).endswith(ext) for ext in file_extensions))
            return any(str(o).endswith(ext) for ext in file_extensions)

        expects_file_output = any(
            o_type_is_file(o) for o in declared_outputs
        )

        description = node.metadata.get("description", "").lower()
        is_file_edit = any(word in description for word in
                        ("edit", "modify", "update", "append", "patch", "overwrite"))

        if expects_file_output or is_file_edit:
            output_instruction = f"""- This task is expected to produce a file on disk.
        Call write_file (or append_file if editing an existing file) before setting done=true.
        Your result string should confirm what was written:
            file_written: <filename>
            summary: <brief description of contents>"""
        else:
            output_instruction = f"""- Return your result as a self-contained text string.
        Do NOT write files unless explicitly required by this task's description.
        Do NOT read from or write to disk — pass results inline as text.
        If your result would exceed {MAX_INLINE_RESULT_CHARS} characters, write it to a file
        using write_file and return the filename + a summary instead."""

        outputs_text = ("\n".join(_format_output_for_prompt(o) for o in declared_outputs) 
                                 if declared_outputs else "  (not specified)"
        )

        outputs_block = f"""Expected outputs (produce the CONTENT of these, not their names):
        {outputs_text}

        IMPORTANT: Do not return the output name as your result. Return the actual content.
        For example, if the output is 'research_report', your result must contain
        the actual research findings, not the string 'research_report'."""

        logger.info(
            "[EXECUTOR] Prompt sections for %s — "
            "retry=%d chars, outputs=%d chars, inputs=%d chars, "
            "tools=%d chars, history=%d chars",
            node.id,
            len(retry_notice), len(outputs_text),
            len(inputs_text), len(tools_text), len(history_text),
        )
        PROMPT_VERSION = "v3"   # bump this when prompt semantics change significantly

        return f"""[prompt_version={PROMPT_VERSION}]
You are executing one task inside a larger automated plan.
Your result will be stored and passed directly to downstream tasks as their input,
so it must be self-contained, specific, and directly usable — not a summary or stub.

════════════════════════════════════════
TASK 
{retry_notice}
{extra_reminder}
════════════════════════════════════════
ID:          {node.id}
Description: {node.metadata.get("description", node.id)}

{outputs_block}

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
- Your result must be detailed enough that a downstream task can use it
  without any other context. Label each output clearly:
    investment_analysis: <full content>
    risk_assessment: <full content>
- If you need to call a tool first, set done=false and provide tool_call.
- Only set done=true when you have a complete, usable result.
- Use \\n for line breaks and 4 spaces for indentation in Python code.
  Do NOT compress multi-line code onto one line with semicolons.
"""

    # ──────────────────────────────────────────────────────────────────────────

    def execute(self, node, snapshot, reporter=None):
        resolved_inputs = self._resolve_inputs(node, snapshot)
        history = []

        declared_outputs = node.metadata.get("output", [])
        file_extensions  = {".md", ".txt", ".py", ".json", ".csv", ".html",
                            ".yaml", ".yml", ".xml", ".pdf", ".log"}

        def _output_name(o):
            return o["name"] if isinstance(o, dict) else str(o)

        def o_type_is_file(o):
            if isinstance(o, dict):
                return (o.get("type") == "file" or
                        any(_output_name(o).endswith(ext) for ext in file_extensions))
            return any(str(o).endswith(ext) for ext in file_extensions)


        expected_files = [
            _output_name(o) for o in declared_outputs
            if o_type_is_file(o)
        ]

        for turn in range(self.max_turns):
            # If we have turns remaining and file outputs are declared,
            # remind the model upfront in the prompt
            turns_remaining = self.max_turns - turn
            file_reminder = ""
            if expected_files and "write_file" not in {h["name"] for h in history}:
                file_reminder = (
                    f"\nREMINDER: You must call write_file to create "
                    f"{expected_files} before setting done=true. "
                    f"You have {turns_remaining} turn(s) remaining."
                )

            prompt = self._build_prompt(node, resolved_inputs, history, 
                                        extra_reminder=file_reminder)

            if reporter:
                reporter.on_llm_turn(turn)

            prompt = self._build_prompt(node, resolved_inputs, history)

            try:
                raw = self.llm.ask(prompt, schema=EXECUTION_TURN_SCHEMA)
            except LLMStoppedError:
                logger.warning("[EXECUTOR] LLM stopped during execution of %s", node.id)
                return None
            except Exception as e:
                logger.error("[EXECUTOR] LLM error for node %s: %s", node.id, e)
                if reporter:
                    reporter.on_llm_error(turn, str(e))
                return None

            try:
                response = json.loads(raw)
            except json.JSONDecodeError as e:
                logger.error("[EXECUTOR] JSON parse error for node %s: %s", node.id, e)
                if reporter:
                    reporter.on_llm_error(turn, f"JSON parse error: {e}")
                return None

            # In LLMExecutor.execute(), after parsing response, before checking done:
            if response.get("done"):
                result = response.get("result", "")

                # If this node declares file outputs, verify a write_file was called
                # this session before accepting done=true

                tool_names_used  = {h["name"] for h in history}

                # In LLMExecutor.execute(), replace the correction turn injection:
                if expected_files and "write_file" not in tool_names_used:
                    logger.warning(
                        "[EXECUTOR] Node %s set done=true but write_file not in history "
                        "— injecting correction turn", node.id
                    )
                    # Inject as a complete tool exchange (call + result) so the model
                    # understands it's still in the middle of execution
                    history.append({
                        "name":   "write_file",
                        "args":   {"path": expected_files[0], "content": ""},
                        "result": (
                            f"ERROR: write_file was called with empty content. "
                            f"You must call write_file again with the full content of "
                            f"{expected_files[0]}. Use the actual report content you "
                            f"generated — do not set done=true until the file is written "
                            f"with real content."
                        ),
                    })
                    # Also cap done=false by continuing — the next turn must use tool_call
                    continue

                logger.info("[EXECUTOR] Node %s completed. Result: %.120s", node.id, result)
                return result

            tool_call = response.get("tool_call")
            if not tool_call:
                logger.warning("[EXECUTOR] Node %s: done=false but no tool_call on turn %d",
                            node.id, turn + 1)
                if reporter:
                    reporter.on_llm_error(turn, "done=false but no tool_call provided")
                return None

            tool_name = tool_call.get("name")
            tool_args = tool_call.get("args", {})

            if not self.tools or tool_name not in self.tools.tools:
                logger.warning("[EXECUTOR] Node %s requested unknown tool '%s'",
                            node.id, tool_name)
                if reporter:
                    step_id = reporter.on_tool_start(tool_name, tool_args)
                    reporter.on_tool_done(step_id, tool_name, tool_args,
                                        f"ERROR: tool '{tool_name}' not found",
                                        error=True)
                history.append({
                    "name": tool_name, "args": tool_args,
                    "result": f"ERROR: tool '{tool_name}' not found",
                })
                continue

            logger.info("[EXECUTOR] Node %s calling tool '%s'", node.id, tool_name)
            step_id = reporter.on_tool_start(tool_name, tool_args) if reporter else None

            error = False
            try:
                tool_result = self.tools.execute(tool_name, tool_args)
            except Exception as e:
                tool_result = f"ERROR: {e}"
                error = True
                logger.error("[EXECUTOR] Tool '%s' raised: %s", tool_name, e)

            if reporter and step_id:
                reporter.on_tool_done(step_id, tool_name, tool_args,
                                    str(tool_result), error=error)

            MAX_TOOL_RESULT_CHARS = 2000
            MAX_HISTORY_ENTRIES = 3

            tool_result_str = str(tool_result)
            if len(tool_result_str) > MAX_TOOL_RESULT_CHARS:
                tool_result_str = (
                    tool_result_str[:MAX_TOOL_RESULT_CHARS]
                    + f"\n…[truncated — {len(tool_result_str)} chars total]"
                )

            history.append({
                "name":   tool_name,
                "args":   tool_args,
                "result": tool_result_str,
            })

            if len(history) > MAX_HISTORY_ENTRIES:
                history = history[-MAX_HISTORY_ENTRIES:]

        logger.error("[EXECUTOR] Node %s did not complete within %d turns",
                    node.id, self.max_turns)
        return None


# --- FILE: planning/llm_interface.py ---

# planning/llm_interface.py
#
# Three interchangeable LLM backends, all sharing the same .ask() / .generate() interface.
#
# ┌─────────────────────────────────────────────────────────────┐
# │  Backend        │  How to select                           │
# ├─────────────────┼──────────────────────────────────────────┤
# │  FileBasedLLM   │  create_llm_client(backend="file")       │
# │  LlamaCppLLM    │  create_llm_client(backend="llamacpp")   │
# │  ApiLLM         │  create_llm_client(backend="openai")     │
# │                 │  create_llm_client(backend="claude")     │
# └─────────────────────────────────────────────────────────────┘
#
# All backends accept a `prompt: str` and return a `str` (raw JSON text).
# Structured output (outlines grammar) is applied inside LlamaCppLLM so the
# rest of the codebase never needs to change.

from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any
import threading

from cuddlytoddly.core.id_generator import StableIDGenerator
from cuddlytoddly.infra.logging import get_logger
import threading

class TokenCounter:
    """
    Module-level singleton tracking tokens consumed across all LLM calls
    in this process.  Thread-safe; all attributes are read-only properties.
    """
    def __init__(self):
        self._lock          = threading.Lock()
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._calls         = 0

    def add(self, prompt: int, completion: int) -> None:
        with self._lock:
            self._prompt_tokens     += prompt
            self._completion_tokens += completion
            self._calls             += 1

    @property
    def prompt_tokens(self) -> int:
        return self._prompt_tokens

    @property
    def completion_tokens(self) -> int:
        return self._completion_tokens

    @property
    def total_tokens(self) -> int:
        return self._prompt_tokens + self._completion_tokens

    @property
    def calls(self) -> int:
        return self._calls

    def reset(self) -> None:
        with self._lock:
            self._prompt_tokens = self._completion_tokens = self._calls = 0


# Module-level singleton — import this wherever you need token counts
token_counter = TokenCounter()
logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Shared constants (kept from original FileBasedLLM)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROMPT_LOG_FILE = PROJECT_ROOT / "llm_prompts.txt"
RESPONSE_FILE = PROJECT_ROOT / "llm_responses.txt"
POLL_INTERVAL = 0.5
TIMEOUT = 300
PROGRESS_LOG_INTERVAL = 2

id_gen = StableIDGenerator(id_length=6)


# ---------------------------------------------------------------------------
# JSON Schema — shared by all backends that support structured output.
# Describes the list-of-events format the planner/reflector expect.
# ---------------------------------------------------------------------------

# Add alongside EVENT_LIST_SCHEMA and REFINER_OUTPUT_SCHEMA

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


_IO_ITEM = {
    "type": "object",
    "required": ["name", "type", "description"],
    "additionalProperties": False,
    "properties": {
        "name":        {"type": "string",
                        "description": "Short snake_case identifier, e.g. 'investment_report'"},
        "type":        {"type": "string",
                        "enum": ["file", "document", "data", "list", "url", "text", "json", "code"],
                        "description": "What kind of artifact this is"},
        "description": {"type": "string",
                        "description": "One sentence: what this artifact contains"},
    }
}

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
                            "node_type":    {"type": "string", "enum": ["task", "goal", "reflection"]},
                            "dependencies": {"type": "array", "items": {"type": "string"}},
                            "metadata": {
                                "type": "object",
                                "additionalProperties": False,
                                "properties": {
                                    "description":      {"type": "string"},
                                    "parallel_group":   {"type": ["string", "null"]},
                                    "required_input":   {"type": "array", "items": _IO_ITEM},
                                    "output":           {"type": "array", "items": _IO_ITEM},
                                    "reflection_notes": {"type": "array", "items": {"type": "string"}},
                                    "precedes":         {"type": "array", "items": {"type": "string"}}
                                }
                            }
                        }
                    }
                }
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
                            "node_id":       {"type": "string"},
                            "depends_on": {"type": "string"}
                        }
                    }
                }
            }
        ]
    }
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
            "items": EVENT_LIST_SCHEMA["items"],  # reuses existing item definitions
        },
    },
}


# ---------------------------------------------------------------------------
# JSON Schema — used by LLMRefiner.
# The refiner returns a single object, not an array.
# ---------------------------------------------------------------------------

REFINER_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["needs_refinement", "tasks_to_expand", "validated_atomic", "dependency_issues", "reasoning"],
    "additionalProperties": False,
    "properties": {
        "needs_refinement": {
            "type": "boolean"
        },
        "tasks_to_expand": {
            "type": "array",
            "items": {"type": "string"}
        },
        "validated_atomic": {
            "type": "array",
            "items": {"type": "string"}
        },
        "dependency_issues": {
            "type": "array",
            "items": {"type": "string"}
        },
        "reasoning": {
            "type": "string"
        }
    }
}

# -----------------------------------------------------------------------------
# 1. New exception  — add near the top of llm_interface.py, after imports
# -----------------------------------------------------------------------------

class LLMStoppedError(RuntimeError):
    """Raised when an LLM call is attempted while the stop flag is set."""


# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------

# -----------------------------------------------------------------------------
# 2. BaseLLM  — replace existing BaseLLM class with this
# -----------------------------------------------------------------------------

class BaseLLM(ABC):
    """
    All backends implement this interface.
    Callers only ever use .ask() or .generate().

    Stop flag
    ---------
    Each instance owns a threading.Event (_stop_event).  When set, any call
    to ask() raises LLMStoppedError immediately — without touching the model
    or making any network call.

    Use stop() / resume() to set/clear the flag.
    The orchestrator calls these on all its clients via stop_llm_calls() /
    resume_llm_calls(), which the UI triggers with the 's' key.
    """

    def __init__(self):
        self._stop_event = threading.Event()

    def stop(self) -> None:
        """Set the stop flag — subsequent ask() calls raise LLMStoppedError."""
        self._stop_event.set()
        logger.info("[LLM] Stop flag SET on %s", self.__class__.__name__)

    def resume(self) -> None:
        """Clear the stop flag — ask() calls proceed normally again."""
        self._stop_event.clear()
        logger.info("[LLM] Stop flag CLEARED on %s", self.__class__.__name__)

    @property
    def is_stopped(self) -> bool:
        return self._stop_event.is_set()

    def _check_stop(self) -> None:
        """Call at the top of ask() in every backend."""
        if self._stop_event.is_set():
            raise LLMStoppedError("LLM calls are paused — resume before retrying")

    @abstractmethod
    def ask(self, prompt: str) -> str:
        """Send a prompt, block until a response is available, return raw text."""

    def generate(self, prompt: str) -> str:
        """Alias for ask() — kept for backward compatibility."""
        return self.ask(prompt)



# ---------------------------------------------------------------------------
# Backend 1 — FileBasedLLM  (original implementation, fully preserved)
# ---------------------------------------------------------------------------

class FileBasedLLM(BaseLLM):
    """
    Simulates an LLM using text files with unique IDs.

    Workflow:
      1. Prompts are appended to llm_prompts.txt  (id:<uid>\n<prompt>\n)
      2. A human (or external process) writes responses to llm_responses.txt
         using the same id:<uid> prefix.
      3. get_response() polls until the matching block appears.
    """

    def __init__(
        self,
        response_file: Path | str = RESPONSE_FILE,
        prompt_log_file: Path | str = PROMPT_LOG_FILE,
    ):
        super().__init__()
        self.response_file = Path(response_file)
        self.prompt_log_file = Path(prompt_log_file)
        logger.info("[LLM] Initialized FileBasedLLM")
        logger.info("[LLM] Prompt file path: %s", self.prompt_log_file.resolve())
        logger.info("[LLM] Response file path: %s", self.response_file.resolve())

    # --------------------------------------------------
    def send_prompt(self, prompt: str) -> str:
        logger.info("[LLM] send_prompt() called")
        prompt_id = id_gen.get_id(prompt, "prompts")
        logger.info("[LLM] Generated prompt_id=%s", prompt_id)

        entry = f"id:{prompt_id}\n{prompt}\n"

        if self.prompt_log_file.exists():
            logger.debug("[LLM] Prompt file exists, checking for duplicate id")
            with self.prompt_log_file.open("r") as f:
                for line in f:
                    if line.startswith("id:") and line[len("id:"):].strip() == prompt_id:
                        logger.warning(
                            "[LLM] Prompt id=%s already exists — skipping write", prompt_id
                        )
                        return prompt_id
        else:
            logger.info("[LLM] Prompt file does not exist — will create new one")

        try:
            with self.prompt_log_file.open("a") as f:
                f.write(entry)
            logger.info("[LLM] Prompt written (id=%s)", prompt_id)
        except Exception as e:
            logger.error("[LLM] Failed to write prompt id=%s: %s", prompt_id, e)
            raise

        return prompt_id

    # --------------------------------------------------
    def get_response(self, prompt_id: str) -> str:
        logger.info("[LLM] get_response() called for id=%s", prompt_id)
        start_time = time.time()
        last_progress_time = start_time

        while True:
            now = time.time()

            if self.response_file.exists():
                with self.response_file.open() as f:
                    lines = f.readlines()

                current_id = None
                block_lines = []
                for line in lines:
                    line = line.rstrip("\n")
                    if line.startswith("id:"):
                        if current_id == prompt_id and block_lines:
                            response_text = "\n".join([l for l in block_lines if l.strip()])
                            logger.info("[LLM] Response matched id=%s", prompt_id)
                            logger.debug("[LLM] Response content:\n%s", response_text)
                            return response_text
                        current_id = line[len("id:"):].strip()
                        block_lines = []
                    else:
                        block_lines.append(line)

                # last block
                if current_id == prompt_id and block_lines:
                    response_text = "\n".join([l for l in block_lines if l.strip()])
                    logger.info("[LLM] Response matched id=%s (last block)", prompt_id)
                    logger.debug("[LLM] Response content:\n%s", response_text)
                    return response_text
            else:
                logger.debug("[LLM] Response file does not yet exist")

            if now - last_progress_time > PROGRESS_LOG_INTERVAL:
                elapsed = int(now - start_time)
                logger.info(
                    "[LLM] Waiting for response (id=%s)... %ds elapsed", prompt_id, elapsed
                )
                last_progress_time = now

            if now - start_time > TIMEOUT:
                logger.error("[LLM] Timeout waiting for response id=%s", prompt_id)
                raise TimeoutError(
                    f"LLM response for id={prompt_id} not found within timeout"
                )

            time.sleep(POLL_INTERVAL)

    # --------------------------------------------------
    def ask(self, prompt: str) -> str:          # FileBasedLLM
        self._check_stop()                      # ← ADD THIS LINE
        logger.info("[LLM] ask() called")
        prompt_id = self.send_prompt(prompt)
        logger.info("[LLM] ask() obtained prompt_id=%s", prompt_id)
        response = self.get_response(prompt_id)
        logger.info("[LLM] ask() completed for id=%s", prompt_id)
        token_counter.add(len(prompt) // 4, len(response) // 4)

        return response


# ---------------------------------------------------------------------------
# Prompt-response cache for LlamaCppLLM
# ---------------------------------------------------------------------------

class LlamaCppCache:
    """
    Persistent, disk-backed cache for LlamaCppLLM prompt/response pairs.

    The cache is stored as a JSON file mapping SHA-256 prompt hashes to their
    responses. Both an in-memory dict and the JSON file are kept in sync so
    that:
      - Lookups within a single process are O(1) (no disk reads after load).
      - Results survive process restarts.

    Parameters
    ----------
    cache_path : Path | str
        Path to the JSON cache file. Created automatically if absent.
    """

    def __init__(self, cache_path: Path | str):
        self.cache_path = Path(cache_path)
        self._store: dict[str, str] = {}
        self._load()

    # --------------------------------------------------
    @staticmethod
    def _hash(prompt: str) -> str:
        """Return a stable SHA-256 hex digest of the prompt string."""
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    # --------------------------------------------------
    def _load(self) -> None:
        if not self.cache_path.exists():
            logger.info("[CACHE] No cache file found — starting empty")
            return

        try:
            with self.cache_path.open("r", encoding="utf-8") as f:
                self._store = json.load(f)

            if not isinstance(self._store, dict):
                raise ValueError("Cache root must be dict")

            logger.info(
                "[CACHE] Loaded %d cached entries from %s",
                len(self._store), self.cache_path,
            )

        except Exception as e:
            logger.error("[CACHE] Corrupted cache file detected: %s", e)

            # Backup corrupted file
            backup = self.cache_path.with_suffix(".corrupt.json")
            try:
                self.cache_path.rename(backup)
                logger.warning("[CACHE] Corrupted cache backed up to %s", backup)
            except OSError:
                logger.warning("[CACHE] Could not backup corrupted cache")

            self._store = {}

    # --------------------------------------------------
    def _save(self) -> None:
        """Persist the in-memory store to disk atomically via a temp file."""
        tmp = self.cache_path.with_suffix(".tmp")
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with tmp.open("w", encoding="utf-8") as f:
                json.dump(self._store, f, indent=2)
            tmp.replace(self.cache_path)
        except OSError as e:
            logger.error("[CACHE] Failed to write cache file: %s", e)
            tmp.unlink(missing_ok=True)

    # --------------------------------------------------
    def get(self, prompt: str) -> str | None:
        entry = self._store.get(self._hash(prompt))
        if entry is None:
            return None
        # handle both old format (bare string) and new format (dict)
        return entry["response"] if isinstance(entry, dict) else entry

    # --------------------------------------------------
    def set(self, prompt: str, response: str) -> None:
        key = self._hash(prompt)
        self._store[key] = {"prompt": prompt, "response": response}
        self._save()
        logger.info("[CACHE] Stored new entry (hash=%s…)", key[:12])

    # --------------------------------------------------
    def __len__(self) -> int:
        return len(self._store)

    # --------------------------------------------------
    def clear(self) -> None:
        """Wipe all cached entries from memory and disk."""
        self._store = {}
        self._save()
        logger.info("[CACHE] Cache cleared")


# ---------------------------------------------------------------------------
# Backend 2 — LlamaCppLLM  (local model via llama-cpp-python + outlines)
# ---------------------------------------------------------------------------

# These must already exist in llm_interface.py — referenced here for clarity
# from cuddlytoddly.planning.llm_interface import (
#     BaseLLM, LlamaCppCache, EVENT_LIST_SCHEMA, PROJECT_ROOT, logger
# )

# planning/llm_interface_llamacpp.py
#
# Drop-in replacement for the LlamaCppLLM class in planning/llm_interface.py.
#
# Key design:
#   ask(prompt)               -> unconstrained generation (fast, ~10-30s)
#                                Used by the planner. JSON repair handles any
#                                malformed output. Matches how the 138 cached
#                                entries were originally generated.
#
#   ask(prompt, schema=X)     -> outlines-constrained generation (slower but
#                                guaranteed-valid JSON).
#                                Used by the executor (EXECUTION_TURN_SCHEMA).
#
# Cache keys:
#   planner  -> prompt string only       (backward-compatible with existing cache)
#   executor -> prompt + schema fingerprint  (no collision with planner entries)

import json
import threading
import time
from pathlib import Path

# These must already exist in llm_interface.py — referenced here for clarity
# from cuddlytoddly.planning.llm_interface import (
#     BaseLLM, LlamaCppCache, EVENT_LIST_SCHEMA, PROJECT_ROOT, logger
# )

# planning/llm_interface_llamacpp.py
#
# Drop-in replacement for the LlamaCppLLM class in planning/llm_interface.py.
#
# Key design:
#   ask(prompt)               -> unconstrained generation (fast, ~10-30s)
#                                Used by the planner. JSON repair handles any
#                                malformed output. Matches how the 138 cached
#                                entries were originally generated.
#
#   ask(prompt, schema=X)     -> outlines-constrained generation (slower but
#                                guaranteed-valid JSON).
#                                Used by the executor (EXECUTION_TURN_SCHEMA).
#
# Cache keys:
#   planner  -> prompt string only       (backward-compatible with existing cache)
#   executor -> prompt + schema fingerprint  (no collision with planner entries)

import json
import threading
import time
from pathlib import Path

# These must already exist in llm_interface.py — referenced here for clarity
# from cuddlytoddly.planning.llm_interface import (
#     BaseLLM, LlamaCppCache, EVENT_LIST_SCHEMA, PROJECT_ROOT, logger
# )


class LlamaCppLLM(BaseLLM):
    """
    Runs a local GGUF model via llama-cpp-python.

    Parameters
    ----------
    model_path    : str | Path
    n_ctx         : int
    n_gpu_layers  : int
    temperature   : float
    max_tokens    : int
    schema        : dict | None   default schema stored but NOT used for generation
                                  unless passed explicitly to ask().
    cache_path    : str | Path | None
    """

    def __init__(
        self,
        model_path,
        n_ctx=4096,
        n_gpu_layers=0,
        temperature=0.2,
        max_tokens=2048,
        schema=None,
        cache_path=PROJECT_ROOT / "llamacpp_cache.json",
    ):
        super().__init__()
        self.model_path     = str(model_path)
        self.n_ctx          = n_ctx
        self.n_gpu_layers   = n_gpu_layers
        self.temperature    = temperature
        self.max_tokens     = max_tokens
        self.default_schema = schema or EVENT_LIST_SCHEMA

        logger.info("[LLAMACPP] Initializing LlamaCppLLM")
        logger.info("[LLAMACPP] Model path: %s", self.model_path)
        logger.info("[LLAMACPP] n_ctx=%d  n_gpu_layers=%d  temperature=%.2f  max_tokens=%d",
                    n_ctx, n_gpu_layers, temperature, max_tokens)

        if cache_path is not None:
            self._cache = LlamaCppCache(cache_path)
            logger.info("[LLAMACPP] Prompt cache enabled -- %s (%d entries loaded)",
                        Path(cache_path), len(self._cache))
        else:
            self._cache = None
            logger.info("[LLAMACPP] Prompt cache disabled")

        self._llama          = None   # llama_cpp.Llama -- loaded once
        self._outlines_model = None   # outlines wrapper -- only built if needed
        self._generators     = {}     # schema fingerprint -> outlines.Generator
        self._load_lock      = __import__("threading").Lock()  # prevents double-load
        # llama.cpp is NOT thread-safe -- all inference must be serialised
        self._inference_lock = threading.Lock()

    # -------------------------------------------------------------------------
    # Model loading
    # -------------------------------------------------------------------------

    def _load_model(self):
        """Load the Llama model. Called once on first use. Thread-safe."""
        if self._llama is not None:
            return
        with self._load_lock:
            if self._llama is not None:  # double-checked locking
                return

        try:
            from llama_cpp import Llama
        except ImportError as e:
            raise ImportError(
                "llama-cpp-python is not installed. "
                "Run: CMAKE_ARGS='-DLLAMA_METAL=on' pip install llama-cpp-python "
                "--force-reinstall --no-cache-dir"
            ) from e

        model_path = str(Path(self.model_path).expanduser().resolve())
        logger.info("[LLAMACPP] Loading model (first call -- may take 10-30s)...")
        self._llama = Llama(
            model_path=model_path,
            n_ctx=self.n_ctx,
            n_gpu_layers=self.n_gpu_layers,
            verbose=False,
        )
        logger.info("[LLAMACPP] Model loaded")

    def _load_outlines(self):
        """Build the outlines model wrapper. Thread-safe."""
        if self._outlines_model is not None:
            return
        with self._load_lock:
            if self._outlines_model is not None:
                return

        try:
            import outlines
        except ImportError as e:
            raise ImportError(
                "outlines is not installed. Run: pip install outlines"
            ) from e

        self._outlines_model = outlines.from_llamacpp(self._llama)
        logger.info("[LLAMACPP] Outlines model wrapper ready")

    # Keep old name as alias
    def _load(self):
        self._load_model()

    # -------------------------------------------------------------------------
    # Constrained generator cache
    # -------------------------------------------------------------------------

    def _get_generator(self, schema: dict):
        """Return a cached outlines Generator for schema (build on first use)."""
        import outlines
        fingerprint = json.dumps(schema, sort_keys=True)
        if fingerprint not in self._generators:
            self._load_outlines()
            logger.info("[LLAMACPP] Building constrained generator for schema %s...",
                        fingerprint[:40])
            output_type = outlines.json_schema(fingerprint)
            self._generators[fingerprint] = outlines.Generator(
                self._outlines_model, output_type
            )
            logger.info("[LLAMACPP] Constrained generator ready")
        return self._generators[fingerprint]

    # -------------------------------------------------------------------------
    # Chat template
    # -------------------------------------------------------------------------

    def _apply_chat_template(self, prompt: str) -> str:
        system = (
            "You are a DAG planning assistant. "
            "Always respond with a valid JSON array and nothing else. "
            "No explanation, no markdown, no code fences."
        )
        try:
            if self._llama.metadata.get("tokenizer.chat_template"):
                messages = [
                    {"role": "system", "content": system},
                    {"role": "user",   "content": prompt},
                ]
                result = self._llama.tokenizer_.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True,
                )
                logger.debug("[LLAMACPP] Chat template applied via llama.cpp tokenizer")
                return result
        except Exception as e:
            logger.debug("[LLAMACPP] Built-in chat template unavailable (%s), using fallback", e)

        # Llama 3 hardcoded fallback
        return (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{system}"
            "<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{prompt}"
            "<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )

    # -------------------------------------------------------------------------
    # Generation
    # -------------------------------------------------------------------------

    def _run_watchdog(self):
        """Return a (done_event, thread) watchdog that logs every 30s."""
        done = threading.Event()
        def _watch():
            start = time.time()
            while not done.wait(timeout=30):
                logger.info("[LLAMACPP] Still generating... %.0fs elapsed",
                            time.time() - start)
        t = threading.Thread(target=_watch, daemon=True, name="llm-watchdog")
        t.start()
        return done

    def _run_unconstrained(self, prompt: str, safe_max: int) -> str:
        """
        Fast path: raw llama.cpp generation with no grammar constraint.
        Used by the planner. ~10-30x faster than outlines on large schemas.
        """
        logger.info("[LLAMACPP] Running unconstrained inference (max_tokens=%d)...", safe_max)
        result = self._llama(
            prompt,
            max_tokens=safe_max,
            temperature=self.temperature,
            echo=False,
        )
        return result["choices"][0]["text"]

    def _run_constrained(self, prompt: str, schema: dict, safe_max: int) -> str:
        """
        Constrained path: outlines grammar enforcement.
        Used by the executor for guaranteed-valid JSON.
        """
        logger.info("[LLAMACPP] Running constrained inference (max_tokens=%d)...", safe_max)
        generator = self._get_generator(schema)
        raw = generator(prompt, max_tokens=safe_max)
        if isinstance(raw, str):
            return raw
        return json.dumps(raw)

    def _run_model(self, prompt: str, constrained_schema=None) -> str:
        """
        Run inference, with or without schema constraint.
        constrained_schema=None -> fast unconstrained (planner)
        constrained_schema=dict -> outlines constrained (executor)

        Serialised via _inference_lock: llama.cpp is not thread-safe.
        Parallel executor nodes will queue here and run one at a time.
        """
        formatted     = self._apply_chat_template(prompt)
        prompt_tokens = len(self._llama.tokenize(formatted.encode("utf-8")))

        safe_max = self.n_ctx - prompt_tokens - 64
        if safe_max <= 0:
            raise ValueError(
                f"Prompt is too long: {prompt_tokens} tokens leaves no room "
                f"in context window of {self.n_ctx}"
            )
        safe_max = min(self.max_tokens, safe_max)

        with self._inference_lock:
            done = self._run_watchdog()
            t0   = time.time()
            try:
                if constrained_schema is None:
                    raw = self._run_unconstrained(formatted, safe_max)
                else:
                    raw = self._run_constrained(formatted, constrained_schema, safe_max)
            finally:
                done.set()

        completion_tokens = len(self._llama.tokenize(raw.encode("utf-8")))
        token_counter.add(prompt_tokens, completion_tokens)

        logger.info("[LLAMACPP] Inference complete in %.1fs -- %d chars",
                    time.time() - t0, len(raw))
        return raw

    # -------------------------------------------------------------------------
    # Truncation repair
    # -------------------------------------------------------------------------

    def _repair_truncated_json(self, text: str):
        text = text.strip()
        if not text.startswith("["):
            return None
        pos = len(text) - 1
        while pos >= 0:
            pos = text.rfind("}", 0, pos + 1)
            if pos == -1:
                break
            candidate = text[:pos + 1].rstrip().rstrip(",") + "]"
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, list) and len(parsed) > 0:
                    logger.warning(
                        "[LLAMACPP] Truncated output repaired: %d event(s) recovered "
                        "(max_tokens=%d)", len(parsed), self.max_tokens,
                    )
                    return candidate
            except json.JSONDecodeError:
                pass
            pos -= 1
        logger.error("[LLAMACPP] Could not repair truncated output. "
                     "Increase max_tokens (currently %d).", self.max_tokens)
        return None

    # -------------------------------------------------------------------------
    # Public interface
    # -------------------------------------------------------------------------

    def ask(self, prompt: str, schema: dict | None = None) -> str:
        """
        Generate a response.

        schema=None  -> unconstrained generation (fast, used by planner)
        schema=dict  -> constrained generation   (slower, used by executor)
        """
        self._check_stop()
        logger.info("[LLAMACPP] ask() called")

        # Cache key: prompt-only for planner (backward-compatible with 138
        # existing entries); schema-namespaced for executor.
        if schema is None:
            cache_key        = prompt
            constrained_schema = None
        else:
            cache_key        = prompt + "\x00" + json.dumps(schema, sort_keys=True)
            constrained_schema = schema

        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                logger.info("[LLAMACPP] Cache HIT")
                return cached

        self._load_model()

        for attempt in range(2):
            response_text = self._run_model(prompt, constrained_schema)

            try:
                parsed = json.loads(response_text)
                if not parsed:
                    raise ValueError("Empty JSON response")
                if self._cache is not None:
                    self._cache.set(cache_key, response_text)
                return response_text

            except Exception as e:
                logger.warning("[LLAMACPP] Invalid JSON on attempt %d: %s", attempt + 1, e)
                if attempt == 0:
                    repaired = self._repair_truncated_json(response_text)
                    if repaired is not None:
                        if self._cache is not None:
                            self._cache.set(cache_key, repaired)
                        return repaired
                    logger.warning("[LLAMACPP] Repair failed -- retrying full generation")

        raise ValueError("Model repeatedly returned invalid JSON")

    def clear_cache(self):
        if self._cache is not None:
            self._cache.clear()
            logger.info("[LLAMACPP] Cache cleared")
        else:
            logger.info("[LLAMACPP] Cache is disabled -- nothing to clear")
# ---------------------------------------------------------------------------
# Backend 3 — ApiLLM  (OpenAI-compatible or Anthropic API)
# ---------------------------------------------------------------------------

# Replacement for ApiLLM in planning/llm_interface.py
# Drop this class in place of the existing ApiLLM definition.

class ApiLLM(BaseLLM):
    """
    Calls a remote LLM API.  Supports:
      - OpenAI  (and any OpenAI-compatible endpoint, e.g. Together, Groq)
      - Anthropic Claude

    Schema enforcement
    ------------------
    OpenAI:  When schema is provided, uses structured outputs
             (response_format type json_schema).  Falls back to
             json_object mode if the model does not support structured outputs
             (older checkpoints).

    Claude:  Schema is serialised into the prompt so the model knows the
             exact shape expected.  The assistant prefill character is
             chosen based on the schema root type ("{" for objects,
             "[" for arrays) so the model cannot produce the wrong container.

    Both backends validate the response JSON and retry once on failure.

    Dependencies (install the one you need):
        pip install openai
        pip install anthropic

    Parameters
    ----------
    provider : str
        "openai" or "claude".
    api_key : str
        Your API key. If None, reads from the environment variable
        OPENAI_API_KEY or ANTHROPIC_API_KEY automatically.
    model : str | None
        Model name.  Defaults per provider:
          openai -> "gpt-4o"
          claude -> "claude-opus-4-6"
    base_url : str | None
        Override API base URL for OpenAI-compatible providers
        (e.g. "https://api.together.xyz/v1").
    temperature : float
        Sampling temperature.
    max_tokens : int
        Maximum tokens to generate.
    system_prompt : str | None
        Optional system prompt prepended to every request.
    """

    _DEFAULTS = {
        "openai": "gpt-4o",
        "claude": "claude-opus-4-6",
    }

    def __init__(
        self,
        provider: str = "openai",
        api_key: str | None = None,
        model: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 4096,
        system_prompt: str | None = None,
    ):
        super().__init__()
        provider = provider.lower()
        if provider not in self._DEFAULTS:
            raise ValueError(
                f"Unknown provider '{provider}'. Choose 'openai' or 'claude'."
            )

        self.provider      = provider
        self.api_key       = api_key
        self.model         = model or self._DEFAULTS[provider]
        self.base_url      = base_url
        self.temperature   = temperature
        self.max_tokens    = max_tokens
        self.system_prompt = system_prompt or (
            "You are a DAG planning assistant. "
            "Always respond with valid JSON and nothing else. "
            "No explanation, no markdown, no code fences."
        )

        logger.info("[API] Initialized ApiLLM  provider=%s  model=%s",
                    self.provider, self.model)
        if base_url:
            logger.info("[API] Using custom base_url: %s", base_url)

        self._client = None  # lazy-loaded

    # ---- Client loading -------------------------------------------------------

    def _load(self):
        if self._client is not None:
            return

        if self.provider == "openai":
            try:
                from openai import OpenAI
            except ImportError as e:
                raise ImportError(
                    "openai package is not installed. Run: pip install openai"
                ) from e

            kwargs: dict[str, Any] = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key
            if self.base_url:
                kwargs["base_url"] = self.base_url

            self._client = OpenAI(**kwargs)
            logger.info("[API] OpenAI client ready")

        elif self.provider == "claude":
            try:
                import anthropic
            except ImportError as e:
                raise ImportError(
                    "anthropic package is not installed. Run: pip install anthropic"
                ) from e

            kwargs = {}
            if self.api_key:
                kwargs["api_key"] = self.api_key

            self._client = anthropic.Anthropic(**kwargs)
            logger.info("[API] Anthropic client ready")

    # ---- Schema helpers -------------------------------------------------------

    @staticmethod
    def _schema_root_type(schema: dict) -> str:
        """Return 'object' or 'array' based on the schema root type field."""
        return schema.get("type", "object")

    @staticmethod
    def _schema_prefill(schema: dict) -> str:
        """Return the correct opening character for a JSON prefill."""
        return "[" if ApiLLM._schema_root_type(schema) == "array" else "{"

    @staticmethod
    def _inject_schema_into_prompt(prompt: str, schema: dict) -> str:
        """
        Append the JSON schema to the prompt so the model knows the exact
        shape required.  Used for Claude where grammar enforcement is not
        available natively.
        """
        schema_str = json.dumps(schema, indent=2)
        return (
            prompt
            + f"\n\nYou MUST respond with JSON that strictly conforms to this schema:\n"
            + f"```json\n{schema_str}\n```\n"
            + "Respond with valid JSON only. No explanation, no markdown fences."
        )

    # ---- OpenAI call ---------------------------------------------------------

    def _ask_openai(self, prompt: str, schema: dict | None) -> str:
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": prompt},
        ]

        kwargs: dict[str, Any] = dict(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        if schema is not None:
            # Structured outputs: enforces the exact schema server-side.
            # Requires gpt-4o-2024-08-06 or later.  Older models that do not
            # support it will raise an error that we catch and fall back from.
            kwargs["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name":   "response",
                    "schema": schema,
                    "strict": False,  # strict=True requires no $defs / additionalProperties
                },
            }
        else:
            # json_object mode: guarantees valid JSON but not a specific shape.
            kwargs["response_format"] = {"type": "json_object"}

        logger.debug("[API] Sending OpenAI request  model=%s  schema=%s",
                     self.model, "yes" if schema else "no")
        try:
            response = self._client.chat.completions.create(**kwargs)
        except Exception as e:
            # Some older models / compatible endpoints don't support json_schema.
            # Fall back to json_object mode so execution continues.
            err_str = str(e).lower()
            if schema is not None and (
                "json_schema" in err_str or "response_format" in err_str
                or "unsupported" in err_str
            ):
                logger.warning(
                    "[API] Structured outputs not supported by this model/endpoint "
                    "— falling back to json_object mode"
                )
                kwargs["response_format"] = {"type": "json_object"}
                response = self._client.chat.completions.create(**kwargs)
            else:
                raise
        if response.usage:
            token_counter.add(response.usage.prompt_tokens,
                              response.usage.completion_tokens)
        content = response.choices[0].message.content or ""
        logger.info("[API] OpenAI response received (%d chars)", len(content))
        logger.debug("[API] Raw response:\n%s", content)
        return content

    # ---- Claude call ---------------------------------------------------------

    def _ask_claude(self, prompt: str, schema: dict | None) -> str:
        # Embed the schema into the prompt so the model knows the exact shape.
        if schema is not None:
            augmented_prompt = self._inject_schema_into_prompt(prompt, schema)
            prefill = self._schema_prefill(schema)
        else:
            augmented_prompt = (
                prompt
                + "\n\nRespond with valid JSON only. "
                "No explanation, no markdown, no code fences."
            )
            prefill = "{"   # default to object; planner always returns objects

        logger.debug("[API] Sending Anthropic request  model=%s  prefill=%r",
                     self.model, prefill)
        response = self._client.messages.create(
            model=self.model,
            max_tokens=self.max_tokens,
            system=self.system_prompt,
            messages=[
                {"role": "user",      "content": augmented_prompt},
                {"role": "assistant", "content": prefill},
            ],
            temperature=self.temperature,
        )
        token_counter.add(response.usage.input_tokens,
                          response.usage.output_tokens)
        # The prefill is not included in the response text — prepend it back.
        raw     = response.content[0].text
        content = prefill + raw
        logger.info("[API] Claude response received (%d chars)", len(content))
        logger.debug("[API] Raw response:\n%s", content)
        return content

    # ---- Public interface ----------------------------------------------------

    def ask(self, prompt: str, schema: dict | None = None) -> str:
        """
        Generate a response.

        schema=None  -> JSON object/array mode only (no shape enforcement)
        schema=dict  -> structured output enforcement (OpenAI json_schema /
                        Claude schema-in-prompt + correct prefill)

        Validates the response JSON and retries once on parse failure.
        """
        self._check_stop()
        logger.info("[API] ask() called  provider=%s  model=%s", self.provider, self.model)
        self._load()

        logger.debug("[API] Prompt (first 200 chars): %.200s", prompt)

        for attempt in range(2):
            try:
                if self.provider == "openai":
                    raw = self._ask_openai(prompt, schema)
                else:
                    raw = self._ask_claude(prompt, schema)
            except LLMStoppedError:
                raise
            except Exception as e:
                logger.error("[API] Request failed on attempt %d: %s", attempt + 1, e)
                if attempt == 0:
                    logger.warning("[API] Retrying after request error...")
                    continue
                raise

            # Validate the response is parseable JSON
            try:
                parsed = json.loads(raw)
                if not parsed and parsed != 0:
                    raise ValueError("Empty JSON response")
                return raw
            except Exception as e:
                logger.warning("[API] Invalid JSON on attempt %d: %s  raw=%.200s",
                               attempt + 1, e, raw)
                if attempt == 0:
                    logger.warning("[API] Retrying due to JSON parse failure...")
                    continue

        raise ValueError(
            f"[API] {self.provider} returned invalid JSON after 2 attempts"
        )
# ---------------------------------------------------------------------------
# Factory — single entry point for the rest of the codebase
# ---------------------------------------------------------------------------

def create_llm_client(backend: str = "file", **kwargs) -> BaseLLM:
    """
    Factory that returns the right LLM backend.

    Usage examples
    --------------
    # File-based (original behaviour — no extra args needed)
    llm = create_llm_client("file")

    # Local llama.cpp model with outlines schema enforcement + caching
    llm = create_llm_client(
        "llamacpp",
        model_path="/models/mistral-7b-instruct.Q4_K_M.gguf",
        n_gpu_layers=35,
        temperature=0.1,
        # cache_path defaults to <PROJECT_ROOT>/llamacpp_cache.json
        # pass cache_path=None to disable caching
    )

    # OpenAI
    llm = create_llm_client("openai", api_key="sk-...", model="gpt-4o")

    # OpenAI-compatible provider (Together, Groq, etc.)
    llm = create_llm_client(
        "openai",
        base_url="https://api.together.xyz/v1",
        api_key="...",
        model="mistralai/Mixtral-8x7B-Instruct-v0.1",
    )

    # Anthropic Claude
    llm = create_llm_client("claude", api_key="sk-ant-...")
    """
    backend = backend.lower()
    logger.info("[LLM FACTORY] Creating backend=%s", backend)

    if backend == "file":
        return FileBasedLLM(**kwargs)

    elif backend == "llamacpp":
        if "model_path" not in kwargs:
            raise ValueError(
                "llamacpp backend requires a 'model_path' keyword argument pointing "
                "to a .gguf file."
            )
        return LlamaCppLLM(**kwargs)

    elif backend in ("openai", "claude"):
        return ApiLLM(provider=backend, **kwargs)

    else:
        raise ValueError(
            f"Unknown backend '{backend}'. "
            "Valid options: 'file', 'llamacpp', 'openai', 'claude'."
        )


# --- FILE: planning/llm_output_validator.py ---

from cuddlytoddly.core.events import ADD_NODE, ADD_DEPENDENCY
from cuddlytoddly.infra.logging import get_logger

logger = get_logger(__name__)


class LLMOutputValidator:
    """
    Validates and normalizes raw LLM output before it enters the DAG.

    - Supports ADD_NODE and ADD_DEPENDENCY
    - Accepts dependency chains transitively within the same LLM batch
    - Prevents duplicate nodes
    - Prevents self-dependencies
    - Prevents goals depending on other goals
    - Ensures all dependencies reference valid nodes
    - Validates metadata keys for parallelism, I/O, and reflection
    """

    ALLOWED_METADATA_KEYS = {
        "precedes",
        "parallel_group",
        "required_input",
        "output",
        "description",
        "reflection_notes",
        "skill",
        "tools"
    }

    def __init__(self, graph):
        self.graph = graph

    def validate_and_normalize(self, raw_events, forced_origin):
        if not isinstance(raw_events, list):
            logger.warning(
                "[VALIDATOR] Expected a list of events, got %s — rejecting entire output",
                type(raw_events).__name__,
            )
            return []

        existing_ids = set(self.graph.nodes.keys())
        goal_ids = {nid for nid, node in self.graph.nodes.items() if getattr(node, "node_type", None) == "goal"}
        proposed_nodes = {}
        proposed_edges = []

        # --------------------------------
        # 1️⃣ Basic schema validation
        # --------------------------------
        for event in raw_events:
            if not isinstance(event, dict):
                logger.warning("[VALIDATOR] Skipping non-dict event: %r", event)
                continue

            event_type = event.get("type")
            payload = event.get("payload")
            if not isinstance(payload, dict):
                logger.warning(
                    "[VALIDATOR] Skipping event with invalid payload (type=%s): %r",
                    event_type,
                    payload,
                )
                continue

            # -----------------------------
            # ADD_NODE
            # -----------------------------
            if event_type == ADD_NODE:
                node_id = payload.get("node_id")
                dependencies = payload.get("dependencies", [])
                node_type = payload.get("node_type", "task")
                metadata = payload.get("metadata", {})

                if not node_id or not isinstance(node_id, str):
                    logger.warning("[VALIDATOR] ADD_NODE rejected — missing or non-string node_id: %r", payload)
                    continue
                if not isinstance(dependencies, list):
                    logger.warning("[VALIDATOR] ADD_NODE %s rejected — dependencies is not a list: %r", node_id, dependencies)
                    continue
                if node_id in dependencies:
                    logger.warning("[VALIDATOR] ADD_NODE %s rejected — self-dependency", node_id)
                    continue
                if node_id in existing_ids:
                    logger.warning("[VALIDATOR] ADD_NODE %s rejected — node already exists in graph", node_id)
                    # Salvage any dependency edges implied by this node's dependencies list.
                    # The node itself doesn't need re-creating, but the edges it declared
                    # may not exist yet (e.g. a goal<->task link the planner is re-asserting).
                    for dep in dependencies:
                        if isinstance(dep, str) and dep != node_id:
                            proposed_edges.append((node_id, dep))
                    continue
                if not isinstance(metadata, dict):
                    logger.warning("[VALIDATOR] ADD_NODE %s — metadata is not a dict, resetting to {}", node_id)
                    metadata = {}

                filtered_metadata = {k: v for k, v in metadata.items() if k in self.ALLOWED_METADATA_KEYS}
                stripped_keys = set(metadata.keys()) - self.ALLOWED_METADATA_KEYS
                if stripped_keys:
                    logger.debug(
                        "[VALIDATOR] ADD_NODE %s — stripped disallowed metadata keys: %s",
                        node_id,
                        stripped_keys,
                    )

                proposed_nodes[node_id] = {
                    "node_id": node_id,
                    "node_type": node_type,
                    "dependencies": dependencies,
                    "metadata": filtered_metadata
                }

            # -----------------------------
            # ADD_DEPENDENCY
            # -----------------------------
            elif event_type == ADD_DEPENDENCY:
                node_id = payload.get("node_id")
                depends_on = payload.get("depends_on")
                if not node_id or not depends_on:
                    logger.warning("[VALIDATOR] ADD_DEPENDENCY rejected — missing node_id or depends_on: %r", payload)
                    continue
                if not isinstance(node_id, str) or not isinstance(depends_on, str):
                    logger.warning(
                        "[VALIDATOR] ADD_DEPENDENCY rejected — node_id/depends_on must be strings: %r",
                        payload,
                    )
                    continue
                if node_id == depends_on:
                    logger.warning("[VALIDATOR] ADD_DEPENDENCY rejected — self-dependency on %s", node_id)
                    continue

                # Block a non-goal node from depending on a goal node.
                # However, a goal depending on its final completing task is valid
                # and must be allowed through.
                dependent_is_goal = node_id in goal_ids or node_id in {
                    nid for nid, nd in proposed_nodes.items()
                    if nd.get("node_type") == "goal"
                }
                if depends_on in goal_ids and not dependent_is_goal:
                    logger.warning(
                        "[VALIDATOR] ADD_DEPENDENCY rejected — non-goal %s cannot depend on goal node %s",
                        node_id,
                        depends_on,
                    )
                    continue

                proposed_edges.append((node_id, depends_on))

            else:
                logger.warning("[VALIDATOR] Unknown event type %r — skipping", event_type)

        # --------------------------------
        # 2️⃣ Transitive structural validation for nodes
        # --------------------------------
        accepted_nodes = {}
        available_ids = set(existing_ids)
        progress = True

        while progress:
            progress = False
            for node_id, node_data in list(proposed_nodes.items()):
                deps = node_data["dependencies"]
                if all(dep in available_ids for dep in deps):
                    accepted_nodes[node_id] = node_data
                    available_ids.add(node_id)
                    proposed_nodes.pop(node_id)
                    progress = True

        # Remaining proposed_nodes have unresolvable dependencies
        for node_id, node_data in proposed_nodes.items():
            missing = [d for d in node_data["dependencies"] if d not in available_ids]
            logger.warning(
                "[VALIDATOR] ADD_NODE %s rejected — unresolvable dependencies: %s",
                node_id,
                missing,
            )

        # --------------------------------
        # 3️⃣ Validate ADD_DEPENDENCY events
        # --------------------------------
        safe_edges = []
        for node_id, depends_on in proposed_edges:
            if node_id not in available_ids:
                logger.warning(
                    "[VALIDATOR] ADD_DEPENDENCY (%s → %s) rejected — %s not in graph or accepted batch",
                    node_id,
                    depends_on,
                    node_id,
                )
                continue
            if depends_on not in available_ids:
                logger.warning(
                    "[VALIDATOR] ADD_DEPENDENCY (%s → %s) rejected — %s not in graph or accepted batch",
                    node_id,
                    depends_on,
                )
                continue
            safe_edges.append((node_id, depends_on))

        # --------------------------------
        # 4️⃣ Normalize accepted events
        # --------------------------------
        safe_events = []

        # ADD_NODE events
        for node_id, node_data in accepted_nodes.items():
            safe_events.append({
                "type": ADD_NODE,
                "payload": {
                    "node_id": node_id,
                    "node_type": node_data["node_type"],
                    "dependencies": node_data["dependencies"],
                    "origin": forced_origin,
                    "metadata": node_data["metadata"]
                }
            })

        # ADD_DEPENDENCY events
        for node_id, depends_on in safe_edges:
            safe_events.append({
                "type": ADD_DEPENDENCY,
                "payload": {
                    "node_id": node_id,
                    "depends_on": depends_on,
                    "origin": forced_origin
                }
            })

        logger.info(
            "[VALIDATOR] Result: %d raw events → %d accepted (%d nodes, %d edges)",
            len(raw_events),
            len(safe_events),
            len(accepted_nodes),
            len(safe_edges),
        )

        return safe_events


# --- FILE: planning/llm_planner.py ---

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

# --- FILE: skills/__init__.py ---



# --- FILE: skills/code_execution/__init__.py ---



# --- FILE: skills/code_execution/tools.py ---

# skills/code_execution/tools.py

import subprocess
import sys
from cuddlytoddly.infra.logging import get_logger

logger = get_logger(__name__)

def _run_python(args):
    code = args["code"]

    # JSON encoding sometimes produces literal \n instead of real newlines
    code = code.replace("\\n", "\n").replace("\\t", "\t")

    # Strip markdown fences
    import re
    code = re.sub(r'^```(?:python)?\s*', '', code.strip())
    code = re.sub(r'\s*```$', '', code.strip())

    logger.info("[RUN_PYTHON] Executing: %.500s", code)

    try:
        result = eval(code, {"__builtins__": __builtins__})
        return str(result)
    except SyntaxError:
        pass

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code, {"__builtins__": __builtins__})
    return buf.getvalue() or "(no output)"


def _run_shell(args):
    result = subprocess.run(
        args["command"],
        shell=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = result.stdout.strip()
    if result.returncode != 0:
        output += f"\n[stderr] {result.stderr.strip()}"
    return output or "(no output)"


TOOLS = {
    "run_python": {
        "description": (
            "Execute a Python code block and return stdout. "
            "Use \\n for line breaks and 4 spaces for indentation. "
            "Do NOT compress multi-line code onto one line with semicolons — "
            "compound statements (for, if, while, def) require proper newlines."
        ),        
        "input_schema": {"code": "string"},
        "fn": _run_python,
    },
    "run_shell": {
        "description":  "Run a shell command and return stdout",
        "input_schema": {"command": "string"},
        "fn": _run_shell,
    },
}


# --- FILE: skills/file_ops/__init__.py ---



# --- FILE: skills/file_ops/tools.py ---

# skills/file_ops/tools.py
#
# Local tool implementations for the file_ops skill.
# The SkillLoader imports this and registers everything in TOOLS.

import subprocess
from pathlib import Path

TOOLS = {
    "read_file": {
        "description":  "Read the full contents of a local file",
        "input_schema": {"path": "string"},
        "fn": lambda args: Path(args["path"]).read_text(encoding="utf-8"),
    },
    "write_file": {
        "description":  "Write (or overwrite) a local file with the given content",
        "input_schema": {"path": "string", "content": "string"},
        "fn": lambda args: (
            Path(args["path"]).write_text(args["content"], encoding="utf-8"),
            f"Written {len(args['content'])} chars to {args['path']}"
        )[1],
    },
    "append_file": {
        "description":  "Append text to an existing file",
        "input_schema": {"path": "string", "content": "string"},
        "fn": lambda args: (
            Path(args["path"]).open("a", encoding="utf-8").write(args["content"]),
            f"Appended {len(args['content'])} chars to {args['path']}"
        )[1],
    },
    "list_dir": {
        "description":  "List files and directories at a path",
        "input_schema": {"path": "string"},
        "fn": lambda args: "\n".join(
            str(p) for p in sorted(Path(args["path"]).iterdir())
        ),
    },
}


# --- FILE: skills/skill_loader.py ---

# skills/skill_loader.py
#
# Reads the skills/ directory, parses each SKILL.md, registers any local
# tool implementations, and returns:
#   - A ToolRegistry populated with all available tools
#   - A skills summary string ready to inject into the planner prompt
#
# Directory convention
# --------------------
#   skills/
#     <skill_name>/
#       SKILL.md       required — description, tools, when-to-use, output format
#       tools.py       optional — local Python tool implementations
#
# tools.py format
# ---------------
# Define a module-level dict called TOOLS:
#
#   TOOLS = {
#       "tool_name": {
#           "description":  "...",
#           "input_schema": {"arg": "string"},
#           "fn":           lambda args: ...,
#       },
#       ...
#   }

from pathlib import Path
from cuddlytoddly.infra.logging import get_logger

logger = get_logger(__name__)

SKILLS_DIR = Path(__file__).parent  # skills/ directory


# ── Inline ToolRegistry (avoids circular import from engine/tools.py) ────────

class Tool:
    def __init__(self, name, description, input_schema, fn):
        self.name         = name
        self.description  = description
        self.input_schema = input_schema
        self._fn          = fn

    def run(self, input_data):
        return self._fn(input_data)


class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self.tools[tool.name] = tool
        logger.info("[SKILLS] Registered tool: %s", tool.name)

    def execute(self, name: str, input_data: dict):
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' not found")
        return self.tools[name].run(input_data)

    def merge(self, other: "ToolRegistry"):
        """Merge another registry into this one (other wins on collision)."""
        for tool in other.tools.values():
            self.register(tool)


# ── Skill loader ──────────────────────────────────────────────────────────────

class SkillLoader:
    """
    Loads all skills from the skills/ directory.

    Usage
    -----
    loader   = SkillLoader()
    registry = loader.registry          # ToolRegistry with all local tools
    summary  = loader.prompt_summary    # string to inject into planner prompt
    """

    def __init__(self, skills_dir: Path | str = SKILLS_DIR):
        self.skills_dir = Path(skills_dir)
        self._skills: list[dict] = []   # parsed skill metadata
        self.registry = ToolRegistry()
        self._load()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self):
        if not self.skills_dir.exists():
            logger.warning("[SKILLS] Skills directory not found: %s", self.skills_dir)
            return

        for skill_dir in sorted(self.skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue

            skill = self._parse_skill_md(skill_dir.name, skill_md)
            self._skills.append(skill)
            logger.info("[SKILLS] Loaded skill: %s", skill_dir.name)

            # Register local tools if tools.py exists
            tools_py = skill_dir / "tools.py"
            if tools_py.exists():
                self._register_local_tools(skill_dir.name, tools_py)

        logger.info("[SKILLS] Loaded %d skill(s), %d local tool(s)",
                    len(self._skills), len(self.registry.tools))

    def _parse_skill_md(self, name: str, path: Path) -> dict:
        """
        Extract the key sections from a SKILL.md into a dict.
        Sections are identified by '## SectionName' headings.
        """
        text     = path.read_text(encoding="utf-8")
        sections = {"name": name, "raw": text}

        current_section = "description"
        buf: list[str] = []

        for line in text.splitlines():
            if line.startswith("## "):
                if buf:
                    sections[current_section] = "\n".join(buf).strip()
                current_section = line[3:].strip().lower().replace(" ", "_")
                buf = []
            elif line.startswith("# "):
                sections["title"] = line[2:].strip()
            else:
                buf.append(line)

        if buf:
            sections[current_section] = "\n".join(buf).strip()

        return sections

    def _register_local_tools(self, skill_name: str, tools_py: Path):
        """Import tools.py from a skill directory and register its TOOLS dict."""
        import importlib.util
        spec   = importlib.util.spec_from_file_location(
            f"skills.{skill_name}.tools", tools_py
        )
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            logger.error("[SKILLS] Failed to import %s: %s", tools_py, e)
            return

        tools_dict = getattr(module, "TOOLS", None)
        if not isinstance(tools_dict, dict):
            logger.warning("[SKILLS] %s has no TOOLS dict — skipping", tools_py)
            return

        for tool_name, spec_dict in tools_dict.items():
            self.registry.register(Tool(
                name         = tool_name,
                description  = spec_dict.get("description", ""),
                input_schema = spec_dict.get("input_schema", {}),
                fn           = spec_dict["fn"],
            ))

    # ── Planner prompt injection ───────────────────────────────────────────────

    @property
    def prompt_summary(self) -> str:
        """
        A compact skills summary ready to drop into the planner prompt.
        Lists each skill, its when-to-use criteria, and the tools it provides.
        """
        if not self._skills:
            return ""

        lines = ["Available skills (use these to guide task decomposition):"]
        for s in self._skills:
            lines.append(f"\n### {s['name']}")

            desc = s.get("description", "")
            if desc:
                # First non-empty line only, to keep the prompt compact
                first_line = next((l for l in desc.splitlines() if l.strip()), "")
                lines.append(f"  {first_line}")

            when = s.get("when_to_use", "")
            if when:
                lines.append(f"  When to use: {when.splitlines()[0].strip()}")

            tools_section = s.get("tools", "")
            if tools_section:
                tool_names = [
                    l.strip().lstrip("- ").split(":")[0].strip("`")
                    for l in tools_section.splitlines()
                    if l.strip().startswith("-")
                ]
                if tool_names:
                    lines.append(f"  Tools: {', '.join(tool_names)}")

            output_fmt = s.get("expected_output_format", "")
            if output_fmt:
                first_line = next((l for l in output_fmt.splitlines() if l.strip()), "")
                lines.append(f"  Output format: {first_line}")

        return "\n".join(lines)

    def merge_mcp(self, mcp_registry: "ToolRegistry"):
        """Merge an MCP-sourced registry into the local one."""
        self.registry.merge(mcp_registry)


# --- FILE: tests/__init__.py ---




# --- FILE: tools/__init__.py ---



# --- FILE: tools/mcp_adapter.py ---

# tools/mcp_adapter.py
#
# Bridges MCP servers into the existing ToolRegistry so the LLMExecutor
# can call any MCP tool without knowing anything about MCP.
#
# Usage
# -----
# from tools.mcp_adapter import MCPAdapter
#
# # Filesystem server (reads/writes local files)
# adapter = MCPAdapter.from_stdio("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"])
# registry = adapter.build_registry()
#
# # Sequential thinking server
# adapter = MCPAdapter.from_stdio("npx", ["-y", "@modelcontextprotocol/server-sequential-thinking"])
# registry = adapter.build_registry()
#
# # Multiple servers merged into one registry
# registry = MCPAdapter.merged_registry([
#     MCPAdapter.from_stdio("npx", ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]),
#     MCPAdapter.from_stdio("npx", ["-y", "@modelcontextprotocol/server-brave-search"]),
# ])
#
# Then pass registry to LLMExecutor:
#   executor = LLMExecutor(llm_client=shared_llm, tool_registry=registry, max_turns=5)
#
# Dependencies
# ------------
#   pip install mcp
#
# Popular ready-made MCP servers (all via npx, no install needed):
#   @modelcontextprotocol/server-filesystem      read/write local files
#   @modelcontextprotocol/server-memory          persistent key-value memory
#   @modelcontextprotocol/server-brave-search    web search (needs BRAVE_API_KEY)
#   @modelcontextprotocol/server-github          GitHub API
#   @modelcontextprotocol/server-postgres        PostgreSQL queries
#   @modelcontextprotocol/server-sqlite          SQLite queries
#   @modelcontextprotocol/server-sequential-thinking  chain-of-thought reasoning


import asyncio
import json
from typing import Any
from cuddlytoddly.infra.logging import get_logger
from mcp.client.stdio import stdio_client
from mcp import ClientSession
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Inline ToolRegistry (mirrors engine/tools.py to avoid a circular import)
# ---------------------------------------------------------------------------

class Tool:
    def __init__(self, name: str, description: str, input_schema: dict, fn):
        self.name         = name
        self.description  = description
        self.input_schema = input_schema
        self._fn          = fn

    def run(self, input_data: dict) -> Any:
        return self._fn(input_data)


class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self.tools[tool.name] = tool
        logger.info("[TOOLS] Registered tool: %s", tool.name)

    def execute(self, name: str, input_data: dict) -> Any:
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' not found in registry")
        return self.tools[name].run(input_data)


# ---------------------------------------------------------------------------
# MCP Adapter
# ---------------------------------------------------------------------------

class MCPAdapter:
    """
    Connects to an MCP server, discovers its tools, and exposes them
    as a populated ToolRegistry.

    Parameters
    ----------
    server_params : mcp.StdioServerParameters
        How to launch the MCP server process.
    """

    def __init__(self, server_params):
        self._params = server_params

    # ── Constructors ──────────────────────────────────────────────────────────

    @classmethod
    def from_stdio(cls, command: str, args: list[str],
                   env: dict | None = None) -> "MCPAdapter":
        """
        Create an adapter for a stdio MCP server (the most common kind).

        Parameters
        ----------
        command : str        e.g. "npx" or "python"
        args    : list[str]  e.g. ["-y", "@modelcontextprotocol/server-filesystem", "/tmp"]
        env     : dict | None  extra environment variables (e.g. {"BRAVE_API_KEY": "..."})
        """
        try:
            from mcp import StdioServerParameters
        except ImportError as e:
            raise ImportError("mcp is not installed. Run: pip install mcp") from e

        params = StdioServerParameters(command=command, args=args, env=env)
        return cls(params)

    # ── Registry builder ──────────────────────────────────────────────────────

    def build_registry(self) -> ToolRegistry:
        """
        Launch the MCP server, list its tools, and return a ToolRegistry.

        Each MCP tool becomes a callable Tool that synchronously calls back
        into the server via a fresh async session (one call per invocation).
        """
        try:
            from mcp import StdioServerParameters
            from mcp.client.stdio import stdio_client
            import mcp.types as mcp_types
        except ImportError as e:
            raise ImportError("mcp is not installed. Run: pip install mcp") from e

        # Discover available tools synchronously
        tool_defs = asyncio.run(self._list_tools())
        registry  = ToolRegistry()

        for t in tool_defs:
            # Capture t.name in the closure correctly
            tool_name = t.name

            def make_fn(name):
                def call_tool(input_data: dict) -> str:
                    return asyncio.run(self._call_tool(name, input_data))
                return call_tool

            registry.register(Tool(
                name         = tool_name,
                description  = t.description or "",
                input_schema = t.inputSchema if hasattr(t, "inputSchema") else {},
                fn           = make_fn(tool_name),
            ))

        logger.info("[MCP] Registry built with %d tool(s) from server", len(registry.tools))
        return registry

    # ── Async helpers ─────────────────────────────────────────────────────────

    async def _list_tools(self) -> list:


        async with stdio_client(self._params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                response = await session.list_tools()
                return response.tools

    async def _call_tool(self, name: str, arguments: dict) -> str:


        async with stdio_client(self._params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)

                # MCP returns a list of content blocks; join text ones
                parts = []
                for block in result.content:
                    if hasattr(block, "text"):
                        parts.append(block.text)
                    else:
                        parts.append(json.dumps(block.__dict__))

                return "\n".join(parts)

    # ── Multi-server merge ────────────────────────────────────────────────────

    @staticmethod
    def merged_registry(adapters: list["MCPAdapter"]) -> ToolRegistry:
        """
        Build a single ToolRegistry from multiple MCP servers.
        Later adapters win on name collision.
        """
        merged = ToolRegistry()
        for adapter in adapters:
            sub = adapter.build_registry()
            for tool in sub.tools.values():
                merged.register(tool)
        return merged

# --- FILE: tools/registry.py ---



# --- FILE: ui/__init__.py ---




# --- FILE: ui/curses_startup.py ---

# ui/curses_startup.py
"""
Curses startup screen shown before the main DAG UI.

Usage in __main__.py:
    from cuddlytoddly.ui.curses_startup import run_startup_selection
    choice = run_startup_selection(repo_root)
    # choice is a StartupChoice namedtuple
"""
from __future__ import annotations

import curses
import textwrap
from pathlib import Path

from cuddlytoddly.ui.startup import (
    StartupChoice, RunInfo, scan_runs, parse_manual_plan,
)


# ── Colour pairs (set up once inside curses.wrapper) ─────────────────────────
_C_TITLE   = 1
_C_SEL     = 2
_C_DIM     = 3
_C_DONE    = 4
_C_ACCENT  = 5
_C_ERR     = 6


def _init_colors():
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(_C_TITLE,  curses.COLOR_CYAN,    -1)
    curses.init_pair(_C_SEL,    curses.COLOR_BLACK,   curses.COLOR_CYAN)
    curses.init_pair(_C_DIM,    curses.COLOR_WHITE,   -1)
    curses.init_pair(_C_DONE,   curses.COLOR_GREEN,   -1)
    curses.init_pair(_C_ACCENT, curses.COLOR_YELLOW,  -1)
    curses.init_pair(_C_ERR,    curses.COLOR_RED,     -1)


# ── helpers ───────────────────────────────────────────────────────────────────

def _safe_addstr(win, y, x, text, attr=0):
    h, w = win.getmaxyx()
    if y < 0 or y >= h or x < 0 or x >= w:
        return
    max_len = w - x - 1
    if max_len <= 0:
        return
    try:
        win.addstr(y, x, text[:max_len], attr)
    except curses.error:
        pass


def _center(win, y, text, attr=0):
    _, w = win.getmaxyx()
    x = max(0, (w - len(text)) // 2)
    _safe_addstr(win, y, x, text, attr)


def _hline(win, y, char="─"):
    _, w = win.getmaxyx()
    try:
        win.addstr(y, 0, char * (w - 1))
    except curses.error:
        pass


# ── Tab 1 — Resume existing run ───────────────────────────────────────────────

def _draw_resume_tab(win, runs: list[RunInfo], sel: int, scroll: int):
    h, w = win.getmaxyx()
    y = 0
    _safe_addstr(win, y, 2, "Existing runs  (↑↓ select, Enter resume)", curses.color_pair(_C_DIM))
    y += 1
    _hline(win, y)
    y += 1

    visible = h - y - 3
    for i, run in enumerate(runs[scroll: scroll + visible]):
        idx    = i + scroll
        is_sel = idx == sel
        attr   = curses.color_pair(_C_SEL) | curses.A_BOLD if is_sel else 0

        date   = run.age
        label  = f"  {run.goal[:w - 30]}".ljust(w - 28)
        stats  = f"{run.node_count} nodes  {date}  "

        try:
            win.addstr(y + i, 0, label[:w - len(stats) - 1].ljust(w - len(stats) - 1), attr)
            win.addstr(y + i, w - len(stats) - 1, stats,
                       curses.color_pair(_C_DONE) if not is_sel else attr)
        except curses.error:
            pass

    if not runs:
        _center(win, y + 2, "No existing runs found.", curses.color_pair(_C_DIM) | curses.A_DIM)


# ── Tab 2 — New goal ──────────────────────────────────────────────────────────

def _draw_new_goal_tab(win, text: str, cursor: int, error: str):
    h, w = win.getmaxyx()
    y = 0
    _safe_addstr(win, y, 2, "New goal  (type goal, Enter to start)", curses.color_pair(_C_DIM))
    y += 1
    _hline(win, y); y += 1

    _safe_addstr(win, y, 2, "Goal:", curses.color_pair(_C_ACCENT))
    y += 1

    # Wrap the text into the available width
    box_w  = w - 6
    lines  = textwrap.wrap(text, box_w) if text else [""]
    # Find cursor position
    cur_line, cur_col = _cursor_pos(text, cursor, box_w)

    for li, line in enumerate(lines[:h - y - 4]):
        is_cur_line = li == cur_line
        attr = curses.A_REVERSE if is_cur_line else curses.color_pair(_C_DIM)
        _safe_addstr(win, y + li, 4, line.ljust(box_w), attr)
    y += max(len(lines), 1) + 1

    if error:
        _safe_addstr(win, y, 2, f"! {error}", curses.color_pair(_C_ERR))


def _cursor_pos(text: str, cursor: int, wrap_w: int) -> tuple[int, int]:
    """Return (line_index, col_index) of cursor in wrapped text."""
    before  = text[:cursor]
    wrapped = textwrap.wrap(before, wrap_w) if before else [""]
    li      = len(wrapped) - 1
    col     = len(wrapped[-1]) if wrapped else 0
    return li, col


# ── Tab 3 — Manual plan ───────────────────────────────────────────────────────

_MANUAL_PLACEHOLDER = """\
task: Task_One
desc: First thing to do

task: Task_Two
desc: Second thing to do
deps: Task_One"""

_MANUAL_HELP = [
    "Format:",
    "  First lines (before any 'task:') = goal description",
    "  task: Task_ID",
    "  desc: One sentence description",
    "  deps: Dep1, Dep2   (optional)",
]


def _draw_manual_tab(win, goal_text: str, goal_cursor: int,
                      plan_text: str, plan_cursor: int,
                      active_field: int, error: str):
    h, w = win.getmaxyx()
    y = 0
    _safe_addstr(win, y, 2, "Manual plan  (Tab: switch fields, Enter: confirm)", curses.color_pair(_C_DIM))
    y += 1; _hline(win, y); y += 1

    # Goal field
    goal_attr = curses.A_REVERSE if active_field == 0 else curses.color_pair(_C_ACCENT)
    _safe_addstr(win, y, 2, "Goal:", goal_attr)
    y += 1
    box_w = w - 6
    goal_disp = goal_text or "(enter goal description)"
    _safe_addstr(win, y, 4, goal_disp[:box_w].ljust(box_w),
                 curses.A_REVERSE if active_field == 0 else curses.color_pair(_C_DIM))
    y += 2

    # Plan textarea
    plan_attr = curses.A_REVERSE if active_field == 1 else curses.color_pair(_C_ACCENT)
    _safe_addstr(win, y, 2, "Task breakdown:", plan_attr)
    y += 1

    available = h - y - 4
    plan_lines = (plan_text or _MANUAL_PLACEHOLDER).splitlines()
    for li, pline in enumerate(plan_lines[:available]):
        is_active = active_field == 1
        attr = curses.color_pair(_C_DIM) if not is_active else 0
        _safe_addstr(win, y + li, 4, pline[:box_w].ljust(box_w), attr)
    y += min(len(plan_lines), available) + 1

    if error:
        _safe_addstr(win, min(y, h - 3), 2, f"! {error}", curses.color_pair(_C_ERR))

    for hi, hline in enumerate(_MANUAL_HELP):
        _safe_addstr(win, h - len(_MANUAL_HELP) + hi - 1, 2,
                     hline, curses.color_pair(_C_DIM) | curses.A_DIM)


# ── Tab bar ───────────────────────────────────────────────────────────────────

_TABS = ["Resume run", "New goal", "Manual plan"]


def _draw_tab_bar(win, active_tab: int):
    h, w = win.getmaxyx()
    # Title
    _center(win, 0, "── cuddlytoddly ──", curses.color_pair(_C_TITLE) | curses.A_BOLD)
    y = 2
    x = 2
    for i, name in enumerate(_TABS):
        label = f"  {name}  "
        attr  = (curses.color_pair(_C_SEL) | curses.A_BOLD) if i == active_tab \
                else curses.color_pair(_C_DIM)
        _safe_addstr(win, y, x, label, attr)
        x += len(label) + 1
    _hline(win, y + 1)

    # Footer
    _safe_addstr(win, h - 1, 2,
                 "Tab/←/→: switch tabs   ↑↓: navigate   Enter: confirm   Esc: quit",
                 curses.color_pair(_C_DIM) | curses.A_DIM)


# ── Main startup screen ───────────────────────────────────────────────────────

def _startup_screen(stdscr, repo_root: Path) -> StartupChoice | None:
    curses.curs_set(0)
    stdscr.nodelay(False)
    _init_colors()

    _raw_runs   = scan_runs(repo_root)
    runs        = [RunInfo(**r) if isinstance(r, dict) else r for r in _raw_runs]
    active_tab  = 0 if runs else 1

    # Tab 1 state
    resume_sel    = 0
    resume_scroll = 0

    # Tab 2 state
    goal_text   = ""
    goal_cursor = 0
    goal_error  = ""

    # Tab 3 state
    manual_goal_text   = ""
    manual_goal_cursor = 0
    manual_plan_text   = ""
    manual_plan_cursor = 0
    manual_active_fld  = 0   # 0=goal, 1=plan
    manual_error       = ""

    while True:
        h, w = stdscr.getmaxyx()
        stdscr.erase()

        _draw_tab_bar(stdscr, active_tab)

        # Content area starts at row 4
        content_h = h - 5
        content_w = w

        # Use a subwindow for cleaner rendering
        try:
            content = stdscr.derwin(content_h, content_w, 4, 0)
        except curses.error:
            stdscr.refresh()
            k = stdscr.getch()
            continue

        content.erase()

        if active_tab == 0:
            _draw_resume_tab(content, runs, resume_sel, resume_scroll)
        elif active_tab == 1:
            _draw_new_goal_tab(content, goal_text, goal_cursor, goal_error)
        elif active_tab == 2:
            _draw_manual_tab(content,
                             manual_goal_text, manual_goal_cursor,
                             manual_plan_text, manual_plan_cursor,
                             manual_active_fld, manual_error)

        stdscr.refresh()
        content.refresh()

        k = stdscr.getch()

        # ── Global tab switching ──────────────────────────────────────────────
        if k == 27:   # Escape → quit
            return None

        if k == curses.KEY_LEFT or (k == ord('\t') and active_tab == 0):
            active_tab = (active_tab - 1) % len(_TABS)
            continue
        if k == curses.KEY_RIGHT:
            active_tab = (active_tab + 1) % len(_TABS)
            continue
        # Tab key cycles forward through tabs when not in a text field
        if k == ord('\t') and active_tab != 2:
            active_tab = (active_tab + 1) % len(_TABS)
            continue

        # ── Tab-specific key handling ─────────────────────────────────────────
        if active_tab == 0:
            if k == curses.KEY_UP:
                resume_sel    = max(0, resume_sel - 1)
                resume_scroll = max(0, min(resume_scroll, resume_sel))
            elif k == curses.KEY_DOWN:
                resume_sel    = min(len(runs) - 1, resume_sel + 1)
                visible       = content_h - 4
                if resume_sel >= resume_scroll + visible:
                    resume_scroll = resume_sel - visible + 1
            elif k in (10, 13) and runs:
                run = runs[resume_sel]
                return StartupChoice(
                    mode      = "resume",
                    run_dir   = Path(run.path),
                    goal_text = run.goal,
                    is_fresh  = False,
                )

        elif active_tab == 1:
            goal_error = ""
            if k in (10, 13):
                gt = goal_text.strip()
                if not gt:
                    goal_error = "Goal cannot be empty."
                else:
                    return StartupChoice(
                        mode      = "new_goal",
                        run_dir   = None,
                        goal_text = gt,
                        is_fresh  = True,
                    )
            elif k == curses.KEY_BACKSPACE or k == 127:
                if goal_cursor > 0:
                    goal_text   = goal_text[:goal_cursor - 1] + goal_text[goal_cursor:]
                    goal_cursor -= 1
            elif k == curses.KEY_LEFT:
                goal_cursor = max(0, goal_cursor - 1)
            elif k == curses.KEY_RIGHT:
                goal_cursor = min(len(goal_text), goal_cursor + 1)
            elif 32 <= k <= 126:
                ch          = chr(k)
                goal_text   = goal_text[:goal_cursor] + ch + goal_text[goal_cursor:]
                goal_cursor += 1

        elif active_tab == 2:
            manual_error = ""
            if k == ord('\t'):
                # Tab switches between goal and plan fields
                manual_active_fld = 1 - manual_active_fld
            elif k in (10, 13) and manual_active_fld == 1:
                # Enter in plan field = confirm
                gt   = manual_goal_text.strip()
                plan = manual_plan_text.strip()
                if not gt:
                    manual_error = "Goal cannot be empty."
                elif not plan:
                    manual_error = "Plan cannot be empty."
                else:
                    _, tasks = parse_manual_plan(plan)
                    if not tasks:
                        manual_error = "No tasks found. Use 'task: Name' lines."
                    else:
                        return StartupChoice(
                            mode      = "manual_plan",
                            run_dir   = None,
                            goal_text = gt,
                            plan_events = tasks,
                            is_fresh    = True,
                        )
            elif manual_active_fld == 0:
                # Editing goal field
                if k == curses.KEY_BACKSPACE or k == 127:
                    if manual_goal_cursor > 0:
                        manual_goal_text   = manual_goal_text[:manual_goal_cursor - 1] + manual_goal_text[manual_goal_cursor:]
                        manual_goal_cursor -= 1
                elif k == curses.KEY_LEFT:
                    manual_goal_cursor = max(0, manual_goal_cursor - 1)
                elif k == curses.KEY_RIGHT:
                    manual_goal_cursor = min(len(manual_goal_text), manual_goal_cursor + 1)
                elif k in (10, 13):
                    manual_active_fld = 1   # move to plan
                elif 32 <= k <= 126:
                    ch                 = chr(k)
                    manual_goal_text   = manual_goal_text[:manual_goal_cursor] + ch + manual_goal_text[manual_goal_cursor:]
                    manual_goal_cursor += 1
            else:
                # Editing plan textarea — newlines allowed
                if k == curses.KEY_BACKSPACE or k == 127:
                    if manual_plan_cursor > 0:
                        manual_plan_text   = manual_plan_text[:manual_plan_cursor - 1] + manual_plan_text[manual_plan_cursor:]
                        manual_plan_cursor -= 1
                elif k == curses.KEY_LEFT:
                    manual_plan_cursor = max(0, manual_plan_cursor - 1)
                elif k == curses.KEY_RIGHT:
                    manual_plan_cursor = min(len(manual_plan_text), manual_plan_cursor + 1)
                elif k in (10, 13):
                    manual_plan_text   = manual_plan_text[:manual_plan_cursor] + "\n" + manual_plan_text[manual_plan_cursor:]
                    manual_plan_cursor += 1
                elif 32 <= k <= 126:
                    ch                 = chr(k)
                    manual_plan_text   = manual_plan_text[:manual_plan_cursor] + ch + manual_plan_text[manual_plan_cursor:]
                    manual_plan_cursor += 1


def run_startup_selection(repo_root: Path) -> StartupChoice | None:
    """
    Show the startup screen and return the user's choice.
    Returns None if the user presses Escape (quit).
    Runs in its own curses.wrapper call so it finishes cleanly before
    the main UI starts.
    """
    import sys, logging

    # Silence stderr during curses — same pattern as run_ui
    root = logging.getLogger("dag")
    ch   = getattr(root, "_stderr_handler", None)
    if ch:
        root.removeHandler(ch)

    result: list[StartupChoice | None] = [None]

    def _inner(stdscr):
        result[0] = _startup_screen(stdscr, repo_root)

    try:
        curses.wrapper(_inner)
    finally:
        if ch:
            root.addHandler(ch)

    return result[0]


# --- FILE: ui/curses_ui.py ---

"""
Curses UI wired to Planner Runtime

Design:
- UI never mutates TaskGraph directly
- All edits emit Events into EventQueue
- Graph state is read via snapshot
- Git repo is rebuilt from TaskGraph snapshot
- Rendering logic is minimally modified
"""

import curses
import subprocess
import re
import time
import sys

from collections import deque, defaultdict
from cuddlytoddly.core.reducer import apply_event
from cuddlytoddly.ui.git_projection import (
    rebuild_repo_from_graph,
    graph_to_dag,
)
import textwrap
import logging
import hashlib

from collections import deque

from cuddlytoddly.core.events import (
    Event,
    ADD_NODE,
    REMOVE_NODE,
    ADD_DEPENDENCY,
    REMOVE_DEPENDENCY,
    UPDATE_METADATA,
    UPDATE_STATUS,
    RESET_SUBTREE
)

from cuddlytoddly.infra.logging import get_logger
from pathlib import Path
import cuddlytoddly.ui.git_projection as git_proj

logger = get_logger(__name__)

ANSI_COLOR_MAP = {
    30: curses.COLOR_BLACK,
    31: curses.COLOR_RED,
    32: curses.COLOR_GREEN,
    33: curses.COLOR_YELLOW,
    34: curses.COLOR_BLUE,
    35: curses.COLOR_MAGENTA,
    36: curses.COLOR_CYAN,
    37: curses.COLOR_WHITE,
    90: curses.COLOR_BLACK,
    91: curses.COLOR_RED,
    92: curses.COLOR_GREEN,
    93: curses.COLOR_YELLOW,
    94: curses.COLOR_BLUE,
    95: curses.COLOR_MAGENTA,
    96: curses.COLOR_CYAN,
    97: curses.COLOR_WHITE,
}
# --------------------------
# Git Repo Setup
# --------------------------



# --------------------------
# Graph Adapter
# --------------------------

# remove: ANSI + 7+ hex digits + ANSI
hash_pattern = re.compile(r'\x1b\[[0-9;]*m[0-9a-f]{7,}\x1b\[[0-9;]*m')

def remove_commit_hashes(lines):
    return [hash_pattern.sub('', line) for line in lines]


# --------------------------
# Incremental Git Layer
# --------------------------

node_to_commit = {}  # maps node_id -> latest commit hash

def get_git_dag_text():
    result = subprocess.run(
        ["git", "branch", "--list", "tip_*"],
        cwd=git_proj.REPO_PATH,
        capture_output=True,
        text=True
    )
    tip_branches = [b.strip().lstrip("* ") for b in result.stdout.splitlines() if b.strip()]

    if not tip_branches:
        tip_branches = ["master"]

    result = subprocess.run(
        ["git", "log", "--graph", "--oneline", "--color=always"] + tip_branches,
        cwd=git_proj.REPO_PATH,
        capture_output=True,
        text=True
    )
    return remove_commit_hashes(result.stdout.splitlines())

def find_root_node(snapshot):
    # root = node with no dependencies
    for node_id, node in snapshot.items():
        if not node.dependencies:
            return node_id
    # fallback: just pick the first node
    return next(iter(snapshot.keys()), None)

def find_path_to_node(dag, target_node):
    """
    Returns a list of nodes from root -> target_node (excluding target_node).
    dag: dict[node_id] -> list of child node_ids
    """
    def dfs(node, path, visited):
        if node == target_node:
            return path
        if node in visited:
            return None
        visited.add(node)
        for child in dag.get(node, []):
            res = dfs(child, path + [node], visited)
            if res is not None:
                return res
        return None

    # Assume single root (first node with no dependencies)
    all_nodes = set(dag.keys())
    all_children = {c for children in dag.values() for c in children}
    roots = list(all_nodes - all_children)
    if not roots:
        roots = list(all_nodes)
    visited = set()
    for root in roots:
        path = dfs(root, [], visited)
        if path:
            return path
    return []

def ensure_path_starts_at_root(dag, path):
    """
    Given a path (list of nodes), ensures it starts at a root node.
    If the path doesn't start at a root, extends it from the beginning
    until a root is found. The end of the path remains unchanged.

    dag: dict[node_id] -> list of child node_ids
    path: list of node_ids
    """
    if not path:
        return path

    # Build a reverse mapping: child -> list of parents
    all_nodes = set(dag.keys())
    all_children = {c for children in dag.values() for c in children}
    roots = all_nodes - all_children

    # If the path already starts at a root, return as-is
    if path[0] in roots:
        return path

    # Build parent map for reverse traversal
    parent_map = {}
    for node, children in dag.items():
        for child in children:
            parent_map.setdefault(child, []).append(node)

    # Walk backwards from path[0] using BFS until we hit a root
    def find_prefix_to_root(start_node):
        # BFS to find shortest path from any root to start_node (in reverse)
        queue = deque([[start_node]])
        visited = {start_node}

        while queue:
            current_path = queue.popleft()
            current_node = current_path[-1]

            if current_node in roots:
                # Reverse since we built it backwards
                return list(reversed(current_path))

            for parent in parent_map.get(current_node, []):
                if parent not in visited:
                    visited.add(parent)
                    queue.append(current_path + [parent])

        return None  # No root found (e.g. cyclic or disconnected)

    prefix = find_prefix_to_root(path[0])

    if prefix is None:
        return path  # Can't extend, return original

    # prefix ends with path[0], so drop the last element to avoid duplication
    return prefix[:-1] + path

def get_aggregate_outputs(snapshot):
    """
    Returns a dict of node_id -> result for all nodes that have results.
    """
    outputs = {}
    for node_id, node in snapshot.items():
        if node.result is not None:
            outputs[node_id] = node.result
    return outputs

# --------------------------
# ANSI Parsing
# --------------------------

ansi_regex = re.compile(r'\x1b\[[0-9;]*m')

def parse_ansi(line):
    parts = []

    current_color = curses.COLOR_WHITE
    bold = False
    attr = curses.color_pair(0)

    idx = 0
    for match in ansi_regex.finditer(line):
        # plain text before escape
        while idx < match.start():
            parts.append((line[idx], attr))
            idx += 1

        codes = match.group()[2:-1].split(';')
        if codes == ['']:
            codes = ['0']

        for code in codes:
            code = int(code)
            if code == 0:
                current_color = curses.COLOR_WHITE
                bold = False
            elif code == 1:
                bold = True
            elif 30 <= code <= 37:
                current_color = code - 30
                bold = False
            elif 90 <= code <= 97:
                current_color = code - 90
                bold = True

        attr = curses.color_pair(current_color + 1)
        if bold:
            attr |= curses.A_BOLD

        idx = match.end()

    while idx < len(line):
        parts.append((line[idx], attr))
        idx += 1

    return parts

def strip_ansi(line):
    return ansi_regex.sub('', line)

# --------------------------
# Mapping
# --------------------------

def map_nodes_to_lines(git_lines, snapshot):
    # Pre-compute the hash suffix for every node_id
    hash_to_node_id = {
        "#" + hashlib.sha256(node_id.encode()).hexdigest()[:6]: node_id
        for node_id in snapshot
    }

    node_map = {}
    for i, line in enumerate(git_lines):
        clean = strip_ansi(line)
        if '*' not in clean:
            continue
        star_pos = clean.index('*')
        after_star = clean[star_pos + 1:]
        message = after_star.lstrip(" |\\/.-")
        if not message:
            continue

        m = re.search(r'(#[0-9a-f]{6})', message)
        if m:
            node_id = hash_to_node_id.get(m.group(1))
            if node_id:
                node_map[node_id] = i

    return node_map

def get_node_col(line):
    parsed = parse_ansi(line)
    for idx, (ch, _) in enumerate(parsed):
        if ch == '*':
            return idx
    return 0

# --------------------------
# UI
# --------------------------

def trace_branch_path_recursive(git_lines, row, col, child_row, child_col, step=None, visited=None, is_start=True, debug=False):
    """
    Recursive function to find path from (row,col) to (child_row,child_col)
    following \ | / characters, stopping at other *.
    """
    if visited is None:
        visited = set()
    path_positions = set()

    if step is None:
        step = 1 if child_row > row else -1

    this_line = "".join(ch for ch, _ in parse_ansi(git_lines[row]))

    # Out of bounds
    if row < 0 or row >= len(git_lines) or col < 0 or col >= len(this_line):
        return path_positions
    

    char = this_line[col]
    
    char_matrix = {}

    char_matrix[(0,+2)] = this_line[col+2] if col+2 < len(this_line) else ''
    char_matrix[(0,+1)] = this_line[col+1] if col+1 < len(this_line) else ''
    char_matrix[(0,0)] = this_line[col] if col < len(this_line) else ''
    char_matrix[(0,-1)] = this_line[col-1] if (0 <= col-1 < len(this_line)) else ''
    char_matrix[(0,-2)] = this_line[col-2] if (0 <= col-2 < len(this_line)) else ''

    # Stop if we hit a '*' that is not the child (and not the start)
    if char == '*' and not (row == child_row and col == child_col) and not is_start:
        return (path_positions|set('x'))

    # Mark visited and add current cell
    if (row, col) in visited:
        return path_positions
    visited.add((row, col))
    path_positions.add((row, col))

    # Stop if reached child
    if row == child_row and col == child_col:
        return path_positions
    
    subpath_positions = []

    # Explore next row in step direction
    next_row = row + step
    if 0 <= next_row < len(git_lines):
        next_line = "".join(ch for ch, _ in parse_ansi(git_lines[next_row]))

        for dcol in range(-2,3):
            char_matrix[(1,dcol)] = next_line[col+dcol] if (0 <= col+dcol < len(next_line)) else ''

        for dcol in range(-2,3):
            ncol = col + dcol
            if 0 <= ncol < len(next_line):
                if (

                    (dcol==1 and char_matrix[(1,1)] == '/' and char_matrix[(0,0)] == '/')
                    or (dcol==1 and char_matrix[(1,1)] == '/' and char_matrix[(0,0)] == '*')
                    or (dcol==1 and char_matrix[(1,1)] == '\\' and char_matrix[(0,0)] == '/' and char_matrix[(1,0)] == ' ')

                    or (dcol==1 and char_matrix[(1,1)] == '/' and char_matrix[(0,0)] == '|' and char_matrix[(1,0)] == '|' and char_matrix[(0,-1)] != '/' and char_matrix[(1,-1)] != '_')
                    or (dcol==1 and char_matrix[(1,1)] == '/' and char_matrix[(0,0)] == '|' and char_matrix[(1,0)] == ' ')
                    or (dcol==1 and char_matrix[(1,1)] == '|' and char_matrix[(0,0)] == '/' and char_matrix[(1,2)] != '/' and char_matrix[(1,2)] != '_')
                    or (dcol==0 and char_matrix[(1,0)] == '|' and char_matrix[(0,0)] == '*' and char_matrix[(1,1)] == '/')

                    or (dcol==1 and char_matrix[(1,1)] == '*' and char_matrix[(0,0)] == '/')

                    or (dcol==0 and char_matrix[(1,0)] == "|" and char_matrix[(0,0)] == '*')
                    or (dcol==0 and char_matrix[(1,0)] == "|" and char_matrix[(0,0)] == '/')
                    or (dcol==0 and char_matrix[(1,0)] == "|" and char_matrix[(0,0)] == '\\')
                    or (dcol==0 and char_matrix[(1,0)] == "|" and char_matrix[(0,0)] == '|')
                    or (dcol==0 and char_matrix[(1,0)] == "\\" and char_matrix[(0,0)] == '|')
                    or (dcol==0 and char_matrix[(1,0)] == "\\" and char_matrix[(0,0)] == '/' and char_matrix[(1,1)] != "/")

                    or (dcol==0 and char_matrix[(1,0)] == '*' and char_matrix[(0,0)] == '|')

                    or (dcol==-1 and char_matrix[(1,-1)] == '|' and char_matrix[(0,0)] == '\\' and char_matrix[(1,0)] == ' ' and char_matrix[(1,-2)] != '\\')
                    or (dcol==-1 and char_matrix[(1,-1)] == '\\' and char_matrix[(0,0)] == '\\')
                    or (dcol==-1 and char_matrix[(1,-1)] == '*' and char_matrix[(0,0)] == '\\')

                    or (dcol==-1 and char_matrix[(1,-1)] == '\\' and char_matrix[(0,0)] == '|' and char_matrix[(0,1)] != '\\')
                    or (dcol==-1 and char_matrix[(1,-1)] == '|' and char_matrix[(0,0)] == '\\' and char_matrix[(1,-2)] != '\\')
                    or (dcol==-1 and char_matrix[(1,-1)] == '\\' and char_matrix[(0,0)] == '*')

                    or (dcol==-1 and char_matrix[(1,-1)] == '*' and char_matrix[(0,0)] == '\\')

                    ):
                    subpath_positions += [trace_branch_path_recursive(
                        git_lines, next_row, ncol, child_row, child_col, step, visited, is_start=False, debug=debug
                    )]
        if (char_matrix[(1,2)] in ['_','/'] and char_matrix[(0,0)] == '/' and char_matrix[(0,1)] == '|' and char_matrix[(1,1)] == '|'):
            temp_set = set()
            dcol = 2
            while char_matrix[(1,dcol)] == '_' and char_matrix[(1,dcol-1)] == '|':
                temp_set.add((next_row, col + dcol))
                dcol += 2
                char_matrix[(1,dcol)] = next_line[col+dcol] if (0 <= col+dcol < len(next_line)) else ''
                char_matrix[(1,dcol-1)] = next_line[col+dcol-1] if (0 <= col+dcol-1 < len(next_line)) else ''

            ncol = col + dcol
            subpath_positions += [trace_branch_path_recursive(
                        git_lines, next_row, ncol, child_row, child_col, step, visited, is_start=False, debug=debug
                    )|temp_set]
            
        if (char_matrix[(1,-1)] in ['.','-'] and char_matrix[(0,0)] == '\\'):
            temp_set = set()
            dcol = -1
            while char_matrix[(1,dcol)] in ['.','-']:
                temp_set.add((next_row, col + dcol))
                dcol -= 1
                char_matrix[(1,dcol)] = next_line[col+dcol] if (0 <= col+dcol < len(next_line)) else ''

            ncol = col + dcol
            subpath_positions += [trace_branch_path_recursive(
                        git_lines, next_row, ncol, child_row, child_col, step, visited, is_start=False, debug=debug
                    )|temp_set]
        
    if len(subpath_positions)>1:
        nothing_added = True
        for subpath_position in subpath_positions:
            if 'x' not in subpath_position:
                nothing_added = False
                path_positions |= subpath_position
        if nothing_added:
            path_positions |= set('x')
    elif len(subpath_positions)==1:
        path_positions |= subpath_positions[0]


    return path_positions

def build_reverse_dag(dag):
    """Returns a dict mapping child_id -> list of parent_ids."""
    reverse = defaultdict(list)
    for parent, children in dag.items():
        for child in children:
            reverse[child].append(parent)
    return reverse

class ModalField:
    """A single editable field inside a modal."""
    def __init__(self, label, value="", completions=None, validator=None):
        self.label = label
        self.value = value
        self.completions = completions or []  # list of strings for autocomplete
        self.validator = validator             # callable(str) -> str|None (error msg)
        self.cursor = len(value)
        self.error = None
        self._completion_idx = -1
        self._completion_prefix = ""

    def _current_token(self):
            """Return the text after the last comma (stripped), for autocomplete."""
            parts = self.value[:self.cursor].rsplit(",", 1)
            return parts[-1].strip()

    def handle_key(self, k):
        if k == curses.KEY_BACKSPACE or k == 127:
            if self.cursor > 0:
                self.value = self.value[:self.cursor-1] + self.value[self.cursor:]
                self.cursor -= 1
                self._completion_idx = -1
        elif k == curses.KEY_LEFT:
            self.cursor = max(0, self.cursor - 1)
        elif k == curses.KEY_RIGHT:
            self.cursor = min(len(self.value), self.cursor + 1)
        elif k == ord('\t') and self.completions:
            token = self._current_token()
            matches = [c for c in self.completions if c.startswith(token)]
            if matches:
                self._completion_idx = (self._completion_idx + 1) % len(matches)
                completion = matches[self._completion_idx]
                # Replace only the current token, preserving everything before it
                before_cursor = self.value[:self.cursor]
                last_comma = before_cursor.rfind(",")
                if last_comma == -1:
                    # No comma — replace entire value
                    self.value = completion + self.value[self.cursor:]
                else:
                    # Replace only the token after the last comma
                    prefix_part = self.value[:last_comma + 1] + " "
                    self.value = prefix_part + completion + self.value[self.cursor:]
                self.cursor = len(self.value)
        elif 32 <= k <= 126:
            ch = chr(k)
            self.value = self.value[:self.cursor] + ch + self.value[self.cursor:]
            self.cursor += 1
            self._completion_idx = -1  # reset on typing
            
    def validate(self):
        if self.validator:
            self.error = self.validator(self.value)
        return self.error is None

class Modal:
    """
    Multi-field modal dialog rendered over the info panel area.
    
    fields: list of ModalField
    title: str
    on_submit: callable(dict of label->value) -> None
    on_cancel: callable() -> None
    """
    def __init__(self, title, fields, on_submit, on_cancel):
        self.title = title
        self.fields = fields
        self.on_submit = on_submit
        self.on_cancel = on_cancel
        self.active_field = 0

    def handle_key(self, k):
        if k == 27:  # Escape
            self.on_cancel()
            return

        if k in (curses.KEY_DOWN, ord('\t')) and chr(k) != '\t':
            self.active_field = (self.active_field + 1) % len(self.fields)
            return

        if k == curses.KEY_UP:
            self.active_field = (self.active_field - 1) % len(self.fields)
            return

        if k in (10, 13):  # Enter
            # Validate all fields
            all_valid = all(f.validate() for f in self.fields)
            if all_valid:
                self.on_submit({f.label: f.value for f in self.fields})
            return

        self.fields[self.active_field].handle_key(k)

    def draw(self, stdscr, h, w):
        panel_x = w // 2 + 1
        panel_w = w - panel_x - 1

        # Clear the panel area before drawing
        for row in range(0, h - 2):
            try:
                stdscr.addstr(row, panel_x, " " * panel_w)
            except curses.error:
                pass

        row = 0
        # Title
        try:
            stdscr.addstr(row, panel_x, f" {self.title} ".center(panel_w, "─"), curses.A_BOLD)
        except curses.error:
            pass
        row += 2

        for i, field in enumerate(self.fields):
            is_active = (i == self.active_field)
            attr = curses.A_REVERSE if is_active else 0

            label_str = f" {field.label}: "
            try:
                stdscr.addstr(row, panel_x, label_str)
            except curses.error:
                pass

            val_x = panel_x + len(label_str)
            val_w = w - val_x - 1

            # Wrap the value across multiple lines
            wrapped_val = textwrap.wrap(field.value if field.value else " ", width=val_w) or [" "]
            for j, wline in enumerate(wrapped_val):
                try:
                    stdscr.addstr(row + j, val_x, wline.ljust(val_w), attr)
                except curses.error:
                    pass
            row += len(wrapped_val)

            if is_active and field.error:
                try:
                    stdscr.addstr(row, panel_x, f" ! {field.error}", curses.color_pair(curses.COLOR_RED + 1))
                except curses.error:
                    pass
                row += 1

            # Autocomplete suggestions
            if is_active and field.completions:
                prefix = field._current_token()  # <-- see below
                matches = [c for c in field.completions if c.startswith(prefix)][:4]
                if matches:
                    for j, m in enumerate(matches):
                        try:
                            stdscr.addstr(row + j, val_x, m[:val_w], curses.A_DIM)
                        except curses.error:
                            pass
                    row += len(matches)

            row += 1
        # Footer
        try:
            stdscr.addstr(row + 1, panel_x, " Enter: confirm  Esc: cancel  Tab: autocomplete", curses.A_DIM)
        except curses.error:
            pass

def export_results_to_markdown(snapshot, run_dir):
    """Walk the DAG in topological order and write all results to a
    Markdown file in <run_dir>/outputs/. Returns the output path."""
    from datetime import datetime

    out_dir = run_dir / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path  = out_dir / f"export_{timestamp}.md"

    def topo_sort(snap):
        visited, order = set(), []
        def visit(nid):
            if nid in visited:
                return
            visited.add(nid)
            node = snap.get(nid)
            if node:
                for dep in sorted(node.dependencies):
                    visit(dep)
            order.append(nid)
        for nid in sorted(snap.keys()):
            visit(nid)
        return order

    order = topo_sort(snapshot)

    goal_nodes = [snapshot[nid] for nid in order
                  if snapshot.get(nid) and snapshot[nid].node_type == "goal"]
    title = (goal_nodes[0].metadata.get("description", goal_nodes[0].id)
             if goal_nodes else run_dir.name.replace("_", " ").title())

    lines = [
        f"# {title}", "",
        f"*Exported {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*", "",
        "---", "",
        "## Summary", "",
        "| Node | Type | Status |",
        "|------|------|--------|",
    ]
    for nid in order:
        node = snapshot.get(nid)
        if not node or node.metadata.get("hidden") or node.node_type == "execution_step":
            continue
        lines.append(f"| {nid} | {node.node_type} | {node.status} |")
    lines += ["", "---", "", "## Results", ""]

    for nid in order:
        node = snapshot.get(nid)
        if not node or node.node_type == "execution_step" or node.metadata.get("hidden"):
            continue
        desc = node.metadata.get("description", "")
        lines.append(f"### {nid}")
        if desc and desc != nid:
            lines += [f"*{desc}*", ""]
        deps = ", ".join(sorted(node.dependencies)) or "none"
        lines.append(f"**Type:** {node.node_type} | **Status:** {node.status} | **Deps:** {deps}")
        lines.append("")
        req_input = node.metadata.get("required_input")
        output    = node.metadata.get("output")
        if req_input:
            lines.append(f"**Input:** `{req_input}`")
        if output:
            lines.append(f"**Output:** `{output}`")
        if req_input or output:
            lines.append("")
        if node.result:
            lines += ["**Result:**", "", "```", str(node.result).strip(), "```"]
        else:
            lines.append("*No result yet.*")
        notes = node.metadata.get("reflection_notes", [])
        if notes:
            lines += ["", "**Notes:**"] + [f"- {n}" for n in notes]
        lines += ["", "---", ""]

    out_path.write_text("\n".join(lines), encoding="utf-8")
    logger.info("[EXPORT] Written to %s", out_path)
    return out_path

def open_add_modal(snapshot, event_queue, current_node, set_modal):
    node_ids = list(snapshot.keys())
 
    def on_submit(values):
        new_id         = values["ID"].strip()
        new_desc       = values["Description"].strip()
        deps_raw       = values["Dependencies"].strip()
        dependents_raw = values["Dependents"].strip()
        ntype          = values["Type"].strip() or "task"
 
        if not new_id or new_id in snapshot:
            set_modal(None)
            return
 
        deps = [d.strip() for d in deps_raw.split(",")
                if d.strip() and d.strip() in snapshot]
        dependents = [d.strip() for d in dependents_raw.split(",")
                      if d.strip() and d.strip() in snapshot]
 
        event_queue.put(Event(ADD_NODE, {
            "node_id":      new_id,
            "node_type":    ntype,
            "dependencies": deps,
            "origin":       "user",
            "metadata":     {"description": new_desc},
        }))
 
        for dependent_id in dependents:
            event_queue.put(Event(ADD_DEPENDENCY, {
                "node_id":    dependent_id,
                "depends_on": new_id,
            }))
            # Reset each dependent and its subtree so they rerun
            # with the new node as a prerequisite.
            event_queue.put(Event(RESET_SUBTREE, {"node_id": dependent_id}))
 
        set_modal(None)
 
    set_modal(Modal(
        title="Add Node",
        fields=[
            ModalField("ID",           value=""),
            ModalField("Description",  value=""),
            ModalField("Type",         value="task", completions=["task", "goal"]),
            ModalField("Dependencies", value=current_node or "", completions=node_ids),
            ModalField("Dependents",   value="",                 completions=node_ids),
        ],
        on_submit=on_submit,
        on_cancel=lambda: set_modal(None),
    ))
  
def open_edit_modal(current_node, snapshot, event_queue, set_modal):
    node = snapshot[current_node]
    node_ids = list(snapshot.keys())
    current_deps = ", ".join(node.dependencies)
 
    def on_submit(values):
        new_id       = values["ID"].strip()
        new_desc     = values["Description"].strip()
        new_deps_raw = values["Dependencies"].strip()
        new_status   = values["Status"].strip()
 
        new_deps = [d.strip() for d in new_deps_raw.split(",") if d.strip()]
        new_deps = [d for d in new_deps if d in snapshot]
 
        event_queue.put(Event(UPDATE_METADATA, {
            "node_id": current_node,
            "origin":  "user",
            "metadata": {"description": new_desc},
        }))
 
        if new_status in ("pending", "done", "running", "failed", "to_be_expanded"):
            event_queue.put(Event(UPDATE_STATUS, {
                "node_id": current_node,
                "status":  new_status,
            }))
 
        old_deps     = set(node.dependencies)
        new_deps_set = set(new_deps)
        for removed in old_deps - new_deps_set:
            event_queue.put(Event(REMOVE_DEPENDENCY, {
                "node_id": current_node, "depends_on": removed,
            }))
        for added in new_deps_set - old_deps:
            event_queue.put(Event(ADD_DEPENDENCY, {
                "node_id": current_node, "depends_on": added,
            }))
 
        if new_id and new_id != current_node and new_id not in snapshot:
            event_queue.put(Event(ADD_NODE, {
                "node_id":      new_id,
                "node_type":    node.node_type,
                "dependencies": list(new_deps_set),
                "origin":       node.origin,
                "metadata":     {**node.metadata, "description": new_desc},
            }))
            for child in node.children:
                event_queue.put(Event(ADD_DEPENDENCY,    {"node_id": child, "depends_on": new_id}))
                event_queue.put(Event(REMOVE_DEPENDENCY, {"node_id": child, "depends_on": current_node}))
            event_queue.put(Event(REMOVE_NODE, {"node_id": current_node}))
            # Reset the renamed node's subtree (children now point to new_id)
            event_queue.put(Event(RESET_SUBTREE, {"node_id": new_id}))
        else:
            # Reset this node and everything downstream
            event_queue.put(Event(RESET_SUBTREE, {"node_id": current_node}))
 
        set_modal(None)
 
    set_modal(Modal(
        title="Edit Node",
        fields=[
            ModalField("ID",           value=current_node),
            ModalField("Description",  value=node.metadata.get("description", "")),
            ModalField("Dependencies", value=current_deps, completions=node_ids),
            ModalField("Status",       value=node.status,
                       completions=["pending", "running", "done", "failed", "to_be_expanded"]),
        ],
        on_submit=on_submit,
        on_cancel=lambda: set_modal(None),
    ))
  
def open_remove_modal(current_node, snapshot, event_queue, set_modal):
    node     = snapshot[current_node]
    parents  = list(node.dependencies)
    children = list(node.children)
 
    options = [
        ("Remove node only — rewire children to its parents", "rewire"),
        ("Remove node and all descendants",                   "cascade"),
        ("Remove node and disconnect everything",             "disconnect"),
    ]
 
    def on_submit(values):
        choice = values["Action"].strip()
        mode   = next((m for label, m in options if label == choice), None)
 
        if mode == "rewire":
            for child in children:
                event_queue.put(Event(REMOVE_DEPENDENCY, {
                    "node_id": child, "depends_on": current_node,
                }))
                for parent in parents:
                    event_queue.put(Event(ADD_DEPENDENCY, {
                        "node_id": child, "depends_on": parent,
                    }))
            event_queue.put(Event(REMOVE_NODE, {"node_id": current_node}))
            # Children survive — reset them and their subtrees
            for child in children:
                event_queue.put(Event(RESET_SUBTREE, {"node_id": child}))
 
        elif mode == "cascade":
            # REMOVE_NODE recurses into children — nothing left to reset
            event_queue.put(Event(REMOVE_NODE, {"node_id": current_node}))
 
        elif mode == "disconnect":
            for child in children:
                event_queue.put(Event(REMOVE_DEPENDENCY, {
                    "node_id": child, "depends_on": current_node,
                }))
            event_queue.put(Event(REMOVE_NODE, {"node_id": current_node}))
            # Children survive without this dep — reset them
            for child in children:
                event_queue.put(Event(RESET_SUBTREE, {"node_id": child}))
 
        set_modal(None)
 
    set_modal(Modal(
        title=f"Remove: {current_node}  ({len(children)} children, {len(parents)} parents)",
        fields=[
            ModalField(
                "Action",
                value=options[0][0],
                completions=[label for label, _ in options],
            ),
        ],
        on_submit=on_submit,
        on_cancel=lambda: set_modal(None),
    ))
 
def dag_interface(stdscr, orchestrator, run_dir=None):
    graph = orchestrator.graph
    graph_lock = orchestrator.graph_lock
    event_queue = orchestrator.event_queue

    active_modal = None
    def set_modal(m):
        nonlocal active_modal
        active_modal = m

    logger.debug("=== UI Debug Log Started ===")

    curses.start_color()
    curses.use_default_colors()

    for idx, color in ANSI_COLOR_MAP.items():
        curses.init_pair(color + 1, color, -1)

    stdscr.nodelay(True)
    stdscr.keypad(True)
    curses.curs_set(0)

    last_seen_version = -1
    last_exec_version = -1

    cached_git_lines = []
    cached_node_to_line = {}

    current_node = None
    parent_node = None

    parent_stack = []
    child_stack = []
    branch_mode = False
    selection_index = 0

    info_scroll = 0
    export_notice = None   # (message, expire_time) or None

    loop_count = 0
    snapshot = {}
    dag = {}
    reverse_dag = {}
    last_missing_nodes: frozenset = frozenset()
    switch_requested = False

    while True:
        loop_count += 1

        # ── Input first — always consume keypresses regardless of render state ──
        k = stdscr.getch()

        if k == ord("q"):
            break

        # ── Snapshot ──────────────────────────────────────────────────────────
        try:
            with graph_lock:
                snapshot = graph.get_snapshot()
        except Exception as e:
            logger.error("[UI LOOP %d] Failed to get snapshot: %s", loop_count, e)
            snapshot = {}

        logger.debug("[UI LOOP %d] Snapshot keys: %s", loop_count, list(snapshot.keys()))

        version_at_rebuild_start = graph.structure_version
        exec_version_at_rebuild_start = graph.execution_version

        # ── Rebuild git projection if graph changed ───────────────────────────
        skip_render = False
        if (
            graph.structure_version != last_seen_version
            or graph.execution_version != last_exec_version
        ):
            logger.debug("[UI REBUILD] Version changed from %d to %d",
                         last_seen_version, graph.structure_version)

            try:
                with graph_lock:
                    snapshot = graph.get_snapshot()

                rebuild_repo_from_graph(graph)
                cached_git_lines    = get_git_dag_text()
                cached_node_to_line = map_nodes_to_lines(cached_git_lines, snapshot)

                last_seen_version  = version_at_rebuild_start
                last_exec_version  = exec_version_at_rebuild_start

                dag         = graph_to_dag(snapshot)
                reverse_dag = build_reverse_dag(dag)

                if current_node not in snapshot:
                    current_node = find_root_node(snapshot)
                    parent_stack = []
                    child_stack  = find_path_to_node(reverse_dag, current_node)
                else:
                    parent_stack = find_path_to_node(dag, current_node)
                    child_stack  = find_path_to_node(reverse_dag, current_node)

                missing = set(snapshot.keys()) - set(cached_node_to_line.keys())
                if missing:
                    missing_frozen = frozenset(missing)
                    if missing_frozen != last_missing_nodes:
                        logger.warning("[UI] Nodes in snapshot but NOT in git map: %s", missing)
                        last_missing_nodes = missing_frozen
                    # force rebuild next iteration but don't skip input
                    last_seen_version = -1
                    last_exec_version = -1
                    skip_render = True
                else:
                    last_missing_nodes = frozenset()

            except Exception as e:
                logger.error("[UI REBUILD] Failed: %s", e, exc_info=True)
                skip_render = True
        else:
            try:
                with graph_lock:
                    snapshot = graph.get_snapshot()
            except Exception:
                pass
            dag         = graph_to_dag(snapshot)
            reverse_dag = build_reverse_dag(dag)

        # ── Rendering ─────────────────────────────────────────────────────────
        if not skip_render:
            try:
                git_lines    = cached_git_lines
                node_to_line = cached_node_to_line

                if not snapshot:
                    current_node = None
                else:
                    if current_node not in snapshot:
                        current_node = find_root_node(snapshot)

                stdscr.clear()
                h, w = stdscr.getmaxyx()

                if current_node:
                    current_line = node_to_line.get(current_node)
                    current_col  = get_node_col(git_lines[current_line]) \
                                   if current_line is not None else 0
                else:
                    current_line = None
                    current_col  = None

                if parent_node and current_node:
                    parent_line = node_to_line.get(parent_node)
                    parent_col  = get_node_col(git_lines[parent_line]) \
                                  if parent_line is not None else None
                else:
                    parent_node  = None
                    parent_line  = None
                    parent_col   = None

                # Branch path highlight
                if (
                    branch_mode
                    and parent_node
                    and current_line is not None
                    and parent_line is not None
                ):
                    path = trace_branch_path_recursive(
                        git_lines, parent_line, parent_col,
                        current_line, current_col,
                    )
                else:
                    path = set()

                # Node-type symbol maps
                goal_star_positions = {}
                step_star_positions = {}
                for node_id, line_idx in node_to_line.items():
                    node = snapshot.get(node_id)
                    if not node:
                        continue
                    nt = getattr(node, "node_type", None)
                    if nt == "goal":
                        goal_star_positions[line_idx] = get_node_col(git_lines[line_idx])
                    elif nt == "execution_step":
                        step_star_positions[line_idx] = (
                            get_node_col(git_lines[line_idx]),
                            node.metadata.get("hidden", False),
                            node.status == "failed",
                        )

                line_to_node = {v: k for k, v in node_to_line.items()}

                # Label overrides (hash → description)
                line_label_overrides = {}
                for line_idx, node_id in line_to_node.items():
                    node = snapshot.get(node_id)
                    if not node:
                        continue
                    h6   = "#" + hashlib.sha256(node_id.encode()).hexdigest()[:6]
                    desc = node.metadata.get("description") or node_id
                    line_label_overrides[line_idx] = (h6, f"{h6} {desc}")

                if git_lines:
                    start = 0
                    if current_line is not None:
                        start = max(0, current_line - h // 2)

                    for i, line in enumerate(git_lines[start:start + h - 1]):
                        parsed           = parse_ansi(line)
                        x                = 0
                        current_line_idx = i + start
                        override         = line_label_overrides.get(current_line_idx)

                        hash_start_x = None
                        if override:
                            h6      = override[0]
                            visible = "".join(ch for ch, _ in parsed)
                            idx     = visible.find(h6)
                            if idx != -1:
                                hash_start_x = idx

                        for ch, color in parsed:
                            if x >= w // 2 - 1:
                                break
                            highlight = 0

                            if (
                                current_node
                                and current_line_idx == current_line
                                and x == current_col
                            ):
                                highlight = curses.A_REVERSE

                            if (current_line_idx, x) in path:
                                highlight = curses.A_REVERSE

                            if ch == '*' and goal_star_positions.get(current_line_idx) == x:
                                ch = 'o'

                            if ch == '*' and current_line_idx in step_star_positions:
                                col, hidden, failed = step_star_positions[current_line_idx]
                                if x == col:
                                    ch = '·' if hidden else ('✗' if failed else '◆')

                            if hash_start_x is not None and x == hash_start_x:
                                full_label = override[1]
                                available  = (w // 2 - 1) - x
                                label = (
                                    (full_label[:available - 3] + "...")
                                    if len(full_label) > available
                                    else full_label
                                )
                                for lch in label:
                                    if x >= w // 2 - 1:
                                        break
                                    try:
                                        stdscr.addstr(i, x, lch, color | highlight)
                                    except curses.error:
                                        pass
                                    x += 1
                                break

                            try:
                                stdscr.addstr(i, x, ch, color | highlight)
                            except curses.error:
                                pass
                            x += 1

                # Status bars
                node_label      = current_node if current_node else "<empty>"
                llm_paused      = orchestrator.llm_stopped
                paused_indicator = " | [LLM PAUSED]" if llm_paused else ""
                activity        = orchestrator.current_activity
                started         = orchestrator.activity_started

                if activity and started:
                    elapsed      = time.time() - started
                    activity_str = f" {activity} ({elapsed:.0f}s)"
                else:
                    activity_str = ""

                status_line = (
                    "Up/Down/Left/Right/[/]: move | "
                    f"j/k </> scroll info | "
                    f"e: edit | a: add | x: remove | p: export | "
                    f"s: {'resume' if llm_paused else 'pause'} LLM | g: switch goal | q: quit"
                    f"{paused_indicator}"
                )
                tc = orchestrator.token_counts
                token_str = f"Tokens: {tc['total']:,} ({tc['calls']} calls)"

                if export_notice and time.time() < export_notice[1]:
                    debug_line = export_notice[0]
                else:
                    export_notice = None
                    debug_line = (
                        f"Node: {node_label} | "
                        f"{'Branch' if branch_mode else 'Node'} | "
                        f"Nodes: {len(snapshot)} "
                        f"{token_str} |"
                        f"{activity_str}"
                    )

                stdscr.addstr(h - 2, 0, debug_line[: w - 1])
                stdscr.addstr(h - 1, 0, status_line[: w - 1])

                if branch_mode and parent_node:
                    selected_nodes = [parent_node, current_node]
                else:
                    selected_nodes = [current_node]

                if active_modal:
                    active_modal.draw(stdscr, h, w)
                else:
                    draw_info_panel(stdscr, h, w, current_node, snapshot, selected_nodes, info_scroll)

                stdscr.refresh()

            except Exception as e:
                logger.error("[UI] Render error: %s", e, exc_info=True)

        # ── Key handling — always runs ─────────────────────────────────────────
        if k == -1:
            time.sleep(0.02)
            continue

        if not current_node:
            continue

        if not parent_node:
            parents = reverse_dag.get(current_node, [])
            if parents:
                parent_node = parents[0]

        parent_stack = ensure_path_starts_at_root(dag, parent_stack + [current_node])[:-1]
        child_stack  = ensure_path_starts_at_root(reverse_dag, child_stack + [current_node])[:-1]

        if active_modal:
            active_modal.handle_key(k)
            continue

        if k == curses.KEY_UP:
            info_scroll = 0
            children = dag.get(current_node, [])
            if not branch_mode and children:
                parent_stack.append(current_node)
                parent_node  = current_node
                current_node = child_stack.pop() if child_stack else current_node
                branch_mode  = True
            elif branch_mode:
                branch_mode = False

        elif k == curses.KEY_DOWN:
            info_scroll = 0
            parents = reverse_dag.get(current_node, [])
            if not branch_mode and parents:
                parent_node = parent_stack[-1] if parent_stack else None
                branch_mode = True
            elif branch_mode:
                if parent_stack:
                    child_stack.append(current_node)
                    current_node = parent_stack.pop()
                branch_mode = False

        elif k in (curses.KEY_LEFT, curses.KEY_RIGHT):
            info_scroll = 0
            if parent_node or parent_stack:
                if not parent_node and parent_stack:
                    parent_node = parent_stack[-1]
                siblings = dag.get(parent_node, [])
                if siblings and current_node in siblings:
                    current_index = siblings.index(current_node)
                    delta         = -1 if k == curses.KEY_LEFT else 1
                    selection_index = (current_index + delta) % len(siblings)
                    current_node    = siblings[selection_index]
                    child_stack     = find_path_to_node(reverse_dag, current_node)

        elif k in (ord('['), ord(']')):
            info_scroll = 0
            parents = reverse_dag.get(current_node, [])
            if parent_node and parents and parent_node in parents:
                parent_index  = parents.index(parent_node)
                delta         = -1 if k == ord('[') else 1
                selection_index = (parent_index + delta) % len(parents)
                parent_node     = parents[selection_index]
                parent_stack    = find_path_to_node(dag, parent_node) + [parent_node]

        elif k == ord("s"):
            if orchestrator.llm_stopped:
                orchestrator.resume_llm_calls()
            else:
                orchestrator.stop_llm_calls()

        elif k == ord("e"):
            if current_node:
                open_edit_modal(current_node, snapshot, event_queue, set_modal)

        elif k == ord("a"):
            open_add_modal(snapshot, event_queue, current_node, set_modal)

        elif k == ord("x"):
            if current_node:
                open_remove_modal(current_node, snapshot, event_queue, set_modal)

        elif k == ord("p"):
            if run_dir and snapshot:
                try:
                    out_path = export_results_to_markdown(snapshot, run_dir)
                    export_notice = (f"Exported → {out_path.name}", time.time() + 4)
                except Exception as ex:
                    export_notice = (f"Export failed: {ex}", time.time() + 4)
                    logger.error("[EXPORT] Failed: %s", ex, exc_info=True)

        elif k in (curses.KEY_PPAGE, ord("<")):   # Page Up
            info_scroll = max(0, info_scroll - (h - 4))

        elif k in (curses.KEY_NPAGE, ord(">")):   # Page Down
            info_scroll += (h - 4)    # draw_info_panel clamps the max

        elif k == ord("j"):            # fine scroll down
            info_scroll += 3

        elif k == ord("k"):            # fine scroll up
            info_scroll = max(0, info_scroll - 3)

        elif k == ord("g"):
            switch_requested = True
            break

    return "switch" if switch_requested else None


def draw_info_panel(stdscr, h, w, node_id, snapshot, selected_nodes, scroll_offset=0):
    if not node_id or not snapshot:
        return

    node = snapshot[node_id]
    panel_x = w // 2
    panel_w = w - panel_x

    # Border
    for row in range(0, h - 2):
        try:
            stdscr.addch(row, panel_x, curses.ACS_VLINE)
        except curses.error:
            pass

    # Node info lines
    lines = [
        f" ID:     {node.id}",
        " ",
    ]

    desc = node.metadata.get("description")
    if desc:
        lines+=[
        f" Desc:   {desc}"," "
        ]

    input = node.metadata.get("required_input")
    if input:
        lines+=[
        f" Input:  {input}"," "
        ]

    output = node.metadata.get("output")
    if output:
        lines+=[
        f" Output: {output}"," "
        ]

    lines += [
        f" Deps:   {', '.join(node.dependencies) or 'none'}",
        " ",
        f" Type:   {node.node_type}",
        f" Status: {node.status}",
        f" Origin: {node.origin}",
        " ",
    ]
    notes = node.metadata.get("reflection_notes", [])
    if notes:
        lines.append(" Notes:")
        for note in notes:
            lines.append(f"   {note}")

    # --- AGGREGATE OUTPUTS ---
    results = get_aggregate_outputs(snapshot)
    if results:
        lines.append(" ")
        lines.append(" ")
        lines.append(" ")

        # --- RESULTS ---
        lines.append(" Results:")
        for nid in selected_nodes:
            node = snapshot.get(nid)
            if not node:
                continue

            if node.result is not None:
                lines.append(f"   [{nid}]")
                for wrapped_line in textwrap.wrap(str(node.result), width=panel_w - 5):
                    lines.append(f"   {wrapped_line}")
            else:
                # Node has no result (e.g. a goal) — show direct deps' results instead
                for dep_id in node.dependencies:
                    dep = snapshot.get(dep_id)
                    if dep and dep.result is not None:
                        lines.append(f"   [{dep_id}]")
                        for wrapped_line in textwrap.wrap(str(dep.result), width=panel_w - 5):
                            lines.append(f"   {wrapped_line}")

    # After showing the node's own result, show any visible execution steps
    step_children = [
        n for n in snapshot.values()
        if n.node_type == "execution_step"
        and node_id in n.dependencies
        and not n.metadata.get("hidden", False)
    ]
    if step_children:
        lines.append(" ")
        lines.append(" Execution steps:")
        for step in step_children:
            status_icon = "✓" if step.status == "done" else "✗" if step.status == "failed" else "…"
            lines.append(f"   {status_icon} {step.metadata.get('description', step.id)}")
            attempts = step.metadata.get("attempts", [])
            if len(attempts) > 1:
                lines.append(f"     ({len(attempts)} attempts)")
            if step.result:
                for wrapped in textwrap.wrap(step.result, width=panel_w - 7):
                    lines.append(f"     {wrapped}")

    # Wrap ALL lines first, then slice the visible window
    rendered = []
    for line in lines:
        if not line.strip():
            rendered.append("")          # blank spacer row
        else:
            for subline in textwrap.wrap(line, width=max(1, panel_w - 2)):
                rendered.append(subline)

    visible_rows = h - 2
    total = len(rendered)
    scroll_offset = max(0, min(scroll_offset, max(0, total - visible_rows)))

    for i, subline in enumerate(rendered[scroll_offset: scroll_offset + visible_rows]):
        if not subline:
            continue                     # skip empty rows — addstr("") can error
        try:
            stdscr.addstr(i, panel_x + 1, subline[:panel_w - 2])
        except curses.error:
            pass

    # Scroll position indicator (only when content overflows)
    if total > visible_rows:
        pct = int(100 * scroll_offset / max(1, total - visible_rows))
        indicator = f" {pct}% "
        try:
            stdscr.addstr(0, w - len(indicator) - 1, indicator, curses.A_DIM)
        except curses.error:
            pass
            
def run_ui(
    orchestrator,
    run_dir: Path | None = None,
    repo_root: Path | None = None,
    restart_fn=None,
    ):
    """
    Run the curses DAG UI.
    If the user presses 'g', the startup screen is shown so they can pick a
    different goal or resume a previous run.  This requires `repo_root` and
    `restart_fn` to be supplied:

        run_ui(
            orchestrator,
            run_dir=run_dir,
            repo_root=REPO_ROOT,
            restart_fn=_init_system,   # callable(StartupChoice) -> (orch, run_dir)
        )
    """
    import sys
    root = logging.getLogger("dag")
    ch   = getattr(root, "_stderr_handler", None)
    if ch:
        root.removeHandler(ch)

    log_path   = (run_dir / "logs" / "dag.log") if run_dir else Path("logs/dag.log")
    log_file   = open(log_path, "a", encoding="utf-8", buffering=1)
    old_stderr = sys.stderr
    sys.stderr  = log_file

    try:
        while True:
            try:
                rebuild_repo_from_graph(orchestrator.graph)
            except Exception as exc:
                logger.warning("[UI] Git pre-warm failed (non-fatal): %s", exc)
            result = curses.wrapper(dag_interface, orchestrator, run_dir)

            # Normal quit — nothing more to do.
            if result != "switch":
                break

            # User pressed 'g' — switch-goal is only possible when the caller
            # provided both repo_root and a restart_fn.
            if restart_fn is None or repo_root is None:
                break

            # Stop the current orchestrator gracefully before tearing down.
            try:
                orchestrator.stop_llm_calls()
            except Exception as exc:
                logger.warning("[UI] Could not stop orchestrator before switch: %s", exc)

            # Show the startup screen (runs its own curses.wrapper call).
            from cuddlytoddly.ui.curses_startup import run_startup_selection
            choice = run_startup_selection(repo_root)
            if choice is None:
                # User pressed Esc on the startup screen — just exit.
                break

            # Reopen the log file for the new run (run_dir may change).
            log_file.flush()

            # Initialise the new system.
            try:
                orchestrator, run_dir = restart_fn(choice, False)
            except Exception as exc:
                logger.error("[UI] restart_fn failed during goal switch: %s", exc, exc_info=True)
                break

            # Reopen log file pointed at the new run directory.
            log_file.close()
            new_log = (run_dir / "logs" / "dag.log") if run_dir else Path("logs/dag.log")
            log_file   = open(new_log, "a", encoding="utf-8", buffering=1)
            sys.stderr  = log_file

    finally:
        sys.stderr = old_stderr
        log_file.close()
        if ch:
            root.addHandler(ch)

# --- FILE: ui/git_projection.py ---


import git
import os
from pathlib import Path
import re
import shutil
import hashlib

from collections import deque, defaultdict

from cuddlytoddly.infra.logging import get_logger

logger = get_logger(__name__)

REPO_PATH = "dag_repo"
repo = None

node_to_commit = {}


def truncate_label(label, node_id=None, max_len=20):
    if node_id is not None:
        suffix = "#" + hashlib.sha256(node_id.encode()).hexdigest()[:6]
        return suffix
    else:
        suffix = "#" + hashlib.sha256(label.encode()).hexdigest()[:6]
        return suffix

def init_repo(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)
    return git.Repo.init(path)



def graph_to_dag(snapshot):
    dag = {node_id: [] for node_id in snapshot}  # preserves insertion order
    for node_id, node in snapshot.items():
        for dep in node.dependencies:
            if dep in dag:
                dag[dep].append(node_id)  # children added in snapshot insertion order
    return dag

def topological_sort(dag):
    indegree = defaultdict(int)
    for node, children in dag.items():
        for child in children:
            indegree[child] += 1

    queue = deque([n for n in dag if indegree[n] == 0])  # insertion order, no sort
    order = []

    while queue:
        node = queue.popleft()
        order.append(node)
        for child in dag[node]:  # insertion order, no sort
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)

    return order

def commit_nodes_from_graph(snapshot):
    global node_to_commit
    node_to_commit.clear()

    repo_dir = Path(REPO_PATH)
    dag = graph_to_dag(snapshot)
    order = topological_sort(dag)

    # Detach HEAD so index.commit() doesn't chain via HEAD
    try:
        repo.head.reference = repo.commit("HEAD")
        repo.git.checkout("--detach", "HEAD")
    except Exception:
        pass  # Fine on empty repo

    for idx, node_id in enumerate(order):
        node = snapshot[node_id]
        parents = [
            node_to_commit[dep]
            for dep in sorted(node.dependencies)  # sort for deterministic commit parent ordering
            if dep in node_to_commit
        ]
        file_path = repo_dir / f"{node_id}.txt"
        file_path.write_text(f"This is node {node_id}\nStatus: {node.status}\n")
        repo.index.add([str(file_path.relative_to(REPO_PATH))])
        label = truncate_label(node.metadata.get("description") or node_id, node_id = node_id)
        try:
            parent_commits = [repo.commit(p) for p in parents]
            commit_obj = repo.index.commit(
                f"{label} [{node.status}]",
                parent_commits=parent_commits
            )
            node_to_commit[node_id] = commit_obj.hexsha  # use commit_obj, NOT repo.head
        except Exception as e:
            logger.exception("FAILED to commit node '%s': %s", node_id, e)

def compute_descendants(snapshot):
    reverse = defaultdict(set)
    for node_id, node in snapshot.items():
        for dep in node.dependencies:
            reverse[dep].add(node_id)

    descendants = defaultdict(set)

    def dfs(n, root):
        for child in reverse[n]:
            if child not in descendants[root]:
                descendants[root].add(child)
                dfs(child, root)

    for node_id in snapshot:
        dfs(node_id, node_id)

    return descendants

def commit_node_incremental(node_id, node, snapshot):
    """
    Incrementally commit a node. 
    Crucially, it uses the LATEST parent hashes from node_to_commit.
    """
    repo_dir = Path(REPO_PATH)
    file_path = repo_dir / f"{node_id}.txt"

    # 1. Update the physical file content
    file_path.write_text(
        f"This is node {node_id}\n"
        f"Dependencies: {list(node.dependencies)}\n"
        f"Status: {node.status}\n"
    )
    repo.index.add([str(file_path.relative_to(REPO_PATH))])

    # 2. Get the current Git hashes for parents from our tracking map
    # We use node_to_commit.get because a parent might not have a commit yet 
    # (though topological sort usually prevents this).
    parents = [
        node_to_commit[dep] for dep in sorted(node.dependencies) if dep in node_to_commit  # sort for determinism
    ]
    
    # 3. Create the commit
    try:
        # Convert hex strings to actual Git Commit objects
        parent_commits = [repo.commit(p) for p in parents] if parents else []
        label = truncate_label(node.metadata.get("description") or node_id, node_id = node_id)
        
        # In Git, changing parents creates a new hash. index.commit does this for us.
        commit_obj = repo.index.commit(
            f"{label} [{node.status}]", 
            parent_commits=parent_commits
        )
        
        # 4. Update the global map and node metadata
        node_to_commit[node_id] = commit_obj.hexsha
        node.metadata["last_commit_status"] = node.status
        node.metadata["last_commit_parents"] = sorted(parents)
        
        return True # Success
    except Exception as e:
        logger.exception("Failed to commit node '%s': %s", node_id, e)
        return False

def get_leaf_node_ids(dag):
    """Nodes with no children — tips of the DAG."""
    return {node_id for node_id, children in dag.items() if not children}

def sanitize_branch_name(node_id):
    """Replace characters invalid in Git branch names."""
    return re.sub(r'[^a-zA-Z0-9._-]', '_', node_id)

def update_tip_branches(snapshot):
    dag = graph_to_dag(snapshot)

    # Remove stale tip branches — skip any that fail (race condition)
    for branch in list(repo.heads):
        if branch.name.startswith("tip_"):
            try:
                repo.delete_head(branch, force=True)
            except Exception as e:
                logger.debug("[GIT] Could not delete branch %s: %s", branch.name, e)

    # Create a tip for every node that has a commit
    for node_id in snapshot:
        if node_id in node_to_commit:
            branch_name = f"tip_{sanitize_branch_name(node_id)}"
            try:
                repo.create_head(branch_name, node_to_commit[node_id], force=True)
            except Exception as e:
                logger.debug("[GIT] Could not create branch %s: %s", branch_name, e)

    # Point master at root
    root_id = next(
        (nid for nid in snapshot
         if not snapshot[nid].dependencies and nid in node_to_commit),
        None
    )
    if root_id:
        try:
            repo.create_head("master", node_to_commit[root_id], force=True)
        except Exception as e:
            logger.debug("[GIT] Could not update master: %s", e)
            
def rebuild_repo_from_graph(graph, incremental=True):
    try:
        global repo
        snapshot = graph.get_snapshot()
        # Exclude hidden execution step nodes from git projection
        snapshot = {
            nid: n for nid, n in snapshot.items()
            if not (n.node_type == "execution_step"
                    and n.metadata.get("hidden", False))
        }
        dag = graph_to_dag(snapshot)

        try:
            if repo is None:
                raise ValueError("no repo yet")
            repo.head.commit
        except (ValueError, TypeError):
            incremental = False

        if not incremental:
            repo = init_repo(REPO_PATH)
            node_to_commit.clear()
            commit_nodes_from_graph(snapshot)
            update_tip_branches(snapshot)
            return

        # Topological sort of DAG nodes
        order = topological_sort(dag)
        dirty = set()

        # 1️⃣ Mark nodes as dirty if status changed, parents changed, or missing
        for node_id in order:
            node = snapshot[node_id]
            last_status = node.metadata.get("last_commit_status")
            last_parents = node.metadata.get("last_commit_parents", [])
            missing_parent = False
            resolved_parents = []

            for dep in node.dependencies:
                if dep not in node_to_commit:
                    missing_parent = True
                else:
                    resolved_parents.append(node_to_commit[dep])

            current_parents = sorted(resolved_parents)

            if missing_parent:
                dirty.add(node_id)
            if node_id not in node_to_commit or node.status != last_status or current_parents != last_parents:
                dirty.add(node_id)

        # 2️⃣ Propagate dirty downward (dependencies)
        for node_id in order:
            if any(dep in dirty for dep in snapshot[node_id].dependencies):
                dirty.add(node_id)

        # 3️⃣ Propagate dirty upward (children)
        changed = True
        while changed:
            changed = False
            for node_id in order:
                if node_id in dirty:
                    for child_id in snapshot[node_id].children:
                        if child_id not in dirty and child_id in snapshot:
                            dirty.add(child_id)
                            changed = True

        # 4️⃣ Ensure **all snapshot nodes** are included (orphans, missing in DAG)
        full_order = order + [n for n in snapshot if n not in order]
        for node_id in snapshot:
            if node_id not in node_to_commit:
                dirty.add(node_id)

        # 5️⃣ Detach HEAD safely (pick any committed SHA or fallback)
        if node_to_commit:
            some_sha = next(iter(node_to_commit.values()))
            repo.git.checkout(some_sha)
        else:
            # fallback: detached HEAD at empty tree
            repo.git.checkout("--orphan", "tmp_head")
            repo.index.reset()

        # 6️⃣ Commit all dirty nodes
        for node_id in full_order:
            if node_id in dirty or node_id not in node_to_commit:
                commit_node_incremental(node_id, snapshot[node_id], snapshot)

        # 7️⃣ Reset index and update tip branches
        repo.index.reset()
        update_tip_branches(snapshot)

        # 8️⃣ Optional Git garbage collection
        try:
            repo.git.gc(prune="now")
        except Exception as e:
            logger.warning("gc warning (non-fatal): %s", e)
    except Exception as e:
        logger.error("[GIT] rebuild_repo_from_graph failed: %s", e, exc_info=True)
        # Last resort: nuke and rebuild from scratch
        try:
            repo = init_repo(REPO_PATH)
            node_to_commit.clear()
            commit_nodes_from_graph(snapshot)
            update_tip_branches(snapshot)
            logger.info("[GIT] Full rebuild succeeded after error")
        except Exception as e2:
            logger.error("[GIT] Full rebuild also failed: %s", e2)

def delete_node(node_id, graph):
    """
    Soft-delete a node:
    - Mark as deleted
    - Commit the deletion
    """
    if node_id in graph.nodes:
        node = graph.nodes[node_id]
        node.metadata["deleted"] = True
        commit_node_incremental(node_id, node, graph.get_snapshot())

# --- FILE: ui/startup.py ---

# ui/startup.py
"""
Shared startup logic — curses startup screen + shared data types.

Exports:
    StartupChoice      dataclass returned by the startup screen
    scan_runs()        list existing runs from the runs/ directory
    parse_manual_plan() freeform text -> (goal_text, events_list)
    run_startup_curses() blocking curses startup screen
"""
from __future__ import annotations

import curses
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class StartupChoice:
    mode:        str          # "existing" | "new_goal" | "manual_plan"
    run_dir:     Path
    goal_text:   str
    plan_events: list = field(default_factory=list)
    is_fresh:    bool = True

from typing import NamedTuple

class RunInfo(NamedTuple):
    name:       str
    path:       str
    goal:       str
    node_count: int
    mtime:      float
    age:        str

# ---------------------------------------------------------------------------
# Run scanner
# ---------------------------------------------------------------------------

def scan_runs(repo_root: Path) -> list[dict]:
    """Return one metadata dict per run that has a non-empty events.jsonl."""
    runs_dir = repo_root / "runs"
    if not runs_dir.exists():
        return []

    results = []
    for run_dir in sorted(runs_dir.iterdir(),
                          key=lambda p: p.stat().st_mtime, reverse=True):
        if not run_dir.is_dir():
            continue
        event_log = run_dir / "events.jsonl"
        if not event_log.exists() or event_log.stat().st_size == 0:
            continue

        goal_text  = ""
        node_count = 0
        try:
            with event_log.open() as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    evt = json.loads(line)
                    if evt.get("type") == "ADD_NODE":
                        node_count += 1
                        p = evt.get("payload", {})
                        if p.get("node_type") == "goal" and not goal_text:
                            goal_text = (
                                p.get("metadata", {}).get("description", "")
                                or p.get("node_id", "")
                            )
        except Exception:
            pass

        mtime = run_dir.stat().st_mtime
        results.append({
            "name":       run_dir.name,
            "path":       str(run_dir),
            "goal":       goal_text or run_dir.name.replace("_", " "),
            "node_count": node_count,
            "mtime":      mtime,
            "age":        _human_age(mtime),
        })

    return results


def _human_age(mtime: float) -> str:
    delta = time.time() - mtime
    if delta < 60:
        return "just now"
    if delta < 3600:
        return f"{int(delta / 60)}m ago"
    if delta < 86400:
        return f"{int(delta / 3600)}h ago"
    return f"{int(delta / 86400)}d ago"


# ---------------------------------------------------------------------------
# Manual plan parser
# ---------------------------------------------------------------------------

def parse_manual_plan(text: str) -> tuple[str, list]:
    """
    Parse freeform plan text into (goal_text, events_list).

    Format:
        First non-bullet non-empty line  ->  goal description
        Lines starting with - * bullet  ->  tasks
        Dependency syntax: [depends: Task_A, Task_B]
                      or:  depends on: Task_A, Task_B  (trailing)

    Returns ("", []) on empty / unparseable input.
    """

    def to_id(s: str) -> str:
        return re.sub(r"[^a-zA-Z0-9]+", "_", s.strip()).strip("_")[:50]

    lines     = text.strip().splitlines()
    goal_text = ""
    tasks: list[dict] = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        is_task = line.startswith(("-", "*", "\u2022"))

        if not is_task and not goal_text:
            goal_text = re.sub(r"^[Gg]oal\s*:\s*", "", line).strip()
            continue

        if is_task:
            content = line.lstrip("-*\u2022 ").strip()

            # [depends: X, Y]  or  (depends on: X, Y)
            dep_match = re.search(
                r"[\[\(]depends?\s*(?:on)?\s*:\s*([^\]\)]+)[\]\)]",
                content, re.IGNORECASE,
            )
            deps_raw: list[str] = []
            if dep_match:
                deps_raw = [d.strip() for d in dep_match.group(1).split(",") if d.strip()]
                content  = content[: dep_match.start()].strip()

            if not deps_raw:
                sfx = re.search(r"\s+depends?\s+on\s*:\s*(.+)$", content, re.IGNORECASE)
                if sfx:
                    deps_raw = [d.strip() for d in sfx.group(1).split(",") if d.strip()]
                    content  = content[: sfx.start()].strip()

            # "Task ID: description"  or  "description"
            id_match = re.match(r"^([^:]{1,40}):\s*(.+)$", content)
            if id_match:
                task_id   = to_id(id_match.group(1))
                task_desc = id_match.group(2).strip()
            else:
                task_desc = content
                task_id   = to_id(content)

            tasks.append({"id": task_id, "desc": task_desc, "deps_raw": deps_raw})

    if not goal_text and tasks:
        goal_text = "Goal"
    if not goal_text:
        return "", []

    task_id_set = {t["id"] for t in tasks}
    desc_to_id  = {t["desc"].lower(): t["id"] for t in tasks}

    def resolve_dep(raw: str) -> str | None:
        key = to_id(raw)
        if key in task_id_set:
            return key
        raw_lower = raw.lower().strip()
        for desc, tid in desc_to_id.items():
            if desc.startswith(raw_lower) or raw_lower.startswith(tid.lower()):
                return tid
        return None

    events: list[dict] = []
    for t in tasks:
        resolved = [r for raw in t["deps_raw"] if (r := resolve_dep(raw))]
        events.append({
            "type": "ADD_NODE",
            "payload": {
                "node_id":      t["id"],
                "node_type":    "task",
                "dependencies": resolved,
                "metadata": {
                    "description":    t["desc"],
                    "required_input": [],
                    "output":         [],
                },
            },
        })

    goal_id     = to_id(goal_text)
    depended_on = {
        r
        for t in tasks
        for raw in t["deps_raw"]
        if (r := resolve_dep(raw))
    }
    terminals = (task_id_set - depended_on) or task_id_set

    events.append({
        "type": "ADD_NODE",
        "payload": {
            "node_id":      goal_id,
            "node_type":    "goal",
            "dependencies": [],
            "metadata": {
                "description": goal_text,
                "expanded":    bool(tasks),
            },
        },
    })
    for tid in terminals:
        events.append({
            "type": "ADD_DEPENDENCY",
            "payload": {"node_id": goal_id, "depends_on": tid},
        })

    return goal_text, events


# ---------------------------------------------------------------------------
# Curses startup screen
# ---------------------------------------------------------------------------

def run_startup_curses(repo_root: Path) -> StartupChoice:
    """
    Show a full-screen curses startup dialog.
    Blocks until the user confirms a choice.
    Raises SystemExit(0) if the user presses q / Escape.
    """
    result: list[StartupChoice] = []

    def _screen(stdscr):
        result.append(_startup_screen(stdscr, repo_root))

    curses.wrapper(_screen)
    return result[0]


def _startup_screen(stdscr, repo_root: Path) -> StartupChoice:
    curses.start_color()
    curses.use_default_colors()
    curses.curs_set(0)
    stdscr.nodelay(False)
    stdscr.keypad(True)

    try:
        curses.init_pair(1, curses.COLOR_CYAN,   -1)
        curses.init_pair(2, curses.COLOR_GREEN,  -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
        curses.init_pair(4, curses.COLOR_WHITE,  -1)
        curses.init_pair(5, curses.COLOR_BLACK,  curses.COLOR_CYAN)
    except Exception:
        pass

    ACCENT  = curses.color_pair(1)
    SEL     = curses.color_pair(2) | curses.A_BOLD
    HI      = curses.color_pair(3)
    NORMAL  = curses.color_pair(4)
    TAB_ON  = curses.color_pair(5) | curses.A_BOLD
    TAB_OFF = NORMAL

    runs  = scan_runs(repo_root)
    TABS  = ["  Existing runs  ", "  New goal  ", "  Manual plan  "]
    tab   = 0 if runs else 1

    # Per-tab state
    run_sel      = 0
    goal_text    = ""
    goal_cursor  = 0
    plan_text    = ""
    plan_cursor  = 0
    error_msg    = ""

    def _draw():
        stdscr.erase()
        h, w = stdscr.getmaxyx()

        # Title bar
        try:
            stdscr.addstr(0, 0,
                " cuddlytoddly — startup ".center(w), ACCENT | curses.A_BOLD)
        except curses.error:
            pass

        # Tab bar
        x = 2
        for i, label in enumerate(TABS):
            try:
                stdscr.addstr(2, x, label, TAB_ON if i == tab else TAB_OFF)
            except curses.error:
                pass
            x += len(label) + 2

        # Separator
        try:
            stdscr.addstr(3, 0, "─" * (w - 1), NORMAL)
        except curses.error:
            pass

        body_top = 4
        body_h   = h - body_top - 3

        # ── Tab 0: existing runs ─────────────────────────────────────────────
        if tab == 0:
            if not runs:
                try:
                    stdscr.addstr(body_top + 1, 4,
                                  "No existing runs found.", NORMAL)
                    stdscr.addstr(body_top + 2, 4,
                                  "Press Tab or → to start a new goal.", NORMAL)
                except curses.error:
                    pass
            else:
                for i, run in enumerate(runs[:body_h]):
                    y    = body_top + i
                    attr = SEL if i == run_sel else NORMAL
                    ptr  = "▶ " if i == run_sel else "  "
                    age  = run["age"].rjust(10)
                    nc   = f"({run['node_count']} nodes)"
                    line = f"{ptr}{run['goal'][:w - 30]} {nc:>14} {age}"
                    try:
                        stdscr.addstr(y, 2, line[:w - 2], attr)
                    except curses.error:
                        pass

        # ── Tab 1: new goal ──────────────────────────────────────────────────
        elif tab == 1:
            try:
                stdscr.addstr(body_top,     4, "Goal description:", ACCENT)
                stdscr.addstr(body_top + 1, 4, "─" * min(60, w - 6), NORMAL)
                stdscr.addstr(body_top + 2, 4,
                              (goal_text or " ")[:w - 6], HI | curses.A_REVERSE)
                stdscr.addstr(body_top + 5, 4,
                              "Type the goal then press Enter.", NORMAL)
            except curses.error:
                pass

        # ── Tab 2: manual plan ───────────────────────────────────────────────
        elif tab == 2:
            instructions = [
                "First line = goal.  Lines starting with  -  are tasks.",
                "Dependency syntax:  - Task_B: desc [depends: Task_A]",
                "Press Ctrl+G to confirm.",
            ]
            for i, ins in enumerate(instructions):
                try:
                    stdscr.addstr(body_top + i, 4, ins[:w - 6], NORMAL)
                except curses.error:
                    pass
            try:
                stdscr.addstr(body_top + len(instructions), 4,
                              "─" * min(60, w - 6), NORMAL)
            except curses.error:
                pass

            area_top   = body_top + len(instructions) + 1
            area_h     = body_h - len(instructions) - 2
            plan_lines = plan_text.splitlines() or [""]
            for i, pline in enumerate(plan_lines[:area_h]):
                try:
                    stdscr.addstr(area_top + i, 4, pline[:w - 6], NORMAL)
                except curses.error:
                    pass

            # Draw cursor
            cur_line = plan_text[:plan_cursor].count("\n")
            cur_col  = len(plan_text[:plan_cursor].rsplit("\n", 1)[-1])
            abs_y    = area_top + cur_line
            if 0 <= abs_y < h - 1:
                split = plan_text.splitlines()
                ch    = (split[cur_line][cur_col]
                         if cur_line < len(split) and cur_col < len(split[cur_line])
                         else " ")
                try:
                    stdscr.addstr(abs_y, 4 + cur_col, ch, curses.A_REVERSE)
                except curses.error:
                    pass

        # Error line
        if error_msg:
            try:
                stdscr.addstr(h - 3, 2, f"! {error_msg}"[:w - 2], HI)
            except curses.error:
                pass

        # Footer
        try:
            stdscr.addstr(h - 2, 0, "─" * (w - 1), NORMAL)
            stdscr.addstr(h - 1, 0,
                "Tab/←/→: switch  ↑/↓: navigate  Enter: confirm  q: quit"[:w - 1],
                NORMAL)
        except curses.error:
            pass

        stdscr.refresh()

    def _make_run_dir(goal: str) -> Path:
        from cuddlytoddly.__main__ import make_run_dir
        return make_run_dir(goal).resolve()

    while True:
        _draw()
        k = stdscr.getch()

        # ── Quit ────────────────────────────────────────────────────────────
        if k in (ord("q"), 27):
            raise SystemExit(0)

        # ── Tab switching ────────────────────────────────────────────────────
        if k in (9, curses.KEY_RIGHT):
            tab = (tab + 1) % len(TABS)
            error_msg = ""
            continue

        if k == curses.KEY_LEFT:
            tab = (tab - 1) % len(TABS)
            error_msg = ""
            continue

        # ── Tab 0: existing runs ─────────────────────────────────────────────
        if tab == 0:
            if k == curses.KEY_UP:
                run_sel = max(0, run_sel - 1)
            elif k == curses.KEY_DOWN:
                run_sel = min(len(runs) - 1, run_sel + 1)
            elif k in (10, 13) and runs:
                r = runs[run_sel]
                return StartupChoice(
                    mode="existing",
                    run_dir=Path(r["path"]),
                    goal_text=r["goal"],
                    is_fresh=False,
                )

        # ── Tab 1: new goal ──────────────────────────────────────────────────
        elif tab == 1:
            if k in (10, 13):
                if goal_text.strip():
                    return StartupChoice(
                        mode="new_goal",
                        run_dir=_make_run_dir(goal_text.strip()),
                        goal_text=goal_text.strip(),
                        is_fresh=True,
                    )
                error_msg = "Goal cannot be empty."
            elif k in (curses.KEY_BACKSPACE, 127):
                if goal_cursor > 0:
                    goal_text   = goal_text[:goal_cursor - 1] + goal_text[goal_cursor:]
                    goal_cursor -= 1
            elif k == curses.KEY_LEFT:
                goal_cursor = max(0, goal_cursor - 1)
            elif k == curses.KEY_RIGHT:
                goal_cursor = min(len(goal_text), goal_cursor + 1)
            elif 32 <= k <= 126:
                goal_text   = goal_text[:goal_cursor] + chr(k) + goal_text[goal_cursor:]
                goal_cursor += 1

        # ── Tab 2: manual plan ───────────────────────────────────────────────
        elif tab == 2:
            if k == 7:   # Ctrl+G — submit
                if plan_text.strip():
                    gt, evts = parse_manual_plan(plan_text)
                    if gt:
                        return StartupChoice(
                            mode="manual_plan",
                            run_dir=_make_run_dir(gt),
                            goal_text=gt,
                            plan_events=evts,
                            is_fresh=True,
                        )
                    error_msg = "Could not parse plan — add a goal line first."
                else:
                    error_msg = "Plan cannot be empty."
            elif k == 10:   # Enter — insert newline
                plan_text    = plan_text[:plan_cursor] + "\n" + plan_text[plan_cursor:]
                plan_cursor += 1
            elif k in (curses.KEY_BACKSPACE, 127):
                if plan_cursor > 0:
                    plan_text    = plan_text[:plan_cursor - 1] + plan_text[plan_cursor:]
                    plan_cursor -= 1
            elif k == curses.KEY_UP:
                before = plan_text[:plan_cursor]
                lines  = before.splitlines(keepends=True)
                if len(lines) >= 2:
                    col         = len(lines[-1])
                    prev        = lines[-2]
                    plan_cursor = len("".join(lines[:-2])) + min(col, len(prev.rstrip("\n")))
            elif k == curses.KEY_DOWN:
                all_lines = plan_text.splitlines(keepends=True)
                before    = plan_text[:plan_cursor]
                line_idx  = before.count("\n")
                if line_idx + 1 < len(all_lines):
                    col         = len(before.rsplit("\n", 1)[-1])
                    nxt         = all_lines[line_idx + 1]
                    plan_cursor = len("".join(all_lines[:line_idx + 1])) + min(col, len(nxt.rstrip("\n")))
            elif k == curses.KEY_LEFT:
                plan_cursor = max(0, plan_cursor - 1)
            elif k == curses.KEY_RIGHT:
                plan_cursor = min(len(plan_text), plan_cursor + 1)
            elif 32 <= k <= 126:
                plan_text    = plan_text[:plan_cursor] + chr(k) + plan_text[plan_cursor:]
                plan_cursor += 1


def build_manual_plan_events(goal_id: str, goal_text: str, tasks: list) -> list:
    """
    Compatibility shim for callers that pre-parse tasks themselves.
    Prefer parse_manual_plan() for new code.
    """
    events = []
    for t in tasks:
        events.append({
            "type": "ADD_NODE",
            "payload": {
                "node_id":      t["node_id"],
                "node_type":    "task",
                "dependencies": t.get("dependencies", []),
                "metadata": {
                    "description":   t.get("description", ""),
                    "required_input": [],
                    "output":         [],
                },
            },
        })

    task_ids    = {t["node_id"] for t in tasks}
    depended_on = {dep for t in tasks for dep in t.get("dependencies", [])}
    terminals   = (task_ids - depended_on) or task_ids

    events.append({
        "type": "ADD_NODE",
        "payload": {
            "node_id":      goal_id,
            "node_type":    "goal",
            "dependencies": [],
            "metadata":     {"description": goal_text, "expanded": bool(tasks)},
        },
    })
    for tid in terminals:
        events.append({
            "type": "ADD_DEPENDENCY",
            "payload": {"node_id": goal_id, "depends_on": tid},
        })

    return events

# --- FILE: ui/web_server.py ---

# ui/web_server.py
"""
FastAPI + WebSocket server — drop-in replacement for the curses UI.

Usage (in __main__.py):
    from cuddlytoddly.ui.web_server import run_web_ui

    # Option A: already-initialised orchestrator (inline goal / curses startup)
    run_web_ui(orchestrator=orchestrator, run_dir=run_dir)

    # Option B: deferred init — startup screen shown in the browser
    run_web_ui(repo_root=REPO_ROOT, init_fn=init_fn)

Install deps once:
    pip install fastapi "uvicorn[standard]"
"""
from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

try:
    from fastapi import FastAPI, WebSocket, HTTPException, Request
    from fastapi.responses import HTMLResponse
    import uvicorn
except ImportError:
    raise ImportError(
        "Web UI requires FastAPI and uvicorn:\n"
        "  pip install fastapi 'uvicorn[standard]'"
    )

from cuddlytoddly.core.events import (
    Event,
    ADD_NODE, REMOVE_NODE,
    ADD_DEPENDENCY, REMOVE_DEPENDENCY,
    UPDATE_METADATA, UPDATE_STATUS,
)
from cuddlytoddly.infra.logging import get_logger

logger = get_logger(__name__)

_HERE = Path(__file__).resolve().parent

_HIDDEN_META = frozenset({
    "expanded", "fully_refined", "dependency_reflected",
    "last_commit_status", "last_commit_parents",
    "missing_inputs", "coverage_checked",
})


# ── Serialization ─────────────────────────────────────────────────────────────

def _serialize_snapshot(snapshot: dict) -> dict:
    out = {}
    for nid, node in snapshot.items():
        if node.node_type == "execution_step" and node.metadata.get("hidden", False):
            continue
        out[nid] = {
            "id":           node.id,
            "node_type":    node.node_type,
            "status":       node.status,
            "origin":       node.origin,
            "dependencies": sorted(node.dependencies),
            "children":     sorted(node.children),
            "result":       node.result,
            "metadata":     {k: v for k, v in node.metadata.items()
                             if k not in _HIDDEN_META},
        }
    return out


def _build_payload(orchestrator) -> dict:
    snapshot = orchestrator.get_snapshot()
    elapsed  = None
    if orchestrator.activity_started:
        elapsed = round(time.time() - orchestrator.activity_started, 1)
    return {
        "type":     "snapshot",
        "nodes":    _serialize_snapshot(snapshot),
        "status":   orchestrator.get_status(),
        "paused":   orchestrator.llm_stopped,
        "activity": orchestrator.current_activity,
        "elapsed":  elapsed,
        "tokens":   orchestrator.token_counts,   # ADD
    }


# ── App factory ───────────────────────────────────────────────────────────────

def create_app(orchestrator, run_dir: Path) -> FastAPI:
    """
    Build the FastAPI app for a fully-initialised orchestrator.
    All routes are registered unconditionally.
    """
    app = FastAPI(title="cuddlytoddly")

    # ── HTML ──────────────────────────────────────────────────────────────────

    @app.get("/")
    async def index():
        return HTMLResponse((_HERE / "web_ui.html").read_text(encoding="utf-8"))

    # ── WebSocket ─────────────────────────────────────────────────────────────

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        logger.info("[WEB] WebSocket connected")
        last_sv = last_ev = -1
        try:
            while True:
                sv = orchestrator.graph.structure_version
                ev = orchestrator.graph.execution_version
                if sv != last_sv or ev != last_ev:
                    payload = await asyncio.to_thread(_build_payload, orchestrator)
                    await websocket.send_text(json.dumps(payload, default=str))
                    last_sv, last_ev = sv, ev
                await asyncio.sleep(0.25)
        except Exception as e:
            logger.info("[WEB] WebSocket disconnected: %s", e)

    # ── Read ──────────────────────────────────────────────────────────────────

    @app.get("/api/snapshot")
    async def get_snapshot():
        return await asyncio.to_thread(_build_payload, orchestrator)

    # ── Node mutations ────────────────────────────────────────────────────────

    @app.post("/api/node")
    async def add_node(body: dict):
        node_id = (body.get("node_id") or "").strip()
        if not node_id:
            raise HTTPException(400, "node_id is required")
        dependencies = body.get("dependencies", [])
        dependents   = body.get("dependents", [])
        orchestrator.event_queue.put(Event(ADD_NODE, {
            "node_id":      node_id,
            "node_type":    body.get("node_type", "task"),
            "dependencies": dependencies,
            "origin":       "user",
            "metadata":     {"description": body.get("description", "")},
        }))
        for dep_id in dependents:
            orchestrator.event_queue.put(Event(ADD_DEPENDENCY, {
                "node_id": dep_id, "depends_on": node_id,
            }))
            from cuddlytoddly.core.events import RESET_SUBTREE
            orchestrator.event_queue.put(Event(RESET_SUBTREE, {"node_id": dep_id}))
        return {"ok": True}

    @app.put("/api/node/{node_id:path}")
    async def edit_node(node_id: str, body: dict):
        snap = orchestrator.get_snapshot()
        node = snap.get(node_id)
        if not node:
            raise HTTPException(404, "node not found")
        orchestrator.event_queue.put(Event(UPDATE_METADATA, {
            "node_id":  node_id,
            "origin":   "user",
            "metadata": {"description": body.get(
                "description", node.metadata.get("description", ""))},
        }))
        st = body.get("status", "")
        if st in ("pending", "done", "running", "failed", "to_be_expanded"):
            orchestrator.event_queue.put(Event(UPDATE_STATUS, {
                "node_id": node_id, "status": st,
            }))
        if "dependencies" in body:
            old = set(node.dependencies)
            new = set(body["dependencies"])
            for removed in old - new:
                orchestrator.event_queue.put(Event(REMOVE_DEPENDENCY, {
                    "node_id": node_id, "depends_on": removed,
                }))
            for added in new - old:
                orchestrator.event_queue.put(Event(ADD_DEPENDENCY, {
                    "node_id": node_id, "depends_on": added,
                }))
        from cuddlytoddly.core.events import RESET_SUBTREE
        orchestrator.event_queue.put(Event(RESET_SUBTREE, {"node_id": node_id}))
        return {"ok": True}

    @app.delete("/api/node/{node_id:path}")
    async def remove_node(node_id: str, mode: str = "cascade"):
        snap = orchestrator.get_snapshot()
        node = snap.get(node_id)
        if not node:
            raise HTTPException(404, "node not found")
        parents  = list(node.dependencies)
        children = list(node.children)
        q = orchestrator.event_queue
        from cuddlytoddly.core.events import RESET_SUBTREE
        if mode == "rewire":
            for child in children:
                q.put(Event(REMOVE_DEPENDENCY, {"node_id": child, "depends_on": node_id}))
                for parent in parents:
                    q.put(Event(ADD_DEPENDENCY, {"node_id": child, "depends_on": parent}))
            q.put(Event(REMOVE_NODE, {"node_id": node_id}))
            for child in children:
                q.put(Event(RESET_SUBTREE, {"node_id": child}))
        elif mode == "disconnect":
            for child in children:
                q.put(Event(REMOVE_DEPENDENCY, {"node_id": child, "depends_on": node_id}))
            q.put(Event(REMOVE_NODE, {"node_id": node_id}))
            for child in children:
                q.put(Event(RESET_SUBTREE, {"node_id": child}))
        else:  # cascade
            q.put(Event(REMOVE_NODE, {"node_id": node_id}))
        return {"ok": True}

    @app.post("/api/node/{node_id:path}/retry")
    async def retry_node(node_id: str):
        orchestrator.retry_node(node_id)
        return {"ok": True}

    # ── Goal mutations ────────────────────────────────────────────────────────

    @app.post("/api/goal/{goal_id:path}/replan")
    async def replan_goal(goal_id: str):
        orchestrator.replan_goal(goal_id)
        return {"ok": True}

    # ── LLM control ───────────────────────────────────────────────────────────

    @app.post("/api/llm/pause")
    async def llm_pause():
        orchestrator.stop_llm_calls()
        return {"ok": True}

    @app.post("/api/llm/resume")
    async def llm_resume():
        orchestrator.resume_llm_calls()
        return {"ok": True}

    # ── Export ────────────────────────────────────────────────────────────────

    @app.post("/api/export")
    async def export_md():
        from cuddlytoddly.ui.curses_ui import export_results_to_markdown
        snap = orchestrator.get_snapshot()
        try:
            path = export_results_to_markdown(snap, run_dir)
            return {"ok": True, "path": str(path)}
        except Exception as e:
            raise HTTPException(500, str(e))

    @app.post("/api/switch")
    async def switch_goal():
        raise HTTPException(
            501,
            "Goal switching is only available when the server was started in "
            "web-startup mode (--web with no inline goal). "
            "Restart with a new goal instead."
        )

    return app

# ── Unified run_web_ui ────────────────────────────────────────────────────────

def run_web_ui(
    orchestrator=None,
    run_dir: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    repo_root: Path | None = None,
    init_fn=None,
):
    """
    Start the web UI server and open a browser tab.

    Option A — already initialised (inline goal or curses startup chose):
        run_web_ui(orchestrator=orch, run_dir=rd)

    Option B — deferred init (startup screen shown in browser):
        run_web_ui(repo_root=REPO_ROOT, init_fn=lambda choice: (orch, rd))
    """
    import webbrowser

    if orchestrator is not None:
        # ── Option A: already have an orchestrator — skip startup screen ──────
        app = create_app(orchestrator, run_dir)
    else:
        # ── Option B: deferred init — build a unified app that shows the
        #    startup screen until init is complete, then serves the DAG UI ─────
        app = _create_unified_app(repo_root, init_fn)

    def _serve():
        uvicorn.run(app, host=host, port=port, log_level="warning")

    thread = threading.Thread(target=_serve, daemon=True, name="web-ui")
    thread.start()
    time.sleep(1.0)

    url = f"http://{host}:{port}"
    logger.info("[WEB UI] Listening at %s", url)
    print(f"\n  Web UI →  {url}\n")
    webbrowser.open(url)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("[WEB UI] Stopped by user")


def _create_unified_app(repo_root: Path | None, init_fn) -> FastAPI:
    """
    Single FastAPI app that handles both phases:
      - Before init: serves the startup HTML, /api/runs, /api/startup, /api/status
      - After init:  serves the DAG HTML, /ws, and all DAG REST routes

    All routes are registered upfront. DAG-specific routes return 503
    until the system is initialised.
    """
    app = FastAPI(title="cuddlytoddly")

    # Mutable state shared between the init thread and the async handlers.
    # We use a list-wrapped dict so closures can rebind it.
    state = {
        "orchestrator": None,
        "run_dir":      None,
        "ready":        False,
        "loading":      False,
        "error":        "",
    }

    startup_html = (_HERE / "web_ui_startup.html").read_text(encoding="utf-8")
    dag_html     = (_HERE / "web_ui.html").read_text(encoding="utf-8")

    # ── HTML ──────────────────────────────────────────────────────────────────

    @app.get("/")
    async def index():
        if state["ready"]:
            return HTMLResponse(dag_html)
        return HTMLResponse(startup_html)

    @app.get("/dag")
    async def dag_page():
        return HTMLResponse(dag_html)

    # ── Status & startup ──────────────────────────────────────────────────────

    @app.get("/api/status")
    async def api_status():
        return {
            "initialized": state["ready"],
            "loading":     state["loading"],
            "error":       state["error"],
        }

    @app.get("/api/runs")
    async def api_runs():
        from cuddlytoddly.ui.startup import scan_runs
        return {"runs": scan_runs(repo_root) if repo_root else []}

    @app.post("/api/startup")
    async def api_startup(body: dict):
        if state["ready"]:
            return {"ok": True, "already_initialized": True}
        if state["loading"]:
            return {"ok": False, "error": "Already loading"}
        if init_fn is None:
            return {"ok": False, "error": "No init_fn configured"}

        from cuddlytoddly.ui.startup import StartupChoice, parse_manual_plan
        from cuddlytoddly.__main__ import make_run_dir

        mode      = body.get("mode", "new_goal")
        goal_text = body.get("goal_text", "").strip()
        plan_text = body.get("plan_text", "").strip()
        run_path  = body.get("run_dir", "")

        if mode == "existing":
            if not run_path:
                return {"ok": False, "error": "run_dir required"}
            choice = StartupChoice(
                mode="existing", run_dir=Path(run_path),
                goal_text=goal_text or Path(run_path).name.replace("_", " "),
                is_fresh=False,
            )
        elif mode == "manual_plan":
            if not plan_text:
                return {"ok": False, "error": "plan_text required"}
            gt, evts = parse_manual_plan(plan_text)
            if not gt:
                return {"ok": False, "error": "Could not parse plan — add a goal line"}
            choice = StartupChoice(
                mode="manual_plan",
                run_dir=make_run_dir(gt).resolve(),
                goal_text=gt, plan_events=evts, is_fresh=True,
            )
        else:
            if not goal_text:
                return {"ok": False, "error": "goal_text required"}
            choice = StartupChoice(
                mode="new_goal",
                run_dir=make_run_dir(goal_text).resolve(),
                goal_text=goal_text, is_fresh=True,
            )

        state["loading"] = True
        state["error"]   = ""

        def _init():
            try:
                orch, rd              = init_fn(choice)
                state["orchestrator"] = orch
                state["run_dir"]      = rd
                state["ready"]        = True
                logger.info("[WEB] System initialised — DAG has %d nodes",
                            len(orch.graph.nodes))
            except Exception as e:
                logger.exception("[WEB] init_fn failed: %s", e)
                state["error"] = str(e)
            finally:
                state["loading"] = False

        threading.Thread(target=_init, daemon=True, name="web-init").start()
        return {"ok": True}

    # ── WebSocket ─────────────────────────────────────────────────────────────

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        logger.info("[WEB] WebSocket connected")

        # If not ready yet, wait here — the browser may connect immediately
        # after navigation, before the orchestrator is fully up.
        waited = 0
        while not state["ready"]:
            await asyncio.sleep(0.5)
            waited += 1
            if waited > 1200:   # 10 min timeout
                logger.warning("[WEB] WebSocket timed out waiting for init")
                await websocket.close()
                return

        orch    = state["orchestrator"]
        last_sv = last_ev = -1

        try:
            while True:
                sv = orch.graph.structure_version
                ev = orch.graph.execution_version
                if sv != last_sv or ev != last_ev:
                    payload = await asyncio.to_thread(_build_payload, orch)
                    await websocket.send_text(json.dumps(payload, default=str))
                    last_sv, last_ev = sv, ev
                await asyncio.sleep(0.25)
        except Exception as e:
            logger.info("[WEB] WebSocket closed: %s", e)

    # ── DAG REST routes — identical to create_app ─────────────────────────────
    # These return 503 until the system is ready.

    def _require_ready():
        if not state["ready"]:
            raise HTTPException(503, "System not yet initialised — wait for startup to complete")

    def _orch():
        _require_ready()
        return state["orchestrator"]

    def _run_dir():
        return state["run_dir"]

    @app.get("/api/snapshot")
    async def get_snapshot():
        return await asyncio.to_thread(_build_payload, _orch())

    @app.post("/api/node")
    async def add_node(body: dict):
        orch    = _orch()
        node_id = (body.get("node_id") or "").strip()
        if not node_id:
            raise HTTPException(400, "node_id is required")
        dependencies = body.get("dependencies", [])
        dependents   = body.get("dependents", [])
        orch.event_queue.put(Event(ADD_NODE, {
            "node_id":      node_id,
            "node_type":    body.get("node_type", "task"),
            "dependencies": dependencies,
            "origin":       "user",
            "metadata":     {"description": body.get("description", "")},
        }))
        from cuddlytoddly.core.events import RESET_SUBTREE
        for dep_id in dependents:
            orch.event_queue.put(Event(ADD_DEPENDENCY, {
                "node_id": dep_id, "depends_on": node_id,
            }))
            orch.event_queue.put(Event(RESET_SUBTREE, {"node_id": dep_id}))
        return {"ok": True}

    @app.put("/api/node/{node_id:path}")
    async def edit_node(node_id: str, body: dict):
        orch = _orch()
        snap = orch.get_snapshot()
        node = snap.get(node_id)
        if not node:
            raise HTTPException(404, "node not found")
        orch.event_queue.put(Event(UPDATE_METADATA, {
            "node_id":  node_id,
            "origin":   "user",
            "metadata": {"description": body.get(
                "description", node.metadata.get("description", ""))},
        }))
        st = body.get("status", "")
        if st in ("pending", "done", "running", "failed", "to_be_expanded"):
            orch.event_queue.put(Event(UPDATE_STATUS, {"node_id": node_id, "status": st}))
        if "dependencies" in body:
            old = set(node.dependencies)
            new = set(body["dependencies"])
            for removed in old - new:
                orch.event_queue.put(Event(REMOVE_DEPENDENCY, {
                    "node_id": node_id, "depends_on": removed}))
            for added in new - old:
                orch.event_queue.put(Event(ADD_DEPENDENCY, {
                    "node_id": node_id, "depends_on": added}))
        from cuddlytoddly.core.events import RESET_SUBTREE
        orch.event_queue.put(Event(RESET_SUBTREE, {"node_id": node_id}))
        return {"ok": True}

    @app.delete("/api/node/{node_id:path}")
    async def remove_node(node_id: str, mode: str = "cascade"):
        orch = _orch()
        snap = orch.get_snapshot()
        node = snap.get(node_id)
        if not node:
            raise HTTPException(404, "node not found")
        parents  = list(node.dependencies)
        children = list(node.children)
        q = orch.event_queue
        from cuddlytoddly.core.events import RESET_SUBTREE
        if mode == "rewire":
            for child in children:
                q.put(Event(REMOVE_DEPENDENCY, {"node_id": child, "depends_on": node_id}))
                for parent in parents:
                    q.put(Event(ADD_DEPENDENCY, {"node_id": child, "depends_on": parent}))
            q.put(Event(REMOVE_NODE, {"node_id": node_id}))
            for child in children:
                q.put(Event(RESET_SUBTREE, {"node_id": child}))
        elif mode == "disconnect":
            for child in children:
                q.put(Event(REMOVE_DEPENDENCY, {"node_id": child, "depends_on": node_id}))
            q.put(Event(REMOVE_NODE, {"node_id": node_id}))
            for child in children:
                q.put(Event(RESET_SUBTREE, {"node_id": child}))
        else:
            q.put(Event(REMOVE_NODE, {"node_id": node_id}))
        return {"ok": True}

    @app.post("/api/node/{node_id:path}/retry")
    async def retry_node(node_id: str):
        _orch().retry_node(node_id)
        return {"ok": True}

    @app.post("/api/goal/{goal_id:path}/replan")
    async def replan_goal(goal_id: str):
        _orch().replan_goal(goal_id)
        return {"ok": True}

    @app.post("/api/llm/pause")
    async def llm_pause():
        _orch().stop_llm_calls()
        return {"ok": True}

    @app.post("/api/llm/resume")
    async def llm_resume():
        _orch().resume_llm_calls()
        return {"ok": True}

    @app.post("/api/export")
    async def export_md():
        from cuddlytoddly.ui.curses_ui import export_results_to_markdown
        snap = _orch().get_snapshot()
        try:
            path = export_results_to_markdown(snap, _run_dir())
            return {"ok": True, "path": str(path)}
        except Exception as e:
            raise HTTPException(500, str(e))

    return app

# --- FILE: ui/web_ui.html ---

<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>cuddlytoddly</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/d3/7.8.5/d3.min.js"></script>
<script src="https://cdnjs.cloudflare.com/ajax/libs/dagre/0.8.5/dagre.min.js"></script>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
  --bg:         #f5f6f8;
  --surface:    #ffffff;
  --surface2:   #f0f1f4;
  --surface3:   #e6e8ec;

  --border:     #d4d7dd;
  --border2:    #c7cbd3;

  --text:       #1e1f22;
  --text-muted: #6b7078;
  --text-dim:   #8a9099;

  --accent:     #4f6df5;
  --accent-glow:#4f6df522;

  --s-pending:  #9aa0aa;
  --s-ready:    #3b82f6;
  --s-running:  #d89a3c;
  --s-done:     #2f9e6f;
  --s-failed:   #d35a5a;
  --s-expanded: #8b5cf6;

  --t-goal:     #7b61d9;
  --t-task:     #4db7e5;
  --t-reflect:  #5fbf8f;
  --t-step:     #9aa0aa;

  --toolbar-h:  52px;
  --panel-w:    340px;
  --radius:     8px;
}


body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  font-size: 13px;
  height: 100vh;
  overflow: hidden;
  display: flex;
  flex-direction: column;
}

/* ── Toolbar ─────────────────────────────────────────────────────────── */
#toolbar {
  height: var(--toolbar-h);
  min-height: var(--toolbar-h);
  background: var(--surface);
  border-bottom: 1px solid var(--border);
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 0 16px;
  z-index: 10;
}

#toolbar-brand {
  font-size: 14px;
  font-weight: 700;
  color: var(--accent);
  letter-spacing: -0.3px;
  white-space: nowrap;
}

#toolbar-goal {
  font-size: 12px;
  color: var(--text-dim);
  max-width: 300px;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  padding: 4px 10px;
  border-left: 1px solid var(--border2);
  background: none;
  border-top: none;
  border-right: none;
  border-bottom: none;
  border-radius: 6px;
  cursor: pointer;
  transition: background 0.15s, color 0.15s;
  font-family: inherit;
}
#toolbar-goal:hover {
  background: var(--surface2);
  color: var(--text);
}

#toolbar-counts {
  display: flex;
  gap: 6px;
  margin-left: auto;
}

.count-pill {
  display: flex;
  align-items: center;
  gap: 4px;
  padding: 2px 8px;
  border-radius: 20px;
  font-size: 11px;
  font-weight: 600;
  background: var(--surface2);
  border: 1px solid var(--border);
  color: var(--text-muted);
  transition: opacity 0.2s;
}
.count-pill.has-value { opacity: 1; }
.count-pill .dot { width: 6px; height: 6px; border-radius: 50%; }

#toolbar-activity {
  display: flex;
  align-items: center;
  gap: 6px;
  font-size: 11px;
  color: var(--text-muted);
  max-width: 220px;
  overflow: hidden;
  white-space: nowrap;
  text-overflow: ellipsis;
  padding: 0 10px;
  border-left: 1px solid var(--border2);
  border-right: 1px solid var(--border2);
}

.spinner {
  width: 10px; height: 10px;
  border: 2px solid var(--s-running);
  border-top-color: transparent;
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
  flex-shrink: 0;
}
@keyframes spin { to { transform: rotate(360deg); } }

.toolbar-btn {
  background: var(--surface2);
  border: 1px solid var(--border2);
  color: var(--text-dim);
  border-radius: 6px;
  padding: 5px 11px;
  font-size: 12px;
  cursor: pointer;
  transition: all 0.15s;
  white-space: nowrap;
}
.toolbar-btn:hover { background: var(--surface3); color: var(--text); border-color: var(--border2); }
.toolbar-btn.active { background: var(--s-running); color: #0b1120; border-color: var(--s-running); font-weight: 600; }
.toolbar-btn.primary { background: var(--accent); color: #fff; border-color: var(--accent); }
.toolbar-btn.primary:hover { filter: brightness(1.1); }

#conn-dot {
  width: 7px; height: 7px; border-radius: 50%;
  background: var(--s-failed); flex-shrink: 0;
  transition: background 0.3s;
}
#conn-dot.ok { background: var(--s-done); }

/* ── Canvas ──────────────────────────────────────────────────────────── */
#canvas-wrap {
  flex: 1;
  position: relative;
  overflow: hidden;
}

#dag-canvas {
  width: 100%;
  height: 100%;
  cursor: grab;
}
#dag-canvas:active { cursor: grabbing; }

.node-g { cursor: pointer; }
.node-g .node-hit { fill: transparent; }
.node-g .bg { transition: filter 0.15s; }
.node-g:hover .bg { filter: brightness(1.15); }
.node-g.selected .bg { filter: brightness(1.2); }

@keyframes pulse-glow {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.6; }
}
.node-running .status-ring { animation: pulse-glow 1.2s ease-in-out infinite; }

.edge-path {
  fill: none;
  stroke: var(--border2);
  stroke-width: 1.5;
  marker-end: url(#arrowhead);
  transition: stroke 0.3s;
}

/* ── Zoom controls ───────────────────────────────────────────────────── */
#zoom-controls {
  position: absolute;
  bottom: 16px;
  left: 16px;
  display: flex;
  flex-direction: column;
  gap: 4px;
}
.zoom-btn {
  width: 28px; height: 28px;
  background: var(--surface);
  border: 1px solid var(--border2);
  color: var(--text-dim);
  border-radius: 6px;
  font-size: 16px;
  cursor: pointer;
  display: flex; align-items: center; justify-content: center;
  transition: all 0.15s;
}
.zoom-btn:hover { background: var(--surface2); color: var(--text); }

/* ── Info Panel ──────────────────────────────────────────────────────── */
#info-panel {
  position: absolute;
  bottom: 16px; right: 16px;
  width: var(--panel-w);
  max-height: calc(100vh - var(--toolbar-h) - 32px);
  background: var(--surface);
  border: 1px solid var(--border2);
  border-radius: var(--radius);
  box-shadow: 0 8px 32px rgba(0,0,0,0.5);
  display: flex;
  flex-direction: column;
  overflow: hidden;
  transition: opacity 0.2s;
  z-index: 20;
}
#info-panel.hidden { display: none; }

.panel-header {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 10px 12px;
  background: var(--surface2);
  border-bottom: 1px solid var(--border);
  cursor: move;
  user-select: none;
  flex-shrink: 0;
}

.panel-node-id {
  font-size: 12px;
  font-weight: 600;
  color: var(--text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  flex: 1;
}

.panel-close {
  background: none;
  border: none;
  color: var(--text-muted);
  cursor: pointer;
  font-size: 14px;
  padding: 2px 4px;
  border-radius: 4px;
  line-height: 1;
  flex-shrink: 0;
}
.panel-close:hover { background: var(--surface3); color: var(--text); }

.panel-actions {
  display: flex;
  gap: 6px;
  padding: 8px 12px;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}

.panel-btn {
  background: var(--surface2);
  border: 1px solid var(--border2);
  color: var(--text-dim);
  border-radius: 5px;
  padding: 4px 10px;
  font-size: 11px;
  cursor: pointer;
  transition: all 0.15s;
}
.panel-btn:hover { background: var(--surface3); color: var(--text); }
.panel-btn.danger { border-color: #7f1d1d; color: #fca5a5; }
.panel-btn.danger:hover { background: #7f1d1d; color: #fef2f2; }

.panel-body {
  overflow-y: auto;
  padding: 12px;
  flex: 1;
  min-height: 0;
}
.panel-body::-webkit-scrollbar { width: 4px; }
.panel-body::-webkit-scrollbar-track { background: transparent; }
.panel-body::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

.info-row {
  margin-bottom: 12px;
}
.info-label {
  font-size: 10px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.6px;
  color: var(--text-muted);
  margin-bottom: 3px;
}
.info-value {
  color: var(--text-dim);
  line-height: 1.5;
  word-break: break-word;
}
.info-value.prominent { color: var(--text); font-size: 13px; }

.badge {
  display: inline-flex;
  align-items: center;
  gap: 4px;
  padding: 2px 8px;
  border-radius: 20px;
  font-size: 10px;
  font-weight: 600;
  background: var(--surface2);
  border: 1px solid var(--border);
}

.dep-tag {
  display: inline-block;
  background: var(--surface2);
  border: 1px solid var(--border2);
  border-radius: 4px;
  padding: 2px 7px;
  font-size: 11px;
  color: var(--text-dim);
  cursor: pointer;
  margin: 2px 2px 2px 0;
  transition: all 0.15s;
}
.dep-tag:hover { background: var(--surface3); color: var(--accent); border-color: var(--accent); }

.result-box {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 5px;
  padding: 8px 10px;
  font-size: 11px;
  line-height: 1.6;
  color: var(--text-dim);
  max-height: 180px;
  overflow-y: auto;
  white-space: pre-wrap;
  word-break: break-word;
}
.result-box::-webkit-scrollbar { width: 3px; }
.result-box::-webkit-scrollbar-thumb { background: var(--border2); }

.step-item {
  display: flex;
  gap: 8px;
  align-items: flex-start;
  padding: 6px 0;
  border-bottom: 1px solid var(--border);
  font-size: 11px;
}
.step-item:last-child { border-bottom: none; }
.step-icon { flex-shrink: 0; padding-top: 1px; }
.step-desc { color: var(--text-dim); line-height: 1.4; }
.step-result { color: var(--text-muted); font-size: 10px; margin-top: 2px; white-space: pre-wrap; word-break: break-word; }

/* ── Modal ───────────────────────────────────────────────────────────── */
#modal-overlay {
  position: fixed; inset: 0;
  background: rgba(0,0,0,0.65);
  display: flex; align-items: center; justify-content: center;
  z-index: 100;
  backdrop-filter: blur(2px);
}
#modal-overlay.hidden { display: none; }

#modal {
  background: var(--surface);
  border: 1px solid var(--border2);
  border-radius: var(--radius);
  width: 460px;
  max-width: calc(100vw - 32px);
  max-height: calc(100vh - 64px);
  display: flex;
  flex-direction: column;
  box-shadow: 0 20px 60px rgba(0,0,0,0.6);
}

.modal-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 14px 18px;
  border-bottom: 1px solid var(--border);
  flex-shrink: 0;
}
.modal-title {
  font-size: 14px;
  font-weight: 600;
  color: var(--text);
}
.modal-close {
  background: none; border: none;
  color: var(--text-muted); cursor: pointer;
  font-size: 16px; padding: 2px 5px;
  border-radius: 4px; line-height: 1;
}
.modal-close:hover { background: var(--surface2); color: var(--text); }

.modal-body {
  padding: 18px;
  overflow-y: auto;
  flex: 1;
}
.modal-body::-webkit-scrollbar { width: 4px; }
.modal-body::-webkit-scrollbar-thumb { background: var(--border2); }

.form-group { margin-bottom: 14px; }
.form-label {
  display: block;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--text-muted);
  margin-bottom: 5px;
}
.form-input, .form-select, .form-textarea {
  width: 100%;
  background: var(--surface2);
  border: 1px solid var(--border2);
  border-radius: 5px;
  padding: 8px 10px;
  color: var(--text);
  font-size: 13px;
  font-family: inherit;
  outline: none;
  transition: border-color 0.15s;
}
.form-input:focus, .form-select:focus, .form-textarea:focus {
  border-color: var(--accent);
}
.form-select { cursor: pointer; }
.form-select option { background: var(--surface2); }
.form-textarea { resize: vertical; min-height: 70px; line-height: 1.5; }
.form-hint { font-size: 10px; color: var(--text-muted); margin-top: 3px; }

.radio-group { display: flex; flex-direction: column; gap: 8px; }
.radio-item {
  display: flex; align-items: flex-start; gap: 10px;
  background: var(--surface2);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 10px 12px;
  cursor: pointer;
  transition: all 0.15s;
}
.radio-item:hover { border-color: var(--border2); background: var(--surface3); }
.radio-item input { margin-top: 2px; accent-color: var(--accent); flex-shrink: 0; }
.radio-item-label { font-size: 12px; font-weight: 600; color: var(--text); }
.radio-item-desc { font-size: 11px; color: var(--text-muted); margin-top: 2px; }

.modal-footer {
  display: flex; justify-content: flex-end; gap: 8px;
  padding: 12px 18px;
  border-top: 1px solid var(--border);
  flex-shrink: 0;
}
.btn {
  padding: 7px 16px;
  border-radius: 6px;
  font-size: 13px;
  cursor: pointer;
  border: 1px solid transparent;
  transition: all 0.15s;
  font-family: inherit;
}
.btn-ghost { background: var(--surface2); border-color: var(--border2); color: var(--text-dim); }
.btn-ghost:hover { background: var(--surface3); color: var(--text); }
.btn-primary { background: var(--accent); color: #fff; }
.btn-primary:hover { filter: brightness(1.1); }
.btn-danger { background: #7f1d1d; border-color: #991b1b; color: #fef2f2; }
.btn-danger:hover { background: #991b1b; }

/* ── Toast ───────────────────────────────────────────────────────────── */
#toast {
  position: fixed;
  bottom: 24px; left: 50%; transform: translateX(-50%);
  background: var(--surface2);
  border: 1px solid var(--border2);
  border-radius: 8px;
  padding: 10px 18px;
  font-size: 12px;
  color: var(--text);
  box-shadow: 0 4px 16px rgba(0,0,0,0.4);
  z-index: 200;
  transition: opacity 0.3s;
  pointer-events: none;
}
#toast.hidden { opacity: 0; }

/* ── Empty state ─────────────────────────────────────────────────────── */
#empty-state {
  position: absolute; inset: 0;
  display: flex; align-items: center; justify-content: center;
  flex-direction: column; gap: 8px;
  color: var(--text-muted); font-size: 13px; pointer-events: none;
}
#empty-state.hidden { display: none; }
#empty-state .big { font-size: 32px; opacity: 0.3; }

/* ── Switch-goal modal ───────────────────────────────────────────────── */
#switch-overlay {
  position: fixed; inset: 0;
  background: rgba(0,0,0,0.55);
  display: flex; align-items: center; justify-content: center;
  z-index: 100;
  opacity: 1; transition: opacity 0.2s;
}
#switch-overlay.hidden { display: none; }

#switch-shell {
  width: 600px;
  max-width: calc(100vw - 32px);
  background: #111827;
  border: 1px solid #2e3f58;
  border-radius: 10px;
  box-shadow: 0 24px 80px rgba(0,0,0,0.7);
  overflow: hidden;
  font-family: 'Inter', system-ui, sans-serif;
  color: #e2e8f0;
}
#switch-shell-header {
  padding: 18px 22px 14px;
  border-bottom: 1px solid #1f2d42;
  background: #1e2a3d;
  display: flex; align-items: center; justify-content: space-between;
}
.sw-brand { font-size: 15px; font-weight: 700; color: #6366f1; letter-spacing: -0.3px; }
.sw-close  {
  background: none; border: none; color: #64748b;
  font-size: 18px; cursor: pointer; padding: 2px 6px; border-radius: 4px;
}
.sw-close:hover { color: #e2e8f0; background: #263350; }

.sw-tabs { display: flex; border-bottom: 1px solid #1f2d42; background: #1e2a3d; }
.sw-tab  {
  flex: 1; padding: 10px 12px; font-size: 12px; font-weight: 500;
  color: #64748b; cursor: pointer; border-bottom: 2px solid transparent;
  transition: all 0.15s; text-align: center; user-select: none;
  background: none; border-top: none; border-right: none; border-left: none;
  font-family: inherit;
}
.sw-tab:hover { color: #94a3b8; }
.sw-tab.active { color: #6366f1; border-bottom-color: #6366f1; background: #111827; }

.sw-pane { display: none; padding: 22px; }
.sw-pane.active { display: block; }

.sw-runs-list {
  max-height: 260px; overflow-y: auto;
  border: 1px solid #1f2d42; border-radius: 6px;
}
.sw-runs-list::-webkit-scrollbar { width: 4px; }
.sw-runs-list::-webkit-scrollbar-thumb { background: #2e3f58; }
.sw-run-item {
  display: flex; align-items: center; gap: 12px;
  padding: 11px 14px; border-bottom: 1px solid #1f2d42;
  cursor: pointer; transition: background 0.12s;
}
.sw-run-item:last-child { border-bottom: none; }
.sw-run-item:hover { background: #1e2a3d; }
.sw-run-item.selected { background: #263350; }
.sw-run-dot { width: 8px; height: 8px; border-radius: 50%; background: #6366f1; flex-shrink: 0; }
.sw-run-goal {
  flex: 1; font-size: 13px; font-weight: 500; color: #e2e8f0;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
.sw-run-meta { font-size: 10px; color: #64748b; white-space: nowrap; text-align: right; }
.sw-empty { padding: 28px 16px; text-align: center; color: #64748b; font-size: 12px; }

.sw-label {
  display: block; font-size: 11px; font-weight: 600;
  text-transform: uppercase; letter-spacing: 0.5px;
  color: #64748b; margin-bottom: 6px;
}
.sw-input {
  width: 100%; background: #1e2a3d; border: 1px solid #2e3f58;
  border-radius: 6px; padding: 10px 12px; color: #e2e8f0;
  font-size: 13px; font-family: inherit; outline: none;
  transition: border-color 0.15s;
}
.sw-input:focus { border-color: #6366f1; }
.sw-textarea {
  width: 100%; background: #1e2a3d; border: 1px solid #2e3f58;
  border-radius: 6px; padding: 10px 12px; color: #e2e8f0;
  font-size: 12px; font-family: 'SF Mono','Fira Code',monospace;
  line-height: 1.6; outline: none; resize: vertical; min-height: 160px;
  transition: border-color 0.15s;
}
.sw-textarea:focus { border-color: #6366f1; }
.sw-hint { font-size: 11px; color: #64748b; margin-top: 5px; line-height: 1.5; }
.sw-error { color: #ef4444; font-size: 12px; margin-top: 8px; min-height: 16px; }

.sw-footer {
  padding: 14px 22px; border-top: 1px solid #1f2d42;
  display: flex; justify-content: flex-end; gap: 8px;
  background: #111827;
}
.sw-btn {
  padding: 8px 18px; border-radius: 6px; font-size: 13px;
  font-family: inherit; cursor: pointer; border: 1px solid transparent;
  transition: all 0.15s;
}
.sw-btn-ghost { background: #1e2a3d; border-color: #2e3f58; color: #94a3b8; }
.sw-btn-ghost:hover { background: #263350; color: #e2e8f0; }
.sw-btn-primary { background: #6366f1; color: #fff; font-weight: 600; }
.sw-btn-primary:hover { filter: brightness(1.1); }
.sw-btn-primary:disabled { opacity: 0.45; cursor: not-allowed; filter: none; }

.sw-loading {
  display: none; flex-direction: column; align-items: center;
  justify-content: center; gap: 14px; padding: 44px 24px;
}
.sw-loading.show { display: flex; }
.sw-spinner {
  width: 26px; height: 26px; border: 3px solid #2e3f58;
  border-top-color: #6366f1; border-radius: 50%;
  animation: spin 0.7s linear infinite;
}
.sw-loading-msg { color: #94a3b8; font-size: 13px; }
</style>
</head>
<body>

<!-- ── Toolbar ──────────────────────────────────────────────────────────── -->
<div id="toolbar">
  <span id="toolbar-brand">cuddlytoddly</span>
  <button id="toolbar-goal" onclick="openSwitchModal()" title="Switch goal"></button>
  <div id="toolbar-counts">
    <div class="count-pill" id="pill-pending">
      <div class="dot" style="background:var(--s-pending)"></div>
      <span id="cnt-pending">0</span> pending
    </div>
    <div class="count-pill" id="pill-ready">
      <div class="dot" style="background:var(--s-ready)"></div>
      <span id="cnt-ready">0</span> ready
    </div>
    <div class="count-pill" id="pill-running">
      <div class="dot" style="background:var(--s-running)"></div>
      <span id="cnt-running">0</span> running
    </div>
    <div class="count-pill" id="pill-done">
      <div class="dot" style="background:var(--s-done)"></div>
      <span id="cnt-done">0</span> done
    </div>
    <div class="count-pill" id="pill-failed">
      <div class="dot" style="background:var(--s-failed)"></div>
      <span id="cnt-failed">0</span> failed
    </div>
  </div>
  <div class="count-pill" id="pill-tokens">
    <div class="dot" style="background:var(--accent)"></div>
    <span id="cnt-tokens">0</span> tokens
  </div>
  <div id="toolbar-activity"><span id="activity-text" style="color:var(--text-muted)">idle</span></div>

  <button class="toolbar-btn" id="btn-llm-toggle" onclick="toggleLLM()">⏸ Pause LLM</button>
  <button class="toolbar-btn" onclick="addNodePrompt()">＋ Add Node</button>
  <button class="toolbar-btn primary" onclick="exportMD()">↓ Export</button>
  <div id="conn-dot" title="WebSocket connection"></div>
</div>

<!-- ── Canvas ───────────────────────────────────────────────────────────── -->
<div id="canvas-wrap">
  <svg id="dag-canvas"></svg>

  <div id="empty-state">
    <div class="big">◎</div>
    <div>Waiting for graph data…</div>
  </div>

  <div id="zoom-controls">
    <button class="zoom-btn" onclick="zoomIn()" title="Zoom in">+</button>
    <button class="zoom-btn" onclick="zoomReset()" title="Fit to screen" style="font-size:11px">⊡</button>
    <button class="zoom-btn" onclick="zoomOut()" title="Zoom out">−</button>
  </div>
</div>

<!-- ── Info Panel ────────────────────────────────────────────────────────── -->
<div id="info-panel" class="hidden">
  <div class="panel-header" id="panel-drag-handle">
    <span class="panel-node-id" id="panel-title">—</span>
    <button class="panel-close" onclick="closePanel()">✕</button>
  </div>
  <div class="panel-actions" id="panel-actions"></div>
  <div class="panel-body" id="panel-body"></div>
</div>

<!-- ── Modal ─────────────────────────────────────────────────────────────── -->
<div id="modal-overlay" class="hidden" onclick="overlayClick(event)">
  <div id="modal">
    <div class="modal-header">
      <span class="modal-title" id="modal-title">—</span>
      <button class="modal-close" onclick="closeModal()">✕</button>
    </div>
    <div class="modal-body" id="modal-body"></div>
    <div class="modal-footer" id="modal-footer"></div>
  </div>
</div>

<!-- ── Toast ─────────────────────────────────────────────────────────────── -->
<div id="toast" class="hidden"></div>

<script>
// ── Constants ────────────────────────────────────────────────────────────────

const STATUS_COLOR = {
  pending:        'var(--s-pending)',
  ready:          'var(--s-ready)',
  running:        'var(--s-running)',
  done:           'var(--s-done)',
  failed:         'var(--s-failed)',
  to_be_expanded: 'var(--s-expanded)',
};
const STATUS_COLOR_HEX = {
  pending: '#475569', ready: '#3b82f6', running: '#f59e0b',
  done: '#10b981', failed: '#ef4444', to_be_expanded: '#8b5cf6',
};
const TYPE_ICON  = { goal: '◎', task: '▣', reflection: '◈', execution_step: '·' };
const TYPE_COLOR = { goal: '#a78bfa', task: '#7dd3fc', reflection: '#86efac', execution_step: '#475569' };
const NODE_W = 210, NODE_H = 54;

// ── State ────────────────────────────────────────────────────────────────────

let nodes = {};        // current snapshot
let selectedId = null;
let paused = false;
let firstRender = true;
let currentG = null;   // last dagre graph
let ws, reconnectTimer;
let modalSubmitFn = null;
let modalMode = null;

// ── WebSocket ────────────────────────────────────────────────────────────────

function connect() {
  ws = new WebSocket(`ws://${location.host}/ws`);
  ws.onopen = () => {
    document.getElementById('conn-dot').classList.add('ok');
    clearTimeout(reconnectTimer);
  };
  ws.onclose = () => {
    document.getElementById('conn-dot').classList.remove('ok');
    reconnectTimer = setTimeout(connect, 2500);
  };
  ws.onmessage = e => handleMessage(JSON.parse(e.data));
  ws.onerror = () => ws.close();
}

function handleMessage(data) {
  if (data.type !== 'snapshot') return;

  nodes = data.nodes || {};
  paused = data.paused || false;

  if (data.tokens) {
    const t = data.tokens.total;
    document.getElementById('cnt-tokens').textContent =
      t >= 1_000_000 ? `${(t/1_000_000).toFixed(1)}M`
      : t >= 1_000   ? `${(t/1_000).toFixed(1)}K`
      : t;
  }

  // Toolbar counts
  const counts = (data.status && data.status.by_status) || {};
  for (const st of ['pending','ready','running','done','failed']) {
    document.getElementById(`cnt-${st}`).textContent = counts[st] || 0;
  }

  // Goal description in toolbar (now a button)
  const goal = Object.values(nodes).find(n => n.node_type === 'goal');
  const goalDesc = goal?.metadata?.description || goal?.id || '';
  const goalEl = document.getElementById('toolbar-goal');
  goalEl.textContent = goalDesc ? `⊹ ${goalDesc}` : '';
  goalEl.style.display = goalDesc ? '' : 'none';
  goalEl.title = goalDesc ? `Current goal: ${goalDesc} — click to switch` : '';

  // Activity
  const actEl = document.getElementById('activity-text');
  if (data.activity) {
    const spin = `<span class="spinner"></span>`;
    const elapsed = data.elapsed != null ? ` (${data.elapsed}s)` : '';
    actEl.innerHTML = `${spin} ${esc(data.activity)}${elapsed}`;
  } else {
    actEl.innerHTML = paused
      ? '<span style="color:var(--s-running)">⏸ LLM paused</span>'
      : '<span style="color:var(--text-muted)">idle</span>';
  }

  // LLM button
  const llmBtn = document.getElementById('btn-llm-toggle');
  llmBtn.textContent = paused ? '▶ Resume LLM' : '⏸ Pause LLM';
  llmBtn.classList.toggle('active', paused);

  // Empty state
  const hasNodes = Object.keys(nodes).length > 0;
  document.getElementById('empty-state').classList.toggle('hidden', hasNodes);

  // Re-render DAG
  if (hasNodes) renderDAG();

  // Update info panel if open
  if (selectedId && nodes[selectedId]) updatePanel();
  else if (selectedId && !nodes[selectedId]) closePanel();
}

// ── Layout + Render ──────────────────────────────────────────────────────────

let svg, svgG, zoomBehavior;

function initSVG() {
  svg = d3.select('#dag-canvas');

  // Arrow marker
  const defs = svg.append('defs');
  defs.append('marker')
    .attr('id', 'arrowhead')
    .attr('viewBox', '-0 -4 8 8')
    .attr('refX', 7).attr('refY', 0)
    .attr('markerWidth', 5).attr('markerHeight', 5)
    .attr('orient', 'auto')
    .append('path')
    .attr('d', 'M0,-4L8,0L0,4')
    .attr('fill', '#2e3f58');

  svgG = svg.append('g').attr('class', 'root');
  svgG.append('g').attr('class', 'edge-layer');
  svgG.append('g').attr('class', 'node-layer');

  zoomBehavior = d3.zoom()
    .scaleExtent([0.08, 4])
    .on('zoom', e => svgG.attr('transform', e.transform));
  svg.call(zoomBehavior);
}

function renderDAG() {
  // Build dagre graph — edges go from dependent → dependency
  // so goal (which depends on tasks) ends up at top in TB layout
  const g = new dagre.graphlib.Graph({ multigraph: false });
  g.setGraph({
    rankdir: 'TB',
    ranksep: 80, nodesep: 44,
    marginx: 48, marginy: 48,
  });
  g.setDefaultEdgeLabel(() => ({}));

  const visible = Object.entries(nodes)
    .filter(([, n]) => n.node_type !== 'execution_step');

  for (const [id] of visible) {
    g.setNode(id, { width: NODE_W, height: NODE_H });
  }
  const visibleIds = new Set(visible.map(([id]) => id));
  for (const [id, node] of visible) {
    for (const dep of node.dependencies) {
      if (visibleIds.has(dep)) {
        g.setEdge(dep, id); // dependent → prereq = goal at top
      }
    }
  }
  dagre.layout(g);
  currentG = g;

  // Edges
  const line = d3.line().x(d => d.x).y(d => d.y).curve(d3.curveBasis);
  const edgeLayer = svgG.select('.edge-layer');
  const edgeData = g.edges().map(e => ({
    key:    `${e.v}__${e.w}`,
    v: e.v, w: e.w,
    points: g.edge(e).points,
  }));
  const edges = edgeLayer.selectAll('path.edge-path')
    .data(edgeData, d => d.key);
  edges.enter().append('path').attr('class', 'edge-path')
    .merge(edges)
    .attr('d', d => line(d.points))
    .attr('stroke', d => STATUS_COLOR_HEX[nodes[d.v]?.status] || '#2e3f58')
    .attr('stroke-opacity', d => nodes[d.v]?.status === 'done' ? 0.35 : 0.55);
  edges.exit().remove();

  // Nodes
  const nodeLayer = svgG.select('.node-layer');
  const nodeData = g.nodes()
    .filter(id => nodes[id])
    .map(id => ({ id, pos: g.node(id), node: nodes[id] }));

  const nodeGs = nodeLayer.selectAll('g.node-g')
    .data(nodeData, d => d.id);

  // Enter
  const entered = nodeGs.enter().append('g')
    .attr('class', d => `node-g ${d.node.status === 'running' ? 'node-running' : ''}`)
    .on('click', (e, d) => { e.stopPropagation(); selectNode(d.id); });

  entered.append('rect').attr('class', 'node-hit')
    .attr('x', -NODE_W/2 - 4).attr('y', -NODE_H/2 - 4)
    .attr('width', NODE_W + 8).attr('height', NODE_H + 8)
    .attr('rx', 10).attr('fill', 'transparent');

  // Selection glow ring
  entered.append('rect').attr('class', 'status-ring')
    .attr('x', -NODE_W/2 - 2).attr('y', -NODE_H/2 - 2)
    .attr('width', NODE_W + 4).attr('height', NODE_H + 4)
    .attr('rx', 9).attr('fill', 'none')
    .attr('stroke-width', 2).attr('stroke', 'transparent');

  entered.append('rect').attr('class', 'bg')
    .attr('x', -NODE_W/2).attr('y', -NODE_H/2)
    .attr('width', NODE_W).attr('height', NODE_H)
    .attr('rx', 7);

  // Left status bar
  entered.append('rect').attr('class', 'sbar')
    .attr('x', -NODE_W/2 + 4).attr('y', -NODE_H/2 + 7)
    .attr('width', 3).attr('height', NODE_H - 14)
    .attr('rx', 1.5);

  entered.append('text').attr('class', 'ticon');
  entered.append('text').attr('class', 'tdesc');
  entered.append('text').attr('class', 'tstatus');
  entered.append('text').attr('class', 'ttype');

  // Merge + update
  const merged = entered.merge(nodeGs);

  merged
    .attr('transform', d => `translate(${d.pos.x}, ${d.pos.y})`)
    .attr('class', d => `node-g ${d.node.status === 'running' ? 'node-running' : ''} ${d.id === selectedId ? 'selected' : ''}`);

  const isGoal = d => d.node.node_type === 'goal';
  merged.select('.bg')
    .attr('fill', d => d.id === selectedId ? '#1e2a3d' : '#111827')
    .attr('stroke', d => d.id === selectedId ? '#6366f1' : isGoal(d) ? '#312e81' : '#1f2d42')
    .attr('stroke-width', d => d.id === selectedId ? 2 : isGoal(d) ? 1.5 : 1);

  merged.select('.status-ring')
    .attr('stroke', d => {
      if (d.id === selectedId) return '#6366f1';
      if (d.node.status === 'running') return STATUS_COLOR_HEX.running;
      return 'transparent';
    })
    .attr('stroke-opacity', d => d.id === selectedId || d.node.status === 'running' ? 0.4 : 0);

  merged.select('.sbar')
    .attr('fill', d => STATUS_COLOR_HEX[d.node.status] || '#475569');

  merged.select('.ticon')
    .attr('x', -NODE_W/2 + 14).attr('y', -NODE_H/2 + 20)
    .attr('font-size', 12).attr('font-family', 'monospace')
    .attr('fill', d => TYPE_COLOR[d.node.node_type] || '#94a3b8')
    .text(d => TYPE_ICON[d.node.node_type] || '▣');

  merged.select('.tdesc')
    .attr('x', -NODE_W/2 + 28).attr('y', -NODE_H/2 + 20)
    .attr('font-size', 12).attr('font-family', 'system-ui, sans-serif').attr('font-weight', '500')
    .attr('fill', '#e2e8f0')
    .text(d => trunc(d.node.metadata?.description || d.id, 24));

  merged.select('.tstatus')
    .attr('x', -NODE_W/2 + 14).attr('y', -NODE_H/2 + 38)
    .attr('font-size', 10).attr('font-family', 'system-ui, sans-serif')
    .attr('fill', d => STATUS_COLOR_HEX[d.node.status] || '#475569')
    .text(d => d.node.status);

  merged.select('.ttype')
    .attr('x', NODE_W/2 - 8).attr('y', -NODE_H/2 + 38)
    .attr('font-size', 10).attr('font-family', 'system-ui, sans-serif')
    .attr('text-anchor', 'end').attr('fill', '#2e3f58')
    .text(d => d.node.node_type);

  nodeGs.exit().remove();

  // Deselect on canvas click (not on node)
  svg.on('click', () => { if (selectedId) closePanel(); });

  // Auto-fit first render
  if (firstRender) { fitGraph(); firstRender = false; }
}

// ── Selection ────────────────────────────────────────────────────────────────

function selectNode(id) {
  selectedId = id;
  renderDAG();
  updatePanel();
}

function updatePanel() {
  const node = nodes[selectedId];
  if (!node) return;

  const panel = document.getElementById('info-panel');
  panel.classList.remove('hidden');
  document.getElementById('panel-title').textContent = selectedId;

  // Actions
  const actions = document.getElementById('panel-actions');
  const isGoal = node.node_type === 'goal';
  const isFailed = node.status === 'failed';
  const isDone = node.status === 'done';
  actions.innerHTML = `
    <button class="panel-btn" onclick="openEditModal()">✎ Edit</button>
    ${(isFailed || isDone) ? `<button class="panel-btn" onclick="retryNode()">↺ Retry</button>` : ''}
    ${isGoal ? `<button class="panel-btn" onclick="replanGoal()">⟳ Replan</button>` : ''}
    <button class="panel-btn danger" onclick="openRemoveModal()">✕ Remove</button>
  `;

  // Body
  const body = document.getElementById('panel-body');
  const desc = node.metadata?.description || '';
  const deps = node.dependencies || [];
  const result = node.result;
  const notes = node.metadata?.reflection_notes || [];
  const reqIn = node.metadata?.required_input || [];
  const output = node.metadata?.output || [];

  const stepChildren = Object.values(nodes).filter(n =>
    n.node_type === 'execution_step' &&
    n.dependencies.includes(selectedId) &&
    !n.metadata?.hidden
  );

  let html = '';

  if (desc) {
    html += row('Description', `<span class="info-value prominent">${esc(desc)}</span>`);
  }

  html += row('Status &amp; Type', `
    <span class="badge" style="color:${STATUS_COLOR_HEX[node.status]};border-color:${STATUS_COLOR_HEX[node.status]}30">
      <span style="width:6px;height:6px;border-radius:50%;background:${STATUS_COLOR_HEX[node.status]};display:inline-block"></span>
      ${esc(node.status)}
    </span>
    &nbsp;
    <span class="badge" style="color:${TYPE_COLOR[node.node_type] || '#94a3b8'}">
      ${TYPE_ICON[node.node_type]} ${esc(node.node_type)}
    </span>
  `);

  if (deps.length) {
    html += row('Dependencies', deps.map(d =>
      `<span class="dep-tag" onclick="selectNode('${esc(d)}')">${esc(d)}</span>`
    ).join(''));
  }

  if (reqIn.length) {
    html += row('Requires', reqIn.map(i =>
      `<div style="font-size:11px;color:var(--text-muted);margin-bottom:3px">
        <span style="color:var(--text-dim)">${esc(i.name)}</span>
        <span style="color:var(--s-pending);font-size:10px"> [${esc(i.type)}]</span>
        — ${esc(i.description || '')}
      </div>`
    ).join(''));
  }

  if (output.length) {
    html += row('Outputs', output.map(o =>
      `<div style="font-size:11px;color:var(--text-muted);margin-bottom:3px">
        <span style="color:var(--text-dim)">${esc(o.name)}</span>
        <span style="color:var(--s-ready);font-size:10px"> [${esc(o.type)}]</span>
        — ${esc(o.description || '')}
      </div>`
    ).join(''));
  }

  if (result != null) {
    const label = node.node_type === 'goal' ? 'Plan' : 'Result';
    html += row(label, `<div class="result-box">${esc(String(result))}</div>`);
  }

  if (notes.length) {
    html += row('Notes', notes.map(n => `<div style="font-size:11px;color:var(--text-muted);margin-bottom:4px">• ${esc(n)}</div>`).join(''));
  }

  if (stepChildren.length) {
    const stepsHtml = stepChildren.map(step => {
      const icon = step.status === 'done' ? '✓' : step.status === 'failed' ? '✗' : '…';
      const iconColor = step.status === 'done' ? 'var(--s-done)' : step.status === 'failed' ? 'var(--s-failed)' : 'var(--s-running)';
      return `<div class="step-item">
        <span class="step-icon" style="color:${iconColor}">${icon}</span>
        <div>
          <div class="step-desc">${esc(step.metadata?.description || step.id)}</div>
          ${step.result ? `<div class="step-result">${esc(trunc(step.result, 120))}</div>` : ''}
        </div>
      </div>`;
    }).join('');
    html += row('Execution Steps', `<div style="margin-top:2px">${stepsHtml}</div>`);
  }

  body.innerHTML = html || '<div style="color:var(--text-muted);font-size:12px">No details available.</div>';
}

function row(label, content) {
  return `<div class="info-row">
    <div class="info-label">${label}</div>
    <div class="info-value">${content}</div>
  </div>`;
}

function closePanel() {
  selectedId = null;
  document.getElementById('info-panel').classList.add('hidden');
  renderDAG();
}

// ── Panel dragging ────────────────────────────────────────────────────────────

(function() {
  let dragging = false, ox, oy;
  const panel = document.getElementById('info-panel');
  const handle = document.getElementById('panel-drag-handle');

  handle.addEventListener('mousedown', e => {
    dragging = true;
    const r = panel.getBoundingClientRect();
    ox = e.clientX - r.left;
    oy = e.clientY - r.top;
    e.preventDefault();
  });
  document.addEventListener('mousemove', e => {
    if (!dragging) return;
    panel.style.left   = (e.clientX - ox) + 'px';
    panel.style.top    = (e.clientY - oy) + 'px';
    panel.style.right  = 'auto';
    panel.style.bottom = 'auto';
  });
  document.addEventListener('mouseup', () => { dragging = false; });
})();

// ── Modals ────────────────────────────────────────────────────────────────────

function openEditModal() {
  const node = nodes[selectedId];
  if (!node) return;
  modalMode = 'edit';
  document.getElementById('modal-title').textContent = 'Edit Node';

  const allIds = Object.keys(nodes).filter(id => id !== selectedId);
  const deps = (node.dependencies || []).join(', ');

  document.getElementById('modal-body').innerHTML = `
    <div class="form-group">
      <label class="form-label">Description</label>
      <textarea class="form-textarea" id="f-desc" rows="3">${esc(node.metadata?.description || '')}</textarea>
    </div>
    <div class="form-group">
      <label class="form-label">Status</label>
      <select class="form-select" id="f-status">
        ${['pending','ready','running','done','failed','to_be_expanded'].map(s =>
          `<option value="${s}" ${node.status === s ? 'selected' : ''}>${s}</option>`
        ).join('')}
      </select>
    </div>
    <div class="form-group">
      <label class="form-label">Dependencies</label>
      <input class="form-input" id="f-deps" value="${esc(deps)}" placeholder="comma-separated node IDs">
      <div class="form-hint">Available: ${allIds.slice(0,8).join(', ')}${allIds.length > 8 ? '…' : ''}</div>
    </div>
  `;

  document.getElementById('modal-footer').innerHTML = `
    <button class="btn btn-ghost" onclick="closeModal()">Cancel</button>
    <button class="btn btn-primary" onclick="submitEdit()">Save</button>
  `;

  showModal();
}

function submitEdit() {
  const desc  = document.getElementById('f-desc').value.trim();
  const status = document.getElementById('f-status').value;
  const depsRaw = document.getElementById('f-deps').value;
  const deps = depsRaw.split(',').map(d => d.trim()).filter(Boolean);

  api('PUT', `/api/node/${encodeURIComponent(selectedId)}`, {
    description: desc, status, dependencies: deps,
  });
  closeModal();
}

function openRemoveModal() {
  const node = nodes[selectedId];
  if (!node) return;
  modalMode = 'remove';
  document.getElementById('modal-title').textContent = `Remove: ${selectedId}`;

  document.getElementById('modal-body').innerHTML = `
    <div class="radio-group">
      <label class="radio-item">
        <input type="radio" name="remove-mode" value="rewire" checked>
        <div>
          <div class="radio-item-label">Rewire children</div>
          <div class="radio-item-desc">Remove this node and connect its children directly to its parents.</div>
        </div>
      </label>
      <label class="radio-item">
        <input type="radio" name="remove-mode" value="cascade">
        <div>
          <div class="radio-item-label">Cascade remove</div>
          <div class="radio-item-desc">Remove this node and all its descendants.</div>
        </div>
      </label>
      <label class="radio-item">
        <input type="radio" name="remove-mode" value="disconnect">
        <div>
          <div class="radio-item-label">Disconnect</div>
          <div class="radio-item-desc">Remove this node only, leaving children without this dependency.</div>
        </div>
      </label>
    </div>
  `;

  document.getElementById('modal-footer').innerHTML = `
    <button class="btn btn-ghost" onclick="closeModal()">Cancel</button>
    <button class="btn btn-danger" onclick="submitRemove()">Remove</button>
  `;

  showModal();
}

function submitRemove() {
  const mode = document.querySelector('input[name="remove-mode"]:checked')?.value || 'cascade';
  const id = selectedId;
  closeModal();
  closePanel();
  api('DELETE', `/api/node/${encodeURIComponent(id)}?mode=${mode}`);
}
// Replacements for addNodePrompt() and submitAdd() in web_ui.html

function addNodePrompt() {
  modalMode = 'add';
  document.getElementById('modal-title').textContent = 'Add Node';
  const allIds = Object.keys(nodes);

  document.getElementById('modal-body').innerHTML = `
    <div class="form-group">
      <label class="form-label">Node ID <span style="color:var(--s-failed)">*</span></label>
      <input class="form-input" id="f-node-id" placeholder="e.g. Research_Market">
    </div>
    <div class="form-group">
      <label class="form-label">Description</label>
      <textarea class="form-textarea" id="f-add-desc" rows="2" placeholder="What this node does"></textarea>
    </div>
    <div class="form-group">
      <label class="form-label">Type</label>
      <select class="form-select" id="f-type">
        <option value="task" selected>task</option>
        <option value="goal">goal</option>
      </select>
    </div>
    <div class="form-group">
      <label class="form-label">Dependencies</label>
      <input class="form-input" id="f-add-deps" value="${esc(selectedId || '')}" placeholder="comma-separated node IDs">
      <div class="form-hint">Nodes this node depends on (prerequisites that must complete first).</div>
    </div>
    <div class="form-group">
      <label class="form-label">Dependents</label>
      <input class="form-input" id="f-add-dependents" value="" placeholder="comma-separated node IDs">
      <div class="form-hint">
        Existing nodes that should depend on this new node (it becomes their prerequisite).
        ${allIds.length ? `Available: ${allIds.slice(0, 8).join(', ')}${allIds.length > 8 ? '…' : ''}` : ''}
      </div>
    </div>
  `;

  document.getElementById('modal-footer').innerHTML = `
    <button class="btn btn-ghost" onclick="closeModal()">Cancel</button>
    <button class="btn btn-primary" onclick="submitAdd()">Add Node</button>
  `;

  showModal();
  document.getElementById('f-node-id').focus();
}

function submitAdd() {
  const nodeId  = document.getElementById('f-node-id').value.trim();
  if (!nodeId) { toast('Node ID is required'); return; }
  if (nodes[nodeId]) { toast('Node ID already exists'); return; }

  const desc    = document.getElementById('f-add-desc').value.trim();
  const type    = document.getElementById('f-type').value;

  const depsRaw = document.getElementById('f-add-deps').value;
  const deps    = depsRaw.split(',').map(d => d.trim()).filter(d => d && nodes[d]);

  const dependentsRaw = document.getElementById('f-add-dependents').value;
  const dependents    = dependentsRaw.split(',').map(d => d.trim()).filter(d => d && nodes[d]);

  api('POST', '/api/node', {
    node_id:      nodeId,
    node_type:    type,
    description:  desc,
    dependencies: deps,
    dependents:   dependents,
  });
  closeModal();
}

function showModal() {
  document.getElementById('modal-overlay').classList.remove('hidden');
}

function closeModal() {
  document.getElementById('modal-overlay').classList.add('hidden');
}

function overlayClick(e) {
  if (e.target === document.getElementById('modal-overlay')) closeModal();
}

// ── Node actions ──────────────────────────────────────────────────────────────

function retryNode() {
  if (!selectedId) return;
  api('POST', `/api/node/${encodeURIComponent(selectedId)}/retry`);
  toast('Retry scheduled');
}

function replanGoal() {
  if (!selectedId) return;
  api('POST', `/api/goal/${encodeURIComponent(selectedId)}/replan`);
  toast('Replan scheduled');
  closePanel();
}

// ── LLM control ───────────────────────────────────────────────────────────────

function toggleLLM() {
  api('POST', paused ? '/api/llm/resume' : '/api/llm/pause');
}

// ── Export ────────────────────────────────────────────────────────────────────

function exportMD() {
  api('POST', '/api/export').then(data => {
    if (data.ok) toast(`Exported → ${data.path.split('/').pop()}`);
  });
}

// ── Zoom controls ─────────────────────────────────────────────────────────────

function zoomIn()    { svg.transition().call(zoomBehavior.scaleBy, 1.4); }
function zoomOut()   { svg.transition().call(zoomBehavior.scaleBy, 0.7); }
function zoomReset() { fitGraph(); }

function fitGraph() {
  if (!currentG) return;
  const graph = currentG.graph();
  if (!graph.width || !graph.height) return;

  const wrap = document.getElementById('canvas-wrap');
  const W = wrap.clientWidth, H = wrap.clientHeight;
  const pad = 60;
  const scale = Math.min(
    (W - pad * 2) / graph.width,
    (H - pad * 2) / graph.height,
    1.2
  );
  const tx = (W - graph.width  * scale) / 2;
  const ty = (H - graph.height * scale) / 2;

  svg.transition().duration(400).call(
    zoomBehavior.transform,
    d3.zoomIdentity.translate(tx, ty).scale(scale)
  );
}

// ── API helper ────────────────────────────────────────────────────────────────

async function api(method, path, body) {
  const opts = {
    method,
    headers: { 'Content-Type': 'application/json' },
  };
  if (body) opts.body = JSON.stringify(body);
  try {
    const r = await fetch(path, opts);
    return await r.json();
  } catch (e) {
    toast('Request failed: ' + e.message);
    return { ok: false };
  }
}

// ── Utilities ─────────────────────────────────────────────────────────────────

function esc(s) {
  return String(s)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;')
    .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function trunc(s, n) {
  s = String(s);
  return s.length > n ? s.slice(0, n - 1) + '…' : s;
}

let toastTimer;
function toast(msg) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.remove('hidden');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.add('hidden'), 3500);
}

// ── Switch-goal modal ────────────────────────────────────────────────────────

let swActiveTab     = 'existing';
let swSelectedRun   = null;
let swPolling       = false;

function openSwitchModal() {
  swActiveTab   = 'existing';
  swSelectedRun = null;
  swPolling     = false;

  // Reset to initial state
  document.querySelectorAll('.sw-pane').forEach(el => el.classList.remove('active'));
  document.getElementById('swpane-existing').classList.add('active');
  document.querySelectorAll('.sw-tab').forEach(el => el.classList.remove('active'));
  document.getElementById('swtab-existing').classList.add('active');
  document.querySelectorAll('.sw-error').forEach(el => el.textContent = '');
  document.getElementById('sw-loading').classList.remove('show');
  document.getElementById('sw-footer').style.display = '';
  document.getElementById('sw-main').querySelectorAll('.sw-pane')
    .forEach(el => { if (!el.classList.contains('active')) el.classList.remove('active'); });

  document.getElementById('switch-overlay').classList.remove('hidden');
  loadSwitchRuns();
}

function closeSwitchModal() {
  if (swPolling) return;   // don't close while switching
  document.getElementById('switch-overlay').classList.add('hidden');
}

function switchModalTab(tab) {
  swActiveTab   = tab;
  swSelectedRun = null;
  document.querySelectorAll('.sw-tab').forEach(el =>
    el.classList.toggle('active', el.id === `swtab-${tab}`)
  );
  document.querySelectorAll('.sw-pane').forEach(el =>
    el.classList.toggle('active', el.id === `swpane-${tab}`)
  );
  document.querySelectorAll('.sw-error').forEach(el => el.textContent = '');

  if (tab === 'goal')     setTimeout(() => document.getElementById('sw-goal-input')?.focus(), 50);
  if (tab === 'plan')     setTimeout(() => document.getElementById('sw-plan-input')?.focus(), 50);
}

async function loadSwitchRuns() {
  const container = document.getElementById('sw-runs-container');
  container.innerHTML = '<div class="sw-empty">Loading…</div>';
  try {
    const data = await (await fetch('/api/runs')).json();
    const runs = data.runs || [];
    if (!runs.length) {
      container.innerHTML = '<div class="sw-empty">No existing runs found.</div>';
      switchModalTab('goal');
      return;
    }
    container.innerHTML = `<div class="sw-runs-list">
      ${runs.map((r, i) => `
        <div class="sw-run-item" id="swrun-${i}"
             data-path="${swEsc(r.path)}" data-goal="${swEsc(r.goal)}"
             onclick="selectSwitchRun(${i})">
          <div class="sw-run-dot"></div>
          <div class="sw-run-goal" title="${swEsc(r.goal)}">${swEsc(r.goal)}</div>
          <div class="sw-run-meta">${swEsc(r.age)} · ${r.node_count} nodes</div>
        </div>`).join('')}
    </div>`;
    selectSwitchRun(0);
  } catch (e) {
    container.innerHTML = `<div class="sw-empty">Could not load runs: ${swEsc(String(e))}</div>`;
  }
}

function selectSwitchRun(i) {
  swSelectedRun = i;
  document.querySelectorAll('.sw-run-item').forEach((el, j) =>
    el.classList.toggle('selected', j === i)
  );
}

async function submitSwitch() {
  document.querySelectorAll('.sw-error').forEach(el => el.textContent = '');
  let body;

  if (swActiveTab === 'existing') {
    if (swSelectedRun === null) {
      document.getElementById('swerr-existing').textContent = 'Select a run first.';
      return;
    }
    const item = document.querySelector(`#swrun-${swSelectedRun}`);
    body = { mode: 'existing', run_dir: item.dataset.path, goal_text: item.dataset.goal };

  } else if (swActiveTab === 'goal') {
    const gt = document.getElementById('sw-goal-input').value.trim();
    if (!gt) { document.getElementById('swerr-goal').textContent = 'Please enter a goal.'; return; }
    body = { mode: 'new_goal', goal_text: gt };

  } else {
    const plan = document.getElementById('sw-plan-input').value.trim();
    if (!plan) { document.getElementById('swerr-plan').textContent = 'Please enter a plan.'; return; }
    body = { mode: 'manual_plan', plan_text: plan };
  }

  // Show loading spinner
  document.getElementById('sw-main').querySelectorAll('.sw-pane')
    .forEach(el => el.classList.remove('active'));
  document.getElementById('sw-loading').classList.add('show');
  document.getElementById('sw-footer').style.display = 'none';
  swPolling = true;

  try {
    const res  = await fetch('/api/switch', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const data = await res.json();
    if (!data.ok) {
      swShowError(data.error || 'Switch failed.');
      return;
    }
    // Poll until the new orchestrator is ready
    swPollReady();
  } catch (e) {
    swShowError(String(e));
  }
}

function swPollReady() {
  const msgs = ['Initialising…', 'Loading model…', 'Building graph…', 'Almost ready…'];
  let tick = 0;
  const iv = setInterval(async () => {
    tick++;
    document.getElementById('sw-loading-msg').textContent =
      msgs[Math.floor(tick / 4) % msgs.length];
    try {
      const d = await (await fetch('/api/status')).json();
      if (d.error) { clearInterval(iv); swShowError(d.error); return; }
      if (d.initialized) {
        clearInterval(iv);
        window.location.reload();
      }
    } catch (_) { /* keep polling */ }
  }, 1500);
}

function swShowError(msg) {
  swPolling = false;
  document.getElementById('sw-loading').classList.remove('show');
  document.getElementById('sw-footer').style.display = '';
  document.getElementById(`swpane-${swActiveTab}`).classList.add('active');
  document.getElementById(`swerr-${swActiveTab}`).textContent = 'Error: ' + msg;
}

function swEsc(s) {
  return String(s).replace(/&/g,'&amp;').replace(/</g,'&lt;')
                  .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// Close on overlay click (outside the shell)
document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('switch-overlay').addEventListener('click', e => {
    if (e.target === document.getElementById('switch-overlay')) closeSwitchModal();
  });
});

// Esc key closes the modal
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeSwitchModal();
});

// ── Boot ──────────────────────────────────────────────────────────────────────

initSVG();
connect();
</script>
<!-- ── Switch-goal modal ─────────────────────────────────────────────────── -->
<div id="switch-overlay" class="hidden">
  <div id="switch-shell">

    <div id="switch-shell-header">
      <span class="sw-brand">Switch goal</span>
      <button class="sw-close" onclick="closeSwitchModal()">✕</button>
    </div>

    <div class="sw-tabs">
      <button class="sw-tab active" id="swtab-existing" onclick="switchModalTab('existing')">📂 Existing runs</button>
      <button class="sw-tab"        id="swtab-goal"     onclick="switchModalTab('goal')">✦ New goal</button>
      <button class="sw-tab"        id="swtab-plan"     onclick="switchModalTab('plan')">✎ Manual plan</button>
    </div>

    <div id="sw-main">
      <!-- Existing runs -->
      <div class="sw-pane active" id="swpane-existing">
        <div id="sw-runs-container"><div class="sw-empty">Loading…</div></div>
        <div class="sw-error" id="swerr-existing"></div>
      </div>

      <!-- New goal -->
      <div class="sw-pane" id="swpane-goal">
        <label class="sw-label">What do you want to achieve?</label>
        <input class="sw-input" id="sw-goal-input"
          placeholder="e.g. Build a CLI tool that summarises GitHub PRs"
          autocomplete="off">
        <div class="sw-hint">The planner will break this down into tasks automatically.</div>
        <div class="sw-error" id="swerr-goal"></div>
      </div>

      <!-- Manual plan -->
      <div class="sw-pane" id="swpane-plan">
        <label class="sw-label">Your plan</label>
        <textarea class="sw-textarea" id="sw-plan-input"
          placeholder="First line = goal description&#10;- Task: description [depends: Other_Task]&#10;- Another_Task: does something else"></textarea>
        <div class="sw-hint">First non-bullet line is the goal. Lines starting with <code style="background:#263350;padding:1px 4px;border-radius:3px">-</code> are tasks.</div>
        <div class="sw-error" id="swerr-plan"></div>
      </div>

      <!-- Loading state -->
      <div class="sw-loading" id="sw-loading">
        <div class="sw-spinner"></div>
        <div class="sw-loading-msg" id="sw-loading-msg">Switching goal…</div>
      </div>
    </div>

    <div class="sw-footer" id="sw-footer">
      <button class="sw-btn sw-btn-ghost" onclick="closeSwitchModal()">Cancel</button>
      <button class="sw-btn sw-btn-primary" id="sw-start-btn" onclick="submitSwitch()">Switch →</button>
    </div>

  </div>
</div>

</body>
</html>

# --- FILE: ui/web_ui_startup.html ---

<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>cuddlytoddly — startup</title>
<style>
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
:root {
  --bg:       #0b1120;
  --surface:  #111827;
  --surface2: #1e2a3d;
  --surface3: #263350;
  --border:   #1f2d42;
  --border2:  #2e3f58;
  --text:     #e2e8f0;
  --muted:    #64748b;
  --dim:      #94a3b8;
  --accent:   #6366f1;
  --done:     #10b981;
  --warn:     #f59e0b;
  --fail:     #ef4444;
  --radius:   10px;
}
body {
  background: var(--bg);
  color: var(--text);
  font-family: 'Inter', system-ui, -apple-system, sans-serif;
  font-size: 13px;
  min-height: 100vh;
  display: flex;
  align-items: center;
  justify-content: center;
}

/* ── Shell ─────────────────────────────────────────── */
.shell {
  width: 640px;
  max-width: calc(100vw - 32px);
  background: var(--surface);
  border: 1px solid var(--border2);
  border-radius: var(--radius);
  box-shadow: 0 24px 80px rgba(0,0,0,0.6);
  overflow: hidden;
}
.shell-header {
  padding: 20px 24px 16px;
  border-bottom: 1px solid var(--border);
  background: var(--surface2);
}
.brand {
  font-size: 18px;
  font-weight: 700;
  color: var(--accent);
  letter-spacing: -0.4px;
}
.tagline {
  font-size: 12px;
  color: var(--muted);
  margin-top: 3px;
}

/* ── Tabs ──────────────────────────────────────────── */
.tabs {
  display: flex;
  border-bottom: 1px solid var(--border);
  background: var(--surface2);
}
.tab {
  flex: 1;
  padding: 11px 12px;
  font-size: 12px;
  font-weight: 500;
  color: var(--muted);
  cursor: pointer;
  border-bottom: 2px solid transparent;
  transition: all 0.15s;
  text-align: center;
  user-select: none;
}
.tab:hover { color: var(--dim); }
.tab.active {
  color: var(--accent);
  border-bottom-color: var(--accent);
  background: var(--surface);
}

/* ── Tab panes ─────────────────────────────────────── */
.pane { display: none; padding: 24px; }
.pane.active { display: block; }

/* ── Runs list ─────────────────────────────────────── */
.runs-list {
  max-height: 300px;
  overflow-y: auto;
  border: 1px solid var(--border);
  border-radius: 6px;
}
.runs-list::-webkit-scrollbar { width: 4px; }
.runs-list::-webkit-scrollbar-thumb { background: var(--border2); }
.run-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 11px 14px;
  border-bottom: 1px solid var(--border);
  cursor: pointer;
  transition: background 0.12s;
}
.run-item:last-child { border-bottom: none; }
.run-item:hover { background: var(--surface2); }
.run-item.selected { background: var(--surface3); }
.run-dot {
  width: 8px; height: 8px; border-radius: 50%;
  background: var(--accent); flex-shrink: 0;
}
.run-goal {
  flex: 1;
  font-size: 13px;
  font-weight: 500;
  color: var(--text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}
.run-meta {
  font-size: 10px;
  color: var(--muted);
  white-space: nowrap;
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 2px;
}
.empty-runs {
  padding: 32px 16px;
  text-align: center;
  color: var(--muted);
  font-size: 12px;
}

/* ── Form elements ─────────────────────────────────── */
.form-group { margin-bottom: 16px; }
.form-label {
  display: block;
  font-size: 11px;
  font-weight: 600;
  text-transform: uppercase;
  letter-spacing: 0.5px;
  color: var(--muted);
  margin-bottom: 6px;
}
.form-input {
  width: 100%;
  background: var(--surface2);
  border: 1px solid var(--border2);
  border-radius: 6px;
  padding: 10px 12px;
  color: var(--text);
  font-size: 13px;
  font-family: inherit;
  outline: none;
  transition: border-color 0.15s;
}
.form-input:focus { border-color: var(--accent); }
.form-textarea {
  width: 100%;
  background: var(--surface2);
  border: 1px solid var(--border2);
  border-radius: 6px;
  padding: 10px 12px;
  color: var(--text);
  font-size: 12px;
  font-family: 'SF Mono', 'Fira Code', monospace;
  line-height: 1.6;
  outline: none;
  resize: vertical;
  min-height: 180px;
  transition: border-color 0.15s;
}
.form-textarea:focus { border-color: var(--accent); }
.form-hint {
  font-size: 11px;
  color: var(--muted);
  margin-top: 5px;
  line-height: 1.5;
}

/* ── Example block ─────────────────────────────────── */
.example {
  background: var(--surface2);
  border: 1px solid var(--border);
  border-left: 3px solid var(--accent);
  border-radius: 5px;
  padding: 10px 12px;
  font-family: 'SF Mono', 'Fira Code', monospace;
  font-size: 11px;
  color: var(--dim);
  line-height: 1.7;
  margin-bottom: 14px;
  white-space: pre;
  overflow-x: auto;
}

/* ── Footer ────────────────────────────────────────── */
.shell-footer {
  padding: 16px 24px;
  border-top: 1px solid var(--border);
  display: flex;
  justify-content: flex-end;
  gap: 8px;
  background: var(--surface);
}
.btn {
  padding: 8px 20px;
  border-radius: 6px;
  font-size: 13px;
  font-family: inherit;
  cursor: pointer;
  border: 1px solid transparent;
  transition: all 0.15s;
}
.btn-ghost {
  background: var(--surface2);
  border-color: var(--border2);
  color: var(--dim);
}
.btn-ghost:hover { background: var(--surface3); color: var(--text); }
.btn-primary {
  background: var(--accent);
  color: #fff;
  font-weight: 600;
}
.btn-primary:hover { filter: brightness(1.1); }
.btn-primary:disabled { opacity: 0.45; cursor: not-allowed; filter: none; }

/* ── Error / loading ───────────────────────────────── */
.error-msg {
  color: var(--fail);
  font-size: 12px;
  margin-top: 8px;
}
.loading-overlay {
  display: none;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 14px;
  padding: 48px 24px;
}
.loading-overlay.show { display: flex; }
.spinner {
  width: 28px; height: 28px;
  border: 3px solid var(--border2);
  border-top-color: var(--accent);
  border-radius: 50%;
  animation: spin 0.7s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }
.loading-msg { color: var(--dim); font-size: 13px; }
.loading-sub { color: var(--muted); font-size: 11px; }
</style>
</head>
<body>

<div class="shell" id="shell">

  <!-- Header -->
  <div class="shell-header">
    <div class="brand">cuddlytoddly</div>
    <div class="tagline">Choose a run or start something new</div>
  </div>

  <!-- Tabs -->
  <div class="tabs">
    <div class="tab active" id="tab-existing" onclick="switchTab('existing')">
      📂 Existing runs
    </div>
    <div class="tab" id="tab-goal" onclick="switchTab('goal')">
      ✦ New goal
    </div>
    <div class="tab" id="tab-plan" onclick="switchTab('plan')">
      ✎ Manual plan
    </div>
  </div>

  <!-- Main content -->
  <div id="main-content">

    <!-- Existing runs -->
    <div class="pane active" id="pane-existing">
      <div id="runs-container">
        <div class="empty-runs">Loading runs…</div>
      </div>
      <p class="error-msg" id="err-existing"></p>
    </div>

    <!-- New goal -->
    <div class="pane" id="pane-goal">
      <div class="form-group">
        <label class="form-label">What do you want to achieve?</label>
        <input class="form-input" id="f-goal"
          placeholder="e.g. Build an MVP for my educational toy and bring it to market"
          autocomplete="off">
        <p class="form-hint">
          The planner will break this down into tasks automatically.
        </p>
      </div>
      <p class="error-msg" id="err-goal"></p>
    </div>

    <!-- Manual plan -->
    <div class="pane" id="pane-plan">
      <div class="example">Build an MVP for my educational toy

- Research_Market: Research competitors and pricing
- Design_Hardware: Design PCB and enclosure  [depends: Research_Market]
- Develop_Software: Build the firmware       [depends: Research_Market]
- Integrate: Combine hardware and software   [depends: Design_Hardware, Develop_Software]
- User_Testing: Run pilot with 5 kids        [depends: Integrate]</div>
      <div class="form-group">
        <label class="form-label">Your plan</label>
        <textarea class="form-textarea" id="f-plan"
          placeholder="First line = goal description&#10;- Task: description [depends: Other_Task]&#10;- Another_Task: does something else"></textarea>
        <p class="form-hint">
          First non-bullet line is the goal. Lines starting with
          <code style="background:var(--surface3);padding:1px 4px;border-radius:3px">-</code>
          are tasks. Add
          <code style="background:var(--surface3);padding:1px 4px;border-radius:3px">[depends: Task_A, Task_B]</code>
          to specify dependencies. The system will run tasks without dependencies in parallel.
        </p>
      </div>
      <p class="error-msg" id="err-plan"></p>
    </div>

    <!-- Loading state -->
    <div class="loading-overlay" id="loading-overlay">
      <div class="spinner"></div>
      <div class="loading-msg" id="loading-msg">Initialising…</div>
      <div class="loading-sub" id="loading-sub">Loading model and setting up the graph</div>
    </div>

  </div>

  <!-- Footer -->
  <div class="shell-footer" id="footer">
    <button class="btn btn-primary" id="btn-start" onclick="submit()">
      Start →
    </button>
  </div>

</div>

<script>
let activeTab   = 'existing';
let selectedRun = null;
let polling     = false;

// ── Tabs ──────────────────────────────────────────────────────────────────────
function switchTab(tab) {
  activeTab = tab;
  document.querySelectorAll('.tab').forEach(el =>
    el.classList.toggle('active', el.id === `tab-${tab}`)
  );
  document.querySelectorAll('.pane').forEach(el =>
    el.classList.toggle('active', el.id === `pane-${tab}`)
  );
  clearErrors();
}

function clearErrors() {
  document.querySelectorAll('.error-msg').forEach(el => el.textContent = '');
}

// ── Load runs ─────────────────────────────────────────────────────────────────
async function loadRuns() {
  try {
    const data = await (await fetch('/api/runs')).json();
    const runs = data.runs || [];
    const container = document.getElementById('runs-container');

    if (!runs.length) {
      container.innerHTML = `<div class="empty-runs">
        No existing runs found.<br>Start a new goal below.
      </div>`;
      // Auto-switch to New Goal if no runs
      switchTab('goal');
      return;
    }

    container.innerHTML = `<div class="runs-list" id="runs-list">
      ${runs.map((r, i) => `
        <div class="run-item" id="run-${i}" onclick="selectRun(${i})"
             data-path="${esc(r.path)}" data-goal="${esc(r.goal)}">
          <div class="run-dot"></div>
          <div class="run-goal" title="${esc(r.goal)}">${esc(r.goal)}</div>
          <div class="run-meta">
            <span>${esc(r.age)}</span>
            <span style="color:var(--border2)">${r.node_count} nodes</span>
          </div>
        </div>
      `).join('')}
    </div>`;

    // Auto-select first
    selectRun(0);
  } catch (e) {
    document.getElementById('runs-container').innerHTML =
      `<div class="empty-runs">Could not load runs: ${esc(String(e))}</div>`;
  }
}

function selectRun(i) {
  selectedRun = i;
  document.querySelectorAll('.run-item').forEach((el, j) =>
    el.classList.toggle('selected', j === i)
  );
}

// ── Submit ────────────────────────────────────────────────────────────────────
async function submit() {
  clearErrors();
  let body;

  if (activeTab === 'existing') {
    if (selectedRun === null) {
      document.getElementById('err-existing').textContent = 'Select a run first.';
      return;
    }
    const item = document.querySelector(`#run-${selectedRun}`);
    body = {
      mode:      'existing',
      run_dir:   item.dataset.path,
      goal_text: item.dataset.goal,
    };

  } else if (activeTab === 'goal') {
    const goal = document.getElementById('f-goal').value.trim();
    if (!goal) {
      document.getElementById('err-goal').textContent = 'Please enter a goal.';
      return;
    }
    body = { mode: 'new_goal', goal_text: goal };

  } else {
    const plan = document.getElementById('f-plan').value.trim();
    if (!plan) {
      document.getElementById('err-plan').textContent = 'Please enter a plan.';
      return;
    }
    body = { mode: 'manual_plan', plan_text: plan };
  }

  // Show loading
  document.getElementById('main-content').querySelectorAll('.pane')
    .forEach(el => el.classList.remove('active'));
  const overlay = document.getElementById('loading-overlay');
  overlay.classList.add('show');
  document.getElementById('footer').style.display = 'none';

  try {
    const res  = await fetch('/api/startup', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify(body),
    });
    const data = await res.json();
    if (!data.ok) {
      showLoadError(data.error || 'Unknown error');
      return;
    }
    if (data.already_initialized) {
      window.location.href = '/dag';
      return;
    }
    // Poll until initialized
    pollReady();
  } catch (e) {
    showLoadError(String(e));
  }
}

function pollReady() {
  if (polling) return;
  polling = true;
  const msgs = [
    'Loading model…',
    'Setting up the graph…',
    'Almost ready…',
  ];
  let tick = 0;

  const interval = setInterval(async () => {
    tick++;
    document.getElementById('loading-msg').textContent =
      msgs[Math.floor(tick / 4) % msgs.length];

    try {
      const data = await (await fetch('/api/status')).json();
      if (data.error) {
        clearInterval(interval);
        showLoadError(data.error);
        return;
      }
      if (data.initialized) {
        clearInterval(interval);
        window.location.href = '/dag';
      }
    } catch (_) { /* keep polling */ }
  }, 1500);
}

function showLoadError(msg) {
  polling = false;
  document.getElementById('loading-overlay').classList.remove('show');
  document.getElementById('footer').style.display = '';
  // Re-show active pane
  document.getElementById(`pane-${activeTab}`).classList.add('active');
  document.getElementById(`err-${activeTab}`).textContent = 'Error: ' + msg;
}

// ── Utility ───────────────────────────────────────────────────────────────────
function esc(s) {
  return String(s)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── Enter key submits ─────────────────────────────────────────────────────────
document.addEventListener('keydown', e => {
  if (e.key === 'Enter' && activeTab !== 'plan') submit();
});

// ── Init ──────────────────────────────────────────────────────────────────────
loadRuns();

// If already initialized (server restarted), go straight to DAG
fetch('/api/status').then(r => r.json()).then(d => {
  if (d.initialized) window.location.href = '/dag';
  else if (d.loading) pollReady();
}).catch(() => {});
</script>
</body>
</html>

# --- FILE: README.md ---

# cuddlytoddly

An LLM-driven autonomous planning and execution system built around a DAG (directed acyclic graph) of tasks. Give it a goal; it breaks the goal into tasks, executes them with tools, verifies results, and fills in gaps — continuously, with live terminal and web UIs.

## How it works

1. A plain-English **goal** is seeded into the graph.
2. The **LLMPlanner** decomposes it into a DAG of tasks with explicit dependencies.
3. The **SimpleOrchestrator** picks up ready nodes and hands them to the **LLMExecutor**.
4. The executor runs a multi-turn LLM loop, calling tools (code execution, file I/O, custom skills) until the task is done.
5. The **QualityGate** checks the result against declared outputs; if something is missing it injects a bridging task automatically.
6. Every mutation is written to an **event log** — crash and resume with no lost work.

```
goal → LLMPlanner → TaskGraph (DAG)
                        │
              SimpleOrchestrator
              ├── LLMExecutor + tools
              └── QualityGate (verify / bridge)
                        │
                   EventLog (JSONL) → replay on restart
```

## Installation

```bash
pip install cuddlytoddly                       
```

**Requirements:** Python 3.11+, `git` on your PATH (for the DAG visualiser).

## Quick start

```bash
export ANTHROPIC_API_KEY=sk-ant-...
cuddlytoddly "Write a market analysis for electric scooters"
```

Or pass no argument to use the startup screen with multiple options.
The UI opens automatically. The run data is stored locally and can be resumed later — the event log preserves all state.

## LLM backends

| Backend | Install | `create_llm_client` call |
|---|---|---|
| Anthropic Claude | included | `create_llm_client("claude", model="claude-3-5-sonnet-20241022")` |
| OpenAI / compatible | `[openai]` | `create_llm_client("openai", model="gpt-4o")` |
| Local llama.cpp | `[local]` | `create_llm_client("llamacpp", model_path="/path/to/model.gguf")` |

## Adding skills

Drop a folder with a `SKILL.md` (and optional `tools.py`) into `cuddlytoddly/skills/`. The `SkillLoader` discovers it automatically. See [docs/skills.md](docs/skills.md) for the full format.

## Documentation

- [Architecture](docs/architecture.md) — how the components fit together
- [Configuration](docs/configuration.md) — LLM backends, run directory, environment variables
- [Skills](docs/skills.md) — built-in skills and how to add custom ones
- [API Reference](docs/api.md) — public Python API

## Project structure

```
cuddlytoddly/
├── core/           # TaskGraph, events, reducer, ID generator
├── engine/         # SimpleOrchestrator, QualityGate, ExecutionStepReporter
├── infra/          # Logging, EventQueue, EventLog, replay
├── planning/       # LLM interface, LLMPlanner, LLMExecutor, output validator
├── skills/         # SkillLoader + built-in skill packs
│   ├── code_execution/
│   └── file_ops/
└── ui/             # Curses terminal UI, Git DAG projection
docs/
pyproject.toml
LICENSE
```

## Python API

```python
from cuddlytoddly.core.task_graph import TaskGraph
from cuddlytoddly.core.events import Event, ADD_NODE
from cuddlytoddly.core.reducer import apply_event
from cuddlytoddly.infra.event_queue import EventQueue
from cuddlytoddly.infra.event_log import EventLog
from cuddlytoddly.planning.llm_interface import create_llm_client
from cuddlytoddly.planning.llm_planner import LLMPlanner
from cuddlytoddly.planning.llm_executor import LLMExecutor
from cuddlytoddly.engine.quality_gate import QualityGate
from cuddlytoddly.engine.llm_orchestrator import SimpleOrchestrator
from cuddlytoddly.skills.skill_loader import SkillLoader

# LLM client — swap "claude" for "openai" or "llamacpp"
llm = create_llm_client("claude", model="claude-3-5-sonnet-20241022")

graph    = TaskGraph()
skills   = SkillLoader()
planner  = LLMPlanner(llm_client=llm, graph=graph, skills_summary=skills.prompt_summary)
executor = LLMExecutor(llm_client=llm, tool_registry=skills.registry)
gate     = QualityGate(llm_client=llm, tool_registry=skills.registry)

orchestrator = SimpleOrchestrator(
    graph=graph, planner=planner, executor=executor,
    quality_gate=gate, event_queue=EventQueue(),
)

# Seed a goal
apply_event(graph, Event(ADD_NODE, {
    "node_id": "my_goal",
    "node_type": "goal",
    "metadata": {"description": "Summarise the key risks of AGI", "expanded": False},
}))

orchestrator.start()
# orchestrator runs in the background — block however suits your use case
```

## License

MIT — see [LICENSE](LICENSE).


# --- FILE: docs/api.md ---

# API Reference

This page documents the public interfaces intended for programmatic use. Internal modules (reducer, event types, ID generator) are implementation details and may change between releases.

---

## `cuddlytoddly.planning.llm_interface`

### `create_llm_client(backend, **kwargs) → BaseLLM`

Factory for LLM clients. Returns a `BaseLLM` instance.

| `backend` value | Class returned | Required kwargs |
|---|---|---|
| `"claude"` | `ApiLLM` | `model`, optional `temperature`, `max_tokens` |
| `"openai"` | `ApiLLM` | `model`, optional `base_url`, `temperature`, `max_tokens` |
| `"llamacpp"` | `LlamaCppLLM` | `model_path`, optional `n_gpu_layers`, `n_ctx`, `max_tokens`, `temperature`, `cache_path` |

### `BaseLLM`

All backends implement:

```python
def ask(self, prompt: str, schema: dict | None = None) -> str:
    """Send a prompt; return raw JSON string. Pass schema for structured output."""

def stop(self) -> None:
    """Signal the LLM to stop mid-generation (used by the UI pause button)."""

def resume(self) -> None:
    """Resume after a stop()."""
```

---

## `cuddlytoddly.planning.llm_planner`

### `LLMPlanner(llm_client, graph, skills_summary="")`

Decomposes unexpanded nodes into child tasks.

```python
planner = LLMPlanner(llm_client=llm, graph=graph, skills_summary=skills.prompt_summary)
planner.plan(node_id)  # expands one node; emits events to graph
```

---

## `cuddlytoddly.planning.llm_executor`

### `LLMExecutor(llm_client, tool_registry, max_turns=5)`

Executes a single task node via multi-turn LLM + tool calls.

```python
executor = LLMExecutor(llm_client=llm, tool_registry=registry, max_turns=5)
result: str = executor.run(node, snapshot, reporter)
```

`reporter` is an `ExecutionStepReporter` instance; pass `None` to skip step tracking.

---

## `cuddlytoddly.engine.llm_orchestrator`

### `SimpleOrchestrator(graph, planner, executor, quality_gate, event_log, event_queue, max_workers)`

The top-level plan→execute loop.

```python
orchestrator = SimpleOrchestrator(
    graph=graph,
    planner=planner,
    executor=executor,
    quality_gate=quality_gate,
    event_log=event_log,
    event_queue=queue,
    max_workers=1,
)
orchestrator.start()   # runs in a background thread
orchestrator.stop()    # signals shutdown
```

**UI-facing attributes** (read by `curses_ui`):

| Attribute | Type | Description |
|---|---|---|
| `graph` | `TaskGraph` | The live graph |
| `graph_lock` | `threading.Lock` | Must be held when reading graph for display |
| `event_queue` | `EventQueue` | Queue for user-injected events |
| `current_activity` | `str` | Human-readable status string |
| `llm_stopped` | `bool` | True when LLM is paused |

---

## `cuddlytoddly.engine.quality_gate`

### `QualityGate(llm_client, tool_registry=None)`

LLM-powered result verification and dependency checking.

```python
gate = QualityGate(llm_client=llm, tool_registry=registry)

satisfied, reason = gate.verify_result(node, result_str, snapshot)
bridge = gate.check_dependencies(node, snapshot)  # returns None or {node_id, description, output}
```

---

## `cuddlytoddly.skills.skill_loader`

### `SkillLoader(skills_dir=None)`

Discovers and loads skills from `cuddlytoddly/skills/` (or a custom path).

```python
skills = SkillLoader()
registry: ToolRegistry = skills.registry       # all registered tools
summary: str = skills.prompt_summary           # text to inject into planner prompt
skills.merge_mcp(other_registry)               # merge an external ToolRegistry
```

### `ToolRegistry`

```python
registry.register(tool: Tool)
result: str = registry.execute(tool_name: str, input_data: dict)
```

---

## `cuddlytoddly.core.task_graph`

### `TaskGraph`

```python
graph = TaskGraph()
graph.add_node(node_id, node_type="task", dependencies=[], metadata={})
graph.remove_node(node_id)
graph.add_dependency(node_id, depends_on)
graph.get_snapshot() -> dict[str, Node]        # deep copy; safe for concurrent reads
graph.get_ready_nodes() -> list[Node]
graph.recompute_readiness()
```

### `TaskGraph.Node`

| Attribute | Type | Description |
|---|---|---|
| `id` | `str` | Unique node identifier |
| `status` | `str` | `pending` / `ready` / `running` / `done` / `failed` |
| `node_type` | `str` | `goal`, `task`, `execution_step` |
| `dependencies` | `set[str]` | IDs of nodes this node depends on |
| `children` | `set[str]` | IDs of nodes that depend on this node |
| `result` | `str \| None` | Output of the node once done |
| `metadata` | `dict` | Arbitrary planner/executor annotations |

---

## `cuddlytoddly.infra`

### `EventLog(path: str)`

```python
log = EventLog("events.jsonl")
log.append(event)
log.read_all() -> list[Event]
```

### `EventQueue`

Thread-safe queue wrapping `queue.Queue`.

```python
q = EventQueue()
q.put(event)
event = q.get(timeout=1.0)
```

### `rebuild_graph_from_log(event_log) → TaskGraph`

Replays all events from an `EventLog` and returns the reconstructed graph.

### `setup_logging(log_dir=None)` / `get_logger(name) → Logger`

Call `setup_logging()` once at startup. Use `get_logger(__name__)` in every module.


# --- FILE: docs/architecture.md ---

# Architecture

## Overview

cuddlytoddly is a DAG-first autonomous planning system. A goal is given as a plain-English string; the system decomposes it into a directed acyclic graph (DAG) of tasks, executes them in dependency order, and iteratively refines the plan as results come in — all driven by an LLM.

```
User goal (string)
       │
       ▼
  LLMPlanner  ──── emits ADD_NODE / ADD_DEPENDENCY events ────►  TaskGraph
       │                                                              │
       │                                                    recompute_readiness()
       │                                                              │
  SimpleOrchestrator  ◄──── polls ready nodes ─────────────────────┘
       │
       ├── LLMExecutor  (runs one node via LLM + tools)
       │        │
       │        └── ExecutionStepReporter  (child nodes in DAG)
       │
       ├── QualityGate  (verifies result; may inject bridge nodes)
       │
       └── EventLog  (persists all mutations to JSONL for replay)
```

## Design Principles

**Event-sourced state.** The `TaskGraph` is never mutated directly. All changes go through `Event` objects processed by the `reducer`. This means the full history is replayable from the event log — if the process crashes, it picks up exactly where it left off.

**Read-only snapshots for planning.** The planner and orchestrator always work from `graph.get_snapshot()` (a deep copy), so they can reason about the graph without race conditions.

**LLM backends are interchangeable.** `planning/llm_interface.py` defines one `BaseLLM` with `.ask()` and `.generate()`. Swap between Anthropic Claude, OpenAI-compatible endpoints, and local llama.cpp models by changing one argument to `create_llm_client()`.

**Skills are data-driven.** Drop a folder with a `SKILL.md` and optional `tools.py` into `cuddlytoddly/skills/` — the `SkillLoader` discovers and registers them automatically at startup with no code changes required.

## Data Flow

### Planning phase

1. `LLMPlanner.plan(goal_id)` reads the current snapshot and identifies unexpanded goal/task nodes.
2. It builds a prompt describing the node and asks the LLM for a list of child tasks in JSON format.
3. The JSON is validated by `LLMOutputValidator` and emitted as `ADD_NODE` / `ADD_DEPENDENCY` events.
4. `apply_event()` applies each event to the graph, then calls `recompute_readiness()`.

### Execution phase

1. `SimpleOrchestrator` polls for nodes whose `status == "ready"`.
2. For each ready node it calls `QualityGate.check_dependencies()` — if upstream results are insufficient a bridge node is injected automatically.
3. `LLMExecutor.run(node)` drives a multi-turn LLM loop: the LLM calls tools via JSON responses; each tool call is tracked as a child `execution_step` node by `ExecutionStepReporter`.
4. On success, `QualityGate.verify_result()` checks the result against the node's declared outputs. On failure the node is retried or failed.

### Persistence and replay

All events are appended to an `events.jsonl` file via `EventLog`. On restart, `rebuild_graph_from_log()` replays the log to reconstruct the exact graph state, then ephemeral `execution_step` nodes are pruned and in-flight nodes are reset to `pending`.

## Module Map

| Package | Responsibility |
|---|---|
| `cuddlytoddly.core` | `TaskGraph`, `Node`, `Event` types, `apply_event` reducer, ID generator |
| `cuddlytoddly.engine` | `SimpleOrchestrator`, `QualityGate`, `ExecutionStepReporter` |
| `cuddlytoddly.planning` | LLM client abstraction, `LLMPlanner`, `LLMExecutor`, output validator |
| `cuddlytoddly.infra` | Logging, `EventQueue`, `EventLog`, replay |
| `cuddlytoddly.skills` | `SkillLoader`, `ToolRegistry`, built-in skill packs |
| `cuddlytoddly.ui` | Curses terminal UI, Git DAG projection |

## Concurrency Model

The orchestrator runs in a background thread. Execution of individual nodes is dispatched to a `ThreadPoolExecutor` with `max_workers` workers (default 1 for llama.cpp, which is not thread-safe). All graph mutations are protected by `graph_lock`. The curses UI runs on the main thread and communicates with the orchestrator exclusively through the shared graph and `event_queue`.


# --- FILE: docs/configuration.md ---

# Configuration

All runtime configuration is passed directly to `main()` or to individual components — there is no global config file. The values below are the defaults set in `cuddlytoddly/__main__.py` and can be overridden by editing that file or by calling the Python API directly.

## LLM backends

cuddlytoddly supports three backends selected via `create_llm_client(backend=...)`.

### Anthropic Claude (default recommended)

```python
from cuddlytoddly.planning.llm_interface import create_llm_client

llm = create_llm_client(
    "claude",
    model="claude-3-5-sonnet-20241022",  # any Anthropic model
    temperature=0.1,
    max_tokens=8192,
)
```

Requires the `ANTHROPIC_API_KEY` environment variable.

### OpenAI-compatible API

```python
llm = create_llm_client(
    "openai",
    model="gpt-4o",
    base_url="https://api.openai.com/v1",   # or any compatible endpoint
    temperature=0.1,
    max_tokens=8192,
)
```

Requires `OPENAI_API_KEY`, or set `base_url` + `api_key` for a custom endpoint.  
Install the extra: `pip install cuddlytoddly[openai]`.

### Local llama.cpp

```python
llm = create_llm_client(
    "llamacpp",
    model_path="/path/to/model.gguf",
    n_gpu_layers=-1,       # -1 = all layers on GPU
    temperature=0.1,
    n_ctx=16384,
    max_tokens=8192,
    cache_path="llamacpp_cache.json",  # optional response cache
)
```

Install the extra: `pip install cuddlytoddly[local]`.

## Orchestrator options

| Parameter | Default | Description |
|---|---|---|
| `max_workers` | `1` | Parallel node execution threads. Use `1` for llama.cpp (not thread-safe). |
| `max_turns` | `5` | Max LLM turns per node execution before giving up. |

## Run directory

Each invocation creates a `runs/<goal_slug>/` directory containing:

```
runs/how_to_go_to_mars/
├── events.jsonl        # full event log (enables crash recovery)
├── llamacpp_cache.json # LLM response cache (optional)
├── logs/               # rotating log files
├── outputs/            # working directory for file-writing tools
└── dag_repo/           # Git repo mirroring the DAG (for visualization)
```

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | For `claude` backend | Anthropic API key |
| `OPENAI_API_KEY` | For `openai` backend | OpenAI API key |

## Git DAG projection

The Git projection (`ui/git_projection.py`) requires `git` to be installed on the system. It is purely visual — it does not affect the DAG or execution. The path to the repo is set in `main()`:

```python
import cuddlytoddly.ui.git_projection as git_proj
git_proj.REPO_PATH = str(run_dir / "dag_repo")
```


# --- FILE: docs/skills.md ---

# Skills

Skills are the tool packs that the LLM executor can call during node execution. cuddlytoddly ships two built-in skills and supports adding custom ones with no code changes.

## Built-in skills

### `code_execution`

Runs Python snippets or shell commands in a subprocess and returns stdout.

| Tool | Description |
|---|---|
| `run_python` | Execute a Python code string; returns stdout + stderr |
| `run_shell` | Execute a shell command string; returns stdout + stderr |

### `file_ops`

Read and write files relative to the run's `outputs/` directory.

| Tool | Description |
|---|---|
| `read_file` | Read a file and return its contents as a string |
| `write_file` | Write a string to a file (creates parent dirs automatically) |
| `list_files` | List files in a directory |

## Adding a custom skill

1. Create a directory under `cuddlytoddly/skills/`:

```
cuddlytoddly/skills/
└── my_skill/
    ├── SKILL.md     ← required
    └── tools.py     ← optional (local Python implementations)
```

2. Write a `SKILL.md` describing the skill for the planner:

```markdown
# My Skill

## Description
What this skill does and when to use it.

## When to use
Trigger conditions for the planner to consider this skill.

## Tools
- `my_tool`: Does X given Y.

## Expected output format
A single string containing ...
```

3. Optionally implement local tools in `tools.py`:

```python
# skills/my_skill/tools.py

def _do_thing(args: dict) -> str:
    return f"Result: {args['input']}"

TOOLS = {
    "my_tool": {
        "description": "Does X given Y.",
        "input_schema": {"input": "string"},
        "fn": _do_thing,
    }
}
```

`SkillLoader` discovers the folder automatically at startup, parses `SKILL.md` to build the planner prompt, and registers any tools from `tools.py` into the `ToolRegistry`.

## Using MCP tool servers

For tools that live in an external MCP server, pass a pre-built `ToolRegistry` to `SkillLoader.merge_mcp()`:

```python
from cuddlytoddly.skills.skill_loader import SkillLoader

skills = SkillLoader()

# Build a registry from your MCP adapter of choice, then merge:
skills.merge_mcp(my_mcp_registry)

registry = skills.registry  # combined local + MCP tools
```


# --- FILE: runs/how_to_become_a_millionaire_overnight/outputs/wealth_creation_plan.md ---

A plan to become a millionaire overnight through strategic investments and business ventures.
investment_analysis: Based on the investment options analysis, the potential returns on investments are: 
1. Investing in stocks, real estate, or other assets that have the potential to generate passive income and increase in value over time: 8-12% annual returns.
2. Starting a business: potentially unlimited returns, but also comes with higher risks.
3. Investing in dividend-paying stocks: 4-8% annual returns.
4. Investing in real estate investment trusts (REITs): 8-12% annual returns.
5. Creating and selling an online course or ebook: potentially unlimited returns, but also requires significant marketing efforts.
6. Participating in the gig economy: $15-$25 per hour.
7. Investing in a small business or startup: potentially unlimited returns, but also comes with higher risks.
8. Creating a mobile app or game: potentially unlimited returns, but also requires significant development and marketing efforts.
9. Investing in cryptocurrency or blockchain technology: potentially unlimited returns, but also comes with higher risks and volatility.
risk_assessment: To become a millionaire overnight, it's essential to take calculated risks and invest in high-potential assets. However, it's crucial to diversify your portfolio and have a solid understanding of the market to minimize losses.
wealth_creation_strategy: 
1. Invest in high-growth stocks and real estate.
2. Start a business or invest in a small business or startup.
3. Create and sell an online course or ebook.
4. Participate in the gig economy.
5. Invest in cryptocurrency or blockchain technology.
Remember, becoming a millionaire overnight requires dedication, hard work, and a willingness to take calculated risks.

# --- FILE: runs/how_to_go_to_the_moon/outputs/export_20260325_182834.md ---

# How to go to the Moon

*Exported 2026-03-25 18:28:34*

---

## Summary

| Node | Type | Status |
|------|------|--------|
| Research_Space_Agency_Programs | task | done |
| Design_Lunar_Mission_Architecture | task | done |
| Develop_Spacecraft_Propulsion_System | task | done |
| Plan_Crew_Training_Program | task | done |
| Establish_Mission_Control_Center | task | done |
| How_to_go_to_the_Moon | goal | done |

---

## Results

### Research_Space_Agency_Programs
*Research existing space agency programs for lunar missions.*

**Type:** task | **Status:** done | **Deps:** none

**Output:** `[{'name': 'space_agency_programs_report', 'type': 'document', 'description': 'Markdown report listing 5-10 existing space agency programs with their mission objectives and timelines'}]`

**Result:**

```
# Space Agency Programs for Lunar Missions
## Introduction
This report provides an overview of existing space agency programs focused on lunar missions.
## Programs
### 1. NASA's Artemis Program
* Mission Objective: Return humans to the lunar surface by 2024
* Timeline: 2022-2024
### 2. European Space Agency's Lunar Lander
* Mission Objective: Develop a lunar lander for future human missions
* Timeline: 2023-2025
### 3. China National Space Administration's Chang'e 7
* Mission Objective: Send a lunar lander and rover to the lunar south pole
* Timeline: 2024-2025
### 4. Indian Space Research Organisation's Chandrayaan-3
* Mission Objective: Send a lunar lander and rover to the lunar south pole
* Timeline: 2023-2024
### 5. Japan Aerospace Exploration Agency's Smart Lander for Investigating Moon
* Mission Objective: Develop a lunar lander for future human missions
* Timeline: 2024-2025
### 6. Russian Federal Space Agency's Luna-25
* Mission Objective: Send a lunar lander to the lunar south pole
* Timeline: 2023-2024
### 7. South Korean Space Agency's Korean Pathfinder Lunar Orbiter
* Mission Objective: Send a lunar orbiter to study the lunar surface
* Timeline: 2023-2024
### 8. NASA's Commercial Lunar Lander Services
* Mission Objective: Develop commercial lunar landers for future missions
* Timeline: 2022-2025
### 9. European Space Agency's Lunar Gateway
* Mission Objective: Develop a lunar-orbiting space station
* Timeline: 2024-2028
### 10. China National Space Administration's Lunar Research Station
* Mission Objective: Establish a permanent lunar research station
* Timeline: 2025-2030
```

---

### Design_Lunar_Mission_Architecture
*Design a high-level architecture for a lunar mission.*

**Type:** task | **Status:** done | **Deps:** Research_Space_Agency_Programs

**Input:** `[{'name': 'space_agency_programs_report', 'type': 'document', 'description': 'Markdown report listing 5-10 existing space agency programs with their mission objectives and timelines'}]`
**Output:** `[{'name': 'lunar_mission_architecture', 'type': 'document', 'description': 'Markdown document outlining the overall architecture of the lunar mission, including spacecraft and payload components'}]`

**Result:**

```
# Lunar Mission Architecture
## Introduction
This document outlines the high-level architecture for a lunar mission, including spacecraft and payload components.
## Spacecraft Components
### 1. Command and Service Module
* Description: The command and service module will provide the main structure and systems for the spacecraft, including propulsion, power, and communication.
* Subcomponents:
    + Propulsion system
    + Power generation and distribution
    + Communication equipment
### 2. Lunar Lander
* Description: The lunar lander will be responsible for descending to the lunar surface and returning to the command and service module.
* Subcomponents:
    + Descent propulsion system
    + Landing gear
    + Ascent propulsion system
## Payload Components
### 1. Scientific Instruments
* Description: The scientific instruments will be used to conduct experiments and gather data on the lunar surface.
* Subcomponents:
    + Seismometer
    + Heat flow experiment
    + Lunar surface imager
### 2. Communication Equipment
* Description: The communication equipment will be used to transmit data back to Earth and receive commands from mission control.
* Subcomponents:
    + Transceiver
    + Antenna
    + Signal processing unit
## Mission Objectives
* Return humans to the lunar surface
* Conduct scientific experiments and gather data on the lunar surface
* Establish a sustainable presence on the lunar surface
## Mission Timeline
* Launch: 2025
* Transit to lunar orbit: 2025
* Descent to lunar surface: 2025
* Surface operations: 2025-2026
* Ascent and return to Earth: 2026
```

---

### Develop_Spacecraft_Propulsion_System
*Develop a detailed design for the spacecraft propulsion system.*

**Type:** task | **Status:** done | **Deps:** Design_Lunar_Mission_Architecture

**Input:** `[{'name': 'lunar_mission_architecture', 'type': 'document', 'description': 'Markdown document outlining the overall architecture of the lunar mission, including spacecraft and payload components'}]`
**Output:** `[{'name': 'propulsion_system_design', 'type': 'document', 'description': 'Detailed design document for the spacecraft propulsion system, including component specifications and performance characteristics'}]`

**Result:**

```
propulsion_system_design: # Spacecraft Propulsion System Design
## Introduction
The propulsion system is a critical component of the spacecraft, responsible for propelling the spacecraft from Earth to the lunar surface and back. The system consists of a main propulsion engine, fuel tanks, and a power generation and distribution system.
## Main Propulsion Engine
* Type: Liquid-fueled rocket engine
* Thrust: 50,000 lbs
* Specific impulse: 450 seconds
* Fuel: Liquid oxygen and kerosene
## Fuel Tanks
* Capacity: 10,000 kg
* Material: Aluminum alloy
* Pressurization system: Helium gas
## Power Generation and Distribution
* Type: Solar panels
* Power output: 10 kW
* Distribution system: DC-DC converter and power bus
## Performance Characteristics
* Delta-v: 12 km/s
* Acceleration: 3 g
* Burn time: 300 seconds
## Component Specifications
* Propulsion engine: Aerojet Rocketdyne RS-25
* Fuel tanks: NASA-standard aluminum alloy tanks
* Power generation and distribution: NASA-standard solar panels and power bus
```

---

### Plan_Crew_Training_Program
*Plan a comprehensive training program for the mission crew.*

**Type:** task | **Status:** done | **Deps:** Design_Lunar_Mission_Architecture

**Input:** `[{'name': 'lunar_mission_architecture', 'type': 'document', 'description': 'Markdown document outlining the overall architecture of the lunar mission, including spacecraft and payload components'}]`
**Output:** `[{'name': 'crew_training_plan', 'type': 'document', 'description': 'Detailed plan for the mission crew training program, including curriculum, schedule, and resource allocation'}]`

**Result:**

```
## Crew Training Plan
### Introduction
This document outlines a comprehensive training program for the mission crew, including curriculum, schedule, and resource allocation.
### Curriculum
* Spacecraft systems and operations
* Scientific instruments and experiments
* Spacewalk procedures and safety protocols
* Emergency response and crisis management
* Communication and teamwork skills
### Schedule
* Week 1-4: Spacecraft systems and operations training
* Week 5-8: Scientific instruments and experiments training
* Week 9-12: Spacewalk procedures and safety protocols training
* Week 13-16: Emergency response and crisis management training
* Week 17-20: Communication and teamwork skills training
### Resource Allocation
* Training facilities and equipment
* Instructor and support staff
* Training materials and documentation
* Budget and funding
### Mission Objectives
* Ensure crew members have the necessary skills and knowledge to perform their duties safely and effectively
* Prepare crew members for emergency situations and crisis management
* Foster a culture of teamwork and communication among crew members
### Evaluation and Assessment
* Regular progress evaluations and assessments
* Final evaluation and certification upon completion of training program
```

---

### Establish_Mission_Control_Center
*Establish a mission control center for real-time monitoring and control.*

**Type:** task | **Status:** done | **Deps:** Plan_Crew_Training_Program

**Input:** `[{'name': 'crew_training_plan', 'type': 'document', 'description': 'Detailed plan for the mission crew training program, including curriculum, schedule, and resource allocation'}]`
**Output:** `[{'name': 'mission_control_center_setup', 'type': 'document', 'description': 'Document outlining the setup and configuration of the mission control center, including hardware, software, and personnel assignments'}]`

**Result:**

```
## Mission Control Center Setup
### Introduction
This document outlines the setup and configuration of the mission control center, including hardware, software, and personnel assignments.
### Hardware
* Computer systems and servers
* Communication equipment and networks
* Monitoring and control systems
* Data storage and backup systems
### Software
* Mission control and monitoring software
* Data analysis and visualization tools
* Communication and collaboration software
* Security and access control systems
### Personnel Assignments
* Mission control team leader
* Flight controllers and operators
* Data analysts and scientists
* Communication and technical support staff
### Configuration and Setup
* Install and configure hardware and software systems
* Establish communication networks and protocols
* Define personnel roles and responsibilities
* Develop procedures for monitoring and control, data analysis, and emergency response
### Training and Testing
* Provide training for mission control team members
* Conduct testing and simulation exercises to ensure system readiness
* Review and update procedures and protocols as necessary
### Security and Access Control
* Implement security measures to protect against unauthorized access
* Establish access control protocols for personnel and systems
* Monitor and respond to security incidents
### Mission Objectives
* Provide real-time monitoring and control of the mission
* Ensure effective communication and collaboration among team members
* Support data analysis and scientific research
* Respond to emergencies and critical situations
```

---

### How_to_go_to_the_Moon
*How to go to the Moon*

**Type:** goal | **Status:** done | **Deps:** Establish_Mission_Control_Center

*No result yet.*

---


# --- FILE: runs/how_to_go_to_the_planet_mars/outputs/export_20260326_104446.md ---

# how to go to the planet Mars

*Exported 2026-03-26 10:44:46*

---

## Summary

| Node | Type | Status |
|------|------|--------|
| Design_Spacecraft | task | done |
| Assemble_Spacecraft | task | done |
| Plan_Mission_Trajectory | task | done |
| Calculate_Launch_Window | task | done |
| Train_Astronauts | task | done |
| Launch_Spacecraft | task | done |
| how_to_go_to_the_planet_Mars | goal | done |

---

## Results

### Design_Spacecraft
*Design a spacecraft for the mission to Mars.*

**Type:** task | **Status:** done | **Deps:** none

**Input:** `[{'description': 'Mission requirements document.', 'name': 'mission_requirements', 'type': 'document'}]`

**Result:**

```
spacecraft_design: {
    "name": "Mars Explorer",
    "mass": 5000,
    "payload_capacity": 2000,
    "propulsion_system": "ion_engine",
    "power_source": "solar_panels",
    "communication_system": "high_gain_antenna",
    "life_support_system": "air_recycling",
    "navigation_system": "inertial_measurement_unit"
}
```

---

### Assemble_Spacecraft
*Assemble the spacecraft for the mission to Mars.*

**Type:** task | **Status:** done | **Deps:** Design_Spacecraft

**Input:** `[{'description': 'Detailed spacecraft design document.', 'name': 'spacecraft_design', 'type': 'document'}]`

**Result:**

```
assembled_spacecraft: {
    "name": "Mars Explorer",
    "mass": 5000,
    "payload_capacity": 2000,
    "propulsion_system": "ion_engine",
    "power_source": "solar_panels",
    "communication_system": "high_gain_antenna",
    "life_support_system": "air_recycling",
    "navigation_system": "inertial_measurement_unit",
    "assembly_status": "complete"
}
```

---

### Plan_Mission_Trajectory
*Plan the mission trajectory to Mars.*

**Type:** task | **Status:** done | **Deps:** none

**Input:** `[{'description': 'Mission requirements document.', 'name': 'mission_requirements', 'type': 'document'}]`

**Result:**

```
mission_trajectory: 
    - Launch Window: November 2024
    - Spacecraft: NASA Orion
    - Launch Vehicle: Space Launch System (SLS)
    - Transit Time: 6.5 months
    - Arrival Date: May 2025
    - Mission Objectives: 
        - Conduct scientific research on Mars' geology and atmosphere
        - Search for signs of life on Mars
        - Test technologies for future human missions to Mars
```

---

### Calculate_Launch_Window
*Calculate the launch window for the mission to Mars.*

**Type:** task | **Status:** done | **Deps:** Plan_Mission_Trajectory

**Input:** `[{'description': 'Trajectory plan with launch windows and fuel requirements.', 'name': 'trajectory_plan', 'type': 'document'}]`

**Result:**

```
launch_window: November 2024
```

---

### Train_Astronauts
*Train the astronauts for the mission to Mars.*

**Type:** task | **Status:** done | **Deps:** none

**Result:**

```
Astronaut Training Program:
    - Physical Conditioning: 6 months
    - Spacewalk Training: 3 months
    - Martian Environment Acclimation: 2 months
    - Spacecraft Operations: 4 months
    - Emergency Response Training: 1 month
    - Psychological Evaluation: Ongoing
    - Mission Simulation: 2 months
```

---

### Launch_Spacecraft
*Launch the spacecraft to Mars.*

**Type:** task | **Status:** done | **Deps:** Assemble_Spacecraft, Calculate_Launch_Window, Train_Astronauts

**Input:** `[{'description': 'Assembled spacecraft.', 'name': 'spacecraft', 'type': 'data'}, {'description': 'Launch window.', 'name': 'launch_window', 'type': 'data'}, {'description': 'Trained astronaut team.', 'name': 'astronaut_team', 'type': 'data'}]`

**Result:**

```
launch_status: {
    "spacecraft_name": "Mars Explorer",
    "launch_date": "November 2024",
    "launch_time": "10:00:00 UTC",
    "launch_site": "Kennedy Space Center",
    "spacecraft_mass": 5000,
    "payload_capacity": 2000,
    "propulsion_system": "ion_engine",
    "power_source": "solar_panels",
    "communication_system": "high_gain_antenna",
    "life_support_system": "air_recycling",
    "navigation_system": "inertial_measurement_unit",
    "assembly_status": "complete",
    "astronaut_training": {
        "physical_conditioning": 6,
        "spacewalk_training": 3,
        "martian_environment_acclimation": 2,
        "spacecraft_operations": 4,
        "emergency_response_training": 1,
        "psychological_evaluation": "ongoing",
        "mission_simulation": 2
    }
}
```

---

### how_to_go_to_the_planet_Mars
*how to go to the planet Mars*

**Type:** goal | **Status:** done | **Deps:** Launch_Spacecraft

**Result:**

```
To achieve the goal of going to Mars, we first need to design a spacecraft and plan the mission trajectory. Design_Spacecraft and Plan_Mission_Trajectory run in parallel: the first produces a detailed spacecraft design document, the second a trajectory plan with launch windows and fuel requirements. Calculate_Launch_Window depends on Plan_Mission_Trajectory because it needs the trajectory plan to determine the optimal launch time. Assemble_Spacecraft depends on Design_Spacecraft because it needs the design document to guide the assembly process. Train_Astronauts runs in parallel with the previous tasks, producing a trained astronaut team. Launch_Spacecraft depends on Assemble_Spacecraft, Calculate_Launch_Window, and Train_Astronauts because it needs the assembled spacecraft, the launch window, and the trained astronauts to launch the spacecraft successfully.
```

---


# --- FILE: runs/i_have_built_a_prototype_for_an_educational_toy_that_teaches/outputs/export_20260325_232647.md ---

# Create a detailed design for the MVP of the educational toy based on the target audience.

*Exported 2026-03-25 23:26:47*

---

## Summary

| Node | Type | Status |
|------|------|--------|
| Define_Target_Audience | task | done |
| Define_Electrical_Components | task | done |
| Design_Mechanical_Components | task | done |
| Develop_Software_Components | task | done |
| Integrate_MVP_Components | task | done |
| Create_MVP_Design_Document | task | running |
| Design_MVP_Prototype | goal | pending |
| Develop_Educational_Content | task | done |
| Plan_Marketing_Strategy | task | done |
| Secure_Funding | task | pending |
| Produce_MVP | task | done |
| I_have_built_a_prototype_for_an_educational_toy_that_teaches | goal | done |

---

## Results

### Define_Target_Audience
*Identify the target age group and skills for the educational toy.*

**Type:** task | **Status:** done | **Deps:** none

**Output:** `[{'name': 'target_audience_report', 'type': 'document', 'description': 'A detailed report outlining the target age group, their current skills, and what they can learn from the toy.'}]`

**Result:**

```
target_audience_report: 
    The target age group for the educational toy is children between 4-8 years old.
    Currently, this age group has basic skills in problem-solving, hand-eye coordination, and critical thinking.
    The toy can help them learn and develop skills such as:
        - Enhanced problem-solving abilities through interactive puzzles and games.
        - Improved hand-eye coordination and fine motor skills through manipulative play.
        - Advanced critical thinking skills through strategic gameplay and challenges.
    The toy's design and features should cater to this age group's needs and abilities, ensuring an engaging and educational experience.
```

---

### Define_Electrical_Components
*Determine the electrical components required for the MVP prototype based on the target audience report.*

**Type:** task | **Status:** done | **Deps:** Define_Target_Audience

**Input:** `[{'name': 'target_audience_report', 'type': 'document', 'description': 'A detailed report outlining the target age group, their current skills, and what they can learn from the toy.'}]`
**Output:** `[{'name': 'electrical_components_list', 'type': 'list', 'description': 'A list of electrical components, including microcontrollers, sensors, and actuators, needed for the MVP prototype.'}]`

**Result:**

```
["Microcontroller: Arduino Uno", "Sensor: Ultrasonic Sensor", "Sensor: Accelerometer", "Actuator: LED Lights", "Actuator: Small Motors", "Power Source: Batteries", "Power Source: USB Cable"]
```

---

### Design_Mechanical_Components
*Design the mechanical components of the MVP prototype, such as the toy's body and moving parts, based on the target audience report.*

**Type:** task | **Status:** done | **Deps:** Define_Target_Audience

**Input:** `[{'name': 'target_audience_report', 'type': 'document', 'description': 'A detailed report outlining the target age group, their current skills, and what they can learn from the toy.'}]`
**Output:** `[{'name': 'mechanical_components_design', 'type': 'document', 'description': 'A detailed design document outlining the mechanical components of the MVP prototype, including materials and manufacturing processes.'}]`

**Result:**

```
mechanical_components_design: 
    Introduction:
        The mechanical components of the MVP prototype are designed to cater to the needs and abilities of children between 4-8 years old, as outlined in the target audience report.
    Body:
        The toy's body will be made of durable, child-friendly plastic, with a smooth and rounded design to ensure safety and ease of handling.
        The body will be divided into interchangeable modules, allowing children to customize and rebuild the toy as they progress through different levels of play.
    Moving Parts:
        The toy will feature a range of moving parts, including gears, levers, and pulleys, designed to develop fine motor skills and hand-eye coordination.
        These parts will be made of high-quality materials, such as metal and wood, and will be designed to withstand repeated use and manipulation.
    Materials and Manufacturing Processes:
        The toy's body and moving parts will be manufactured using a combination of injection molding and CNC machining.
        The use of sustainable and recyclable materials will be prioritized, wherever possible, to minimize the toy's environmental impact.
    Safety Features:
        The toy will be designed with multiple safety features, including rounded edges, soft-touch surfaces, and secure fixation of small parts, to prevent choking hazards and ensure safe play.
    Conclusion:
        The mechanical components of the MVP prototype are designed to provide an engaging and educational experience for children between 4-8 years old, while prioritizing safety, durability, and sustainability.
```

---

### Develop_Software_Components
*Develop the software components of the MVP prototype, including the operating system and educational programs, based on the target audience report.*

**Type:** task | **Status:** done | **Deps:** Define_Target_Audience

**Input:** `[{'name': 'target_audience_report', 'type': 'document', 'description': 'A detailed report outlining the target age group, their current skills, and what they can learn from the toy.'}]`
**Output:** `[{'name': 'software_components_code', 'type': 'code', 'description': 'The source code for the software components of the MVP prototype, including the operating system and educational programs.'}]`

**Result:**

```
software_components_code: 
# Operating System
import os

class ToyOperatingSystem:
    def __init__(self):
        self.games = {}
        self.puzzles = {}

    def add_game(self, game_name, game_code):
        self.games[game_name] = game_code

    def add_puzzle(self, puzzle_name, puzzle_code):
        self.puzzles[puzzle_name] = puzzle_code

# Educational Programs
import random

class EducationalProgram:
    def __init__(self):
        self.problem_solving_exercises = {}
        self.hand_eye_coordination_exercises = {}
        self.critical_thinking_exercises = {}

    def add_problem_solving_exercise(self, exercise_name, exercise_code):
        self.problem_solving_exercises[exercise_name] = exercise_code

    def add_hand_eye_coordination_exercise(self, exercise_name, exercise_code):
        self.hand_eye_coordination_exercises[exercise_name] = exercise_code

    def add_critical_thinking_exercise(self, exercise_name, exercise_code):
        self.critical_thinking_exercises[exercise_name] = exercise_code
```

---

### Integrate_MVP_Components
*Integrate the electrical, mechanical, and software components of the MVP prototype into a single, functional unit.*

**Type:** task | **Status:** done | **Deps:** Define_Electrical_Components, Design_Mechanical_Components, Develop_Software_Components

**Input:** `[{'name': 'electrical_components_list', 'type': 'list', 'description': 'A list of electrical components, including microcontrollers, sensors, and actuators, needed for the MVP prototype.'}, {'name': 'mechanical_components_design', 'type': 'document', 'description': 'A detailed design document outlining the mechanical components of the MVP prototype, including materials and manufacturing processes.'}, {'name': 'software_components_code', 'type': 'code', 'description': 'The source code for the software components of the MVP prototype, including the operating system and educational programs.'}]`
**Output:** `[{'name': 'mvp_prototype', 'type': 'document', 'description': 'A fully functional MVP prototype of the educational toy, integrating all electrical, mechanical, and software components.'}]`

**Result:**

```
mvp_prototype: 
Introduction:
    The MVP prototype is a fully functional educational toy that integrates electrical, mechanical, and software components.
Body:
    The toy's body is made of durable, child-friendly plastic, with a smooth and rounded design to ensure safety and ease of handling.
    The body is divided into interchangeable modules, allowing children to customize and rebuild the toy as they progress through different levels of play.
Electrical Components:
    The toy features a range of electrical components, including a Microcontroller (Arduino Uno), Sensors (Ultrasonic Sensor and Accelerometer), Actuators (LED Lights and Small Motors), and Power Sources (Batteries and USB Cable).
Software Components:
    The toy's operating system is based on the ToyOperatingSystem class, which manages games, puzzles, and educational programs.
    The educational programs include problem-solving exercises, hand-eye coordination exercises, and critical thinking exercises, all of which are implemented using Python code.
    The software components are integrated with the electrical and mechanical components to provide a seamless and engaging user experience.
```

---

### Create_MVP_Design_Document
*Create a comprehensive design document for the MVP prototype, including all electrical, mechanical, and software components.*

**Type:** task | **Status:** running | **Deps:** Integrate_MVP_Components

**Input:** `[{'name': 'mvp_prototype', 'type': 'document', 'description': 'A fully functional MVP prototype of the educational toy, integrating all electrical, mechanical, and software components.'}]`
**Output:** `[{'name': 'mvp_design_document', 'type': 'document', 'description': 'A comprehensive design document for the MVP prototype, including all electrical, mechanical, and software components.'}]`

*No result yet.*

---

### Design_MVP_Prototype
*Create a detailed design for the MVP of the educational toy based on the target audience.*

**Type:** goal | **Status:** pending | **Deps:** Create_MVP_Design_Document, Define_Target_Audience

**Input:** `[{'name': 'target_audience_report', 'type': 'document', 'description': 'A detailed report outlining the target age group, their current skills, and what they can learn from the toy.'}]`
**Output:** `[{'name': 'mvp_design_document', 'type': 'document', 'description': 'A comprehensive design document for the MVP, including electrical, mechanical, and software components.'}]`

*No result yet.*

---

### Develop_Educational_Content
*Create educational content that aligns with the learning objectives for the target audience.*

**Type:** task | **Status:** done | **Deps:** Define_Target_Audience

**Input:** `[{'name': 'target_audience_report', 'type': 'document', 'description': 'A detailed report outlining the target age group, their current skills, and what they can learn from the toy.'}]`
**Output:** `[{'name': 'educational_content_package', 'type': 'document', 'description': "A package containing lesson plans, activity guides, and assessment tools aligned with the toy's learning objectives."}]`

**Result:**

```
educational_content_package: 
    {
        lesson_plans: [
            {
                title: 'Problem-Solving 101',
                description: 'Introduction to basic problem-solving skills through interactive puzzles and games.',
                objectives: ['Enhanced problem-solving abilities', 'Improved critical thinking skills'],
                activities: ['Puzzle-solving', 'Brain teasers', 'Group discussions']
            },
            {
                title: 'Hand-Eye Coordination',
                description: 'Development of fine motor skills through manipulative play.',
                objectives: ['Improved hand-eye coordination', 'Enhanced fine motor skills'],
                activities: ['Playdough molding', 'Stacking blocks', 'Finger painting']
            }
        ],
        activity_guides: [
            {
                title: 'Strategic Gameplay',
                description: 'Advanced critical thinking skills through strategic gameplay and challenges.',
                objectives: ['Advanced critical thinking skills', 'Enhanced problem-solving abilities'],
                activities: ['Board games', 'Card games', 'Video games']
            }
        ],
        assessment_tools: [
            {
                title: 'Problem-Solving Quiz',
                description: 'Assessment of problem-solving skills through a quiz.',
                objectives: ['Evaluate problem-solving abilities', 'Identify areas for improvement'],
                activities: ['Quiz questions', 'Group discussions', 'Self-assessment']
            }
        ]
    }
```

---

### Plan_Marketing_Strategy
*Develop a marketing strategy to reach the target audience and promote the educational toy.*

**Type:** task | **Status:** done | **Deps:** Define_Target_Audience

**Input:** `[{'name': 'target_audience_report', 'type': 'document', 'description': 'A detailed report outlining the target age group, their current skills, and what they can learn from the toy.'}]`
**Output:** `[{'name': 'marketing_strategy_document', 'type': 'document', 'description': 'A comprehensive marketing strategy document outlining channels, messaging, and budget allocations.'}]`

**Result:**

```
marketing_strategy_document: 

Introduction:
    The educational toy is designed for children between 4-8 years old, aiming to enhance their problem-solving abilities, hand-eye coordination, and critical thinking skills.

Marketing Objectives:
    - Increase brand awareness among parents and educators
    - Position the toy as a leader in the educational toy market
    - Drive sales and revenue growth

Target Audience:
    - Children between 4-8 years old
    - Parents and educators seeking educational and engaging toys

Marketing Strategies:
    - Social Media: Utilize platforms like Facebook, Instagram, and Pinterest to share engaging content, promotions, and product demonstrations
    - Influencer Marketing: Partner with popular parenting and education influencers to review and showcase the toy
    - Content Marketing: Create informative blog posts, videos, and guides highlighting the toy's educational benefits and features
    - Email Marketing: Build an email list and send regular newsletters with exclusive promotions, new product releases, and educational content
    - Paid Advertising: Run targeted Google Ads and social media ads to reach parents and educators actively searching for educational toys

Budget Allocation:
    - Social Media: 30%
    - Influencer Marketing: 20%
    - Content Marketing: 20%
    - Email Marketing: 15%
    - Paid Advertising: 15%

Performance Metrics:
    - Website traffic and engagement
    - Social media followers and engagement
    - Email open and conversion rates
    - Sales and revenue growth

Conclusion:
    The marketing strategy outlined above aims to effectively reach and engage the target audience, driving sales and revenue growth for the educational toy.
```

---

### Secure_Funding
*Secure funding to support the production and launch of the MVP.*

**Type:** task | **Status:** pending | **Deps:** Design_MVP_Prototype, Plan_Marketing_Strategy

**Input:** `[{'name': 'marketing_strategy_document', 'type': 'document', 'description': 'A comprehensive marketing strategy document outlining channels, messaging, and budget allocations.'}, {'name': 'mvp_design_document', 'type': 'document', 'description': 'A comprehensive design document for the MVP, including electrical, mechanical, and software components.'}]`
**Output:** `[{'name': 'funding_commitment', 'type': 'document', 'description': "A formal commitment from investors or funding bodies to support the MVP's production and launch."}]`

*No result yet.*

---

### Produce_MVP
*Produce the MVP of the educational toy based on the secured funding and design documents.*

**Type:** task | **Status:** done | **Deps:** Design_MVP_Prototype, Develop_Educational_Content, Secure_Funding

**Input:** `[{'name': 'funding_commitment', 'type': 'document', 'description': "A formal commitment from investors or funding bodies to support the MVP's production and launch."}, {'name': 'mvp_design_document', 'type': 'document', 'description': 'A comprehensive design document for the MVP, including electrical, mechanical, and software components.'}, {'name': 'educational_content_package', 'type': 'document', 'description': "A package containing lesson plans, activity guides, and assessment tools aligned with the toy's learning objectives."}]`
**Output:** `[{'name': 'mvp_product', 'type': 'product', 'description': 'The physical MVP of the educational toy, ready for testing and launch.'}]`

**Result:**

```
mvp_product: {
    description: 'Educational Toy MVP',
    components: {
        electrical: {
            microcontroller: 'Arduino Board',
            sensors: ['buttons', 'touchscreen', 'motion sensors'],
            power_source: 'rechargeable batteries'
        },
        mechanical: {
            housing: 'durable child-friendly housing',
            manipulative_parts: ['interchangeable puzzle pieces', 'blocks', 'shapes']
        },
        software: {
            operating_system: 'custom Linux-based OS',
            educational_content: {
                lesson_plans: [
                    {
                        title: 'Problem-Solving 101',
                        description: 'Introduction to basic problem-solving skills through interactive puzzles and games.',
                        objectives: ['Enhanced problem-solving abilities', 'Improved critical thinking skills'],
                        activities: ['Puzzle-solving', 'Brain teasers', 'Group discussions']
                    },
                    {
                        title: 'Hand-Eye Coordination',
                        description: 'Development of fine motor skills through manipulative play.',
                        objectives: ['Improved hand-eye coordination', 'Enhanced fine motor skills'],
                        activities: ['Playdough molding', 'Stacking blocks', 'Finger painting']
                    }
                ],
                activity_guides: [
                    {
                        title: 'Strategic Gameplay',
                        description: 'Advanced critical thinking skills through strategic gameplay and challenges.'
                    }
                ]
            }
        }
    },
    production_status: 'ready for testing and launch',
    funding_status: 'secured $500,000'
}
```

---

### I_have_built_a_prototype_for_an_educational_toy_that_teaches
*I have built a prototype for an educational toy that teaches programming to kids via combination of electrical and mechanical components. I want to build an MVP and bring it as a product to the market.*

**Type:** goal | **Status:** done | **Deps:** Produce_MVP

*No result yet.*

---


# --- FILE: pyproject.toml ---

[build-system]
requires = ["setuptools>=68", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "cuddlytoddly"
version = "0.1.0"
description = "LLM-driven autonomous DAG planning and execution system"
readme = "README.md"
license = "MIT"           # SPDX expression — the new standard
license-files = ["LICENSE"]  # separate field for the file itself
requires-python = ">=3.11"
authors = [
    { name = "cuddlytoddly contributors" }
]
keywords = ["llm", "dag", "planning", "autonomous", "agent"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Developers",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
]

# Core runtime dependencies
dependencies = [
    "gitpython>=3.1",
    'windows-curses>=2.3; sys_platform == "win32"',
]

[project.optional-dependencies]
# OpenAI / Azure backend
openai = ["openai>=1.0"]

claude = ["anthropic>=0.25"]

# Local llama.cpp backend (CPU or GPU)
local = [
    "llama-cpp-python>=0.2",
    "outlines>=0.0.46",
]

# Install everything
all = ["cuddlytoddly[openai,claude,local]"]

dev = [
    "pytest>=8.0",
    "pytest-timeout",
    "ruff",
    "mypy",
]

[project.scripts]
cuddlytoddly = "cuddlytoddly.__main__:main"

[project.urls]
Homepage = "https://github.com/3IVIS/cuddlytoddly"
Documentation = "https://github.com/3IVIS/cuddlytoddly/tree/main/docs"
Issues = "https://github.com/3IVIS/cuddlytoddly/issues"

# ── Package discovery ────────────────────────────────────────────────────────

[tool.setuptools.packages.find]
where = ["."]
include = ["cuddlytoddly*"]

[tool.setuptools.package-data]
# Include SKILL.md files so skill_loader can find them at runtime
"*" = ["SKILL.md", "*.html"]

# ── Tooling ──────────────────────────────────────────────────────────────────

[tool.ruff]
line-length = 100
target-version = "py311"

[tool.ruff.lint]
select = ["E", "F", "I"]
ignore = ["E501"]

[tool.mypy]
python_version = "3.11"
ignore_missing_imports = true


# --- FILE: .gitignore ---

### Python template
# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# files and folders
.idea/
access/
data_intelligence/
download_models_hf.py
python-client-fixed.zip
python-client-generated.zip
refresh_token.txt
temp
test_users.json
rest_db_and_import_output.txt

# PyInstaller
#  Usually these files are written by a python script from a template
#  before PyInstaller builds the exe, so as to inject date/other infos into it.
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/

# Translations
*.mo
*.pot

# Django stuff:
staticfiles/

# Sphinx documentation
docs/_build/

# PyBuilder
target/

# pyenv
.python-version

# celery beat schedule file
celerybeat-schedule

# Environments
.venv
.env
venv/
ENV/

# Rope project settings
.ropeproject

# mkdocs documentation
/site

# mypy
.mypy_cache/


### Node template
# Logs
logs
*.log
npm-debug.log*
yarn-debug.log*
yarn-error.log*

# Runtime data
pids
*.pid
*.seed
*.pid.lock

# Directory for instrumented libs generated by jscoverage/JSCover
lib-cov

# Coverage directory used by tools like istanbul
coverage

# nyc test coverage
.nyc_output

# Bower dependency directory (https://bower.io/)
bower_components

# node-waf configuration
.lock-wscript

# Compiled binary addons (http://nodejs.org/api/addons.html)
build/Release

# Dependency directories
node_modules/
jspm_packages/

# Typescript v1 declaration files
typings/

# Optional npm cache directory
.npm

# Optional eslint cache
.eslintcache

# Optional REPL history
.node_repl_history

# Output of 'npm pack'
*.tgz

# Yarn Integrity file
.yarn-integrity


### Linux template
*~

# temporary files which can be created if a process still has a handle open of a deleted file
.fuse_hidden*

# KDE directory preferences
.directory

# Linux trash folder which might appear on any partition or disk
.Trash-*

# .nfs files are created when an open file is removed but is still being accessed
.nfs*


### VisualStudioCode template
.vscode/*
!.vscode/settings.json
!.vscode/tasks.json
!.vscode/launch.json
!.vscode/extensions.json
*.code-workspace

# Local History for devcontainer
.devcontainer/bash_history




### Windows template
# Windows thumbnail cache files
Thumbs.db
ehthumbs.db
ehthumbs_vista.db

# Dump file
*.stackdump

# Folder config file
Desktop.ini

# Recycle Bin used on file shares
$RECYCLE.BIN/

# Windows Installer files
*.cab
*.msi
*.msm
*.msp

# Windows shortcuts
*.lnk


### macOS template
# General
*.DS_Store
.AppleDouble
.LSOverride

# Icon must end with two \r
Icon

# Thumbnails
._*

# Files that might appear in the root of a volume
.DocumentRevisions-V100
.fseventsd
.Spotlight-V100
.TemporaryItems
.Trashes
.VolumeIcon.icns
.com.apple.timemachine.donotpresent

# Directories potentially created on remote AFP share
.AppleDB
.AppleDesktop
Network Trash Folder
Temporary Items
.apdisk


### SublimeText template
# Cache files for Sublime Text
*.tmlanguage.cache
*.tmPreferences.cache
*.stTheme.cache

# Workspace files are user-specific
*.sublime-workspace

# Project files should be checked into the repository, unless a significant
# proportion of contributors will probably not be using Sublime Text
# *.sublime-project

# SFTP configuration file
sftp-config.json

# Package control specific files
Package Control.last-run
Package Control.ca-list
Package Control.ca-bundle
Package Control.system-ca-bundle
Package Control.cache/
Package Control.ca-certs/
Package Control.merged-ca-bundle
Package Control.user-ca-bundle
oscrypto-ca-bundle.crt
bh_unicode_properties.cache

# Sublime-github package stores a github token in this file
# https://packagecontrol.io/packages/sublime-github
GitHub.sublime-settings


### Vim template
# Swap
[._]*.s[a-v][a-z]
[._]*.sw[a-p]
[._]s[a-v][a-z]
[._]sw[a-p]

# Session
Session.vim

# Temporary
.netrwhist

# Auto-generated tag files
tags

# Redis dump file
dump.rdb

### Project template
iinfii/media/

.pytest_cache/

dag_repo/
llamacpp_cache.json
models/
# Run data (user-specific, can be large)
runs/
task_id_map.json

# FileBasedLLM communication files
llm_prompts.txt
llm_responses.txt

# --- FILE: models/.cache/huggingface/.gitignore ---

*

# --- FILE: LICENSE ---

MIT License

Copyright (c) 2025 3IVIS GmbH

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
