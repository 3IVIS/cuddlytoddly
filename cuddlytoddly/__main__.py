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
