# __main__.py

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
from cuddlytoddly.config import (
    load_config, DATA_DIR, resolve_model_path, preflight_check,
    get_executor_cfg, get_planner_cfg, get_orchestrator_cfg, get_file_llm_cfg,
)
import cuddlytoddly.planning.llm_interface as llm_iface

REPO_ROOT = Path(__file__).resolve().parent   # package code location

setup_logging()
logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Deferred LLM — used by the web UI two-phase init so the DAG is shown
# immediately while the real model loads in the background.
# ---------------------------------------------------------------------------

class _DeferredLLM:
    """
    Placeholder LLM that appears stopped until a real client is attached.

    The orchestrator checks `is_stopped` before every LLM call.  While this
    object is stopped, all call attempts raise LLMStoppedError — identical to
    the user pressing the pause button — so no LLM work is attempted.

    Once `attach(real_llm)` is called (from the background LLM-loader thread)
    the deferred LLM becomes transparent: `is_stopped` reflects the real
    client's state, and `ask()` delegates straight through.
    """

    def __init__(self):
        self._real: object | None = None
        self._lock = threading.Lock()

    @property
    def is_stopped(self) -> bool:
        with self._lock:
            if self._real is None:
                return True
            return getattr(self._real, "is_stopped", False)

    def stop(self) -> None:
        with self._lock:
            if self._real is not None and hasattr(self._real, "stop"):
                self._real.stop()

    def resume(self) -> None:
        with self._lock:
            if self._real is not None and hasattr(self._real, "resume"):
                self._real.resume()

    def ask(self, prompt: str, schema=None) -> str:
        from cuddlytoddly.planning.llm_interface import LLMStoppedError
        with self._lock:
            real = self._real
        if real is None:
            raise LLMStoppedError(
                "LLM is still loading — execution will resume automatically"
            )
        return real.ask(prompt, schema=schema) if schema is not None else real.ask(prompt)

    def generate(self, prompt: str) -> str:
        return self.ask(prompt)

    def attach(self, real_llm) -> None:
        """Swap in the real client; from this point on the LLM is live."""
        with self._lock:
            self._real = real_llm
        logger.info("[DEFERRED LLM] Real LLM attached — execution enabled")


def make_run_dir(goal_text: str) -> Path:
    safe    = goal_text.lower().replace(" ", "_")
    safe    = "".join(c for c in safe if c.isalnum() or c == "_")[:60]
    run_dir = DATA_DIR / "runs" / safe
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "outputs").mkdir(exist_ok=True)
    return run_dir


def _print_preflight_issues(issues: list[dict]) -> None:
    errors   = [i for i in issues if i["level"] == "error"]
    warnings = [i for i in issues if i["level"] != "error"]

    if errors:
        print("\n  ✗ Configuration errors (will fail at runtime):", file=sys.stderr)
        for issue in errors:
            print(f"    • {issue['message']}", file=sys.stderr)
            if issue.get("fix"):
                print(f"      → {issue['fix']}", file=sys.stderr)
    if warnings:
        print("\n  ⚠ Configuration warnings:", file=sys.stderr)
        for issue in warnings:
            print(f"    • {issue['message']}", file=sys.stderr)
            if issue.get("fix"):
                print(f"      → {issue['fix']}", file=sys.stderr)
    print(file=sys.stderr)


def main():
    # ── Load config ───────────────────────────────────────────────────────────
    cfg        = load_config()
    server_cfg = cfg.get("server", {})

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    issues = preflight_check(cfg)
    if issues:
        for issue in issues:
            logger.warning("[PREFLIGHT] %s: %s — %s",
                           issue["level"].upper(), issue["message"],
                           issue.get("fix", ""))
        _print_preflight_issues(issues)

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
        default=server_cfg.get("host", "127.0.0.1"),
        help="Host for the web UI server (default: from config.toml).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=server_cfg.get("port", 8765),
        help="Port for the web UI server (default: from config.toml).",
    )
    parser.add_argument(
        "goal",
        nargs="*",
        help=(
            "Goal text to start immediately, skipping the startup screen. "
            "If omitted the startup screen is shown."
        ),
    )
    args    = parser.parse_args()
    use_web = not args.terminal

    # ── Build a bound init function that carries cfg through all call sites ───
    def _init(choice: StartupChoice, _use_web: bool = use_web,
              on_graph_ready=None):
        return _init_system(choice, _use_web, cfg,
                            on_graph_ready=on_graph_ready)

    # ── Startup screen ────────────────────────────────────────────────────────
    inline_goal = " ".join(args.goal).strip()

    if inline_goal:
        choice = StartupChoice(
            mode="new_goal",
            run_dir=make_run_dir(inline_goal).resolve(),
            goal_text=inline_goal,
            is_fresh=True,
        )
    elif use_web:
        choice = None
    else:
        try:
            choice = run_startup_curses(DATA_DIR, issues=issues)
        except SystemExit:
            return

    # ── Web UI with deferred init ─────────────────────────────────────────────
    if use_web and choice is None:
        run_web_ui(
            repo_root=DATA_DIR,
            init_fn=_init,
            cfg=cfg,
            host=args.host,
            port=args.port,
        )
        return

    # ── Eager init ────────────────────────────────────────────────────────────
    orchestrator, run_dir = _init(choice)

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
            repo_root=DATA_DIR,
            restart_fn=_init,
        )

    # ── Final log ─────────────────────────────────────────────────────────────
    snap = orchestrator.graph.get_snapshot()
    logger.info("=== Final graph state (%d nodes) ===", len(snap))
    for nid, n in snap.items():
        logger.info("  [%s] %s", n.status, nid)


def _init_system(choice: "StartupChoice", use_web: bool, cfg: dict,
                 on_graph_ready=None):
    """
    Build the full orchestrator from a StartupChoice and a loaded config dict.
    Returns (orchestrator, run_dir).

    All numeric tuning parameters are read from cfg so no source files need
    editing to adjust behaviour.
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

    # ── Git repo — per run ────────────────────────────────────────────────────
    import cuddlytoddly.ui.git_projection as git_proj
    git_proj.REPO_PATH = str(run_dir / "dag_repo")

    # ── Working directory — sandbox for file tools ────────────────────────────
    os.chdir(run_dir / "outputs")
    _logger.info("Working directory: %s", Path.cwd())

    llm_iface.id_gen = StableIDGenerator(
        mapping_file=run_dir / "task_id_map.json",
        id_length=6,
    )

    # ── Read config sections (with defaults for old configs) ──────────────────
    orch_cfg     = get_orchestrator_cfg(cfg)
    exec_cfg     = get_executor_cfg(cfg)
    planner_cfg  = get_planner_cfg(cfg)

    max_workers  = orch_cfg["max_workers"]
    max_turns    = orch_cfg["max_turns"]

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

    # ── LLM client ────────────────────────────────────────────────────────────
    use_deferred = (not fresh_start) and (on_graph_ready is not None)
    if use_deferred:
        deferred_llm = _DeferredLLM()
        shared_llm   = deferred_llm
    else:
        deferred_llm = None
        shared_llm   = _build_llm_client(cfg, run_dir)

    # ── Components ────────────────────────────────────────────────────────────
    skills   = SkillLoader()
    registry = skills.registry

    planner = LLMPlanner(
        llm_client=shared_llm,
        graph=graph,
        skills_summary=skills.prompt_summary,
        min_tasks_per_goal=planner_cfg["min_tasks_per_goal"],
        max_tasks_per_goal=planner_cfg["max_tasks_per_goal"],
    )

    executor = LLMExecutor(
        llm_client=shared_llm,
        tool_registry=registry,
        max_turns=max_turns,
        max_inline_result_chars=exec_cfg["max_inline_result_chars"],
        max_total_input_chars=exec_cfg["max_total_input_chars"],
        max_tool_result_chars=exec_cfg["max_tool_result_chars"],
        max_history_entries=exec_cfg["max_history_entries"],
    )

    quality_gate = QualityGate(llm_client=shared_llm, tool_registry=registry)

    queue        = EventQueue()
    orchestrator = SimpleOrchestrator(
        graph=graph,
        planner=planner,
        executor=executor,
        quality_gate=quality_gate,
        event_log=event_log,
        event_queue=queue,
        max_workers=max_workers,
        max_gap_fill_attempts=orch_cfg["max_gap_fill_attempts"],
        idle_sleep=orch_cfg["idle_sleep"],
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

    if use_deferred:
        on_graph_ready(orchestrator, run_dir)

        def _load_real_llm():
            _logger.info("[STARTUP] Background LLM load starting…")
            try:
                real_llm = _build_llm_client(cfg, run_dir)
                deferred_llm.attach(real_llm)
                _logger.info("[STARTUP] LLM ready — starting background verification")
                orchestrator.verify_restored_nodes()
            except Exception as exc:
                _logger.error("[STARTUP] Background LLM load failed: %s", exc)

        threading.Thread(target=_load_real_llm, daemon=True,
                         name="startup-llm").start()

    else:
        if not fresh_start:
            def _bg_verify():
                _logger.info("[STARTUP] Background verification pass starting...")
                orchestrator.verify_restored_nodes()
                _logger.info("[STARTUP] Background verification complete")

            threading.Thread(target=_bg_verify, daemon=True,
                             name="startup-verify").start()

    return orchestrator, run_dir


def _build_llm_client(cfg: dict, run_dir: Path):
    """
    Construct and return the correct BaseLLM from the loaded config.
    Extracted so it can be unit-tested independently of the full startup.
    """
    backend  = cfg["llm"]["backend"]           # already validated by load_config()
    llm_cfg  = cfg.get(backend, {})

    _logger = get_logger(__name__)
    _logger.info("[LLM] Backend: %s", backend)

    if backend == "llamacpp":
        model_path   = resolve_model_path(cfg)
        cache_path   = (
            str(run_dir / "llamacpp_cache.json")
            if llm_cfg.get("cache_enabled", True)
            else None
        )
        return create_llm_client(
            "llamacpp",
            model_path   = model_path,
            n_gpu_layers = llm_cfg.get("n_gpu_layers",  -1),
            n_ctx        = llm_cfg.get("n_ctx",         16384),
            max_tokens   = llm_cfg.get("max_tokens",    8192),
            temperature  = llm_cfg.get("temperature",   0.1),
            cache_path   = cache_path,
        )

    if backend == "claude":
        cache_path = (
            str(run_dir / "api_cache.json")
            if llm_cfg.get("cache_enabled", True)
            else None
        )
        return create_llm_client(
            "claude",
            model       = llm_cfg.get("model",       "claude-opus-4-6"),
            temperature = llm_cfg.get("temperature", 0.1),
            max_tokens  = llm_cfg.get("max_tokens",  8192),
            cache_path  = cache_path,
        )

    if backend == "openai":
        cache_path = (
            str(run_dir / "api_cache.json")
            if llm_cfg.get("cache_enabled", True)
            else None
        )
        kwargs: dict = dict(
            model       = llm_cfg.get("model",       "gpt-4o"),
            temperature = llm_cfg.get("temperature", 0.1),
            max_tokens  = llm_cfg.get("max_tokens",  8192),
            cache_path  = cache_path,
        )
        if "base_url" in llm_cfg:
            kwargs["base_url"] = llm_cfg["base_url"]
        if "api_key" in llm_cfg:
            kwargs["api_key"] = llm_cfg["api_key"]
        return create_llm_client("openai", **kwargs)

    if backend == "file":
        file_cfg   = get_file_llm_cfg(cfg)
        cache_path = (
            str(run_dir / "file_llm_cache.json")
            if file_cfg.get("cache_enabled", True)
            else None
        )
        return create_llm_client(
            "file",
            poll_interval         = file_cfg["poll_interval"],
            timeout               = file_cfg["timeout"],
            progress_log_interval = file_cfg["progress_log_interval"],
            cache_path            = cache_path,
        )

    # Should never reach here — _validate() in load_config() guards this.
    raise ValueError(f"Unknown backend: {backend!r}")


if __name__ == "__main__":
    main()