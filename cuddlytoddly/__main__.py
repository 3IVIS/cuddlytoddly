# __main__.py

import argparse
import json
import sys
import threading
from pathlib import Path

import cuddlytoddly.ui.git_projection as git_proj
import toddly.skills.file_ops.tools as file_ops_tools
from cuddlytoddly.config import (
    DATA_DIR,
    get_executor_cfg,
    get_orchestrator_cfg,
    get_planner_cfg,
    load_config,
    preflight_check,
)
from cuddlytoddly.engine.orchestrator import Orchestrator
from cuddlytoddly.engine.quality_gate import QualityGate
from cuddlytoddly.planning.llm_executor import LLMExecutor
from cuddlytoddly.planning.llm_planner import LLMPlanner
from cuddlytoddly.ui.curses_ui import run_ui
from cuddlytoddly.ui.startup import StartupChoice, run_startup_curses
from cuddlytoddly.ui.ui_config import make_cuddlytoddly_config
from cuddlytoddly.ui.web_server import run_web_ui
from toddly.core.events import ADD_NODE, Event
from toddly.core.id_generator import StableIDGenerator
from toddly.core.reducer import apply_event
from toddly.core.task_graph import TaskGraph
from toddly.infra.event_log import EventLog
from toddly.infra.event_queue import EventQueue
from toddly.infra.logging import get_logger, setup_logging
from toddly.infra.replay import rebuild_graph_from_log
from toddly.planning.llm_interface import (
    TokenCounter,
    token_counter,
)
from toddly.skills.skill_loader import SkillLoader
from toddly.utils.build_llm_client import build_llm_client as _build_llm_client_impl
from toddly.utils.make_run_dir import make_run_dir as _make_run_dir_impl

REPO_ROOT = Path(__file__).resolve().parent  # package code location

# FIX #8: do NOT call setup_logging() at module import time.  The module is
# imported before any run directory is known, so a module-level call produces
# log output with no per-run log file and accumulates duplicate handlers if the
# module is imported more than once (e.g. in tests).  setup_logging() is now
# called once, early in main(), before any log messages are emitted.

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
        # FIX #5: snapshot _real under the lock, then call stop() outside it.
        # Holding _lock while delegating risks deadlock if the backend's stop()
        # tries to acquire its own lock while a concurrent ask() thread holds it.
        with self._lock:
            real = self._real
        if real is not None and hasattr(real, "stop"):
            real.stop()

    def resume(self) -> None:
        # FIX #5: same pattern as stop() — release lock before delegating.
        with self._lock:
            real = self._real
        if real is not None and hasattr(real, "resume"):
            real.resume()

    def ask(self, prompt: str, schema=None) -> str:
        from toddly.planning.llm_interface import LLMStoppedError

        with self._lock:
            real = self._real
        if real is None:
            raise LLMStoppedError("LLM is still loading — execution will resume automatically")
        return real.ask(prompt, schema=schema) if schema is not None else real.ask(prompt)

    def generate(self, prompt: str) -> str:
        # FIX #10: delegate to the real client's generate() method once
        # attached, rather than routing through ask() which has different
        # semantics (ask() always accepts a schema kwarg; generate() is a
        # simpler text-completion call without structured output).  Before
        # attachment we raise LLMStoppedError consistent with ask().
        from toddly.planning.llm_interface import LLMStoppedError

        with self._lock:
            real = self._real
        if real is None:
            raise LLMStoppedError("LLM is still loading — execution will resume automatically")
        if hasattr(real, "generate"):
            return real.generate(prompt)
        # Fallback: if the real client exposes only ask(), use it without schema.
        return real.ask(prompt)

    def attach(self, real_llm) -> None:
        """Swap in the real client; from this point on the LLM is live."""
        with self._lock:
            self._real = real_llm
        logger.info("[DEFERRED LLM] Real LLM attached — execution enabled")


def make_run_dir(goal_text: str) -> Path:
    """Create a timestamped run directory under DATA_DIR/runs/."""
    return _make_run_dir_impl(goal_text, base_dir=DATA_DIR)


def _print_preflight_issues(issues: list[dict]) -> None:
    errors = [i for i in issues if i["level"] == "error"]
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
    # Bootstrap stderr logging so that preflight warnings and config-load
    # messages are visible before the run directory is known.  _init_system()
    # calls setup_logging(log_dir=…) which clears these handlers and installs
    # the full rotating file handlers, so there is only ever one complete
    # initialisation rather than two overlapping ones.
    import logging as _logging

    _dag_root = _logging.getLogger("dag")
    if not _dag_root.hasHandlers():
        _dag_root.setLevel(_logging.DEBUG)
        _bootstrap_ch = _logging.StreamHandler()
        _bootstrap_ch.setLevel(_logging.WARNING)
        _bootstrap_ch.setFormatter(_logging.Formatter("[%(levelname)s] %(name)s: %(message)s"))
        _dag_root.addHandler(_bootstrap_ch)

    # ── Load config ───────────────────────────────────────────────────────────
    cfg = load_config()
    server_cfg = cfg.get("server", {})

    # ── Pre-flight checks ─────────────────────────────────────────────────────
    issues = preflight_check(cfg)
    if issues:
        for issue in issues:
            logger.warning(
                "[PREFLIGHT] %s: %s — %s",
                issue["level"].upper(),
                issue["message"],
                issue.get("fix", ""),
            )
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
    args = parser.parse_args()
    use_web = not args.terminal

    # ── Build a bound init function that carries cfg through all call sites ───
    def _init(choice: StartupChoice, _use_web: bool = use_web, on_graph_ready=None):
        return _init_system(choice, _use_web, cfg, on_graph_ready=on_graph_ready)

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

    # One UIConfig instance shared by both UI paths so the domain behaviour is
    # defined in a single place.  The config only uses the already-initialised
    # orchestrator via closures built inside each hook function, so it can be
    # constructed here after _init() completes.
    ui_config = make_cuddlytoddly_config()

    if use_web:
        run_web_ui(
            orchestrator=orchestrator,
            run_dir=run_dir,
            init_fn=_init,
            cfg=cfg,
            host=args.host,
            port=args.port,
            config=ui_config,
        )
    else:
        run_ui(
            orchestrator,
            run_dir=run_dir,
            repo_root=DATA_DIR,
            restart_fn=_init,
            git_proj_instance=git_proj.configure(run_dir / "dag_repo"),
            config=ui_config,
        )

    # ── Final log ─────────────────────────────────────────────────────────────
    snap = orchestrator.graph.get_snapshot()
    logger.info("=== Final graph state (%d nodes) ===", len(snap))
    for nid, n in snap.items():
        logger.info("  [%s] %s", n.status, nid)


def _init_system(choice: "StartupChoice", use_web: bool, cfg: dict, on_graph_ready=None):
    """
    Build the full orchestrator from a StartupChoice and a loaded config dict.
    Returns (orchestrator, run_dir).

    All numeric tuning parameters are read from cfg so no source files need
    editing to adjust behaviour.
    """
    goal_text = choice.goal_text

    # FIX #7: sanitise goal_id with the same alnum+underscore filter used in
    # make_run_dir so that special characters (quotes, backslashes, unicode…)
    # never leak into event-log node IDs and corrupt JSONL replay.
    goal_id = "".join(c for c in goal_text.lower().replace(" ", "_") if c.isalnum() or c == "_")[
        :60
    ]
    if not goal_id:
        goal_id = "goal"

    run_dir = choice.run_dir.resolve()

    # ── Logging ───────────────────────────────────────────────────────────────
    setup_logging(log_dir=run_dir / "logs")
    _logger = get_logger(__name__)
    _logger.info(
        "=== cuddlytoddly starting  mode=%s  ui=%s ===",
        choice.mode,
        "web" if use_web else "curses",
    )
    _logger.info("Run directory: %s", run_dir)

    # ── Event log ─────────────────────────────────────────────────────────────
    event_log_path = run_dir / "events.jsonl"
    event_log = EventLog(str(event_log_path))

    # ── Git repo — per run ────────────────────────────────────────────────────
    # configure() resets the module-level default GitProjection instance to this
    # run's repo path, replacing the old git_proj.REPO_PATH = ... assignment.
    # For truly concurrent web-mode runs a per-instance approach is preferred;
    # this is safe for the single-active-run terminal mode.
    git_proj.configure(run_dir / "dag_repo")

    # FIX #1: removed os.chdir() here — the working directory is now stored on
    # the executor and set/restored around each individual tool call so that
    # concurrent runs in web-server mode cannot clobber each other's CWD.
    working_dir = run_dir / "outputs"
    _logger.info("Task working directory: %s", working_dir)

    # FIX (security): configure the file_ops sandbox so that read_file,
    # write_file, append_file, and list_dir are restricted to the run's
    # outputs/ directory.  Absolute paths outside this sandbox raise
    # ValueError and fail the tool call cleanly rather than silently
    # reading or writing arbitrary files on the host filesystem.
    file_ops_tools.configure(working_dir)

    # FIX #5: create the StableIDGenerator per-run and pass it directly to the
    # LLM client rather than replacing a shared module-level global.  This
    # prevents two concurrent web-mode runs from racing on the same id_gen.
    run_id_gen = StableIDGenerator(
        mapping_file=run_dir / "task_id_map.json",
        id_length=6,
    )

    # ── Read config sections (with defaults for old configs) ──────────────────
    orch_cfg = get_orchestrator_cfg(cfg)
    exec_cfg = get_executor_cfg(cfg)
    planner_cfg = get_planner_cfg(cfg)

    max_workers = orch_cfg["max_workers"]
    max_successful_turns = orch_cfg["max_successful_turns"]
    max_unsuccessful_turns = orch_cfg["max_unsuccessful_turns"]

    # ── Per-run token counter ─────────────────────────────────────────────────
    # FIX #3: created here — before the graph-replay block — so it is in scope
    # when the seed call inside the replay branch fires (the seed call must
    # happen before the first LLM call, not after).
    run_token_counter = TokenCounter()

    # ── Graph init ────────────────────────────────────────────────────────────
    if not choice.is_fresh and event_log_path.exists() and event_log_path.stat().st_size > 0:
        _logger.info("[STARTUP] Replaying event log")
        graph = rebuild_graph_from_log(event_log)
        fresh_start = False
        _logger.info("[STARTUP] Restored %d nodes", len(graph.nodes))

        for step_id in [n.id for n in graph.nodes.values() if n.node_type == "execution_step"]:
            if step_id in graph.nodes:
                graph.detach_node(step_id)

        for node_id in {
            n.id
            for n in graph.nodes.values()
            if n.status in ("running", "failed") and n.node_type != "execution_step"
        }:
            n = graph.nodes.get(node_id)
            if n:
                n.status = "pending"
                n.result = None
                n.metadata.pop("retry_count", None)
                n.metadata.pop("verification_failure", None)
                n.metadata.pop("verified", None)
                # FIX: also clear retry_after so a node that crashed while
                # inside an exponential-backoff window is not silently skipped
                # by _execution_pass() after restart.  Without this, the node
                # would sit in "ready" status but never be launched until the
                # stale timestamp expires (up to 60 s after the crash).
                n.metadata.pop("retry_after", None)

        graph.recompute_readiness()

        # ── Restore historical token counts ───────────────────────────────────
        # Read llamacpp_cache.json (present for llama.cpp runs; absent for
        # Anthropic/OpenAI runs — skipped silently in that case).
        # FIX #9: the original code approximated token counts as len(text) // 4,
        # which is an English-centric heuristic that can be off by 2-5× for
        # non-English text.  We now use tiktoken (if available) for a much more
        # accurate count, and fall back to the character-based approximation only
        # when tiktoken is not installed.  A warning is logged on fallback so the
        # discrepancy is visible in the run logs.
        cache_path = run_dir / "llamacpp_cache.json"
        if cache_path.exists():
            try:
                entries = json.loads(cache_path.read_text(encoding="utf-8"))
                prompt_total = 0
                completion_total = 0

                try:
                    import tiktoken

                    enc = tiktoken.get_encoding("cl100k_base")

                    def _count_tokens(text: str) -> int:
                        return len(enc.encode(text))

                except ImportError:
                    _logger.warning(
                        "[STARTUP] tiktoken not installed — token counts seeded from cache "
                        "will use the len//4 approximation which may be inaccurate for "
                        "non-English text.  Install tiktoken for accurate counts."
                    )

                    def _count_tokens(text: str) -> int:  # type: ignore[misc]
                        return len(text) // 4

                for entry in entries.values():
                    prompt_total += _count_tokens(entry.get("prompt", ""))
                    completion_total += _count_tokens(entry.get("response", ""))

                run_token_counter.seed(prompt_total, completion_total, calls=len(entries))
                token_counter.seed(prompt_total, completion_total, calls=len(entries))
                _logger.info(
                    "[STARTUP] Seeded token counter from cache: "
                    "%d prompt + %d completion = %d total (%d calls)",
                    prompt_total,
                    completion_total,
                    prompt_total + completion_total,
                    len(entries),
                )
            except Exception as exc:
                _logger.warning("[STARTUP] Could not seed token counter from cache: %s", exc)
        # ─────────────────────────────────────────────────────────────────────

    else:
        graph = TaskGraph()
        fresh_start = True

    # ── LLM client ────────────────────────────────────────────────────────────
    use_deferred = (not fresh_start) and (on_graph_ready is not None)
    if use_deferred:
        deferred_llm = _DeferredLLM()
        shared_llm = deferred_llm
    else:
        deferred_llm = None
        shared_llm = _build_llm_client(cfg, run_dir, run_id_gen, run_token_counter)

    # ── Components ────────────────────────────────────────────────────────────
    skills = SkillLoader()
    registry = skills.registry

    planner = LLMPlanner(
        llm_client=shared_llm,
        graph=graph,
        skills_summary=skills.prompt_summary,
        min_tasks_per_goal=planner_cfg["min_tasks_per_goal"],
        max_tasks_per_goal=planner_cfg["max_tasks_per_goal"],
        scrutinize_plan=planner_cfg["scrutinize_plan"],
        min_clarification_fields=planner_cfg["min_clarification_fields"],
        max_clarification_fields=planner_cfg["max_clarification_fields"],
    )

    # FIX #1: pass working_dir to the executor so it can chdir/restore around
    # each individual tool call instead of relying on the process-wide CWD.
    from toddly.infra.tool_call_log import ToolCallLog

    tool_call_log = ToolCallLog(run_dir / "tool_calls.jsonl")

    executor = LLMExecutor(
        llm_client=shared_llm,
        tool_registry=registry,
        max_successful_turns=max_successful_turns,
        max_unsuccessful_turns=max_unsuccessful_turns,
        max_inline_result_chars=exec_cfg["max_inline_result_chars"],
        max_total_input_chars=exec_cfg["max_total_input_chars"],
        max_tool_result_chars=exec_cfg["max_tool_result_chars"],
        max_history_entries=exec_cfg["max_history_entries"],
        working_dir=working_dir,
        tool_call_log=tool_call_log,
    )

    # FIX #5: pass the executor's working_dir to QualityGate so that declared
    # file output paths (e.g. "report.md") are resolved against the correct
    # directory during file-existence checks rather than the process CWD.
    #
    # verify_prompt_fn and check_deps_prompt_fn default to the cuddlytoddly
    # implementations in planning/prompts.py, so they only need to be passed
    # here if overriding the defaults.  verify_schema / check_deps_schema are
    # no longer constructor params — they are now owned by the prompt functions.
    quality_gate = QualityGate(
        llm_client=shared_llm,
        tool_registry=registry,
        max_total_input_chars=exec_cfg["max_total_input_chars"],
        working_dir=working_dir,
    )

    queue = EventQueue()
    orchestrator = Orchestrator(
        graph=graph,
        planner=planner,
        executor=executor,
        quality_gate=quality_gate,
        event_log=event_log,
        event_queue=queue,
        max_workers=max_workers,
        max_gap_fill_attempts=orch_cfg["max_gap_fill_attempts"],
        max_retries=orch_cfg["max_retries"],
        idle_sleep=orch_cfg["idle_sleep"],
        token_counter_instance=run_token_counter,
    )

    # ── Seed graph ────────────────────────────────────────────────────────────
    if fresh_start:
        if choice.mode == "manual_plan" and choice.plan_events:
            _logger.info("[STARTUP] Seeding manual plan (%d events)", len(choice.plan_events))
            for evt_dict in choice.plan_events:
                apply_event(
                    graph,
                    Event(evt_dict["type"], evt_dict["payload"]),
                    event_log=event_log,
                )
        else:
            _logger.info("[STARTUP] Seeding new goal: %s", goal_text)
            apply_event(
                graph,
                Event(
                    ADD_NODE,
                    {
                        "node_id": goal_id,
                        "node_type": "goal",
                        "dependencies": [],
                        "origin": "user",
                        "metadata": {"description": goal_text, "expanded": False},
                    },
                ),
                event_log=event_log,
            )

    orchestrator.start()

    if use_deferred:
        on_graph_ready(orchestrator, run_dir)

        def _load_real_llm():
            _logger.info("[STARTUP] Background LLM load starting…")
            try:
                real_llm = _build_llm_client(cfg, run_dir, run_id_gen, run_token_counter)
                deferred_llm.attach(real_llm)
                _logger.info("[STARTUP] LLM ready — starting background verification")
                orchestrator.verify_restored_nodes()
            except Exception as exc:
                _logger.error("[STARTUP] Background LLM load failed: %s", exc)
                # FIX: StatusEvent now lives in event_queue (it was previously
                # imported inside a try/except because it didn't exist there,
                # so failures were always silently swallowed).  Import at the
                # top of the except block — no defensive try needed.
                from toddly.infra.event_queue import StatusEvent

                orchestrator.event_queue.put(
                    StatusEvent(
                        "llm_load_failed",
                        {"error": str(exc)},
                    )
                )

        threading.Thread(target=_load_real_llm, daemon=True, name="startup-llm").start()

    else:
        if not fresh_start:

            def _bg_verify():
                _logger.info("[STARTUP] Background verification pass starting...")
                orchestrator.verify_restored_nodes()
                _logger.info("[STARTUP] Background verification complete")

            threading.Thread(target=_bg_verify, daemon=True, name="startup-verify").start()

    return orchestrator, run_dir


def _build_llm_client(
    cfg: dict,
    run_dir: Path,
    id_gen: "StableIDGenerator | None" = None,
    run_token_counter: "TokenCounter | None" = None,
):
    """Delegate to cuddly.utils.build_llm_client, supplying DATA_DIR."""
    return _build_llm_client_impl(
        cfg,
        run_dir=run_dir,
        data_dir=DATA_DIR,
        id_gen=id_gen,
        run_token_counter=run_token_counter,
    )


if __name__ == "__main__":
    main()
