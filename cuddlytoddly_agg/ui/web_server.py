# --- FILE: cuddlytoddly/ui/web_server.py ---

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
from datetime import datetime, timezone
from pathlib import Path

try:
    import uvicorn
    from fastapi import FastAPI, HTTPException, WebSocket
    from fastapi.responses import HTMLResponse
except ImportError:
    raise ImportError(
        "Web UI requires FastAPI and uvicorn:\n  pip install fastapi 'uvicorn[standard]'"
    )

from cuddlytoddly.core.events import (
    ADD_DEPENDENCY,
    ADD_NODE,
    REMOVE_DEPENDENCY,
    REMOVE_NODE,
    RESET_NODE,
    SET_RESULT,
    UPDATE_METADATA,
    UPDATE_STATUS,
    Event,
)
from cuddlytoddly.infra.logging import get_logger

from cuddlytoddly.ui.ui_config import UIConfig

logger = get_logger(__name__)

_HERE = Path(__file__).resolve().parent

_HIDDEN_META = frozenset(
    {
        "expanded",
        "fully_refined",
        "dependency_reflected",
        "last_commit_status",
        "last_commit_parents",
        "missing_inputs",
        "coverage_checked",
    }
)


# ── Serialization ─────────────────────────────────────────────────────────────


def _serialize_snapshot(snapshot: dict, config: UIConfig | None = None) -> dict:
    _include = (
        config.snapshot_filter_fn if config and config.snapshot_filter_fn else lambda _n: True
    )
    out = {}
    for nid, node in snapshot.items():
        if not _include(node):
            continue
        out[nid] = {
            "id": node.id,
            "node_type": node.node_type,
            "status": node.status,
            "origin": node.origin,
            "dependencies": sorted(node.dependencies),
            "children": sorted(node.children),
            "result": node.result,
            "metadata": {k: v for k, v in node.metadata.items() if k not in _HIDDEN_META},
        }
    return out


def _build_payload(orchestrator, config: UIConfig | None = None) -> dict:
    snapshot = orchestrator.get_snapshot()
    elapsed = None
    if orchestrator.activity_started:
        elapsed = round(time.time() - orchestrator.activity_started, 1)
    return {
        "type": "snapshot",
        "nodes": _serialize_snapshot(snapshot, config),
        "status": orchestrator.get_status(),
        "paused": orchestrator.llm_stopped,
        "activity": orchestrator.current_activity,
        "elapsed": elapsed,
        "tokens": orchestrator.token_counts,
    }


# ── Static HTML export ────────────────────────────────────────────────────────


def _build_static_html(
    snapshot: dict,
    run_dir: Path,
    token_counts: dict | None = None,
    config: UIConfig | None = None,
) -> tuple[str, Path]:
    """
    Generate a standalone, self-contained HTML file from the current snapshot.

    The template (web_ui_static.html) contains three placeholder tokens:
      "SNAPSHOT_DATA_PLACEHOLDER"  — serialised nodes dict
      "EXPORT_META_PLACEHOLDER"    — {goal, timestamp, tokens}
                                     tokens: {prompt, completion, total, calls}
      "REPLAY_EVENTS_PLACEHOLDER"  — ordered list of events from events.jsonl
                                     (empty list when the log is not found)

    The file is written to <run_dir>/outputs/ and (html_string, path) is returned.
    """
    template = (_HERE / "web_ui_static.html").read_text(encoding="utf-8")

    nodes_json = json.dumps(snapshot, default=str, ensure_ascii=False)

    if config and config.find_title_fn:
        goal_title = config.find_title_fn(snapshot)
    else:
        goal_node = next((n for n in snapshot.values() if n.get("node_type") == "goal"), None)
        goal_title = (
            (goal_node.get("metadata") or {}).get("description") or goal_node.get("id", "")
            if goal_node
            else ""
        )
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    meta_json = json.dumps(
        {
            "goal": goal_title,
            "timestamp": ts,
            "tokens": token_counts or {"prompt": 0, "completion": 0, "total": 0, "calls": 0},
        }
    )

    # ── Read and embed the event log ──────────────────────────────────────────
    # Events are read in order from events.jsonl and embedded verbatim so the
    # standalone file can replay the full history without any server connection.
    events: list[dict] = []
    events_path = run_dir / "events.jsonl"
    if events_path.exists():
        for raw_line in events_path.read_text(encoding="utf-8").splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue
            try:
                events.append(json.loads(raw_line))
            except json.JSONDecodeError:
                logger.warning("[EXPORT] Skipping corrupt event line: %.80s", raw_line)
    events_json = json.dumps(events, default=str, ensure_ascii=False)

    html = (
        template.replace('"SNAPSHOT_DATA_PLACEHOLDER"', nodes_json)
        .replace('"EXPORT_META_PLACEHOLDER"', meta_json)
        .replace('"REPLAY_EVENTS_PLACEHOLDER"', events_json)
    )

    safe_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = run_dir / "outputs" / f"dag_snapshot_{safe_ts}.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")

    logger.info("[EXPORT] Static HTML written to %s (%d events embedded)", out_path, len(events))
    return html, out_path


# ── App factory ───────────────────────────────────────────────────────────────


def create_app(orchestrator, run_dir: Path, config: UIConfig | None = None) -> FastAPI:
    """
    Build the FastAPI app for a fully-initialised orchestrator.
    All generic routes are registered unconditionally; domain-specific routes
    are registered via config.extra_routes_fn (if supplied).
    """
    app = FastAPI(title="cuddlytoddly")

    def _build_payload_with_config() -> dict:
        snapshot = orchestrator.get_snapshot()
        elapsed = None
        if orchestrator.activity_started:
            elapsed = round(time.time() - orchestrator.activity_started, 1)
        return {
            "type": "snapshot",
            "nodes": _serialize_snapshot(snapshot, config),
            "status": orchestrator.get_status(),
            "paused": orchestrator.llm_stopped,
            "activity": orchestrator.current_activity,
            "elapsed": elapsed,
            "tokens": orchestrator.token_counts,
        }

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
        last_paused = None  # sentinel — forces a push on first tick
        last_activity = None
        try:
            while True:
                sv = orchestrator.graph.structure_version
                ev = orchestrator.graph.execution_version
                paused = orchestrator.llm_stopped
                activity = orchestrator.current_activity
                if (
                    sv != last_sv
                    or ev != last_ev
                    or paused != last_paused
                    or activity != last_activity
                ):
                    payload = await asyncio.to_thread(_build_payload_with_config)
                    await websocket.send_text(json.dumps(payload, default=str))
                    last_sv, last_ev = sv, ev
                    last_paused = paused
                    last_activity = activity
                await asyncio.sleep(0.25)
        except Exception as e:
            logger.info("[WEB] WebSocket disconnected: %s", e)

    # ── Read ──────────────────────────────────────────────────────────────────

    @app.get("/api/snapshot")
    async def get_snapshot():
        return await asyncio.to_thread(_build_payload_with_config)

    # ── Node mutations ────────────────────────────────────────────────────────

    @app.post("/api/node")
    async def add_node(body: dict):
        node_id = (body.get("node_id") or "").strip()
        if not node_id:
            raise HTTPException(400, "node_id is required")
        dependencies = body.get("dependencies", [])
        dependents = body.get("dependents", [])
        orchestrator.event_queue.put(
            Event(
                ADD_NODE,
                {
                    "node_id": node_id,
                    "node_type": body.get("node_type", "task"),
                    "dependencies": dependencies,
                    "origin": "user",
                    "metadata": {"description": body.get("description", "")},
                },
            )
        )
        for dep_id in dependents:
            orchestrator.event_queue.put(
                Event(
                    ADD_DEPENDENCY,
                    {
                        "node_id": dep_id,
                        "depends_on": node_id,
                    },
                )
            )
            orchestrator.event_queue.put(Event(RESET_NODE, {"node_id": dep_id}))
        return {"ok": True}

    @app.put("/api/node/{node_id:path}")
    async def edit_node(node_id: str, body: dict):
        snap = orchestrator.get_snapshot()
        node = snap.get(node_id)
        if not node:
            raise HTTPException(404, "node not found")
        _meta_update = {
            "description": body.get("description", node.metadata.get("description", ""))
        }
        if "execution_steps" in body:
            _meta_update["execution_steps"] = body["execution_steps"]
        orchestrator.event_queue.put(
            Event(
                UPDATE_METADATA,
                {
                    "node_id": node_id,
                    "origin": "user",
                    "metadata": _meta_update,
                },
            )
        )
        st = body.get("status", "")
        if st in (
            config.valid_status_values if config else ("pending", "done", "running", "failed")
        ):
            orchestrator.event_queue.put(
                Event(
                    UPDATE_STATUS,
                    {
                        "node_id": node_id,
                        "status": st,
                    },
                )
            )
        if "dependencies" in body:
            old = set(node.dependencies)
            new = set(body["dependencies"])
            for removed in old - new:
                orchestrator.event_queue.put(
                    Event(
                        REMOVE_DEPENDENCY,
                        {
                            "node_id": node_id,
                            "depends_on": removed,
                        },
                    )
                )
            for added in new - old:
                orchestrator.event_queue.put(
                    Event(
                        ADD_DEPENDENCY,
                        {
                            "node_id": node_id,
                            "depends_on": added,
                        },
                    )
                )
        if "dependents" in body:
            snap2 = orchestrator.get_snapshot()
            old_deps = {nid for nid, n in snap2.items() if node_id in n.dependencies}
            new_deps = {d for d in body["dependents"] if d in snap2 and d != node_id}
            for removed in old_deps - new_deps:
                orchestrator.event_queue.put(
                    Event(
                        REMOVE_DEPENDENCY,
                        {
                            "node_id": removed,
                            "depends_on": node_id,
                        },
                    )
                )
                orchestrator.event_queue.put(Event(RESET_NODE, {"node_id": removed}))
            for added in new_deps - old_deps:
                orchestrator.event_queue.put(
                    Event(
                        ADD_DEPENDENCY,
                        {
                            "node_id": added,
                            "depends_on": node_id,
                        },
                    )
                )
                orchestrator.event_queue.put(Event(RESET_NODE, {"node_id": added}))

        result_changed = "result" in body and body.get("result", "") != (node.result or "")
        desc_changed = "description" in body and body["description"] != node.metadata.get(
            "description", ""
        )
        deps_changed = "dependencies" in body and set(body["dependencies"]) != set(
            node.dependencies
        )
        steps_changed = "execution_steps" in body and body["execution_steps"] != node.metadata.get(
            "execution_steps", []
        )

        if result_changed:
            orchestrator.event_queue.put(
                Event(
                    SET_RESULT,
                    {
                        "node_id": node_id,
                        "result": body["result"] if body["result"] else None,
                    },
                )
            )
            for child_id in node.children:
                child = snap.get(child_id)
                if child and child.status == "done":
                    orchestrator.event_queue.put(Event(RESET_NODE, {"node_id": child_id}))
        elif desc_changed or deps_changed or steps_changed:
            orchestrator.event_queue.put(Event(RESET_NODE, {"node_id": node_id}))

        return {"ok": True}

    @app.delete("/api/node/{node_id:path}")
    async def remove_node(node_id: str, mode: str = "cascade"):
        snap = orchestrator.get_snapshot()
        node = snap.get(node_id)
        if not node:
            raise HTTPException(404, "node not found")
        parents = list(node.dependencies)
        children = list(node.children)
        q = orchestrator.event_queue
        if mode == "rewire":
            for child in children:
                q.put(Event(REMOVE_DEPENDENCY, {"node_id": child, "depends_on": node_id}))
                for parent in parents:
                    q.put(Event(ADD_DEPENDENCY, {"node_id": child, "depends_on": parent}))
            q.put(Event(REMOVE_NODE, {"node_id": node_id}))
            for child in children:
                q.put(Event(RESET_NODE, {"node_id": child}))
        elif mode == "disconnect":
            for child in children:
                q.put(Event(REMOVE_DEPENDENCY, {"node_id": child, "depends_on": node_id}))
            q.put(Event(REMOVE_NODE, {"node_id": node_id}))
            for child in children:
                q.put(Event(RESET_NODE, {"node_id": child}))
        else:  # cascade
            q.put(Event(REMOVE_NODE, {"node_id": node_id}))
        return {"ok": True}

    @app.post("/api/node/{node_id:path}/retry")
    async def retry_node(node_id: str):
        orchestrator.retry_node(node_id)
        return {"ok": True}

    @app.post("/api/node/{node_id:path}/resume")
    async def resume_node(node_id: str):
        """
        Explicitly resume an awaiting_input node.

        Normally resumption happens automatically via _resume_unblocked_pass
        when the user fills in the required clarification fields.  This
        endpoint allows the UI to trigger an immediate manual resume — for
        example if the user edits the clarification node directly and wants
        to unblock the task without waiting for the next loop tick.
        """
        resumed = orchestrator.resume_node(node_id)
        if not resumed:
            snap = orchestrator.get_snapshot()
            node = snap.get(node_id)
            if not node:
                raise HTTPException(404, f"node '{node_id}' not found")
            raise HTTPException(
                400,
                f"node '{node_id}' is not awaiting_input (status={node.status})",
            )
        return {"ok": True}

    @app.post("/api/node/{node_id:path}/confirm")
    async def confirm_node(node_id: str):
        """
        Confirm that the user has completed the real-world steps for an
        awaiting_user node, transitioning it to done and unblocking downstream
        dependents.
        """
        confirmed = orchestrator.confirm_node(node_id)
        if not confirmed:
            snap = orchestrator.get_snapshot()
            node = snap.get(node_id)
            if not node:
                raise HTTPException(404, f"node '{node_id}' not found")
            raise HTTPException(
                400,
                f"node '{node_id}' is not awaiting_user (status={node.status})",
            )
        return {"ok": True}

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
            path = export_results_to_markdown(snap, run_dir, config)
            return {"ok": True, "path": str(path)}
        except Exception as e:
            raise HTTPException(500, str(e))

    @app.post("/api/export/html")
    async def export_html():
        snap = _serialize_snapshot(orchestrator.get_snapshot(), config)
        try:
            _, path = await asyncio.to_thread(
                _build_static_html,
                snap,
                run_dir,
                orchestrator.token_counts,
                config,
            )
            return {"ok": True, "path": str(path)}
        except Exception as e:
            raise HTTPException(500, str(e))

    @app.post("/api/switch")
    async def switch_goal(body: dict = None):
        # create_app is used when the server was launched with a pre-built
        # orchestrator (inline goal).  Switching requires the unified app
        # (started without an inline goal) which holds init_fn.  Return a
        # proper JSON error so the UI displays a readable message instead of
        # failing to parse an HTTPException response.
        return {
            "ok": False,
            "error": (
                "Goal switching is not available when the server was started "
                "with an inline goal. Restart without a goal argument to enable switching."
            ),
        }

    # ── Domain-specific routes ─────────────────────────────────────────────────
    # Registered last so generic routes always take priority.  The cuddlytoddly
    # config adds /api/node/{id}/clarification/confirm and /api/goal/{id}/replan.
    if config and config.extra_routes_fn:
        config.extra_routes_fn(app, orchestrator, run_dir)

    return app


# ── Unified run_web_ui ────────────────────────────────────────────────────────


def run_web_ui(
    orchestrator=None,
    run_dir=None,
    repo_root: Path | None = None,
    init_fn=None,
    cfg: dict | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    config: UIConfig | None = None,
):
    """
    Start the web UI server and open a browser tab.

    Option A — already initialised (inline goal or curses startup chose):
        run_web_ui(orchestrator=orch, run_dir=rd, config=config)

    Option B — deferred init (startup screen shown in browser):
        run_web_ui(repo_root=REPO_ROOT, init_fn=lambda choice: (orch, rd))
    """
    import webbrowser

    if orchestrator is not None:
        app = create_app(orchestrator, run_dir, config)
    else:
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


def _create_unified_app(
    repo_root: Path | None,
    init_fn,
    cfg: dict | None = None,
) -> "FastAPI":
    """
    Single FastAPI app that handles both phases:
      - Before init: serves the startup HTML, /api/runs, /api/startup, /api/status
      - After init:  serves the DAG HTML, /ws, and all DAG REST routes

    All routes are registered upfront. DAG-specific routes return 503
    until the system is initialised.
    """
    app = FastAPI(title="cuddlytoddly")

    # _create_unified_app is inherently cuddlytoddly-specific (it owns startup,
    # goal-switching, and preflight).  Build a local UIConfig so that serialisation
    # and export use the same domain filter as the rest of the UI layer.
    from cuddlytoddly.ui.ui_config import make_cuddlytoddly_config as _make_cfg

    _ui_config = _make_cfg()

    state = {
        "orchestrator": None,
        "run_dir": None,
        "ready": False,
        "loading": False,
        "error": "",
    }
    # Fix #5: protect compound check-and-set operations on `state` from race
    # conditions between concurrent FastAPI handler coroutines (which run on
    # the ASGI thread pool) and the background _init() thread.  A plain dict
    # in CPython is GIL-safe for individual key reads/writes, but patterns like
    # `if state["loading"]: ... state["loading"] = True` are not atomic.
    # Without the lock two simultaneous POST /api/startup requests could both
    # pass the loading check and each spin up an _init() thread.
    _state_lock = threading.Lock()

    startup_html = (_HERE / "web_ui_startup.html").read_text(encoding="utf-8")
    dag_html = (_HERE / "web_ui.html").read_text(encoding="utf-8")

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
            "loading": state["loading"],
            "error": state["error"],
        }

    @app.get("/api/runs")
    async def api_runs():
        from cuddlytoddly.ui.startup import scan_runs

        return {"runs": scan_runs(repo_root) if repo_root else []}

    @app.get("/api/preflight")
    async def api_preflight():
        if cfg is None:
            return {"issues": []}
        from cuddlytoddly.config import preflight_check

        return {"issues": preflight_check(cfg)}

    @app.post("/api/startup")
    async def api_startup(body: dict):
        with _state_lock:
            if state["ready"]:
                return {"ok": True, "already_initialized": True}
            if state["loading"]:
                return {"ok": False, "error": "Already loading"}
            if init_fn is None:
                return {"ok": False, "error": "No init_fn configured"}

            from cuddlytoddly.__main__ import make_run_dir
            from cuddlytoddly.ui.startup import StartupChoice, parse_manual_plan

            mode = body.get("mode", "new_goal")
            goal_text = body.get("goal_text", "").strip()
            plan_text = body.get("plan_text", "").strip()
            run_path = body.get("run_dir", "")

            if mode == "existing":
                if not run_path:
                    return {"ok": False, "error": "run_dir required"}
                choice = StartupChoice(
                    mode="existing",
                    run_dir=Path(run_path),
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
                    goal_text=gt,
                    plan_events=evts,
                    is_fresh=True,
                )
            else:
                if not goal_text:
                    return {"ok": False, "error": "goal_text required"}
                choice = StartupChoice(
                    mode="new_goal",
                    run_dir=make_run_dir(goal_text).resolve(),
                    goal_text=goal_text,
                    is_fresh=True,
                )

            # Set loading=True inside the lock so that a concurrent request
            # that also passes the loading check above sees it immediately.
            state["loading"] = True
            state["error"] = ""

        def _init():
            try:
                if mode == "existing":

                    def _on_graph_ready(orch, rd):
                        # FIX: wrap all three writes in the state lock so no
                        # concurrent reader can observe ready=True while
                        # orchestrator is still None.
                        with _state_lock:
                            state["orchestrator"] = orch
                            state["run_dir"] = rd
                            state["ready"] = True
                        logger.info(
                            "[WEB] Graph ready — DAG visible (%d nodes)",
                            len(orch.graph.nodes),
                        )

                    init_fn(choice, on_graph_ready=_on_graph_ready)
                else:
                    orch, rd = init_fn(choice)
                    # FIX: wrap all three writes in the state lock so no
                    # concurrent reader can observe ready=True while
                    # orchestrator is still None.
                    with _state_lock:
                        state["orchestrator"] = orch
                        state["run_dir"] = rd
                        state["ready"] = True
                    logger.info(
                        "[WEB] System initialised — DAG has %d nodes",
                        len(orch.graph.nodes),
                    )
            except Exception as e:
                logger.exception("[WEB] init_fn failed: %s", e)
                with _state_lock:
                    state["error"] = str(e)
            finally:
                with _state_lock:
                    state["loading"] = False

        threading.Thread(target=_init, daemon=True, name="web-init").start()
        return {"ok": True}

    # ── Switch goal ───────────────────────────────────────────────────────────

    @app.post("/api/switch")
    async def api_switch(body: dict):
        """
        Tear down the current orchestrator and start a new one.

        Accepts the same body shape as /api/startup so the frontend can reuse
        the same payload-building logic for both initial start and mid-session
        switching.  After returning {ok: true} the client polls /api/status
        (same as after /api/startup) and reloads when initialized flips true.
        """
        with _state_lock:
            if state["loading"]:
                return {
                    "ok": False,
                    "error": "Already loading — wait for the current operation to finish",
                }
            if init_fn is None:
                return {"ok": False, "error": "No init_fn configured"}

            from cuddlytoddly.__main__ import make_run_dir
            from cuddlytoddly.ui.startup import StartupChoice, parse_manual_plan

            mode = body.get("mode", "new_goal")
            goal_text = body.get("goal_text", "").strip()
            plan_text = body.get("plan_text", "").strip()
            run_path = body.get("run_dir", "")

            if mode == "existing":
                if not run_path:
                    return {"ok": False, "error": "run_dir required"}
                choice = StartupChoice(
                    mode="existing",
                    run_dir=Path(run_path),
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
                    goal_text=gt,
                    plan_events=evts,
                    is_fresh=True,
                )
            else:  # new_goal
                if not goal_text:
                    return {"ok": False, "error": "goal_text required"}
                choice = StartupChoice(
                    mode="new_goal",
                    run_dir=make_run_dir(goal_text).resolve(),
                    goal_text=goal_text,
                    is_fresh=True,
                )

            # Stop the running orchestrator before replacing it so its background
            # thread and thread-pool don't keep consuming resources.
            old_orch = state["orchestrator"]
            if old_orch is not None:
                try:
                    old_orch.stop()
                except Exception as exc:
                    logger.warning("[WEB] Could not cleanly stop old orchestrator: %s", exc)

            # FIX: clear state atomically inside the lock so no concurrent
            # reader sees ready=True with a None or stale orchestrator.
            state["orchestrator"] = None
            state["run_dir"] = None
            state["ready"] = False
            state["loading"] = True
            state["error"] = ""

        def _switch():
            try:
                if mode == "existing":

                    def _on_graph_ready(orch, rd):
                        # FIX: wrap all three writes in the state lock.
                        with _state_lock:
                            state["orchestrator"] = orch
                            state["run_dir"] = rd
                            state["ready"] = True
                        logger.info(
                            "[WEB] Switch complete — DAG visible (%d nodes)",
                            len(orch.graph.nodes),
                        )

                    init_fn(choice, on_graph_ready=_on_graph_ready)
                else:
                    orch, rd = init_fn(choice)
                    # FIX: wrap all three writes in the state lock.
                    with _state_lock:
                        state["orchestrator"] = orch
                        state["run_dir"] = rd
                        state["ready"] = True
                    logger.info(
                        "[WEB] Switch complete — DAG has %d nodes",
                        len(orch.graph.nodes),
                    )
            except Exception as e:
                logger.exception("[WEB] switch failed: %s", e)
                with _state_lock:
                    state["error"] = str(e)
            finally:
                with _state_lock:
                    state["loading"] = False

        threading.Thread(target=_switch, daemon=True, name="web-switch").start()
        return {"ok": True}

    # ── WebSocket ─────────────────────────────────────────────────────────────

    @app.websocket("/ws")
    async def ws_endpoint(websocket: WebSocket):
        await websocket.accept()
        logger.info("[WEB] WebSocket connected")

        waited = 0
        while not state["ready"]:
            await asyncio.sleep(0.5)
            waited += 1
            if waited > 1200:
                logger.warning("[WEB] WebSocket timed out waiting for init")
                await websocket.close()
                return

        last_sv = last_ev = -1
        last_paused = None
        last_activity = None
        # FIX: track which orchestrator instance was last polled.  When
        # /api/switch replaces the orchestrator we reset the version sentinels
        # so the new state is pushed to the client immediately, rather than
        # the WebSocket silently continuing to stream from the old (stopped)
        # orchestrator for the lifetime of the connection.
        last_orch: object = None

        try:
            while True:
                # Re-read the orchestrator on every tick — it may have been
                # replaced by /api/switch since the last iteration.
                orch = state["orchestrator"]

                if orch is None:
                    # A switch is in progress; wait for ready to flip back.
                    await asyncio.sleep(0.5)
                    continue

                # Reset version sentinels whenever the orchestrator identity
                # changes so the first tick after a switch sends a full snapshot.
                if orch is not last_orch:
                    last_sv = last_ev = -1
                    last_paused = None
                    last_activity = None
                    last_orch = orch

                sv = orch.graph.structure_version
                ev = orch.graph.execution_version
                paused = orch.llm_stopped
                activity = orch.current_activity
                if (
                    sv != last_sv
                    or ev != last_ev
                    or paused != last_paused
                    or activity != last_activity
                ):
                    payload = await asyncio.to_thread(_build_payload, orch, _ui_config)
                    await websocket.send_text(json.dumps(payload, default=str))
                    last_sv, last_ev = sv, ev
                    last_paused = paused
                    last_activity = activity
                await asyncio.sleep(0.25)
        except Exception as e:
            logger.info("[WEB] WebSocket closed: %s", e)

    # ── DAG REST routes ───────────────────────────────────────────────────────

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
        return await asyncio.to_thread(_build_payload, _orch(), _ui_config)

    @app.post("/api/node")
    async def add_node(body: dict):
        orch = _orch()
        node_id = (body.get("node_id") or "").strip()
        if not node_id:
            raise HTTPException(400, "node_id is required")
        dependencies = body.get("dependencies", [])
        dependents = body.get("dependents", [])
        orch.event_queue.put(
            Event(
                ADD_NODE,
                {
                    "node_id": node_id,
                    "node_type": body.get("node_type", "task"),
                    "dependencies": dependencies,
                    "origin": "user",
                    "metadata": {"description": body.get("description", "")},
                },
            )
        )
        for dep_id in dependents:
            orch.event_queue.put(
                Event(
                    ADD_DEPENDENCY,
                    {
                        "node_id": dep_id,
                        "depends_on": node_id,
                    },
                )
            )
            orch.event_queue.put(Event(RESET_NODE, {"node_id": dep_id}))
        return {"ok": True}

    @app.put("/api/node/{node_id:path}")
    async def edit_node(node_id: str, body: dict):
        orch = _orch()
        snap = orch.get_snapshot()
        node = snap.get(node_id)
        if not node:
            raise HTTPException(404, "node not found")
        _meta_update = {
            "description": body.get("description", node.metadata.get("description", ""))
        }
        if "execution_steps" in body:
            _meta_update["execution_steps"] = body["execution_steps"]
        orch.event_queue.put(
            Event(
                UPDATE_METADATA,
                {
                    "node_id": node_id,
                    "origin": "user",
                    "metadata": _meta_update,
                },
            )
        )
        st = body.get("status", "")
        if st in ("pending", "done", "running", "failed", "to_be_expanded"):
            orch.event_queue.put(Event(UPDATE_STATUS, {"node_id": node_id, "status": st}))
        if "dependencies" in body:
            old = set(node.dependencies)
            new = set(body["dependencies"])
            for removed in old - new:
                orch.event_queue.put(
                    Event(REMOVE_DEPENDENCY, {"node_id": node_id, "depends_on": removed})
                )
            for added in new - old:
                orch.event_queue.put(
                    Event(ADD_DEPENDENCY, {"node_id": node_id, "depends_on": added})
                )
        orch.event_queue.put(Event(RESET_NODE, {"node_id": node_id}))
        return {"ok": True}

    @app.delete("/api/node/{node_id:path}")
    async def remove_node(node_id: str, mode: str = "cascade"):
        orch = _orch()
        snap = orch.get_snapshot()
        node = snap.get(node_id)
        if not node:
            raise HTTPException(404, "node not found")
        parents = list(node.dependencies)
        children = list(node.children)
        q = orch.event_queue
        if mode == "rewire":
            for child in children:
                q.put(Event(REMOVE_DEPENDENCY, {"node_id": child, "depends_on": node_id}))
                for parent in parents:
                    q.put(Event(ADD_DEPENDENCY, {"node_id": child, "depends_on": parent}))
            q.put(Event(REMOVE_NODE, {"node_id": node_id}))
            for child in children:
                q.put(Event(RESET_NODE, {"node_id": child}))
        elif mode == "disconnect":
            for child in children:
                q.put(Event(REMOVE_DEPENDENCY, {"node_id": child, "depends_on": node_id}))
            q.put(Event(REMOVE_NODE, {"node_id": node_id}))
            for child in children:
                q.put(Event(RESET_NODE, {"node_id": child}))
        else:
            q.put(Event(REMOVE_NODE, {"node_id": node_id}))
        return {"ok": True}

    @app.post("/api/node/{node_id:path}/retry")
    async def retry_node(node_id: str):
        _orch().retry_node(node_id)
        return {"ok": True}

    @app.post("/api/node/{node_id:path}/confirm")
    async def confirm_node(node_id: str):
        """
        Confirm that the user has completed the real-world steps for an
        awaiting_user node, transitioning it to done and unblocking downstream
        dependents.
        """
        confirmed = _orch().confirm_node(node_id)
        if not confirmed:
            snap = _orch().get_snapshot()
            node = snap.get(node_id)
            if not node:
                raise HTTPException(404, f"node '{node_id}' not found")
            raise HTTPException(
                400,
                f"node '{node_id}' is not awaiting_user (status={node.status})",
            )
        return {"ok": True}

    @app.post("/api/node/{node_id:path}/clarification/confirm")
    async def confirm_clarification(node_id: str, body: dict):
        orch = _orch()
        snap = orch.get_snapshot()
        node = snap.get(node_id)
        if not node or node.node_type != "clarification":
            raise HTTPException(404, "clarification node not found")

        updated_fields = body.get("updated_fields")
        if not isinstance(updated_fields, list):
            raise HTTPException(400, "updated_fields must be a list")

        import json as _json

        new_result = _json.dumps(updated_fields, ensure_ascii=False)

        q = orch.event_queue
        q.put(
            Event(
                UPDATE_METADATA,
                {
                    "node_id": node_id,
                    "origin": "user",
                    "metadata": {"fields": updated_fields},
                },
            )
        )
        q.put(Event(SET_RESULT, {"node_id": node_id, "result": new_result}))

        for child_id in node.children:
            child = snap.get(child_id)
            if child and child.node_type != "clarification":
                q.put(Event(RESET_NODE, {"node_id": child_id}))

        goal_id = node.metadata.get("parent_goal")
        if goal_id and goal_id in snap:
            q.put(
                Event(
                    UPDATE_METADATA,
                    {
                        "node_id": goal_id,
                        "origin": "user",
                        "metadata": {"expanded": False},
                    },
                )
            )

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

    # ── Export ────────────────────────────────────────────────────────────────

    @app.post("/api/export")
    async def export_md():
        from cuddlytoddly.ui.curses_ui import export_results_to_markdown

        snap = _orch().get_snapshot()
        try:
            path = export_results_to_markdown(snap, _run_dir(), _ui_config)
            return {"ok": True, "path": str(path)}
        except Exception as e:
            raise HTTPException(500, str(e))

    @app.post("/api/export/html")
    async def export_html():
        snap = _serialize_snapshot(_orch().get_snapshot(), _ui_config)
        try:
            _, path = await asyncio.to_thread(
                _build_static_html,
                snap,
                _run_dir(),
                _orch().token_counts,
                _ui_config,
            )
            return {"ok": True, "path": str(path)}
        except Exception as e:
            raise HTTPException(500, str(e))

    return app
