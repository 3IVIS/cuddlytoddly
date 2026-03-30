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
    SET_RESULT, RESET_SUBTREE,
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
        if "dependents" in body:
            snap2     = orchestrator.get_snapshot()
            old_deps  = {nid for nid, n in snap2.items() if node_id in n.dependencies}
            new_deps  = {d for d in body["dependents"] if d in snap2 and d != node_id}
            for removed in old_deps - new_deps:
                orchestrator.event_queue.put(Event(REMOVE_DEPENDENCY, {
                    "node_id": removed, "depends_on": node_id,
                }))
                orchestrator.event_queue.put(Event(RESET_SUBTREE, {"node_id": removed}))
            for added in new_deps - old_deps:
                orchestrator.event_queue.put(Event(ADD_DEPENDENCY, {
                    "node_id": added, "depends_on": node_id,
                }))
                orchestrator.event_queue.put(Event(RESET_SUBTREE, {"node_id": added}))

        if "result" in body:
            new_result   = body["result"]
            old_result   = node.result or ""
            result_changed = new_result != old_result
            if result_changed:
                # Update result in-place; the node keeps its current status.
                # Reset only children so they rerun with the new upstream result.
                orchestrator.event_queue.put(Event(SET_RESULT, {
                    "node_id": node_id,
                    "result":  new_result if new_result else None,
                }))
                for child_id in node.children:
                    orchestrator.event_queue.put(Event(RESET_SUBTREE, {"node_id": child_id}))
                return {"ok": True}

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
    run_dir=None,
    repo_root: Path | None = None,
    init_fn=None,
    cfg: dict | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
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

    @app.get("/api/preflight")
    async def api_preflight():
        """
        Return pre-flight configuration issues for the current backend.
        The startup UI fetches this on load and displays a banner if issues exist.
        Response shape:
            { "issues": [ { "level": "error"|"warning", "message": "...", "fix": "..." } ] }
        """
        if cfg is None:
            return {"issues": []}
        from cuddlytoddly.config import preflight_check
        return {"issues": preflight_check(cfg)}

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