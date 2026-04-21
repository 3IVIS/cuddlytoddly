# skills/file_ops/tools.py
#
# Local tool implementations for the file_ops skill.
# The SkillLoader imports this and registers everything in TOOLS.
#
# SECURITY: all four filesystem tools validate that the resolved path stays
# inside the configured sandbox root (the run's outputs/ directory).  Call
# configure(sandbox_root) once per run before any tool executes.  Without
# a configured sandbox the tools still work, but a warning is emitted and
# no path restriction is enforced — do not deploy this way in production.

from pathlib import Path

from cuddlytoddly.infra.logging import get_logger

logger = get_logger(__name__)

# Module-level sandbox root.  Set by configure() in _init_system().
# Pattern follows git_proj.configure() used elsewhere in the codebase.
# NOTE: for truly concurrent web-mode runs a per-instance approach would be
# preferable; this is safe for the single-active-run terminal mode and for
# web mode where each goal switch calls configure() before any tool runs.
_SANDBOX_ROOT: Path | None = None


def configure(sandbox_root: "Path | str | None") -> None:
    """
    Set the directory that all file tools are restricted to.

    Must be called once per run (from _init_system) before any tool
    executes.  Passing None disables the sandbox and emits a warning.
    """
    global _SANDBOX_ROOT
    if sandbox_root is None:
        logger.warning(
            "[FILE_OPS] No sandbox_root configured — path traversal checks disabled. "
            "All filesystem paths will be accepted without restriction."
        )
        _SANDBOX_ROOT = None
    else:
        _SANDBOX_ROOT = Path(sandbox_root).resolve()
        logger.info("[FILE_OPS] Sandbox root set to: %s", _SANDBOX_ROOT)


def _safe_resolve(path: str) -> Path:
    """
    Resolve *path* and enforce the sandbox boundary.

    Relative paths are resolved against the current working directory,
    which LLMExecutor._run_tool() sets to the run's outputs/ directory
    before every tool call — so bare filenames like "report.md" always
    land inside the sandbox naturally.

    Absolute paths (e.g. "/etc/passwd") are checked explicitly: if they
    resolve to anything outside _SANDBOX_ROOT, ValueError is raised and
    the tool call fails cleanly rather than reading or writing an
    arbitrary file.

    If no sandbox is configured (configure() was never called or was
    called with None), the resolved path is returned unchecked and a
    warning is logged.
    """
    resolved = Path(path).resolve()

    if _SANDBOX_ROOT is None:
        logger.warning(
            "[FILE_OPS] Sandbox not configured — allowing unrestricted access to '%s'",
            resolved,
        )
        return resolved

    try:
        resolved.relative_to(_SANDBOX_ROOT)
    except ValueError:
        raise ValueError(
            f"Access denied: '{path}' resolves to '{resolved}', which is outside "
            f"the permitted sandbox '{_SANDBOX_ROOT}'. "
            "Only paths inside the run's outputs directory are allowed."
        )

    return resolved


# ── Tool implementations ──────────────────────────────────────────────────────


def _read_file(args: dict) -> str:
    path = _safe_resolve(args["path"])
    return path.read_text(encoding="utf-8")


def _write_file(args: dict) -> str:
    path = _safe_resolve(args["path"])
    content = args["content"]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return f"Written {len(content)} chars to {path}"


def _append_file(args: dict) -> str:
    path = _safe_resolve(args["path"])
    content = args["content"]
    # Use a context manager so the file handle is always closed,
    # even if write() raises (e.g. disk full).
    with path.open("a", encoding="utf-8") as fh:
        fh.write(content)
    return f"Appended {len(content)} chars to {path}"


def _list_dir(args: dict) -> str:
    path = _safe_resolve(args["path"])
    return "\n".join(str(p) for p in sorted(path.iterdir()))


TOOLS = {
    "read_file": {
        "description": "Read the full contents of a local file",
        "input_schema": {"path": "string"},
        "fn": _read_file,
    },
    "write_file": {
        "description": "Write (or overwrite) a local file with the given content",
        "input_schema": {"path": "string", "content": "string"},
        "fn": _write_file,
    },
    "append_file": {
        "description": "Append text to an existing file",
        "input_schema": {"path": "string", "content": "string"},
        "fn": _append_file,
    },
    "list_dir": {
        "description": "List files and directories at a path",
        "input_schema": {"path": "string"},
        "fn": _list_dir,
    },
}
