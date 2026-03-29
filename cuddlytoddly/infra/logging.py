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
