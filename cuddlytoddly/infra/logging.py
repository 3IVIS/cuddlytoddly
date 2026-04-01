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
from datetime import datetime
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


def _rotate_existing_log(path: Path) -> None:
    """
    Archive *path* and all its numbered size-rotation siblings
    (dag.log.1, dag.log.2, …) to timestamped copies before the current
    session begins, so no history is ever overwritten or lost.

    All files from the same session share one timestamp prefix:
        dag.log    → dag.log.20240115_100711
        dag.log.1  → dag.log.20240115_100711.1
        dag.log.2  → dag.log.20240115_100711.2
        dag.log.3  → dag.log.20240115_100711.3

    Using a timestamp prefix (rather than plain numbers) keeps these
    session archives completely separate from the .1/.2/.3 space that
    ``RotatingFileHandler`` reuses for within-session size rotation, so
    the two mechanisms never overwrite each other.

    Does nothing when the main file does not exist or is already empty.
    """
    if not path.exists() or path.stat().st_size == 0:
        return

    ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_archive = Path(str(path) + f".{ts}")

    # Archive numbered size-rotation siblings first so their destination
    # names (base_archive.N) are free before we rename the main file.
    # Process highest number first — purely defensive ordering.
    for n in range(9, 0, -1):
        numbered = Path(str(path) + f".{n}")
        if numbered.exists():
            try:
                numbered.rename(Path(str(base_archive) + f".{n}"))
            except OSError as exc:
                logging.getLogger("dag").warning(
                    "[LOGGING] Could not rotate %s: %s", numbered, exc
                )

    # Archive the main log file last.
    try:
        path.rename(base_archive)
    except OSError as exc:
        logging.getLogger("dag").warning(
            "[LOGGING] Could not rotate %s → %s: %s", path, base_archive, exc
        )


class _DeduplicateFilter(logging.Filter):
    """
    Suppress consecutive log records that carry identical content.

    Two records are considered identical when they share the same log level,
    logger name, and formatted message text.  The timestamp is intentionally
    excluded so that the same line repeating across multiple seconds is still
    deduplicated.

    One instance must be attached to each handler independently so that the
    deduplication state for the main log and the debug log are tracked
    separately (the debug handler sees a different subset of records, so its
    "last seen" key will naturally diverge).

    Not applied to stderr — the live terminal output is a separate stream
    where a user may want to see repeated warnings.
    """

    def __init__(self) -> None:
        super().__init__()
        self._last_key: tuple | None = None

    def filter(self, record: logging.LogRecord) -> bool:
        key = (record.levelno, record.name, record.getMessage())
        if key == self._last_key:
            return False   # identical to previous — suppress
        self._last_key = key
        return True


def setup_logging(
    log_level: int = logging.INFO,
    log_dir: Path | str | None = None,
    debug_modules: tuple[str, ...] = (
        "dag.engine",
        "dag.planning",
        "dag.skills"
    ),
    max_bytes: int = 5 * 1024 * 1024,   # 5 MB per file
    backup_count: int = 3,               # keep .1 .2 .3 within a session
) -> logging.Logger:
    """
    Configure the application root logger.

    Creates:
      logs/dag.log         — INFO+ from all modules, appended, rotated at 5 MB
      logs/dag_debug.log   — DEBUG from debug_modules only, rotated at 5 MB
      stderr               — WARNING+ (removed during curses session)

    Session rotation
    ----------------
    If ``dag.log`` (or its numbered size-rotation siblings ``dag.log.1``
    through ``dag.log.3``) already contain data from a previous session,
    they are all renamed to ``dag.log.YYYYMMDD_HHMMSS`` (and
    ``dag.log.YYYYMMDD_HHMMSS.1`` / ``.2`` / ``.3``) before new handlers
    are opened.  No log data is ever deleted.

    Within-session size rotation
    ----------------------------
    ``RotatingFileHandler`` rotates the active ``dag.log`` into ``dag.log.1``
    / ``.2`` / ``.3`` when the file exceeds ``max_bytes``.  These numbered
    backups are separate from the timestamped session archives above.
    """
    log_dir = Path(log_dir) if log_dir else LOG_DIR
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Archive previous session logs before starting fresh ──────────────────
    # Each call to setup_logging() marks the start of a new session.
    # Existing files are renamed to dag.log.YYYYMMDD_HHMMSS so history is
    # preserved.  Empty files (e.g. from a failed prior startup) are skipped.
    for fname in ("dag.log", "dag_debug.log"):
        _rotate_existing_log(log_dir / fname)

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
    fh_main.addFilter(_DeduplicateFilter())
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
    fh_debug.addFilter(_DeduplicateFilter())
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