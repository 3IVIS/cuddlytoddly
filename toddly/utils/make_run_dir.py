"""
cuddly.utils.make_run_dir
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Create a timestamped, collision-safe directory for a single agent run.

Extracted from cuddlytoddly/__main__.py so any host application can reuse
the logic without importing the full cuddlytoddly package.

Usage::

    from cuddly.utils.make_run_dir import make_run_dir
    from cuddlytoddly.config import DATA_DIR

    run_dir = make_run_dir("build a todo app", base_dir=DATA_DIR)
"""

from __future__ import annotations

import secrets
import time
from pathlib import Path


def make_run_dir(goal_text: str, base_dir: Path) -> Path:
    """
    Create and return a unique run directory under ``base_dir/runs/``.

    The directory name is ``<slug>_<timestamp>_<random>`` where:

    * ``slug``      – the goal text sanitised to ``[a-z0-9_]``, max 60 chars.
    * ``timestamp`` – full Unix epoch seconds (not truncated).
    * ``random``    – 8 cryptographically random hex chars.

    This triple makes collisions effectively impossible even for goals
    started within the same second.  ``mkdir(exist_ok=False)`` is used so
    any (vanishingly unlikely) collision raises immediately rather than
    silently sharing a directory.

    Also creates the ``outputs/`` subdirectory expected by file-ops tools.

    Raises
    ------
    FileExistsError
        On the extraordinarily unlikely event of a name collision.
    """
    safe = goal_text.lower().replace(" ", "_")
    safe = "".join(c for c in safe if c.isalnum() or c == "_")[:60]
    if not safe:
        # Goal text was pure unicode / emoji — use a neutral fallback so the
        # directory name is never just "_{ts}_{rand}" which confuses humans.
        safe = "goal"

    ts = str(int(time.time()))
    rand = secrets.token_hex(4)  # 8 random hex chars
    run_dir = base_dir / "runs" / f"{safe}_{ts}_{rand}"
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "outputs").mkdir(exist_ok=True)
    return run_dir
