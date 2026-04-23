# --- FILE: toddly/infra/tool_call_log.py ---

"""
toddly.infra.tool_call_log
~~~~~~~~~~~~~~~~~~~~~~~~~~
Append-only, thread-safe JSONL log of every tool call made during a run.

Each line is a self-contained JSON record:

    {
        "ts":           "2026-04-23T10:00:38.412Z",   # ISO-8601 UTC
        "node_id":      "Identify_Search_Terms",
        "tool_name":    "web_search",
        "args":         {"query": "python repo review"},
        "result":       "... full untruncated result ...",
        "result_chars": 4821,
        "duration_ms":  1340,
        "error":        false
    }

The file is written to ``<run_dir>/tool_calls.jsonl`` alongside the
existing ``events.jsonl`` and ``dag.log`` files, making it easy to
grep, tail, or load into a notebook for post-mortem analysis.

Design notes
------------
* ``result`` is the **full** tool output before the executor's
  ``max_tool_result_chars`` truncation.  Truncation only affects what
  the LLM sees; the log always has the complete payload.
* The file is opened in append mode on every write so no handle is held
  open between calls — safe for long-running processes and crash recovery.
* A ``NullToolCallLog`` is provided for tests and configurations where
  logging is disabled, so call sites never need ``if log is not None``
  guards.
"""

from __future__ import annotations

import json
import threading
from datetime import datetime, timezone
from pathlib import Path

from toddly.infra.logging import get_logger

logger = get_logger(__name__)


class ToolCallLog:
    """
    Thread-safe, append-only log of tool calls and their results.

    Parameters
    ----------
    path:
        Destination file.  Created (with parents) if it does not exist.
    """

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.touch(exist_ok=True)
        self._lock = threading.Lock()
        logger.info("[TOOL_LOG] Tool call log: %s", self.path)

    def record(
        self,
        *,
        node_id: str,
        tool_name: str,
        args: dict,
        result: str,
        duration_ms: float,
        error: bool,
    ) -> None:
        """Append one tool-call record to the log.

        Parameters
        ----------
        node_id:
            The DAG node that triggered the call.
        tool_name:
            Registered name of the tool (e.g. ``"web_search"``).
        args:
            Arguments passed to the tool, exactly as the LLM supplied them
            (``_cwd`` injection stripped so internal paths don't clutter the
            log).
        result:
            The **full** tool output before any executor-side truncation.
        duration_ms:
            Wall-clock time from call to return, in milliseconds.
        error:
            ``True`` if the tool raised an exception or returned an
            ``ERROR:``-prefixed string.
        """
        record = {
            "ts": datetime.now(tz=timezone.utc).isoformat(timespec="milliseconds"),
            "node_id": node_id,
            "tool_name": tool_name,
            "args": args,
            "result": result,
            "result_chars": len(result),
            "duration_ms": round(duration_ms, 1),
            "error": error,
        }
        line = json.dumps(record, ensure_ascii=False)
        # Belt-and-suspenders: a tool result could theoretically contain
        # literal newlines (e.g. shell output).  Escape them so each record
        # stays on one line and the file remains valid JSONL.
        line = line.replace("\r\n", "\\r\\n").replace("\r", "\\r").replace("\n", "\\n")
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


class NullToolCallLog:
    """No-op drop-in for tests or when tool-call logging is disabled."""

    def record(self, **kwargs) -> None:  # noqa: D102
        pass
