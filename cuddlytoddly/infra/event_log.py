# infra/event_log.py

# FIX #11: moved `import logging` to module level — it was previously inside
# the except block on every replay iteration (harmless due to Python's import
# cache, but misleading and contrary to PEP 8 style).
import json
import logging
import threading
from pathlib import Path

from cuddlytoddly.core.events import Event

_replay_logger = logging.getLogger("dag.infra.event_log")


class EventLog:
    """
    Append-only JSONL event log.

    Each line is one JSON-serialized event. The file is safe to replay
    after a crash because:
      - append() sanitizes the serialized line to guarantee no embedded
        newlines slip through (LLM results can contain raw \\r\\n)
      - replay() skips and logs any lines that fail to parse rather than
        raising, so a single corrupt entry never blocks a full restore

    Thread safety: a per-instance lock serializes all append() AND clear()
    calls so that no operation can observe a partially-written or partially-
    truncated file.
    """

    def __init__(self, path="event_log.jsonl"):
        self.path = Path(path)
        self.path.touch(exist_ok=True)
        self._lock = threading.Lock()

    def append(self, event: Event):
        # ensure_ascii=False preserves unicode but keeps the output compact;
        # json.dumps always escapes \n inside strings — the extra replace is a
        # safety net for any control characters the LLM smuggles in.
        line = json.dumps(event.to_dict(), ensure_ascii=False)
        # Belt-and-suspenders: strip any literal newlines that somehow survived
        line = line.replace("\r\n", "\\r\\n").replace("\r", "\\r").replace("\n", "\\n")
        with self._lock:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def replay(self):
        """
        Yield Event objects in order, skipping unparseable lines.
        """
        with self.path.open("r", encoding="utf-8") as f:
            for lineno, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    data = json.loads(raw)
                    yield Event.from_dict(data)
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    # Log and skip — one bad line should not abort a full restore
                    _replay_logger.warning(
                        "[EVENT LOG] Skipping corrupt line %d: %s — %s",
                        lineno,
                        repr(raw[:80]),
                        e,
                    )

    def clear(self):
        # FIX: acquire the lock before truncating so that a concurrent append()
        # call (which also holds self._lock while writing) cannot interleave with
        # the truncation.  Without the lock, the two file-open calls race: on
        # Linux, whichever open() wins determines whether the appended line lands
        # before or after the truncation — potentially producing a corrupt log
        # where a WAL entry exists in memory but not on disk.
        #
        # We use an atomic tmp-file + rename pattern rather than write_text("")
        # so that a crash mid-clear leaves either the original file intact or an
        # empty file — never a partially truncated one.
        tmp = self.path.with_suffix(".clearing")
        with self._lock:
            try:
                tmp.write_text("", encoding="utf-8")
                tmp.replace(self.path)
            except OSError:
                tmp.unlink(missing_ok=True)
                raise
