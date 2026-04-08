# infra/event_log.py

import json
import threading
from pathlib import Path

from cuddlytoddly.core.events import Event


class EventLog:
    """
    Append-only JSONL event log.

    Each line is one JSON-serialized event. The file is safe to replay
    after a crash because:
      - append() sanitizes the serialized line to guarantee no embedded
        newlines slip through (LLM results can contain raw \\r\\n)
      - replay() skips and logs any lines that fail to parse rather than
        raising, so a single corrupt entry never blocks a full restore

    Thread safety: a per-instance lock serializes all append() calls so
    concurrent threads cannot interleave their writes (critical on Windows
    where open-append-close from two threads can silently drop lines).
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
                    import logging

                    logging.getLogger("dag.infra.event_log").warning(
                        "[EVENT LOG] Skipping corrupt line %d: %s — %s",
                        lineno,
                        repr(raw[:80]),
                        e,
                    )

    def clear(self):
        self.path.write_text("", encoding="utf-8")