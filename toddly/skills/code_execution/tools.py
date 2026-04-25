# skills/code_execution/tools.py
#
# SECURITY NOTE: run_python and run_shell execute arbitrary code/commands with
# the same privileges as the cuddlytoddly process.  They are intentionally
# unrestricted so that the LLM can perform real filesystem and network work.
# Do NOT expose these tools in untrusted environments or over a network-
# accessible interface without additional sandboxing (e.g. a container or
# seccomp profile).

import contextlib
import io
import os
import re
import signal
import subprocess
import threading

from toddly.infra.logging import get_logger

logger = get_logger(__name__)

# Maximum characters returned by any tool before the output is truncated.
# Mirrors the cap used by web_research/tools.py and prevents a runaway
# command (e.g. `yes`, `cat /dev/urandom`) from exhausting process memory.
_MAX_OUTPUT_CHARS = 8000

# Lock narrowed to in-process Python execution only.
#
# Python's os.chdir() is process-wide, so concurrent eval/exec calls that
# each need a different CWD must still serialise around the chdir.  However,
# in-process Python snippets are typically short-lived (data manipulation,
# text processing) so holding this lock is far less costly than the old
# _CWD_LOCK in llm_executor.py which was held for the full duration of every
# tool call — including 30-second shell commands.
#
# Shell commands avoid this lock entirely by using subprocess cwd= instead.
_PYTHON_CWD_LOCK = threading.Lock()


def _truncate(text: str, label: str = "output") -> str:
    """Truncate *text* to _MAX_OUTPUT_CHARS and append a notice when trimmed."""
    if len(text) <= _MAX_OUTPUT_CHARS:
        return text
    return text[:_MAX_OUTPUT_CHARS] + f"\n[{label} truncated at {_MAX_OUTPUT_CHARS} chars]"


# ---------------------------------------------------------------------------
# Python execution — inner helper (no CWD logic)
# ---------------------------------------------------------------------------


def _eval_code(code: str) -> str:
    """Evaluate *code* in the current working directory and return a result string."""
    # Try eval first so single expressions return their value directly.
    try:
        result = eval(code, {"__builtins__": __builtins__})
        return _truncate(str(result))
    except SyntaxError:
        pass
    except Exception as e:
        # eval() raised a runtime error (not a syntax error) — report it
        # cleanly rather than falling through to exec().
        return f"ERROR: {e}"

    # Fall back to exec() for multi-statement code blocks.
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {"__builtins__": __builtins__})  # noqa: S102
    except Exception as e:
        # exec() was previously unguarded.  Any runtime exception
        # (NameError, ZeroDivisionError, etc.) would propagate up and surface
        # as an opaque tool failure rather than a clean "ERROR: ..." string
        # that the LLM can reason about and correct.
        captured = buf.getvalue()
        error_msg = f"ERROR: {e}"
        if captured:
            error_msg = _truncate(captured) + "\n" + error_msg
        return error_msg

    return _truncate(buf.getvalue()) or "(no output)"


# ---------------------------------------------------------------------------
# Tool implementations
# ---------------------------------------------------------------------------


def _run_python(args):
    # Pop the injected working-directory key before touching user code so
    # it never leaks into the execution namespace or appears in error messages.
    cwd = args.pop("_cwd", None)

    code = args["code"]

    # JSON encoding sometimes produces literal \n instead of real newlines
    code = code.replace("\\n", "\n").replace("\\t", "\t")

    # Strip markdown fences
    code = re.sub(r"^```(?:python)?\s*", "", code.strip())
    code = re.sub(r"\s*```$", "", code.strip())

    logger.info("[RUN_PYTHON] Executing: %.500s", code)

    if cwd is not None:
        # Hold _PYTHON_CWD_LOCK only for the duration of in-process
        # execution, not for the entire tool call.  This is much narrower than
        # the old process-wide _CWD_LOCK in llm_executor.py which was held
        # across all tool types including 30-second shell commands.
        with _PYTHON_CWD_LOCK:
            prev_cwd = os.getcwd()
            try:
                os.chdir(cwd)
                return _eval_code(code)
            finally:
                os.chdir(prev_cwd)

    return _eval_code(code)


def _run_shell(args):
    #
    # 1. ORPHAN-PROCESS LEAK — subprocess.run(timeout=...) kills only the
    #    direct child; grandchildren keep running.  We now launch in its own
    #    process group (start_new_session=True) and on timeout send SIGKILL
    #    to the entire group via os.killpg.
    #
    # 2. UNHANDLED TimeoutExpired — now caught and returned as a clean
    #    "ERROR: ..." string the LLM can reason about.
    #
    # 3. UNBOUNDED OUTPUT — both stdout and stderr are truncated to
    #    _MAX_OUTPUT_CHARS before being returned.
    #
    # 4. CWD WITHOUT LOCK — passes cwd= to subprocess.Popen so the shell
    #    starts in the correct working directory without mutating the process
    #    CWD or holding any lock.  None means inherit process CWD (safe for
    #    single-run terminal mode and for isolated web-mode runs).

    # Pop the injected key so it doesn't appear in the command string.
    cwd = args.pop("_cwd", None)
    command = args["command"]

    try:
        proc = subprocess.Popen(
            command,
            shell=True,  # noqa: S602  (intentional — see module docstring)
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,  # child gets its own process group
            cwd=cwd,  # no os.chdir(); no lock required
        )

        try:
            stdout, stderr = proc.communicate(timeout=30)
        except subprocess.TimeoutExpired:
            # Kill the entire process group so forked children are also reaped.
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except ProcessLookupError:
                pass  # process already exited between the timeout and the kill
            proc.wait()
            return "ERROR: command timed out after 30 seconds"

    except Exception as e:
        return f"ERROR: {e}"

    output = _truncate(stdout.strip())
    if proc.returncode != 0:
        stderr_snippet = _truncate(stderr.strip(), label="stderr")
        if stderr_snippet:
            output += f"\n[stderr] {stderr_snippet}"
    return output or "(no output)"


TOOLS = {
    "run_python": {
        "description": (
            "Execute a Python code block and return stdout. "
            "Use \\n for line breaks and 4 spaces for indentation. "
            "Do NOT compress multi-line code onto one line with semicolons — "
            "compound statements (for, if, while, def) require proper newlines."
        ),
        "input_schema": {"code": "string"},
        "fn": _run_python,
    },
    "run_shell": {
        "description": "Run a shell command and return stdout",
        "input_schema": {"command": "string"},
        "fn": _run_shell,
    },
}
