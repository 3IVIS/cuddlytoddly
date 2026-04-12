# skills/code_execution/tools.py
#
# SECURITY NOTE: run_python and run_shell execute arbitrary code/commands with
# the same privileges as the cuddlytoddly process.  They are intentionally
# unrestricted so that the LLM can perform real filesystem and network work.
# Do NOT expose these tools in untrusted environments or over a network-
# accessible interface without additional sandboxing (e.g. a container or
# seccomp profile).

import subprocess

from cuddlytoddly.infra.logging import get_logger

logger = get_logger(__name__)


def _run_python(args):
    code = args["code"]

    # JSON encoding sometimes produces literal \n instead of real newlines
    code = code.replace("\\n", "\n").replace("\\t", "\t")

    # Strip markdown fences
    import re

    code = re.sub(r"^```(?:python)?\s*", "", code.strip())
    code = re.sub(r"\s*```$", "", code.strip())

    logger.info("[RUN_PYTHON] Executing: %.500s", code)

    # Try eval first so single expressions return their value directly.
    try:
        result = eval(code, {"__builtins__": __builtins__})
        return str(result)
    except SyntaxError:
        pass
    except Exception as e:
        # eval() raised a runtime error (not a syntax error) — report it
        # cleanly rather than falling through to exec().
        return f"ERROR: {e}"

    # Fall back to exec() for multi-statement code blocks.
    import contextlib
    import io

    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            exec(code, {"__builtins__": __builtins__})  # noqa: S102
    except Exception as e:
        # FIX: exec() was previously unguarded.  Any runtime exception
        # (NameError, ZeroDivisionError, etc.) would propagate up and surface
        # as an opaque tool failure rather than a clean "ERROR: ..." string
        # that the LLM can reason about and correct.
        captured = buf.getvalue()
        error_msg = f"ERROR: {e}"
        if captured:
            error_msg = captured + "\n" + error_msg
        return error_msg

    return buf.getvalue() or "(no output)"


def _run_shell(args):
    result = subprocess.run(
        args["command"],
        shell=True,  # noqa: S602  (intentional — see module docstring)
        capture_output=True,
        text=True,
        timeout=30,
    )
    output = result.stdout.strip()
    if result.returncode != 0:
        output += f"\n[stderr] {result.stderr.strip()}"
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
