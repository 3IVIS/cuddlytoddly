# skills/code_execution/tools.py

import subprocess
import sys
from cuddlytoddly.infra.logging import get_logger

logger = get_logger(__name__)

def _run_python(args):
    code = args["code"]

    # JSON encoding sometimes produces literal \n instead of real newlines
    code = code.replace("\\n", "\n").replace("\\t", "\t")

    # Strip markdown fences
    import re
    code = re.sub(r'^```(?:python)?\s*', '', code.strip())
    code = re.sub(r'\s*```$', '', code.strip())

    logger.info("[RUN_PYTHON] Executing: %.500s", code)

    try:
        result = eval(code, {"__builtins__": __builtins__})
        return str(result)
    except SyntaxError:
        pass

    import io, contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        exec(code, {"__builtins__": __builtins__})
    return buf.getvalue() or "(no output)"


def _run_shell(args):
    result = subprocess.run(
        args["command"],
        shell=True,
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
        "description":  "Run a shell command and return stdout",
        "input_schema": {"command": "string"},
        "fn": _run_shell,
    },
}
