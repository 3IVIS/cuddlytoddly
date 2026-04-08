# skills/file_ops/tools.py
#
# Local tool implementations for the file_ops skill.
# The SkillLoader imports this and registers everything in TOOLS.

from pathlib import Path

TOOLS = {
    "read_file": {
        "description": "Read the full contents of a local file",
        "input_schema": {"path": "string"},
        "fn": lambda args: Path(args["path"]).read_text(encoding="utf-8"),
    },
    "write_file": {
        "description": "Write (or overwrite) a local file with the given content",
        "input_schema": {"path": "string", "content": "string"},
        "fn": lambda args: (
            Path(args["path"]).write_text(args["content"], encoding="utf-8"),
            f"Written {len(args['content'])} chars to {args['path']}",
        )[1],
    },
    "append_file": {
        "description": "Append text to an existing file",
        "input_schema": {"path": "string", "content": "string"},
        "fn": lambda args: (
            Path(args["path"]).open("a", encoding="utf-8").write(args["content"]),
            f"Appended {len(args['content'])} chars to {args['path']}",
        )[1],
    },
    "list_dir": {
        "description": "List files and directories at a path",
        "input_schema": {"path": "string"},
        "fn": lambda args: "\n".join(
            str(p) for p in sorted(Path(args["path"]).iterdir())
        ),
    },
}
