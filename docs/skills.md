# Skills

Skills are the tool packs that the LLM executor can call during node execution. cuddlytoddly ships two built-in skills and supports adding custom ones with no code changes.

## Built-in skills

### `code_execution`

Runs Python snippets or shell commands in a subprocess and returns stdout.

| Tool | Description |
|---|---|
| `run_python` | Execute a Python code string; returns stdout + stderr |
| `run_shell` | Execute a shell command string; returns stdout + stderr |

### `file_ops`

Read and write files relative to the run's `outputs/` directory.

| Tool | Description |
|---|---|
| `read_file` | Read a file and return its contents as a string |
| `write_file` | Write a string to a file (creates parent dirs automatically) |
| `list_files` | List files in a directory |

### `web_research`

Search the web and fetch page content. Used for tasks that require current information, salary data, market research, company research, news, or any fact that cannot be reliably answered from training knowledge alone.

| Tool | Description |
|---|---|
| `web_search` | Search the web and return a list of results (title, URL, snippet) for a query |
| `fetch_url` | Fetch the content of a URL and return cleaned plain text |

## Adding a custom skill

1. Create a directory under `toddly/skills/`:

```
toddly/skills/
└── my_skill/
    ├── SKILL.md     ← required
    └── tools.py     ← optional (local Python implementations)
```

2. Write a `SKILL.md` describing the skill for the planner:

```markdown
# My Skill

## Description
What this skill does and when to use it.

## When to use
Trigger conditions for the planner to consider this skill.

## Tools
- `my_tool`: Does X given Y.

## Expected output format
A single string containing ...
```

3. Optionally implement local tools in `tools.py`:

```python
# skills/my_skill/tools.py

def _do_thing(args: dict) -> str:
    return f"Result: {args['input']}"

TOOLS = {
    "my_tool": {
        "description": "Does X given Y.",
        "input_schema": {"input": "string"},
        "fn": _do_thing,
    }
}
```

`SkillLoader` discovers the folder automatically at startup, parses `SKILL.md` to build the planner prompt, and registers any tools from `tools.py` into the `ToolRegistry`.

## Using MCP tool servers

For tools that live in an external MCP server, pass a pre-built `ToolRegistry` to `SkillLoader.merge_mcp()`:

```python
from toddly.skills.skill_loader import SkillLoader

skills = SkillLoader()

# Build a registry from your MCP adapter of choice, then merge:
skills.merge_mcp(my_mcp_registry)

registry = skills.registry  # combined local + MCP tools
```