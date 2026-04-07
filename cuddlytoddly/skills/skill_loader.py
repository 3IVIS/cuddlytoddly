# skills/skill_loader.py
#
# Reads the skills/ directory, parses each SKILL.md, registers any local
# tool implementations, and returns:
#   - A ToolRegistry populated with all available tools
#   - A skills summary string ready to inject into the planner prompt
#
# Directory convention
# --------------------
#   skills/
#     <skill_name>/
#       SKILL.md       required — description, tools, when-to-use, output format
#       tools.py       optional — local Python tool implementations
#
# tools.py format
# ---------------
# Define a module-level dict called TOOLS:
#
#   TOOLS = {
#       "tool_name": {
#           "description":  "...",
#           "input_schema": {"arg": "string"},
#           "fn":           lambda args: ...,
#       },
#       ...
#   }

from pathlib import Path

from cuddlytoddly.infra.logging import get_logger

logger = get_logger(__name__)

SKILLS_DIR = Path(__file__).parent  # skills/ directory


# ── Inline ToolRegistry (avoids circular import from engine/tools.py) ────────

class Tool:
    def __init__(self, name, description, input_schema, fn):
        self.name         = name
        self.description  = description
        self.input_schema = input_schema
        self._fn          = fn

    def run(self, input_data):
        return self._fn(input_data)


class ToolRegistry:
    def __init__(self):
        self.tools: dict[str, Tool] = {}

    def register(self, tool: Tool):
        self.tools[tool.name] = tool
        logger.info("[SKILLS] Registered tool: %s", tool.name)

    def execute(self, name: str, input_data: dict):
        if name not in self.tools:
            raise KeyError(f"Tool '{name}' not found")
        return self.tools[name].run(input_data)

    def merge(self, other: "ToolRegistry"):
        """Merge another registry into this one (other wins on collision)."""
        for tool in other.tools.values():
            self.register(tool)


# ── Skill loader ──────────────────────────────────────────────────────────────

class SkillLoader:
    """
    Loads all skills from the skills/ directory.

    Usage
    -----
    loader   = SkillLoader()
    registry = loader.registry          # ToolRegistry with all local tools
    summary  = loader.prompt_summary    # string to inject into planner prompt
    """

    def __init__(self, skills_dir: Path | str = SKILLS_DIR):
        self.skills_dir = Path(skills_dir)
        self._skills: list[dict] = []   # parsed skill metadata
        self.registry = ToolRegistry()
        self._load()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _load(self):
        if not self.skills_dir.exists():
            logger.warning("[SKILLS] Skills directory not found: %s", self.skills_dir)
            return

        for skill_dir in sorted(self.skills_dir.iterdir()):
            if not skill_dir.is_dir():
                continue
            skill_md = skill_dir / "SKILL.md"
            if not skill_md.exists():
                continue

            skill = self._parse_skill_md(skill_dir.name, skill_md)
            self._skills.append(skill)
            logger.info("[SKILLS] Loaded skill: %s", skill_dir.name)

            # Register local tools if tools.py exists
            tools_py = skill_dir / "tools.py"
            if tools_py.exists():
                self._register_local_tools(skill_dir.name, tools_py)

        logger.info("[SKILLS] Loaded %d skill(s), %d local tool(s)",
                    len(self._skills), len(self.registry.tools))

    def _parse_skill_md(self, name: str, path: Path) -> dict:
        """
        Extract the key sections from a SKILL.md into a dict.
        Sections are identified by '## SectionName' headings.
        """
        text     = path.read_text(encoding="utf-8")
        sections = {"name": name, "raw": text}

        current_section = "description"
        buf: list[str] = []

        for line in text.splitlines():
            if line.startswith("## "):
                if buf:
                    sections[current_section] = "\n".join(buf).strip()
                current_section = line[3:].strip().lower().replace(" ", "_")
                buf = []
            elif line.startswith("# "):
                sections["title"] = line[2:].strip()
            else:
                buf.append(line)

        if buf:
            sections[current_section] = "\n".join(buf).strip()

        return sections

    def _register_local_tools(self, skill_name: str, tools_py: Path):
        """Import tools.py from a skill directory and register its TOOLS dict."""
        import importlib.util
        spec   = importlib.util.spec_from_file_location(
            f"skills.{skill_name}.tools", tools_py
        )
        module = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(module)
        except Exception as e:
            logger.error("[SKILLS] Failed to import %s: %s", tools_py, e)
            return

        tools_dict = getattr(module, "TOOLS", None)
        if not isinstance(tools_dict, dict):
            logger.warning("[SKILLS] %s has no TOOLS dict — skipping", tools_py)
            return

        for tool_name, spec_dict in tools_dict.items():
            self.registry.register(Tool(
                name         = tool_name,
                description  = spec_dict.get("description", ""),
                input_schema = spec_dict.get("input_schema", {}),
                fn           = spec_dict["fn"],
            ))

    # ── Planner prompt injection ───────────────────────────────────────────────

    @property
    def prompt_summary(self) -> str:
        """
        A compact skills summary ready to drop into the planner prompt.
        Lists each skill, its when-to-use criteria, and the tools it provides.
        """
        if not self._skills:
            return ""

        lines = ["Available skills (use these to guide task decomposition):"]
        for s in self._skills:
            lines.append(f"\n### {s['name']}")

            desc = s.get("description", "")
            if desc:
                # First non-empty line only, to keep the prompt compact
                first_line = next((ln for ln in desc.splitlines() if ln.strip()), "")
                lines.append(f"  {first_line}")

            when = s.get("when_to_use", "")
            if when:
                lines.append(f"  When to use: {when.splitlines()[0].strip()}")

            tools_section = s.get("tools", "")
            if tools_section:
                tool_names = [
                    ln.strip().lstrip("- ").split(":")[0].strip("`")
                    for ln in tools_section.splitlines()
                    if ln.strip().startswith("-")
                ]
                if tool_names:
                    lines.append(f"  Tools: {', '.join(tool_names)}")

            output_fmt = s.get("expected_output_format", "")
            if output_fmt:
                first_line = next((ln for ln in output_fmt.splitlines() if ln.strip()), "")
                lines.append(f"  Output format: {first_line}")

        return "\n".join(lines)

    def merge_mcp(self, mcp_registry: "ToolRegistry"):
        """Merge an MCP-sourced registry into the local one."""
        self.registry.merge(mcp_registry)


