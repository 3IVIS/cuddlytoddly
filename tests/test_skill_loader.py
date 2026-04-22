"""Tests for cuddlytoddly.skills.skill_loader."""

import shutil
from pathlib import Path

import pytest

from toddly.skills.skill_loader import SkillLoader, Tool, ToolRegistry

# ── ToolRegistry ──────────────────────────────────────────────────────────────


class TestToolRegistry:
    def test_register_and_execute(self):
        registry = ToolRegistry()
        tool = Tool(
            name="add",
            description="adds two numbers",
            input_schema={"a": "string", "b": "string"},
            fn=lambda args: str(int(args["a"]) + int(args["b"])),
        )
        registry.register(tool)
        result = registry.execute("add", {"a": "3", "b": "4"})
        assert result == "7"

    def test_execute_unknown_tool_raises(self):
        registry = ToolRegistry()
        with pytest.raises(KeyError):
            registry.execute("ghost_tool", {})

    def test_register_overwrites_on_duplicate(self):
        registry = ToolRegistry()
        registry.register(Tool("t", "v1", {}, lambda args: "v1"))
        registry.register(Tool("t", "v2", {}, lambda args: "v2"))
        assert registry.execute("t", {}) == "v2"

    def test_merge_combines_registries(self):
        r1 = ToolRegistry()
        r1.register(Tool("tool_a", "", {}, lambda args: "a"))
        r2 = ToolRegistry()
        r2.register(Tool("tool_b", "", {}, lambda args: "b"))
        r1.merge(r2)
        assert "tool_a" in r1.tools
        assert "tool_b" in r1.tools

    def test_merge_second_wins_on_collision(self):
        r1 = ToolRegistry()
        r1.register(Tool("t", "", {}, lambda args: "first"))
        r2 = ToolRegistry()
        r2.register(Tool("t", "", {}, lambda args: "second"))
        r1.merge(r2)
        assert r1.execute("t", {}) == "second"


# ── SkillLoader with real skills ──────────────────────────────────────────────
# The built-in skills (file_ops, code_execution) ship tools.py but not SKILL.md.
# This fixture copies those skills into a tmp dir and adds the missing SKILL.md
# so SkillLoader can discover them, matching the intended package behaviour.


def _build_real_skills_dir(tmp_path: Path) -> Path:
    from toddly.skills.skill_loader import SKILLS_DIR

    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    for skill_name in ["file_ops", "code_execution"]:
        src = SKILLS_DIR / skill_name
        if not src.exists():
            continue
        dst = skills_dir / skill_name
        shutil.copytree(src, dst)
        skill_md = dst / "SKILL.md"
        if not skill_md.exists():
            skill_md.write_text(
                f"# {skill_name}\n\n"
                f"## Description\n{skill_name} built-in skill.\n\n"
                f"## When to use\nUse for {skill_name} tasks.\n\n"
                f"## Tools\n- tool: Executes {skill_name} operations.\n"
            )
    return skills_dir


class TestSkillLoaderWithRealSkills:
    @pytest.fixture
    def loader(self, tmp_path):
        skills_dir = _build_real_skills_dir(tmp_path)
        return SkillLoader(skills_dir=skills_dir)

    def test_loads_file_ops_skill(self, loader):
        assert "read_file" in loader.registry.tools
        assert "write_file" in loader.registry.tools

    def test_loads_code_execution_skill(self, loader):
        assert "run_python" in loader.registry.tools
        assert "run_shell" in loader.registry.tools

    def test_prompt_summary_is_non_empty(self, loader):
        assert len(loader.prompt_summary) > 0

    def test_prompt_summary_contains_skill_names(self, loader):
        summary = loader.prompt_summary
        assert "file_ops" in summary or "code_execution" in summary

    def test_write_file_tool_works(self, loader, tmp_path):
        path = str(tmp_path / "test.txt")
        loader.registry.execute("write_file", {"path": path, "content": "hello"})
        assert Path(path).read_text() == "hello"

    def test_read_file_tool_works(self, loader, tmp_path):
        path = tmp_path / "read_me.txt"
        path.write_text("read this")
        result = loader.registry.execute("read_file", {"path": str(path)})
        assert result == "read this"

    def test_run_python_tool_works(self, loader):
        # Use an expression rather than a print statement — eval() returns the
        # value directly, whereas print() returns None (stdout is not captured
        # on the eval path).
        result = loader.registry.execute("run_python", {"code": "1 + 1"})
        assert result == "2"

    def test_run_shell_tool_works(self, loader):
        result = loader.registry.execute("run_shell", {"command": "echo hello"})
        assert "hello" in result


# ── SkillLoader with synthetic skill dir ──────────────────────────────────────


class TestSkillLoaderSynthetic:
    @pytest.fixture
    def skills_dir(self, tmp_path):
        skill_dir = tmp_path / "my_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(
            "# My Skill\n\n"
            "## Description\nDoes cool things.\n\n"
            "## When to use\nWhen you need cool things.\n\n"
            "## Tools\n- `do_thing`: Does the thing.\n\n"
            "## Expected output format\nA string with the result.\n"
        )
        (skill_dir / "tools.py").write_text(
            "TOOLS = {\n"
            '    "do_thing": {\n'
            '        "description": "Does the thing.",\n'
            '        "input_schema": {"input": "string"},\n'
            '        "fn": lambda args: f"did: {args[\'input\']}",\n'
            "    }\n"
            "}\n"
        )
        return tmp_path

    def test_loads_synthetic_skill(self, skills_dir):
        loader = SkillLoader(skills_dir=skills_dir)
        assert "do_thing" in loader.registry.tools

    def test_executes_synthetic_tool(self, skills_dir):
        loader = SkillLoader(skills_dir=skills_dir)
        result = loader.registry.execute("do_thing", {"input": "foo"})
        assert result == "did: foo"

    def test_prompt_summary_includes_skill(self, skills_dir):
        loader = SkillLoader(skills_dir=skills_dir)
        assert "my_skill" in loader.prompt_summary

    def test_skill_without_tools_py_loaded_but_no_tools(self, tmp_path):
        skill_dir = tmp_path / "no_tools_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# NoTools\n\n## Description\nNothing.\n")
        loader = SkillLoader(skills_dir=tmp_path)
        assert "no_tools_skill" in loader.prompt_summary
        assert len(loader.registry.tools) == 0

    def test_skill_dir_not_found_does_not_crash(self, tmp_path):
        loader = SkillLoader(skills_dir=tmp_path / "nonexistent")
        assert loader.prompt_summary == ""
        assert len(loader.registry.tools) == 0

    def test_broken_tools_py_skipped_gracefully(self, tmp_path):
        skill_dir = tmp_path / "broken_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# Broken\n\n## Description\nBroken tools.\n")
        (skill_dir / "tools.py").write_text("this is not valid python )()(")
        loader = SkillLoader(skills_dir=tmp_path)
        assert len(loader.registry.tools) == 0

    def test_tools_py_without_tools_dict_skipped(self, tmp_path):
        skill_dir = tmp_path / "no_dict_skill"
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text("# NoDictSkill\n\n## Description\nNo dict.\n")
        (skill_dir / "tools.py").write_text("TOOLS = 'not a dict'\n")
        loader = SkillLoader(skills_dir=tmp_path)
        assert len(loader.registry.tools) == 0

    def test_merge_mcp_adds_external_tools(self, skills_dir):
        loader = SkillLoader(skills_dir=skills_dir)
        external = ToolRegistry()
        external.register(Tool("mcp_tool", "external", {}, lambda args: "mcp"))
        loader.merge_mcp(external)
        assert "mcp_tool" in loader.registry.tools
        assert "do_thing" in loader.registry.tools
