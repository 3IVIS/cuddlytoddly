# Contributing to cuddlytoddly

Thank you for your interest in contributing! cuddlytoddly is an open-source project and contributions of all kinds are welcome — bug fixes, new features, documentation improvements, new skills, and additional backend support.

---

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [Ways to Contribute](#ways-to-contribute)
- [Getting Started](#getting-started)
- [Development Workflow](#development-workflow)
- [Project Structure](#project-structure)
- [Adding a New Skill](#adding-a-new-skill)
- [Adding a New Backend](#adding-a-new-backend)
- [Testing](#testing)
- [Commit Style](#commit-style)
- [Pull Request Guidelines](#pull-request-guidelines)
- [Reporting Bugs](#reporting-bugs)
- [Requesting Features](#requesting-features)

---

## Code of Conduct

Be respectful, constructive, and collaborative. We're all here to build something useful together.

---

## Ways to Contribute

You don't have to write code to contribute meaningfully:

- **Fix a bug** — browse [open issues](https://github.com/3IVIS/cuddlytoddly/issues) labelled `bug`
- **Implement a feature** — pick up an issue labelled `enhancement` or `help wanted`
- **Add a skill** — the easiest entry point; no core code changes required (see [Adding a New Skill](#adding-a-new-skill))
- **Add a backend** — bring support for a new LLM provider (see [Adding a New Backend](#adding-a-new-backend))
- **Improve documentation** — fix typos, clarify explanations, add examples
- **Share a demo run** — submit an interesting interactive snapshot as an example
- **Report a bug** — a well-written issue is a genuine contribution
- **Propose a feature** — open a discussion before building something large

---

## Getting Started

### Prerequisites

- Python 3.11 or higher
- `git` on your PATH
- At least one of: `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, or a local llama.cpp installation

### Fork and clone

```bash
# 1. Fork the repo on GitHub, then clone your fork
git clone https://github.com/<your-username>/cuddlytoddly.git
cd cuddlytoddly

# 2. Add the upstream remote so you can pull future changes
git remote add upstream https://github.com/3IVIS/cuddlytoddly.git
```

### Install in development mode

```bash
# Install all extras plus development dependencies
pip install -e ".[all,dev]"
```

### Verify your setup

```bash
# Run the test suite to confirm everything is working
pytest

# Run a quick smoke test against your preferred backend
export ANTHROPIC_API_KEY=sk-ant-...
cuddlytoddly "List three ways to learn Python"
```

---

## Development Workflow

cuddlytoddly follows a standard **fork → branch → PR** flow.

```
upstream/main  ←─────────── your PR
      │
      └──→ your fork/main
                 │
                 └──→ feature/your-branch  ← your work lives here
```

### Step by step

```bash
# 1. Keep your fork's main up to date
git fetch upstream
git checkout main
git merge upstream/main

# 2. Create a focused branch for your change
git checkout -b feature/my-new-skill
# or: fix/crash-on-empty-graph
# or: docs/clarify-config-options

# 3. Make your changes, commit often
git add .
git commit -m "feat(skills): add web-scraping skill with BeautifulSoup"

# 4. Push and open a PR against upstream/main
git push origin feature/my-new-skill
```

Then open a pull request on GitHub from your branch to `3IVIS/cuddlytoddly:main`.

---

## Project Structure

Understanding where things live will help you find the right place for your change:

```
cuddlytoddly/
├── core/
│   └── task_graph.py          # TaskGraph — the mutable DAG at the heart of everything
├── planning/
│   ├── llm_interface.py       # create_llm_client() — backend abstraction layer
│   ├── llm_planner.py         # LLMPlanner — goal → DAG decomposition
│   └── llm_executor.py        # LLMExecutor — per-task LLM execution loop
├── engine/
│   ├── orchestrator.py        # Orchestrator — concurrency, scheduling, event loop
│   └── quality_gate.py        # QualityGate — output verification + gap bridging
├── skills/
│   ├── skill_loader.py        # Discovers and loads SKILL.md skill folders
│   └── builtin/               # Built-in skills (code execution, file I/O, web access)
├── ui/
│   ├── terminal/              # Curses-based terminal UI
│   └── web/                   # Web UI — task graph visualiser, node editor
├── config.py                  # Config loading, CONFIG_PATH, defaults
docs/
├── architecture.md
├── configuration.md
├── skills.md
└── api.md
skills/                        # Drop custom skill folders here
tests/
```

---

## Adding a New Skill

Skills are the easiest way to extend cuddlytoddly — no changes to core code required. A skill is a folder containing two files:

```
skills/
└── my_skill/
    ├── SKILL.md     # Description of what the skill does (shown to the LLM planner)
    └── tools.py     # Tool implementations exposed to the LLM executor
```

### SKILL.md

Write a clear, concise description of what the skill does and when to use it. This text is injected directly into the planner's prompt, so plain English works best:

```markdown
# my_skill

Use this skill when the task requires [describe the use case].

## Available tools

- `tool_name(arg1, arg2)` — does X, returns Y
- `another_tool(arg)` — does Z
```

### tools.py

Implement your tools as plain Python functions decorated with `@tool`:

```python
from cuddlytoddly.skills.registry import tool

@tool
def tool_name(arg1: str, arg2: int) -> str:
    """Does X. Returns Y as a string."""
    # your implementation
    return result
```

### Testing your skill

Drop the folder into the `skills/` directory at the project root and run a goal that would naturally invoke it:

```bash
cuddlytoddly "Your goal that exercises the new skill"
```

Check that the planner picks up the skill in its summary and that the executor calls it correctly.

---

## Adding a New Backend

Backends live behind the `create_llm_client()` abstraction in `agent_core/planning/llm_interface.py`. To add support for a new LLM provider:

1. **Add a client class** in `agent_core/planning/llm_interface.py` that implements the same interface as the existing `ApiLLM` and `LlamaCppLLM` classes (i.e. an `ask()` method with the same signature, plus optional `ask_with_tools()` for native tool-use APIs).

2. **Register the backend name** in the `create_llm_client()` factory function in that same file.

3. **Add a pip extra** in `pyproject.toml` (e.g. `[project.optional-dependencies]` → `myprovider = ["their-sdk"]`).

4. **Update the backends table** in `README.md` and `docs/configuration.md`.

5. **Add an integration test** in `tests/test_backends.py` that skips unless the relevant API key is present.

Please open an issue first if you're planning a new backend — it's a good way to align on the interface before writing code.

---

## Testing

```bash
# Run the full test suite
pytest

# Run a specific file
pytest tests/test_task_graph.py

# Run with coverage
pytest --cov=cuddlytoddly

# Run only fast unit tests (skip integration tests that hit real APIs)
pytest -m "not integration"
```

If you're adding new functionality, please include:
- **Unit tests** for logic that can be tested in isolation
- **An integration test** (marked `@pytest.mark.integration`) for anything that exercises a real LLM call, gated on the relevant env var being set

---

## Commit Style

We use [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <short description>

[optional body]

[optional footer]
```

**Types:** `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`

**Scopes** (use the relevant module): `planner`, `executor`, `graph`, `orchestrator`, `quality-gate`, `skills`, `ui`, `config`, `backends`

**Examples:**

```
feat(skills): add PDF extraction skill using pdfplumber
fix(orchestrator): prevent duplicate task dispatch on fast resume
docs(skills): add example SKILL.md template
refactor(planner): extract context extraction into its own method
test(graph): add edge cases for cycle detection in PlanConstraintChecker
```

Keep the subject line under 72 characters. Use the body to explain *why*, not *what*.

---

## Pull Request Guidelines

- **Keep PRs focused** — one logical change per PR makes review much faster
- **Reference the issue** — include `Closes #123` or `Fixes #123` in the PR description if applicable
- **Fill in the PR template** — describe what changed, why, and how you tested it
- **Pass CI** — all tests must pass before a PR will be reviewed
- **Keep commits clean** — squash fixup commits before requesting review (`git rebase -i`)
- **Be responsive** — if a reviewer asks a question, try to respond within a few days

### PR description template

```markdown
## What does this PR do?
<!-- A concise summary of the change -->

## Why?
<!-- The motivation — link to issue if applicable -->

## How was it tested?
<!-- What did you run to verify it works? -->

## Checklist
- [ ] Tests added or updated
- [ ] Docs updated if behaviour changed
- [ ] `CONTRIBUTING.md` updated if new patterns introduced
```

---

## Reporting Bugs

Please use the [GitHub Issues](https://github.com/3IVIS/cuddlytoddly/issues) page and include:

- **cuddlytoddly version** (`pip show cuddlytoddly`)
- **Python version** (`python --version`)
- **Backend and model** (e.g. `claude / claude-opus-4-6`)
- **Operating system**
- **What you did** — the goal you ran, or the code you called
- **What you expected** to happen
- **What actually happened** — include the full error traceback if there is one
- **Event log** if available — the JSONL file from the run's output directory (remove any API keys first)

---

## Requesting Features

Open a [GitHub Issue](https://github.com/3IVIS/cuddlytoddly/issues) with the label `enhancement`. Describe:

- The problem you're trying to solve (not just the solution)
- How you'd expect it to work from a user perspective
- Any ideas you have on implementation

For large changes (new subsystems, breaking changes to the API), please open a discussion issue before writing code so we can align on the design first.

---

## Questions?

Open a [GitHub Discussion](https://github.com/3IVIS/cuddlytoddly/discussions) or file an issue labelled `question`. We're happy to help.