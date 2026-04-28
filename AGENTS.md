# AGENTS.md

## Quick start

- Install dependencies and run the standard build before starting new work.

## Architecture & layout

- Start with `README.md` when you need repo orientation or architectural context.
- Repository: cuddlytoddly.
- Primary languages: Python.
- Key source directories: `tests/`.
- Library-style project; expect reusable crates and packages.
- CI workflows detected under `.github/workflows/`; match those expectations locally.

## Important instructions

- Use Conventional Commits (`type(scope): subject`).

## Code style

- Follow PEP 8, prefer Black-compatible formatting, and add type hints when practical.

## Testing

- Keep CI green by mirroring workflow steps locally before pushing.

## Performance & simplicity

- Do not guess at bottlenecks; measure before optimizing.
- Prefer simple algorithms and data structures until workload data proves otherwise.
- Keep performance changes surgical and behavior-preserving.

## PR guidelines

- Write descriptive, imperative commit messages.
- Reference issues with `Fixes #123` or `Closes #123` when applicable.
- Keep pull requests focused and include test evidence for non-trivial changes.

## Additional guidance

- Preferred orientation doc: `README.md`.
- Repository docs spotted: CONTRIBUTING.md, LICENSE, README.md.

