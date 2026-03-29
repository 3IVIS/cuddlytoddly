from pathlib import Path

ROOT_DIR = Path(".").resolve()
OUTPUT_FILE = (ROOT_DIR / "bundle.py").resolve()
EXCLUDE_KEYWORD = "__pycache__"
THIS_FILE = Path(__file__).resolve()

# File extensions/patterns you want to include
EXTENSIONS = ["*.py", "*.html", "*.js", "*.md", "*.toml", ".gitignore"]

# Explicit filenames (no extension) you want to include
EXTRA_FILES = ["LICENSE"]

with OUTPUT_FILE.open("w", encoding="utf-8") as out:
    # Handle extension-based patterns
    for pattern in EXTENSIONS:
        for path in sorted(ROOT_DIR.rglob(pattern)):
            path = path.resolve()

            if EXCLUDE_KEYWORD in path.name.lower():
                continue
            if path in {THIS_FILE, OUTPUT_FILE}:
                continue

            out.write(f"\n\n# --- FILE: {path.relative_to(ROOT_DIR)} ---\n\n")
            out.write(path.read_text(encoding="utf-8"))

    # Handle explicit files with no extension
    for filename in EXTRA_FILES:
        path = ROOT_DIR / filename
        if path.exists():
            out.write(f"\n\n# --- FILE: {path.relative_to(ROOT_DIR)} ---\n\n")
            out.write(path.read_text(encoding="utf-8"))

print(f"Bundled files into {OUTPUT_FILE}")
