"""Check for unwanted build artifacts and OS metadata."""

from __future__ import annotations

import sys
from pathlib import Path


BAD_DIR_NAMES = {
    "__pycache__",
    "__MACOSX",
}
BAD_FILE_NAMES = {
    ".DS_Store",
}
BAD_SUFFIXES = {
    ".pyc",
}
BAD_TOP_LEVEL_DIRS = {
    "dist",
    "build",
    "site",
}

IGNORE_PREFIXES = (
    ".venv",
    ".mypy_cache",
    ".pytest_cache",
    ".hypothesis",
    ".ruff_cache",
)


def _is_ignored(path: Path, root: Path) -> bool:
    try:
        rel = path.relative_to(root)
    except ValueError:
        return True
    if not rel.parts:
        return True
    first = rel.parts[0]
    return first.startswith(IGNORE_PREFIXES)


def main() -> int:
    root = Path.cwd()
    violations: list[str] = []

    for path in root.rglob("*"):
        if _is_ignored(path, root):
            continue
        rel = path.relative_to(root)

        if path.is_dir():
            if rel.parts and rel.parts[0] in BAD_TOP_LEVEL_DIRS:
                violations.append(str(rel))
                continue
            if path.name in BAD_DIR_NAMES or path.name.endswith(".egg-info"):
                violations.append(str(rel))
            continue

        if path.is_file():
            if path.name in BAD_FILE_NAMES:
                violations.append(str(rel))
                continue
            if path.suffix in BAD_SUFFIXES:
                violations.append(str(rel))

    if violations:
        print("Repository hygiene check failed. Remove the following artifacts:")
        for item in sorted(set(violations)):
            print(f"- {item}")
        return 1

    print("Repository hygiene check passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
