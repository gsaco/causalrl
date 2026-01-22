"""Validate docs estimator pages against code implementations."""

from __future__ import annotations

import importlib
import re
import sys
from pathlib import Path

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required to run docs validation.") from exc


ESTIMATOR_SECTION = "Estimator Reference"
IMPLEMENTATION_RE = re.compile(r"^Implementation:\s*`([^`]+)`", re.MULTILINE)


def _collect_estimator_pages(nav: list) -> list[str]:
    pages: list[str] = []

    def _walk(items):
        if isinstance(items, dict):
            for key, value in items.items():
                if key == ESTIMATOR_SECTION:
                    _walk(value)
                else:
                    _walk(value)
        elif isinstance(items, list):
            for item in items:
                _walk(item)
        elif isinstance(items, str):
            if items.startswith("reference/estimators/") and items.endswith(".md"):
                if items.endswith("index.md"):
                    return
                pages.append(items)

    _walk(nav)
    return pages


def _load_impl_path(md_path: Path) -> str:
    text = md_path.read_text(encoding="utf-8")
    match = IMPLEMENTATION_RE.search(text)
    if not match:
        raise ValueError(f"Missing Implementation line in {md_path}")
    return match.group(1).strip()


def _validate_import(path: str) -> None:
    if "." not in path:
        raise ValueError(f"Invalid implementation path '{path}'")
    module_name, class_name = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    if not hasattr(module, class_name):
        raise ValueError(f"{class_name} not found in module {module_name}")


def main() -> int:
    mkdocs_path = Path("mkdocs.yml")
    if not mkdocs_path.exists():
        print("mkdocs.yml not found", file=sys.stderr)
        return 1

    config = yaml.safe_load(mkdocs_path.read_text(encoding="utf-8"))
    nav = config.get("nav", [])
    pages = _collect_estimator_pages(nav)
    if not pages:
        print("No estimator pages found in mkdocs nav.")
        return 1

    errors: list[str] = []
    for page in pages:
        md_path = Path("docs") / page
        if not md_path.exists():
            errors.append(f"Missing doc file: {md_path}")
            continue
        try:
            impl_path = _load_impl_path(md_path)
            _validate_import(impl_path)
        except Exception as exc:  # pragma: no cover - validation errors
            errors.append(f"{md_path}: {exc}")

    if errors:
        print("Docs validation failed:")
        for error in errors:
            print(f"- {error}")
        return 1

    print(f"Docs validation passed ({len(pages)} estimator pages).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
