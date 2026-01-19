# Release Process

This checklist documents the release workflow for CausalRL.

## 1. Prep and version bump

1. Update `crl/version.py`.
2. Update `CHANGELOG.md` with a new version section.
3. Update `CITATION.cff` (version) and `CITATION.bib` if the year/version changes.

## 2. Run quality gates

```bash
ruff check .
ruff format --check .
mypy crl
pytest
make notebooks-smoke
mkdocs build --strict
python -m build
twine check dist/*
```

## 3. Tag and release

```bash
git tag vX.Y.Z
git push origin vX.Y.Z
```

Create a GitHub release with the changelog highlights.

## 4. Publish to PyPI

```bash
twine upload dist/*
```

## 5. Post-release

- Verify the PyPI page renders correctly and install works in a clean venv.
- Announce the release and link to the documentation.
