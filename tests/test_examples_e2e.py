from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run_example(rel_path: str) -> str:
    result = subprocess.run(
        [sys.executable, str(ROOT / rel_path)],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=120,
    )
    return (result.stdout or "") + (result.stderr or "")


def test_quickstart_bandit_runs():
    output = _run_example("examples/quickstart/bandit_ope.py")
    assert "estimate=" in output
    assert "true=" in output
    assert "warnings=" in output


def test_quickstart_mdp_runs():
    output = _run_example("examples/quickstart/mdp_ope.py")
    assert "estimate=" in output
    assert "true=" in output
    assert "warnings=" in output
