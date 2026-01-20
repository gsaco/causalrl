"""Plotting helpers for documentation figures."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

STYLE_PATH = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "assets"
    / "styles"
    / "crl_style.mplstyle"
)


def apply_style() -> None:
    """Apply the CausalRL Matplotlib style if present."""

    if STYLE_PATH.exists():
        plt.style.use(str(STYLE_PATH))


def save_figure_bundle(
    fig: plt.Figure,
    out_dir: Path,
    name: str,
    *,
    dpi_high: int = 500,
    dpi_web: int = 220,
    optimize: bool = True,
) -> dict[str, Path]:
    """Save SVG + high-res PNG + web PNG for a figure."""

    out_dir.mkdir(parents=True, exist_ok=True)
    svg_path = out_dir / f"{name}.svg"
    png_path = out_dir / f"{name}.png"
    web_path = out_dir / f"{name}_web.png"

    fig.savefig(svg_path, format="svg")
    fig.savefig(png_path, dpi=dpi_high)
    fig.savefig(web_path, dpi=dpi_web)

    if optimize:
        _optimize_pngs([png_path, web_path])

    return {"svg": svg_path, "png": png_path, "web": web_path}


def _optimize_pngs(paths: Iterable[Path]) -> None:
    optimizer = shutil.which("oxipng")
    if optimizer is None:
        return
    for path in paths:
        subprocess.run(
            [optimizer, "-o4", str(path)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
