"""Matplotlib styling helpers for journal-ready plots.

Design goals:
- Consistent figure sizes (single/double column)
- Consistent fonts + mathtext (LaTeX-like without requiring TeX)
- High-quality export: PDF (vector) + PNG (high DPI)
- No global matplotlib dependency at import time (imports are local)
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping


@dataclass(frozen=True)
class FigureSpec:
    """Figure sizing standard for papers/docs."""

    width_in: float
    height_in: float
    constrained_layout: bool = True


def paper_figspec(column: str = "single", aspect: float = 0.62) -> FigureSpec:
    """
    Typical journal widths:
    - single column ~ 3.25-3.5 in
    - double column ~ 6.5-7.0 in
    """
    if column not in {"single", "double"}:
        raise ValueError("column must be 'single' or 'double'")
    width = 3.5 if column == "single" else 7.0
    height = max(1.9, width * aspect)
    return FigureSpec(width_in=width, height_in=height)


def _journal_rcparams() -> dict[str, Any]:
    # No TeX required; use STIX fonts to make math look paper-like.
    return {
        # Typography
        "font.family": "STIXGeneral",
        "mathtext.fontset": "stix",
        "font.size": 9.0,
        "axes.labelsize": 9.0,
        "axes.titlesize": 9.0,
        "xtick.labelsize": 8.0,
        "ytick.labelsize": 8.0,
        "legend.fontsize": 8.0,
        "legend.frameon": False,
        # Lines/axes
        "axes.linewidth": 1.0,
        "axes.grid": False,
        "lines.linewidth": 2.0,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.width": 1.0,
        "ytick.major.width": 1.0,
        "xtick.major.size": 5.0,
        "ytick.major.size": 5.0,
        # Background/export
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        # Embed fonts nicely in PDFs
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }


@contextmanager
def journal_style(extra_rc: Mapping[str, Any] | None = None):
    """Context manager for consistent styling without global side-effects."""
    try:
        import matplotlib as mpl
    except ImportError as exc:  # pragma: no cover
        raise ImportError("matplotlib is required for plotting.") from exc

    rc = _journal_rcparams()
    if extra_rc:
        rc.update(dict(extra_rc))

    with mpl.rc_context(rc):
        yield


def new_figure(spec: FigureSpec | None = None):
    """Create a styled figure/axes pair."""
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("matplotlib is required for plotting.") from exc

    spec = spec or paper_figspec(column="single")
    fig, ax = plt.subplots(
        figsize=(spec.width_in, spec.height_in),
        constrained_layout=spec.constrained_layout,
    )
    return fig, ax


def apply_axes_style(ax: Any) -> None:
    """Small, consistent cosmetics that match journal histograms."""
    ax.tick_params(axis="both", which="major", direction="out")
    # Keep all spines (matches journal-style framing).
    for spine in ax.spines.values():
        spine.set_visible(True)


def save_figure(fig: Any, outpath: str | Path, *, dpi: int = 500) -> dict[str, str]:
    """
    Save both PDF (vector) and PNG (high DPI) with the same stem.
    Returns the generated file paths.
    """
    outpath = Path(outpath)
    outpath.parent.mkdir(parents=True, exist_ok=True)

    pdf_path = outpath.with_suffix(".pdf")
    png_path = outpath.with_suffix(".png")

    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)  # vector
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight", pad_inches=0.02)

    return {"pdf": str(pdf_path), "png": str(png_path)}


def figure_to_base64(fig: Any, *, dpi: int = 500) -> str:
    """Render a figure to base64-encoded PNG."""
    import base64

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def configure_notebook_display(*, dpi: int = 300, use_retina: bool = True) -> None:
    """Improve inline figure clarity in notebooks."""
    try:
        import matplotlib as mpl
    except ImportError as exc:  # pragma: no cover
        raise ImportError("matplotlib is required for plotting.") from exc

    mpl.rcParams["figure.dpi"] = dpi
    if use_retina:
        try:
            from matplotlib_inline.backend_inline import set_matplotlib_formats

            set_matplotlib_formats("retina")
        except Exception:
            pass


def set_style(style: str | None = None) -> None:
    """Apply the journal rcParams globally (legacy helper)."""
    _ = style
    try:
        import matplotlib as mpl
    except ImportError as exc:  # pragma: no cover
        raise ImportError("matplotlib is required for plotting.") from exc

    mpl.rcParams.update(_journal_rcparams())


@contextmanager
def paper_context(style: str | None = None):
    """Backwards-compatible alias for journal_style()."""
    _ = style
    with journal_style():
        yield
