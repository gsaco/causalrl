"""CLI entrypoints for CRL."""

from __future__ import annotations

from pathlib import Path

import typer

from crl.config import load_evaluation_spec
from crl.ope import evaluate

app = typer.Typer(help="CausalRL command line interface.")


def _run_ope(config: str, out: str) -> None:
    output_dir = Path(out)
    output_dir.mkdir(parents=True, exist_ok=True)

    spec = load_evaluation_spec(config)
    result = evaluate(spec)
    result.save_bundle(str(output_dir))


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    config: str | None = typer.Option(None, help="Path to OPE config YAML."),
    out: str | None = typer.Option(None, help="Output directory for reports."),
) -> None:
    """Run OPE based on a YAML config file."""

    if ctx.invoked_subcommand is not None:
        return
    if config is None or out is None:
        typer.echo(ctx.get_help())
        raise typer.Exit(code=1)
    _run_ope(config, out)


@app.command("ope")
def ope(
    config: str = typer.Option(..., help="Path to OPE config YAML."),
    out: str = typer.Option(..., help="Output directory for reports."),
) -> None:
    """Run OPE based on a YAML config file."""

    _run_ope(config, out)


if __name__ == "__main__":
    app()
